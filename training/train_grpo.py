"""
GRPO training — MIMIC Discharge Planning
=========================================
Model           : Qwen/Qwen2.5-3B-Instruct  (bfloat16, LoRA r=16)

Curriculum 
----------
  Steps    0 -  199 : Task 1  noise=clean    curriculum=medium_only  (200 steps ~1.9h)
  Steps  200 -  349 : Task 2  noise=clean    curriculum=medium_only  (150 steps ~1.5h)
  Steps  350 -  449 : Task 3  noise=partial  curriculum=random       (100 steps ~1.5h)
  Steps  450 -  549 : Task 4  noise=partial  curriculum=hard_only    (100 steps ~1.7h)

Checkpointing & Resume
-----------------------
  After every chunk: HuggingFace model checkpoint + train_state.json saved.
  To resume a crashed run:

      python -m training.train_grpo --resume_from ./checkpoints/chunk_004

  This reloads LoRA weights + full training state (step, chunk,
  zero-reward buffers, elapsed time) and continues exactly where it stopped.

Usage
-----
  # Fresh run
  python -m training.train_grpo

  # Resume
  python -m training.train_grpo --resume_from ./checkpoints/chunk_004

  # Re-plot only
  python -m training.train_grpo --replot ./logs/training_20260426_134017_fresh.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*AttentionMaskConverter.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*attention mask API.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Passing `generation_config` together with generation-related arguments.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*`max_new_tokens`.*and `max_length`.*",
)
# ── End warning suppression ───────────────────────────────────────────────────

import random
import requests
import torch

# ── Optional Unsloth ──────────────────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
    _UNSLOTH = True
    print("[init] Unsloth found — using 4-bit quantised loading", flush=True)
except ImportError:
    _UNSLOTH = False
    print("[init] Unsloth not found — using bfloat16 loading", flush=True)

from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# ── Hardware / training constants for L4 24 GB ───────────────────────────────
ENV_TIMEOUT       = 30
MAX_SEQ_LENGTH    = 2560   # prompt (1536) + completion (768) + buffer
MAX_PROMPT_LENGTH = 1536
MAX_COMP_LENGTH   = 768    # Task-1 reasoning was hitting 512 → truncation → parse fail
NUM_GENERATIONS   = 8      # min for non-degenerate GRPO advantage
BATCH_SIZE        = 2      # 8 gen x 2 batch x 2560 tok fits in 24 GB
GRAD_ACCUM        = 8      # effective batch = 2 x 8 = 16
ZERO_WARN_AFTER   = 40     # buffer fill before firing zero-rate warning
LORA_R            = 16
MIN_EASY_POOL     = 10    # easy tier works with fewer patients (home-discharge cases, no hospice)
MIN_TIER_POOL     = 25    # minimum for medium/hard tier curriculum
ZERO_REWARD_THRESHOLD = 0.12
# Dead-gradient stop thresholds per task (fraction of "near-zero" rewards).
DEAD_THRESHOLDS       = {1: 0.80, 2: 0.90, 3: 0.90, 4: 0.90}

_SEED_N_BY_MODE: Dict[str, int] = {
    "easy_only":   44,   # 21 × 2
    "medium_only": 220,  # 109 × 2  (T1 + T2)
    "hard_only":   210,  # 103 × 2  (T4)
    "random":      466,  # 233 × 2  (T3 — full patient pool)
    "easy_medium": 260,  # (21+109) × 2 — fallback blend
}


def _chunk_seed_n(curr_mode: str, pool_sizes: Optional[Dict[str, int]], seed_n_floor: int) -> int:
    """Return seed dataset size for one chunk.

    Auto-scales to 2× the active patient pool so every unique patient appears
    at least twice per chunk.  seed_n_floor acts as a user-supplied minimum.
    """
    mode_key = curr_mode if curr_mode in _SEED_N_BY_MODE else "random"
    # Re-derive from live pool sizes when available so demo vs. full dataset works.
    if pool_sizes:
        easy   = pool_sizes.get("easy",   21)
        medium = pool_sizes.get("medium", 109)
        hard   = pool_sizes.get("hard",   103)
        total  = pool_sizes.get("total",  easy + medium + hard)
        pool_n = {
            "easy_only":   easy,
            "medium_only": medium,
            "hard_only":   hard,
            "random":      total,
            "easy_medium": easy + medium,
        }.get(mode_key, total)
        auto = pool_n * 2
    else:
        auto = _SEED_N_BY_MODE.get(mode_key, 466)
    return max(seed_n_floor, auto)


# ── VRAM helpers ──────────────────────────────────────────────────────────────

def _vram_mb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}
    alloc    = torch.cuda.memory_allocated()  / 1024**2
    reserved = torch.cuda.memory_reserved()   / 1024**2
    total    = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return {
        "allocated": round(alloc, 1),
        "reserved":  round(reserved, 1),
        "free":      round(total - reserved, 1),
        "total":     round(total, 1),
    }


def _vram_str() -> str:
    v = _vram_mb()
    return (f"VRAM  alloc={v['allocated']:.0f}MB  "
            f"reserved={v['reserved']:.0f}MB  "
            f"free={v['free']:.0f}MB / {v['total']:.0f}MB")


# ── Complexity pool helpers ───────────────────────────────────────────────────

def fetch_pool_sizes(env_url: str) -> Dict[str, int]:
    """
    Query /episodes/by_complexity and return {tier: count}.
    Falls back to {"easy": 0, "medium": 0, "hard": 0} on error so callers
    can still decide — they will see counts < MIN_TIER_POOL and fall back
    to "random" automatically.
    """
    default = {"easy": 0, "medium": 0, "hard": 0}
    try:
        r = requests.get(
            f"{env_url.rstrip('/')}/episodes/by_complexity",
            timeout=ENV_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        totals = data.get("totals", {})
        return {
            "easy":   int(totals.get("easy",   0)),
            "medium": int(totals.get("medium", 0)),
            "hard":   int(totals.get("hard",   0)),
        }
    except Exception as exc:
        print(f"[pool] /episodes/by_complexity failed: {exc} — defaulting to random",
              file=sys.stderr, flush=True)
        return default


def _resolve_mode(preferred_mode: str, pool_sizes: Dict[str, int]) -> str:
    """Return the effective curriculum mode for this phase.

    easy_only uses MIN_EASY_POOL (10) as its threshold — 21 easy patients
    is plenty for Phase 1 and keeps hospice/cancer medium cases OUT of the
    early curriculum.  Only if easy < 10 do we fall back to easy_medium.
    Other tiers use the larger MIN_TIER_POOL (25).
    """
    tier_map = {"easy_only": "easy", "medium_only": "medium", "hard_only": "hard"}
    tier = tier_map.get(preferred_mode)
    if tier:
        count   = pool_sizes.get(tier, 0)
        min_req = MIN_EASY_POOL if preferred_mode == "easy_only" else MIN_TIER_POOL
        if count < min_req:
            if preferred_mode == "easy_only":
                return "easy_medium"
            return "random"
    return preferred_mode


def _pick_mode(mode: str, pool_sizes: Dict[str, int]) -> str:
    """Resolve the pseudo-mode 'easy_medium' to a concrete env curriculum mode.

    Randomly picks 'easy_only' or 'medium_only' weighted by their pool sizes
    so each episode reflects the natural patient distribution.  All other modes
    pass through unchanged.
    """
    if mode != "easy_medium":
        return mode
    easy   = pool_sizes.get("easy",   0)
    medium = pool_sizes.get("medium", 0)
    total  = easy + medium
    if total == 0:
        return "random"
    return "easy_only" if random.random() < (easy / total) else "medium_only"


# ── Curriculum ────────────────────────────────────────────────────────────────

_CURRICULUM_PREFERRED = {
    (0,   199): (1, "clean",   "medium_only"),
    (200, 349): (2, "clean",   "medium_only"),
    (350, 449): (3, "partial", "random"),
    (450, 549): (4, "partial", "hard_only"),
}


def _curriculum(step: int, pool_sizes: Optional[Dict[str, int]] = None) -> Tuple[int, str, str]:
    """Return (task_id, noise_level, curriculum_mode) for a global step.

    If pool_sizes is provided (fetched from the env at startup), the
    preferred curriculum mode is validated against the real patient counts
    and downgraded to "random" when the tier is too small to provide
    meaningful diversity.
    """
    if step < 200:
        tid, noise, preferred = 1, "clean",   "medium_only"
    elif step < 350:
        tid, noise, preferred = 2, "clean",   "medium_only"
    elif step < 450:
        tid, noise, preferred = 3, "partial", "random"
    else:
        tid, noise, preferred = 4, "partial", "hard_only"

    mode = _resolve_mode(preferred, pool_sizes) if pool_sizes else preferred
    return tid, noise, mode


def _phase_of(task_id: int) -> int:
    return {1: 1, 2: 2, 3: 3, 4: 4}.get(task_id, 1)


# ── Logger ────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Dual-sink logger: machine-readable JSONL + human-readable .log file.
    Also captures TRL internal metrics (loss, entropy, clipped_ratio).
    """

    def __init__(self, log_dir: str, resume: bool = False) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "resume" if resume else "fresh"
        self.jsonl_path = self.log_dir / f"training_{ts}_{mode}.jsonl"
        self.text_path  = self.log_dir / f"training_{ts}_{mode}.log"
        self._jfh = open(self.jsonl_path, "w", buffering=1)
        self._tfh = open(self.text_path,  "w", buffering=1)
        self._t0  = time.time()
        self.all_records:    List[Dict] = []
        self.trl_records:    List[Dict] = []
        self._chunk_records: List[Dict] = []
        self._p("=" * 68)
        self._p("  MIMIC Discharge Planning - GRPO Training Log")
        self._p(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._p(f"  Mode     : {mode.upper()}")
        self._p(f"  JSONL    : {self.jsonl_path}")
        self._p(f"  Text log : {self.text_path}")
        self._p(f"  {_vram_str()}")
        self._p("=" * 68)

    def _elapsed(self) -> float:
        return time.time() - self._t0

    def _p(self, msg: str) -> None:
        line = f"[{self._elapsed():>8.1f}s] {msg}"
        print(line, flush=True)
        self._tfh.write(line + "\n")

    def _eta(self, step: int, max_steps: int) -> str:
        if step == 0:
            return "n/a"
        rate = self._elapsed() / step
        return str(timedelta(seconds=int((max_steps - step) * rate)))

    # ── Public API ────────────────────────────────────────────────────────────

    def log_phase_start(self, phase: int, task_id: int,
                        noise: str, curriculum: str, step: int) -> None:
        self._p("=" * 68)
        self._p(f"  PHASE {phase} START  task={task_id}  noise={noise}  "
                f"curriculum={curriculum}  step={step}")
        self._p(f"  {_vram_str()}")
        self._p("=" * 68)
        self._chunk_records = []

    def log_chunk_start(self, chunk: int, step: int, max_steps: int,
                        task_id: int, noise: str, curriculum: str) -> None:
        self._p("-" * 68)
        self._p(f"  Chunk {chunk:03d}  |  step {step:5d}/{max_steps}  "
                f"|  task={task_id}  noise={noise}  curriculum={curriculum}")
        self._p(f"  {_vram_str()}")
        self._p("-" * 68)
        self._chunk_records = []

    def log_seed_build(self, task_id: int, built: int, requested: int) -> None:
        self._p(f"  [seed] Built {built}/{requested} prompts for task={task_id}")

    def log_reward_batch(
        self,
        global_step: int,
        chunk: int,
        max_steps: int,
        task_id: int,
        noise_level: str,
        curriculum_mode: str,
        rewards: List[float],
        parse_ok: List[bool],
        sample_response: Optional[str] = None,
        n_truncated: int = 0,
    ) -> None:
        n      = len(rewards) or 1
        mean_r = sum(rewards) / n
        min_r  = min(rewards) if rewards else 0.0
        max_r  = max(rewards) if rewards else 0.0
        p_rate = sum(parse_ok) / len(parse_ok) if parse_ok else 0.0
        z_rate = sum(r == 0.0 for r in rewards) / n
        v      = _vram_mb()

        record = {
            "ts":               time.time(),
            "elapsed_s":        round(self._elapsed(), 2),
            "step":             global_step,
            "chunk":            chunk,
            "task_id":          task_id,
            "noise_level":      noise_level,
            "curriculum_mode":  curriculum_mode,
            "mean_reward":      round(mean_r, 5),
            "min_reward":       round(min_r,  5),
            "max_reward":       round(max_r,  5),
            "rewards":          [round(r, 4) for r in rewards],
            "parse_ok_rate":    round(p_rate, 4),
            "zero_reward_rate": round(z_rate, 4),
            "n_truncated":      n_truncated,
            "vram_alloc_mb":    v["allocated"],
        }
        self._jfh.write(json.dumps(record) + "\n")
        self.all_records.append(record)
        self._chunk_records.append(record)

        r_str      = " ".join(f"{r:.3f}" for r in rewards)
        eta        = self._eta(global_step, max_steps)
        trunc_str  = f"  trunc={n_truncated}/{n}" if n_truncated else ""
        parse_icon = "OK" if p_rate >= 0.80 else ("~" if p_rate >= 0.50 else "!!")
        zero_icon  = "OK" if z_rate < 0.30  else ("~" if z_rate < 0.50  else "!!")

        self._p(
            f"  step={global_step:5d}  "
            f"mean={mean_r:.4f}  min={min_r:.3f}  max={max_r:.3f}  "
            f"parse[{parse_icon}]={p_rate:.0%}  zeros[{zero_icon}]={z_rate:.0%}"
            f"{trunc_str}  ETA={eta}"
        )
        self._p(f"    rewards: [{r_str}]")
        self._p(f"    VRAM: alloc={v['allocated']:.0f}MB  "
                f"reserved={v['reserved']:.0f}MB  free={v['free']:.0f}MB")
        if sample_response is not None:
            preview = sample_response.replace("\n", " ")[:300]
            self._p(f"    sample: {preview}")

    def log_trl_metrics(self, step: int, metrics: Dict) -> None:
        """Intercept TRL's own metrics dict and re-log with our format."""
        clipped  = float(metrics.get("completions/clipped_ratio", -1))
        comp_len = float(metrics.get("completions/mean_length", -1))
        term_len = float(metrics.get("completions/mean_terminated_length", -1))
        entropy  = float(metrics.get("entropy", -1))
        loss     = float(metrics.get("loss", -1))
        grad_n   = float(metrics.get("grad_norm", -1))
        lr_now   = float(metrics.get("learning_rate", -1))

        record = {
            "ts": time.time(), "step": step,
            "clipped_ratio": clipped, "mean_comp_length": comp_len,
            "mean_term_length": term_len, "entropy": entropy,
            "loss": loss, "grad_norm": grad_n, "lr": lr_now,
        }
        self.trl_records.append(record)

        flags = []
        if clipped >= 0.90:
            flags.append(f"!! CLIPPED={clipped:.0%} — completions hitting length limit!")
        if clipped == 1.00:
            flags.append("!! ALL COMPLETIONS TRUNCATED — increase MAX_COMP_LENGTH")
        if term_len == 0.0 and comp_len > 0:
            flags.append("!! ZERO natural terminations — model never finishes JSON")
        if 0 <= entropy < 0.5:
            flags.append(f"!! LOW ENTROPY={entropy:.3f} — model output collapsing")
        if 0 <= loss < 1e-7:
            flags.append(f"!! NEAR-ZERO LOSS={loss:.2e} — no gradient signal")

        self._p(
            f"  [TRL] loss={loss:.3e}  grad={grad_n:.4f}  lr={lr_now:.2e}  "
            f"entropy={entropy:.3f}  clipped={clipped:.0%}  "
            f"comp_len={comp_len:.0f}  term_len={term_len:.1f}"
        )
        for flag in flags:
            self._p(f"    {flag}")

    def log_zero_warn(self, task_id: int, rate: float, buf_len: int) -> None:
        self._p(f"  !! WARN  Task {task_id} zero-reward rate={rate:.1%} "
                f"over last {buf_len} rollouts")

    def log_dead_gradient(self, task_id: int, rate: float) -> None:
        self._p("!" * 68)
        self._p(f"  DEAD GRADIENT  Task {task_id} zero-reward={rate:.1%} "
                f"over last 50 rollouts — halting.")
        self._p("!" * 68)

    def log_checkpoint(self, path: str, step: int, chunk: int) -> None:
        self._p(f"  [ckpt] Saved -> {path}  step={step}  chunk={chunk}")

    def log_chunk_summary(self, chunk: int, elapsed_min: float,
                          zero_rates: Dict[int, float]) -> None:
        if not self._chunk_records:
            return
        rws    = [r["mean_reward"]   for r in self._chunk_records]
        parses = [r["parse_ok_rate"] for r in self._chunk_records]
        zr_str = "  ".join(f"T{t}={v:.0%}" for t, v in sorted(zero_rates.items()))
        self._p("=" * 68)
        self._p(f"  CHUNK {chunk:03d} DONE  "
                f"mean_reward={sum(rws)/len(rws):.4f}  "
                f"parse={sum(parses)/len(parses):.0%}  "
                f"elapsed={elapsed_min:.1f}min")
        self._p(f"  zero-rates: {zr_str}")
        self._p(f"  {_vram_str()}")
        self._p("=" * 68)

    def log_training_end(self, output_dir: str, step: int) -> None:
        self._p("=" * 68)
        self._p(f"  TRAINING COMPLETE  step={step}")
        self._p(f"  Model    -> {output_dir}")
        self._p(f"  JSONL    -> {self.jsonl_path}")
        self._p(f"  Text log -> {self.text_path}")
        self._p(f"  Elapsed  -> {self._elapsed()/60:.1f} min")
        self._p("=" * 68)

    def close(self) -> None:
        self._jfh.close()
        self._tfh.close()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_dir(output_dir: str, chunk: int) -> Path:
    return Path(output_dir) / f"chunk_{chunk:03d}"


def save_train_state(
    output_dir:    str,
    chunk:         int,
    step_counter:  List[int],
    chunk_counter: List[int],
    zero_bufs:     Dict[int, Deque[bool]],
    t0:            float,
    logger:        TrainingLogger,
) -> str:
    ckpt = _ckpt_dir(output_dir, chunk)
    ckpt.mkdir(parents=True, exist_ok=True)
    state = {
        "chunk":     chunk,
        "step":      step_counter[0],
        "zero_bufs": {str(k): list(v) for k, v in zero_bufs.items()},
        "elapsed_s": round(time.time() - t0, 2),
        "saved_at":  datetime.now().isoformat(),
    }
    state_path = ckpt / "train_state.json"
    state_path.write_text(json.dumps(state, indent=2))
    logger.log_checkpoint(str(ckpt), step_counter[0], chunk)
    return str(ckpt)


def load_train_state(
    resume_from:   str,
    step_counter:  List[int],
    chunk_counter: List[int],
    zero_bufs:     Dict[int, Deque[bool]],
) -> float:
    """Load state from checkpoint dir. Returns previous elapsed_s."""
    state_path = Path(resume_from) / "train_state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"No train_state.json in {resume_from}.\n"
            f"Point --resume_from at a chunk_NNN directory."
        )
    state = json.loads(state_path.read_text())
    step_counter[0]  = state["step"]
    chunk_counter[0] = state["chunk"] + 1
    for k, v in state.get("zero_bufs", {}).items():
        tid = int(k)
        if tid in zero_bufs:
            zero_bufs[tid] = deque(v, maxlen=50)
    elapsed = float(state.get("elapsed_s", 0.0))
    print(f"[resume] step={step_counter[0]}  chunk={chunk_counter[0]}  "
          f"elapsed_prev={elapsed/60:.1f}min", flush=True)
    return elapsed


# ── Visualisations ────────────────────────────────────────────────────────────

_TASK_COLORS  = {1: "#4361EE", 2: "#F4A261", 3: "#2DC653", 4: "#E63946"}
_PHASE_COLORS = {1: "#E8EEFF", 2: "#FFF3E0", 3: "#E8F8EE", 4: "#FDECEA"}

_TASK_META = {
    1: {"label": "Task 1 – Disposition",    "diff": "Easy",      "steps": "0–199"},
    2: {"label": "Task 2 – Care Plan",      "diff": "Medium",    "steps": "200–349"},
    3: {"label": "Task 3 – Discharge Note", "diff": "Hard",      "steps": "350–449"},
    4: {"label": "Task 4 – ICU Workflow",   "diff": "Very Hard", "steps": "450–549"},
}
_DIFF_COLOR = {"Easy": "#4361EE", "Medium": "#F4A261", "Hard": "#2DC653", "Very Hard": "#E63946"}

# Curriculum phase boundaries — must match _curriculum() exactly
_PHASE_BANDS = [
    (0,   200, "Task 1\nDisposition\n(Easy)",        1),
    (200, 350, "Task 2\nCare Plan\n(Medium)",        2),
    (350, 450, "Task 3\nDischarge Note\n(Hard)",     3),
    (450, 550, "Task 4\nICU Workflow\n(Very Hard)",  4),
]
_PHASE_TRANSITIONS = [200, 350, 450]


def _rolling(arr: List[float], w: int) -> List[float]:
    out = []
    for i in range(len(arr)):
        win = arr[max(0, i - w + 1): i + 1]
        out.append(float(sum(win) / len(win)))
    return out


def _setup_ax(ax: Any, title: str, xlabel: str, ylabel: str,
              ylim: Optional[Tuple] = None) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#1a1a2e")
    ax.set_xlabel(xlabel, fontsize=10, color="#333")
    ax.set_ylabel(ylabel, fontsize=10, color="#333")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.18, linestyle="--", color="#aaa")
    ax.tick_params(labelsize=9, colors="#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#ccc")


def _shade_phases(ax: Any, steps: List[int]) -> None:
    if not steps:
        return
    max_s = max(steps)
    for lo, hi, _, phase in _PHASE_BANDS:
        hi_c = min(hi, max_s)
        if hi_c > lo:
            ax.axvspan(lo, hi_c, alpha=0.09, color=_PHASE_COLORS[phase], zorder=0)


def _draw_transitions(ax: Any, max_s: int, ymax: float = 1.05) -> None:
    for b in _PHASE_TRANSITIONS:
        if b < max_s:
            ax.axvline(b, color="#888", linestyle="--", lw=1.1, alpha=0.55, zorder=1)


def _task_legend_handles(present_tasks: List[int]) -> List[Any]:
    from matplotlib.patches import Patch
    handles = []
    for t in present_tasks:
        m = _TASK_META[t]
        handles.append(Patch(
            facecolor=_TASK_COLORS[t], alpha=0.80,
            label=f"{m['label']}  [{m['diff']}]  steps {m['steps']}",
        ))
    return handles


def _save_fig(fig: Any, path: str) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout(pad=1.8)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [viz] -> {path}", flush=True)


def plot_all(
    records:     List[Dict],
    trl_records: List[Dict],
    log_dir:     str,
) -> List[str]:
    """Generate all 9 PNGs. Safe to call with partial data mid-training."""
    if not records:
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "white",
    })

    out_dir = Path(log_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    steps        = [r["step"]             for r in records]
    mean_rewards = [r["mean_reward"]      for r in records]
    parse_rates  = [r["parse_ok_rate"]    for r in records]
    zero_rates   = [r["zero_reward_rate"] for r in records]
    task_ids     = [r["task_id"]          for r in records]
    vram_mb      = [r.get("vram_alloc_mb", 0) for r in records]

    present_tasks = sorted(set(task_ids))
    max_s = max(steps) if steps else 1

    sm50  = _rolling(mean_rewards, 50)
    sm100 = _rolling(parse_rates, 100)
    sm50z = _rolling(zero_rates, 50)

    # ── 01 Reward Curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_phases(ax, steps)
    _draw_transitions(ax, max_s)

    # per-task scatter
    for tid in present_tasks:
        xs = [s for s, t in zip(steps, task_ids) if t == tid]
        ys = [r for r, t in zip(mean_rewards, task_ids) if t == tid]
        ax.scatter(xs, ys, s=6, color=_TASK_COLORS[tid], alpha=0.25, zorder=2)

    ax.plot(steps, mean_rewards, alpha=0.18, color="#555", linewidth=0.7, zorder=3)
    ax.plot(steps, sm50, color="#1a1a2e", linewidth=2.2, label="Rolling-50 mean", zorder=4)

    # annotate peak reward per phase
    for lo, hi, _, tid in _PHASE_BANDS:
        phase_y = [y for s, y in zip(steps, mean_rewards) if lo <= s < hi]
        if phase_y:
            peak = max(phase_y)
            peak_s = [s for s, y in zip(steps, mean_rewards) if lo <= s < hi and y == peak][0]
            ax.annotate(f"peak {peak:.2f}", xy=(peak_s, peak),
                        xytext=(peak_s, peak + 0.07),
                        fontsize=7.5, ha="center", color=_TASK_COLORS[tid],
                        arrowprops=dict(arrowstyle="-", color=_TASK_COLORS[tid], lw=0.8))

    leg = _task_legend_handles(present_tasks)
    leg.append(plt.Line2D([0], [0], color="#1a1a2e", lw=2.2, label="Rolling-50 mean"))
    ax.legend(handles=leg, fontsize=8.5, loc="upper left",
              framealpha=0.9, edgecolor="#ccc")
    _setup_ax(ax, "Training Reward Curve — MIMIC Discharge Planning",
              "Global Training Step", "Mean Reward  (0 – 1)", (0, 1.12))
    p = str(out_dir / "01_reward_curve.png"); _save_fig(fig, p); saved.append(p)

    # ── 02 JSON Parse Rate ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    _shade_phases(ax, steps)
    _draw_transitions(ax, max_s)
    ax.fill_between(steps, parse_rates, alpha=0.12, color="#F4A261")
    ax.plot(steps, parse_rates, alpha=0.20, color="#F4A261", linewidth=0.7)
    ax.plot(steps, sm100, color="#c44b00", linewidth=2.2, label="Rolling-100 parse rate")
    ax.axhline(0.80, color="#c0392b", linestyle="--", lw=1.4, label="80% target floor")
    ax.axhline(0.50, color="#e67e22", linestyle=":",  lw=1.1, label="50% critical floor")
    leg = _task_legend_handles(present_tasks)
    leg += [
        plt.Line2D([0], [0], color="#c44b00",  lw=2.2, label="Rolling-100"),
        plt.Line2D([0], [0], color="#c0392b",  lw=1.4, linestyle="--", label="80% target"),
    ]
    ax.legend(handles=leg, fontsize=8.5, loc="lower left", framealpha=0.9, edgecolor="#ccc")
    _setup_ax(ax, "JSON Parse Success Rate by Training Phase",
              "Global Training Step", "Parse Rate", (0, 1.08))
    p = str(out_dir / "02_parse_rate.png"); _save_fig(fig, p); saved.append(p)

    # ── 03 Dead-Gradient Monitor ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    _shade_phases(ax, steps)
    _draw_transitions(ax, max_s)
    for tid in present_tasks:
        col = _TASK_COLORS[tid]
        tx_steps = [s for s, t in zip(steps, task_ids) if t == tid]
        tx_zero  = [z for z, t in zip(zero_rates, task_ids) if t == tid]
        if not tx_steps:
            continue
        tx_sm = _rolling(tx_zero, 50)
        ax.plot(tx_steps, tx_zero, alpha=0.18, color=col, lw=0.7)
        m = _TASK_META[tid]
        ax.plot(tx_steps, tx_sm, color=col, lw=2.2,
                label=f"{m['label']} [{m['diff']}]")
    ax.axhspan(0.60, 1.08, alpha=0.08, color="#c0392b")
    ax.axhspan(0.40, 0.60, alpha=0.06, color="#e67e22")
    ax.axhline(0.60, color="#c0392b", linestyle="--", lw=1.5, label="60%  → HALT training")
    ax.axhline(0.40, color="#e67e22", linestyle=":",  lw=1.2, label="40%  → WARN zone")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9, edgecolor="#ccc")
    _setup_ax(ax, "Dead-Gradient Monitor — Zero-Reward Rate per Task",
              "Global Training Step", "Zero-Reward Rate", (0, 1.08))
    p = str(out_dir / "03_dead_gradient.png"); _save_fig(fig, p); saved.append(p)

    # ── 04 Reward by Task (boxplot) ────────────────────────────────────────────
    task_reward_map: Dict[int, List[float]] = {t: [] for t in range(1, 5)}
    for r in records:
        tid = r["task_id"]
        if tid in task_reward_map:
            task_reward_map[tid].extend(r.get("rewards", [r["mean_reward"]]))
    present = sorted(t for t in task_reward_map if task_reward_map[t])
    if present:
        fig, ax = plt.subplots(figsize=(9, 6))
        tick_labels = [
            f"{_TASK_META[t]['label']}\n({_TASK_META[t]['diff']})" for t in present
        ]
        bp = ax.boxplot(
            [task_reward_map[t] for t in present],
            tick_labels=tick_labels,
            patch_artist=True, notch=False, widths=0.50,
        )
        for patch, t in zip(bp["boxes"], present):
            patch.set_facecolor(_TASK_COLORS[t])
            patch.set_alpha(0.68)
        for el in ["whiskers", "caps"]:
            for item in bp[el]:
                item.set(color="#555", linewidth=1.3)
        for flier in bp["fliers"]:
            flier.set(marker="o", color="#888", alpha=0.4, markersize=3)
        for med in bp["medians"]:
            med.set(color="white", linewidth=2.2)
        for i, t in enumerate(present, 1):
            data = task_reward_map[t]
            m = sum(data) / len(data)
            ax.plot(i, m, "D", color="#c0392b", markersize=7, zorder=6,
                    label="Mean" if i == 1 else "")
            ax.text(i + 0.28, m, f"μ={m:.3f}", va="center",
                    fontsize=8, color="#c0392b")
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#ccc")
        ax.set_xlabel("Task  (sorted by difficulty)", fontsize=10)
        _setup_ax(ax, "Reward Distribution by Task & Difficulty",
                  "Task  (sorted by difficulty)", "Reward  (0 – 1)", (0, 1.12))
        p = str(out_dir / "04_reward_by_task.png"); _save_fig(fig, p); saved.append(p)

    # ── 05 Phase Timeline ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    if steps:
        for lo, hi, label, tid in _PHASE_BANDS:
            lo_c, hi_c = min(lo, max_s), min(hi, max_s)
            if hi_c <= lo_c:
                continue
            color = _PHASE_COLORS[tid]
            ax.axvspan(lo_c, hi_c, alpha=0.38, color=color, zorder=0)
            mid = (lo_c + hi_c) / 2
            # Two-line label: task label + difficulty badge
            m = _TASK_META[tid]
            ax.text(mid, 0.97, m["label"], ha="center", va="top",
                    fontsize=9, fontweight="bold", color=_TASK_COLORS[tid],
                    transform=ax.get_xaxis_transform())
            ax.text(mid, 0.90, f"[{m['diff']}]  steps {m['steps']}", ha="center", va="top",
                    fontsize=7.5, color="#555",
                    transform=ax.get_xaxis_transform())

        ax.plot(steps, sm50, color="#1a1a2e", lw=2.2, label="Rolling-50 reward", zorder=4)

        for b in _PHASE_TRANSITIONS:
            if b < max_s:
                ax.axvline(b, color="#666", linestyle="--", lw=1.2, alpha=0.6, zorder=2)
                ax.text(b + 3, 0.04, f"step {b}", fontsize=7.5, color="#666",
                        rotation=90, va="bottom")

        # annotate rolling mean at each transition
        for b in _PHASE_TRANSITIONS:
            idx = min(range(len(steps)), key=lambda i: abs(steps[i] - b))
            val = sm50[idx]
            ax.annotate(f"{val:.2f}", xy=(b, val), xytext=(b + 18, val + 0.05),
                        fontsize=8, color="#333",
                        arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))

    leg = _task_legend_handles(present_tasks)
    leg.append(plt.Line2D([0], [0], color="#1a1a2e", lw=2.2, label="Rolling-50 reward"))
    ax.legend(handles=leg, fontsize=8.5, loc="lower right",
              framealpha=0.9, edgecolor="#ccc")
    _setup_ax(ax, "Curriculum Phase Timeline — Reward by Task & Difficulty",
              "Global Training Step", "Rolling Mean Reward  (0 – 1)", (0, 1.12))
    p = str(out_dir / "05_phase_timeline.png"); _save_fig(fig, p); saved.append(p)

    # ── 06 Reward Histogram ────────────────────────────────────────────────────
    all_rewards: List[float] = []
    for r in records:
        all_rewards.extend(r.get("rewards", [r["mean_reward"]]))
    if all_rewards:
        fig, ax = plt.subplots(figsize=(10, 5))
        bands = [
            (0.00, 0.10, "Near-zero  (parse fail / wrong class)", "#ffcccc"),
            (0.10, 0.40, "Partial credit  (adjacent / incomplete)", "#fff3cc"),
            (0.40, 0.70, "Good  (solid clinical reasoning)", "#d5f5e3"),
            (0.70, 1.01, "Excellent  (exact / near-perfect)", "#a9dfbf"),
        ]
        for lo, hi, label, color in bands:
            ax.axvspan(lo, hi, alpha=0.30, color=color, label=label, zorder=0)
        ax.hist(all_rewards, bins=44, color="#4361EE",
                alpha=0.72, edgecolor="white", lw=0.4, zorder=3)
        mean_val = sum(all_rewards) / len(all_rewards)
        ax.axvline(mean_val, color="#c0392b", linestyle="--",
                   lw=2.0, label=f"Overall mean = {mean_val:.3f}", zorder=5)
        # per-task means
        for tid in present_tasks:
            data = task_reward_map.get(tid, [])
            if data:
                m = sum(data) / len(data)
                ax.axvline(m, color=_TASK_COLORS[tid], linestyle=":",
                           lw=1.4, alpha=0.80, zorder=4,
                           label=f"T{tid} mean = {m:.3f}  [{_TASK_META[tid]['diff']}]")
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9, edgecolor="#ccc")
        _setup_ax(ax, "Reward Distribution Across All Tasks",
                  "Reward  (0 – 1)", "Sample Count")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        p = str(out_dir / "06_reward_histogram.png"); _save_fig(fig, p); saved.append(p)

    # ── 07 Per-Chunk Summary ───────────────────────────────────────────────────
    chunk_data: Dict[int, Dict] = {}
    for r in records:
        c = r["chunk"]
        if c not in chunk_data:
            chunk_data[c] = {"rewards": [], "task_ids": []}
        chunk_data[c]["rewards"].append(r["mean_reward"])
        chunk_data[c]["task_ids"].append(r["task_id"])
    if chunk_data:
        c_ids  = sorted(chunk_data)
        c_vals = [sum(chunk_data[c]["rewards"]) / len(chunk_data[c]["rewards"])
                  for c in c_ids]
        c_task = [max(set(chunk_data[c]["task_ids"]),
                      key=chunk_data[c]["task_ids"].count) for c in c_ids]
        bar_colors = [_TASK_COLORS.get(t, "#AAAAAA") for t in c_task]
        fig, ax = plt.subplots(figsize=(max(9, len(c_ids) * 0.7 + 2), 5))
        bars = ax.bar(c_ids, c_vals, color=bar_colors, alpha=0.75,
                      edgecolor="white", lw=0.6)
        ax.plot(c_ids, c_vals, "o--", color="#1a1a2e", lw=1.4, markersize=6, zorder=5)
        for x, y, b in zip(c_ids, c_vals, bars):
            ax.text(x, y + 0.012, f"{y:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#222")
            tid = c_task[c_ids.index(x)]
            ax.text(x, -0.045, _TASK_META[tid]["diff"],
                    ha="center", va="top", fontsize=7, color=_TASK_COLORS[tid],
                    transform=ax.transData)
        leg = _task_legend_handles(sorted(set(c_task)))
        ax.legend(handles=leg, fontsize=8.5, loc="upper right",
                  framealpha=0.9, edgecolor="#ccc")
        ax.set_xticks(c_ids)
        ax.set_xticklabels([f"Chunk {c}" for c in c_ids], fontsize=8, rotation=40, ha="right")
        top = max(c_vals) * 1.22 + 0.04 if c_vals else 1.0
        _setup_ax(ax, "Per-Chunk Mean Reward — Training Progression",
                  "Training Chunk  (50 steps each)", "Mean Reward  (0 – 1)", (0, top))
        p = str(out_dir / "07_chunk_summary.png"); _save_fig(fig, p); saved.append(p)

    # ── 08 Per-Task Learning Curves ───────────────────────────────────────────
    if len(present_tasks) >= 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes_flat = axes.flatten()
        diff_labels = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Very Hard"}

        for panel, tid in enumerate([1, 2, 3, 4]):
            ax = axes_flat[panel]
            t_steps   = [s for s, t in zip(steps, task_ids) if t == tid]
            t_rewards = [r for r, t in zip(mean_rewards, task_ids) if t == tid]

            if not t_steps:
                ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                        transform=ax.transAxes, fontsize=13, color="#aaa")
                ax.set_facecolor(_PHASE_COLORS[tid])
                meta = _TASK_META[tid]
                _setup_ax(ax,
                          f"{meta['label']}\n[{diff_labels[tid]}]  Steps {meta['steps']}",
                          "Global Training Step", "Mean Reward", (0, 1.1))
                continue

            color = _TASK_COLORS[tid]
            meta  = _TASK_META[tid]
            ax.set_facecolor(_PHASE_COLORS[tid])

            ax.scatter(t_steps, t_rewards, s=9, color=color, alpha=0.28, zorder=2)
            ax.plot(t_steps, t_rewards, color=color, alpha=0.15, lw=0.7, zorder=2)

            win = max(10, len(t_steps) // 8)
            sm  = _rolling(t_rewards, win)
            ax.plot(t_steps, sm, color=color, lw=2.2, zorder=4,
                    label=f"Rolling-{win} mean")

            mean_val = sum(t_rewards) / len(t_rewards)
            ax.axhline(mean_val, color=color, lw=1.2, linestyle="--", alpha=0.65,
                       label=f"Phase mean  {mean_val:.3f}")

            if t_rewards:
                peak   = max(t_rewards)
                pk_s   = t_steps[t_rewards.index(peak)]
                offset = min(0.12, 1.05 - peak)
                ax.annotate(f"peak {peak:.2f}",
                            xy=(pk_s, peak),
                            xytext=(pk_s, peak + offset),
                            fontsize=7.5, ha="center", color=color,
                            arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

            ax.legend(fontsize=8, framealpha=0.9, edgecolor="#ccc", loc="lower right")
            ax.text(0.015, 0.97, diff_labels[tid],
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    ha="left", va="top", color=color,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=color, alpha=0.85))
            _setup_ax(ax,
                      f"{meta['label']}  ·  Steps {meta['steps']}",
                      "Global Training Step", "Mean Reward", (0, 1.1))

        fig.suptitle("Per-Task Learning Curves — MIMIC Discharge Planning",
                     fontsize=13, fontweight="bold", color="#1a1a2e", y=1.01)
        fig.tight_layout(pad=2.0)
        p = str(out_dir / "08_per_task_curves.png"); _save_fig(fig, p); saved.append(p)

    # ── 09 VRAM Usage ─────────────────────────────────────────────────────────  # noqa: E265
    if any(v > 0 for v in vram_mb):
        fig, ax = plt.subplots(figsize=(14, 3))
        _shade_phases(ax, steps)
        ax.plot(steps, vram_mb, color="#7B2D8B", lw=1.6, label="Allocated VRAM (MB)")
        ax.fill_between(steps, vram_mb, alpha=0.14, color="#7B2D8B")
        ax.axhline(24 * 1024, color="#c0392b", linestyle="--",
                   lw=1.2, label="L4 24 GB limit")
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#ccc")
        _setup_ax(ax, "GPU VRAM Usage — NVIDIA L4 24 GB",
                  "Global Training Step", "VRAM Allocated (MB)")
        p = str(out_dir / "09_vram_usage.png"); _save_fig(fig, p); saved.append(p)

    # ── 09 Entropy + Loss + Clipped Ratio ─────────────────────────────────────
    if trl_records:
        trl_steps = [r["step"]          for r in trl_records]
        entropies = [r["entropy"]       for r in trl_records]
        losses    = [r["loss"]          for r in trl_records]
        clipped   = [r["clipped_ratio"] for r in trl_records]

        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        a1.plot(trl_steps, entropies, color="#0077B6", lw=1.6)
        a1.axhline(0.5, color="#c0392b", linestyle="--", lw=1.2,
                   label="Collapse threshold (0.5)")
        a1.legend(fontsize=8.5, framealpha=0.9)
        _setup_ax(a1, "Entropy — Higher = More Diverse Outputs (want > 0.5)",
                  "Step", "Entropy")

        a2.semilogy(trl_steps, [max(abs(l), 1e-12) for l in losses],
                    color="#E63946", lw=1.6)
        _setup_ax(a2, "|Loss|  (log scale)", "Step", "|Loss|")

        a3.plot(trl_steps, clipped, color="#F4A261", lw=1.6)
        a3.axhline(0.90, color="#c0392b", linestyle="--", lw=1.2,
                   label="Danger: > 90% clipped")
        a3.legend(fontsize=8.5, framealpha=0.9)
        _setup_ax(a3, "Completion Clipped Ratio  (want < 0.50)",
                  "Step", "Clipped Ratio", (0, 1.08))

        fig.tight_layout(pad=2.0)
        p = str(out_dir / "10_entropy_loss.png")
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [viz] -> {p}", flush=True)
        saved.append(p)

    return saved


# ── Env helpers ───────────────────────────────────────────────────────────────

def _env_post(env_url: str, path: str, body: Dict) -> Dict:
    r = requests.post(
        f"{env_url.rstrip('/')}{path}", json=body, timeout=ENV_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _extract_json(text: str) -> Optional[Dict]:
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except Exception:
        pass
    for pat in [r"```json\s*(.*?)```", r"```\s*(.*?)```", r"(\{.*\})"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(1).strip())
                return result if isinstance(result, dict) else None
            except Exception:
                continue
    return None


# ── Format-quality partial reward ─────────────────────────────────────────────

_REQUIRED_FIELDS: Dict[int, List[str]] = {
    1: ["disposition"],
    2: ["follow_up_specialties", "medications_to_continue", "key_instructions"],
    3: ["discharge_note"],
    4: ["final_note"],
}

_T4_ADVANCE_DEFAULTS: List[Dict] = [
    {"task_id": 4, "task4": {"triage_level": "icu"}},
    {"task_id": 4, "task4": {"priority_labs": ["CBC", "BMP", "LFTs"], "priority_consults": ["Primary Care"]}},
    {"task_id": 4, "task4": {"interventions": ["IV access", "continuous monitoring"]}},
    {"task_id": 4, "task4": {"high_risk_medications": []}},
    {"task_id": 4, "task4": {"antibiotic_strategy": "none", "antibiotics": []}},
    {"task_id": 4, "task4": {"fluid_strategy": "maintain"}},
    {"task_id": 4, "task4": {"ready_for_stepdown": False, "barriers": ["medical complexity"]}},
    {"task_id": 4, "task4": {"predicted_disposition": "snf", "los_remaining_days": 3.0}},
    {"task_id": 4, "task4": {"medications_to_continue": []}},
]


def _format_score(action_dict: Optional[Dict], task_id: int) -> float:
    """
    0.0  — no valid JSON at all
    0.2  — valid JSON but missing task key
    0.5  — has correct task key (e.g. "task1")
    1.0  — has all required sub-fields populated

    This gives a non-zero gradient signal even when env reward = 0,
    so the model learns JSON structure before clinical accuracy.
    """
    if action_dict is None:
        return 0.0
    task_key = f"task{task_id}"
    sub = action_dict.get(task_key)
    if sub is None:
        return 0.2
    if not isinstance(sub, dict):
        return 0.3
    fields = _REQUIRED_FIELDS.get(task_id, [])
    if not fields:
        return 0.5
    present = sum(1 for f in fields if sub.get(f))
    return round(0.5 + 0.5 * (present / len(fields)), 4)


def _reward_weights(task_id: int) -> Tuple[float, float]:
    """
    Returns (env_weight, format_weight).

    Task 1: env=0.80 so correct dispositions produce meaningfully different
    rewards (0.80 vs 0.40 vs 0.20 vs 0.04) rather than all collapsing to
    the 0.35 format floor.  Format weight 0.20 still gives gradient when
    the model can't parse — just less dominance once JSON works.
    """
    if task_id == 1:
        return 0.80, 0.20
    elif task_id == 2:
        return 0.85, 0.15
    elif task_id == 3:
        return 0.90, 0.10
    else:  # task 4
        return 0.90, 0.10


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name:  str,
    resume_from: Optional[str] = None,
):
    load_path = resume_from if resume_from else model_name
    print(f"[model] Loading {'checkpoint' if resume_from else 'base model'}: "
          f"{load_path}", flush=True)

    if _UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        if not resume_from:
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=LORA_R,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        print(f"[model] Unsloth 4-bit load done.  {_vram_str()}", flush=True)
        return model, tokenizer

    import transformers
    from peft import LoraConfig, get_peft_model, PeftModel

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base.gradient_checkpointing_enable()
    print(f"[model] Base model loaded.  {_vram_str()}", flush=True)

    if resume_from:
        print(f"[model] Loading LoRA adapter from {resume_from}", flush=True)
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_R, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()

    print(f"[model] LoRA ready.  {_vram_str()}", flush=True)
    return model, tokenizer


# ── Seed dataset ──────────────────────────────────────────────────────────────

def build_seed_dataset(
    env_url:         str,
    task_id:         int,
    n:               int,
    noise_level:     str,
    curriculum_mode: str,
    logger:          Optional[TrainingLogger] = None,
    pool_sizes:      Optional[Dict[str, int]] = None,
) -> Dataset:
    from training.rollout_collector import format_observation, _SYSTEM_PROMPT

    print(f"[seed] Building {n} prompts  task={task_id}  "
          f"noise={noise_level}  curriculum={curriculum_mode} ...", flush=True)
    rows: List[Dict] = []
    fail = 0
    for i in range(n):
        try:
            # Resolve pseudo-modes (e.g. "easy_medium") per episode so each
            # reset independently samples from the correct tier distribution.
            episode_mode = _pick_mode(curriculum_mode, pool_sizes) if pool_sizes else curriculum_mode
            obs = _env_post(env_url, "/reset", {
                "task_id":         task_id,
                "noise_level":     noise_level,
                "curriculum_mode": episode_mode,
            })
            if task_id == 2:
                meds_empty = not (obs.get("pharmacy_active") or obs.get("medications"))
                labs_empty = not obs.get("lab_flags")
                if meds_empty or labs_empty:
                    result = _env_post(env_url, "/step", {
                        "task_id":            2,
                        "information_request": ["labs", "medications", "microbiology"],
                    })
                    enriched = result.get("observation")
                    if enriched:
                        obs = enriched
            if task_id == 4:
                for adv_action in _T4_ADVANCE_DEFAULTS:
                    try:
                        r4 = _env_post(env_url, "/step", adv_action)
                        if r4.get("done"):
                            break
                        enriched = r4.get("observation")
                        if enriched:
                            obs = enriched
                    except Exception:
                        break
            rows.append({
                "prompt": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": format_observation(obs, task_id=task_id)},
                ],
                "hadm_id": str(obs.get("hadm_id", "")),
            })
        except Exception as exc:
            fail += 1
            if fail <= 3:
                print(f"  [seed] episode {i} failed: {exc}", file=sys.stderr)

    built = len(rows)
    print(f"[seed] Done: {built}/{n}  (failed={fail})", flush=True)
    if logger:
        logger.log_seed_build(task_id, built, n)
    if not rows:
        raise RuntimeError(
            f"[seed] All {n} episodes failed for task={task_id}. "
            "Check env_url and server logs before training."
        )
    return Dataset.from_list(rows)


# ── Reward function ───────────────────────────────────────────────────────────

def _comp_text(c: Any) -> str:
    """Extract plain text from a completion that may be a string or a TRL
    message-list (list of dicts with role/content keys).  TRL >= 0.12 passes
    message-list completions when prompts are in conversational format."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        for msg in c:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        # fallback: join all content fields
        return " ".join(str(msg.get("content", "")) for msg in c if isinstance(msg, dict))
    return str(c)


def make_reward_fn(
    env_url:         str,
    step_counter:    List[int],
    chunk_counter:   List[int],
    zero_bufs:       Dict[int, Deque[bool]],
    logger:          TrainingLogger,
    max_steps:       int,
    task_id:         int,
    noise_level:     str,
    curriculum_mode: str,
    sample_every:    int = 32,
    pool_sizes:      Optional[Dict[str, int]] = None,
) -> Any:
    """Build the reward function with a FIXED task locked at chunk start.

    task_id / noise_level / curriculum_mode are captured in the closure and do
    NOT change during the chunk's trainer.train() call.  This prevents the
    mid-chunk curriculum switch that collapsed all rewards to 0.030 when
    step_counter crossed a phase boundary while the seed dataset was still the prior task's prompts.
    """

    # Import here to avoid circular imports at module level
    from training.rollout_collector import _normalize_action as _norm_action

    env_w, fmt_w = _reward_weights(task_id)

    def reward_fn(
        prompts:     List[Any],
        completions: List[Any],
        hadm_id:     Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        rewards:  List[float] = []
        parse_ok: List[bool]  = []

        # Normalise completions to plain strings (TRL passes message-list dicts
        # when prompts are in conversational format).
        texts = [_comp_text(c) for c in completions]

        # Truncation monitor
        n_truncated = sum(
            1 for t in texts
            if t.rstrip().count("{") > t.rstrip().count("}")
        )
        if n_truncated > len(texts) // 2:
            print(
                f"  [TRUNC WARN] {n_truncated}/{len(texts)} completions "
                f"appear truncated (unclosed JSON). "
                f"Consider raising MAX_COMP_LENGTH={MAX_COMP_LENGTH}.",
                file=sys.stderr, flush=True,
            )

        # Periodic sample
        emit_sample = bool(texts) and (
            (step_counter[0] % sample_every) < len(texts)
        )
        sample_completion: Optional[str] = texts[0] if emit_sample else None

        print(
            f"  [reward] step={step_counter[0]}  task={task_id}  "
            f"n={len(texts)}  scoring...",
            flush=True,
        )

        # Score each completion
        for i, text in enumerate(texts):
            action_dict = _extract_json(text)
            parse_failed = action_dict is None
            parse_ok.append(not parse_failed)

            # Format score: non-zero even on parse failure → always some gradient
            fmt_score = _format_score(action_dict, task_id)

            # Short-circuit: parse failure → fmt=0, env=0, no round-trip needed
            if parse_failed:
                rewards.append(0.0)
                if task_id in zero_bufs:
                    zero_bufs[task_id].append(True)
                print(
                    f"    [{i+1}/{len(texts)}] parse=FAIL  reward=0.000  "
                    f"preview={repr(text[:60])}",
                    flush=True,
                )
                continue

            action_dict["task_id"] = task_id
            # Normalise disposition string: spaces/hyphens → underscores, uppercase
            # so "home with services" and "HOME WITH SERVICES" both become
            # "HOME_WITH_SERVICES" before the env round-trip.
            if task_id == 1:
                sub1 = action_dict.get("task1", {})
                if isinstance(sub1, dict) and sub1.get("disposition"):
                    sub1["disposition"] = (
                        sub1["disposition"].strip().upper().replace(" ", "_").replace("-", "_")
                    )
            # Fix common LLM formatting mistakes (flat fields, wrong types, etc.)
            action_dict = _norm_action(action_dict, task_id)

            try:
                episode_mode = _pick_mode(curriculum_mode, pool_sizes or {})
                # Pin the env to the same patient the model saw in the prompt.
                # The seed dataset stores hadm_id per row; TRL forwards it here
                # as the hadm_id kwarg list.  Without this, /reset picks a random
                # patient and the reward doesn't measure "did you classify THIS
                # patient correctly" — just population-level distribution matching.
                reset_body: Dict = {
                    "task_id":         task_id,
                    "noise_level":     noise_level,
                    "curriculum_mode": episode_mode,
                }
                if hadm_id is not None and i < len(hadm_id) and hadm_id[i]:
                    try:
                        reset_body["hadm_id"] = int(hadm_id[i])
                    except (ValueError, TypeError):
                        pass  # fall back to random if id is malformed
                _env_post(env_url, "/reset", reset_body)
                if task_id == 2:
                    _env_post(env_url, "/step", {
                        "task_id":            2,
                        "information_request": ["labs", "medications", "microbiology"],
                    })
                    result = _env_post(env_url, "/step", action_dict)
                elif task_id == 4:
                    # Advance env steps 1-9 with minimal defaults (reward=0 each),
                    # then submit the model's final_note as step 10 for the reward.
                    for adv in _T4_ADVANCE_DEFAULTS:
                        r4 = _env_post(env_url, "/step", adv)
                        if r4.get("done"):
                            break
                    t4 = action_dict.get("task4", {})
                    final_note = ""
                    if isinstance(t4, dict):
                        final_note = (t4.get("final_note") or
                                      t4.get("discharge_note") or
                                      t4.get("note") or "")
                    result = _env_post(env_url, "/step", {
                        "task_id": 4,
                        "task4":   {"final_note": str(final_note)},
                    })
                else:
                    result = _env_post(env_url, "/step", action_dict)
                env_reward = float(result.get("reward", 0.0))
            except Exception as exc:
                print(f"  [reward] env error: {exc}", file=sys.stderr, flush=True)
                env_reward = 0.0

            # Blended reward: format gives gradient early; env dominates later
            reward = round(env_w * env_reward + fmt_w * fmt_score, 5)

            print(
                f"    [{i+1}/{len(texts)}] parse=OK  "
                f"env={env_reward:.3f}  fmt={fmt_score:.2f}  reward={reward:.4f}",
                flush=True,
            )

            rewards.append(reward)
            if task_id in zero_bufs:
                zero_bufs[task_id].append(reward < ZERO_REWARD_THRESHOLD)

        # Log disposition distribution for task 1 so mode collapse is visible immediately
        if task_id == 1:
            from collections import Counter as _Counter
            disps = []
            for text in texts:
                d = _extract_json(text)
                if d:
                    sub = d.get("task1", {})
                    if isinstance(sub, dict) and sub.get("disposition"):
                        disps.append(sub["disposition"].lower().strip())
            if disps:
                counts = _Counter(disps)
                n_uniq = len(counts)
                d_str  = "  ".join(
                    f"{dsp}:{cnt/len(disps):.0%}" for dsp, cnt in counts.most_common(5)
                )
                collapse_tag = "  !! MODE COLLAPSE" if n_uniq == 1 else ""
                print(
                    f"  [disp] unique={n_uniq}/{len(disps)}  {d_str}{collapse_tag}",
                    flush=True,
                )

        step_counter[0] += 1  # count optimizer steps, not completions

        logger.log_reward_batch(
            global_step=step_counter[0],
            chunk=chunk_counter[0],
            max_steps=max_steps,
            task_id=task_id,
            noise_level=noise_level,
            curriculum_mode=curriculum_mode,
            rewards=rewards,
            parse_ok=parse_ok,
            sample_response=sample_completion,
            n_truncated=n_truncated,
        )

        # Zero-rate warnings (only after buffer is meaningful)
        for tid, buf in zero_bufs.items():
            if len(buf) >= ZERO_WARN_AFTER:
                rate = sum(buf) / len(buf)
                if rate > 0.40:
                    logger.log_zero_warn(tid, rate, len(buf))

        return rewards

    return reward_fn


# ── TRL log callback ─────────────────────────────────────────────────────────

class _TRLLogCallback(TrainerCallback):
    """Intercepts TRL on_log to capture metrics into our logger."""
    def __init__(self, logger: TrainingLogger, step_counter: List[int]) -> None:
        self._logger       = logger
        self._step_counter = step_counter

    def on_log(self, args: Any, state: Any, control: Any,
               logs: Optional[Dict] = None, **kwargs) -> None:
        if logs:
            self._logger.log_trl_metrics(self._step_counter[0], logs)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model_name:  str           = "Qwen/Qwen2.5-3B-Instruct",
    env_url:     str           = "http://localhost:7860",
    output_dir:  str           = "./checkpoints",
    log_dir:     str           = "./logs",
    max_steps:   int           = 550,
    eval_every:  int           = 50,
    batch_size:  int           = BATCH_SIZE,
    grad_accum:  int           = GRAD_ACCUM,
    lr:          float         = 5e-6,
    seed_n:      int           = 50,
    resume_from: Optional[str] = None,
) -> None:

    print("\n" + "=" * 68, flush=True)
    print("  MIMIC Discharge Planning - GRPO Training", flush=True)
    print(f"  model        : {model_name}", flush=True)
    print(f"  env_url      : {env_url}", flush=True)
    print(f"  output_dir   : {output_dir}", flush=True)
    print(f"  max_steps    : {max_steps}", flush=True)
    print(f"  chunk_size   : {eval_every}", flush=True)
    print(f"  batch_size   : {batch_size}  grad_accum={grad_accum}  "
          f"effective={batch_size * grad_accum}", flush=True)
    print(f"  num_gen      : {NUM_GENERATIONS}  "
          f"comp_len={MAX_COMP_LENGTH}  prompt_len={MAX_PROMPT_LENGTH}", flush=True)
    print(f"  resume_from  : {resume_from or 'None (fresh run)'}", flush=True)
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"  GPU          : {dev.name}  "
              f"{dev.total_memory/1024**3:.1f} GB", flush=True)
    print("=" * 68 + "\n", flush=True)

    model, tokenizer = load_model_and_tokenizer(model_name, resume_from)

    logger        = TrainingLogger(log_dir, resume=resume_from is not None)
    step_counter  = [0]
    chunk_counter = [0]
    zero_bufs: Dict[int, Deque[bool]] = {
        1: deque(maxlen=50),
        2: deque(maxlen=50),
        3: deque(maxlen=50),
        4: deque(maxlen=50),
    }
    t0             = time.time()
    elapsed_offset = 0.0

    # Fetch patient pool sizes once so curriculum mode can be validated
    # against the real distribution instead of blindly using easy_only/medium_only.
    pool_sizes = fetch_pool_sizes(env_url)
    total_pts  = sum(pool_sizes.values())
    print(
        f"[pool] Patient complexity distribution — "
        f"easy={pool_sizes['easy']}  medium={pool_sizes['medium']}  "
        f"hard={pool_sizes['hard']}  total={total_pts}  "
        f"(MIN_TIER_POOL={MIN_TIER_POOL})",
        flush=True,
    )
    easy   = pool_sizes.get("easy",   0)
    medium = pool_sizes.get("medium", 0)
    hard   = pool_sizes.get("hard",   0)
    print(
        f"[pool] Curriculum tiers — easy={easy} (all HOME, trivial) | "
        f"medium={medium} (56% HWS / 39% HOME, real discrimination) | "
        f"hard={hard} (SNF/expired/rehab/hospice)",
        flush=True,
    )
    print(
        f"[pool] Phase 1 uses medium_only → model must distinguish HOME vs "
        f"HOME_WITH_SERVICES from clinical features (not just 'always HOME').",
        flush=True,
    )
    if medium < MIN_TIER_POOL:
        print(
            f"[pool] WARNING: medium tier only {medium} patients (need ≥{MIN_TIER_POOL}) "
            f"— Phase 1 will fall back to 'random'.",
            flush=True,
        )

    if resume_from:
        elapsed_offset = load_train_state(
            resume_from, step_counter, chunk_counter, zero_bufs,
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prev_task_id = None

    while step_counter[0] < max_steps:
        chunk   = chunk_counter[0]
        task_id, noise_level, curr_mode = _curriculum(step_counter[0], pool_sizes)

        if task_id != prev_task_id:
            logger.log_phase_start(
                _phase_of(task_id), task_id, noise_level,
                curr_mode, step_counter[0],
            )
            prev_task_id = task_id

        logger.log_chunk_start(
            chunk, step_counter[0], max_steps, task_id, noise_level, curr_mode,
        )

        # Build reward_fn fresh each chunk with the task LOCKED so it cannot
        # switch mid-chunk as step_counter advances past a phase boundary.
        reward_fn = make_reward_fn(
            env_url, step_counter, chunk_counter, zero_bufs, logger, max_steps,
            task_id=task_id, noise_level=noise_level, curriculum_mode=curr_mode,
            pool_sizes=pool_sizes,
        )

        chunk_seed_n = _chunk_seed_n(curr_mode, pool_sizes, seed_n)
        print(f"[seed] Task {task_id} mode={curr_mode}  seed_n={chunk_seed_n}  "
              f"(floor={seed_n}, auto=2×pool)", flush=True)
        dataset = build_seed_dataset(
            env_url, task_id, chunk_seed_n, noise_level, curr_mode, logger,
            pool_sizes=pool_sizes,
        )

        # Phase-specific temperature: high early to escape local optima (hospice collapse),
        # lower later when the model already produces diverse structured output.
        phase_temperature = {1: 1.3, 2: 1.1, 3: 0.9}.get(task_id, 0.9)

        grpo_cfg = GRPOConfig(
            output_dir=str(_ckpt_dir(output_dir, chunk)),
            max_steps=eval_every,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            logging_steps=5,
            save_steps=eval_every // 2,
            warmup_steps=10,
            num_generations=NUM_GENERATIONS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMP_LENGTH,
            temperature=phase_temperature,
            top_p=0.95,
            beta=0.04,               # KL penalty: keeps policy close to base model's diversity
            top_entropy_quantile=0.8, # drop bottom 20% entropy completions per group
            seed=42,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_fn],
            args=grpo_cfg,
            train_dataset=dataset,
            callbacks=[_TRLLogCallback(logger, step_counter)],
        )

        print(f"\n[trainer] Chunk {chunk:03d}  "
              f"steps {step_counter[0]}->{step_counter[0]+eval_every}",
              flush=True)
        trainer.train()
        print(f"[trainer] Chunk {chunk:03d} done.", flush=True)

        # Save model + state
        ckpt_path = str(_ckpt_dir(output_dir, chunk))
        trainer.save_model(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        save_train_state(
            output_dir, chunk, step_counter,
            chunk_counter, zero_bufs, t0, logger,
        )

        # Dead-gradient check: stop if the CURRENT task's near-zero rate is
        # consistently above its threshold.  Thresholds are higher for tasks
        # 2/3 because partial credit is harder to obtain early in those phases.
        dead = False
        for tid, buf in zero_bufs.items():
            if len(buf) < 50:
                continue
            zr = sum(buf) / len(buf)
            threshold = DEAD_THRESHOLDS.get(tid, 0.90)
            if tid == task_id and zr > threshold:
                print(
                    f"  [DEAD] task={tid}  near-zero rate={zr:.0%} > {threshold:.0%}"
                    f" — stopping training.",
                    flush=True,
                )
                logger.log_dead_gradient(tid, zr)
                dead = True
                break
            if zr > 0.90:
                logger.log_zero_warn(tid, zr, len(buf))

        if dead:
            break

        zr_now = {
            tid: (sum(buf) / len(buf) if buf else 0.0)
            for tid, buf in zero_bufs.items()
        }
        elapsed_total = (time.time() - t0 + elapsed_offset) / 60
        logger.log_chunk_summary(chunk, elapsed_total, zr_now)

        saved_plots = plot_all(logger.all_records, logger.trl_records, log_dir)
        if saved_plots:
            print(f"  [viz] {len(saved_plots)} plots -> {log_dir}/plots/", flush=True)

        chunk_counter[0] += 1

    # Final save
    final_dir = str(Path(output_dir) / "final")
    print(f"\n[save] Writing final model -> {final_dir}", flush=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    plot_all(logger.all_records, logger.trl_records, log_dir)
    logger.log_training_end(final_dir, step_counter[0])
    logger.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO training — MIMIC Discharge Planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name",  default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--env_url",     default="https://iinovaii-mimic-discharge-env-v2.hf.space")
    parser.add_argument("--output_dir",  default="./checkpoints")
    parser.add_argument("--log_dir",     default="./logs")
    parser.add_argument("--max_steps",   type=int,   default=550,
                        help="Total training steps. 7-hour L4 budget: T1=0-199(200), "
                             "T2=200-349(150), T3=350-449(100), T4=450-549(100). "
                             "T1/T2 ~35s/step; T3/T4 ~55s/step due to longer outputs.")
    parser.add_argument("--eval_every",  type=int,   default=50,
                        help="Steps per training chunk (checkpoint + seed rebuild frequency)")
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE,
                        help="Per-device batch (L4 24GB: keep at 2)")
    parser.add_argument("--grad_accum",  type=int,   default=GRAD_ACCUM,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--seed_n",      type=int,   default=50,
                        help="Minimum seed episodes per chunk. Auto-scales to 2× the "
                             "active patient pool (T1/T2 medium→220, T3 all→466, T4 hard→210). "
                             "Pass a larger value to override the auto-minimum.")
    parser.add_argument("--resume_from", default=None,
                        help="Path to chunk_NNN dir to resume from")
    parser.add_argument("--replot",      default=None,
                        help="Path to JSONL log — re-generates plots and exits")
    args = parser.parse_args()

    if args.replot:
        records: List[Dict] = []
        with open(args.replot) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        print(f"[replot] {len(records)} records from {args.replot}")
        saved = plot_all(records, [], args.log_dir)
        print(f"[replot] {len(saved)} plots -> {args.log_dir}/plots/")
        return

    train(
        model_name=args.model_name,
        env_url=args.env_url,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed_n=args.seed_n,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()