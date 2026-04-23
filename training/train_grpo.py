"""
GRPO training — MIMIC Discharge Planning
=========================================
Hardware target : 1x NVIDIA L4  (24 GB VRAM, 28 vCPU, 124 GB RAM)
Model           : Qwen/Qwen2.5-3B-Instruct  (bfloat16, LoRA r=16)
Framework       : TRL >= 0.12  +  Transformers >= 4.40

Curriculum
----------
  Steps    0 -  999 : Task 1  noise=clean    curriculum=random
  Steps 1000 - 2999 : Task 2  noise=clean    curriculum=random
  Steps 3000+       : Task 3  noise=partial  curriculum=random
  (Tier-based modes like easy_only/medium_only require a large patient pool;
   the demo dataset has ~275 admissions so "easy" filtered to only 2 patients.)

Checkpointing & Resume
-----------------------
  After every chunk: HuggingFace model checkpoint + train_state.json saved.
  To resume a crashed run:

      python -m training.train_grpo --resume_from ./checkpoints/chunk_004

  This reloads LoRA weights + full training state (step, chunk,
  zero-reward buffers, elapsed time) and continues exactly where it stopped.

Visualisations  (logs/plots/, updated every chunk)
---------------------------------------------------
  01_reward_curve.png      raw + rolling-50 reward over time
  02_parse_rate.png        JSON parse-success rate
  03_dead_gradient.png     Task-1 zero-reward monitor
  04_reward_by_task.png    box-plot per task
  05_phase_timeline.png    curriculum phases + reward
  06_reward_histogram.png  reward distribution with banded zones
  07_chunk_summary.png     per-chunk mean reward bar chart
  08_vram_usage.png        GPU VRAM over time (MB)
  09_entropy_loss.png      TRL entropy + loss + clipped-ratio

All bugs fixed vs previous versions
-------------------------------------
  reward_funcs=[fn]         list required in TRL >= 0.9
  completions not responses TRL >= 0.12 rename
  processing_class          TRL >= 0.9 rename from tokenizer
  warmup_steps not ratio    TRL >= 5.2
  max_completion_length=512 was 256; JSON was truncated 100% of the time
  num_generations=8         was 4; advantage washed out with 4 gens
  batch_size=2 / accum=8   L4 VRAM budget (effective batch stays 16)
  Trainer re-instantiated   avoids stale LR scheduler + dataloader
  Zero-warn threshold=40    avoids false alarms in early steps
  Truncation monitor        warns when >50% of completions are cut off
  _normalize_action called  fixes flat/mistyped JSON before env step
  _format_score blended     non-zero reward for valid JSON structure
  temperature=0.9 / top_p   encourages diverse generations for GRPO
  dead-gradient threshold   raised 60%→80% (was too aggressive early)

Usage
-----
  # Fresh run
  python -m training.train_grpo

  # Resume
  python -m training.train_grpo --resume_from ./checkpoints/chunk_004

  # Re-plot only
  python -m training.train_grpo --replot ./logs/training_20240101_120000.jsonl
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

# ── Silence noisy but harmless deprecation warnings ───────────────────────────
# 1. Transformers 5.3 deprecated AttentionMaskConverter; removed in 5.10
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
# 2. Unsloth/TRL passes both generation_config + loose kwargs in Transformers 5.3
#    max_new_tokens=512 correctly takes precedence; warning is cosmetic only
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
MIN_TIER_POOL     = 25    # minimum patients in a complexity tier to use tier-based curriculum


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

    When easy_only is below MIN_TIER_POOL, return "easy_medium" so the caller
    can sample proportionally from easy+medium patients, keeping hard cases out
    of Phase 1.  Other under-sized tiers fall back to "random".
    """
    tier_map = {"easy_only": "easy", "medium_only": "medium", "hard_only": "hard"}
    tier = tier_map.get(preferred_mode)
    if tier and pool_sizes.get(tier, 0) < MIN_TIER_POOL:
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

# Desired tier per phase.  _resolve_mode() will downgrade to "random" at
# runtime if the actual patient count in a tier is below MIN_TIER_POOL.
_CURRICULUM_PREFERRED = {
    # (step_lo, step_hi): (task_id, noise_level, preferred_curriculum_mode)
    (0,    999):  (1, "clean",   "easy_only"),
    (1000, 2999): (2, "clean",   "medium_only"),
    (3000, 9999): (3, "partial", "random"),
}


def _curriculum(step: int, pool_sizes: Optional[Dict[str, int]] = None) -> Tuple[int, str, str]:
    """Return (task_id, noise_level, curriculum_mode) for a global step.

    If pool_sizes is provided (fetched from the env at startup), the
    preferred curriculum mode is validated against the real patient counts
    and downgraded to "random" when the tier is too small to provide
    meaningful diversity.
    """
    if step < 1000:
        tid, noise, preferred = 1, "clean",   "easy_only"
    elif step < 3000:
        tid, noise, preferred = 2, "clean",   "medium_only"
    else:
        tid, noise, preferred = 3, "partial", "random"

    mode = _resolve_mode(preferred, pool_sizes) if pool_sizes else preferred
    return tid, noise, mode


def _phase_of(task_id: int) -> int:
    return {1: 1, 2: 2, 3: 3}.get(task_id, 1)


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

_TASK_COLORS  = {1: "#4C8CF5", 2: "#F5A623", 3: "#7ED321"}
_PHASE_COLORS = {1: "#DDEEFF", 2: "#FFF3CD", 3: "#D5F5E3"}


def _rolling(arr: List[float], w: int) -> List[float]:
    out = []
    for i in range(len(arr)):
        win = arr[max(0, i - w + 1): i + 1]
        out.append(float(sum(win) / len(win)))
    return out


def _setup_ax(ax: Any, title: str, xlabel: str, ylabel: str,
              ylim: Optional[Tuple] = None) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.tick_params(labelsize=8)


def _shade_phases(ax: Any, steps: List[int]) -> None:
    if not steps:
        return
    max_s = max(steps)
    for i, (lo, hi) in enumerate([(0, 1000), (1000, 3000), (3000, max_s + 1)]):
        clamp = min(hi, max_s)
        if clamp > lo:
            ax.axvspan(lo, clamp, alpha=0.07,
                       color=list(_PHASE_COLORS.values())[i], zorder=0)


def _save_fig(fig: Any, path: str) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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

    out_dir = Path(log_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    steps        = [r["step"]             for r in records]
    mean_rewards = [r["mean_reward"]      for r in records]
    parse_rates  = [r["parse_ok_rate"]    for r in records]
    zero_rates   = [r["zero_reward_rate"] for r in records]
    task_ids     = [r["task_id"]          for r in records]
    vram_mb      = [r.get("vram_alloc_mb", 0) for r in records]

    sm50  = _rolling(mean_rewards, 50)
    sm100 = _rolling(parse_rates, 100)
    sm50z = _rolling(zero_rates, 50)

    # 01 Reward Curve
    fig, ax = plt.subplots(figsize=(12, 4))
    _shade_phases(ax, steps)
    ax.plot(steps, mean_rewards, alpha=0.15, color="steelblue",
            linewidth=0.6, label="raw")
    ax.plot(steps, sm50, color="steelblue", linewidth=2.0, label="rolling-50")
    for tid, col in _TASK_COLORS.items():
        xs = [s for s, t in zip(steps, task_ids) if t == tid]
        ys = [r for r, t in zip(mean_rewards, task_ids) if t == tid]
        if xs:
            ax.scatter(xs, ys, s=4, color=col, alpha=0.22, zorder=2)
    legend_els = [Patch(facecolor=_TASK_COLORS[t], label=f"Task {t}", alpha=0.6)
                  for t in sorted(_TASK_COLORS)]
    legend_els.append(plt.Line2D([0], [0], color="steelblue", lw=2, label="rolling-50"))
    ax.legend(handles=legend_els, fontsize=8, loc="upper left")
    _setup_ax(ax, "01 - Reward Curve", "Global step", "Mean reward", (0, 1.05))
    p = str(out_dir / "01_reward_curve.png"); _save_fig(fig, p); saved.append(p)

    # 02 JSON Parse Rate
    fig, ax = plt.subplots(figsize=(12, 4))
    _shade_phases(ax, steps)
    ax.fill_between(steps, parse_rates, alpha=0.10, color="darkorange")
    ax.plot(steps, parse_rates, alpha=0.15, color="darkorange", linewidth=0.6)
    ax.plot(steps, sm100, color="darkorange", linewidth=2.0, label="rolling-100")
    ax.axhline(0.80, color="red",      linestyle="--", lw=1.2, label="80% target")
    ax.axhline(0.50, color="goldenrod", linestyle=":",  lw=1.0, label="50% floor")
    ax.legend(fontsize=8)
    _setup_ax(ax, "02 - JSON Parse Success Rate", "Global step", "Parse rate", (0, 1.05))
    p = str(out_dir / "02_parse_rate.png"); _save_fig(fig, p); saved.append(p)

    # 03 Dead-Gradient Monitor
    fig, ax = plt.subplots(figsize=(12, 4))
    t1_steps = [s for s, t in zip(steps, task_ids) if t == 1]
    t1_zero  = [z for z, t in zip(zero_rates, task_ids) if t == 1]
    t1_sm    = _rolling(t1_zero, 50) if t1_zero else []
    if t1_steps:
        ax.fill_between(t1_steps, t1_zero, alpha=0.10, color="crimson")
        ax.plot(t1_steps, t1_zero, alpha=0.15, color="crimson", lw=0.6, label="raw T1")
        ax.plot(t1_steps, t1_sm,   color="crimson", lw=2.0, label="rolling-50")
    ax.axhspan(0.60, 1.05, alpha=0.07, color="red")
    ax.axhspan(0.40, 0.60, alpha=0.05, color="goldenrod")
    ax.axhline(0.60, color="black",     linestyle="--", lw=1.5, label="60% HALT")
    ax.axhline(0.40, color="goldenrod", linestyle=":",  lw=1.2, label="40% WARN")
    ax.legend(fontsize=8)
    _setup_ax(ax, "03 - Dead-Gradient Monitor (Task 1)",
              "Global step (Task 1 only)", "Zero-reward rate", (0, 1.05))
    p = str(out_dir / "03_dead_gradient.png"); _save_fig(fig, p); saved.append(p)

    # 04 Reward by Task (boxplot)
    task_reward_map: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    for r in records:
        tid = r["task_id"]
        if tid in task_reward_map:
            task_reward_map[tid].extend(r.get("rewards", [r["mean_reward"]]))
    present = sorted(t for t in task_reward_map if task_reward_map[t])
    if present:
        fig, ax = plt.subplots(figsize=(7, 5))
        bp = ax.boxplot(
            [task_reward_map[t] for t in present],
            labels=[f"Task {t}" for t in present],
            patch_artist=True, notch=False, widths=0.45,
        )
        for patch, t in zip(bp["boxes"], present):
            patch.set_facecolor(_TASK_COLORS.get(t, "#AAAAAA"))
            patch.set_alpha(0.70)
        for el in ["whiskers", "caps", "fliers"]:
            for item in bp[el]:
                item.set(color="#444", linewidth=1.1)
        for med in bp["medians"]:
            med.set(color="white", linewidth=2.0)
        for i, t in enumerate(present, 1):
            m = sum(task_reward_map[t]) / len(task_reward_map[t])
            ax.plot(i, m, "D", color="red", markersize=6, zorder=5,
                    label="mean" if i == 1 else "")
        ax.legend(fontsize=8)
        _setup_ax(ax, "04 - Reward Distribution by Task",
                  "Task", "Reward", (0, 1.05))
        p = str(out_dir / "04_reward_by_task.png"); _save_fig(fig, p); saved.append(p)

    # 05 Phase Timeline
    fig, ax = plt.subplots(figsize=(12, 4))
    if steps:
        max_s = max(steps)
        for lo, hi, label, color in [
            (0,    min(1000, max_s), "Phase 1 / Task 1", _PHASE_COLORS[1]),
            (1000, min(3000, max_s), "Phase 2 / Task 2", _PHASE_COLORS[2]),
            (3000, max_s,            "Phase 3 / Task 3", _PHASE_COLORS[3]),
        ]:
            if hi > lo:
                ax.axvspan(lo, hi, alpha=0.35, color=color, label=label)
                ax.text((lo + hi) / 2, 1.02, label, ha="center", va="bottom",
                        fontsize=7, transform=ax.get_xaxis_transform(), alpha=0.7)
        ax.plot(steps, sm50, color="#1a1a2e", lw=2.0, label="rolling-50")
        for b in [1000, 3000]:
            if b < max_s:
                ax.axvline(b, color="#555", linestyle="--", lw=1.0, alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    _setup_ax(ax, "05 - Curriculum Phase Timeline + Reward",
              "Global step", "Rolling mean reward", (0, 1.05))
    p = str(out_dir / "05_phase_timeline.png"); _save_fig(fig, p); saved.append(p)

    # 06 Reward Histogram
    all_rewards: List[float] = []
    for r in records:
        all_rewards.extend(r.get("rewards", [r["mean_reward"]]))
    if all_rewards:
        fig, ax = plt.subplots(figsize=(9, 5))
        for lo, hi, label, color in [
            (0.00, 0.10, "near-zero", "#ffcccc"),
            (0.10, 0.40, "partial",   "#fff3cc"),
            (0.40, 0.70, "good",      "#d5f5e3"),
            (0.70, 1.01, "excellent", "#a9dfbf"),
        ]:
            ax.axvspan(lo, hi, alpha=0.25, color=color, label=label)
        ax.hist(all_rewards, bins=40, color="steelblue",
                alpha=0.70, edgecolor="white", lw=0.4, zorder=3)
        mean_val = sum(all_rewards) / len(all_rewards)
        ax.axvline(mean_val, color="red", linestyle="--",
                   lw=1.8, label=f"mean={mean_val:.3f}")
        ax.legend(fontsize=8)
        _setup_ax(ax, "06 - Overall Reward Histogram",
                  "Reward", "Count", (0, ax.get_ylim()[1] * 1.1))
        p = str(out_dir / "06_reward_histogram.png"); _save_fig(fig, p); saved.append(p)

    # 07 Per-Chunk Summary
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
        colors = [_TASK_COLORS.get(t, "#AAAAAA") for t in c_task]
        fig, ax = plt.subplots(figsize=(max(8, len(c_ids) * 0.5 + 2), 5))
        ax.bar(c_ids, c_vals, color=colors, alpha=0.72, edgecolor="white", lw=0.5)
        ax.plot(c_ids, c_vals, "o--", color="#333", lw=1.2, markersize=5, zorder=5)
        for x, y in zip(c_ids, c_vals):
            ax.text(x, y + 0.008, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        legend_els = [Patch(facecolor=_TASK_COLORS[t], label=f"Task {t}", alpha=0.72)
                      for t in sorted(_TASK_COLORS) if t in set(c_task)]
        ax.legend(handles=legend_els, fontsize=8)
        ax.set_xticks(c_ids)
        ax.set_xticklabels([f"C{c}" for c in c_ids], fontsize=7, rotation=45)
        top = max(c_vals) * 1.18 + 0.05 if c_vals else 1.0
        _setup_ax(ax, "07 - Per-Chunk Mean Reward", "Chunk", "Mean reward", (0, top))
        p = str(out_dir / "07_chunk_summary.png"); _save_fig(fig, p); saved.append(p)

    # 08 VRAM Usage
    if any(v > 0 for v in vram_mb):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(steps, vram_mb, color="mediumpurple", lw=1.5, label="allocated MB")
        ax.fill_between(steps, vram_mb, alpha=0.12, color="mediumpurple")
        ax.axhline(24 * 1024, color="red", linestyle="--", lw=1.0, label="24 GB limit")
        ax.legend(fontsize=8)
        _setup_ax(ax, "08 - GPU VRAM Allocated (MB)", "Global step", "VRAM (MB)")
        p = str(out_dir / "08_vram_usage.png"); _save_fig(fig, p); saved.append(p)

    # 09 Entropy + Loss + Clipped Ratio
    if trl_records:
        trl_steps = [r["step"]           for r in trl_records]
        entropies = [r["entropy"]        for r in trl_records]
        losses    = [r["loss"]           for r in trl_records]
        clipped   = [r["clipped_ratio"]  for r in trl_records]

        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        a1.plot(trl_steps, entropies, color="teal", lw=1.5)
        a1.axhline(0.5, color="red", linestyle="--", lw=1.0, label="collapse threshold")
        a1.legend(fontsize=8)
        _setup_ax(a1, "09a - Entropy (higher = more diverse outputs)",
                  "Step", "Entropy")

        a2.semilogy(trl_steps, [max(abs(l), 1e-12) for l in losses],
                    color="coral", lw=1.5)
        _setup_ax(a2, "09b - |Loss| (log scale)", "Step", "|Loss|")

        a3.plot(trl_steps, clipped, color="goldenrod", lw=1.5)
        a3.axhline(0.90, color="red", linestyle="--", lw=1.2, label="danger: 90%")
        a3.legend(fontsize=8)
        _setup_ax(a3, "09c - Completion Clipped Ratio (want < 0.5)",
                  "Step", "Clipped ratio", (0, 1.05))

        fig.tight_layout()
        p = str(out_dir / "09_entropy_loss.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
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
}


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
    else:
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
            # Store as proper message list so TRL applies the chat template
            # correctly: system prompt in the <|im_start|>system turn, not user.
            # Train-inference mismatch was the main cause of JSON format failures.
            rows.append({"prompt": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": format_observation(obs, task_id=task_id)},
            ]})
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
    env_url:       str,
    step_counter:  List[int],
    chunk_counter: List[int],
    zero_bufs:     Dict[int, Deque[bool]],
    logger:        TrainingLogger,
    max_steps:     int,
    sample_every:  int = 32,
    pool_sizes:    Optional[Dict[str, int]] = None,
) -> Any:

    # Import here to avoid circular imports at module level
    from training.rollout_collector import _normalize_action as _norm_action

    def reward_fn(
        prompts:     List[Any],
        completions: List[Any],
        **kwargs,
    ) -> List[float]:

        task_id, noise_level, curriculum_mode = _curriculum(step_counter[0], pool_sizes)
        env_w, fmt_w = _reward_weights(task_id)
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
            # Fix common LLM formatting mistakes (flat fields, wrong types, etc.)
            action_dict = _norm_action(action_dict, task_id)

            try:
                episode_mode = _pick_mode(curriculum_mode, pool_sizes or {})
                _env_post(env_url, "/reset", {
                    "task_id":         task_id,
                    "noise_level":     noise_level,
                    "curriculum_mode": episode_mode,
                })
                if task_id == 2:
                    _env_post(env_url, "/step", {
                        "task_id":            2,
                        "information_request": ["labs", "medications", "microbiology"],
                    })
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
                zero_bufs[task_id].append(reward == 0.0)

        step_counter[0] += len(rewards)

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
    max_steps:   int           = 5000,
    eval_every:  int           = 200,
    batch_size:  int           = BATCH_SIZE,
    grad_accum:  int           = GRAD_ACCUM,
    lr:          float         = 5e-6,
    seed_n:      int           = 256,
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
    easy = pool_sizes.get("easy", 0)
    if 0 < easy < MIN_TIER_POOL:
        medium = pool_sizes.get("medium", 0)
        pct_easy = easy / (easy + medium) * 100 if (easy + medium) else 0
        print(
            f"[pool] WARNING: 'easy' tier has only {easy} patients (need ≥{MIN_TIER_POOL}) — "
            f"Phase 1 will use 'easy_medium' mix (~{pct_easy:.0f}% easy / "
            f"~{100-pct_easy:.0f}% medium) to avoid training on hard cases.",
            flush=True,
        )

    if resume_from:
        elapsed_offset = load_train_state(
            resume_from, step_counter, chunk_counter, zero_bufs,
        )

    reward_fn = make_reward_fn(
        env_url, step_counter, chunk_counter, zero_bufs, logger, max_steps,
        pool_sizes=pool_sizes,
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

        dataset = build_seed_dataset(
            env_url, task_id, seed_n, noise_level, curr_mode, logger,
            pool_sizes=pool_sizes,
        )

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
            max_prompt_length=MAX_PROMPT_LENGTH,    # left-truncate long prompts
            max_completion_length=MAX_COMP_LENGTH,
            temperature=0.9,   # encourage diverse outputs; helps gradient flow
            top_p=0.95,
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

        # Dead-gradient check (threshold relaxed: model needs time to learn JSON)
        dead = False
        for tid, buf in zero_bufs.items():
            if len(buf) < 50:
                continue
            zr = sum(buf) / len(buf)
            if tid == task_id == 1 and zr > 0.80:
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
    parser.add_argument("--max_steps",   type=int,   default=5000)
    parser.add_argument("--eval_every",  type=int,   default=200,
                        help="Steps per training chunk")
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE,
                        help="Per-device batch (L4 24GB: keep at 2)")
    parser.add_argument("--grad_accum",  type=int,   default=GRAD_ACCUM,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--seed_n",      type=int,   default=256,
                        help="Seed episodes per chunk")
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