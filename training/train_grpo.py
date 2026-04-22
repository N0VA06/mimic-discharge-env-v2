"""
GRPO training script for MIMIC Discharge Planning.
Uses Unsloth (optional) + TRL GRPOTrainer.

Curriculum phases:
  Steps    0-999:  Task 1, noise=clean,   curriculum=easy_only
  Steps 1000-2999: Task 2, noise=partial, curriculum=medium_only
  Steps  3000+:   Task 3, noise=noisy,   curriculum=random

Dead-gradient guard:
  If Task 1 zero-reward rate > 60% over the last 50 rollouts → halt with warning.

Usage:
    python -m training.train_grpo \\
        --model_name Qwen/Qwen2.5-3B-Instruct \\
        --env_url http://localhost:7860 \\
        --output_dir ./checkpoints \\
        --log_dir ./logs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import requests
import torch

try:
    from unsloth import FastLanguageModel
    _UNSLOTH = True
except ImportError:
    _UNSLOTH = False

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

ENV_TIMEOUT = 30

# ─── Curriculum config ────────────────────────────────────────────────────────
# Task 4 is excluded — its sparse reward (only step 10 signals) and 10-step
# trajectory add noise without clean gradient signal at this training scale.

def _curriculum(step: int) -> Tuple[int, str, str]:
    """Returns (task_id, noise_level, curriculum_mode) for training step."""
    if step < 1000:
        return 1, "clean",   "easy_only"
    elif step < 3000:
        return 2, "clean",   "medium_only"   # clean (not partial) so meds are populated
    else:
        return 3, "partial", "random"


# ─── Training logger ──────────────────────────────────────────────────────────

class TrainingLogger:
    """Writes one JSONL record per reward batch; generates matplotlib plots."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{ts}.jsonl"
        self._fh = open(self.log_file, "w", buffering=1)

    def log(
        self,
        global_step: int,
        task_id: int,
        noise_level: str,
        curriculum_mode: str,
        rewards: List[float],
        parse_ok: List[bool],
    ) -> None:
        record = {
            "ts": time.time(),
            "step": global_step,
            "task_id": task_id,
            "noise_level": noise_level,
            "curriculum_mode": curriculum_mode,
            "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
            "rewards": [round(r, 4) for r in rewards],
            "parse_ok_rate": float(sum(parse_ok) / len(parse_ok)) if parse_ok else 0.0,
            "zero_reward_rate": float(sum(r == 0.0 for r in rewards) / len(rewards)) if rewards else 1.0,
        }
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()

    def plot(self, output_dir: Optional[str] = None) -> str:
        """Generate and save training visualizations. Returns output path."""
        return plot_training_run(str(self.log_file), output_dir or str(self.log_dir))


# ─── Visualization ────────────────────────────────────────────────────────────

_TASK_COLORS = {1: "#4C8CF5", 2: "#F5A623", 3: "#7ED321", 4: "#9B59B6"}
_PHASE_COLORS = {1: "#DDEEFF", 2: "#FFF3CD", 3: "#D5F5E3"}


def plot_training_run(log_file: str, output_dir: str) -> str:
    """
    Read JSONL log and produce a 6-panel training dashboard PNG.
    Returns the path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    records: List[Dict] = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    if not records:
        return ""

    steps       = [r["step"] for r in records]
    mean_rewards = [r["mean_reward"] for r in records]
    parse_rates  = [r["parse_ok_rate"] for r in records]
    zero_rates   = [r["zero_reward_rate"] for r in records]
    task_ids     = [r["task_id"] for r in records]

    def _rolling(arr: List[float], w: int = 50) -> List[float]:
        out = []
        for i in range(len(arr)):
            window = arr[max(0, i - w + 1): i + 1]
            out.append(float(sum(window) / len(window)))
        return out

    smooth_rewards = _rolling(mean_rewards, 50)
    smooth_parse   = _rolling(parse_rates, 100)
    smooth_zero    = _rolling(zero_rates, 50)

    # Collect all individual rewards per task for boxplot
    task_reward_map: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}
    for r in records:
        tid = r["task_id"]
        task_reward_map[tid].extend(r.get("rewards", [r["mean_reward"]]))

    # Phase boundary indices (step values where curriculum changes)
    phase_boundaries = [1000, 3000]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("GRPO Training Dashboard — MIMIC Discharge Planning", fontsize=14, fontweight="bold")
    ax_reward, ax_parse, ax_zero = axes[0]
    ax_box, ax_phase, ax_hist = axes[1]

    # ── Panel 1: Reward curve ─────────────────────────────────────────────────
    _shade_phases(ax_reward, steps, phase_boundaries)
    ax_reward.plot(steps, mean_rewards, alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
    ax_reward.plot(steps, smooth_rewards, color="steelblue", linewidth=1.8, label="rolling-50")
    ax_reward.set_xlabel("Global step")
    ax_reward.set_ylabel("Mean reward")
    ax_reward.set_title("Reward Curve")
    ax_reward.set_ylim(0, 1.05)
    ax_reward.legend(fontsize=8)
    ax_reward.grid(True, alpha=0.3)

    # ── Panel 2: JSON parse success rate ─────────────────────────────────────
    _shade_phases(ax_parse, steps, phase_boundaries)
    ax_parse.plot(steps, parse_rates, alpha=0.25, color="darkorange", linewidth=0.8)
    ax_parse.plot(steps, smooth_parse, color="darkorange", linewidth=1.8, label="rolling-100")
    ax_parse.set_xlabel("Global step")
    ax_parse.set_ylabel("Parse success rate")
    ax_parse.set_title("JSON Parse Success Rate")
    ax_parse.set_ylim(0, 1.05)
    ax_parse.axhline(0.80, color="red", linestyle="--", linewidth=1, alpha=0.6, label="80% target")
    ax_parse.legend(fontsize=8)
    ax_parse.grid(True, alpha=0.3)

    # ── Panel 3: Dead-gradient monitor ────────────────────────────────────────
    _shade_phases(ax_zero, steps, phase_boundaries)
    task1_steps   = [s for s, t in zip(steps, task_ids) if t == 1]
    task1_zero    = [z for z, t in zip(zero_rates, task_ids) if t == 1]
    task1_smooth  = _rolling(task1_zero, 50) if task1_zero else []
    if task1_steps:
        ax_zero.plot(task1_steps, task1_zero, alpha=0.25, color="crimson", linewidth=0.8)
        ax_zero.plot(task1_steps, task1_smooth, color="crimson", linewidth=1.8, label="Task 1 rolling-50")
    ax_zero.axhline(0.60, color="black", linestyle="--", linewidth=1.5, label="60% halt threshold")
    ax_zero.set_xlabel("Global step")
    ax_zero.set_ylabel("Zero-reward rate")
    ax_zero.set_title("Dead-Gradient Monitor (Task 1)")
    ax_zero.set_ylim(0, 1.05)
    ax_zero.legend(fontsize=8)
    ax_zero.grid(True, alpha=0.3)

    # ── Panel 4: Reward by task (boxplot) ─────────────────────────────────────
    present_tasks = sorted(t for t in task_reward_map if task_reward_map[t])
    box_data   = [task_reward_map[t] for t in present_tasks]
    box_labels = [f"Task {t}" for t in present_tasks]
    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, notch=False)
    for patch, t in zip(bp["boxes"], present_tasks):
        patch.set_facecolor(_TASK_COLORS.get(t, "#AAAAAA"))
        patch.set_alpha(0.7)
    ax_box.set_ylabel("Reward")
    ax_box.set_title("Reward Distribution by Task")
    ax_box.set_ylim(0, 1.05)
    ax_box.grid(True, axis="y", alpha=0.3)

    # ── Panel 5: Curriculum phase timeline ────────────────────────────────────
    ax_phase.set_title("Curriculum Phase Timeline + Mean Reward")
    if steps:
        max_step = max(steps)
        phase_spans = [
            (0, min(1000, max_step), "Phase 1\nTask 1", _PHASE_COLORS[1]),
            (1000, min(3000, max_step), "Phase 2\nTask 2", _PHASE_COLORS[2]),
            (3000, max_step, "Phase 3\nTask 3", _PHASE_COLORS[3]),
        ]
        for (lo, hi, label, color) in phase_spans:
            if hi > lo:
                ax_phase.axvspan(lo, hi, alpha=0.4, color=color)
                ax_phase.text((lo + hi) / 2, 0.95, label, ha="center", va="top",
                              fontsize=8, transform=ax_phase.get_xaxis_transform())
        ax_phase.plot(steps, smooth_rewards, color="steelblue", linewidth=1.5)
        ax_phase.set_xlabel("Global step")
        ax_phase.set_ylabel("Rolling mean reward")
        ax_phase.set_ylim(0, 1.05)
        ax_phase.grid(True, alpha=0.3)

    # ── Panel 6: Overall reward histogram ────────────────────────────────────
    all_rewards: List[float] = []
    for r in records:
        all_rewards.extend(r.get("rewards", [r["mean_reward"]]))
    ax_hist.hist(all_rewards, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax_hist.set_xlabel("Reward")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Overall Reward Histogram")
    ax_hist.axvline(float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0,
                    color="red", linestyle="--", linewidth=1.5, label="mean")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_stem = Path(log_file).stem
    out_path = str(out_dir / f"{log_stem}_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved training dashboard → {out_path}")
    return out_path


def _shade_phases(ax: Any, steps: List[int], boundaries: List[int]) -> None:
    """Draw light background shading for curriculum phase regions."""
    if not steps:
        return
    max_step = max(steps)
    edges = [0] + boundaries + [max_step + 1]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi > lo:
            ax.axvspan(lo, min(hi, max_step), alpha=0.06,
                       color=list(_PHASE_COLORS.values())[min(i, 2)])


# ─── Env helpers ──────────────────────────────────────────────────────────────

def _env_post(env_url: str, path: str, body: Dict) -> Dict:
    r = requests.post(f"{env_url.rstrip('/')}{path}", json=body, timeout=ENV_TIMEOUT)
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


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, max_seq_length: int = 2048):
    if _UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer

    import transformers
    from peft import LoraConfig, get_peft_model

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=16, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(base, lora_cfg), tokenizer


# ─── Seed dataset ─────────────────────────────────────────────────────────────

def build_seed_dataset(env_url: str, task_id: int, n: int, noise_level: str, curriculum_mode: str) -> Dataset:
    from training.rollout_collector import format_observation
    rows: List[Dict] = []
    for _ in range(n):
        try:
            obs = _env_post(env_url, "/reset", {
                "task_id":         task_id,
                "noise_level":     noise_level,
                "curriculum_mode": curriculum_mode,
            })
            # Task 2: step past the info-request so the seed prompt has full medication/lab data.
            # The model is trained on step-1 observations (enriched), not step-0 (empty).
            if task_id == 2:
                meds_empty = not (obs.get("pharmacy_active") or obs.get("medications"))
                labs_empty = not obs.get("lab_flags")
                if meds_empty or labs_empty:
                    result = _env_post(env_url, "/step", {
                        "task_id": 2,
                        "information_request": ["labs", "medications", "microbiology"],
                    })
                    enriched = result.get("observation")
                    if enriched:
                        obs = enriched

            rows.append({"prompt": format_observation(obs, task_id=task_id)})
        except Exception:
            continue
    return Dataset.from_list(rows) if rows else Dataset.from_list([{"prompt": ""}])


# ─── Reward function factory ──────────────────────────────────────────────────

def make_reward_fn(
    env_url: str,
    step_counter: List[int],
    zero_buf: Deque[bool],
    logger: Optional[TrainingLogger] = None,
) -> Any:
    def reward_fn(prompts: List[str], responses: List[str], **kwargs) -> List[float]:
        task_id, noise_level, curriculum_mode = _curriculum(step_counter[0])
        rewards:   List[float] = []
        parse_ok:  List[bool]  = []

        for response in responses:
            action_dict = _extract_json(response)
            parsed = action_dict is not None
            parse_ok.append(parsed)

            if action_dict is None:
                action_dict = {"task_id": task_id}
            action_dict["task_id"] = task_id

            try:
                _env_post(env_url, "/reset", {
                    "task_id":         task_id,
                    "noise_level":     noise_level,
                    "curriculum_mode": curriculum_mode,
                })
                # Task 2: unlock lab/medication data before scoring the model's care plan.
                # Without this, the env has no meds on step 0 and all recommendations
                # are scored as hallucinations, giving near-zero gradient signal.
                if task_id == 2:
                    _env_post(env_url, "/step", {
                        "task_id": 2,
                        "information_request": ["labs", "medications", "microbiology"],
                    })
                result = _env_post(env_url, "/step", action_dict)
                reward = float(result.get("reward", 0.0))
            except Exception:
                reward = 0.0

            rewards.append(reward)
            if task_id == 1:
                zero_buf.append(reward == 0.0)

        step_counter[0] += len(rewards)

        if logger is not None:
            logger.log(
                global_step=step_counter[0],
                task_id=task_id,
                noise_level=noise_level,
                curriculum_mode=curriculum_mode,
                rewards=rewards,
                parse_ok=parse_ok,
            )

        return rewards

    return reward_fn


# ─── Training entry point ─────────────────────────────────────────────────────

def train(
    model_name:  str            = "Qwen/Qwen2.5-3B-Instruct",
    env_url:     str            = "http://localhost:7860",
    output_dir:  str            = "./checkpoints",
    log_dir:     str            = "./logs",
    max_steps:   int            = 5000,
    eval_every:  int            = 200,
    batch_size:  int            = 4,
    grad_accum:  int            = 4,
    lr:          float          = 5e-6,
    seed_n:      int            = 256,
) -> None:
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)

    logger = TrainingLogger(log_dir)
    step_counter: List[int] = [0]
    zero_buf: Deque[bool]   = deque(maxlen=50)
    reward_fn = make_reward_fn(env_url, step_counter, zero_buf, logger)

    task_id, noise_level, curr_mode = _curriculum(0)
    seed_dataset = build_seed_dataset(env_url, task_id, seed_n, noise_level, curr_mode)
    print(f"Seed dataset: {len(seed_dataset)} prompts")

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        max_steps=eval_every,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=10,
        save_steps=eval_every,
        warmup_ratio=0.05,
        num_generations=4,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=seed_dataset,
    )

    print("Starting GRPO training…")
    t0    = time.time()
    chunk = 0

    while step_counter[0] < max_steps:
        task_id, noise_level, curr_mode = _curriculum(step_counter[0])
        print(
            f"\n[Chunk {chunk}  step~{step_counter[0]}]  "
            f"task={task_id}  noise={noise_level}  curriculum={curr_mode}"
        )

        trainer.train_dataset = build_seed_dataset(
            env_url, task_id, seed_n, noise_level, curr_mode
        )
        trainer.train(resume_from_checkpoint=(chunk > 0))

        # Dead-gradient detection (Task 1 only)
        if task_id == 1 and len(zero_buf) == 50:
            zero_rate = sum(zero_buf) / len(zero_buf)
            if zero_rate > 0.60:
                print(
                    f"[WARNING] Dead gradient: {zero_rate:.1%} zero-reward rate on Task 1 "
                    f"over last 50 rollouts. Halting."
                )
                break

        # Checkpoint visualization
        logger.plot(log_dir)

        chunk += 1
        elapsed_min = round((time.time() - t0) / 60, 1)
        print(f"  Elapsed: {elapsed_min} min")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final visualization
    final_plot = logger.plot(log_dir)
    logger.close()
    print(f"Done. Model saved → {output_dir}")
    print(f"Training log  → {logger.log_file}")
    print(f"Final dashboard → {final_plot}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--env_url",     default="http://localhost:7860")
    parser.add_argument("--output_dir",  default="./checkpoints")
    parser.add_argument("--log_dir",     default="./logs",
                        help="Directory for JSONL training logs and PNG dashboards")
    parser.add_argument("--max_steps",   type=int,   default=5000)
    parser.add_argument("--eval_every",  type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--grad_accum",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--seed_n",      type=int,   default=256)
    args = parser.parse_args()

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
    )


# ─── Standalone plot utility ──────────────────────────────────────────────────

def plot_from_cli() -> None:
    """Entry point: python -m training.train_grpo --plot logs/training_*.jsonl"""
    parser = argparse.ArgumentParser(description="Generate training dashboard from JSONL log")
    parser.add_argument("log_file",   help="Path to training JSONL log file")
    parser.add_argument("--out_dir",  default="./logs", help="Output directory for PNGs")
    args = parser.parse_args()
    plot_training_run(args.log_file, args.out_dir)


if __name__ == "__main__":
    main()
