"""
Rollout collector for MIMIC Discharge Planning environment.

Connects to the env server, runs N episodes with a local LLM, and saves
the results as a HuggingFace Dataset.

Error handling:
  - 422 Unprocessable Content: action fails Pydantic validation → normalize then retry
  - JSON parse failures: LLM output not parseable → noop action (reward 0)
  - Network errors: exponential backoff retry (max 3 attempts)
  - OOM during generation: skip episode and continue

Usage:
    python -m training.rollout_collector \\
        --env_url http://localhost:7860 \\
        --model_name Qwen/Qwen2.5-3B-Instruct \\
        --n_episodes 64 \\
        --task_id 1 \\
        --output_path ./rollouts

Dataset columns:
    prompt       str    formatted observation text
    response     str    LLM-generated action JSON (raw text)
    reward       float  environment reward for this step
    partial      str    JSON-encoded per-component scores
    hadm_id      int
    task_id      int
    step_num     int
    parse_ok     bool   whether action JSON was valid
    error_type   str    "" | "parse_fail" | "schema_422" | "oom" | "network"
    episode_idx  int
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ENV_TIMEOUT = 30
MAX_RETRIES = 3


# ─── Noop / fallback actions ──────────────────────────────────────────────────

_NOOP_ACTIONS: Dict[int, Dict] = {
    1: {"task_id": 1, "task1": {"disposition": "other", "reasoning": ""}},
    2: {"task_id": 2, "information_request": ["labs"]},
    3: {"task_id": 3, "task3": {"discharge_note": "Discharge summary not generated."}},
    4: {"task_id": 4, "task4": {"triage_level": "floor"}},
}


def _noop_action(task_id: int) -> Dict:
    return dict(_NOOP_ACTIONS.get(task_id, {"task_id": task_id}))


# ─── JSON extraction (dicts only) ────────────────────────────────────────────

def _try_parse_dict(s: str) -> Optional[Dict]:
    """Parse JSON and return only if result is a dict."""
    try:
        result = json.loads(s.strip())
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _extract_json(text: str) -> Optional[Dict]:
    """
    Extract the first JSON object from LLM output.
    Returns None if no valid dict-shaped JSON found.
    Non-dict JSON (arrays, strings) is treated as parse failure.
    """
    text = text.strip()

    # 1. Direct parse
    r = _try_parse_dict(text)
    if r is not None:
        return r

    # 2. ```json ... ``` or ``` ... ``` blocks
    for pattern in [r"```json\s*(.*?)```", r"```\s*(.*?)```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            r = _try_parse_dict(m.group(1).strip())
            if r is not None:
                return r

    # 3. Find outermost { ... } span
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        r = _try_parse_dict(text[start : end + 1])
        if r is not None:
            return r

    return None


# ─── Action normalisation ────────────────────────────────────────────────────

def _normalize_action(action_dict: Dict, task_id: int) -> Dict:
    """
    Fix common LLM formatting mistakes before posting to the API.

    Handles:
      - task_id at wrong type or missing
      - task1 fields at root level instead of nested
      - task1/task3 as string instead of object
      - list fields sent as a single string
      - ready_for_stepdown as string "true"/"false"
    """
    if not isinstance(action_dict, dict):
        return _noop_action(task_id)

    out = {k: v for k, v in action_dict.items()}
    out["task_id"] = task_id  # always force correct task_id type

    # ── Task 1 fixes ──────────────────────────────────────────────────────────
    if task_id == 1:
        # Flatten root-level fields into task1 sub-object
        if "task1" not in out and "disposition" in out:
            out["task1"] = {
                "disposition": str(out.pop("disposition", "other")).lower().strip(),
                "reasoning":   str(out.pop("reasoning", "")),
            }
        # task1 is a bare string (LLM wrote disposition directly)
        if isinstance(out.get("task1"), str):
            out["task1"] = {"disposition": out["task1"].lower().strip()}
        # Normalise disposition value to lowercase
        if isinstance(out.get("task1"), dict) and "disposition" in out["task1"]:
            out["task1"]["disposition"] = str(out["task1"]["disposition"]).lower().strip()

    # ── Task 2 fixes ──────────────────────────────────────────────────────────
    if task_id == 2 and isinstance(out.get("task2"), dict):
        t2 = out["task2"]
        for f in ["follow_up_specialties", "medications_to_continue",
                  "medications_to_discontinue", "key_instructions"]:
            if f in t2:
                if isinstance(t2[f], str):
                    t2[f] = [t2[f]] if t2[f] else []
                elif not isinstance(t2[f], list):
                    t2[f] = []

    # ── Task 3 fixes ──────────────────────────────────────────────────────────
    if task_id == 3:
        # Flatten root-level discharge_note
        if "task3" not in out and "discharge_note" in out:
            out["task3"] = {"discharge_note": str(out.pop("discharge_note", ""))}
        # task3 is a bare string
        if isinstance(out.get("task3"), str):
            out["task3"] = {"discharge_note": out["task3"]}
        if isinstance(out.get("task3"), dict) and "discharge_note" in out["task3"]:
            out["task3"]["discharge_note"] = str(out["task3"]["discharge_note"])

    # ── Task 4 fixes ──────────────────────────────────────────────────────────
    if task_id == 4 and isinstance(out.get("task4"), dict):
        t4 = out["task4"]
        for f in ["priority_labs", "priority_consults", "interventions",
                  "high_risk_medications", "antibiotics", "barriers",
                  "medications_to_continue"]:
            if f in t4:
                if isinstance(t4[f], str):
                    t4[f] = [t4[f]] if t4[f] else []
                elif not isinstance(t4[f], list):
                    t4[f] = []
        # Boolean coercion
        if "ready_for_stepdown" in t4:
            v = t4["ready_for_stepdown"]
            if isinstance(v, str):
                t4["ready_for_stepdown"] = v.lower() in ("true", "yes", "1")
        # Numeric coercion
        if "los_remaining_days" in t4:
            try:
                t4["los_remaining_days"] = float(t4["los_remaining_days"])
            except (TypeError, ValueError):
                del t4["los_remaining_days"]

    return out


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

class _EnvClient:
    def __init__(self, env_url: str) -> None:
        self.env_url = env_url.rstrip("/")

    def post(self, path: str, body: Dict, retries: int = MAX_RETRIES) -> Dict:
        """POST with exponential backoff for network errors; no retry on 422."""
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(retries):
            try:
                r = requests.post(
                    f"{self.env_url}{path}", json=body, timeout=ENV_TIMEOUT
                )
                if r.status_code == 422:
                    # Schema validation failure — expose detail for debugging
                    raise requests.exceptions.HTTPError(
                        f"422 Unprocessable: {r.text[:300]}", response=r
                    )
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError:
                raise  # Don't retry 4xx; let caller handle
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt < retries - 1:
                    time.sleep(1.5 ** attempt)
        raise last_exc


# ─── Observation formatter ────────────────────────────────────────────────────

def format_observation(obs: Dict[str, Any]) -> str:
    """Convert raw observation dict to structured text prompt for the LLM."""
    lines: List[str] = []

    lines.append("=== PATIENT CLINICAL SUMMARY ===")
    lines.append(f"Task: {obs.get('task_description', '')}")
    lines.append(
        f"Hadm ID: {obs.get('hadm_id', '?')} | Subject: {obs.get('subject_id', '?')} | "
        f"Step: {obs.get('step_num', 0)}/{obs.get('max_steps', 1)}"
    )
    lines.append("")

    lines.append("--- DEMOGRAPHICS ---")
    lines.append(f"Age: {obs.get('age', '?')} | Gender: {obs.get('gender', '?')}")
    lines.append(f"Admission type: {obs.get('admission_type', '?')}")
    lines.append(f"LOS: {obs.get('hospital_los_days', 0):.1f} days | Complexity: {obs.get('complexity', '?')}")
    lines.append("")

    dx = obs.get("diagnoses") or []
    if dx:
        lines.append("--- DIAGNOSES ---")
        for d in dx[:5]:
            lines.append(f"  [{d.get('icd_code','?')}] {d.get('description','?')}")
        lines.append("")

    icu = obs.get("icu_stays") or []
    if icu:
        lines.append("--- ICU STAYS ---")
        for s in icu:
            lines.append(
                f"  {s.get('first_careunit','?')} → {s.get('last_careunit','?')}  "
                f"({s.get('los_days', 0):.1f} days)"
            )
        lines.append("")

    vitals = obs.get("vitals") or []
    if vitals:
        lines.append("--- VITALS ---")
        for v in vitals:
            crit = " [CRITICAL]" if v.get("critical_flag") else ""
            lines.append(
                f"  {v.get('name','?')}: adm={v.get('admission_value','?')} "
                f"dc={v.get('discharge_value','?')}  "
                f"[{v.get('min_value','?')}–{v.get('max_value','?')}]{crit}"
            )
        lines.append("")

    labs = obs.get("lab_flags") or []
    abnormal = [l for l in labs if str(l.get("flag", "")).lower() in ("abnormal", "critical", "high", "low")]
    if abnormal:
        lines.append("--- ABNORMAL LABS ---")
        for l in abnormal[:10]:
            lines.append(f"  {l.get('label','?')}: {l.get('value','?')} [{l.get('flag','?')}]")
        lines.append("")

    meds = obs.get("medications") or []
    if meds:
        lines.append("--- MEDICATIONS ---")
        for m in meds[:10]:
            lines.append(f"  {m.get('drug','?')} {m.get('route','')} {m.get('dose_val_rx','')}")
        lines.append("")

    emar = obs.get("emar_summary") or []
    active_emar = [e for e in emar if e.get("active_at_discharge")]
    if active_emar:
        lines.append("--- ACTIVE AT DISCHARGE (eMAR) ---")
        for e in active_emar[:8]:
            lines.append(f"  {e.get('medication','?')} (last: {e.get('last_given','?')})")
        lines.append("")

    micro = obs.get("microbiology") or []
    if micro:
        lines.append("--- MICROBIOLOGY ---")
        for m in micro[:5]:
            org = m.get("organism") or "No growth"
            res = ", ".join(m.get("resistant_to") or [])
            lines.append(f"  {m.get('specimen','?')}: {org}" + (f"  R: {res}" if res else ""))
        lines.append("")

    fb = obs.get("fluid_balance")
    if fb:
        lines.append("--- FLUID BALANCE ---")
        lines.append(
            f"  Input: {fb.get('total_input_ml',0):.0f} mL  "
            f"Output: {fb.get('total_output_ml',0):.0f} mL  "
            f"Net: {fb.get('net_balance_ml',0):+.0f} mL"
        )
        flags = [f for f, k in [("OVERLOADED", "fluid_overloaded"), ("OLIGURIA", "oliguria")] if fb.get(k)]
        if flags:
            lines.append(f"  Flags: {', '.join(flags)}")
        lines.append("")

    history = obs.get("episode_history") or []
    if history:
        lines.append("--- PRIOR STEPS ---")
        for h in history:
            lines.append(f"  Step {h.get('step_num','?')}: {h.get('action_summary','—')}")
        lines.append("")

    lines.append("--- ACTION REQUIRED ---")
    lines.append(obs.get("action_space_description", ""))
    lines.append("")
    lines.append("Respond ONLY with valid JSON matching the action schema above. No prose, no markdown.")

    return "\n".join(lines)


# ─── Rollout collector ────────────────────────────────────────────────────────

class RolloutCollector:

    def __init__(
        self,
        env_url: str,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
    ) -> None:
        self.client     = _EnvClient(env_url)
        self.model_name = model_name

        print(f"Loading {model_name}…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        print("Model ready.")

    def _llm(self, prompt: str, max_new_tokens: int = 512) -> Tuple[str, str]:
        """Run LLM inference. Returns (response_text, error_type)."""
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=3500
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            return response, ""
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "", "oom"
        except Exception as e:
            return "", f"llm_error:{type(e).__name__}"

    def run_episode(
        self,
        task_id: int,
        hadm_id: Optional[int] = None,
        noise_level: str = "clean",
    ) -> Dict:
        reset_body: Dict[str, Any] = {"task_id": task_id, "noise_level": noise_level}
        if hadm_id is not None:
            reset_body["hadm_id"] = hadm_id

        obs   = self.client.post("/reset", reset_body)
        done  = False
        steps: List[Dict] = []

        while not done:
            prompt             = format_observation(obs)
            response_txt, llm_err = self._llm(prompt)

            # Parse + normalize action
            error_type = llm_err
            if llm_err:
                action_dict = _noop_action(task_id)
                parse_ok    = False
            else:
                raw = _extract_json(response_txt)
                parse_ok = raw is not None
                if parse_ok:
                    action_dict = _normalize_action(raw, task_id)
                    error_type  = ""
                else:
                    action_dict = _noop_action(task_id)
                    error_type  = "parse_fail"

            # Submit to env; handle schema validation errors gracefully
            try:
                result = self.client.post("/step", action_dict)
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", True))
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 422:
                    # Schema validation failed even after normalization — use noop
                    print(
                        f"  [422] schema error after normalize. "
                        f"Sending noop. Detail: {e.args[0][:120]}"
                    )
                    error_type = "schema_422"
                    parse_ok   = False
                    try:
                        result = self.client.post("/step", _noop_action(task_id))
                        reward = float(result.get("reward", 0.0))
                        done   = bool(result.get("done", True))
                    except Exception:
                        reward = 0.0
                        done   = True
                        result = {}
                else:
                    raise

            steps.append({
                "prompt":     prompt,
                "response":   response_txt,
                "reward":     reward,
                "partial":    result.get("partial_signals", {}),
                "parse_ok":   parse_ok,
                "error_type": error_type,
                "step_num":   int(obs.get("step_num", 0)),
            })

            if not done:
                obs = result.get("observation") or {}

        return {
            "hadm_id":      int(obs.get("hadm_id", hadm_id or 0)),
            "task_id":      task_id,
            "total_reward": sum(s["reward"] for s in steps),
            "steps":        steps,
            "parse_rate":   sum(s["parse_ok"] for s in steps) / max(1, len(steps)),
        }

    def collect(
        self,
        task_id: int,
        n_episodes: int = 64,
        noise_level: str = "clean",
        output_path: Optional[str] = None,
    ) -> Dataset:
        rows: List[Dict] = []
        stats = {"ok": 0, "failed": 0, "parse_fail": 0, "schema_422": 0, "oom": 0}

        for ep_idx in range(n_episodes):
            t0 = time.time()
            try:
                episode = self.run_episode(task_id, noise_level=noise_level)
            except Exception as e:
                print(f"Episode {ep_idx} failed: {type(e).__name__}: {e}")
                stats["failed"] += 1
                continue

            for step in episode["steps"]:
                rows.append({
                    "prompt":      step["prompt"],
                    "response":    step["response"],
                    "reward":      step["reward"],
                    "partial":     json.dumps(step["partial"]),
                    "hadm_id":     int(episode.get("hadm_id") or 0),
                    "task_id":     task_id,
                    "step_num":    int(step["step_num"]),
                    "parse_ok":    bool(step["parse_ok"]),
                    "error_type":  step.get("error_type", ""),
                    "episode_idx": ep_idx,
                })
                et = step.get("error_type", "")
                if et in stats:
                    stats[et] += 1
                else:
                    stats["ok"] += 1

            stats["ok"] += 1
            elapsed = round(time.time() - t0, 1)
            print(
                f"Episode {ep_idx+1}/{n_episodes}  "
                f"reward={episode['total_reward']:.4f}  "
                f"steps={len(episode['steps'])}  "
                f"parse={episode['parse_rate']:.0%}  ({elapsed}s)"
            )

        print(f"\nCollection stats: {stats}")

        if not rows:
            print("Warning: no rows collected — dataset is empty.")
            rows = [{"prompt": "", "response": "", "reward": 0.0, "partial": "{}",
                     "hadm_id": 0, "task_id": task_id, "step_num": 0,
                     "parse_ok": False, "error_type": "empty", "episode_idx": 0}]

        dataset = Dataset.from_list(rows)
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(output_path)
            print(f"Saved {len(rows)} rows → {output_path}")
        return dataset


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_url",     default="http://localhost:7860")
    parser.add_argument("--model_name",  default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_episodes",  type=int, default=64)
    parser.add_argument("--task_id",     type=int, default=1)
    parser.add_argument("--noise_level", default="clean")
    parser.add_argument("--output_path", default="./rollouts")
    parser.add_argument("--device",      default="auto")
    args = parser.parse_args()

    collector = RolloutCollector(args.env_url, args.model_name, args.device)
    collector.collect(
        task_id=args.task_id,
        n_episodes=args.n_episodes,
        noise_level=args.noise_level,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
