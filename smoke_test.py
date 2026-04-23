#!/usr/bin/env python3
"""
smoke_test.py — pre-flight check for GRPO training.

Run this BEFORE train_grpo.py to catch env / model / config issues early.
All checks are fast (no GPU training, <2 min total with model load).

Usage:
    python smoke_test.py                              # env only, no model
    python smoke_test.py --model Qwen/Qwen2.5-3B-Instruct   # + model checks
    python smoke_test.py --env_url http://localhost:7860

Exit code 0 = all checks passed. Non-zero = at least one check failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import requests

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

_passed: List[str] = []
_failed: List[str] = []
_warned: List[str] = []


def ok(name: str, detail: str = "") -> None:
    msg = f"{GREEN}[PASS]{RESET} {name}" + (f"  {detail}" if detail else "")
    print(msg)
    _passed.append(name)


def fail(name: str, detail: str = "") -> None:
    msg = f"{RED}[FAIL]{RESET} {name}" + (f"\n       {detail}" if detail else "")
    print(msg)
    _failed.append(name)


def warn(name: str, detail: str = "") -> None:
    msg = f"{YELLOW}[WARN]{RESET} {name}" + (f"\n       {detail}" if detail else "")
    print(msg)
    _warned.append(name)


# ── Env HTTP helpers ──────────────────────────────────────────────────────────

def _post(env_url: str, path: str, body: Dict, timeout: int = 30) -> Optional[Dict]:
    try:
        r = requests.post(f"{env_url.rstrip('/')}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e)}


def _get(env_url: str, path: str, timeout: int = 10) -> Optional[Dict]:
    try:
        r = requests.get(f"{env_url.rstrip('/')}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": str(e)}


# ── Check groups ──────────────────────────────────────────────────────────────

def check_env_reachable(env_url: str) -> bool:
    print("\n--- Env connectivity ---")
    r = _get(env_url, "/health")
    if r and "__error__" not in r:
        ok("Env /health", str(r))
        return True
    # Some servers don't have /health; try /reset
    r2 = _post(env_url, "/reset", {"task_id": 1})
    if r2 and "__error__" not in r2:
        ok("Env reachable (via /reset)")
        return True
    fail("Env unreachable", str(r.get("__error__", r)))
    return False


def check_complexity_pool(env_url: str) -> Dict[str, int]:
    """
    Query /episodes/by_complexity and report the patient distribution.
    Warns when any tier used by the curriculum has fewer than MIN_TIER_POOL
    patients — that causes the seed dataset to cycle the same few hadm_ids.
    """
    print("\n--- Patient complexity pool ---")
    MIN_TIER_POOL = 25  # must match training/train_grpo.py

    r = _get(env_url, "/episodes/by_complexity")
    if not r or "__error__" in r:
        warn("Cannot reach /episodes/by_complexity", str(r))
        return {}

    totals: Dict[str, int] = r.get("totals", {})
    total  = sum(totals.values())

    easy   = totals.get("easy",   0)
    medium = totals.get("medium", 0)
    hard   = totals.get("hard",   0)

    pct = lambda n: f"{n/total*100:.0f}%" if total else "?"
    print(f"  easy  : {easy:4d}  ({pct(easy)})")
    print(f"  medium: {medium:4d}  ({pct(medium)})")
    print(f"  hard  : {hard:4d}  ({pct(hard)})")
    print(f"  total : {total:4d}")

    # Check if curriculum tiers are usable
    tier_to_phase = {"easy": "Task 1 (easy_only)", "medium": "Task 2 (medium_only)"}
    all_ok = True
    for tier, phase in tier_to_phase.items():
        n = totals.get(tier, 0)
        if n == 0:
            fail(f"{tier} pool", f"0 patients — {phase} curriculum impossible")
            all_ok = False
        elif n < MIN_TIER_POOL:
            warn(
                f"{tier} pool too small ({n} < {MIN_TIER_POOL})",
                f"{phase} will fall back to 'random' automatically in train_grpo.py",
            )
        else:
            ok(f"{tier} pool", f"{n} patients ≥ MIN_TIER_POOL={MIN_TIER_POOL} ✓")

    if total > 0 and all_ok:
        # Check rough balance: no tier should dominate >80% of total
        for tier, n in totals.items():
            pct_val = n / total
            if pct_val > 0.80:
                warn(
                    f"Imbalanced pool: {tier}={n} is {pct_val:.0%} of total",
                    "Training will over-represent this complexity tier.",
                )

    return totals


def check_env_reset(env_url: str) -> Dict[int, Optional[Dict]]:
    print("\n--- /reset for each task ---")
    obs_by_task: Dict[int, Optional[Dict]] = {}
    for tid in (1, 2, 3):
        r = _post(env_url, "/reset", {"task_id": tid, "noise_level": "clean", "curriculum_mode": "random"})
        if r and "__error__" not in r:
            hadm = r.get("hadm_id", "?")
            ok(f"Task {tid} /reset", f"hadm_id={hadm}")
            obs_by_task[tid] = r
        else:
            fail(f"Task {tid} /reset", str(r))
            obs_by_task[tid] = None
    return obs_by_task


def check_patient_pool(env_url: str, n: int = 20) -> None:
    print(f"\n--- Patient pool diversity (sampling {n} resets, task=1) ---")
    seen: set = set()
    for _ in range(n):
        r = _post(env_url, "/reset", {"task_id": 1, "curriculum_mode": "random"})
        if r and "__error__" not in r and "hadm_id" in r:
            seen.add(r["hadm_id"])
    if len(seen) >= 5:
        ok("Patient diversity", f"{len(seen)} unique hadm_ids in {n} resets")
    elif len(seen) >= 2:
        warn("Patient diversity", f"Only {len(seen)} unique hadm_ids in {n} resets — pool may be small")
    else:
        fail("Patient diversity", f"Only {len(seen)} unique hadm_ids in {n} resets — curriculum filter too strict")


def check_env_step_valid(env_url: str, obs_by_task: Dict[int, Optional[Dict]]) -> None:
    print("\n--- /step with valid actions ---")

    # Task 1: home disposition
    if obs_by_task.get(1):
        _post(env_url, "/reset", {"task_id": 1})
        r = _post(env_url, "/step", {
            "task_id": 1,
            "task1": {"disposition": "home", "reasoning": "Stable patient, short LOS, oral meds."},
        })
        if r and "__error__" not in r and "reward" in r:
            ok("Task 1 valid /step", f"reward={r['reward']:.4f}")
        else:
            fail("Task 1 valid /step", str(r))

    # Task 2: care plan
    if obs_by_task.get(2):
        _post(env_url, "/reset", {"task_id": 2})
        _post(env_url, "/step", {"task_id": 2, "information_request": ["labs", "medications"]})
        r = _post(env_url, "/step", {
            "task_id": 2,
            "task2": {
                "follow_up_specialties": ["Primary Care", "Cardiology"],
                "medications_to_continue": [],
                "medications_to_discontinue": [],
                "key_instructions": [
                    "Follow up with primary care within 1 week.",
                    "Take all medications as prescribed.",
                    "Return to ED for chest pain or shortness of breath.",
                    "Weigh yourself daily; call if weight up >2 lbs.",
                    "Maintain low-sodium diet under 2000 mg/day.",
                ],
                "reasoning": "Stable discharge.",
            },
        })
        if r and "__error__" not in r and "reward" in r:
            ok("Task 2 valid /step", f"reward={r['reward']:.4f}")
        else:
            fail("Task 2 valid /step", str(r))

    # Task 3: discharge note
    if obs_by_task.get(3):
        _post(env_url, "/reset", {"task_id": 3})
        note = (
            "PRINCIPAL DIAGNOSIS: Acute systolic heart failure\n\n"
            "BRIEF HOSPITAL COURSE: The patient is a 72-year-old male admitted via the emergency "
            "room with acute systolic heart failure. The hospital stay lasted 4.2 days. "
            "The patient was managed with diuresis and optimisation of guideline-directed medical therapy. "
            "All active medical problems were addressed. The patient's condition stabilised and was "
            "deemed safe for discharge.\n\n"
            "KEY PROCEDURES PERFORMED: Routine monitoring and laboratory evaluation\n\n"
            "DISCHARGE CONDITION: Stable, improved from admission baseline.\n\n"
            "DISCHARGE DISPOSITION: The patient was discharged home with home health services.\n\n"
            "DISCHARGE MEDICATIONS:\n- No medications prescribed at discharge\n\n"
            "FOLLOW-UP INSTRUCTIONS: Follow up with primary care within 1 week. "
            "Return to ED for fever, chest pain, or worsening shortness of breath."
        )
        r = _post(env_url, "/step", {"task_id": 3, "task3": {"discharge_note": note}})
        if r and "__error__" not in r and "reward" in r:
            ok("Task 3 valid /step", f"reward={r['reward']:.4f}")
        else:
            fail("Task 3 valid /step", str(r))


def check_env_step_invalid(env_url: str) -> None:
    print("\n--- /step with invalid/empty action (should not crash, reward=0) ---")
    _post(env_url, "/reset", {"task_id": 1})
    r = _post(env_url, "/step", {"task_id": 1})
    if r and "__error__" not in r:
        rw = float(r.get("reward", -1))
        if rw <= 0.0:
            ok("Empty action graceful", f"reward={rw}")
        else:
            warn("Empty action gave non-zero reward", f"reward={rw} — fallback scoring may be too generous")
    else:
        warn("Empty action /step errored", str(r) + " — env may reject missing fields (422 expected)")


def check_reward_function_logic() -> None:
    print("\n--- Reward blending logic (unit test) ---")
    sys.path.insert(0, ".")
    try:
        from training.train_grpo import _extract_json, _format_score, _reward_weights

        # Parse valid JSON
        j = _extract_json('{"task_id": 1, "task1": {"disposition": "home", "reasoning": "stable"}}')
        assert j is not None, "valid JSON not parsed"
        ok("_extract_json valid", f"keys={list(j.keys())}")

        # Parse JSON with markdown fences
        j2 = _extract_json('```json\n{"task_id":1,"task1":{"disposition":"snf"}}\n```')
        assert j2 is not None, "fenced JSON not parsed"
        ok("_extract_json fenced markdown")

        # Parse failure returns None
        j3 = _extract_json("this is not json at all")
        assert j3 is None, "non-JSON should return None"
        ok("_extract_json parse failure → None")

        # Format scores
        assert _format_score(None, 1) == 0.0
        assert _format_score({"task1": {"disposition": "home"}}, 1) == 1.0
        assert _format_score({"task1": {}}, 1) == 0.5
        assert _format_score({"other_key": {}}, 1) == 0.2
        ok("_format_score tiers correct")

        # Reward weights sum to 1.0
        for tid in (1, 2, 3):
            ew, fw = _reward_weights(tid)
            assert abs(ew + fw - 1.0) < 1e-9, f"weights don't sum to 1 for task {tid}"
        ok("_reward_weights sum to 1.0")

    except ImportError as e:
        warn("Reward logic checks skipped", f"import error: {e}")
    except AssertionError as e:
        fail("Reward logic", str(e))
    except Exception as e:
        fail("Reward logic unexpected error", traceback.format_exc(limit=3))


def check_format_observation() -> None:
    print("\n--- format_observation output ---")
    try:
        from training.rollout_collector import format_observation, _SYSTEM_PROMPT
        sample_obs = {
            "hadm_id": 99999,
            "task_id": 1,
            "age": 65,
            "gender": "M",
            "hospital_los_days": 3.5,
            "complexity": "medium",
            "admission_type": "EMERGENCY",
            "admission_location": "EMERGENCY ROOM",
            "diagnoses": [
                {"seq_num": 1, "icd_version": 10, "icd_code": "I10", "description": "Essential hypertension"},
                {"seq_num": 2, "icd_version": 10, "icd_code": "E11", "description": "Type 2 diabetes"},
            ],
            "pharmacy_active": ["Metoprolol 50mg", "Lisinopril 10mg"],
            "pharmacy_stopped": [],
            "lab_flags": [],
            "vitals": [],
            "discharge_orders": {},
            "icu_stays": [],
            "icu_procedures": {},
        }
        content = format_observation(sample_obs, task_id=1)
        assert "=== PATIENT SUMMARY ===" in content, "missing patient summary header"
        assert _SYSTEM_PROMPT not in content, "system prompt should NOT be in format_observation output"
        assert "=== TASK 1 ===" in content, "missing task section"
        assert "=== REQUIRED OUTPUT FORMAT ===" in content, "missing format section"
        assert "disposition" in content.lower(), "missing disposition field hint"
        ok("format_observation structure", f"{len(content)} chars, system prompt correctly excluded")
    except ImportError as e:
        warn("format_observation check skipped", f"import error: {e}")
    except AssertionError as e:
        fail("format_observation", str(e))
    except Exception as e:
        fail("format_observation unexpected error", traceback.format_exc(limit=3))


def check_seed_dataset(env_url: str, n: int = 3) -> None:
    print(f"\n--- build_seed_dataset (n={n}, task=1) ---")
    try:
        from training.train_grpo import build_seed_dataset
        ds = build_seed_dataset(env_url, task_id=1, n=n,
                                noise_level="clean", curriculum_mode="random")
        assert len(ds) == n, f"expected {n} rows, got {len(ds)}"
        row = ds[0]
        prompt = row["prompt"]
        assert isinstance(prompt, list), f"prompt should be list of dicts, got {type(prompt)}"
        assert len(prompt) == 2, f"expected [system, user] messages, got {len(prompt)} messages"
        assert prompt[0]["role"] == "system", "first message must be system"
        assert prompt[1]["role"] == "user",   "second message must be user"
        assert "=== PATIENT SUMMARY ===" in prompt[1]["content"], \
            "user message missing patient summary"
        ok("build_seed_dataset", f"{len(ds)} rows, prompt format=message-list ✓")
    except RuntimeError as e:
        fail("build_seed_dataset", str(e))
    except ImportError as e:
        warn("build_seed_dataset skipped", f"import error: {e}")
    except AssertionError as e:
        fail("build_seed_dataset", str(e))
    except Exception as e:
        fail("build_seed_dataset unexpected error", traceback.format_exc(limit=3))


def check_tokenizer_chat_template(model_name: str) -> None:
    print(f"\n--- Tokenizer chat template ({model_name}) ---")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        messages = [
            {"role": "system", "content": "You are a physician."},
            {"role": "user",   "content": "=== PATIENT SUMMARY ===\nAge: 65"},
        ]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert "system" in formatted.lower() or "<|im_start|>" in formatted, \
            "chat template not applied (system token missing)"
        tokens = tok.encode(formatted)
        ok("Chat template applied", f"{len(tokens)} tokens for sample prompt")

        # Check a realistic full prompt fits in MAX_PROMPT_LENGTH
        from training.rollout_collector import _SYSTEM_PROMPT
        long_user = "=== PATIENT SUMMARY ===\n" + "X" * 3000
        msgs2 = [{"role": "system", "content": _SYSTEM_PROMPT},
                 {"role": "user",   "content": long_user}]
        toks2 = tok.apply_chat_template(msgs2, tokenize=True, add_generation_prompt=True)
        if len(toks2) > 1536:
            warn("Long prompt exceeds MAX_PROMPT_LENGTH=1536",
                 f"got {len(toks2)} tokens — TRL will left-truncate. OK if intentional.")
        else:
            ok("Prompt length OK", f"{len(toks2)} tokens ≤ 1536")

    except ImportError as e:
        warn("Tokenizer check skipped", f"import error: {e}")
    except AssertionError as e:
        fail("Tokenizer chat template", str(e))
    except Exception as e:
        fail("Tokenizer unexpected error", traceback.format_exc(limit=3))


def check_model_forward(model_name: str) -> None:
    print(f"\n--- Model forward pass ({model_name}) ---")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            warn("No CUDA", "Model check skipped — no GPU available")
            return
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        messages = [
            {"role": "system", "content": "Output ONLY valid JSON."},
            {"role": "user",   "content": '=== PATIENT SUMMARY ===\nAge: 72  Gender: M\n\n=== TASK 1 ===\nPredict discharge.\n\n=== REQUIRED OUTPUT FORMAT ===\n{"task_id":1,"task1":{"disposition":"home","reasoning":"short"}}'},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        elapsed = time.time() - t0
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok("Model forward pass", f"{elapsed:.1f}s  response[:80]={repr(response[:80])}")

        # Check if response contains JSON
        try:
            import re
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                ok("Model produces parseable JSON", str(list(parsed.keys())))
            else:
                warn("Model response has no JSON", repr(response[:120]))
        except Exception:
            warn("Model response JSON parse failed", repr(response[:120]))

        del model
        torch.cuda.empty_cache()
    except ImportError as e:
        warn("Model check skipped", f"import error: {e}")
    except Exception as e:
        fail("Model forward pass", traceback.format_exc(limit=4))


def check_task1_grader() -> None:
    print("\n--- Task 1 disposition grader ---")
    try:
        from environment.tasks.task1_disposition import DispositionGrader, normalize_mimic_location

        cases = [
            ("HOME",                    "home",             True),
            ("HOME HEALTH CARE",        "home_with_services", True),
            ("SKILLED NURSING FACILITY","snf",              True),
            ("HOSPICE-HOME",            "hospice",          True),
            ("REHABILITATION",          "rehab",            True),
            ("EXPIRED",                 "expired",          True),
            ("AGAINST MEDICAL ADVICE",  "ama",              True),
            ("SOME UNKNOWN PLACE",      "other",            True),
        ]
        for raw_loc, expected, should_pass in cases:
            got = normalize_mimic_location(raw_loc)
            if got == expected:
                ok(f"normalize_mimic_location({raw_loc!r})", f"→ {got}")
            else:
                fail(f"normalize_mimic_location({raw_loc!r})", f"expected {expected!r}, got {got!r}")

        # Grader scoring
        grader = DispositionGrader()

        class FakeTask1:
            class task1:
                disposition = "home"
                reasoning   = "Stable patient with short LOS and oral medications only."
        score, partial, done, info = grader.grade(FakeTask1(), {"discharge_location": "HOME"})
        assert score >= 1.0, f"exact match should be 1.0+, got {score}"
        ok("DispositionGrader exact match", f"score={score}")

        class FakeTask1Wrong:
            class task1:
                disposition = "snf"
                reasoning   = "Needs skilled nursing."
        score2, _, _, _ = grader.grade(FakeTask1Wrong(), {"discharge_location": "HOME"})
        assert score2 < 0.5, f"wrong group should be <0.5, got {score2}"
        ok("DispositionGrader wrong group", f"score={score2}")

    except ImportError as e:
        warn("Task1 grader check skipped", f"import error: {e}")
    except AssertionError as e:
        fail("Task1 grader", str(e))
    except Exception as e:
        fail("Task1 grader unexpected error", traceback.format_exc(limit=3))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight smoke test for GRPO training")
    parser.add_argument("--env_url",  default="https://iinovaii-mimic-discharge-env-v2.hf.space",
                        help="Environment server URL")
    parser.add_argument("--model",    default=None,
                        help="HuggingFace model name to test (optional, slow)")
    parser.add_argument("--skip_env", action="store_true",
                        help="Skip env connectivity checks (unit-test only mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MIMIC Discharge Planning — Smoke Test")
    print(f"  env_url : {args.env_url}")
    print(f"  model   : {args.model or '(skipped)'}")
    print("=" * 60)

    # ── Unit tests (no env needed) ────────────────────────────────────────────
    check_reward_function_logic()
    check_format_observation()
    check_task1_grader()

    # ── Env tests ─────────────────────────────────────────────────────────────
    if not args.skip_env:
        reachable = check_env_reachable(args.env_url)
        if reachable:
            check_complexity_pool(args.env_url)
            obs_by_task = check_env_reset(args.env_url)
            check_patient_pool(args.env_url)
            check_env_step_valid(args.env_url, obs_by_task)
            check_env_step_invalid(args.env_url)
            check_seed_dataset(args.env_url)
        else:
            fail("Env checks skipped", "Cannot reach env server")

    # ── Model tests (optional, slow) ──────────────────────────────────────────
    if args.model:
        check_tokenizer_chat_template(args.model)
        check_model_forward(args.model)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {GREEN}PASSED{RESET}: {len(_passed)}")
    print(f"  {YELLOW}WARNED{RESET}: {len(_warned)}")
    print(f"  {RED}FAILED{RESET}: {len(_failed)}")
    print("=" * 60)

    if _failed:
        print(f"\n{RED}Failed checks:{RESET}")
        for f in _failed:
            print(f"  • {f}")
        return 1

    if _warned:
        print(f"\n{YELLOW}Warnings:{RESET}")
        for w in _warned:
            print(f"  • {w}")

    print(f"\n{GREEN}All required checks passed. Safe to start training.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
