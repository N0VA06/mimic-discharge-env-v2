"""
inference.py — MIMIC Discharge Planning (simplified)
Structured stdout: [START], [STEP], [END]
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ── Config ────────────────────────────────────────────────────────────────────
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
BENCHMARK    = "mimic-discharge-planning"
MAX_STEPS    = 3

_TASK_NAMES = {1: "discharge-disposition", 2: "care-plan", 3: "discharge-note"}

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

# ── Stdout loggers (competition spec) ─────────────────────────────────────────

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    err = error or "null"
    print(f"[STEP] step={step} action={action[:80]} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}", flush=True)

# ── Health check wait ─────────────────────────────────────────────────────────

def wait_for_server(url: str, timeout: int = 120) -> bool:
    """Poll /health until server is ready or timeout."""
    health_url = url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"[INFO] Server ready at {url}", flush=True)
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"[WARN] Server not ready after {timeout}s — proceeding anyway", flush=True)
    return False

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM = (
    "You are a clinical physician assistant producing structured discharge planning outputs. "
    "Output ONLY a single valid JSON object. No markdown fences, no text outside JSON. "
    "Close every string. Never invent drug names or diagnoses not in the patient record."
)

# ── Task schemas ──────────────────────────────────────────────────────────────

T1_SCHEMA = """Output exactly:
{
  "task_id": 1,
  "task1": {
    "disposition": "<ONE OF: expired|hospice|ama|snf|rehab|home_with_services|home|other>",
    "reasoning": "<max 15 words>"
  }
}

Choose disposition:
- snf/rehab if: ventilation used, dialysis used, LOS>7 days and age>70, or IV meds at discharge
- home_with_services if: needs visiting nurse or home PT
- home if: fully independent, no follow-up care needed
- Default facility case → snf (safer than rehab unless clear rapid recovery post-surgery)"""

T2_SCHEMA = """Output exactly:
{
  "task_id": 2,
  "task2": {
    "follow_up_specialties": ["<spec1>", "<spec2>", "<spec3>"],
    "medications_to_continue": ["<drug1>", "<drug2>"],
    "medications_to_discontinue": ["<drug1>"],
    "key_instructions": [
      "<specific instruction with number/threshold/timeframe>",
      "<specific instruction with number/threshold/timeframe>",
      "<specific instruction with number/threshold/timeframe>",
      "<specific instruction with number/threshold/timeframe>",
      "<specific instruction with number/threshold/timeframe>"
    ],
    "reasoning": "<max 15 words>"
  }
}

Rules:
- Include ALL active medications in medications_to_continue (omit only vasopressors/paralytics/propofol)
- Each instruction MUST have a specific value: "Weigh daily; call if gain >2 lbs in 1 day" not "Monitor weight"
- Include follow-up timeframe: "See cardiologist within 2 weeks" not "Follow up with cardiologist" """

T3_SCHEMA = """Output exactly:
{
  "task_id": 3,
  "task3": {
    "discharge_note": "<FULL NOTE minimum 300 words with sections: PRINCIPAL DIAGNOSIS, BRIEF HOSPITAL COURSE, KEY PROCEDURES PERFORMED, DISCHARGE CONDITION, DISCHARGE DISPOSITION, DISCHARGE MEDICATIONS, FOLLOW-UP INSTRUCTIONS>"
  }
}

DISCHARGE DISPOSITION phrase must be exactly one of:
- "The patient was discharged home."
- "The patient was discharged home with home health services."
- "The patient was transferred to a skilled nursing facility."
- "The patient was transferred to inpatient rehabilitation."
- "The patient was transitioned to hospice care."
- "The patient expired during this hospitalization."

Discharge medications: ONLY use drugs from the active medications list."""

SCHEMAS = {1: T1_SCHEMA, 2: T2_SCHEMA, 3: T3_SCHEMA}

# ── Observation → prompt ──────────────────────────────────────────────────────

def _format_obs(obs: Dict[str, Any], task_id: int) -> str:
    lines = [
        "=== PATIENT SUMMARY ===",
        f"Age: {obs.get('age','?')}  Gender: {obs.get('gender','?')}  LOS: {obs.get('hospital_los_days','?')} days",
        f"Admission: {obs.get('admission_type','?')} via {obs.get('admission_location','?')}",
        f"Insurance: {obs.get('insurance','?')}",
    ]

    icu_p = obs.get("icu_procedures", {}) or {}
    vent  = float(icu_p.get("ventilation_hours", 0) or 0)
    dial  = bool(icu_p.get("has_dialysis", False))
    if vent > 0:
        lines.append(f"ICU ventilation: {vent:.1f} hours  <- FACILITY INDICATOR")
    if dial:
        lines.append("Dialysis used: YES  <- FACILITY INDICATOR")

    traj = obs.get("care_trajectory", [])
    if traj:
        lines.append(f"Care path: {' -> '.join(traj)}")

    if obs.get("diagnoses"):
        lines.append("\n-- DIAGNOSES --")
        for dx in obs["diagnoses"][:10]:
            lines.append(f"  [{dx.get('seq_num',0):2d}] {dx.get('icd_code',''):10s} {dx.get('description','')}")

    meds = obs.get("pharmacy_active", [])
    if meds:
        lines.append("\n-- ACTIVE MEDICATIONS --")
        for m in meds[:12]:
            lines.append(f"  - {m}")

    stopped = obs.get("pharmacy_stopped", [])
    if stopped:
        lines.append("\n-- STOPPED MEDICATIONS --")
        for m in stopped[:8]:
            lines.append(f"  - {m}")

    if obs.get("medications"):
        lines.append("\n-- MEDICATION ORDERS --")
        for m in obs["medications"][:10]:
            drug  = m.get("drug", "")
            route = m.get("route", "")
            dose  = m.get("dose_val_rx", "")
            lines.append(f"  {drug} {dose} [{route}]".strip())

    if obs.get("lab_flags"):
        lines.append("\n-- ABNORMAL LABS --")
        for lab in obs["lab_flags"][:8]:
            lines.append(f"  {lab.get('label','?'):30s} {str(lab.get('value','?')):>10}  [{lab.get('flag','?')}]")

    if obs.get("procedures"):
        lines.append("\n-- PROCEDURES --")
        for p in obs["procedures"][:6]:
            lines.append(f"  {p}")

    if obs.get("drg_codes"):
        lines.append("\n-- DRG --")
        for d in obs["drg_codes"][:2]:
            lines.append(f"  {d}")

    lines += [f"\n=== TASK {task_id} ===", obs.get("task_description", "")]
    lines += ["", "=== SCHEMA ===", SCHEMAS[task_id]]
    return "\n".join(lines)

# ── Fallbacks ─────────────────────────────────────────────────────────────────

def _fallback(task_id: int, obs: Dict[str, Any]) -> Dict:
    drugs   = [str(m) for m in (obs.get("pharmacy_active") or [])[:6]]
    stopped = [str(m) for m in (obs.get("pharmacy_stopped") or [])[:3]]
    los     = float(obs.get("hospital_los_days", 0) or 0)
    icu_p   = obs.get("icu_procedures", {}) or {}
    vent    = float(icu_p.get("ventilation_hours", 0) or 0)
    dial    = bool(icu_p.get("has_dialysis", False))
    age     = int(obs.get("age", 65) or 65)
    facility = vent > 0 or dial or (los > 7 and age > 70)

    if task_id == 1:
        return {"task_id": 1, "task1": {
            "disposition": "snf" if facility else "home_with_services",
            "reasoning": "Fallback: ICU signals and LOS reviewed.",
        }}
    if task_id == 2:
        return {"task_id": 2, "task2": {
            "follow_up_specialties": ["Primary Care", "Cardiology", "Nephrology"],
            "medications_to_continue": drugs,
            "medications_to_discontinue": stopped,
            "key_instructions": [
                "Follow up with primary care within 1 week of discharge.",
                "Take all medications as prescribed; do not stop without calling your doctor.",
                "Maintain low-sodium diet under 2 grams per day.",
                "Weigh yourself every morning; call doctor if gain exceeds 2 lbs in 1 day.",
                "Return to ED immediately for chest pain, shortness of breath, or fever over 38.5 C.",
            ],
            "reasoning": "Fallback care plan based on episode data.",
        }}
    drug_list = "\n".join(f"- {d}" for d in drugs) or "- No medications recorded"
    return {"task_id": 3, "task3": {"discharge_note": (
        f"PRINCIPAL DIAGNOSIS: Acute medical illness requiring inpatient management.\n\n"
        f"BRIEF HOSPITAL COURSE: The patient was admitted and hospitalised for {los:.1f} days. "
        "Appropriate clinical management was provided. The patient's condition stabilised and improved. "
        "All active medical problems were addressed and specialist consultations were obtained as indicated.\n\n"
        "KEY PROCEDURES PERFORMED: Routine monitoring and laboratory evaluation.\n\n"
        "DISCHARGE CONDITION: Stable, improved from admission baseline.\n\n"
        "DISCHARGE DISPOSITION: The patient was discharged home with home health services.\n\n"
        f"DISCHARGE MEDICATIONS:\n{drug_list}\n\n"
        "FOLLOW-UP INSTRUCTIONS: Follow up with primary care within 1 week. Continue all medications as "
        "prescribed. Maintain a low-sodium diet under 2 grams per day. Weigh every morning; call doctor "
        "if gain exceeds 2 lbs in one day. Return to ED for chest pain, shortness of breath, or fever over 38.5 C."
    )}}

# ── JSON extraction ────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJ_RE   = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def _extract_json(raw: str) -> Optional[Dict]:
    raw       = _THINK_RE.sub("", raw).strip()
    fence     = _FENCE_RE.search(raw)
    candidate = fence.group(1) if fence else raw
    obj       = _OBJ_RE.search(candidate)
    if obj:
        candidate = obj.group(0)
    for text in [candidate, raw]:
        try:
            return json.loads(text)
        except Exception:
            pass
    # Try to close truncated JSON
    fixed = candidate.rstrip()
    stack, in_str = [], False
    i = 0
    while i < len(fixed):
        ch = fixed[i]
        if ch == '\\' and in_str:
            i += 2; continue
        if ch == '"':
            in_str = not in_str
        if not in_str:
            if ch in '{[':
                stack.append('}' if ch == '{' else ']')
            elif ch in '}]' and stack:
                stack.pop()
        i += 1
    if in_str:
        fixed += '"'
    fixed += ''.join(reversed(stack))
    try:
        return json.loads(fixed)
    except Exception:
        return None

# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int) -> Optional[str]:
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", flush=True)
        return None

_MAX_TOKENS = {1: 512, 2: 1024, 3: 3072}

def _get_action_dict(obs: Dict, task_id: int) -> Dict:
    prompt = _format_obs(obs, task_id)
    raw    = _call_llm(prompt, _MAX_TOKENS[task_id])
    if raw:
        parsed = _extract_json(raw)
        if parsed:
            parsed["task_id"] = task_id
            return parsed
    return _fallback(task_id, obs)

# ── Build Pydantic Action ─────────────────────────────────────────────────────

def _build_action(d: Dict, task_id: int):
    from environment import Action
    from environment.models import Task1Action, Task2Action, Task3Action

    if task_id == 1:
        t = d.get("task1", {})
        return Action(task_id=1, task1=Task1Action(
            disposition=t.get("disposition", "home"),
            reasoning=t.get("reasoning", ""),
        ))
    if task_id == 2:
        t = d.get("task2", {})
        return Action(task_id=2, task2=Task2Action(
            follow_up_specialties=t.get("follow_up_specialties", []),
            medications_to_continue=t.get("medications_to_continue", []),
            medications_to_discontinue=t.get("medications_to_discontinue", []),
            key_instructions=t.get("key_instructions", []),
            reasoning=t.get("reasoning", ""),
        ))
    t = d.get("task3", {})
    return Action(task_id=3, task3=Task3Action(
        discharge_note=t.get("discharge_note", ""),
    ))

def _action_summary(d: Dict, task_id: int) -> str:
    if task_id == 1:
        return f"disposition={d.get('task1', {}).get('disposition', '?')}"
    if task_id == 2:
        specs = d.get("task2", {}).get("follow_up_specialties", [])[:2]
        return f"specialties=[{','.join(specs)}]"
    note = d.get("task3", {}).get("discharge_note", "")
    return f"note({len(note.split())}words)"

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: int) -> float:
    from environment import MIMICDischargeEnv

    task_name = _TASK_NAMES[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_name)

    try:
        env = MIMICDischargeEnv()
        obs_model = env.reset(task_id=task_id)
        obs = obs_model.model_dump()
        max_steps = obs.get("max_steps", MAX_STEPS)

        for step in range(1, max_steps + 1):
            try:
                action_dict = _get_action_dict(obs, task_id)
                action      = _build_action(action_dict, task_id)
                result      = env.step(action)

                reward = float(result.reward or 0.0)
                done   = bool(result.done)

                rewards.append(reward)
                steps_taken = step

                log_step(step, _action_summary(action_dict, task_id), reward, done)

                if done:
                    break
                if result.observation is not None:
                    obs = result.observation.model_dump()

            except Exception as e:
                rewards.append(0.0)
                steps_taken = step
                log_step(step, "error", 0.0, True, str(e)[:80])
                break

        score   = sum(rewards)
        success = score > 0.0

    except Exception as e:
        print(f"[ERROR] task={task_id} error={e}", flush=True)
        rewards     = [0.0]
        steps_taken = 1
        score       = 0.0

    log_end(success, steps_taken, score, rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    wait_for_server(ENV_URL, timeout=120)

    scores: Dict[int, float] = {}
    for task_id in [1, 2, 3]:
        try:
            scores[task_id] = run_episode(task_id)
        except Exception as e:
            print(f"[ERROR] task={task_id} unhandled={e}", flush=True)
            scores[task_id] = 0.0

    print("\n=== FINAL SCORES ===", flush=True)
    for t, s in scores.items():
        print(f"  Task {t} ({_TASK_NAMES[t]}): {s:.4f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  Overall: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()