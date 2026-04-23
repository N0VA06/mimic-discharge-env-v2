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

# ─── Task-specific schemas (mirror of inference.py) ───────────────────────────

_T1_SCHEMA = """Output EXACTLY this JSON and nothing else:
{"task_id": 1, "task1": {"disposition": "<one of the values below>", "reasoning": "<ONE sentence, max 20 words>"}}

Valid disposition values: expired | hospice | ama | snf | rehab | home_with_services | home | other

DECISION ORDER (first match wins):
1. HOSPICE  → ICD V667/Z51.5/palliative, secondary malignant neoplasm + DRG mortality=4, GCS≤5 + terminal dx
2. EXPIRED  → patient died during admission
3. SNF      → ventilation hours>0 OR dialysis, non-terminal, age>60
4. REHAB    → orthopedic fracture + surgical fixation, younger patient, good functional baseline
5. HOME WITH SERVICES → wound care, IV antibiotics, or home nurse needed at discharge
6. HOME     → stable, oral meds only, short LOS, discharge orders finalized

CRITICAL: Terminal cancer / palliative ICD → hospice ONLY. Never snf or home.
CRITICAL: reasoning MUST be one sentence under 20 words. Do NOT elaborate."""

_T2_SCHEMA = """Output EXACTLY:
{"task_id": 2, "task2": {"follow_up_specialties": ["<spec>"], "medications_to_continue": ["<drug>"], "medications_to_discontinue": ["<drug>"], "key_instructions": ["<instruction with numbers/thresholds>", ...x5], "reasoning": "<max 20 words>"}}

SPECIALTY: Derive ONLY from actual diagnoses.
ICD-9: 001-139→Infectious Disease, 140-239→Oncology, 240-279→Endocrinology,
320-389→Neurology, 390-459→Cardiology, 460-519→Pulmonology,
520-579→Gastroenterology, 580-629→Nephrology, 800-999→Trauma Surgery
ICD-10 first letter: A,B→Infectious Disease, C→Oncology, E→Endocrinology,
G→Neurology, I→Cardiology, J→Pulmonology, K→Gastroenterology, N→Nephrology
Microbiology organisms → always add Infectious Disease. Always include Primary Care.

MEDICATIONS:
  medications_to_continue   = EXACT names from ACTIVE MEDICATIONS list ONLY
  medications_to_discontinue = EXACT names from STOPPED MEDICATIONS list ONLY
  DO NOT invent drug names."""

_T3_SCHEMA = """Output EXACTLY:
{"task_id": 3, "task3": {"discharge_note": "<FULL NOTE ≥300 words>"}}

MANDATORY:
1. Name each top-5 diagnosis by its EXACT description keywords in prose
2. DISCHARGE MEDICATIONS: ONLY exact names from ACTIVE MEDICATIONS list
3. LOS: state "[N.N] days" matching the hospital_los_days shown
4. Use one verbatim disposition phrase:
   "The patient was discharged home." |
   "The patient was discharged home with home health services." |
   "The patient was transferred to a skilled nursing facility." |
   "The patient was transferred to inpatient rehabilitation." |
   "The patient was transitioned to hospice care." |
   "The patient expired during this hospitalization."
5. Sections in order: PRINCIPAL DIAGNOSIS | BRIEF HOSPITAL COURSE | KEY PROCEDURES PERFORMED | DISCHARGE CONDITION | DISCHARGE DISPOSITION | DISCHARGE MEDICATIONS | FOLLOW-UP INSTRUCTIONS"""

_TASK_SCHEMAS = {1: _T1_SCHEMA, 2: _T2_SCHEMA, 3: _T3_SCHEMA}

_SYSTEM_PROMPT = (
    "You are a clinical physician producing structured discharge planning outputs. "
    "Output ONLY a single valid JSON object. No markdown, no prose outside JSON. "
    "Never invent drug names not in the patient record."
)

# ─── ICD-9 specialty mapping ───────────────────────────────────────────────────

_ICD9_RANGES = [
    (1,   139, "Infectious Disease"),
    (140, 239, "Oncology"),
    (240, 279, "Endocrinology"),
    (290, 319, "Psychiatry"),
    (320, 389, "Neurology"),
    (390, 459, "Cardiology"),
    (460, 519, "Pulmonology"),
    (520, 579, "Gastroenterology"),
    (580, 629, "Nephrology"),
    (680, 709, "Dermatology"),
    (710, 739, "Rheumatology"),
    (800, 999, "Trauma Surgery"),
]

_ICD10_PREFIXES = {
    'A': 'Infectious Disease', 'B': 'Infectious Disease',
    'C': 'Oncology',           'E': 'Endocrinology',
    'F': 'Psychiatry',         'G': 'Neurology',
    'I': 'Cardiology',         'J': 'Pulmonology',
    'K': 'Gastroenterology',   'N': 'Nephrology',
    'M': 'Rheumatology',       'S': 'Trauma Surgery',
}


def _specialty_from_dx(diagnoses: List[Dict]) -> List[str]:
    specs: List[str] = ["Primary Care"]
    seen = {"Primary Care"}
    for dx in diagnoses[:8]:
        code    = str(dx.get("icd_code", "")).strip().upper()
        version = int(dx.get("icd_version", 10) or 10)
        spec    = None
        if version == 9:
            digits = re.sub(r"\D", "", code)[:3]
            if digits:
                num = int(digits)
                for lo, hi, s in _ICD9_RANGES:
                    if lo <= num <= hi:
                        spec = s
                        break
        else:
            spec = _ICD10_PREFIXES.get(code[0]) if code else None
        if spec and spec not in seen:
            seen.add(spec)
            specs.append(spec)
    return specs


def _is_end_of_life(obs: Dict) -> bool:
    diagnoses = obs.get("diagnoses", []) or []
    dx_descs  = [str(d.get("description", "")).lower() for d in diagnoses]
    dx_codes  = [str(d.get("icd_code", "")).upper() for d in diagnoses]
    drg_codes = [str(d).upper() for d in (obs.get("drg_codes") or [])]
    if any("V667" in c or "Z515" in c or "Z51.5" in c for c in dx_codes):
        return True
    if any("palliative" in d or "comfort care" in d for d in dx_descs):
        return True
    if any("secondary malignant neoplasm" in d for d in dx_descs) and \
       any("MALIGNANCY" in g or "MALIGNANT" in g for g in drg_codes):
        return True
    vitals = obs.get("vitals", []) or []
    for v in vitals:
        if v.get("name") == "GCS Total":
            gcs = v.get("discharge_value")
            if gcs is not None and float(gcs) <= 5:
                terminal = any(
                    kw in d for d in dx_descs
                    for kw in ("malignant", "malignancy", "metastasis", "neoplasm")
                )
                if terminal:
                    return True
    return False


def _fallback_action(task_id: int, obs: Dict) -> Dict:
    """Deterministic fallback when LLM fails — uses clinical heuristics."""
    drugs_active  = [str(m) for m in (obs.get("pharmacy_active")  or [])[:8]]
    drugs_stopped = [str(m) for m in (obs.get("pharmacy_stopped") or [])[:5]]
    los           = float(obs.get("hospital_los_days", 0) or 0)
    icu_p         = obs.get("icu_procedures", {}) or {}
    vent          = float(icu_p.get("ventilation_hours", 0) or 0)
    dial          = bool(icu_p.get("has_dialysis", False))
    age           = int(obs.get("age", 65) or 65)
    diagnoses     = obs.get("diagnoses", []) or []
    eol           = _is_end_of_life(obs)
    facility      = vent > 0 or dial or (los > 7 and age > 70)

    if task_id == 1:
        dx_descs = [str(d.get("description","")).lower() for d in diagnoses]
        is_ortho = any(kw in d for d in dx_descs for kw in ("fracture","femur","hip"))
        if eol:
            disp, reason = "hospice", "Terminal malignancy with palliative care indicators."
        elif is_ortho and not facility:
            disp, reason = "rehab", "Orthopedic fracture with surgical fixation requiring rehab."
        elif facility:
            disp, reason = "snf", "ICU or ventilation stay with advanced age requires skilled nursing."
        else:
            disp, reason = "home", "Hemodynamically stable, short LOS, oral medications only."
        return {"task_id": 1, "task1": {"disposition": disp, "reasoning": reason}}

    if task_id == 2:
        specs = _specialty_from_dx(diagnoses)
        if obs.get("microbiology") and "Infectious Disease" not in specs:
            specs.append("Infectious Disease")
        return {"task_id": 2, "task2": {
            "follow_up_specialties": specs[:4],
            "medications_to_continue":   drugs_active,
            "medications_to_discontinue": drugs_stopped[:3],
            "key_instructions": [
                "Follow up with primary care within 1 week of discharge.",
                "Complete the full antibiotic course as prescribed; do not stop early.",
                "Maintain a low-sodium diet under 2000 mg per day.",
                "Weigh yourself every morning; call doctor if weight increases >2 lbs/day or >5 lbs/week.",
                "Return to ED immediately for fever >38.5°C, chest pain, or severe worsening symptoms.",
            ],
            "reasoning": "Derived from ICD code analysis and active medication list.",
        }}

    if task_id == 3:
        dx_list  = "\n".join(f"{i+1}. {d.get('description','Unknown')}" for i, d in enumerate(diagnoses[:5]))
        drug_list = "\n".join(f"- {d}" for d in drugs_active) or "- No medications prescribed at discharge"
        disp_phrase = (
            "The patient was transitioned to hospice care." if eol else
            "The patient was transferred to a skilled nursing facility." if facility else
            "The patient was discharged home with home health services." if drugs_active else
            "The patient was discharged home."
        )
        los_text = f"{los:.1f}"
        procs    = obs.get("procedures") or []
        proc_text = "; ".join(procs[:3]) if procs else "Routine monitoring and laboratory evaluation"
        note = (
            f"PRINCIPAL DIAGNOSIS: {diagnoses[0].get('description','Acute medical illness') if diagnoses else 'Acute medical illness'}\n\n"
            f"BRIEF HOSPITAL COURSE: The patient is a {obs.get('age','?')}-year-old "
            f"{'male' if str(obs.get('gender','')).upper()=='M' else 'female'} "
            f"admitted via the emergency room with {diagnoses[0].get('description','acute illness') if diagnoses else 'acute illness'}. "
            f"The hospital stay lasted {los_text} days. "
            f"The patient was managed for the following diagnoses:\n{dx_list}\n"
            "All active medical problems were addressed and specialist consultations were obtained as indicated. "
            "The patient's condition stabilised with appropriate treatment and the patient was deemed safe for discharge.\n\n"
            f"KEY PROCEDURES PERFORMED: {proc_text}\n\n"
            "DISCHARGE CONDITION: Stable, improved from admission baseline. Vital signs within acceptable limits.\n\n"
            f"DISCHARGE DISPOSITION: {disp_phrase}\n\n"
            f"DISCHARGE MEDICATIONS:\n{drug_list}\n\n"
            "FOLLOW-UP INSTRUCTIONS: Follow up with primary care within 1 week. Continue all prescribed medications. "
            "Maintain a low-sodium diet under 2000 mg per day. "
            "Weigh yourself every morning; call your doctor if weight increases more than 2 lbs in one day or 5 lbs in one week. "
            "Return to the emergency department immediately for fever over 38.5°C, chest pain, worsening shortness of breath, "
            "severe pain, or any sudden change in your condition."
        )
        return {"task_id": 3, "task3": {"discharge_note": note}}

    return {"task_id": task_id}


# ─── Noop actions (used only on parse failure, not as default) ────────────────

def _noop_action(task_id: int) -> Dict:
    return {"task_id": task_id, "task1": {"disposition": "other", "reasoning": ""}} if task_id == 1 else \
           {"task_id": task_id, "task3": {"discharge_note": "Discharge summary not generated."}} if task_id == 3 else \
           {"task_id": task_id}


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

def format_observation(obs: Dict[str, Any], task_id: Optional[int] = None) -> str:
    """Convert raw observation dict to structured clinical prompt for the LLM.

    Returns the USER-facing content only — caller is responsible for wrapping
    with the system prompt in a proper chat template.  (_llm() and
    build_seed_dataset both do this via message-list formatting.)
    """
    tid = task_id or int(obs.get("task_id") or obs.get("task_id_hint") or 1)
    lines: List[str] = ["=== PATIENT SUMMARY ==="]

    los = obs.get("hospital_los_days", 0)
    lines.append(
        f"Age: {obs.get('age','?')}  Gender: {obs.get('gender','?')}  "
        f"LOS: {los:.1f} days  Complexity: {obs.get('complexity','?')}"
    )
    lines.append(f"Admission: {obs.get('admission_type','?')} via {obs.get('admission_location','?')}")

    icu_p = obs.get("icu_procedures", {}) or {}
    vent  = float(icu_p.get("ventilation_hours", 0) or 0)
    dial  = bool(icu_p.get("has_dialysis", False))
    if vent > 0:
        lines.append(f"ICU ventilation: {vent:.1f} hours")
    if dial:
        lines.append("Dialysis used: YES")

    icu_stays = obs.get("icu_stays") or []
    if icu_stays:
        s = icu_stays[0]
        lines.append(f"ICU: {float(s.get('los_days',0)):.2f} days in {s.get('first_careunit','?')}")

    traj = obs.get("care_trajectory") or []
    if traj:
        lines.append(f"Care path: {' → '.join(traj)}")

    # Vitals — GCS is critical for end-of-life decisions
    vitals = obs.get("vitals") or []
    if vitals:
        lines.append("\n-- VITALS (admission → discharge) --")
        for v in vitals:
            crit = "  ← CRITICAL" if v.get("critical_flag") else ""
            lines.append(
                f"  {v.get('name','?'):20s}: {v.get('admission_value','?')} → "
                f"{v.get('discharge_value','?')}{crit}"
            )

    # Discharge orders — strong signal for disposition
    dorders = obs.get("discharge_orders") or {}
    if dorders:
        lines.append("\n-- DISCHARGE ORDERS --")
        lines.append(f"  Finalized: {dorders.get('discharge_planning_finalized', False)}")
        otypes = dorders.get("documented_discharge_orders", [])
        if otypes:
            lines.append(f"  Types: {', '.join(otypes)}")

    # Diagnoses (shown with ICD codes for specialty derivation)
    diagnoses = obs.get("diagnoses") or []
    if diagnoses:
        lines.append("\n-- DIAGNOSES (derive specialties from ICD codes) --")
        for d in diagnoses[:10]:
            lines.append(
                f"  [{d.get('seq_num',0):2d}] ICD{d.get('icd_version','?')}-"
                f"{d.get('icd_code',''):10s}  {d.get('description','')}"
            )

    # Active medications — the ONLY source for medications_to_continue
    meds_active = obs.get("pharmacy_active") or []
    if meds_active:
        lines.append("\n-- ACTIVE MEDICATIONS AT DISCHARGE (use EXACT names for medications_to_continue) --")
        for m in meds_active:
            lines.append(f"  ✓ {m}")
    else:
        lines.append("\n-- ACTIVE MEDICATIONS AT DISCHARGE: None --")

    # Stopped medications — source for medications_to_discontinue
    meds_stopped = obs.get("pharmacy_stopped") or []
    if meds_stopped:
        lines.append("\n-- STOPPED/DISCONTINUED MEDICATIONS --")
        for m in meds_stopped[:10]:
            lines.append(f"  ✗ {m}")

    # Medication orders (reference context, NOT the discharge active list)
    meds_orders = obs.get("medications") or []
    if meds_orders:
        lines.append("\n-- MEDICATION ORDERS during admission (reference only) --")
        for m in meds_orders[:10]:
            lines.append(f"  {m.get('drug','?')} {m.get('dose_val_rx','')} [{m.get('route','')}]".strip())

    # Abnormal labs
    labs = obs.get("lab_flags") or []
    if labs:
        lines.append("\n-- ABNORMAL LABS --")
        for lab in labs[:10]:
            lines.append(f"  {lab.get('label','?'):30s} {str(lab.get('value','?')):>10}  [{lab.get('flag','?')}]")

    # Microbiology — triggers Infectious Disease specialty
    micro = obs.get("microbiology") or []
    if micro:
        lines.append("\n-- MICROBIOLOGY (→ add Infectious Disease to specialties) --")
        for m in micro[:5]:
            org = m.get("organism") or m.get("org_name") or "?"
            lines.append(f"  ⚠ {org}")

    # Procedures
    procs = obs.get("procedures") or []
    if procs:
        lines.append("\n-- PROCEDURES --")
        for p in procs[:6]:
            lines.append(f"  {p}")

    # DRG
    drg = obs.get("drg_codes") or []
    if drg:
        lines.append("\n-- DRG --")
        for d in drg[:2]:
            lines.append(f"  {d}")

    # eMAR active drugs (additional confirmation of discharge medications)
    emar = obs.get("emar_summary") or []
    active_emar = [e.get("medication","") for e in emar if e.get("active_at_discharge")]
    if active_emar and tid in (2, 3):
        lines.append("\n-- eMAR CONFIRMED ACTIVE AT DISCHARGE --")
        for med in active_emar[:10]:
            lines.append(f"  ✓ {med}")

    # Task schema
    lines.append(f"\n=== TASK {tid} ===")
    lines.append(obs.get("task_description", ""))
    lines.append("\n=== REQUIRED OUTPUT FORMAT ===")
    lines.append(_TASK_SCHEMAS.get(tid, obs.get("action_space_description", "")))
    lines.append("\nRespond ONLY with valid JSON. No markdown, no prose outside JSON.")

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
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
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
        if task_id not in (1, 2, 3):
            raise ValueError(f"task_id {task_id} not supported for training (only 1-3)")

        reset_body: Dict[str, Any] = {"task_id": task_id, "noise_level": noise_level}
        if hadm_id is not None:
            reset_body["hadm_id"] = hadm_id

        obs   = self.client.post("/reset", reset_body)
        done  = False
        steps: List[Dict] = []

        while not done:
            step_num = int(obs.get("step_num", 0))

            # Task 2: auto-request labs+meds+microbiology on step 0 when data is sparse
            # This unlocks the full medication/lab data before the LLM generates the care plan.
            if task_id == 2 and step_num == 0:
                meds_empty = not (obs.get("pharmacy_active") or obs.get("medications"))
                labs_empty = not obs.get("lab_flags")
                if meds_empty or labs_empty:
                    info_req = {"task_id": 2, "information_request": ["labs", "medications", "microbiology"]}
                    try:
                        info_result = self.client.post("/step", info_req)
                        # Update obs with the enriched observation (has meds+labs now)
                        enriched = info_result.get("observation")
                        if enriched:
                            obs = enriched
                        # Info requests return reward=0, done=False — continue to LLM step
                        if info_result.get("done"):
                            done = True
                            steps.append({
                                "prompt": "", "response": "", "reward": 0.0,
                                "partial": {}, "parse_ok": False,
                                "error_type": "info_req_done", "step_num": step_num,
                            })
                            break
                    except Exception as e:
                        print(f"  [INFO_REQ] failed: {e}")

            prompt             = format_observation(obs, task_id)
            response_txt, llm_err = self._llm(
                prompt,
                max_new_tokens=512 if task_id == 1 else (1024 if task_id == 2 else 3072),
            )

            # Parse + normalize action
            error_type = llm_err
            if llm_err:
                action_dict = _fallback_action(task_id, obs)
                parse_ok    = False
            else:
                raw = _extract_json(response_txt)
                parse_ok = raw is not None
                if parse_ok:
                    action_dict = _normalize_action(raw, task_id)
                    error_type  = ""
                else:
                    action_dict = _fallback_action(task_id, obs)
                    error_type  = "parse_fail"

            # For Task 2 on step >= 1: never send another information_request
            if task_id == 2 and step_num >= 1 and action_dict.get("information_request"):
                action_dict = _fallback_action(task_id, obs)
                error_type  = "t2_forced_plan"
                parse_ok    = False

            # Submit to env; handle schema validation errors gracefully
            try:
                result = self.client.post("/step", action_dict)
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", True))
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 422:
                    print(f"  [422] schema error. Using fallback. Detail: {e.args[0][:120]}")
                    error_type = "schema_422"
                    parse_ok   = False
                    try:
                        result = self.client.post("/step", _fallback_action(task_id, obs))
                        reward = float(result.get("reward", 0.0))
                        done   = bool(result.get("done", True))
                    except Exception:
                        reward = 0.0
                        done   = True
                        result = {}
                else:
                    raise

            # When the LLM failed to produce a valid action the env was stepped with
            # a heuristic fallback to keep the episode moving.  That fallback's reward
            # belongs to the heuristic, not to the LLM's output — zero it so the
            # dataset never contains (garbled_text → 0.3-0.6 reward) pairs.
            if not parse_ok:
                reward = 0.0

            steps.append({
                "prompt":     prompt,
                "response":   response_txt,
                "reward":     reward,
                "partial":    result.get("partial_signals", {}),
                "parse_ok":   parse_ok,
                "error_type": error_type,
                "step_num":   step_num,
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
