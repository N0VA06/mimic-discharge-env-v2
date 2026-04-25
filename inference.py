"""
inference.py — MIMIC Discharge Planning
Structured stdout: [START], [STEP], [END]
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "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")
BENCHMARK    = "mimic-discharge-planning"
MAX_STEPS    = 10

_TASK_NAMES = {
    1: "discharge-disposition",
    2: "care-plan",
    3: "discharge-note",
    4: "admission-to-discharge-workflow",
}

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

# ── Stdout loggers ─────────────────────────────────────────────────────────────

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    err = error or "null"
    print(f"[STEP] step={step} action={action[:80]} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}", flush=True)

# ── Health check wait ──────────────────────────────────────────────────────────

def wait_for_server(url: str, timeout: int = 120) -> bool:
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

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM = (
    "You are a clinical physician producing structured discharge planning outputs. "
    "Output ONLY a single valid JSON object. No markdown fences, no text outside JSON. "
    "Close every string. Never invent drug names or diagnoses not in the patient record. "
    "Always base your decisions on the specific clinical data provided."
)

# ── Task schemas ───────────────────────────────────────────────────────────────

T1_SCHEMA = """Analyze ALL clinical data and output EXACTLY:
{
  "task_id": 1,
  "task1": {
    "disposition": "<ONE OF: expired|hospice|ama|snf|rehab|home_with_services|home|other>",
    "reasoning": "<20-50 words citing specific clinical evidence from this patient>"
  }
}

DECISION ALGORITHM (apply in strict order — stop at first match):
═══════════════════════════════════════════════════════════
STEP 1 → HOSPICE (choose if ANY are true):
  • Diagnoses contain "palliative care" (ICD V667 or Z51.5)
  • Diagnoses contain "secondary malignant neoplasm" (metastatic cancer to brain/lung/other)
  • DRG text contains "MALIGNANCY" AND severity=4.0 AND mortality=4.0
  • GCS Total discharge value ≤ 5 WITH terminal/malignant diagnosis
  • Multiple metastatic sites (brain + lung + adrenal, etc.)

STEP 2 → EXPIRED:
  • DRG or diagnoses explicitly state DIED, EXPIRED, DECEASED

STEP 3 → SNF (NOT terminal, any apply):
  • Ventilation hours > 0 AND patient age > 60 AND non-terminal
  • Active dialysis AND non-terminal
  • LOS > 7 days + age > 70 + functional decline

STEP 4 → REHAB (applies for trauma/orthopedic recovery):
  • Open or closed fracture with surgical fixation (femur, hip, spine)
  • Young patient (<65) needing intense physical/occupational therapy

STEP 5 → HOME WITH SERVICES:
  • Needs IV antibiotics, wound care, or visiting nurse at home
  • Discharge orders include home health care

STEP 6 → HOME (default for stable patients):
  • Discharge orders finalized + "Discharge Now" present
  • Pharmacy active = only oral/inhaled medications (no IV)
  • Short LOS (< 5 days), hemodynamically stable

═══════════════════════════════════════════════════════════
CRITICAL: Terminal cancer / palliative care ICD → ALWAYS hospice, NEVER snf/home"""

T2_SCHEMA = """Recommend post-discharge care plan. Output EXACTLY:
{
  "task_id": 2,
  "task2": {
    "follow_up_specialties": ["<specialty1>", "<specialty2>"],
    "medications_to_continue": ["<drug1>", "<drug2>"],
    "medications_to_discontinue": ["<drug1>"],
    "key_instructions": [
      "<specific instruction with numbers/thresholds/timeframes>",
      "<specific instruction with numbers/thresholds/timeframes>",
      "<specific instruction with numbers/thresholds/timeframes>",
      "<specific instruction with numbers/thresholds/timeframes>",
      "<specific instruction with numbers/thresholds/timeframes>"
    ],
    "reasoning": "<max 20 words>"
  }
}

SPECIALTY RULES — Derive ONLY from actual diagnoses listed:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ICD-9 numeric code ranges:
  001–139  → Infectious Disease     520–579  → Gastroenterology
  140–239  → Oncology               580–629  → Nephrology
  240–279  → Endocrinology          630–677  → Obstetrics/Gynecology
  290–319  → Psychiatry             680–709  → Dermatology
  320–389  → Neurology              710–739  → Rheumatology
  390–459  → Cardiology             800–999  → Trauma Surgery / Orthopedics
  460–519  → Pulmonology

ICD-10 first-letter:
  A,B → Infectious Disease   I → Cardiology    K → Gastroenterology
  C   → Oncology             J → Pulmonology   N → Nephrology
  E   → Endocrinology        G → Neurology     M → Rheumatology

Microbiology organisms identified → ALWAYS add Infectious Disease
ALWAYS include: Primary Care
Only list specialties supported by the patient's actual diagnoses.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEDICATION RULES (critical — determines your score):
  medications_to_continue   = EXACT drug names from "ACTIVE MEDICATIONS AT DISCHARGE" list ONLY
  medications_to_discontinue = EXACT drug names from "STOPPED/DISCONTINUED MEDICATIONS" list ONLY
  DO NOT invent or modify drug names. DO NOT use medications from "MEDICATION ORDERS" section
  unless the drug also appears in the ACTIVE MEDICATIONS list.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTION RULES (each must have specific measurable values):
  Activity  : "Walk 10 minutes 3x/day; avoid lifting > 10 lbs for 4 weeks"
  Diet      : "Low-sodium diet under 2000 mg/day; limit fluids to 1.5L/day"
  Medication: "Take levofloxacin 750mg once daily for 14 days with food"
  Follow-up : "See gastroenterology within 2 weeks for follow-up imaging"
  Warning   : "Return to ED immediately for fever > 38.5°C, severe abdominal pain, or jaundice" """

T3_SCHEMA = """Generate a complete clinical discharge note. Output EXACTLY:
{
  "task_id": 3,
  "task3": {
    "discharge_note": "<FULL DISCHARGE NOTE — MINIMUM 300 WORDS>"
  }
}

MANDATORY REQUIREMENTS (each directly scored):
═══════════════════════════════════════════════════════════
1. DIAGNOSES: Name EACH of the top diagnoses using their EXACT description words in the note.
   The note text must contain keywords from each diagnosis title. For example:
   - "Abscess of liver"      → write "liver abscess" or "hepatic abscess"
   - "Intracerebral hemorrhage" → write "intracerebral hemorrhage"
   - "Cerebral edema"        → write "cerebral edema"
   Do NOT just list ICD codes. Use the medical description words.

2. MEDICATIONS: List ONLY drugs from "ACTIVE MEDICATIONS AT DISCHARGE" — use exact names.
   NEVER invent medications. NEVER use drugs from "MEDICATION ORDERS" unless also in active list.
   If pharmacy active is empty, write "No medications prescribed at discharge."

3. LOS: Must explicitly state the length of stay as a number of days matching the LOS shown
   (e.g., "admitted for 3.3 days" or "hospital stay of 3 days"). Within 25% of actual is accepted.

4. DISPOSITION PHRASE: Use exactly ONE of these verbatim in the DISCHARGE DISPOSITION section:
   • "The patient was discharged home."
   • "The patient was discharged home with home health services."
   • "The patient was transferred to a skilled nursing facility."
   • "The patient was transferred to inpatient rehabilitation."
   • "The patient was transitioned to hospice care."
   • "The patient expired during this hospitalization."

5. REQUIRED SECTIONS (in this order, use these exact headers):
   PRINCIPAL DIAGNOSIS
   BRIEF HOSPITAL COURSE
   KEY PROCEDURES PERFORMED
   DISCHARGE CONDITION
   DISCHARGE DISPOSITION
   DISCHARGE MEDICATIONS
   FOLLOW-UP INSTRUCTIONS
═══════════════════════════════════════════════════════════"""

T4_SCHEMA = {
    1:  """Output: {"task_id": 4, "task4": {"triage_level": "<icu|stepdown|floor>"}}
Rules: icu if any ICU stay exists; stepdown if intermediate care; floor otherwise.""",

    2:  """Output: {"task_id": 4, "task4": {"priority_labs": ["<lab1>", ...], "priority_consults": ["<specialty>", ...]}}
Labs: Order labs matching abnormal flags (hematology, metabolic, renal, hepatic, cardiac).
Consults: Derive from ICD codes. ICD9: 001-139→Infectious Disease, 240-279→Endocrinology,
390-459→Cardiology, 460-519→Pulmonology, 520-579→Gastroenterology, 580-629→Nephrology,
710-739→Rheumatology, 800-999→Trauma Surgery/Orthopedics.""",

    3:  """Output: {"task_id": 4, "task4": {"interventions": ["<intervention1>", ...]}}
Include interventions matching icu_procedures shown (ventilation→intubation,
arterial lines, central lines, dialysis, fluid resuscitation as needed).""",

    4:  """Output: {"task_id": 4, "task4": {"high_risk_medications": ["<drug1>", ...]}}
List all high-risk medications from the medication list:
anticoagulants (heparin, warfarin, enoxaparin), vasopressors (norepinephrine, dopamine),
opioids (morphine, hydromorphone, fentanyl), sedatives (propofol, midazolam),
insulin, vancomycin, aminoglycosides, chemotherapy agents.""",

    5:  """Output: {"task_id": 4, "task4": {"antibiotic_strategy": "<none|targeted|broad|empiric>", "antibiotics": ["<drug>", ...]}}
Strategy: targeted if specific organisms identified in microbiology;
empiric if suspected infection but no culture; broad if multiple resistant organisms;
none if no infection present. List antibiotics from medication list.""",

    6:  """Output: {"task_id": 4, "task4": {"fluid_strategy": "<restrict_diuresis|aggressive_resuscitation|maintain>"}}
restrict_diuresis: fluid overloaded (positive balance, edema, pulmonary edema diagnosis)
aggressive_resuscitation: hypovolemic (oliguria, negative balance, septic shock)
maintain: hemodynamically stable with normal fluid balance""",

    7:  """Output: {"task_id": 4, "task4": {"ready_for_stepdown": <true|false>, "barriers": ["<barrier1>", ...]}}
Ready if: patient stable, LOS > 3 days, not on vasopressors, not on ventilation.
Barriers: ongoing ventilation, hemodynamic instability, active sepsis, IV vasopressors,
pending urgent procedures, altered mental status requiring ICU monitoring.""",

    8:  """Output: {"task_id": 4, "task4": {"predicted_disposition": "<home|snf|rehab|hospice|home_with_services|other>", "los_remaining_days": <number>}}
Disposition: same rules as Task 1. LOS remaining = total_hospital_los - days_already_spent.
Use the hospital_los_days field minus days elapsed (step_num / 10 * hospital_los_days approx).""",

    9:  """Output: {"task_id": 4, "task4": {"medications_to_continue": ["<drug1>", ...]}}
List all medications from pharmacy_active that should continue after discharge.
Omit vasopressors, paralytics, IV-only drugs not suitable for outpatient use.""",

    10: """Output: {"task_id": 4, "task4": {"final_note": "<FULL DISCHARGE NOTE — MINIMUM 300 WORDS>"}}
Requirements identical to Task 3:
1. Name EACH diagnosis by its description keywords
2. Use ONLY drugs from pharmacy_active list
3. State exact LOS in days
4. Use exact disposition phrase from: "The patient was discharged home." /
   "The patient was discharged home with home health services." /
   "The patient was transferred to a skilled nursing facility." /
   "The patient was transferred to inpatient rehabilitation." /
   "The patient was transitioned to hospice care." /
   "The patient expired during this hospitalization."
5. Include all 7 sections: PRINCIPAL DIAGNOSIS, BRIEF HOSPITAL COURSE, KEY PROCEDURES PERFORMED,
   DISCHARGE CONDITION, DISCHARGE DISPOSITION, DISCHARGE MEDICATIONS, FOLLOW-UP INSTRUCTIONS""",
}

SCHEMAS = {1: T1_SCHEMA, 2: T2_SCHEMA, 3: T3_SCHEMA}

# ── ICD-9 specialty mapping (for fallbacks) ────────────────────────────────────

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
    'T': 'Trauma Surgery',
}


def _specialty_from_dx(diagnoses: List[Dict]) -> List[str]:
    specs: List[str] = ["Primary Care"]
    seen: Set[str] = {"Primary Care"}
    for dx in diagnoses[:8]:
        code    = str(dx.get("icd_code", "")).strip().upper()
        version = int(dx.get("icd_version", 10) or 10)
        if not code:
            continue
        spec = None
        if version == 9:
            digits = re.sub(r"\D", "", code)[:3]
            if digits:
                num = int(digits)
                for lo, hi, s in _ICD9_RANGES:
                    if lo <= num <= hi:
                        spec = s
                        break
        else:
            spec = _ICD10_PREFIXES.get(code[0])
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
       any("MALIGNANCY" in drg or "MALIGNANT" in drg for drg in drg_codes):
        return True
    # Low GCS with terminal disease
    vitals = obs.get("vitals", []) or []
    for v in vitals:
        if v.get("name") == "GCS Total":
            gcs_disch = v.get("discharge_value")
            if gcs_disch is not None and float(gcs_disch) <= 5:
                terminal_dx = any(
                    any(kw in d for kw in ["malignant", "malignancy", "metastasis", "metastatic", "neoplasm"])
                    for d in dx_descs
                )
                if terminal_dx:
                    return True
    return False


# ── Observation → prompt ───────────────────────────────────────────────────────

def _format_obs(obs: Dict[str, Any], task_id: int, step_num: int = 0) -> str:
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
        lines.append(f"ICU ventilation: {vent:.1f} hours")
    if dial:
        lines.append("Dialysis used: YES")

    icu_stays = obs.get("icu_stays", []) or []
    if icu_stays:
        stay = icu_stays[0] if isinstance(icu_stays[0], dict) else {}
        lines.append(f"ICU: {stay.get('los_days','?'):.2f} days in {stay.get('first_careunit','?')}")

    traj = obs.get("care_trajectory", []) or []
    if traj:
        lines.append(f"Care path: {' → '.join(traj)}")

    # Vitals — especially GCS (key for end-of-life decisions)
    vitals = obs.get("vitals", []) or []
    if vitals:
        lines.append("\n-- VITALS (admission → discharge) --")
        for v in vitals:
            name     = v.get("name", "")
            adm      = v.get("admission_value", "?")
            dis      = v.get("discharge_value", "?")
            critical = "  ← CRITICAL" if v.get("critical_flag") else ""
            lines.append(f"  {name:20s}: {adm} → {dis}{critical}")

    # Discharge orders — strong signal for disposition
    dorders = obs.get("discharge_orders") or {}
    if dorders:
        lines.append("\n-- DISCHARGE ORDERS --")
        lines.append(f"  Finalized: {dorders.get('discharge_planning_finalized', False)}")
        order_types = dorders.get("documented_discharge_orders", [])
        if order_types:
            lines.append(f"  Types: {', '.join(order_types)}")

    if obs.get("diagnoses"):
        lines.append("\n-- DIAGNOSES (derive specialties from these ICD codes) --")
        for dx in obs["diagnoses"][:10]:
            lines.append(
                f"  [{dx.get('seq_num',0):2d}] ICD{dx.get('icd_version','?')}-{dx.get('icd_code',''):10s}  {dx.get('description','')}"
            )

    # Active medications — use THESE for medications_to_continue
    meds_active = obs.get("pharmacy_active", []) or []
    if meds_active:
        lines.append("\n-- ACTIVE MEDICATIONS AT DISCHARGE (use EXACT names for medications_to_continue) --")
        for m in meds_active:
            lines.append(f"  ✓ {m}")
    else:
        lines.append("\n-- ACTIVE MEDICATIONS AT DISCHARGE: None --")

    # Stopped medications — use THESE for medications_to_discontinue
    meds_stopped = obs.get("pharmacy_stopped", []) or []
    if meds_stopped:
        lines.append("\n-- STOPPED/DISCONTINUED MEDICATIONS (use for medications_to_discontinue) --")
        for m in meds_stopped[:10]:
            lines.append(f"  ✗ {m}")

    # Medication orders (broader context, but NOT the active discharge list)
    if obs.get("medications"):
        lines.append("\n-- MEDICATION ORDERS during admission (reference only) --")
        for m in obs["medications"][:10]:
            drug  = m.get("drug", "")
            route = m.get("route", "")
            dose  = m.get("dose_val_rx", "")
            lines.append(f"  {drug} {dose} [{route}]".strip())

    if obs.get("lab_flags"):
        lines.append("\n-- ABNORMAL LABS --")
        for lab in obs["lab_flags"][:10]:
            lines.append(f"  {lab.get('label','?'):30s} {str(lab.get('value','?')):>10}  [{lab.get('flag','?')}]")

    # Microbiology — triggers Infectious Disease specialty
    microbio = obs.get("microbiology", []) or []
    if microbio:
        lines.append("\n-- MICROBIOLOGY (organisms → add Infectious Disease) --")
        for m in microbio[:5]:
            org = m.get("organism") or m.get("org_name") or str(m)
            lines.append(f"  ⚠ {org}")

    if obs.get("procedures"):
        lines.append("\n-- PROCEDURES --")
        for p in obs["procedures"][:6]:
            lines.append(f"  {p}")

    if obs.get("drg_codes"):
        lines.append("\n-- DRG --")
        for d in obs["drg_codes"][:2]:
            lines.append(f"  {d}")

    # eMAR active drugs — additional source for discharge medications (tasks 2, 3)
    emar = obs.get("emar_summary", []) or []
    if emar and task_id in (2, 3, 4):
        active_emar = [e.get("medication", "") for e in emar if e.get("active_at_discharge")]
        if active_emar:
            lines.append("\n-- MEDICATIONS ACTIVE AT DISCHARGE (eMAR confirmation) --")
            for med in active_emar[:10]:
                lines.append(f"  ✓ {med}")

    if obs.get("fluid_balance") and task_id == 4:
        fb = obs["fluid_balance"]
        lines.append(f"\n-- FLUID BALANCE --")
        lines.append(f"  Net balance: {fb.get('net_balance_ml','?')} mL")
        lines.append(f"  Overloaded: {fb.get('fluid_overloaded','?')}  Oliguria: {fb.get('oliguria','?')}")

    # Task 4: episode history
    if task_id == 4 and obs.get("episode_history"):
        lines.append("\n-- PRIOR STEPS --")
        for h in obs["episode_history"]:
            lines.append(f"  Step {h.get('step_num')}: {h.get('action_summary','')}")

    lines.append(f"\n=== TASK {task_id} ===")
    lines.append(obs.get("task_description", ""))

    if task_id == 4:
        schema = T4_SCHEMA.get(step_num, T4_SCHEMA[10])
    else:
        schema = SCHEMAS[task_id]
    lines += ["", "=== SCHEMA ===", schema]
    return "\n".join(lines)


# ── Fallbacks ──────────────────────────────────────────────────────────────────

def _fallback(task_id: int, obs: Dict[str, Any], step_num: int = 0) -> Dict:
    drugs_active  = [str(m) for m in (obs.get("pharmacy_active")  or [])[:8]]
    drugs_stopped = [str(m) for m in (obs.get("pharmacy_stopped") or [])[:6]]
    los           = float(obs.get("hospital_los_days", 0) or 0)
    icu_p         = obs.get("icu_procedures", {}) or {}
    vent          = float(icu_p.get("ventilation_hours", 0) or 0)
    dial          = bool(icu_p.get("has_dialysis", False))
    age           = int(obs.get("age", 65) or 65)
    diagnoses     = obs.get("diagnoses", []) or []

    eol     = _is_end_of_life(obs)
    facility = vent > 0 or dial or (los > 7 and age > 70)

    if task_id == 1:
        dx_descs = [str(d.get("description", "")).lower() for d in diagnoses]
        is_ortho = any(kw in d for d in dx_descs for kw in ("fracture", "orthop", "femur", "hip"))
        if eol:
            disp   = "hospice"
            reason = "Terminal malignancy with palliative care indicators; end-of-life trajectory."
        elif is_ortho and not facility:
            disp   = "rehab"
            reason = "Orthopedic fracture with surgical fixation requiring rehabilitation."
        elif facility:
            disp   = "snf"
            reason = "ICU stay, ventilation, or dialysis with advanced age requires skilled nursing."
        else:
            disp   = "home"
            reason = "Hemodynamically stable, short LOS, oral medications only."
        return {"task_id": 1, "task1": {"disposition": disp, "reasoning": reason}}

    if task_id == 2:
        specs = _specialty_from_dx(diagnoses)
        microbio = obs.get("microbiology", []) or []
        if microbio and "Infectious Disease" not in specs:
            specs.append("Infectious Disease")
        return {"task_id": 2, "task2": {
            "follow_up_specialties": specs[:4],
            "medications_to_continue": drugs_active,
            "medications_to_discontinue": drugs_stopped[:3],
            "key_instructions": [
                "Follow up with primary care within 1 week of discharge.",
                "Take all medications as prescribed; complete full antibiotic course if applicable.",
                "Maintain a low-sodium diet under 2000 mg per day.",
                "Weigh yourself every morning; call doctor if weight gain exceeds 2 lbs in 1 day or 5 lbs in 1 week.",
                "Return to ED immediately for fever over 38.5°C, chest pain, shortness of breath, or severe pain.",
            ],
            "reasoning": "Care plan derived from diagnosis codes and active medication list.",
        }}

    if task_id == 3:
        dx_list = "\n".join(
            f"{i+1}. {d.get('description','Unknown')}"
            for i, d in enumerate(diagnoses[:5])
        )
        drug_list = "\n".join(f"- {d}" for d in drugs_active) or "- No medications prescribed at discharge"
        disp_phrase = "The patient was transitioned to hospice care." if eol else (
            "The patient was transferred to a skilled nursing facility." if facility else
            "The patient was discharged home with home health services." if drugs_active else
            "The patient was discharged home."
        )
        procs = obs.get("procedures", []) or []
        proc_text = "; ".join(procs[:3]) if procs else "Routine monitoring and laboratory evaluation"

        # Build LOS text
        los_text = f"{los:.1f}" if los else "the required number of"

        note = (
            f"PRINCIPAL DIAGNOSIS: {diagnoses[0].get('description', 'Acute medical illness') if diagnoses else 'Acute medical illness'}\n\n"
            f"BRIEF HOSPITAL COURSE: The patient is a {obs.get('age','?')}-year-old "
            f"{obs.get('gender','?').lower() if obs.get('gender') else 'patient'} "
            f"admitted via {obs.get('admission_location','the emergency room').lower()} "
            f"with {diagnoses[0].get('description','acute illness') if diagnoses else 'acute illness'}. "
            f"The hospital stay lasted {los_text} days. "
            f"The patient was managed for the following diagnoses:\n{dx_list}\n"
            "All active medical problems were addressed. Specialist consultations were obtained as indicated. "
            "The patient's condition stabilised with appropriate treatment and the patient was deemed safe for discharge.\n\n"
            f"KEY PROCEDURES PERFORMED: {proc_text}\n\n"
            "DISCHARGE CONDITION: Stable, improved from admission baseline. The patient was alert and oriented "
            "with vital signs within acceptable limits at the time of discharge.\n\n"
            f"DISCHARGE DISPOSITION: {disp_phrase}\n\n"
            f"DISCHARGE MEDICATIONS:\n{drug_list}\n\n"
            "FOLLOW-UP INSTRUCTIONS: Follow up with primary care within 1 week. "
            "Continue all prescribed medications for the full course. "
            "Maintain a low-sodium diet under 2000 mg per day. "
            "Weigh yourself every morning and call your doctor if weight increases more than 2 lbs in one day or 5 lbs in one week. "
            "Return to the emergency department immediately for fever over 38.5°C, chest pain, worsening shortness of breath, "
            "severe abdominal pain, or any sudden change in your condition."
        )
        return {"task_id": 3, "task3": {"discharge_note": note}}

    # Task 4 fallbacks
    if task_id == 4:
        return _fallback_task4(obs, step_num)

    return {"task_id": task_id}


def _fallback_task4(obs: Dict, step_num: int) -> Dict:
    icu_p    = obs.get("icu_procedures", {}) or {}
    vent     = float(icu_p.get("ventilation_hours", 0) or 0)
    dial     = bool(icu_p.get("has_dialysis", False))
    drugs    = [str(m) for m in (obs.get("pharmacy_active") or [])[:8]]
    diagnoses = obs.get("diagnoses", []) or []
    eol       = _is_end_of_life(obs)
    los       = float(obs.get("hospital_los_days", 0) or 0)
    icu_stays = obs.get("icu_stays", []) or []
    facility  = vent > 0 or dial or (los > 7 and int(obs.get("age", 65) or 65) > 70)

    if step_num == 1:
        tier = "icu" if icu_stays else ("stepdown" if vent > 0 else "floor")
        return {"task_id": 4, "task4": {"triage_level": tier}}
    if step_num == 2:
        specs   = _specialty_from_dx(diagnoses)
        labs    = ["CBC", "CMP", "Coagulation panel", "Lactate", "Blood cultures"]
        return {"task_id": 4, "task4": {"priority_labs": labs, "priority_consults": specs[1:]}}
    if step_num == 3:
        ivs = ["IV access", "Continuous monitoring"]
        if vent > 0:
            ivs.insert(0, "Mechanical ventilation")
        if icu_p.get("has_arterial_line"):
            ivs.append("Arterial line")
        if icu_p.get("has_central_line"):
            ivs.append("Central venous catheter")
        if dial:
            ivs.append("Hemodialysis")
        return {"task_id": 4, "task4": {"interventions": ivs}}
    if step_num == 4:
        all_meds = obs.get("medications") or []
        high_risk_kw = {"heparin", "warfarin", "insulin", "morphine", "hydromorph",
                        "fentanyl", "vancomycin", "norepinephrine", "dopamine", "propofol",
                        "midazolam", "lorazepam", "enoxaparin"}
        hr_meds = [
            m.get("drug","") for m in all_meds
            if any(kw in m.get("drug","").lower() for kw in high_risk_kw)
        ]
        return {"task_id": 4, "task4": {"high_risk_medications": hr_meds or ["Heparin", "Insulin"]}}
    if step_num == 5:
        microbio = obs.get("microbiology", []) or []
        meds     = obs.get("medications", []) or []
        abx_kw   = {"ceftri", "vancom", "pipera", "meropenem", "levoflox",
                    "metronidazole", "ampicillin", "ciproflox", "cefazolin"}
        abx = [m.get("drug","") for m in meds
               if any(kw in m.get("drug","").lower() for kw in abx_kw)]
        strategy = "targeted" if microbio else ("empiric" if abx else "none")
        return {"task_id": 4, "task4": {"antibiotic_strategy": strategy, "antibiotics": abx}}
    if step_num == 6:
        fb = obs.get("fluid_balance") or {}
        overloaded = bool(fb.get("fluid_overloaded"))
        oliguria   = bool(fb.get("oliguria"))
        dx_descs   = [str(d.get("description","")).lower() for d in diagnoses]
        pulm_edema = any("edema" in d and ("lung" in d or "pulmonary" in d) for d in dx_descs)
        if overloaded or pulm_edema:
            strategy = "restrict_diuresis"
        elif oliguria or (fb.get("net_balance_ml") is not None and float(fb.get("net_balance_ml", 0)) < -1000):
            strategy = "aggressive_resuscitation"
        else:
            strategy = "maintain"
        return {"task_id": 4, "task4": {"fluid_strategy": strategy}}
    if step_num == 7:
        barriers = []
        if vent > 0:
            barriers.append("Ongoing mechanical ventilation")
        if dial:
            barriers.append("Active dialysis dependency")
        ready = not barriers and los > 3 and bool(icu_stays)
        return {"task_id": 4, "task4": {"ready_for_stepdown": ready, "barriers": barriers}}
    if step_num == 8:
        disp = "hospice" if eol else ("snf" if facility else "home")
        los_remain = max(0, los - step_num * 0.5)
        return {"task_id": 4, "task4": {"predicted_disposition": disp, "los_remaining_days": round(los_remain, 1)}}
    if step_num == 9:
        return {"task_id": 4, "task4": {"medications_to_continue": drugs}}
    # step 10
    return {"task_id": 4, "task4": _build_final_note(obs)}


def _build_final_note(obs: Dict) -> Dict:
    fallback_note = _fallback(3, obs)
    return {"final_note": fallback_note["task3"]["discharge_note"]}


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
    fixed   = candidate.rstrip()
    stack   = []
    in_str  = False
    i       = 0
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

# ── LLM call ───────────────────────────────────────────────────────────────────

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
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", flush=True)
        return None

_MAX_TOKENS = {1: 512, 2: 1024, 3: 3072, 4: 3072}

def _get_action_dict(obs: Dict, task_id: int, step_num: int = 0) -> Dict:
    # Task 2: first step should request labs+meds+microbiology to improve care plan quality
    if task_id == 2 and step_num == 0:
        meds_empty = not (obs.get("pharmacy_active") or obs.get("medications"))
        labs_empty = not obs.get("lab_flags")
        if meds_empty or labs_empty:
            return {"task_id": 2, "information_request": ["labs", "medications", "microbiology"]}

    prompt = _format_obs(obs, task_id, step_num)
    raw    = _call_llm(prompt, _MAX_TOKENS.get(task_id, 1024))
    if raw:
        parsed = _extract_json(raw)
        if parsed:
            parsed["task_id"] = task_id
            # For Task 2 on step >= 1, never send an info_request — must submit plan
            if task_id == 2 and step_num >= 1 and parsed.get("information_request"):
                del parsed["information_request"]
                if not parsed.get("task2"):
                    return _fallback(task_id, obs, step_num)
            return parsed
    return _fallback(task_id, obs, step_num)

# ── Build Pydantic Action ──────────────────────────────────────────────────────

def _build_action(d: Dict, task_id: int):
    from environment import Action
    from environment.models import Task1Action, Task2Action, Task3Action, Task4Action

    if task_id == 1:
        t = d.get("task1", {})
        return Action(task_id=1, task1=Task1Action(
            disposition=t.get("disposition", "home"),
            reasoning=t.get("reasoning", ""),
        ))
    if task_id == 2:
        if d.get("information_request"):
            return Action(task_id=2, information_request=d["information_request"])
        t = d.get("task2", {})
        return Action(task_id=2, task2=Task2Action(
            follow_up_specialties=t.get("follow_up_specialties", []),
            medications_to_continue=t.get("medications_to_continue", []),
            medications_to_discontinue=t.get("medications_to_discontinue", []),
            key_instructions=t.get("key_instructions", []),
            reasoning=t.get("reasoning", ""),
        ))
    if task_id == 3:
        t = d.get("task3", {})
        return Action(task_id=3, task3=Task3Action(
            discharge_note=t.get("discharge_note", ""),
        ))
    if task_id == 4:
        t = d.get("task4", {})
        return Action(task_id=4, task4=Task4Action(**{k: v for k, v in t.items() if v is not None}))

    raise ValueError(f"Unknown task_id: {task_id}")

def _action_summary(d: Dict, task_id: int) -> str:
    if task_id == 1:
        return f"disposition={d.get('task1', {}).get('disposition', '?')}"
    if task_id == 2:
        if d.get("information_request"):
            return f"info_request={d['information_request']}"
        specs = d.get("task2", {}).get("follow_up_specialties", [])[:2]
        return f"specialties=[{','.join(specs)}]"
    if task_id == 3:
        note = d.get("task3", {}).get("discharge_note", "")
        return f"note({len(note.split())}words)"
    if task_id == 4:
        t4 = d.get("task4", {})
        for key in ("triage_level", "antibiotic_strategy", "fluid_strategy",
                    "ready_for_stepdown", "predicted_disposition"):
            if key in t4:
                return f"step_{key}={t4[key]}"
        if "final_note" in t4:
            return f"final_note({len(str(t4['final_note']).split())}words)"
        return str(list(t4.keys())[:2])
    return "?"

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_id: int) -> float:
    from environment import MIMICDischargeEnv

    task_name  = _TASK_NAMES[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task_name)

    try:
        env       = MIMICDischargeEnv()
        obs_model = env.reset(task_id=task_id)
        obs       = obs_model.model_dump()
        max_steps = obs.get("max_steps", MAX_STEPS)

        for step in range(1, max_steps + 1):
            try:
                step_num    = obs.get("step_num", step)
                # T4_SCHEMA keys are 1-indexed; env step_num is 0-indexed (0 before any step taken)
                t4_step     = (step_num + 1) if task_id == 4 else step_num
                action_dict = _get_action_dict(obs, task_id, t4_step)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    wait_for_server(ENV_URL, timeout=120)

    scores: Dict[int, float] = {}
    for task_id in [1, 2, 3, 4]:
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
