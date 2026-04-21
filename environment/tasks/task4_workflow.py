"""
Task 4 (Very Hard) — Long-Horizon Admission-to-Discharge Workflow.

10-step episode simulating inpatient management from admission triage to final note.
Sparse reward: only step 10 contributes to the GRPO training gradient.
Steps 1-9 return reward=0 but record partial scores in shaping_log.

Revision mechanism: agent may revise any prior step (max 2, -0.02 final score each)
by including revise_step + revision fields in their action.

Step max contributions (normalised 0–1 then weighted in step 10):
  Step 1:  0.10  triage level accuracy
  Step 2:  0.15  priority labs + consult plan
  Step 3:  0.15  intervention plan vs icu_procedures
  Step 4:  0.10  high-risk medication identification
  Step 5:  0.10  antibiotic stewardship plan
  Step 6:  0.08  fluid management strategy
  Step 7:  0.08  ICU-to-stepdown readiness
  Step 8:  0.10  discharge disposition + LOS estimate
  Step 9:  0.10  discharge medication reconciliation
  Step 10: NoteGrader×0.60 + shaping_avg×0.40
           +0.10 consistency bonus (disposition matches step 8)
           +0.05 trajectory bonus (≥50% specialty overlap with step 2)
           −0.02 per revision used (max 2)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .task1_disposition import normalize_mimic_location, BROAD_GROUP, _ADJACENT
from .task3_note import NoteGrader as _NoteGrader


# ─── ICU care unit classification ─────────────────────────────────────────────

_ICU_KW: Set[str] = {
    "micu", "sicu", "ccu", "cvicu", "tsicu", "neuro icu", "burn icu",
    "medical intensive care", "surgical intensive care",
    "cardiac vascular", "coronary care", "trauma sicu", "neuro intensive",
}
_STEPDOWN_KW: Set[str] = {
    "stepdown", "step down", "step-down", "msicu",
    "medical/surgical intensive", "intermediate", "imu", "progressive care",
}


def _careunit_tier(careunit: str) -> str:
    cu = careunit.lower()
    if any(kw in cu for kw in _ICU_KW):
        return "icu"
    if any(kw in cu for kw in _STEPDOWN_KW):
        return "stepdown"
    return "floor"


# ─── Step max rewards (for normalisation) ─────────────────────────────────────

_STEP_MAX: Dict[int, float] = {
    1: 0.10, 2: 0.15, 3: 0.15, 4: 0.10,
    5: 0.10, 6: 0.08, 7: 0.08, 8: 0.10, 9: 0.10,
}


# ─── Step 1 — Acuity Triage ───────────────────────────────────────────────────

def _grade_step_1(t4, episode: Dict) -> Tuple[float, Dict]:
    pred = (t4.triage_level or "").strip().lower()
    icu_stays = episode.get("icu_stays") or []
    if icu_stays:
        first_cu  = icu_stays[0].get("first_careunit", "") if isinstance(icu_stays[0], dict) else ""
        true_tier = _careunit_tier(first_cu)
    else:
        true_tier = "floor"

    if pred == true_tier:
        score = 1.0
    elif {pred, true_tier} <= {"icu", "stepdown"}:
        score = 0.5
    elif pred in ("floor", "stepdown") and true_tier == "floor":
        score = 0.5
    else:
        score = 0.0

    return round(score * _STEP_MAX[1], 4), {
        "step1_triage_score": round(score, 4),
        "predicted_tier": pred,
        "true_tier": true_tier,
    }


# ─── Step 2 — Priority Labs + Consults ────────────────────────────────────────

_LAB_CATS: Dict[str, List[str]] = {
    "renal":      ["creatinine", "bun", "potassium", "sodium"],
    "cardiac":    ["troponin", "bnp", "pro-bnp"],
    "hepatic":    ["ast", "alt", "bilirubin", "albumin"],
    "hematology": ["hemoglobin", "hematocrit", "platelet", "inr"],
    "infection":  ["wbc", "lactate", "procalcitonin"],
    "metabolic":  ["glucose", "calcium", "phosphorus", "magnesium"],
}

_SPEC_ICD: Dict[str, List[str]] = {
    "cardiology":         ["I", "41", "42", "414"],
    "nephrology":         ["N18", "N17", "585", "584"],
    "pulmonology":        ["J", "496", "491", "J96"],
    "neurology":          ["G", "430", "431", "433"],
    "oncology":           ["C", "140", "172", "174"],
    "infectious_disease": ["A", "B", "038"],
    "endocrinology":      ["E", "250", "E11"],
    "gastroenterology":   ["K", "520", "K74"],
    "hematology":         ["D", "280"],
    "rheumatology":       ["M", "710", "714"],
    "psychiatry":         ["F", "296", "300"],
}


def _specs_from_diagnoses(diagnoses: List[Dict]) -> Set[str]:
    found: Set[str] = set()
    for dx in diagnoses:
        code = str(dx.get("icd_code", ""))
        for spec, prefixes in _SPEC_ICD.items():
            if any(code.startswith(p) for p in prefixes):
                found.add(spec)
    return found


def _grade_step_2(t4, episode: Dict) -> Tuple[float, Dict]:
    lab_flags  = episode.get("lab_flags") or []
    diagnoses  = episode.get("diagnoses") or []

    abnormal = {
        str(lf.get("label", "")).lower()
        for lf in lab_flags
        if str(lf.get("flag", "")).lower() in ("abnormal", "critical", "high", "low")
    }
    expected_cats: Set[str] = {
        cat
        for cat, kws in _LAB_CATS.items()
        if any(kw in lbl for lbl in abnormal for kw in kws)
    }

    priority_labs = [l.lower() for l in (t4.priority_labs or [])]
    if expected_cats:
        hits = sum(
            1 for cat in expected_cats
            if any(kw in lab for kw in _LAB_CATS[cat] for lab in priority_labs)
        )
        lab_score = hits / len(expected_cats)
    else:
        lab_score = 1.0 if not priority_labs else 0.8

    expected_specs = _specs_from_diagnoses(diagnoses)
    consults = [
        c.lower().replace(" ", "_").replace("-", "_")
        for c in (t4.priority_consults or [])
    ]
    if expected_specs:
        spec_hits = sum(
            1 for sp in expected_specs
            if any(sp in c or c in sp for c in consults)
        )
        spec_score = min(1.0, spec_hits / max(1, len(expected_specs)))
    else:
        spec_score = 0.8

    combined = lab_score * 0.5 + spec_score * 0.5
    return round(combined * _STEP_MAX[2], 4), {
        "step2_lab_score":        round(lab_score, 4),
        "step2_spec_score":       round(spec_score, 4),
        "expected_lab_cats":      sorted(expected_cats),
        "expected_specialties":   sorted(expected_specs),
    }


# ─── Step 3 — Interventions ───────────────────────────────────────────────────

def _grade_step_3(t4, episode: Dict) -> Tuple[float, Dict]:
    icu_proc = episode.get("icu_procedure_summary") or {}
    fb       = episode.get("fluid_balance") or {}
    ivs_text = " ".join(t4.interventions or []).lower()

    needed: List[Tuple[str, List[str]]] = []
    if float(icu_proc.get("ventilation_hours", 0) or 0) > 0:
        needed.append(("ventilation", ["intubat", "mechanical ventil", "vent"]))
    if icu_proc.get("has_arterial_line"):
        needed.append(("arterial_line", ["arterial line", "a-line", "radial art"]))
    if icu_proc.get("has_dialysis"):
        needed.append(("dialysis", ["dialysis", "crrt", "hemodialysis", "cvvh"]))
    if fb.get("fluid_overloaded"):
        needed.append(("fluid_management", ["fluid bolus", "volume", "resuscitat", "iv fluid"]))

    if not needed:
        return round(0.10, 4), {"step3_intervention_score": 1.0, "needed": []}

    hits  = sum(1 for _n, kws in needed if any(kw in ivs_text for kw in kws))
    score = hits / len(needed)
    return round(score * _STEP_MAX[3], 4), {
        "step3_intervention_score": round(score, 4),
        "needed": [n for n, _ in needed],
        "hits": hits,
    }


# ─── Step 4 — High-Risk Medications ──────────────────────────────────────────

_ANTICOAG   = {"heparin", "warfarin", "enoxaparin", "lovenox", "apixaban", "rivaroxaban", "dabigatran"}
_VASOPRESS  = {"norepinephrine", "epinephrine", "dopamine", "vasopressin", "phenylephrine", "dobutamine"}
_SEDATIVES  = {"propofol", "midazolam", "fentanyl", "lorazepam", "ketamine", "dexmedetomidine"}
_HR_DRUGS   = _ANTICOAG | _VASOPRESS | _SEDATIVES


def _grade_step_4(t4, episode: Dict) -> Tuple[float, Dict]:
    all_drug_names: Set[str] = set()
    for m in (episode.get("medications") or []):
        all_drug_names.add(str(m.get("drug", "")).lower())
    for e in (episode.get("emar_summary") or []):
        all_drug_names.add(str(e.get("medication", "")).lower())

    present_hr: Set[str] = {
        hr for d in all_drug_names for hr in _HR_DRUGS
        if hr in d or d.startswith(hr[:6])
    }

    flagged = [f.lower() for f in (t4.high_risk_medications or [])]

    if not present_hr:
        score = 1.0 if not flagged else 0.6
        return round(score * _STEP_MAX[4], 4), {"step4_hr_med_score": score, "present_hr_meds": []}

    tp = sum(1 for hr in present_hr if any(hr in f or f.startswith(hr[:5]) for f in flagged))
    fp = sum(1 for f in flagged if not any(hr in f or f.startswith(hr[:5]) for hr in present_hr))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / len(present_hr)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return round(f1 * _STEP_MAX[4], 4), {
        "step4_hr_med_score": round(f1, 4),
        "present_hr_meds":    sorted(present_hr),
        "tp": tp, "fp": fp,
    }


# ─── Step 5 — Antibiotic Plan ─────────────────────────────────────────────────

_RESISTANT_ORGS = {"mrsa", "vre", "esbl", "kpc", "cre", "mdr"}


def _grade_step_5(t4, episode: Dict) -> Tuple[float, Dict]:
    micro    = episode.get("microbiology") or []
    organisms = [str(m.get("organism", "")).lower() for m in micro if m.get("organism")]
    sensitive = [
        str(s).lower() for m in micro for s in (m.get("sensitive_to") or [])
    ]

    strategy    = (t4.antibiotic_strategy or "").strip().lower()
    antibiotics = [a.lower() for a in (t4.antibiotics or [])]

    iso_bonus = 0.02 if any(
        any(r in org for r in _RESISTANT_ORGS) for org in organisms
    ) else 0.0

    if not organisms:
        score = 1.0 if strategy in ("none", "no antibiotics", "prophylaxis") else (
            0.4 if strategy in ("broad", "empiric") else 0.2
        )
    elif strategy in ("targeted", "culture-directed"):
        if sensitive and antibiotics:
            match = sum(1 for a in antibiotics if any(s in a or a in s for s in sensitive))
            score = min(1.0, 0.5 + 0.5 * match / max(1, len(sensitive)))
        else:
            score = 0.5
    elif strategy in ("broad", "empiric"):
        score = 0.3
    elif strategy in ("none", "no antibiotics"):
        score = 0.0
    else:
        score = 0.2

    final = min(1.0, score + iso_bonus)
    return round(final * _STEP_MAX[5], 4), {
        "step5_abx_score":   round(final, 4),
        "strategy":          strategy,
        "isolation_bonus":   iso_bonus,
        "organisms_found":   organisms[:5],
    }


# ─── Step 6 — Fluid Strategy ──────────────────────────────────────────────────

def _grade_step_6(t4, episode: Dict) -> Tuple[float, Dict]:
    fb          = episode.get("fluid_balance") or {}
    overloaded  = bool(fb.get("fluid_overloaded", False))
    oliguria    = bool(fb.get("oliguria", False))
    net_balance = float(fb.get("net_balance_ml", 0) or 0)
    strategy    = (t4.fluid_strategy or "").strip().lower().replace("-", "_").replace(" ", "_")

    if overloaded:
        correct = "restrict_diuresis"
    elif oliguria and net_balance < 0:
        correct = "aggressive_resuscitation"
    else:
        correct = "maintain"

    if strategy == correct:
        score = 1.0
    elif correct in ("restrict_diuresis", "aggressive_resuscitation") and strategy == "maintain":
        score = 0.4
    else:
        score = 0.0

    return round(score * _STEP_MAX[6], 4), {
        "step6_fluid_score":  round(score, 4),
        "correct_strategy":   correct,
        "predicted_strategy": strategy,
    }


# ─── Step 7 — ICU Readiness ───────────────────────────────────────────────────

_BARRIER_KW: Dict[str, List[str]] = {
    "vent":        ["vent", "intubat", "trach", "mechanical"],
    "hemodynamic": ["pressor", "vasopress", "norepinephrine", "hemodynamic"],
    "renal":       ["dialysis", "crrt", "renal", "kidney"],
    "mobility":    ["mobility", "physical therapy", "decondit", "ambul"],
}


def _grade_step_7(t4, episode: Dict) -> Tuple[float, Dict]:
    icu_proc   = episode.get("icu_procedure_summary") or {}
    los_days   = float(episode.get("hospital_los_days", 0) or 0)
    traj       = episode.get("care_trajectory") or []
    fb         = episode.get("fluid_balance") or {}

    vent_hours = float(icu_proc.get("ventilation_hours", 0) or 0)
    last_traj  = str(traj[-1]).lower() if traj else ""
    on_vent    = vent_hours > 0
    heading_out = any(kw in last_traj for kw in ("stepdown", "floor", "ward", "step-down"))

    true_ready  = los_days > 5 and not on_vent and heading_out
    pred_ready  = t4.ready_for_stepdown

    if pred_ready is None:
        readiness_score = 0.0
    elif pred_ready == true_ready:
        readiness_score = 1.0
    else:
        readiness_score = 0.2

    actual_barriers: List[str] = []
    if on_vent:
        actual_barriers.append("vent")
    if fb.get("oliguria"):
        actual_barriers.append("renal")
    if icu_proc.get("has_dialysis"):
        actual_barriers.append("renal")

    barriers_text = " ".join(t4.barriers or []).lower()
    if actual_barriers:
        b_hits = sum(
            1 for b in actual_barriers
            if any(kw in barriers_text for kw in _BARRIER_KW.get(b, []))
        )
        barrier_score = b_hits / len(actual_barriers)
    else:
        barrier_score = 1.0

    combined = readiness_score * 0.6 + barrier_score * 0.4
    return round(combined * _STEP_MAX[7], 4), {
        "step7_readiness_score": round(combined, 4),
        "true_ready":            true_ready,
        "pred_ready":            pred_ready,
        "barrier_score":         round(barrier_score, 4),
    }


# ─── Step 8 — Disposition + LOS Estimate ─────────────────────────────────────

def _grade_step_8(t4, episode: Dict) -> Tuple[float, Dict]:
    pred_dispo = (t4.predicted_disposition or "").strip().lower().replace(" ", "_")
    true_loc   = str(episode.get("discharge_location", "")).strip()
    true_canon = normalize_mimic_location(true_loc)

    if pred_dispo == true_canon:
        dispo_score = 0.06
    elif BROAD_GROUP.get(pred_dispo) == BROAD_GROUP.get(true_canon):
        dispo_score = 0.03
    elif true_canon in _ADJACENT.get(pred_dispo, []):
        dispo_score = 0.015
    else:
        dispo_score = 0.0

    los_actual = float(episode.get("hospital_los_days", 0) or 0)
    los_pred   = t4.los_remaining_days
    if los_pred is not None:
        diff = abs(float(los_pred) - los_actual)
        los_score = 0.04 if diff <= 2 else (0.02 if diff <= 5 else 0.0)
    else:
        los_score = 0.0

    return round(dispo_score + los_score, 4), {
        "step8_dispo_score":  dispo_score,
        "step8_los_score":    los_score,
        "predicted_dispo":    pred_dispo,
        "true_dispo":         true_canon,
        "predicted_los":      los_pred,
        "actual_los":         round(los_actual, 2),
    }


# ─── Step 9 — Medication Reconciliation ──────────────────────────────────────

def _med_stem(drug: str) -> str:
    tokens = [t for t in drug.lower().split() if len(t) >= 4]
    return tokens[0] if tokens else drug.lower()[:6]


def _grade_step_9(t4, episode: Dict) -> Tuple[float, Dict]:
    emar = episode.get("emar_summary") or []
    active_meds = [
        _med_stem(e.get("medication", ""))
        for e in emar
        if e.get("active_at_discharge") and e.get("medication")
    ]
    if not active_meds:
        active_meds = [_med_stem(m) for m in (episode.get("pharmacy_active") or []) if m]

    recommended = [_med_stem(m) for m in (t4.medications_to_continue or []) if m]

    if not active_meds:
        return round(0.07, 4), {"step9_med_recon_score": 0.7, "active_count": 0}

    active_set = set(active_meds)
    tp = sum(1 for r in recommended if any(
        r.startswith(a[:4]) or a.startswith(r[:4]) for a in active_set
    ))
    fp = max(0, len(recommended) - tp)
    fn = max(0, len(active_set) - tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return round(f1 * _STEP_MAX[9], 4), {
        "step9_med_recon_score": round(f1, 4),
        "active_count":          len(active_set),
        "recommended_count":     len(recommended),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ─── Step 10 — Final Note + Consistency Bonuses ───────────────────────────────

_NOTE_GRADER = _NoteGrader()

_DISPO_KW: Dict[str, List[str]] = {
    "home":               ["home", "self-care", "returned home"],
    "home_with_services": ["home health", "home with", "vna"],
    "snf":                ["skilled nursing", "snf", "nursing facility"],
    "rehab":              ["rehab", "rehabilitation"],
    "hospice":            ["hospice", "comfort care", "palliative"],
    "expired":            ["expired", "deceased", "died"],
    "ama":                ["against medical advice", "ama"],
    "other":              ["transferred", "transfer"],
}


def _grade_step_10(t4, episode: Dict, shaping_log: Dict) -> Tuple[float, Dict]:
    class _FTask3:
        discharge_note = t4.final_note or ""

    class _FAction:
        task3 = _FTask3()

    base_reward, note_partial, _, _ = _NOTE_GRADER.grade(_FAction(), episode)

    # Shaping average (normalised per-step scores 0–1)
    step_scores = [
        shaping_log[f"step{i}_reward"] / _STEP_MAX[i]
        for i in range(1, 10)
        if f"step{i}_reward" in shaping_log and _STEP_MAX.get(i, 0) > 0
    ]
    shaping_avg = sum(step_scores) / len(step_scores) if step_scores else 0.0

    raw = base_reward * 0.60 + shaping_avg * 0.40

    # Consistency bonus: final note disposition matches step 8 prediction
    note_lower        = (t4.final_note or "").lower()
    step8_dispo       = shaping_log.get("step8_predicted_dispo", "")
    consistency_bonus = 0.0
    if step8_dispo:
        kws = _DISPO_KW.get(step8_dispo, [])
        if kws and any(kw in note_lower for kw in kws):
            consistency_bonus = 0.10

    # Trajectory bonus: ≥50% specialty overlap with step 2 consults
    step2_specs      = set(shaping_log.get("step2_expected_specialties", []))
    trajectory_bonus = 0.0
    if step2_specs:
        hits = sum(1 for sp in step2_specs if sp.replace("_", " ") in note_lower or sp in note_lower)
        if hits / len(step2_specs) >= 0.5:
            trajectory_bonus = 0.05

    revision_cost = float(shaping_log.get("revisions_used", 0)) * 0.02
    final = round(max(0.0, min(1.0, raw + consistency_bonus + trajectory_bonus - revision_cost)), 4)

    partial = {
        **{f"note_{k}": v for k, v in note_partial.items()},
        "shaping_average":   round(shaping_avg, 4),
        "raw_combined":      round(raw, 4),
        "consistency_bonus": consistency_bonus,
        "trajectory_bonus":  trajectory_bonus,
        "revision_cost":     revision_cost,
    }
    return final, partial


# ─── Revision proxy ───────────────────────────────────────────────────────────

class _RevisionProxy:
    """Wraps a revision dict so step graders can treat it like a Task4Action."""

    def __init__(self, revision: Dict[str, Any]) -> None:
        for k, v in revision.items():
            setattr(self, k, v)

    def __getattr__(self, name: str):
        return None


# ─── Main grader ──────────────────────────────────────────────────────────────

class Task4Grader:
    """
    Dispatcher for the 10-step workflow.

    Calling convention (env.py passes extra args):
        reward, partial, done, info = grader.grade(action, episode, step_num, shaping_log)

    Steps 1-9: reward=0 (sparse); shaping scores stored in shaping_log.
    Step 10:   returns composite final reward; done=True.
    """

    _DISPATCH = {
        1: _grade_step_1,
        2: _grade_step_2,
        3: _grade_step_3,
        4: _grade_step_4,
        5: _grade_step_5,
        6: _grade_step_6,
        7: _grade_step_7,
        8: _grade_step_8,
        9: _grade_step_9,
    }
    MAX_STEPS = 10

    def grade(
        self,
        action: Any,
        episode: Dict[str, Any],
        step_num: int,
        shaping_log: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], bool, Dict[str, Any]]:

        if action.task4 is None:
            return 0.0, {"error_no_task4": -1.0}, True, {"error": "Action.task4 is missing"}

        t4 = action.task4

        # ── Revision mechanism ────────────────────────────────────────────────
        if t4.revise_step is not None and t4.revision is not None:
            used = shaping_log.get("revisions_used", 0)
            if used < 2 and 1 <= int(t4.revise_step) <= 9:
                shaping_log["revisions_used"] = used + 1
                fn = self._DISPATCH.get(int(t4.revise_step))
                if fn:
                    proxy = _RevisionProxy(t4.revision)
                    rev_reward, rev_partial = fn(proxy, episode)
                    for k, v in rev_partial.items():
                        shaping_log[k] = v
                    shaping_log[f"step{t4.revise_step}_reward"] = rev_reward
                    shaping_log[f"step{t4.revise_step}_revised"] = True
            return 0.0, {"revision_accepted": 1.0}, False, {
                "revise_step": t4.revise_step,
                "revisions_used": shaping_log.get("revisions_used", 0),
            }

        # ── Step 10: final note ───────────────────────────────────────────────
        if step_num >= self.MAX_STEPS:
            final_reward, partial = _grade_step_10(t4, episode, shaping_log)
            return final_reward, partial, True, {
                "workflow_complete": True,
                "steps_graded": sum(1 for i in range(1, 10) if f"step{i}_reward" in shaping_log),
            }

        # ── Steps 1-9 ─────────────────────────────────────────────────────────
        fn = self._DISPATCH.get(step_num)
        if fn is None:
            return 0.0, {}, False, {"error": f"no grader for step {step_num}"}

        step_reward, step_partial = fn(t4, episode)

        # Persist keys needed for step 10 bonuses
        for k, v in step_partial.items():
            shaping_log[k] = v
        shaping_log[f"step{step_num}_reward"] = step_reward

        if step_num == 2:
            shaping_log["step2_expected_specialties"] = step_partial.get("expected_specialties", [])
        if step_num == 8:
            shaping_log["step8_predicted_dispo"] = step_partial.get("predicted_dispo", "")

        return 0.0, step_partial, False, {
            "step_num":        step_num,
            "step_raw_reward": step_reward,
        }
