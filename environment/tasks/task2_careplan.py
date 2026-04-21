"""
Task 2 (Medium) — Discharge Care Plan Recommendation. v3.

v3 improvements over v2:
  1. Hallucination check uses BOTH prescriptions AND emar_drug_set (actual administered
     drugs from eMAR). A drug is only hallucinated if absent from both sources.
  2. Ghost specialty penalty: recommended specialties without any supporting ICD code
     or critical lab flag are penalised (0.05 each, max 0.10).
  3. max_steps increased to 4 with step_efficiency_discount applied by env.py.

Score = 0.35 × specialty_f1
      + 0.25 × medication_f1
      + 0.25 × instruction_quality
      + 0.15 × discontinue_accuracy
      − hallucination_penalty   (max 0.10)
      − ghost_specialty_penalty (max 0.10)
"""

from __future__ import annotations
import re
from typing import Tuple, Dict, Any, List, Set


# ─── ICD → specialty mappings ─────────────────────────────────────────────────

ICD10_SPECIALTY: Dict[str, List[str]] = {
    "A": ["infectious disease"], "B": ["infectious disease"],
    "C": ["oncology"],           "D": ["hematology", "oncology"],
    "E": ["endocrinology"],      "F": ["psychiatry"],
    "G": ["neurology"],          "H": ["ophthalmology"],
    "I": ["cardiology"],         "J": ["pulmonology"],
    "K": ["gastroenterology"],   "L": ["dermatology"],
    "M": ["rheumatology"],       "N": ["nephrology"],
    "O": ["obstetrics"],         "Q": ["genetics"],
    "S": ["trauma surgery"],     "T": ["toxicology"],
}

ICD9_SPECIALTY: Dict[str, List[str]] = {
    "001": ["infectious disease"], "140": ["oncology"],
    "240": ["endocrinology"],      "290": ["psychiatry"],
    "320": ["neurology"],          "390": ["cardiology"],
    "460": ["pulmonology"],        "520": ["gastroenterology"],
    "580": ["nephrology"],         "630": ["obstetrics"],
    "680": ["dermatology"],        "710": ["rheumatology"],
    "800": ["trauma surgery"],
}

_HCPCS_SPECIALTY: Dict[str, str] = {
    "cardiovascular": "cardiology",
    "cardiac":        "cardiology",
    "echocardiograph": "cardiology",
    "catheterization": "cardiology",
    "pulmonary":      "pulmonology",
    "respiratory":    "pulmonology",
    "neuro":          "neurology",
    "dialysis":       "nephrology",
    "renal":          "nephrology",
    "oncol":          "oncology",
    "chemo":          "oncology",
    "endoscop":       "gastroenterology",
    "colono":         "gastroenterology",
    "orthop":         "orthopedics",
    "joint":          "orthopedics",
    "ophthalm":       "ophthalmology",
    "psych":          "psychiatry",
    "rehab":          "physical therapy",
    "physical therapy": "physical therapy",
}

_ORGANISM_SPECIALTY: Dict[str, str] = {
    "staphylococcus": "infectious disease",
    "streptococcus":  "infectious disease",
    "klebsiella":     "infectious disease",
    "pseudomonas":    "infectious disease",
    "escherichia":    "infectious disease",
    "candida":        "infectious disease",
    "aspergillus":    "infectious disease",
    "enterococcus":   "infectious disease",
    "mrsa":           "infectious disease",
    "vre":            "infectious disease",
    "clostridioid":   "infectious disease",
    "clostridium":    "infectious disease",
}

# ── Specialty → ICD prefixes for ghost-specialty detection ────────────────────
# ICD-10 single-letter prefixes OR ICD-9 numeric prefixes
_SPECIALTY_ICD_PREFIXES: Dict[str, List[str]] = {
    "cardiology":         ["I", "42", "41", "V45"],
    "nephrology":         ["N18", "N17", "N19", "585", "584", "403"],
    "pulmonology":        ["J", "496", "491", "518"],
    "neurology":          ["G", "430", "431", "433", "434"],
    "oncology":           ["C", "140", "172", "174"],
    "infectious disease": ["A", "B", "038", "0"],
    "endocrinology":      ["E", "240", "250"],
    "gastroenterology":   ["K", "520", "550", "560", "570"],
    "hematology":         ["D", "280"],
    "rheumatology":       ["M", "710", "714", "720"],
    "psychiatry":         ["F", "290", "296", "300"],
    "nephrology":         ["N", "580", "585"],
}


def _get_icd9_specialty(code: str) -> List[str]:
    num = int(re.sub(r"\D", "", code)[:3] or "0") if re.search(r"\d", code) else 0
    for threshold, specs in sorted(
        ((k, v) for k, v in ICD9_SPECIALTY.items() if k.isdigit()),
        key=lambda x: int(x[0]), reverse=True,
    ):
        if num >= int(threshold):
            return specs
    return []


def get_expected_specialties(
    diagnoses:    List[Dict],
    microbiology: List[Dict] = None,
    hcpcs:        List[str]  = None,
) -> Set[str]:
    specialties: Set[str] = {"primary care"}

    for dx in diagnoses[:5]:
        code    = str(dx.get("icd_code", "")).strip().upper()
        version = int(dx.get("icd_version", 10) or 10)
        if not code:
            continue
        if version == 10:
            for spec in ICD10_SPECIALTY.get(code[0], []):
                specialties.add(spec.lower())
        else:
            for spec in _get_icd9_specialty(code):
                specialties.add(spec.lower())

    if microbiology:
        for result in microbiology:
            org = str(result.get("organism", "")).lower()
            for key, spec in _ORGANISM_SPECIALTY.items():
                if key in org:
                    specialties.add(spec)
                    break

    if hcpcs:
        for desc in hcpcs:
            dl = desc.lower()
            for key, spec in _HCPCS_SPECIALTY.items():
                if key in dl:
                    specialties.add(spec)
                    break

    return specialties


# ─── Specialty F1 ─────────────────────────────────────────────────────────────

def _specialty_f1(
    predicted: List[str], expected: Set[str]
) -> Tuple[float, float, float]:
    if not expected:
        return 1.0, 1.0, 1.0

    pred_lower = [s.lower().strip() for s in predicted if s]
    if not pred_lower:
        return 0.0, 0.0, 0.0

    pred_lower = pred_lower[:8]

    def _fuzzy_match(a: str, b: str) -> bool:
        a_words = [w for w in a.split() if len(w) > 3]
        return a in b or b in a or any(w in b for w in a_words)

    matched_expected = {
        exp for exp in expected
        if any(_fuzzy_match(exp, pred) for pred in pred_lower)
    }
    matched_pred = {
        pred for pred in pred_lower
        if any(_fuzzy_match(exp, pred) for exp in expected)
    }

    recall    = len(matched_expected) / len(expected)
    precision = len(matched_pred)     / len(pred_lower)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return round(recall, 4), round(precision, 4), round(f1, 4)


# ─── Ghost specialty penalty ──────────────────────────────────────────────────

def _ghost_specialty_penalty(
    predicted:    List[str],
    diagnoses:    List[Dict],
    lab_flags:    List[Dict],
) -> Tuple[float, List[str]]:
    """
    For each predicted specialty, check for a supporting ICD code prefix.
    Falls back to any existing lab flag as general support.
    Returns (penalty, list_of_ghost_specialties).
    """
    has_any_lab = len(lab_flags) > 0

    # Collect all ICD codes from episode
    icd_codes = [str(d.get("icd_code", "")).strip().upper() for d in diagnoses]

    def _has_icd_support(spec_lower: str) -> bool:
        prefixes = _SPECIALTY_ICD_PREFIXES.get(spec_lower, [])
        for prefix in prefixes:
            for code in icd_codes:
                if code.startswith(prefix):
                    return True
        return False

    ghost_specialties = []
    for spec in predicted:
        spec_lower = spec.lower().strip()
        if spec_lower in ("primary care", "general medicine", "hospitalist"):
            continue
        if _has_icd_support(spec_lower):
            continue
        # Fallback: any abnormal lab provides broad support for common medical specialties
        if has_any_lab:
            continue
        ghost_specialties.append(spec)

    penalty = min(0.10, len(ghost_specialties) * 0.05)
    return round(penalty, 4), ghost_specialties


# ─── Medication F1 + hallucination (v3: checks emar_drug_set) ─────────────────

def _med_stem(drug: str) -> str:
    tokens = [t for t in str(drug).lower().split() if len(t) >= 4]
    return tokens[0] if tokens else str(drug).lower()[:4]


def _medication_f1_and_halluc(
    recommended_continue:    List[str],
    recommended_discontinue: List[str],
    episode_meds:            List[Dict],
    pharmacy_active:         List[str] = None,
    emar_drug_set:           Set[str]  = None,
) -> Tuple[float, float]:
    """
    v3: drug is NOT a hallucination if it appears in prescriptions,
    pharmacy_active, OR emar_drug_set.
    """
    all_ep_drugs: List[str] = [m.get("drug", "") for m in episode_meds if m.get("drug")]
    if pharmacy_active:
        all_ep_drugs = list({*all_ep_drugs, *pharmacy_active})

    emar_stems: Set[str] = set()
    if emar_drug_set:
        emar_stems = {_med_stem(d) for d in emar_drug_set if d}

    # Need at least one drug source to score
    if not all_ep_drugs and not emar_stems:
        return 0.5, 0.0

    ep_stems   = {_med_stem(d) for d in all_ep_drugs}
    top5_stems = {_med_stem(d) for d in all_ep_drugs[:5]}

    all_known_stems = ep_stems | emar_stems

    pred_continue_stems = {_med_stem(d) for d in recommended_continue if d}

    def _stem_match(a: str, b: str) -> bool:
        return a.startswith(b[:4]) or b.startswith(a[:4])

    tp_continue = {
        s for s in pred_continue_stems
        if any(_stem_match(s, ep) for ep in all_known_stems)
    }
    recall = (
        len({ep for ep in top5_stems if any(_stem_match(ep, s) for s in pred_continue_stems)})
        / len(top5_stems)
    ) if top5_stems else 0
    all_pred = len(pred_continue_stems)
    precision = len(tp_continue) / all_pred if all_pred > 0 else (
        1.0 if not top5_stems else 0.0
    )
    f1 = (
        (2 * recall * precision / (recall + precision))
        if (recall + precision) > 0 else 0.0
    )

    all_recommended = [
        _med_stem(d)
        for d in (recommended_continue + recommended_discontinue) if d
    ]
    # Drug is hallucinated only if absent from ALL known drug sources
    hallucinated = [
        s for s in all_recommended
        if not any(_stem_match(s, ep) for ep in all_known_stems)
    ]
    halluc_rate = len(hallucinated) / len(all_recommended) if all_recommended else 0.0

    return round(f1, 4), round(halluc_rate, 4)


# ─── Instruction quality ─────────────────────────────────────────────────────

_INSTRUCTION_CATEGORIES = {
    "activity": {
        "triggers": ["activity", "exercise", "walk", "rest", "lift", "physical"],
        "quality":  ["daily", "week", "minute", "pound", "kg", "lb", "avoid", "limit",
                     "restrict", "moderate", "light", "30", "15", "20"],
    },
    "diet": {
        "triggers": ["diet", "sodium", "fluid", "eat", "drink", "food", "water",
                     "nutrition", "calorie", "carb", "protein", "fat"],
        "quality":  ["gram", "mg", "litre", "liter", "oz", "low", "restrict", "limit",
                     "avoid", "per day", "daily", "2g", "2000", "1500"],
    },
    "medication": {
        "triggers": ["medication", "medicine", "drug", "dose", "pill", "tablet", "take",
                     "prescription", "refill"],
        "quality":  ["daily", "twice", "morning", "evening", "mg", "do not stop",
                     "continue", "as prescribed", "with food", "without food"],
    },
    "follow_up": {
        "triggers": ["follow", "appointment", "return", "visit", "clinic", "schedule", "see"],
        "quality":  ["week", "day", "month", "within", "doctor", "physician", "specialist",
                     "primary care", "1 week", "2 week", "one week", "two week"],
    },
    "warnings": {
        "triggers": ["call", "emergency", "warning", "sign", "symptom", "seek", "if you"],
        "quality":  ["chest", "breath", "pain", "fever", "swelling", "weight", "dizziness",
                     "blood", "bleed", "vision", "confusion", "worse"],
    },
}

_FILLER_PHRASES = {
    "take all medications as prescribed",
    "follow up with your doctor",
    "follow up with your physician",
    "seek medical attention",
    "contact your doctor",
    "as directed",
}


def _instruction_quality(instructions: List[str]) -> float:
    if not instructions:
        return 0.0
    cleaned = []
    for instr in instructions:
        instr_lower = instr.lower().strip()
        if instr_lower in _FILLER_PHRASES or len(instr_lower.split()) < 4:
            continue
        cleaned.append(instr_lower)
    cleaned = cleaned[:10]

    category_scores = []
    for _cat, specs in _INSTRUCTION_CATEGORIES.items():
        matched = any(
            any(t in instr for t in specs["triggers"]) and
            any(q in instr for q in specs["quality"])
            for instr in cleaned
        )
        category_scores.append(1.0 if matched else 0.0)

    base         = sum(category_scores) / len(category_scores)
    unique_bonus = min(0.1, len(cleaned) / 50)
    return round(min(1.0, base + unique_bonus), 4)


# ─── Discontinue accuracy ─────────────────────────────────────────────────────

def _discontinue_accuracy(
    recommended_discontinue: List[str],
    episode_meds:            List[Dict],
    pharmacy_stopped:        List[str] = None,
) -> float:
    if not recommended_discontinue:
        return 0.5

    known_stopped: Set[str] = set()
    if pharmacy_stopped:
        known_stopped = {_med_stem(d) for d in pharmacy_stopped if d}

    ep_stems   = {_med_stem(m.get("drug", "")) for m in episode_meds if m.get("drug")}
    disc_stems = [_med_stem(d) for d in recommended_discontinue if d]

    if known_stopped:
        correct = [s for s in disc_stems if any(
            s.startswith(k[:4]) or k.startswith(s[:4]) for k in known_stopped
        )]
        if disc_stems:
            return round(len(correct) / len(disc_stems), 4)

    valid = [s for s in disc_stems if any(
        s.startswith(ep[:4]) or ep.startswith(s[:4]) for ep in ep_stems
    )]
    return round(len(valid) / len(disc_stems), 4) if disc_stems else 0.5


# ─── Grader class ─────────────────────────────────────────────────────────────

class CarePlanGrader:
    """v3 grader for Task 2."""

    def grade(
        self,
        action:  Any,
        episode: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], bool, Dict[str, Any]]:

        if action.task2 is None:
            return 0.0, {"error_no_task2": -1.0}, True, {"error": "Action.task2 is missing"}

        t2            = action.task2
        diagnoses     = episode.get("diagnoses",        [])
        medications   = episode.get("medications",      [])
        microbio      = episode.get("microbiology",     [])
        hcpcs         = episode.get("hcpcs_categories", [])
        pharm_act     = episode.get("pharmacy_active",  [])
        pharm_stop    = episode.get("pharmacy_stopped", [])
        emar_drug_set = episode.get("_emar_drug_set",   set())
        lab_flags     = episode.get("lab_flags",        [])

        expected_specs             = get_expected_specialties(diagnoses, microbio, hcpcs)
        recall, precision, spec_f1 = _specialty_f1(t2.follow_up_specialties, expected_specs)

        med_f1, halluc_rate = _medication_f1_and_halluc(
            t2.medications_to_continue,
            t2.medications_to_discontinue,
            medications,
            pharm_act,
            emar_drug_set,
        )
        instr_quality = _instruction_quality(t2.key_instructions)
        disc_acc      = _discontinue_accuracy(
            t2.medications_to_discontinue, medications, pharm_stop
        )

        halluc_penalty = round(min(0.10, halluc_rate * 0.10), 4)

        ghost_penalty, ghost_specs = _ghost_specialty_penalty(
            t2.follow_up_specialties, diagnoses, lab_flags
        )

        raw = (
            0.35 * spec_f1
            + 0.25 * med_f1
            + 0.25 * instr_quality
            + 0.15 * disc_acc
        )
        final = round(max(0.0, min(1.0, raw - halluc_penalty - ghost_penalty)), 4)

        partial = {
            "specialty_recall":       recall,
            "specialty_precision":    precision,
            "specialty_f1":           spec_f1,
            "medication_f1":          med_f1,
            "medication_halluc_rate": halluc_rate,
            "hallucination_penalty":  halluc_penalty,
            "ghost_specialty_penalty": ghost_penalty,
            "instruction_quality":    instr_quality,
            "discontinue_accuracy":   disc_acc,
        }
        info = {
            "expected_specialties":  sorted(expected_specs),
            "predicted_specialties": t2.follow_up_specialties,
            "ghost_specialties":     ghost_specs,
            "n_meds_in_episode":     len(medications),
            "n_emar_drugs":          len(emar_drug_set),
            "n_instructions_given":  len(t2.key_instructions),
            "microbio_organisms":    [m.get("organism") for m in microbio],
            "hcpcs_found":           hcpcs,
        }

        return final, partial, True, info
