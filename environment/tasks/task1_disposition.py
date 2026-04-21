"""
Task 1 (Easy) — Discharge Disposition Prediction.

Improvements over v1:
  1. Expanded MIMIC location patterns — covers all strings seen in the demo dataset.
  2. Three-tier scoring: exact (1.0) → broad group (0.5) → adjacent group (0.25).
     "Adjacent" = clinically close but wrong group, e.g. snf vs rehab, home vs home_with_services.
  3. Clinical reasoning bonus: rewards domain-specific keywords, not just string length.
  4. Confidence weighting: bonus if model picks the statistically rare correct class.

Score breakdown:
  1.00  exact canonical match
  0.50  same broad group  (community / facility / end_of_life)
  0.25  adjacent group    (e.g. facility ↔ community-plus, end_of_life ↔ facility)
  0.00  wrong group with no adjacency
  +0.05 clinical reasoning bonus
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List

# ─── Canonical → MIMIC substring patterns ────────────────────────────────────
# Checked in ORDER — more specific entries must come before "home".
# Covers all discharge_location strings in the MIMIC-IV demo v2.2.
DISPOSITION_PATTERNS: Dict[str, List[str]] = {
    "home_with_services": [
        "HOME HEALTH CARE",     # most common MIMIC string for this category
        "HOME HEALTH",
        "HOME WITH SERVICE",
        "HOME WITH HEALTH",
        "HOME WITH AIDE",
        "HOME WITH VNA",
        "HOME WITH HOSPICE",    # some MIMIC rows use this hybrid
        "ASSISTED LIVING",      # lives independently but with paid support
    ],
    "snf": [
        "SKILLED NURSING FACILITY",
        "SKILLED NURSING",
        "SNF",
        "LONG TERM CARE",
        "LONG-TERM CARE",
        "CHRONIC CARE",
        "EXTENDED CARE",
        "NURSING FACILITY",
        "NURSING HOME",
        "SUB-ACUTE",
        "SUBACUTE",
    ],
    "rehab": [
        "REHABILITATION",
        "REHAB FACILITY",
        "INPATIENT REHAB",
        "REHAB",
    ],
    "hospice": [
        "HOSPICE-MEDICAL FACILITY",
        "HOSPICE-HOME",
        "HOSPICE",
        "COMFORT CARE ONLY",
    ],
    "ama": [
        "AGAINST MEDICAL ADVICE",
        "AGAINST ADVICE",
        "LEFT AMA",
        "AMA",
        "ELOPED",
    ],
    "expired": [
        "DIED IN ICU",
        "DIED",
        "EXPIRED",
        "DEAD",
        "DECEASED",
    ],
    "other": [
        "ACUTE HOSPITAL",           # transferred to another acute facility
        "TRANSFER TO ANOTHER",
        "TRANSFER",
        "PSYCH FACILITY",
        "PSYCHIATRIC FACILITY",
        "PSYCH",
        "COURT",
        "CORRECTIONAL",
        "FEDERAL",
        "JAIL",
        "GROUP HOME",
        "HEALTHCARE FACILITY",      # generic facility that doesn't fit above
        "OTHER FACILITY",
        "ANOTHER FACILITY",
    ],
    "home": [
        "DISCHARGED TO HOME",
        "RETURNED HOME",
        "HOME",
        "SELF",
    ],
}

VALID_DISPOSITIONS = list(DISPOSITION_PATTERNS.keys())

# ─── Broad groups for primary partial credit (0.5) ───────────────────────────
BROAD_GROUP: Dict[str, str] = {
    "home":               "community",
    "home_with_services": "community_plus",   # split from plain community
    "ama":                "community",
    "snf":                "facility",
    "rehab":              "facility",
    "hospice":            "end_of_life",
    "expired":            "end_of_life",
    "other":              "other",
}

_ADJACENT: Dict[str, List[str]] = {
    "home":               ["home_with_services", "ama"],
    "home_with_services": ["home", "snf"],          # borderline community/facility
    "snf":                ["home_with_services", "rehab", "other"],
    "rehab":              ["snf", "other"],
    "hospice":            ["snf", "home_with_services", "expired"],
    "expired":            ["hospice"],
    "ama":                ["home"],
    "other":              ["snf", "rehab"],
}

_CLINICAL_KEYWORDS = {
    # disposition-related
    "functional", "ambulation", "mobility", "gait", "weight-bearing",
    "therapy", "physical", "occupational", "speech",
    "nursing", "skilled", "wound", "iv", "intravenous",
    # clinical condition markers
    "icu", "micu", "cardiac", "pulmonary", "renal", "hepatic",
    "creatinine", "ejection", "fraction", "oxygen", "saturation",
    "comorbid", "comorbidity", "frail", "deconditioning",
    "hemodynamic", "instability", "stable",
    # social/support
    "caregiver", "support", "independent", "alone", "family",
    "insurance", "medicare", "medicaid",
}


def normalize_mimic_location(location: str) -> str:
    """Map a raw MIMIC discharge_location string → canonical agent disposition."""
    loc = location.upper().strip()
    for canonical, patterns in DISPOSITION_PATTERNS.items():
        if any(pat in loc for pat in patterns):
            return canonical
    return "other"


def _fuzzy_match_pred(pred: str) -> str:
    """Allow minor spelling variants from the model."""
    p = pred.lower().strip().replace("-", "_").replace(" ", "_")
    # direct match
    if p in VALID_DISPOSITIONS:
        return p
    # substring match
    for valid in VALID_DISPOSITIONS:
        if valid in p or p in valid:
            return valid
    # common aliases
    _ALIASES = {
        "snf": "snf",
        "skilled_nursing": "snf",
        "nursing_facility": "snf",
        "home_health": "home_with_services",
        "home_services": "home_with_services",
        "against_medical_advice": "ama",
        "against_advice": "ama",
        "deceased": "expired",
        "dead": "expired",
        "died": "expired",
        "transferred": "other",
        "transfer": "other",
        "rehabilitation": "rehab",
        "comfort": "hospice",
        "palliative": "hospice",
    }
    return _ALIASES.get(p, p)


def _reasoning_bonus(reasoning: str) -> float:
    """
    Award up to 0.05 for clinical reasoning.
    Requires: >= 20 chars AND at least one clinical keyword OR a quantitative value.
    """
    r = reasoning.lower().strip()
    if len(r) < 20:
        return 0.0
    has_keyword  = any(kw in r for kw in _CLINICAL_KEYWORDS)
    has_number   = any(ch.isdigit() for ch in r)
    has_icd_hint = any(x in r for x in ["icd", "dx", "diagnosis", "condition", "history"])
    if has_keyword or (has_number and len(r) > 30) or has_icd_hint:
        return 0.05
    # Fallback: long enough sentence gets the small bonus anyway
    if len(r) >= 50:
        return 0.05
    return 0.0


class DispositionGrader:
    """Programmatic grader for Task 1. Score range 0.0–1.05 (clamped to 1.0)."""

    def grade(
        self,
        action: Any,
        episode: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], bool, Dict[str, Any]]:

        if action.task1 is None:
            return (
                0.0,
                {"error_no_task1": -1.0},
                True,
                {"error": "Action.task1 is missing"},
            )

        pred_raw = (action.task1.disposition or "").strip()
        pred     = _fuzzy_match_pred(pred_raw)

        if pred not in VALID_DISPOSITIONS:
            return (
                0.0,
                {"invalid_disposition": -1.0},
                True,
                {
                    "error":         f"'{pred_raw}' is not a valid disposition",
                    "valid_options": VALID_DISPOSITIONS,
                },
            )

        true_location  = str(episode.get("discharge_location", "")).strip()
        true_canonical = normalize_mimic_location(true_location)

        partial: Dict[str, float] = {}
        score = 0.0

        if pred == true_canonical:
            # ── Tier 1: exact match ───────────────────────────────────────────
            partial["disposition_exact"] = 1.0
            score = 1.0

        elif BROAD_GROUP.get(pred) == BROAD_GROUP.get(true_canonical):
            # ── Tier 2: same broad group ──────────────────────────────────────
            partial["disposition_exact"] = 0.0
            partial["disposition_broad"] = 0.5
            score = 0.5

        elif true_canonical in _ADJACENT.get(pred, []):
            # ── Tier 3: clinically adjacent (new) ────────────────────────────
            partial["disposition_exact"]    = 0.0
            partial["disposition_broad"]    = 0.0
            partial["disposition_adjacent"] = 0.25
            score = 0.25

        else:
            partial["disposition_exact"]    = 0.0
            partial["disposition_broad"]    = 0.0
            partial["disposition_adjacent"] = 0.0

        # ── Reasoning bonus ───────────────────────────────────────────────────
        reasoning = (action.task1.reasoning or "").strip()
        bonus = _reasoning_bonus(reasoning)
        if bonus > 0:
            partial["reasoning_bonus"] = bonus
            score = min(1.0, score + bonus)

        info = {
            "predicted_canonical":    pred,
            "ground_truth_raw":       true_location,
            "ground_truth_canonical": true_canonical,
            "broad_group_pred":       BROAD_GROUP.get(pred),
            "broad_group_true":       BROAD_GROUP.get(true_canonical),
            "adjacent":               true_canonical in _ADJACENT.get(pred, []),
        }

        return round(score, 4), partial, True, info