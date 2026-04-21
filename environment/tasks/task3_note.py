"""
Task 3 (Hard) — Discharge Note Generation. v3.

v3 improvements over v2:
  1. Hallucination check uses BOTH prescriptions AND emar_drug_set — a drug is
     only hallucinated if absent from both sources.
  2. follow_up structure penalty: if discharge_orders.discharge_planning_finalized
     is True in the episode but the generated note omits "follow-up"/"follow up",
     apply 0.05 structure penalty.

Score (all components clamped to [0, 1] before weighting):
  0.30 × diagnosis_coverage        (contextual, anti-stuffing)
  0.20 × disposition_accuracy
  0.20 × medication_precision_recall  (F1)
  0.15 × los_accuracy
  0.10 × structure_score
  0.05 × information_density
  − hallucination_penalty          (subtracted after weighting, floor 0, max 0.15)
  − followup_structure_penalty     (0.05 if planning finalized but follow-up omitted)
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Tuple, Dict, Any, List, Set


# ─── Shared helpers ───────────────────────────────────────────────────────────

_STOPWORDS = {
    "with", "without", "unspecified", "other", "acute", "chronic", "type",
    "stage", "disease", "disorder", "condition", "history", "patient",
    "admission", "hospital", "including", "related", "associated",
    "secondary", "primary", "initial", "subsequent",
}


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _words(text: str) -> List[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


# ─── 1. Diagnosis coverage ────────────────────────────────────────────────────

def _diagnosis_coverage(note: str, diagnoses: List[Dict]) -> float:
    if not diagnoses:
        return 0.5

    top5 = [d for d in diagnoses if d.get("long_title")][:5]
    if not top5:
        return 0.5

    sents = _sentences(note)

    dx_keywords: List[List[str]] = []
    for dx in top5:
        title = str(dx.get("long_title", "")).lower()
        kws = [w for w in re.findall(r"\b[a-z]{5,}\b", title) if w not in _STOPWORDS][:4]
        dx_keywords.append(kws)

    sent_matches: List[Set[int]] = []
    for sent in sents:
        sent_lower = sent.lower()
        word_count = len(sent_lower.split())
        if word_count < 5:
            sent_matches.append(set())
            continue
        matched = set()
        for i, kws in enumerate(dx_keywords):
            if kws and any(kw in sent_lower for kw in kws):
                matched.add(i)
        sent_matches.append(matched)

    valid_sent_matches: List[Set[int]] = []
    for matched in sent_matches:
        if len(matched) >= 3:
            valid_sent_matches.append(set())
        else:
            valid_sent_matches.append(matched)

    covered_indices: Set[int] = set()
    for matched in valid_sent_matches:
        covered_indices |= matched

    base_coverage = len(covered_indices) / len(top5)

    all_kws    = {kw for kws in dx_keywords for kw in kws}
    note_words = _words(note)
    if note_words:
        kw_density = sum(1 for w in note_words if w in all_kws) / len(note_words)
        if kw_density > 0.08:
            base_coverage *= 0.5

    return round(base_coverage, 4)


# ─── 2. Disposition accuracy ─────────────────────────────────────────────────

_DISPO_SYNONYMS: Dict[str, List[str]] = {
    "home_with_services": ["home health", "home with services", "home care"],
    "snf":                ["skilled nursing", "snf", "nursing facility", "long-term care"],
    "rehab":              ["rehabilitation", "rehab facility", "inpatient rehab"],
    "hospice":            ["hospice", "comfort care", "palliative"],
    "expired":            ["expired", "deceased", "passed away", "death", "died"],
    "ama":                ["against medical advice", "left ama", "against advice"],
    "home":               ["discharged home", "home with", "returned home", "discharge home"],
    "other":              ["transferred to", "transfer to"],
}


def _true_canonical(location: str) -> str:
    loc = location.upper().strip()
    if "HEALTH CARE" in loc or "HOME WITH" in loc:                      return "home_with_services"
    if "SKILLED NURSING" in loc or "SNF" in loc or "LONG TERM" in loc: return "snf"
    if "REHAB" in loc:                                                  return "rehab"
    if "HOSPICE" in loc:                                                return "hospice"
    if "AGAINST ADVICE" in loc or "AMA" in loc:                        return "ama"
    if "DIED" in loc or "EXPIRED" in loc or "DEAD" in loc:             return "expired"
    if "TRANSFER" in loc:                                               return "other"
    if "HOME" in loc or "SELF" in loc:                                  return "home"
    return "other"


def _disposition_mentioned(note: str, episode: Dict) -> float:
    true_location = str(episode.get("discharge_location", "")).strip()
    if not true_location:
        return 0.5
    note_lower = note.lower()
    canonical  = _true_canonical(true_location)
    synonyms   = _DISPO_SYNONYMS.get(canonical, [])
    if any(syn in note_lower for syn in synonyms):
        return 1.0
    if "discharg" in note_lower:
        return 0.3
    return 0.0


# ─── 3. Medication F1 with emar_drug_set (v3) ─────────────────────────────────

def _extract_mentioned_drugs(
    note: str,
    known_drugs: List[str],
    emar_drug_set: Set[str] = None,
) -> Tuple[Set[str], Set[str]]:
    """
    Returns (true_positives, false_positives).
    v3: a detected drug is NOT a false positive if it matches episode prescriptions
    OR the emar_drug_set.
    """
    note_lower = note.lower()

    ep_stems: Set[str] = set()
    for drug in known_drugs:
        tokens = [t for t in str(drug).lower().split() if len(t) >= 4]
        if tokens:
            ep_stems.add(tokens[0])

    emar_stems: Set[str] = set()
    if emar_drug_set:
        for drug in emar_drug_set:
            tokens = [t for t in str(drug).lower().split() if len(t) >= 4]
            if tokens:
                emar_stems.add(tokens[0])

    all_known_stems = ep_stems | emar_stems

    true_positives = {stem for stem in ep_stems if stem in note_lower}

    drug_suffixes = re.compile(
        r"\b\w*(?:mab|nib|pril|sartan|olol|pam|lam|statin|mycin|cillin|"
        r"oxacin|cycline|azole|prazole|tidine|triptan|vir|mide|zide|"
        r"done|pine|xine|zine|dine|line|rine|mine|sine|vine|lone)\b",
        re.IGNORECASE,
    )
    note_drug_tokens = {m.group().lower() for m in drug_suffixes.finditer(note)}

    med_context = re.compile(
        r"(?:medication|drug|prescribed|continued|started|taking|given|dose of|mg of)\s+([a-z]{4,})",
        re.IGNORECASE,
    )
    for m in med_context.finditer(note):
        note_drug_tokens.add(m.group(1).lower())

    # False positives: detected tokens not matching ANY known drug source
    false_positives = {
        t for t in note_drug_tokens
        if not any(
            t.startswith(stem[:4]) or stem.startswith(t[:4])
            for stem in all_known_stems
        )
    }

    return true_positives, false_positives


def _medication_f1(
    note: str,
    medications: List[Dict],
    emar_drug_set: Set[str] = None,
) -> Tuple[float, float]:
    if not medications:
        return 0.5, 0.0

    known_drugs = [m.get("drug", "") for m in medications[:10] if m.get("drug")]
    top5_drugs  = known_drugs[:5]

    tp, fp = _extract_mentioned_drugs(note, known_drugs, emar_drug_set)

    recall       = len(tp) / len(top5_drugs) if top5_drugs else 0.0
    all_detected = len(tp) + len(fp)
    precision    = len(tp) / all_detected if all_detected > 0 else (1.0 if not tp else 0.0)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    hallucination_rate = len(fp) / all_detected if all_detected > 0 else 0.0

    return round(f1, 4), round(hallucination_rate, 4)


# ─── 4. LOS accuracy ──────────────────────────────────────────────────────────

def _los_accuracy(note: str, los_days: float) -> float:
    note_lower = note.lower()
    los_kws    = ["day", "days", "admitted for", "hospital stay", "length of stay",
                  "los", "hospitalized for", "overnight", "week"]
    has_context = any(kw in note_lower for kw in los_kws)
    if not has_context:
        return 0.0
    numbers   = [int(n) for n in re.findall(r"\b(\d{1,3})\b", note)]
    los_r     = round(los_days)
    tolerance = max(1, round(los_r * 0.25))
    if any(abs(n - los_r) <= tolerance for n in numbers):
        return 1.0
    return 0.3


# ─── 5. Structure score ───────────────────────────────────────────────────────

_REQUIRED_SECTIONS = [
    ("diagnosis",   ["diagnosis", "diagnos", "presenting", "chief complaint", "admission dx"]),
    ("course",      ["hospital course", "clinical course", "course of", "during admission",
                     "during hospitalization", "inpatient course"]),
    ("medications", ["medication", "medicines", "drugs", "prescri", "discharge med"]),
    ("disposition", ["discharg", "disposition", "transfer", "home", "facility"]),
    ("followup",    ["follow", "follow-up", "appointment", "clinic", "return", "outpatient"]),
    ("warnings",    ["call", "return to", "seek", "emergency", "warning", "symptom",
                     "chest pain", "shortness of breath", "fever", "worsening"]),
]


def _structure_score(note: str) -> float:
    note_lower = note.lower()
    sents      = _sentences(note)
    long_sents = [s.lower() for s in sents if len(s.split()) >= 5]

    sections_present = 0
    for _name, triggers in _REQUIRED_SECTIONS:
        if any(any(t in s for t in triggers) for s in long_sents):
            sections_present += 1

    word_count = len(note.split())
    if word_count < 100:
        return 0.0

    base = sections_present / len(_REQUIRED_SECTIONS)

    import math
    length_factor = min(1.0, math.log2(max(1, word_count / 100)) / math.log2(5))

    return round(0.70 * base + 0.30 * length_factor, 4)


# ─── 6. Information density ───────────────────────────────────────────────────

def _information_density(note: str) -> float:
    words = _words(note)
    if not words:
        return 0.0

    window = 100
    ttr_scores = []
    for i in range(0, len(words), window):
        chunk = words[i:i + window]
        if len(chunk) >= 20:
            ttr_scores.append(len(set(chunk)) / len(chunk))
    mean_ttr = sum(ttr_scores) / len(ttr_scores) if ttr_scores else 0.5

    sents       = _sentences(note)
    sent_tokens = [set(_words(s)) for s in sents if len(_words(s)) >= 5]
    duplicate_pairs = 0
    total_pairs     = 0
    for i in range(len(sent_tokens)):
        for j in range(i + 1, len(sent_tokens)):
            a, b  = sent_tokens[i], sent_tokens[j]
            union = a | b
            if not union:
                continue
            overlap = len(a & b) / len(union)
            total_pairs += 1
            if overlap >= 0.70:
                duplicate_pairs += 1

    repeat_penalty = (duplicate_pairs / total_pairs) if total_pairs > 0 else 0.0
    density        = max(0.0, mean_ttr - repeat_penalty)
    normalised     = min(1.0, max(0.0, (density - 0.30) / 0.45))
    return round(normalised, 4)


# ─── Grader class ─────────────────────────────────────────────────────────────

class NoteGrader:
    """
    v3 grader for Task 3.

    Scoring formula:
        raw = 0.30×dx + 0.20×dispo + 0.20×med_f1 + 0.15×los + 0.10×structure + 0.05×density
        penalty = hallucination_rate × 0.15          (max 0.15 deduction)
        followup_penalty = 0.05 if planning finalized but follow-up omitted
        final = max(0.0, raw − penalty − followup_penalty)
    """

    def grade(
        self,
        action:  Any,
        episode: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], bool, Dict[str, Any]]:

        if action.task3 is None:
            return 0.0, {"error_no_task3": -1.0}, True, {"error": "Action.task3 is missing"}

        note          = action.task3.discharge_note or ""
        diagnoses     = episode.get("diagnoses",  [])
        medications   = episode.get("medications", [])
        emar_drug_set = episode.get("_emar_drug_set", set())
        los_days      = float(episode.get("hospital_los_days", 0) or 0)

        dx_cov         = _diagnosis_coverage(note, diagnoses)
        dispo          = _disposition_mentioned(note, episode)
        med_f1, halluc = _medication_f1(note, medications, emar_drug_set)
        los_acc        = _los_accuracy(note, los_days)
        structure      = _structure_score(note)
        density        = _information_density(note)

        halluc_penalty = round(min(0.15, halluc * 0.15), 4)

        # Follow-up structure penalty (v3)
        followup_penalty = 0.0
        do_raw = episode.get("discharge_orders") or {}
        if do_raw.get("discharge_planning_finalized", False):
            note_lower = note.lower()
            if "follow-up" not in note_lower and "follow up" not in note_lower:
                followup_penalty = 0.05

        raw = (
            0.30 * dx_cov
            + 0.20 * dispo
            + 0.20 * med_f1
            + 0.15 * los_acc
            + 0.10 * structure
            + 0.05 * density
        )
        final = round(
            max(0.0, min(1.0, raw - halluc_penalty - followup_penalty)), 4
        )

        partial = {
            "diagnosis_coverage":    round(dx_cov,   4),
            "disposition_score":     round(dispo,     4),
            "medication_f1":         round(med_f1,    4),
            "hallucination_rate":    round(halluc,    4),
            "hallucination_penalty": halluc_penalty,
            "followup_structure_penalty": followup_penalty,
            "los_accuracy":          round(los_acc,   4),
            "structure_score":       round(structure, 4),
            "information_density":   round(density,   4),
        }

        info = {
            "note_word_count":    len(note.split()),
            "hospital_los_days":  round(los_days, 2),
            "n_diagnoses":        len(diagnoses),
            "n_medications":      len(medications),
            "n_emar_drugs":       len(emar_drug_set),
            "discharge_location": episode.get("discharge_location", ""),
            "discharge_planning_finalized": bool(
                do_raw.get("discharge_planning_finalized", False)
            ),
        }

        return final, partial, True, info
