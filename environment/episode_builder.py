"""
episode_builder.py — Static synthetic episodes (no MIMIC data required).

Replaces the pandas/CSV EpisodeBuilder with a hardcoded dict of 6 realistic
synthetic patients. The public interface is identical to the original:

    builder = EpisodeBuilder()
    builder.hadm_ids          # List[int]
    builder.get_episode(None) # Dict[str, Any]  — random patient
    builder.get_episode(1001) # Dict[str, Any]  — pinned patient

All field names and types match exactly what env.py and the three graders expect.
Server starts in <1 s instead of >60 s. Healthcheck passes. Pipeline unblocked.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic patient episodes
# ---------------------------------------------------------------------------
# Each dict mirrors the output of the original EpisodeBuilder.get_episode().
# Fields are chosen to give the LLM enough signal for all three tasks.
#
# Critical fields per grader
#  Task 1: discharge_location  (canonical MIMIC string — grader parses it)
#  Task 2: diagnoses, medications, pharmacy_active, pharmacy_stopped,
#          microbiology, hcpcs_categories
#  Task 3: diagnoses, medications, hospital_los_days, discharge_location
# ---------------------------------------------------------------------------

_EPISODES: Dict[int, Dict[str, Any]] = {

    # ── 1001 — Cardiac patient → HOME HEALTH CARE ────────────────────────────
    1001: {
        "hadm_id":            1001,
        "subject_id":         10001,
        "discharge_location": "HOME HEALTH CARE",
        "age":                68,
        "gender":             "M",
        "admission_type":     "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "insurance":          "Medicare",
        "language":           "ENGLISH",
        "hospital_los_days":  5.3,
        "diagnoses": [
            {"seq_num": 1, "icd_code": "I50.9",  "icd_version": 10,
             "long_title": "Heart failure, unspecified"},
            {"seq_num": 2, "icd_code": "I10",    "icd_version": 10,
             "long_title": "Essential (primary) hypertension"},
            {"seq_num": 3, "icd_code": "E11.9",  "icd_version": 10,
             "long_title": "Type 2 diabetes mellitus without complications"},
            {"seq_num": 4, "icd_code": "N18.3",  "icd_version": 10,
             "long_title": "Chronic kidney disease, stage 3"},
            {"seq_num": 5, "icd_code": "I48.91", "icd_version": 10,
             "long_title": "Unspecified atrial fibrillation"},
        ],
        "procedures": [
            {"icd_code": "B215ZZZ", "long_title": "Echocardiography, transthoracic"},
            {"icd_code": "4A023N7", "long_title": "Measurement of cardiac output"},
        ],
        "drgcodes": [
            {"drg_code": "291", "description": "Heart failure and shock with MCC",
             "drg_severity": 3, "drg_mortality": 2},
        ],
        "icu_stays": [
            {"stay_id": 20001, "los": 1.8,
             "first_careunit": "Cardiac Vascular Intensive Care Unit (CVICU)",
             "last_careunit": "Cardiac Vascular Intensive Care Unit (CVICU)"},
        ],
        "medications": [
            {"drug": "Furosemide",   "route": "IV",  "dose_val_rx": "40mg"},
            {"drug": "Metoprolol",   "route": "PO",  "dose_val_rx": "25mg"},
            {"drug": "Lisinopril",   "route": "PO",  "dose_val_rx": "5mg"},
            {"drug": "Apixaban",     "route": "PO",  "dose_val_rx": "5mg"},
            {"drug": "Atorvastatin", "route": "PO",  "dose_val_rx": "40mg"},
            {"drug": "Metformin",    "route": "PO",  "dose_val_rx": "500mg"},
        ],
        "pharmacy_active":  ["Metoprolol", "Lisinopril", "Apixaban", "Atorvastatin", "Metformin"],
        "pharmacy_stopped": ["Furosemide IV", "Heparin drip"],
        "lab_flags": [
            {"label": "BNP",        "flag": "H", "value": "1240"},
            {"label": "Creatinine", "flag": "H", "value": "1.8"},
            {"label": "Potassium",  "flag": "L", "value": "3.2"},
            {"label": "Sodium",     "flag": "L", "value": "132"},
        ],
        "microbiology": [],
        "weight_kg":        88.0,
        "bmi":              29.1,
        "care_trajectory":  ["Emergency Department", "CVICU", "Medicine"],
        "icu_procedure_summary": {
            "ventilation_hours": 0.0,
            "has_arterial_line": False,
            "has_central_line":  True,
            "has_dialysis":      False,
            "procedure_names":   ["Central venous catheter"],
        },
        "hcpcs_categories": ["Cardiovascular monitoring", "Echocardiography"],
    },

    # ── 1002 — Elderly post-ICU → SKILLED NURSING FACILITY ───────────────────
    1002: {
        "hadm_id":            1002,
        "subject_id":         10002,
        "discharge_location": "SKILLED NURSING FACILITY",
        "age":                81,
        "gender":             "F",
        "admission_type":     "URGENT",
        "admission_location": "TRANSFER FROM HOSPITAL",
        "insurance":          "Medicare",
        "language":           "ENGLISH",
        "hospital_los_days":  12.7,
        "diagnoses": [
            {"seq_num": 1, "icd_code": "J18.9",  "icd_version": 10,
             "long_title": "Unspecified pneumonia"},
            {"seq_num": 2, "icd_code": "J96.01", "icd_version": 10,
             "long_title": "Acute respiratory failure with hypoxia"},
            {"seq_num": 3, "icd_code": "A41.9",  "icd_version": 10,
             "long_title": "Sepsis, unspecified organism"},
            {"seq_num": 4, "icd_code": "N17.9",  "icd_version": 10,
             "long_title": "Acute kidney injury, unspecified"},
            {"seq_num": 5, "icd_code": "I10",    "icd_version": 10,
             "long_title": "Essential (primary) hypertension"},
        ],
        "procedures": [
            {"icd_code": "0BH17EZ", "long_title": "Endotracheal intubation"},
            {"icd_code": "5A1935Z", "long_title": "Respiratory ventilation, 24-96 hours"},
            {"icd_code": "4A023N7", "long_title": "Measurement of cardiac output"},
        ],
        "drgcodes": [
            {"drg_code": "207", "description": "Respiratory system diagnosis with ventilator support 96+ hours",
             "drg_severity": 4, "drg_mortality": 4},
        ],
        "icu_stays": [
            {"stay_id": 20002, "los": 7.4,
             "first_careunit": "Medical Intensive Care Unit (MICU)",
             "last_careunit":  "Medical Intensive Care Unit (MICU)"},
        ],
        "medications": [
            {"drug": "Piperacillin-tazobactam", "route": "IV",  "dose_val_rx": "3.375g"},
            {"drug": "Vancomycin",              "route": "IV",  "dose_val_rx": "1250mg"},
            {"drug": "Norepinephrine",          "route": "IV",  "dose_val_rx": "0.1 mcg/kg/min"},
            {"drug": "Amoxicillin-clavulanate", "route": "PO",  "dose_val_rx": "875mg"},
            {"drug": "Amlodipine",              "route": "PO",  "dose_val_rx": "5mg"},
        ],
        "pharmacy_active":  ["Amoxicillin-clavulanate", "Amlodipine"],
        "pharmacy_stopped": ["Piperacillin-tazobactam IV", "Vancomycin IV", "Norepinephrine"],
        "lab_flags": [
            {"label": "WBC",        "flag": "H", "value": "18.4"},
            {"label": "Lactate",    "flag": "H", "value": "3.9"},
            {"label": "Creatinine", "flag": "H", "value": "3.1"},
            {"label": "Procalcitonin", "flag": "H", "value": "28.3"},
        ],
        "microbiology": [
            {
                "specimen":     "Sputum",
                "organism":     "Klebsiella pneumoniae",
                "resistant_to": ["Ampicillin", "Ciprofloxacin"],
                "sensitive_to": ["Piperacillin-tazobactam", "Meropenem"],
            },
        ],
        "weight_kg":        61.0,
        "bmi":              22.8,
        "care_trajectory":  ["Emergency Department", "MICU", "Medicine"],
        "icu_procedure_summary": {
            "ventilation_hours": 58.5,
            "has_arterial_line": True,
            "has_central_line":  True,
            "has_dialysis":      False,
            "procedure_names":   ["Invasive mechanical ventilation", "Arterial line", "Central venous catheter"],
        },
        "hcpcs_categories": ["Respiratory therapy", "Pulmonary diagnostics"],
    },

    # ── 1003 — Simple infection → HOME ────────────────────────────────────────
    1003: {
        "hadm_id":            1003,
        "subject_id":         10003,
        "discharge_location": "HOME",
        "age":                34,
        "gender":             "F",
        "admission_type":     "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "insurance":          "Private",
        "language":           "ENGLISH",
        "hospital_los_days":  2.1,
        "diagnoses": [
            {"seq_num": 1, "icd_code": "L03.011", "icd_version": 10,
             "long_title": "Cellulitis of right foot"},
            {"seq_num": 2, "icd_code": "L03.115", "icd_version": 10,
             "long_title": "Cellulitis of right lower limb"},
            {"seq_num": 3, "icd_code": "E11.9",   "icd_version": 10,
             "long_title": "Type 2 diabetes mellitus without complications"},
        ],
        "procedures": [
            {"icd_code": "0HBMXZZ", "long_title": "Excision of right foot skin, external approach"},
        ],
        "drgcodes": [
            {"drg_code": "603", "description": "Cellulitis without MCC",
             "drg_severity": 1, "drg_mortality": 1},
        ],
        "icu_stays": [],
        "medications": [
            {"drug": "Cefazolin",   "route": "IV", "dose_val_rx": "1g"},
            {"drug": "Cephalexin", "route": "PO", "dose_val_rx": "500mg"},
            {"drug": "Metformin",  "route": "PO", "dose_val_rx": "1000mg"},
            {"drug": "Ibuprofen",  "route": "PO", "dose_val_rx": "400mg"},
        ],
        "pharmacy_active":  ["Cephalexin", "Metformin"],
        "pharmacy_stopped": ["Cefazolin IV", "Ibuprofen"],
        "lab_flags": [
            {"label": "WBC",     "flag": "H", "value": "13.2"},
            {"label": "Glucose", "flag": "H", "value": "194"},
        ],
        "microbiology": [
            {
                "specimen":     "Wound swab",
                "organism":     "Staphylococcus aureus",
                "resistant_to": [],
                "sensitive_to": ["Cefazolin", "Clindamycin", "Trimethoprim-sulfamethoxazole"],
            },
        ],
        "weight_kg":        72.0,
        "bmi":              26.3,
        "care_trajectory":  ["Emergency Department", "Medicine"],
        "icu_procedure_summary": {
            "ventilation_hours": 0.0,
            "has_arterial_line": False,
            "has_central_line":  False,
            "has_dialysis":      False,
            "procedure_names":   [],
        },
        "hcpcs_categories": ["Infectious disease consultation"],
    },

    # ── 1004 — Stroke patient → REHABILITATION ────────────────────────────────
    1004: {
        "hadm_id":            1004,
        "subject_id":         10004,
        "discharge_location": "REHABILITATION",
        "age":                62,
        "gender":             "M",
        "admission_type":     "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "insurance":          "Medicare",
        "language":           "ENGLISH",
        "hospital_los_days":  6.8,
        "diagnoses": [
            {"seq_num": 1, "icd_code": "I63.9",  "icd_version": 10,
             "long_title": "Cerebral infarction, unspecified"},
            {"seq_num": 2, "icd_code": "G81.90", "icd_version": 10,
             "long_title": "Hemiplegia, unspecified affecting unspecified side"},
            {"seq_num": 3, "icd_code": "I10",    "icd_version": 10,
             "long_title": "Essential (primary) hypertension"},
            {"seq_num": 4, "icd_code": "E78.5",  "icd_version": 10,
             "long_title": "Hyperlipidemia, unspecified"},
            {"seq_num": 5, "icd_code": "I48.19", "icd_version": 10,
             "long_title": "Other persistent atrial fibrillation"},
        ],
        "procedures": [
            {"icd_code": "B030ZZZ", "long_title": "Plain radiography of intracranial arteries"},
            {"icd_code": "B030Y0Z", "long_title": "MRI brain without contrast"},
            {"icd_code": "3E030GC", "long_title": "Thrombolytic alteplase administration"},
        ],
        "drgcodes": [
            {"drg_code": "61", "description": "Ischemic stroke precerebral occlusion with thrombolytic agent with MCC",
             "drg_severity": 3, "drg_mortality": 2},
        ],
        "icu_stays": [
            {"stay_id": 20004, "los": 2.1,
             "first_careunit": "Neuro Stepdown",
             "last_careunit":  "Neuro Stepdown"},
        ],
        "medications": [
            {"drug": "Alteplase",    "route": "IV", "dose_val_rx": "0.9mg/kg"},
            {"drug": "Aspirin",      "route": "PO", "dose_val_rx": "81mg"},
            {"drug": "Atorvastatin", "route": "PO", "dose_val_rx": "80mg"},
            {"drug": "Apixaban",     "route": "PO", "dose_val_rx": "5mg"},
            {"drug": "Amlodipine",   "route": "PO", "dose_val_rx": "10mg"},
            {"drug": "Lisinopril",   "route": "PO", "dose_val_rx": "10mg"},
        ],
        "pharmacy_active":  ["Aspirin", "Atorvastatin", "Apixaban", "Amlodipine", "Lisinopril"],
        "pharmacy_stopped": ["Alteplase IV"],
        "lab_flags": [
            {"label": "LDL",           "flag": "H", "value": "162"},
            {"label": "INR",           "flag": "H", "value": "2.4"},
            {"label": "Blood glucose", "flag": "H", "value": "188"},
        ],
        "microbiology": [],
        "weight_kg":        84.0,
        "bmi":              27.9,
        "care_trajectory":  ["Emergency Department", "Neuro Stepdown", "Neurology"],
        "icu_procedure_summary": {
            "ventilation_hours": 0.0,
            "has_arterial_line": False,
            "has_central_line":  False,
            "has_dialysis":      False,
            "procedure_names":   [],
        },
        "hcpcs_categories": ["Neurology consultation", "Physical therapy evaluation"],
    },

    # ── 1005 — Terminal cancer → HOSPICE ──────────────────────────────────────
    1005: {
        "hadm_id":            1005,
        "subject_id":         10005,
        "discharge_location": "HOSPICE-HOME",
        "age":                74,
        "gender":             "F",
        "admission_type":     "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "insurance":          "Medicare",
        "language":           "ENGLISH",
        "hospital_los_days":  4.2,
        "diagnoses": [
            {"seq_num": 1, "icd_code": "C34.12", "icd_version": 10,
             "long_title": "Malignant neoplasm of upper lobe, left bronchus or lung"},
            {"seq_num": 2, "icd_code": "C78.01", "icd_version": 10,
             "long_title": "Secondary malignant neoplasm of right lung"},
            {"seq_num": 3, "icd_code": "C79.51", "icd_version": 10,
             "long_title": "Secondary malignant neoplasm of bone"},
            {"seq_num": 4, "icd_code": "R06.09", "icd_version": 10,
             "long_title": "Other forms of dyspnea"},
            {"seq_num": 5, "icd_code": "R52",    "icd_version": 10,
             "long_title": "Pain, unspecified"},
        ],
        "procedures": [
            {"icd_code": "BB04ZZZ", "long_title": "CT scan of chest"},
            {"icd_code": "B04KZZ",  "long_title": "Fluoroscopy of whole body for metastatic survey"},
        ],
        "drgcodes": [
            {"drg_code": "541", "description": "Tracheostomy with MV 96+ hours with extensive OR procedure",
             "drg_severity": 4, "drg_mortality": 4},
        ],
        "icu_stays": [],
        "medications": [
            {"drug": "Morphine",       "route": "IV",  "dose_val_rx": "2mg"},
            {"drug": "Ondansetron",    "route": "IV",  "dose_val_rx": "4mg"},
            {"drug": "Dexamethasone",  "route": "PO",  "dose_val_rx": "4mg"},
            {"drug": "Lorazepam",      "route": "PO",  "dose_val_rx": "0.5mg"},
            {"drug": "Oxycodone",      "route": "PO",  "dose_val_rx": "5mg"},
        ],
        "pharmacy_active":  ["Morphine oral", "Dexamethasone", "Lorazepam", "Oxycodone"],
        "pharmacy_stopped": ["Morphine IV", "Ondansetron IV"],
        "lab_flags": [
            {"label": "Albumin",    "flag": "L", "value": "2.1"},
            {"label": "Hemoglobin", "flag": "L", "value": "8.4"},
            {"label": "Calcium",    "flag": "H", "value": "11.8"},
        ],
        "microbiology": [],
        "weight_kg":        53.0,
        "bmi":              20.1,
        "care_trajectory":  ["Emergency Department", "Oncology"],
        "icu_procedure_summary": {
            "ventilation_hours": 0.0,
            "has_arterial_line": False,
            "has_central_line":  False,
            "has_dialysis":      False,
            "procedure_names":   [],
        },
        "hcpcs_categories": ["Oncology consultation", "Palliative care consultation"],
    },

    # ── 1006 — COPD exacerbation → HOME WITH SERVICES ─────────────────────────
    1006: {
        "hadm_id":            1006,
        "subject_id":         10006,
        "discharge_location": "HOME WITH SERVICE",
        "age":                71,
        "gender":             "M",
        "admission_type":     "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "insurance":          "Medicare",
        "language":           "ENGLISH",
        "hospital_los_days":  4.9,
        "diagnoses": [
            {"seq_num": 1, "icd_code": "J44.1",  "icd_version": 10,
             "long_title": "Chronic obstructive pulmonary disease with (acute) exacerbation"},
            {"seq_num": 2, "icd_code": "J18.9",  "icd_version": 10,
             "long_title": "Unspecified pneumonia"},
            {"seq_num": 3, "icd_code": "I10",    "icd_version": 10,
             "long_title": "Essential (primary) hypertension"},
            {"seq_num": 4, "icd_code": "E11.65", "icd_version": 10,
             "long_title": "Type 2 diabetes with hyperglycemia"},
            {"seq_num": 5, "icd_code": "G47.33", "icd_version": 10,
             "long_title": "Obstructive sleep apnea (adult) (pediatric)"},
        ],
        "procedures": [
            {"icd_code": "0BH17EZ", "long_title": "Intubation cervical trachea"},
            {"icd_code": "GZJ1ZZZ", "long_title": "Spirometry"},
        ],
        "drgcodes": [
            {"drg_code": "190", "description": "Chronic obstructive pulmonary disease with MCC",
             "drg_severity": 3, "drg_mortality": 2},
        ],
        "icu_stays": [
            {"stay_id": 20006, "los": 1.2,
             "first_careunit": "Medical Intensive Care Unit (MICU)",
             "last_careunit":  "Medical Intensive Care Unit (MICU)"},
        ],
        "medications": [
            {"drug": "Albuterol",               "route": "Inhalation", "dose_val_rx": "2.5mg"},
            {"drug": "Ipratropium",             "route": "Inhalation", "dose_val_rx": "0.5mg"},
            {"drug": "Prednisone",              "route": "PO",         "dose_val_rx": "40mg"},
            {"drug": "Azithromycin",            "route": "PO",         "dose_val_rx": "500mg"},
            {"drug": "Tiotropium",              "route": "Inhalation", "dose_val_rx": "18mcg"},
            {"drug": "Fluticasone-salmeterol",  "route": "Inhalation", "dose_val_rx": "250/50mcg"},
            {"drug": "Lisinopril",              "route": "PO",         "dose_val_rx": "10mg"},
            {"drug": "Metformin",               "route": "PO",         "dose_val_rx": "1000mg"},
        ],
        "pharmacy_active":  [
            "Albuterol inhaler", "Tiotropium", "Fluticasone-salmeterol",
            "Prednisone", "Lisinopril", "Metformin",
        ],
        "pharmacy_stopped": ["Azithromycin", "Ipratropium nebulizer"],
        "lab_flags": [
            {"label": "pO2",     "flag": "L", "value": "58"},
            {"label": "pCO2",    "flag": "H", "value": "62"},
            {"label": "WBC",     "flag": "H", "value": "14.1"},
            {"label": "Glucose", "flag": "H", "value": "231"},
        ],
        "microbiology": [],
        "weight_kg":        91.0,
        "bmi":              30.4,
        "care_trajectory":  ["Emergency Department", "MICU", "Medicine"],
        "icu_procedure_summary": {
            "ventilation_hours": 14.0,
            "has_arterial_line": False,
            "has_central_line":  False,
            "has_dialysis":      False,
            "procedure_names":   ["Non-invasive positive pressure ventilation"],
        },
        "hcpcs_categories": ["Respiratory therapy", "Pulmonary function testing"],
    },
}


# ---------------------------------------------------------------------------
# EpisodeBuilder — same public interface as the original
# ---------------------------------------------------------------------------

class EpisodeBuilder:
    """
    Serves synthetic patient episodes without loading any files from disk.
    Identical interface to the MIMIC-IV-based EpisodeBuilder.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        # data_root is accepted but ignored — no files are read
        if data_root:
            logger.info(
                "EpisodeBuilder: data_root='%s' supplied but using synthetic data — "
                "no files loaded.",
                data_root,
            )
        self.hadm_ids: List[int] = list(_EPISODES.keys())
        logger.info(
            "EpisodeBuilder ready — %d synthetic episodes available (no MIMIC data required)",
            len(self.hadm_ids),
        )

    def get_episode(self, hadm_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Return one episode dict.
        hadm_id=None → random episode
        hadm_id=int  → pinned episode (raises ValueError if unknown)
        """
        if hadm_id is None:
            hadm_id = random.choice(self.hadm_ids)
        elif hadm_id not in _EPISODES:
            raise ValueError(
                f"hadm_id {hadm_id} not found. "
                f"Available: {self.hadm_ids}"
            )
        # Return a shallow copy so callers can't mutate the master dict
        return dict(_EPISODES[hadm_id])

    def sample_hadm_ids(self, n: Optional[int] = None) -> List[int]:
        if n is None:
            return list(self.hadm_ids)
        return random.sample(self.hadm_ids, min(n, len(self.hadm_ids)))