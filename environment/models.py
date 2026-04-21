from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─── Sub-models for Observation ──────────────────────────────────────────────

class DiagnosisInfo(BaseModel):
    icd_code:    str
    icd_version: int
    description: str
    seq_num:     int


class ICUStay(BaseModel):
    stay_id:        int
    los_days:       float
    first_careunit: str
    last_careunit:  str


class LabFlag(BaseModel):
    label: str
    flag:  str
    value: Optional[str] = None


class Medication(BaseModel):
    drug:        str
    route:       Optional[str] = None
    dose_val_rx: Optional[str] = None


class MicrobiologyResult(BaseModel):
    specimen:     str
    organism:     str
    resistant_to: List[str] = Field(default_factory=list)
    sensitive_to: List[str] = Field(default_factory=list)


class ICUProcedureSummary(BaseModel):
    ventilation_hours: float     = 0.0
    has_arterial_line: bool      = False
    has_central_line:  bool      = False
    has_dialysis:      bool      = False
    procedure_names:   List[str] = Field(default_factory=list)


class VitalSign(BaseModel):
    name:            str
    admission_value: Optional[float] = None
    discharge_value: Optional[float] = None
    min_value:       Optional[float] = None
    max_value:       Optional[float] = None
    critical_flag:   bool = False


class FluidBalance(BaseModel):
    total_input_ml:  float
    total_urine_ml:  float
    total_output_ml: float
    net_balance_ml:  float
    fluid_overloaded: bool
    oliguria:         bool


class EmarMedication(BaseModel):
    medication:          str
    first_given:         str
    last_given:          str
    total_doses:         int
    active_at_discharge: bool


class DischargeOrders(BaseModel):
    discharge_planning_finalized: bool
    documented_discharge_orders:  List[str]


# ─── Core OpenEnv Models ──────────────────────────────────────────────────────

class Observation(BaseModel):
    """Structured patient observation returned by reset() and step()."""
    task_id:    int
    subject_id: int
    hadm_id:    int
    step_num:   int
    max_steps:  int

    # Demographics
    age:                int
    gender:             str
    admission_type:     str
    admission_location: str
    insurance:          str
    language:           str

    # Clinical data
    diagnoses:         List[DiagnosisInfo]    = Field(default_factory=list)
    icu_stays:         List[ICUStay]          = Field(default_factory=list)
    medications:       List[Medication]       = Field(default_factory=list)
    lab_flags:         List[LabFlag]          = Field(default_factory=list)
    procedures:        List[str]              = Field(default_factory=list)
    drg_codes:         List[str]              = Field(default_factory=list)
    hospital_los_days: float                  = 0.0

    # v2 enrichments
    microbiology:    List[MicrobiologyResult] = Field(default_factory=list)
    weight_kg:       Optional[float]          = None
    bmi:             Optional[float]          = None
    care_trajectory: List[str]                = Field(default_factory=list)
    icu_procedures:  ICUProcedureSummary      = Field(default_factory=ICUProcedureSummary)
    hcpcs_categories: List[str]              = Field(default_factory=list)
    pharmacy_active:  List[str]              = Field(default_factory=list)
    pharmacy_stopped: List[str]              = Field(default_factory=list)

    # v3 enrichments
    vitals:           List[VitalSign]          = Field(default_factory=list)
    fluid_balance:    Optional[FluidBalance]   = None
    emar_summary:     List[EmarMedication]     = Field(default_factory=list)
    discharge_orders: Optional[DischargeOrders] = None
    complexity:       str                      = "medium"
    noise_level:      str                      = "clean"

    # Task context shown to the agent
    task_description:         str
    action_space_description: str

    # Multi-step memory (populated from step 2 onward for Task 4)
    episode_history: List[Dict[str, Any]] = Field(default_factory=list)


# ─── Action sub-models ───────────────────────────────────────────────────────

class Task1Action(BaseModel):
    disposition: str
    reasoning:   Optional[str] = None


class Task2Action(BaseModel):
    follow_up_specialties:      List[str] = Field(default_factory=list)
    medications_to_continue:    List[str] = Field(default_factory=list)
    medications_to_discontinue: List[str] = Field(default_factory=list)
    key_instructions:           List[str] = Field(default_factory=list)
    reasoning:                  Optional[str] = None


class Task3Action(BaseModel):
    discharge_note: str


class Task4Action(BaseModel):
    # Step 1
    triage_level: Optional[str] = None           # "icu" | "stepdown" | "floor"
    # Step 2
    priority_labs:      Optional[List[str]] = None
    priority_consults:  Optional[List[str]] = None
    # Step 3
    interventions:      Optional[List[str]] = None
    # Step 4
    high_risk_medications: Optional[List[str]] = None
    # Step 5
    antibiotic_strategy: Optional[str] = None    # "none"|"targeted"|"broad"|"empiric"
    antibiotics:         Optional[List[str]] = None
    # Step 6
    fluid_strategy: Optional[str] = None         # "restrict_diuresis"|"aggressive_resuscitation"|"maintain"
    # Step 7
    ready_for_stepdown: Optional[bool] = None
    barriers:           Optional[List[str]] = None
    # Step 8
    predicted_disposition: Optional[str] = None  # same values as Task 1
    los_remaining_days:    Optional[float] = None
    # Step 9
    medications_to_continue: Optional[List[str]] = None
    # Step 10
    final_note: Optional[str] = None
    # Revision (any step)
    revise_step: Optional[int]              = None
    revision:    Optional[Dict[str, Any]]   = None


class Action(BaseModel):
    task_id:             int
    task1:               Optional[Task1Action]  = None
    task2:               Optional[Task2Action]  = None
    task3:               Optional[Task3Action]  = None
    task4:               Optional[Task4Action]  = None
    information_request: Optional[List[str]]    = None
    # valid values: "labs", "vitals", "medications", "microbiology", "fluid_balance"


# ─── Response models ─────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation:     Optional[Observation] = None
    reward:          float
    done:            bool
    info:            Dict[str, Any]        = Field(default_factory=dict)
    partial_signals: Dict[str, float]      = Field(default_factory=dict)


class StateInfo(BaseModel):
    current_task_id:    int
    current_subject_id: Optional[int]          = None
    current_hadm_id:    Optional[int]          = None
    step_num:           int
    total_episodes_run: int
    last_reward:        Optional[float]        = None
    ground_truth:       Optional[Dict[str, Any]] = None
    task4_shaping_log:  Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    task_id:         int           = 1
    hadm_id:         Optional[int] = None
    noise_level:     str           = "clean"
    curriculum_mode: str           = "random"
