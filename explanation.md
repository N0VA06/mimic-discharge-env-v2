# MIMIC Discharge Planning — Full Technical Reference

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Layer — MIMIC-IV Episode Builder](#2-data-layer--mimic-iv-episode-builder)
3. [Pydantic Schema Layer](#3-pydantic-schema-layer)
4. [Environment Core — MIMICDischargeEnv](#4-environment-core--mimicdischargeenv)
5. [Task 1 — Discharge Disposition (Easy)](#5-task-1--discharge-disposition-easy)
6. [Task 2 — Care Plan Recommendation (Medium)](#6-task-2--care-plan-recommendation-medium)
7. [Task 3 — Discharge Note Generation (Hard)](#7-task-3--discharge-note-generation-hard)
8. [Task 4 — Long-Horizon Workflow (Very Hard)](#8-task-4--long-horizon-workflow-very-hard)
9. [Stochastic Observation Masking](#9-stochastic-observation-masking)
10. [Curriculum Learning](#10-curriculum-learning)
11. [Multi-Agent / Multi-Step Design](#11-multi-agent--multi-step-design)
12. [Training Pipeline](#12-training-pipeline)
13. [Server API Layer](#13-server-api-layer)
14. [End-to-End Request Flow](#14-end-to-end-request-flow)
15. [curl Test Reference](#15-curl-test-reference)

---

## 1. System Overview

The environment is a **deterministic, server-backed reinforcement learning environment** for clinical discharge planning, grounded in real patient data from the MIMIC-IV Clinical Database Demo v2.2 (PhysioNet). An AI agent plays the role of a physician making sequential discharge decisions.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent / LLM                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Server (server/app.py)              │
│   POST /reset  POST /step  GET /health  GET /metrics        │
│   POST /rollout  GET /complexity/{id}  GET /episodes/...    │
└───────────────────────────┬─────────────────────────────────┘
                            │ Python call
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MIMICDischargeEnv (environment/env.py)          │
│   reset() → Observation                                     │
│   step(Action) → StepResult                                 │
│   state() → StateInfo                                       │
└──────────┬────────────────────────┬────────────────────────┘
           │                        │
           ▼                        ▼
┌────────────────────┐   ┌──────────────────────────────────┐
│ EpisodeBuilder     │   │           Graders                 │
│ old_episode_       │   │  Task1: DispositionGrader         │
│ builder.py         │   │  Task2: CarePlanGrader            │
│                    │   │  Task3: NoteGrader                │
│ Reads MIMIC-IV CSV │   │  Task4: Task4Grader               │
│ files at startup   │   └──────────────────────────────────┘
└────────────────────┘
```

### Key Design Principles

- **No LLM judge** — all graders are deterministic Python functions operating on structured MIMIC fields
- **Reproducible** — pinning `hadm_id` + `noise_level` gives identical observations every run
- **Progressive difficulty** — 4 tasks from single-step classification (Task 1) to 10-step sparse-reward workflow (Task 4)
- **Training-aware** — stochastic masking, curriculum modes, and GRPO-compatible reward structure built in from the start

---

## 2. Data Layer — MIMIC-IV Episode Builder

### File: `environment/old_episode_builder.py`

The `EpisodeBuilder` class loads MIMIC-IV CSV tables once at startup and builds rich episode dicts on demand.

### Startup Loading

```python
class EpisodeBuilder:
    def __init__(self, data_root: Optional[str] = None):
        # Loads all tables into memory (pandas DataFrames)
        # ~100MB of CSV for the demo dataset
        self._load_tables()
        self._build_complexity_index()  # pre-classifies all hadm_ids
```

Tables loaded:
| Table | Use |
|-------|-----|
| `hosp/admissions.csv` | Admission metadata, discharge location, LOS |
| `hosp/patients.csv` | Age, gender |
| `hosp/diagnoses_icd.csv` | ICD codes ranked by seq_num |
| `hosp/prescriptions.csv` | Drug orders (drug name, route, dose) |
| `hosp/pharmacy.csv` | Active vs stopped pharmacy records |
| `hosp/labevents.csv` | Lab results (joined with labitems for labels) |
| `hosp/microbiologyevents.csv` | Cultures, organisms, sensitivities |
| `hosp/procedures_icd.csv` | ICD procedure codes |
| `hosp/hcpcsevents.csv` | HCPCS categories |
| `hosp/drgcodes.csv` | DRG codes with severity/mortality |
| `hosp/emar.csv` | Electronic medication administration records |
| `hosp/poe.csv` + `poe_detail.csv` | Provider order entries (discharge orders) |
| `icu/icustays.csv` | ICU unit names, LOS per stay |
| `icu/chartevents.csv` | Vital sign time series (lazy-filtered) |
| `icu/inputevents.csv` | Fluid inputs (mL) |
| `icu/outputevents.csv` | Fluid outputs including urine (mL) |
| `icu/procedureevents.csv` | ICU procedures (ventilation, dialysis, lines) |
| `icu/transfers.csv` | Care unit trajectory |

### `get_episode(hadm_id, noise_level)` — core method

Builds a complete episode dict for one hospitalisation:

```python
ep = {
    "subject_id": int,
    "hadm_id": int,
    "age": int,
    "gender": str,                    # "M" or "F"
    "admission_type": str,            # "EMERGENCY", "ELECTIVE", etc.
    "admission_location": str,
    "discharge_location": str,        # ground truth for Task 1/4
    "insurance": str,
    "language": str,
    "hospital_los_days": float,       # computed from admittime/dischtime delta

    # Clinical data
    "diagnoses": [{"icd_code", "icd_version", "long_title", "seq_num"}],
    "icu_stays": [{"stay_id", "first_careunit", "last_careunit", "los"}],
    "medications": [{"drug", "route", "dose_val_rx"}],
    "lab_flags": [{"label", "flag", "value"}],
    "procedures": [{"icd_code", "long_title"}],
    "drgcodes": [{"drg_code", "description", "drg_severity", "drg_mortality"}],
    "microbiology": [{"specimen", "organism", "resistant_to", "sensitive_to"}],
    "pharmacy_active": [str],         # drug names active at discharge
    "pharmacy_stopped": [str],        # drug names stopped during admission
    "care_trajectory": [str],         # ordered list of care units
    "icu_procedure_summary": {
        "ventilation_hours": float,
        "has_arterial_line": bool,
        "has_central_line": bool,
        "has_dialysis": bool,
        "procedure_names": [str],
    },
    "hcpcs_categories": [str],
    "weight_kg": float | None,
    "bmi": float | None,

    # v3 enrichments
    "vitals": {
        "Heart Rate": {"admission_value", "discharge_value", "min_value", "max_value", "critical_flag"},
        "Systolic BP": {...},
        ...
    },
    "fluid_balance": {
        "total_input_ml": float,
        "total_urine_ml": float,
        "total_output_ml": float,
        "net_balance_ml": float,      # input - output
        "fluid_overloaded": bool,     # net_balance > 3000mL
        "oliguria": bool,             # urine < 400mL/day average
    },
    "emar_summary": [{
        "medication": str,
        "first_given": str,           # ISO timestamp
        "last_given": str,
        "total_doses": int,
        "active_at_discharge": bool,  # last_given within 24h of dischtime
    }],
    "discharge_orders": {
        "discharge_planning_finalized": bool,
        "documented_discharge_orders": [str],
    },

    # Derived fields
    "complexity": "easy" | "medium" | "hard",
    "noise_level": str,               # set by _apply_noise()

    # Private (not sent to agent)
    "_emar_drug_set": Set[str],       # lowercased medication names from eMAR
}
```

### Vital Sign Extraction — `_get_vitals(hadm_id)`

Vital items tracked (MIMIC `itemid` mapping):
```python
_VITAL_ITEMS = {
    220045: "Heart Rate",
    220179: "Systolic BP",
    220180: "Diastolic BP",
    220277: "SpO2",
    223761: "Temperature F",
    220210: "Respiratory Rate",
    223900: "GCS Total",
}
```

The chartevents table is the largest in MIMIC (~10GB for full dataset, ~50MB for demo). It's loaded with column filters and only rows with `warning == 0` (no data quality alerts) are kept. For each vital sign per admission, the method computes:
- `admission_value`: median of the first 4 hours of data
- `discharge_value`: median of the last 4 hours of data
- `min_value`, `max_value`: over the entire stay
- `critical_flag`: True if any value exceeded clinical thresholds (HR>150 or <40, SpO2<88, etc.)

### Fluid Balance — `_get_fluid_balance(hadm_id)`

- `total_input_ml`: sum of all `inputevents.amount` where `amountuom` is mL-convertible
- `total_urine_ml`: sum of `outputevents` where `label` contains "urine" or "foley"
- `total_output_ml`: sum of all `outputevents.value`
- `net_balance_ml = total_input_ml - total_output_ml`
- `fluid_overloaded = net_balance_ml > 3000`
- `oliguria`: average daily urine output < 400 mL/day (0.5 mL/kg/h threshold converted to daily)

### eMAR Timeline — `_get_emar_summary(hadm_id, dischtime)`

Groups emar rows by medication name, computes first/last given timestamps and dose count. A medication is `active_at_discharge = True` if its `last_given` timestamp is within 24 hours before `dischtime`.

The `_emar_drug_set` private key stores a flat `Set[str]` of lowercased medication names from eMAR. This is used by Task 2 and Task 3 graders as a second source of truth for hallucination detection — a drug flagged as hallucinated must be absent from BOTH prescriptions AND eMAR.

### Discharge Orders — `_get_discharge_orders(hadm_id)`

The `poe` table has no `hadm_id` column in poe_detail — discharge order details require a join:
```
poe (subject_id, poe_id, order_type) 
  → join poe_detail (poe_id, field_name, field_value)
  → filter order_type == "Discharge Planning"
```
`discharge_planning_finalized = True` if the order set contains a finalized discharge planning note.

### Complexity Classification — `classify_complexity(ep)`

A `@staticmethod` that reads the episode dict and assigns a tier:

```
easy:   discharge_location == HOME (not with services)
        AND hospital_los_days <= 4
        AND len(icu_stays) == 0
        AND len(diagnoses) <= 5

hard:   discharge_location IN {SNF, REHAB, HOSPICE, DIED, ...}
        OR hospital_los_days > 14
        OR ventilation_hours > 24
        OR any drug resistance (MRSA, VRE, ESBL, etc.)
        OR oliguria == True

medium: everything else
```

The complexity index `{complexity: [hadm_id, ...]}` is built at init time by calling `classify_complexity` on a lightweight version of each episode, avoiding full extraction overhead.

---

## 3. Pydantic Schema Layer

### File: `environment/models.py`

All data crossing the HTTP boundary is validated by Pydantic v2 models. This prevents malformed agent actions from crashing the environment.

### Observation

Returned by `reset()` and by `step()` when `done=False`:

```python
class Observation(BaseModel):
    task_id:    int
    subject_id: int
    hadm_id:    int
    step_num:   int          # 0-indexed at reset, increments each step
    max_steps:  int          # 1 for Task 1/3, 4 for Task 2, 10 for Task 4

    # Always visible
    age, gender, admission_type, admission_location, insurance, language
    hospital_los_days: float
    complexity: str          # "easy" | "medium" | "hard"
    noise_level: str         # "clean" | "partial" | "noisy"

    # Clinical data (may be gated for Tasks 2/4)
    diagnoses:       List[DiagnosisInfo]
    icu_stays:       List[ICUStay]
    medications:     List[Medication]
    lab_flags:       List[LabFlag]
    procedures:      List[str]
    drg_codes:       List[str]
    microbiology:    List[MicrobiologyResult]
    icu_procedures:  ICUProcedureSummary
    care_trajectory: List[str]
    weight_kg, bmi:  Optional[float]
    pharmacy_active, pharmacy_stopped, hcpcs_categories: List[str]

    # v3 enrichments (may be gated)
    vitals:           List[VitalSign]
    fluid_balance:    Optional[FluidBalance]
    emar_summary:     List[EmarMedication]
    discharge_orders: Optional[DischargeOrders]

    # Task 4 only: prior step memory
    episode_history: List[Dict[str, Any]]  # [{step_num, action_summary, reward}]

    task_description:         str
    action_space_description: str
```

### Action

Sent by agent to `POST /step`:

```python
class Action(BaseModel):
    task_id: int
    task1:   Optional[Task1Action]   # {"disposition": str, "reasoning": str}
    task2:   Optional[Task2Action]   # care plan fields
    task3:   Optional[Task3Action]   # {"discharge_note": str}
    task4:   Optional[Task4Action]   # all 10 step fields (all optional)
    information_request: Optional[List[str]]  # Task 2 gating requests
```

### Task4Action — unified multi-step action

All fields optional; each step submission fills in only the relevant subset:

```python
class Task4Action(BaseModel):
    # Step 1
    triage_level: Optional[str]            # "icu"|"stepdown"|"floor"
    # Step 2
    priority_labs: Optional[List[str]]
    priority_consults: Optional[List[str]]
    # Step 3
    interventions: Optional[List[str]]
    # Step 4
    high_risk_medications: Optional[List[str]]
    # Step 5
    antibiotic_strategy: Optional[str]    # "none"|"targeted"|"broad"|"empiric"
    antibiotics: Optional[List[str]]
    # Step 6
    fluid_strategy: Optional[str]         # "restrict_diuresis"|"aggressive_resuscitation"|"maintain"
    # Step 7
    ready_for_stepdown: Optional[bool]
    barriers: Optional[List[str]]
    # Step 8
    predicted_disposition: Optional[str]
    los_remaining_days: Optional[float]
    # Step 9
    medications_to_continue: Optional[List[str]]
    # Step 10
    final_note: Optional[str]
    # Revision mechanism
    revise_step: Optional[int]            # 1-9
    revision: Optional[Dict[str, Any]]   # corrected fields for that step
```

### StepResult

Returned by every `step()` call:

```python
class StepResult(BaseModel):
    observation:     Optional[Observation]   # None when done=True
    reward:          float                   # 0.0–1.0
    done:            bool
    info:            Dict[str, Any]          # grader-specific debug info
    partial_signals: Dict[str, float]        # per-component scores
```

---

## 4. Environment Core — MIMICDischargeEnv

### File: `environment/env.py`

The environment is a single-instance stateful object. In the server, one instance is shared across all HTTP requests (single-process, no locking — intended for single-agent use).

### State Variables

```python
self._episode:           Dict       # raw episode dict from EpisodeBuilder
self._task_id:           int
self._step_num:          int        # 0 at reset, increments in step()
self._cumulative_reward: float
self._last_reward:       float
self._total_episodes:    int
self.active:             bool
self._revealed_info:     Set[str]   # Task 2: which info categories revealed
self._noise_level:       str
self._shaping_log:       Dict       # Task 4: per-step scores and metadata
self._episode_history:   List[Dict] # Task 4: prior step summaries for obs
```

### `reset()` Flow

```
reset(task_id, hadm_id=None, noise_level="clean", curriculum_mode="random")
  │
  ├─ Clamp task_id to [1, 4]
  ├─ Sample hadm_id if None using _sample_hadm_id(curriculum_mode)
  ├─ builder.get_episode(hadm_id, noise_level) → raw episode dict
  ├─ Zero all step state (_step_num, _cumulative_reward, etc.)
  ├─ Reset _revealed_info to _TASK2_INITIAL (demographics + los + complexity)
  ├─ Reset _shaping_log = {}
  ├─ Reset _episode_history = []
  └─ _build_observation() → Observation
```

### `step()` Dispatch

```
step(action: Action)
  │
  ├─ Increment _step_num
  ├─ Read cfg = TASK_CONFIG[_task_id]
  │
  ├─ Task 2 branch:
  │   ├─ If action.information_request → reveal categories, return obs (reward=0, done=False)
  │   ├─ If action.task2 → grade, apply discount, done=True
  │   └─ If neither → hint or timeout
  │
  ├─ Task 4 branch:
  │   ├─ grade(action, episode, step_num, _shaping_log) → (reward=0 for steps 1-9)
  │   ├─ Append to _episode_history
  │   └─ Return StepResult (done=False until step 10)
  │
  └─ Tasks 1/3 branch:
      ├─ grade(action, episode) → (reward, partial, done_grader, info)
      └─ Apply excess step penalty if step_num > max_steps
```

### `_build_observation()` — Gating Logic

The key function that constructs the `Observation` Pydantic model from the raw episode dict. It applies field gating based on task:

```python
_T4_THRESHOLDS = {
    "labs": 2, "vitals": 3, "icu_procedures": 3, "medications": 4,
    "emar": 4, "microbiology": 5, "fluid_balance": 6,
    "care_trajectory": 7, "discharge_orders": 8,
}

def _gate(category, value, default):
    if task2:
        return value if category in self._revealed_info else default
    if task4:
        return value if self._step_num >= _T4_THRESHOLDS.get(category, 0) else default
    return value  # Tasks 1 and 3: no gating
```

For Tasks 1 and 3, the full observation is returned immediately. For Task 2, fields are hidden until the agent explicitly requests them. For Task 4, fields are revealed automatically as the step counter advances — mimicking how a physician progressively orders tests and consults over the course of an admission.

---

## 5. Task 1 — Discharge Disposition (Easy)

### File: `environment/tasks/task1_disposition.py`

**Single-step, full observation.** The agent sees the complete patient record and must pick one of 8 canonical dispositions.

### Scoring

```
score = 1.0  if exact canonical match
      = 0.5  if same broad group (community / facility / end_of_life)
      = 0.25 if clinically adjacent (e.g., snf ↔ home_with_services)
      = 0.0  otherwise
      + 0.05 reasoning bonus (if ≥20 chars and contains clinical keyword or number)
```

### `normalize_mimic_location(location)` — Ground Truth Canonicalization

MIMIC stores free-text discharge locations like "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "DIED". This function maps them to the 8 canonical categories via ordered substring matching (more specific patterns checked first to avoid "HOME" matching "HOME HEALTH CARE").

### Broad Group Adjacency

```python
BROAD_GROUP = {
    "home":               "community",
    "home_with_services": "community_plus",
    "ama":                "community",
    "snf":                "facility",
    "rehab":              "facility",
    "hospice":            "end_of_life",
    "expired":            "end_of_life",
    "other":              "other",
}
```

Adjacent pairs (clinically close but wrong): `home ↔ home_with_services`, `snf ↔ rehab`, `hospice ↔ snf`, `expired ↔ hospice`.

### Fuzzy Prediction Matching

The agent's prediction string is normalized before comparison: hyphens replaced with underscores, common aliases resolved (`"skilled_nursing"` → `"snf"`, `"rehabilitation"` → `"rehab"`, etc.).

---

## 6. Task 2 — Care Plan Recommendation (Medium)

### File: `environment/tasks/task2_careplan.py`

**Multi-step with 4-step information revelation protocol.** The agent starts with limited data and must request additional categories before submitting a care plan.

### 4-Step Protocol

```
Step 0 (reset):  Always visible: demographics, top 5 diagnoses, LOS, complexity
Steps 1-3:       information_request → unlock one or more of:
                   "labs" | "vitals" | "medications" | "microbiology" | "fluid_balance"
Final step:      Submit task2 care plan → receive graded reward
```

The agent can submit the care plan at any step (step 1 through 4), trading off information quality against the step efficiency multiplier.

### Step Efficiency Multiplier

```python
{1: 1.0, 2: 1.0, 3: 0.85, 4: 0.70}
```

Optimal strategy: 1 information request + 1 care plan submission = 2 steps = 1.0× multiplier.

### Scoring Formula

```
raw = 0.35 × specialty_F1
    + 0.25 × medication_F1
    + 0.25 × instruction_quality
    + 0.15 × discontinue_accuracy
    − hallucination_penalty (max 0.10)
    − ghost_specialty_penalty (max 0.10)

final = raw × step_efficiency_discount
```

### Specialty F1 — Ground Truth Generation

Expected follow-up specialties are derived programmatically from the episode:
1. **ICD prefix mapping**: ICD code prefixes → specialty (e.g., `I*` → cardiology, `N18*` → nephrology)
2. **Microbiology**: any positive culture → infectious disease
3. **HCPCS categories**: e.g., cardiac HCPCS codes → cardiology

This produces a set of expected specialties. The agent's `follow_up_specialties` list is matched using token-stem fuzzy matching (e.g., "Cardiology" matches "cardiology", "Cardiac Surgery" also matches via substring).

### Medication F1

Expected medications = active pharmacy + top eMAR medications at discharge.

Drug names are stem-matched using `_med_stem(drug)`: take the first token of at least 4 characters. For example, "metoprolol succinate 25mg" → stem "meto". This tolerates brand/generic differences.

```
F1 = 2 × precision × recall / (precision + recall)
```

Where:
- `precision` = fraction of recommended drugs that are in the true medication set
- `recall` = fraction of true discharge meds mentioned by the agent

A drug recommended by the agent is NOT a false positive if it appears in either `prescriptions` OR `emar_summary` (dual-source check introduced in v3). This prevents penalising agents who correctly identify drugs from eMAR that weren't in the prescription table.

### Ghost Specialty Penalty

Penalises specialties the agent recommends when there's no clinical evidence for them:

```python
_SPECIALTY_ICD_PREFIXES = {
    "cardiology": ["I", "42", "41", "V45"],
    "nephrology": ["N18", "N17", "585"],
    "pulmonology": ["J", "496", "491"],
    ...
}
```

A specialty is a "ghost" if none of the patient's ICD codes map to that specialty's prefix list. Ghost penalty = min(0.10, 0.04 × ghost_count).

### Instruction Quality

Evaluated using a keyword density approach on the `key_instructions` list:
- Specific thresholds ("call if weight gain > 2 lbs") score higher than vague instructions ("monitor weight")
- Medical condition keywords, numeric values, and action verbs all contribute
- Duplicate instructions (high sentence overlap) are penalised

### Discontinue Accuracy

Cross-checks `medications_to_discontinue` against `pharmacy_stopped` (drugs actually stopped before discharge). F1 between the agent's discontinue list and the true stopped list.

---

## 7. Task 3 — Discharge Note Generation (Hard)

### File: `environment/tasks/task3_note.py`

**Single-step, full observation, free-text output.** The agent generates a complete clinical discharge summary (≥300 words) that is scored against structured MIMIC fields.

### Scoring Formula

```
raw = 0.30 × diagnosis_coverage
    + 0.20 × disposition_accuracy
    + 0.20 × medication_F1
    + 0.15 × LOS_accuracy
    + 0.10 × structure_score
    + 0.05 × information_density
    − hallucination_penalty (max 0.15)
    − followup_structure_penalty (0.05)

final = max(0.0, min(1.0, raw))
```

### Diagnosis Coverage — Anti-Stuffing Logic

For each of the top 5 diagnoses, extract keywords (≥5 chars, not stopwords) from the ICD long title. A diagnosis is "covered" if at least one sentence of ≥5 words contains any of its keywords.

**Anti-stuffing rule**: A single sentence that matches ≥3 different diagnoses simultaneously is treated as having zero coverage (indicates vague generic statements like "patient with multiple comorbidities including...").

**Keyword density guard**: If >8% of all note words are diagnosis keywords, the coverage score is halved (detects keyword-stuffed notes that repeat medical terms without forming coherent sentences).

### Medication F1 with Hallucination Detection

Two passes:
1. **True positives**: drug stems from the episode's prescription list that appear in the note
2. **False positives**: tokens matching drug suffix patterns (`-olol`, `-pril`, `-statin`, `-mycin`, etc.) or following medication context phrases ("prescribed X", "given X") that don't match ANY known drug (from both prescriptions AND eMAR)

`hallucination_rate = len(false_positives) / (len(true_positives) + len(false_positives))`
`hallucination_penalty = min(0.15, hallucination_rate × 0.15)`

### LOS Accuracy

The note must:
1. Contain at least one LOS context keyword ("day", "days", "hospital stay", "admitted for", etc.)
2. Mention a number within ±25% (or ±1 day, whichever is larger) of the actual LOS

If the note mentions LOS context but the number is wrong: 0.3. No LOS context at all: 0.0.

### Structure Score

Checks 6 required sections by scanning sentences of ≥5 words for trigger phrases:
```python
_REQUIRED_SECTIONS = [
    ("diagnosis",   ["diagnosis", "diagnos", "presenting", "chief complaint"]),
    ("course",      ["hospital course", "clinical course", "during admission"]),
    ("medications", ["medication", "medicines", "prescri", "discharge med"]),
    ("disposition", ["discharg", "disposition", "home", "facility"]),
    ("followup",    ["follow", "appointment", "clinic", "outpatient"]),
    ("warnings",    ["call", "return to", "emergency", "warning symptoms"]),
]
```

`structure_score = 0.70 × (sections_present/6) + 0.30 × length_factor`

Where `length_factor = min(1.0, log2(words/100) / log2(5))` — a note needs ~500 words for full length credit.

### Information Density

Uses Type-Token Ratio (TTR) in sliding 100-word windows to measure lexical diversity. A high-quality note uses varied vocabulary; keyword stuffing produces artificially low TTR.

Also penalises duplicate sentences: sentence pairs with ≥70% Jaccard overlap are counted as duplicates.

`density = mean_ttr - duplicate_pair_rate`
Normalised to [0, 1] range: `(density - 0.30) / 0.45`

### Follow-Up Structure Penalty (v3)

If `discharge_orders.discharge_planning_finalized == True` (the doctor marked discharge planning complete) but the note doesn't contain "follow-up" or "follow up", subtract 0.05. This ensures agents don't skip the follow-up section when the clinical record indicates planning was completed.

---

## 8. Task 4 — Long-Horizon Workflow (Very Hard)

### File: `environment/tasks/task4_workflow.py`

**10 sequential steps, sparse reward.** Steps 1-9 return `reward=0.0` but record shaping scores internally. Only Step 10 (the final discharge note) returns a non-zero reward that combines the note quality with how well the agent performed across all prior steps.

### Why Sparse Reward for GRPO?

GRPO (Group Relative Policy Optimization) works by comparing multiple rollouts of the same prompt to compute relative advantage. With dense rewards, the policy receives gradient signal at every step — but this can cause the model to optimize intermediate steps in isolation rather than learning coherent long-horizon strategy.

With sparse reward at Step 10 only, the GRPO gradient is assigned entirely to the final note generation, but the note's score incorporates a weighted average of all prior step scores (the `shaping_avg` term). This means the model must learn that good early decisions (correct triage, appropriate antibiotic selection, accurate medication reconciliation) causally improve the final note score.

### Per-Step Graders

#### Step 1 — Acuity Triage (max 0.10)

Maps the patient's first ICU care unit to a tier and checks if the agent agrees:
```
"Medical Intensive Care Unit (MICU)" → "icu"
"Stepdown Unit" → "stepdown"
"Medical Ward" → "floor"
```
Exact match: 1.0. Adjacent (icu↔stepdown): 0.5. Wrong by 2 tiers: 0.0.

#### Step 2 — Priority Labs + Consults (max 0.15)

**Labs**: Checks which lab flag categories (renal, cardiac, hepatic, hematology, infection, metabolic) have abnormal results in the episode. Scores the agent's `priority_labs` list against the expected categories.

**Consults**: Maps ICD codes to expected follow-up specialties using prefix tables (same logic as Task 2 ghost specialty check). Scores the agent's `priority_consults` list using substring matching against expected specialties.

`combined = (lab_score × 0.5) + (spec_score × 0.5)`

Step 2 also stores `step2_expected_specialties` in the shaping log — used at Step 10 for the trajectory bonus.

#### Step 3 — Interventions (max 0.15)

Checks which interventions are actually needed based on `icu_procedure_summary` and `fluid_balance`:
- `ventilation_hours > 0` → needs "intubation"/"mechanical ventilation"
- `has_arterial_line` → needs "arterial line"/"a-line"
- `has_dialysis` → needs "dialysis"/"crrt"
- `fluid_overloaded` → needs "fluid bolus"/"resuscitation"

Scores the agent's `interventions` list against what's actually needed (recall-oriented: credit for each correct intervention).

#### Step 4 — High-Risk Medications (max 0.10)

Identifies which high-risk drugs are actually in the patient's medication/eMAR record:
- Anticoagulants: heparin, warfarin, enoxaparin, apixaban, dabigatran, rivaroxaban
- Vasopressors: norepinephrine, epinephrine, dopamine, vasopressin, phenylephrine
- Sedatives: propofol, midazolam, fentanyl, lorazepam, dexmedetomidine

Uses 5-character stem matching to handle generic/brand variation. Computes F1 between the agent's flagged list and the true high-risk drug set.

#### Step 5 — Antibiotic Plan (max 0.10)

Checks microbiology results:
- **No organisms**: "none" strategy = 1.0; "broad"/"empiric" = 0.4 (unnecessary but not harmful)
- **Organisms present**: "targeted" + matching sensitive antibiotics = 1.0; "targeted" alone = 0.5; "broad" = 0.3; "none" = 0.0

**Isolation bonus (+0.02)**: If any resistant organism (MRSA, VRE, ESBL, KPC, CRE) is found, a small bonus is added if the agent correctly notes the need for isolation — this is separate from the antibiotic score and encourages infection control awareness.

#### Step 6 — Fluid Strategy (max 0.08)

Three-way classification based on `fluid_balance`:
```
fluid_overloaded == True → correct = "restrict_diuresis"
oliguria == True AND net_balance < 0 → correct = "aggressive_resuscitation"
otherwise → correct = "maintain"
```

Exact match: 1.0. Wrong direction (restrict vs aggressive): 0.0. Off-by-one (maintain when restrict/resuscitate needed): 0.4.

#### Step 7 — ICU Readiness (max 0.08)

Determines if the patient is clinically ready to move to stepdown/floor:
```
true_ready = los_days > 5 AND ventilation_hours == 0 AND last care_trajectory unit is "floor"/"stepdown"
```

Also scores the agent's `barriers` list against actual documented barriers (on-vent, oliguria, dialysis-dependent).

`combined = readiness_accuracy × 0.6 + barrier_detection × 0.4`

#### Step 8 — Disposition + LOS Estimate (max 0.10)

Disposition scoring (same algorithm as Task 1 but lower max):
- Exact match: 0.06
- Broad group match: 0.03
- Adjacent match: 0.015

LOS estimate scoring (compare `los_remaining_days` to actual `hospital_los_days`):
- Within ±2 days: 0.04
- Within ±5 days: 0.02
- Further: 0.0

Step 8 stores `step8_predicted_dispo` in the shaping log — used at Step 10 for the consistency bonus.

#### Step 9 — Medication Reconciliation (max 0.10)

Finds eMAR medications `active_at_discharge = True` (or falls back to `pharmacy_active` list if eMAR is empty). Computes F1 between the agent's `medications_to_continue` list and the true active discharge medications, using 4-character stem matching.

#### Step 10 — Final Note + Composite Reward

```python
# Base note quality from NoteGrader (Tasks 3 algorithm)
base_reward = NoteGrader().grade(final_note, episode)[0]

# Prior step performance average
step_scores = [shaping_log[f"step{i}_reward"] / STEP_MAX[i] for i in range(1, 10)]
shaping_avg = mean(step_scores)

# Combined
raw = base_reward × 0.60 + shaping_avg × 0.40

# Consistency bonus: final note disposition matches Step 8 prediction
if final_note mentions same disposition as step8_predicted_dispo:
    consistency_bonus = 0.10

# Trajectory bonus: ≥50% specialty overlap with Step 2 expected consults
if note mentions ≥50% of step2_expected_specialties:
    trajectory_bonus = 0.05

# Revision cost
revision_cost = shaping_log["revisions_used"] × 0.02

final = clamp(raw + consistency_bonus + trajectory_bonus - revision_cost, 0.0, 1.0)
```

### Revision Mechanism

At any step 1-9, the agent can include `revise_step + revision` in their action instead of (or alongside) the normal step fields. This causes:
1. The grader for the specified prior step is re-run with the revised content
2. The `step{N}_reward` entry in `shaping_log` is overwritten with the new score
3. `shaping_log["revisions_used"]` increments by 1

Maximum 2 revisions. Each costs 0.02 from the final reward. This enables the agent to backtrack if it realises (e.g., at step 7) that its step 5 antibiotic choice was wrong once microbiology results were revealed at step 5.

### Progressive Observation Gating for Task 4

The agent only sees what's clinically relevant at each step — mimicking how a physician orders tests and receives results over time:

| Step | Newly Revealed |
|------|---------------|
| 1 | Demographics, diagnoses, ICU stays (always) |
| 2 | Lab flags |
| 3 | Vitals, ICU procedures |
| 4 | Medications, eMAR summary |
| 5 | Microbiology |
| 6 | Fluid balance |
| 7 | Care trajectory |
| 8+ | Discharge orders, full observation |

### Episode History — In-Context Memory

After each step, `env._episode_history` is updated with a one-line summary of the action taken (`_format_action_summary()`). From step 2 onward, the `Observation.episode_history` field contains this list, allowing the agent to see its prior decisions in context when generating the next step's action. This is essential for coherent multi-step reasoning — without it, the LLM has no memory of what it decided in step 3 when it reaches step 9.

---

## 9. Stochastic Observation Masking

### `_apply_noise(ep, noise_level, hadm_id)` in EpisodeBuilder

Controlled stochasticity with per-(hadm_id, noise_level) seeded RNG for reproducibility:

```python
rng = random.Random(f"{hadm_id}:{noise_level}")
```

| noise_level | What's masked |
|-------------|--------------|
| `clean` | Nothing — identical output per hadm_id every time |
| `partial` | 30% random lab drop, 40% chance weight/BMI → None, medications truncated to 7, 1 random care unit dropped from trajectory |
| `noisy` | All of partial + diagnoses shuffled (sequence numbers randomised), 25% microbiology rows dropped, 2 random vitals removed, 20% chance fluid_balance → None, 15% drug names in medications replaced with formulary codes |

**Training use**: `noise_level="partial"` for GRPO training (forces the model to generalise from incomplete data). `noise_level="clean"` for evaluation (fully reproducible).

The seeded RNG means the same (hadm_id, noise_level) combination always produces the same masked observation, making training rollouts reproducible and comparable across model versions.

---

## 10. Curriculum Learning

### `_sample_hadm_id(curriculum_mode, task_id)` in MIMICDischargeEnv

The `curriculum_mode` parameter controls which complexity tier episodes are sampled from:

| Mode | Pool | Use case |
|------|------|----------|
| `random` | All hadm_ids | Default / general evaluation |
| `easy_only` | Home discharge, LOS≤4, no ICU, ≤5 diagnoses | Early training |
| `medium_only` | Intermediate complexity | Mid-phase training |
| `hard_only` | SNF/rehab/hospice/died, LOS>14, vent>24h, resistance, oliguria | Advanced training / Task 4 |
| `progressive` | Easy (eps 0-199) → Medium (200-499) → Random (500+) | Automated curriculum |

The complexity index is built once at `EpisodeBuilder.__init__()` by classifying all available admissions. It's stored as `{complexity: [hadm_id, ...]}` and accessed via `sample_by_complexity(tier)`.

For Task 4, hard_only is recommended since the 10-step ICU workflow is only meaningful for patients who actually had ICU stays with multi-system illness.

---

## 11. Multi-Agent / Multi-Step Design

### How the Environment Handles Long-Horizon Agents

The environment is designed to support three distinct long-horizon patterns:

#### Pattern 1: Information Gathering (Task 2)
The agent reasons about what data it needs, requests it, then acts. This mirrors a clinical decision-making pattern: "I don't know enough yet; let me order more tests."

```
reset() → partial obs
step(info_request: ["labs", "vitals"]) → enriched obs, reward=0
step(task2: {care_plan}) → final obs=None, reward=0.7
```

The key architectural feature: `_revealed_info` is a Set stored on the env instance that accumulates across steps. Each `_build_observation()` call consults this set to decide which fields to include. The agent's history of information requests is implicit in the growing observation.

#### Pattern 2: Sequential Clinical Workflow (Task 4)
The agent manages a patient through an admission, making different types of decisions at each step. Prior decisions affect later scoring (shaping) and the agent has explicit memory via `episode_history`.

```
reset() → step 0 obs (minimal: demographics + diagnoses)
step(task4: {triage_level}) → step 1 obs (now includes labs)
step(task4: {priority_labs, priority_consults}) → step 2 obs (now includes vitals)
...
step(task4: {final_note}) → reward=0.7, done=True
```

The `_shaping_log` dict persists across all 10 steps on the env instance. It's updated by the grader after each step call. At step 10, the grader reads this accumulated log to compute the composite reward.

#### Pattern 3: Iterative Refinement (Task 4 Revision)
The agent can revise earlier decisions after receiving new information. This is structurally different from most RL environments — the "state" includes a mutable history that can be edited:

```
step 5: receive microbiology results (now visible)
step 5 action: {antibiotic_strategy: "targeted", antibiotics: ["vancomycin"]}
step 6: receive fluid_balance
step 6 action: {
  fluid_strategy: "restrict_diuresis",
  revise_step: 5,
  revision: {antibiotic_strategy: "targeted", antibiotics: ["vancomycin", "pip-tazo"]}
}
```

The revision re-grades step 5 with the new content and overwrites the shaping log entry. This allows the agent to correct mistakes when it receives new information, at a small cost.

### Single-Process Limitation

The current server uses a single `MIMICDischargeEnv` instance, meaning only one episode can be active at a time. For multi-agent training at scale, either:
1. Run multiple server instances on different ports
2. Extend the server to support session-keyed env instances (e.g., `POST /reset` returns a `session_id` that's passed to `POST /step`)

The `/rollout` endpoint partially addresses this by accepting a complete list of actions and executing the full episode server-side in one request, which is safe for parallel training calls as long as each rollout call is atomic.

---

## 12. Training Pipeline

### Rollout Collection — `training/rollout_collector.py`

#### `format_observation(obs: Dict) → str`

The core function for converting a structured `Observation` dict to a text prompt for the LLM. It formats all visible fields into clearly labelled sections:

```
=== PATIENT CLINICAL SUMMARY ===
Task: ...
Hadm ID: ... | Subject: ... | Step: 0/1

--- DEMOGRAPHICS ---
Age: 68 | Gender: M
Admission type: EMERGENCY
LOS: 12.3 days | Complexity: hard

--- DIAGNOSES ---
  [N18.4] Chronic kidney disease, stage 4
  [I50.9] Heart failure, unspecified

--- VITALS ---
  Heart Rate: adm=92 dc=78 [58–118]
  Systolic BP: adm=145 dc=132 [98–185] [CRITICAL]

--- ABNORMAL LABS ---
  Creatinine: 3.2 [Abnormal]
  BNP: 1840 [Critical]

--- ACTION REQUIRED ---
JSON: {"task_id": 1, "task1": {"disposition": "...", "reasoning": "..."}}
Valid dispositions: home | home_with_services | snf | rehab | hospice | ama | expired | other

Respond ONLY with valid JSON matching the action schema above.
```

The prompt ends with an explicit instruction to respond only with JSON, which is critical for `_extract_json()` to parse the output reliably.

#### `_extract_json(text) → Optional[Dict]`

Three-pass extraction:
1. Direct `json.loads()` (handles clean JSON output)
2. Regex extraction of ` ```json ... ``` ` or ` ``` ... ``` ` code blocks
3. Greedy `{...}` pattern extraction (fallback for JSON embedded in prose)

Returns `None` if all passes fail. The caller substitutes `{"task_id": task_id}` (no-op action) on parse failure, which receives reward 0.0.

#### `RolloutCollector.collect(task_id, n_episodes) → Dataset`

Runs `n_episodes` full episodes (each potentially multi-step for Task 2/4) and saves results as a HuggingFace Dataset with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Formatted observation text |
| `response` | str | Raw LLM output |
| `reward` | float | Environment reward for this step |
| `partial` | str | JSON-encoded partial signals |
| `hadm_id` | int | Episode identifier |
| `task_id` | int | Task |
| `step_num` | int | Step within episode |
| `parse_ok` | bool | Whether action JSON was valid |
| `episode_idx` | int | Episode index within collection run |

---

### GRPO Training — `training/train_grpo.py`

#### What is GRPO?

Group Relative Policy Optimization (GRPO) is a variant of PPO adapted for language model fine-tuning. Instead of a value network, it uses the mean reward of a **group** of rollouts from the same prompt as a baseline:

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

This is more stable for LLMs than standard PPO because:
1. No value network needed (fewer parameters to train)
2. The baseline adapts naturally to the difficulty of each prompt
3. Works well with sparse rewards (the group variance captures relative performance)

`num_generations=4` means 4 independent rollouts per prompt per training step. The model learns to improve the high-reward rollouts relative to the group mean.

#### Model Loading

With Unsloth installed:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,    # QLoRA: 4-bit quantisation
)
model = FastLanguageModel.get_peft_model(model,
    r=16,                 # LoRA rank
    lora_alpha=16,        # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

Without Unsloth (fallback): standard `transformers` + `peft` LoraConfig with the same rank-16 configuration.

#### Reward Function

The `make_reward_fn()` factory returns a function that TRL calls after each generation batch:

```python
def reward_fn(prompts, responses, **kwargs) -> List[float]:
    task_id, noise_level, curriculum_mode = _curriculum(step_counter[0])
    for prompt, response in zip(prompts, responses):
        action_dict = _extract_json(response)
        env.reset(task_id=task_id, noise_level=noise_level, curriculum_mode=curriculum_mode)
        result = env.step(action_dict)
        rewards.append(result.reward)
    step_counter[0] += len(responses)
    return rewards
```

**Important**: For Tasks 1 and 3 (single-step), this is straightforward — each response gets a scalar reward. For Tasks 2 and 4 (multi-step), the reward function handles only a single step call, which means the TRL framework treats each step as an independent training example. Full-episode multi-step GRPO would require a custom rollout loop (the `/rollout` endpoint supports this use case).

#### Curriculum Phases

```python
def _curriculum(step):
    if step < 1000:   return 1, "clean",   "easy_only"
    if step < 3000:   return 2, "partial", "medium_only"
    return               3, "noisy",   "random"
```

The training loop refreshes the seed dataset at each evaluation checkpoint (`eval_every` steps) to match the current curriculum phase. This ensures the prompt distribution matches the task and noise level being trained on.

#### Dead-Gradient Detection

Monitors whether Task 1 reward is consistently 0.0 over the last 50 rollouts:

```python
zero_buf = deque(maxlen=50)  # stores (reward == 0.0) booleans

if task_id == 1 and len(zero_buf) == 50:
    if sum(zero_buf) / len(zero_buf) > 0.60:
        print("Dead gradient detected — halting")
        break
```

This catches the case where the model collapses to outputting invalid JSON or nonsensical dispositions, which produces all-zero rewards and provides no training signal. In this case, the LoRA adapter may need to be reset or the learning rate reduced.

---

## 13. Server API Layer

### File: `server/app.py`

FastAPI application with async request handling, JSON structured logging, and per-route latency metrics.

### Middleware Stack

1. **CORS**: All origins allowed (open hackathon environment)
2. **Request ID middleware**: Every request gets a UUID8 header `X-Request-ID` and timing in `X-Response-Time`
3. **Metrics middleware**: Updates `_Metrics` with per-route latency (ring buffer of last 500 per route) and status codes

### Metrics Collected

```python
class _Metrics:
    request_count:    Dict[str, int]           # per route
    error_count:      Dict[str, int]           # per route (4xx/5xx)
    latency_ms:       Dict[str, List[float]]   # ring buffer, last 500
    episode_count:    int
    total_reward:     float
    rewards_by_task:  Dict[int, List[float]]
    json_parse_attempts: int
    json_parse_success:  int                   # always = attempts (Pydantic validates)
    revision_count:   int                      # Task 4 revisions
    complexity_counts: Dict[str, int]          # episodes per complexity tier
```

Exposed at `GET /metrics` with p50/p95/p99 latency percentiles per route and per-task reward statistics.

### Episode History Buffer

A ring deque of the last 50 completed episodes:
```python
class _EpisodeHistory:
    MAX_HISTORY = 50
    _buf: Deque[Dict]  # appendleft (newest first)
```

Each entry: `{task_id, hadm_id, reward, steps, complexity, partial_signals, completed_at}`.

Accessible at `GET /history?n=10`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start new episode, returns Observation |
| POST | `/step` | Submit action, returns StepResult |
| POST | `/rollout` | Run full episode from action list |
| GET | `/health` | Server readiness + basic stats |
| GET | `/metrics` | Latency percentiles + reward stats |
| GET | `/history` | Last N completed episodes |
| GET | `/state` | Current episode ground truth (debug) |
| GET | `/tasks` | Task catalogue with scoring weights |
| GET | `/episodes` | Total count + random sample of hadm_ids |
| GET | `/complexity/{hadm_id}` | Complexity tier for specific admission |
| GET | `/episodes/by_complexity` | All hadm_ids grouped by tier |

### Error Handling

Pydantic validation errors on the action body return HTTP 422 with field-level detail. All unhandled exceptions return HTTP 500 with the request ID for log correlation. The `_require_env()` helper returns 503 if the environment is still initialising (MIMIC tables loading). The `_require_active()` helper returns 409 if `POST /step` is called without a prior `POST /reset`.

---

## 14. End-to-End Request Flow

### Task 1 (single step)

```
POST /reset {"task_id": 1}
  → EpisodeBuilder.get_episode(random_hadm_id, "clean")
  → _build_observation() → full Observation (no gating)
  → 200 OK {age: 68, diagnoses: [...], ...}

POST /step {"task_id": 1, "task1": {"disposition": "snf", "reasoning": "..."}}
  → DispositionGrader.grade()
      → normalize_mimic_location("SKILLED NURSING FACILITY") = "snf"
      → pred "snf" == true "snf" → score = 1.0
      → reasoning bonus: "ICU stay" in reasoning → +0.05 (clamped to 1.0)
  → StepResult(reward=1.0, done=True, partial_signals={disposition_exact: 1.0, ...})
  → 200 OK
```

### Task 2 (multi-step)

```
POST /reset {"task_id": 2, "curriculum_mode": "medium_only"}
  → Sample medium-complexity hadm_id
  → _build_observation() → partial obs (only demographics + top5 dx + LOS + complexity)
  → 200 OK {diagnoses: [...top5], medications: [], lab_flags: [], ...}

POST /step {"task_id": 2, "information_request": ["labs", "vitals"]}
  → _revealed_info updated: {"demographics", "labs", "vitals", ...}
  → _build_observation() → now includes lab_flags and vitals
  → StepResult(reward=0.0, done=False, observation={lab_flags: [...], vitals: [...]})
  → 200 OK

POST /step {"task_id": 2, "task2": {"follow_up_specialties": ["Nephrology"], ...}}
  → CarePlanGrader.grade()
      → step_num = 2 → efficiency discount = 1.0
      → specialty F1: expected={"nephrology"}, rec={"nephrology"} → F1=1.0
      → ...
  → StepResult(reward=0.72, done=True)
  → 200 OK
```

### Task 4 (10 steps)

```
POST /reset {"task_id": 4, "curriculum_mode": "hard_only"}
  → Sample hard-complexity hadm_id (ICU patient)
  → _build_observation() → minimal obs (step 0, only demos + diagnoses + icu_stays)

POST /step {"task_id": 4, "task4": {"triage_level": "icu"}}
  → Task4Grader.grade(action, episode, step_num=1, shaping_log={})
      → _grade_step_1() → score=1.0, scaled_reward=0.10
      → shaping_log["step1_reward"] = 0.10
      → Returns reward=0.0 (sparse)
  → episode_history appended: {step_num:1, action_summary:"Triaged as icu"}
  → _build_observation() → now includes lab_flags (step 2 threshold met)
  → StepResult(reward=0.0, done=False)

... (steps 2-9) ...

POST /step {"task_id": 4, "task4": {"final_note": "Patient is a 72-year-old..."}}
  → Task4Grader.grade(action, episode, step_num=10, shaping_log={...})
      → NoteGrader on final_note → base_reward=0.62
      → shaping_avg = mean([0.10/0.10, 0.12/0.15, ...]) = 0.74 (normalised)
      → raw = 0.62×0.60 + 0.74×0.40 = 0.668
      → note mentions "home" → consistency_bonus=0.10 (matches step8_predicted_dispo)
      → note mentions "nephrology" → trajectory_bonus=0.05 (>50% of step2 specialties)
      → no revisions → revision_cost=0.0
      → final = 0.668 + 0.10 + 0.05 = 0.818
  → StepResult(reward=0.818, done=True)
```

---

## 15. curl Test Reference

### Health check

```bash
curl http://localhost:7860/health
```

### Task 1 — full episode

```bash
# Reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "noise_level": "clean"}' | python3 -m json.tool

# Step
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 1,
    "task1": {
      "disposition": "home",
      "reasoning": "Patient is medically stable with no IV medications, ambulating independently, family support at home."
    }
  }' | python3 -m json.tool
```

### Task 2 — multi-step with information request

```bash
# Reset Task 2
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 2, "noise_level": "clean"}' | python3 -m json.tool

# Request labs and medications
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 2,
    "information_request": ["labs", "medications"]
  }' | python3 -m json.tool

# Submit care plan
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 2,
    "task2": {
      "follow_up_specialties": ["Cardiology", "Nephrology"],
      "medications_to_continue": ["metoprolol", "lisinopril", "furosemide"],
      "medications_to_discontinue": ["heparin", "insulin drip"],
      "key_instructions": [
        "Weigh yourself daily; call doctor if weight increases more than 2 lbs overnight",
        "Take all medications as prescribed; do not skip doses",
        "Fluid restriction: no more than 1.5 litres per day",
        "Return to ED if shortness of breath worsens or you develop chest pain",
        "Follow-up with Cardiology within 1 week"
      ],
      "reasoning": "Heart failure with preserved EF; creatinine elevated suggesting cardiorenal syndrome"
    }
  }' | python3 -m json.tool
```

### Task 3 — discharge note

```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 3}' > /dev/null

curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 3,
    "task3": {
      "discharge_note": "DISCHARGE SUMMARY\n\nPRINCIPAL DIAGNOSIS: Acute decompensated heart failure\n\nBRIEF HOSPITAL COURSE: This 68-year-old male with known systolic heart failure presented with a 3-day history of worsening dyspnea and bilateral lower extremity edema. He was admitted to the MICU for 2 days before transfer to the general medical floor. Hospital length of stay was 7 days. He received IV furosemide 80mg twice daily with significant diuresis. Echo showed EF 35%. BNP trended down from 2400 to 680. He was transitioned to oral diuretics prior to discharge.\n\nKEY PROCEDURES: Echocardiogram, right heart catheterisation\n\nDISCHARGE CONDITION: Stable, ambulating with assistance\n\nDISCHARGE DISPOSITION: Discharged home with home health services for daily weights and wound care\n\nDISCHARGE MEDICATIONS: Furosemide 40mg daily, Metoprolol succinate 50mg daily, Lisinopril 10mg daily, Spironolactone 25mg daily\n\nFOLLOW-UP INSTRUCTIONS: Follow up with Cardiology in 1 week. Daily weights — call if gain exceeds 2 lbs. Return to emergency department if shortness of breath worsens, chest pain develops, or systolic BP falls below 90."
    }
  }' | python3 -m json.tool
```

### Task 4 — first 3 steps of workflow

```bash
# Reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "curriculum_mode": "hard_only"}' | python3 -m json.tool

# Step 1: triage
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "task4": {"triage_level": "icu"}}' | python3 -m json.tool

# Step 2: labs + consults
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 4,
    "task4": {
      "priority_labs": ["creatinine", "bnp", "troponin", "lactate"],
      "priority_consults": ["Cardiology", "Nephrology", "Infectious Disease"]
    }
  }' | python3 -m json.tool

# Step 3: interventions
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 4,
    "task4": {
      "interventions": ["mechanical ventilation", "arterial line", "central line", "continuous renal replacement therapy"]
    }
  }' | python3 -m json.tool
```

### Utility endpoints

```bash
# Get complexity for a specific admission
curl http://localhost:7860/complexity/29079034

# List all episodes by complexity tier
curl http://localhost:7860/episodes/by_complexity | python3 -m json.tool

# Recent episode history
curl "http://localhost:7860/history?n=5" | python3 -m json.tool

# Metrics dashboard
curl http://localhost:7860/metrics | python3 -m json.tool

# Current episode state (ground truth — for debugging)
curl http://localhost:7860/state | python3 -m json.tool

# Task catalogue with scoring weights
curl http://localhost:7860/tasks | python3 -m json.tool

# Replay episode with pre-written actions
curl -s -X POST http://localhost:7860/rollout \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 1,
    "hadm_id": 29079034,
    "noise_level": "clean",
    "actions": [
      {"task_id": 1, "task1": {"disposition": "snf", "reasoning": "Long-stay ventilated patient needs skilled nursing"}}
    ]
  }' | python3 -m json.tool
```

### Pin a specific episode for reproducible evaluation

```bash
# Always test against the same patient
HADM=29079034
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d "{\"task_id\": 1, \"hadm_id\": $HADM, \"noise_level\": \"clean\"}" | python3 -m json.tool
```

---

*This document covers the complete technical design of MIMIC Discharge Planning v3.1.0. For the formal machine-readable spec, see `openenv.yaml`.*
