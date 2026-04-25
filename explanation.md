# MIMIC Discharge Planning — Full Technical Reference

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Layer — MIMIC-IV Episode Builder](#2-data-layer--mimic-iv-episode-builder)
3. [Observation Space](#3-observation-space)
4. [Environment Core — MIMICDischargeEnv](#4-environment-core--mimicdischargeenv)
5. [Task 1 — Discharge Disposition](#5-task-1--discharge-disposition)
6. [Task 2 — Care Plan Recommendation](#6-task-2--care-plan-recommendation)
7. [Task 3 — Discharge Note Generation](#7-task-3--discharge-note-generation)
8. [Task 4 — ICU Admission-to-Discharge Workflow](#8-task-4--icu-admission-to-discharge-workflow)
9. [Stochastic Observation Masking](#9-stochastic-observation-masking)
10. [GRPO Training Pipeline](#10-grpo-training-pipeline)
11. [Server API Reference](#11-server-api-reference)
12. [End-to-End Data Flow](#12-end-to-end-data-flow)
13. [curl Test Reference](#13-curl-test-reference)

---

## 1. System Overview

The environment is a **deterministic, server-backed RL environment** for clinical discharge planning, grounded in real patient records from MIMIC-IV Clinical Database Demo v2.2.  An agent plays the role of an attending physician making sequential discharge decisions.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent / LLM                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Server  (server/app.py)             │
│   POST /reset    POST /step    GET /health   GET /metrics   │
└───────────────────────────┬─────────────────────────────────┘
                            │ Python
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MIMICDischargeEnv  (environment/env.py)         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐│
│  │EpisodeBuilder│  │Task Graders  │  │Observation Builder ││
│  │(MIMIC-IV CSV)│  │T1 / T2 / T3  │  │Gating / Noise      ││
│  └──────────────┘  │/ T4          │  └────────────────────┘│
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- All graders are **fully deterministic** — no LLM judge, no randomness in scoring
- Every reward decomposes into clinically named partial signals
- Task 2 uses information gating (agent requests data in steps before committing)
- Task 4 has sparse reward (only step 10 scores; steps 1–9 accumulate shaping)
- Observation noise is injected stochastically at reset time (clean / partial / noisy)

---

## 2. Data Layer — MIMIC-IV Episode Builder

**File:** `environment/old_episode_builder.py`

Loads and joins MIMIC-IV CSV tables into structured episode dictionaries at startup.

### Source Tables

| Table | Usage |
|-------|-------|
| `admissions.csv` | hadm_id, admission_type, admission_location, discharge_location, ethnicity, subject_id |
| `patients.csv` | gender, anchor_age |
| `diagnoses_icd.csv` + `d_icd_diagnoses.csv` | ICD codes, descriptions, sequence numbers |
| `prescriptions.csv` | medication orders (drug, route, dose, start/stop time) |
| `pharmacy.csv` | active discharge meds (`pharmacy_active`), stopped meds (`pharmacy_stopped`) |
| `labevents.csv` + `d_labitems.csv` | lab results with abnormal flags |
| `microbiologyevents.csv` | culture organisms |
| `chartevents.csv` | vitals (HR, BP, SpO2, GCS, temp) |
| `procedureevents.csv` + `d_items.csv` | ICU procedures (ventilation hours, dialysis, lines) |
| `inputevents.csv` / `outputevents.csv` | fluid balance, net balance, oliguria flag |
| `icustays.csv` | ICU unit names, LOS in ICU |
| `emar.csv` / `emar_detail.csv` | medication administration records (`active_at_discharge` flag) |
| `drgcodes.csv` | DRG codes with severity and mortality scores |
| `transfers.csv` | care trajectory (ward path: ED → MICU → floor) |

### Complexity Tier Classification

Patients are classified into three tiers at build time:

**Easy:** Short LOS (≤ 10 days), no ICU stay, ≤ 12 diagnoses, plain home discharge.
All easy-tier patients discharge to plain `home` — no discrimination is needed, making them useless for training clinical reasoning.

**Hard:** Any of:
- LOS > 14 days
- Ventilation > 24 hours
- Hard discharge location: `SKILLED NURSING FACILITY`, `REHAB`, `HOSPICE`, `DIED`, `CHRONIC/LONG TERM ACUTE CARE`, `OTHER FACILITY`
- Resistant organisms in microbiology: MRSA, VRE, ESBL, KPC, CRE, MDR
- Oliguria flag

**Medium:** Everything in between — the most clinically informative tier.
- 56% home_with_services / 39% home
- Requires reading actual clinical features to distinguish

**Pool sizes (MIMIC-IV demo):**
- Easy: 21 patients — all plain HOME, constant function, useless for training
- Medium: 109 patients — real HOME vs HOME_WITH_SERVICES discrimination
- Hard: 103 patients — complex dispositions (SNF / hospice / expired)
- Total: 233 patients

### Fallback Rule

If any complexity tier has fewer than 10 unique patients, the environment falls back to sampling from the full 233-patient pool rather than halting.

---

## 3. Observation Space

Every `/reset` response is an `Observation` object.  All fields are populated from the episode dict built by EpisodeBuilder.

| Field | Type | Description |
|-------|------|-------------|
| `hadm_id` | int | Hospital admission ID |
| `task_id` | int | Active task (1–4) |
| `age` | int | Patient age at admission |
| `gender` | str | M / F |
| `admission_type` | str | EMERGENCY / ELECTIVE / URGENT / OBSERVATION ADMIT |
| `admission_location` | str | Emergency room / physician referral / transfer |
| `hospital_los_days` | float | Total hospital length of stay |
| `complexity` | str | easy / medium / hard |
| `diagnoses` | list[dict] | ICD code, description, sequence number — capped at 15 total |
| `pharmacy_active` | list[str] | **Drugs active at discharge — the ONLY valid source for `medications_to_continue`** |
| `pharmacy_stopped` | list[str] | **Drugs stopped during admission — the ONLY valid source for `medications_to_discontinue`** |
| `medications` | list[dict] | Full medication orders (drug, route, dose) — reference only |
| `lab_flags` | list[dict] | Abnormal lab results (name, flag: CRITICAL/HIGH/LOW/ABNORMAL, value) |
| `procedures` | list[str] | ICD procedure descriptions — capped at 10 |
| `icu_stays` | list[dict] | ICU unit name + LOS per stay |
| `icu_procedures` | dict | `ventilation_hours`, `has_dialysis`, `has_arterial_line`, `has_central_line` |
| `vitals` | list[dict] | HR, BP, SpO2, GCS Total, Temp — admission → discharge values |
| `drg_codes` | list[dict] | DRG code, description, severity_score (1–4), mortality_score (1–4) |
| `microbiology` | list[dict] | Culture organisms with susceptibility data |
| `fluid_balance` | dict | `net_balance_ml`, `fluid_overloaded` (bool), `oliguria` (bool) |
| `care_trajectory` | list[str] | Ordered care unit path (ED → MICU → floor) |
| `emar_summary` | list[dict] | Medication admin records with `active_at_discharge` flag |
| `discharge_orders` | dict | `discharge_planning_finalized` (bool), order types |

**Critical fields for grading:**
- `pharmacy_active` — only valid medication source for Tasks 2/3/4
- `vitals.GCS Total` — key end-of-life indicator for Task 1
- `diagnoses[*].icd_code` — specialty derivation for Task 2
- `drg_codes[*].mortality_score` — hospice trigger (score = 4) for Task 1
- `microbiology` — Infectious Disease specialty trigger for Task 2

### Task 2 Information Gating

Task 2 starts with only: `demographics`, `admission_type`, `hospital_los_days`, `complexity`, and **top 5 diagnoses only** (not all 15).

The agent must request additional fields before submitting:

```json
{"task_id": 2, "information_request": ["labs", "medications", "microbiology"]}
```

Requestable fields: `labs` · `vitals` · `medications` · `microbiology` · `fluid_balance`

After request, the full versions of those fields unlock in the observation.

### Task 4 Progressive Revelation

Task 4 reveals EHR fields by step number, simulating real-time ICU data arrival:

| Unlocks at step | Fields |
|----------------|--------|
| 1 | Demographics, admission_type, LOS, complexity, diagnoses, DRG codes |
| 2 | `lab_flags` |
| 3 | `vitals`, `icu_procedures` |
| 4 | `medications`, `emar_summary` |
| 5 | `microbiology` |
| 6 | `fluid_balance` |
| 7 | `care_trajectory` |
| 8 | `discharge_orders` |

---

## 4. Environment Core — MIMICDischargeEnv

**File:** `environment/env.py`

### reset()

```python
def reset(
    task_id: int = 1,
    hadm_id: Optional[int] = None,
    noise_level: str = "clean",
    curriculum_mode: str = "random",
) -> Observation
```

- Clamps `task_id` to [1, 4]
- If `hadm_id` is not provided, samples from the appropriate complexity pool based on `curriculum_mode`
- Resets internal state: `_step_num = 0`, `_cumulative_reward = 0.0`, `_shaping_log = {}`, `_revealed_info`
- Initializes Task 2 revealed fields to the default initial set
- Calls `EpisodeBuilder.get_episode(hadm_id, noise_level)` → applies noise masking → returns Observation

**Curriculum modes:**
- `easy_only` — sample from easy tier (21 patients)
- `medium_only` — sample from medium tier (109 patients)
- `hard_only` — sample from hard tier (103 patients)
- `random` — sample from all 233 patients
- `progressive` — episodes 0–199: easy, 200–499: medium, 500+: random

### step()

```python
def step(action: Action) -> StepResult
```

Returns `StepResult` = `(observation, reward, done, partial_signals, info)`

**Task 1 / Task 3:** Single-step. Grade immediately. `done = True`. Extra steps beyond `max_steps` incur a -0.10/step penalty.

**Task 2:**
1. If `action.information_request` is set → update `_revealed_info`, return updated Observation with `reward=0, done=False`
2. If `action.task2` is set → grade via `CarePlanGrader`, apply step efficiency discount, return `done=True`

**Task 4:**
1. Steps 1–9 → dispatch to step-specific grader, store score in `_shaping_log[step_num]`, return `reward=0, done=False`
2. Revision handling: if `action.revise_step` + `action.revision` set and revisions_used < 2 → re-grade that step, update `_shaping_log`, return `reward=0` (no-op, costs a step)
3. Step 10 → grade final note, compute composite reward: `0.60 × note + 0.40 × shaping_avg + bonuses − revision_cost`, return `done=True`

### TASK_CONFIG

```python
TASK_CONFIG = {
    1: {"max_steps": 1},
    2: {"max_steps": 4, "step_efficiency_discount": {1: 1.0, 2: 1.0, 3: 0.85, 4: 0.70}},
    3: {"max_steps": 1},
    4: {"max_steps": 10},
}
```

---

## 5. Task 1 — Discharge Disposition

**File:** `environment/tasks/task1_disposition.py`  
**Difficulty:** Easy · 1 step · All EHR fields visible

### Action Format

```json
{
  "task_id": 1,
  "task1": {
    "disposition": "home_with_services",
    "reasoning": "Patient requires IV antibiotics and wound care post-op."
  }
}
```

### Valid Disposition Values

| Value | Clinical Meaning |
|-------|-----------------|
| `home` | Fully independent, no professional follow-up needed |
| `home_with_services` | Visiting nurse, home PT, IV antibiotics, wound care |
| `snf` | Skilled nursing facility — 24h nursing + ongoing medical needs |
| `rehab` | Inpatient rehabilitation — intensive PT/OT, medically stable |
| `hospice` | Terminal prognosis, comfort-focused care only |
| `ama` | Left against medical advice |
| `expired` | Patient died during this admission |
| `other` | Transfer to acute hospital or psychiatric unit |

### Scoring Formula

```
score = tier_score + reasoning_bonus
      = [1.0 | 0.50 | 0.25 | 0.0] + [0 or 0.05]
      clamped to [0.0, 1.0]
```

**Tier scoring:**

| Match level | Score |
|------------|-------|
| Exact canonical match | 1.00 |
| Same broad group | 0.50 |
| Clinically adjacent | 0.25 |
| No match | 0.00 |

**Broad groups:**

| Group | Members |
|-------|---------|
| `community` | home, ama |
| `community_plus` | home_with_services |
| `facility` | snf, rehab |
| `end_of_life` | hospice, expired |
| `other` | other |

**Adjacency table:**

| Predicted | Adjacent to |
|-----------|-------------|
| home | home_with_services, ama |
| home_with_services | home, snf |
| snf | home_with_services, rehab, other |
| rehab | snf, other |
| hospice | snf, home_with_services, expired |
| expired | hospice |
| ama | home |
| other | snf, rehab |

### Reasoning Bonus (+0.05)

Awarded when reasoning string:
- Is ≥ 20 characters, AND any of:
  - Contains ≥1 clinical keyword (functional, ambulation, therapy, nursing, icu, cardiac, creatinine, ejection fraction, renal, ventil, dialysis, fracture, wound, infection, malignant, hospice, terminal, palliative…)
  - Contains a number and is > 30 chars
  - Contains "icd", "dx", "diagnosis", "condition", or "history"
  - Is ≥ 50 chars (fallback)

### MIMIC Discharge Location Normalization

The grader normalizes raw MIMIC discharge_location strings to canonical values using ordered pattern matching:

| Canonical | MIMIC strings matched (order matters) |
|-----------|---------------------------------------|
| `home_with_services` | HOME HEALTH CARE, HOME HEALTH, HOME WITH SERVICE, HOME WITH AIDE, HOME WITH VNA, ASSISTED LIVING |
| `snf` | SKILLED NURSING FACILITY, SNF, LONG TERM CARE, CHRONIC CARE, EXTENDED CARE, NURSING HOME, SUB-ACUTE |
| `rehab` | REHABILITATION, REHAB FACILITY, INPATIENT REHAB |
| `hospice` | HOSPICE-MEDICAL FACILITY, HOSPICE-HOME, COMFORT CARE ONLY |
| `ama` | AGAINST MEDICAL ADVICE, LEFT AMA, ELOPED |
| `expired` | DIED IN ICU, DIED, EXPIRED, DECEASED |
| `other` | ACUTE HOSPITAL, TRANSFER, PSYCH FACILITY, CORRECTIONAL, GROUP HOME |
| `home` | DISCHARGED TO HOME, RETURNED HOME, HOME, SELF |

---

## 6. Task 2 — Care Plan Recommendation

**File:** `environment/tasks/task2_careplan.py`  
**Difficulty:** Medium · ≤4 steps (gated) · Efficiency-discounted

### Action Format (submission step)

```json
{
  "task_id": 2,
  "task2": {
    "follow_up_specialties": ["Cardiology", "Nephrology"],
    "medications_to_continue": ["Lisinopril 10mg", "Metoprolol 25mg"],
    "medications_to_discontinue": ["Heparin drip"],
    "key_instructions": [
      "Weigh daily; call if weight increases > 2 lbs in 24 hours",
      "Take lisinopril at the same time each morning",
      "Call 911 if chest pain or shortness of breath",
      "Follow low-sodium diet < 2g/day",
      "Follow-up with cardiology within 1 week"
    ]
  }
}
```

### Scoring Formula

```
raw = 0.35 × specialty_f1
    + 0.25 × medication_f1
    + 0.25 × instruction_quality
    + 0.15 × discontinue_accuracy
    − hallucination_penalty   (max 0.10)
    − ghost_specialty_penalty (max 0.10)

final = max(0.0, min(1.0, raw)) × step_efficiency_discount
```

**Step efficiency discount:**

| Steps used | Multiplier |
|-----------|-----------|
| 1–2 | 1.00× |
| 3 | 0.85× |
| 4+ | 0.70× |

**Optimal strategy:** `information_request: ["labs", "medications", "microbiology"]` on step 1 → submit care plan on step 2.

### ICD → Specialty Mappings

**ICD-10 (first letter):**

| Prefix | Specialty |
|--------|-----------|
| A, B | Infectious Disease |
| C | Oncology |
| D | Hematology, Oncology |
| E | Endocrinology |
| F | Psychiatry |
| G | Neurology |
| H | Ophthalmology |
| I | Cardiology |
| J | Pulmonology |
| K | Gastroenterology |
| L | Dermatology |
| M | Rheumatology |
| N | Nephrology |
| O | Obstetrics |
| Q | Genetics |
| S | Trauma Surgery |
| T | Toxicology |

**ICD-9 (numeric range):**

| Range | Specialty |
|-------|-----------|
| 001–139 | Infectious Disease |
| 140–239 | Oncology |
| 240–279 | Endocrinology |
| 290–319 | Psychiatry |
| 320–389 | Neurology |
| 390–459 | Cardiology |
| 460–519 | Pulmonology |
| 520–579 | Gastroenterology |
| 580–629 | Nephrology |
| 630–677 | Obstetrics |
| 680–709 | Dermatology |
| 710–739 | Rheumatology |
| 800–999 | Trauma Surgery |

**Microbiology → Specialty:**  
Any of `staphylococcus`, `streptococcus`, `klebsiella`, `pseudomonas`, `escherichia`, `candida`, `aspergillus`, `enterococcus`, `mrsa`, `vre`, `clostridioid` → **Infectious Disease**

**HCPCS procedure keywords:** `cardiovascular/cardiac` → Cardiology · `pulmonary/respiratory` → Pulmonology · `dialysis/renal` → Nephrology · `oncol/chemo` → Oncology · `endoscop/colono` → Gastroenterology · `orthop/joint` → Orthopedics

### Specialty F1

- Fuzzy word-overlap matching between predicted and expected specialty sets
- Predictions capped at 8 specialties
- If expected set is empty → returns (1.0, 1.0, 1.0)
- `recall = matched_expected / len(expected)`
- `precision = matched_pred / len(predicted)`
- `f1 = 2 × recall × precision / (recall + precision)`

### Ghost Specialty Penalty

Fires when a predicted specialty has no supporting ICD code in the patient record.

- Penalty: `min(0.10, len(ghosts) × 0.05)` — max 2 unsupported specialties = full 0.10 penalty
- Skipped for: "primary care", "general medicine", "hospitalist"
- Bypass: if ANY abnormal lab flag exists, general medical specialties get a pass

### Medication F1 and Hallucination

Drug stem extraction: first token ≥4 chars, else first 4 chars of the drug name.

- Known sources: `prescriptions` + `pharmacy_active` + `emar_summary`
- Drug is hallucinated ONLY if absent from **all** sources
- Recall computed against top 5 drugs from episode
- `hallucination_penalty = min(0.10, halluc_rate × 0.10)`

### Instruction Quality Score

Filters and scores up to 10 instructions across 6 clinical categories:

| Category | Trigger keywords | Quality keywords |
|----------|-----------------|-----------------|
| activity | activity, exercise, walk, mobilize | daily, week, minute, miles, steps |
| diet | diet, sodium, fluid, calorie, protein | gram, mg, litre, low, restrict, avoid |
| medication | medication, medicine, drug, take, dose | daily, twice, morning, mg, tablet |
| follow_up | follow, appointment, return, clinic, call | week, day, month, doctor, specialist |
| warnings | call, emergency, warning, seek, if | chest, breath, pain, fever, bleeding |

`score = categories_hit / 6 + unique_bonus` where `unique_bonus = min(0.1, len(instructions) / 50)`

---

## 7. Task 3 — Discharge Note Generation

**File:** `environment/tasks/task3_note.py`  
**Difficulty:** Hard · 1 step · Full EHR visible · Min 300 words

### Action Format

```json
{
  "task_id": 3,
  "task3": {
    "discharge_note": "PRINCIPAL DIAGNOSIS:\n1. Nonrheumatic aortic valve stenosis...\n\nBRIEF HOSPITAL COURSE:\nThe patient was admitted for 12.8 days...\n\n[7 required sections]"
  }
}
```

### Required Sections (in order)

1. PRINCIPAL DIAGNOSIS
2. BRIEF HOSPITAL COURSE *(must state LOS in days numerically, e.g. "admitted for 12.8 days")*
3. KEY PROCEDURES PERFORMED
4. DISCHARGE CONDITION
5. DISCHARGE DISPOSITION *(must use one of 6 exact canonical phrases)*
6. DISCHARGE MEDICATIONS *(exact drug names from `pharmacy_active` only)*
7. FOLLOW-UP INSTRUCTIONS

**Required disposition phrases (verbatim):**

| Disposition | Required phrase |
|-------------|----------------|
| home | "The patient was discharged home." |
| home_with_services | "The patient was discharged home with home health services." |
| snf | "The patient was transferred to a skilled nursing facility." |
| rehab | "The patient was transferred to inpatient rehabilitation." |
| hospice | "The patient was transitioned to hospice care." |
| expired | "The patient expired during this hospitalization." |

### Scoring Formula

```
raw = 0.30 × diagnosis_coverage
    + 0.20 × disposition_accuracy
    + 0.20 × medication_f1
    + 0.15 × los_accuracy
    + 0.10 × structure_score
    + 0.05 × information_density

halluc_penalty   = min(0.15, hallucination_rate × 0.15)
followup_penalty = 0.05  (if discharge_planning_finalized=True but no follow-up section)

final = max(0.0, raw − halluc_penalty − followup_penalty)
```

### 1. Diagnosis Coverage (0.30)

- Top 5 diagnoses extracted from `long_title`
- Keywords: words ≥5 chars, excluding 17 clinical stopwords (`unspecified`, `without`, `chronic`, `acute`, `disease`, `disorder`, etc.)
- Up to 4 keywords per diagnosis
- Sentence must be ≥5 words to count
- **Anti-stuffing:** keyword density > 0.08 → coverage halved
- **Anti-catch-all:** sentence matching ≥3 diagnoses simultaneously → discarded (vague)

### 2. Disposition Accuracy (0.20)

- 1.0 → synonym for true disposition present in note
- 0.3 → generic "discharg" keyword present (note mentions discharge but no specific location)
- 0.0 → no match

### 3. Medication F1 (0.20)

- Drug suffix regex detects 40+ suffixes: `mab`, `nib`, `pril`, `sartan`, `olol`, `statin`, `mycin`, `cillin`, `oxacin`, `cycline`, `azole`, `prazole`, etc.
- Context regex also matches: "medication X", "dose of X", "prescribed X"
- True positives: detected drugs matching `prescriptions`
- False positives: detected drugs NOT in `prescriptions` OR `emar_summary`
- `hallucination_rate = fp / (tp + fp)` → penalty = `min(0.15, rate × 0.15)`

### 4. LOS Accuracy (0.15)

- Tolerance = `max(1, round(LOS × 0.25))` days
- 1.0 → LOS number within tolerance of actual
- 0.3 → note mentions LOS-related keyword ("days", "hospital stay", "admitted for") without valid number
- 0.0 → no LOS reference

### 5. Structure Score (0.10)

6 required section types, each detected by keyword patterns:

```
word_count < 100 → score = 0.0
base = sections_present / 6
length_factor = min(1.0, log₂(word_count / 100) / log₂(5))
score = 0.70 × base + 0.30 × length_factor
```

### 6. Information Density (0.05)

Type-Token Ratio computed in 100-word sliding windows, penalized for repeated sentences (≥70% word overlap):

```
density = mean_ttr − duplicate_ratio
score = min(1.0, max(0.0, (density − 0.30) / 0.45))
```

---

## 8. Task 4 — ICU Admission-to-Discharge Workflow

**File:** `environment/tasks/task4_workflow.py`  
**Difficulty:** Very Hard · 10 steps · Sparse reward (step 10 only) · Hard patients only

### Sparse Reward Architecture

Steps 1–9 return `reward = 0`.  Each step's score is stored in `_shaping_log`.  Step 10 combines all of them into a single composite reward.

The agent sees progressively revealed EHR data (see §3) — it doesn't have full information until step 8+.

### Step-by-Step Breakdown

#### Step 1 — Acuity Triage (max 0.10)

```json
{"task_id": 4, "task4": {"triage_level": "icu"}}
```

Valid levels: `icu` | `stepdown` | `floor`

Classification logic from `care_trajectory` and ICU stay data:
- ICU keywords: micu, sicu, ccu, cvicu, tsicu, neuro icu, burn icu, cardiac vascular, trauma sicu
- Stepdown keywords: stepdown, step-down, msicu, intermediate, imu, progressive care

Scoring:
- Exact match → 1.0 × 0.10
- ICU/stepdown confusion → 0.5 × 0.10
- Floor/stepdown confusion → 0.5 × 0.10
- Wrong → 0.0

#### Step 2 — Priority Labs + Specialist Consults (max 0.15)

```json
{"task_id": 4, "task4": {
  "priority_labs": ["CBC", "BMP", "LFTs"],
  "priority_consults": ["Cardiology", "Nephrology"]
}}
```

Lab categories scored (from `lab_flags` with CRITICAL/HIGH/LOW/ABNORMAL):
- renal: creatinine, BUN, potassium, sodium
- cardiac: troponin, BNP, pro-BNP
- hepatic: AST, ALT, bilirubin, albumin
- hematology: hemoglobin, platelet, INR
- infection: WBC, lactate, procalcitonin
- metabolic: glucose, calcium, phosphorus, magnesium

Consult specialties derived same way as Task 2 (ICD-10/ICD-9 prefix mapping).

```
lab_score  = hits / expected_labs   (0.8 if no expected labs)
spec_score = hits / expected_specs  (0.8 if no expected specs)
score = (0.5 × lab_score + 0.5 × spec_score) × 0.15
```

#### Step 3 — Interventions (max 0.15)

```json
{"task_id": 4, "task4": {
  "interventions": ["intubation", "mechanical ventilation", "arterial line"]
}}
```

Needed interventions derived from episode:
- `ventilation_hours > 0` → needs one of: intubat, mechanical ventil, vent
- `has_arterial_line` → needs one of: arterial line, a-line, radial art
- `has_dialysis` → needs one of: dialysis, crrt, hemodialysis, cvvh
- `fluid_overloaded` → needs one of: fluid bolus, volume resuscitation, iv fluid

```
score = (matched_needed / total_needed) × 0.15
if no interventions needed → 0.10 (default credit)
```

#### Step 4 — High-Risk Medication Identification (max 0.10)

```json
{"task_id": 4, "task4": {"high_risk_medications": ["heparin", "propofol"]}}
```

High-risk drug sets:
- Anticoagulants: heparin, warfarin, enoxaparin, apixaban, rivaroxaban, dabigatran
- Vasopressors: norepinephrine, epinephrine, dopamine, vasopressin, phenylephrine, dobutamine
- Sedatives: propofol, midazolam, fentanyl, lorazepam, ketamine, dexmedetomidine

```
F1 = 2 × (tp / present) × (tp / flagged) / sum
score = F1 × 0.10
if no high-risk drugs present → 0.10 if agent unflagged, 0.06 if agent flagged anyway
```

#### Step 5 — Antibiotic Stewardship (max 0.10)

```json
{"task_id": 4, "task4": {
  "antibiotic_strategy": "targeted",
  "antibiotics": ["vancomycin", "piperacillin-tazobactam"]
}}
```

Valid strategies: `none` | `targeted` | `broad` | `empiric` | `culture-directed` | `prophylaxis`

Scoring:
```
if no organisms detected:
  "none" → 1.0; "broad"/"empiric" → 0.4; else → 0.2
if organisms detected:
  "targeted"/"culture-directed" → 0.5 + 0.5 × (matched_antibiotics / organisms)
  "broad"/"empiric" → 0.3
  "none" → 0.0
  else → 0.2
resistant_organisms_bonus: +0.02 if MRSA/VRE/ESBL present and identified
score = min(1.0, base + bonus) × 0.10
```

#### Step 6 — Fluid Strategy (max 0.08)

```json
{"task_id": 4, "task4": {"fluid_strategy": "restrict_diuresis"}}
```

Valid strategies: `restrict_diuresis` | `aggressive_resuscitation` | `maintain`

Correct strategy derivation:
- `fluid_overloaded = True` → correct = `restrict_diuresis`
- `oliguria = True AND net_balance < 0` → correct = `aggressive_resuscitation`
- Otherwise → correct = `maintain`

```
exact match → 1.0
correct requires action but predicted "maintain" → 0.4
wrong → 0.0
score = score × 0.08
```

#### Step 7 — ICU-to-Stepdown Readiness (max 0.08)

```json
{"task_id": 4, "task4": {
  "ready_for_stepdown": false,
  "barriers": ["mechanical ventilation", "hemodynamic instability"]
}}
```

Barrier categories detected from episode:
- vent: `ventilation_hours > 0`
- renal: `oliguria` or `has_dialysis`
- hemodynamic: vasopressors in active medications

True readiness: `los_days > 5 AND not on vent AND not on vasopressors`

```
readiness_score = 1.0 if pred_ready == true_ready, else 0.2
barrier_score = matched_barriers / actual_barriers (1.0 if no barriers)
score = (0.6 × readiness_score + 0.4 × barrier_score) × 0.08
```

#### Step 8 — Discharge Disposition + LOS Estimate (max 0.10)

```json
{"task_id": 4, "task4": {
  "predicted_disposition": "snf",
  "los_remaining_days": 3.0
}}
```

```
disposition:
  exact → 0.06; same broad group → 0.03; adjacent → 0.015; wrong → 0.0

LOS error = |predicted − actual_remaining|:
  ≤ 2 days → 0.04; ≤ 5 days → 0.02; > 5 days → 0.0

score = disposition_score + los_score
```

#### Step 9 — Medication Reconciliation (max 0.10)

```json
{"task_id": 4, "task4": {
  "medications_to_continue": ["Lisinopril 10mg", "Metoprolol 25mg"]
}}
```

Active discharge medications sourced from `emar_summary` where `active_at_discharge = True`.

Drug stem matching: first 4+ char token or first 6 chars.

```
F1 = 2 × recall × precision / (recall + precision)
recall = matched_active / total_active
score = F1 × 0.10
if no active meds in episode → 0.07 fixed credit
```

#### Step 10 — Final Discharge Note (composite reward)

```json
{"task_id": 4, "task4": {
  "final_note": "PRINCIPAL DIAGNOSIS:\n...[full note with all 7 sections]..."
}}
```

```
note_score  = Task3Grader.grade(final_note, episode)
shaping_avg = mean([shaping_log[i] / _STEP_MAX[i] for i in 1..9])

raw = 0.60 × note_score + 0.40 × shaping_avg

consistency_bonus = +0.10 if step8_predicted_dispo found in final_note
trajectory_bonus  = +0.05 if ≥50% of step2_expected_specialties appear in final_note
revision_cost     = −0.02 × revisions_used  (max 2 revisions)

final = min(1.0, max(0.0, raw + bonuses − revision_cost))
```

### Revision Mechanism

The agent can revise any step 1–9 decision (max 2 revisions total):

```json
{
  "task_id": 4,
  "task4": {"revise_step": 3, "revision": {"interventions": ["dialysis", "arterial line"]}}
}
```

Each revision: returns `reward=0, done=False`, consumes one revision slot, updates `_shaping_log`.

### Step Max Contributions

| Step | Max contribution | Clinical focus |
|------|-----------------|----------------|
| 1 | 0.10 | Triage acuity |
| 2 | 0.15 | Lab + consult prioritization |
| 3 | 0.15 | Intervention selection |
| 4 | 0.10 | High-risk med identification |
| 5 | 0.10 | Antibiotic stewardship |
| 6 | 0.08 | Fluid management |
| 7 | 0.08 | Stepdown readiness |
| 8 | 0.10 | Disposition + LOS forecasting |
| 9 | 0.10 | Medication reconciliation |
| 10 | 1.00 (cap) | Final note (composite) |

---

## 9. Stochastic Observation Masking

At `/reset`, the `noise_level` parameter controls how much EHR data is corrupted before presenting to the agent.

| Level | Description |
|-------|-------------|
| `clean` | Full data — all fields present and accurate |
| `partial` | 30% of lab results dropped; some medication fields truncated; minor value perturbations |
| `noisy` | Diagnosis sequence numbers scrambled; up to 50% of labs dropped; medication routes/doses may be wrong; false abnormal flags injected |

**Training curriculum noise assignment:**
- Task 1 / Task 2 → `clean` (clear signal needed for early learning)
- Task 3 / Task 4 → `partial` (introduces real-world incompleteness without destroying all signal)

---

## 10. GRPO Training Pipeline

**File:** `training/train_grpo.py`

### Model Configuration

- Model: Qwen/Qwen2.5-3B-Instruct (bfloat16)
- Adapter: LoRA r=16, alpha=32, dropout=0.05, target modules = all linear layers
- Max sequence length: 2560 tokens (prompt 1536 + completion 768 + buffer)
- Loaded via 4-bit NF4 quantization (bitsandbytes) if available

### GRPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_generations` | 8 | Min for non-degenerate GRPO advantage estimation |
| `batch_size` | 2 | L4 24GB VRAM limit |
| `grad_accum` | 8 | Effective batch = 16 |
| `learning_rate` | 5e-6 | Conservative for small RL budget |
| `beta` | 0.04 | KL penalty — prevents policy from collapsing away from base model diversity |
| `top_entropy_quantile` | 0.8 | Drop bottom 20% entropy completions — prunes mode-collapsed outputs |
| `temperature` | 1.3 (T1) / 1.1 (T2) / 0.9 (T3/T4) | Higher early to escape hospice collapse |
| `warmup_steps` | 10 | Short warm-up per chunk |

### Curriculum (7-hour L4 budget)

| Phase | Steps | Task | Patient pool | Seed N |
|-------|-------|------|-------------|--------|
| 1 | 0–199 | Disposition | medium_only (109) | 220 |
| 2 | 200–349 | Care Plan | medium_only (109) | 220 |
| 3 | 350–449 | Discharge Note | random (all 233) | 466 |
| 4 | 450–549 | ICU Workflow | hard_only (103) | 210 |

**Seed auto-scaling:** `seed_n = max(floor, pool_size × 2)` — ensures every unique patient appears at least twice per chunk.

### Chunk Loop

Every `eval_every=50` steps:
1. Determine `(task_id, noise_level, curriculum_mode)` from `_curriculum(global_step)`
2. Build fresh seed dataset (new env resets, stores `hadm_id` in each row)
3. Build reward function closure with task locked
4. Instantiate GRPOTrainer with new dataset + reward function
5. Train for 50 steps
6. Save model checkpoint + `train_state.json`
7. Check dead-gradient stop conditions

### hadm_id Coupling (Critical Fix)

Each seed dataset row stores the `hadm_id` of the patient used in the prompt:

```python
rows.append({
    "prompt": [...],
    "hadm_id": str(obs.get("hadm_id", "")),
})
```

The reward function pins `/reset` to that specific patient before scoring:

```python
reset_body["hadm_id"] = int(hadm_id[i])
_env_post(env_url, "/reset", reset_body)
# ... then step with the model's action
```

Without this, the reward scores a randomly sampled patient — completely decoupling the reward from the prompt the model saw. This was the primary cause of near-zero learning in early runs.

### Reward Function

```python
reward = alpha × env_score + (1 - alpha) × format_score
```

- Task 1: α = 0.80 (env), 0.20 (format)
- Task 2: α = 0.85, 0.15
- Task 3: α = 0.90, 0.10
- Task 4: α = 0.90, 0.10

Format score = 1.0 if valid JSON parseable as the correct task schema, else 0.0.

**Task 4 reward function:** Runs 9 default advance actions (one per step) to reveal all EHR data before scoring the model's `final_note` at step 10.

### Dead-Gradient Guard

Training halts if the zero-reward rate for the current task exceeds its threshold over the last 50 rollouts:

| Task | Halt threshold |
|------|---------------|
| T1 | 80% zero-reward |
| T2 | 90% zero-reward |
| T3 | 90% zero-reward |
| T4 | 90% zero-reward |

Zero is defined as reward < 0.12 (above wrong-task-key floor of 0.03, below minimum valid same-task reward of 0.15).

### Training Results Summary

| Task | Steps | Peak reward | Stabilized at | Key observation |
|------|-------|-------------|---------------|-----------------|
| Disposition | 0–199 | **0.73** | — | Climbs steadily from 0.24; model learns HOME vs HWS |
| Care Plan | 200–349 | — | **~0.60** | Fast stabilization; specialty + med F1 both contributing |
| Discharge Note | 350–435 | — | **~0.45** | High variance; longer outputs harder to optimize |
| **Overall** | — | — | **0.517** | Mean across all rollouts |

Parse rate: ≥95% through Tasks 1–2; ~85% in Task 3.
Zero-reward rate: <2% for T1/T2; spikes to 53% at T3 start, recovers to ~30%.

---

## 11. Server API Reference

**File:** `server/app.py`  
**Base URL:** `http://localhost:7860`

### Core RL Loop

#### POST /reset

```json
{
  "task_id": 1,
  "hadm_id": null,
  "noise_level": "clean",
  "curriculum_mode": "random"
}
```

Returns: full `Observation` JSON  
Errors: 400 (invalid task_id) · 503 (not ready)

#### POST /step

```json
{
  "task_id": 1,
  "task1": {"disposition": "home_with_services", "reasoning": "..."},
  "information_request": null
}
```

Returns: `StepResult` = `{observation, reward, done, partial_signals, info}`  
Errors: 409 (no active episode) · 422 (Pydantic validation failed) · 503 (not ready)

#### POST /rollout

Run a full multi-step rollout in one call:

```json
{
  "task_id": 2,
  "actions": [
    {"task_id": 2, "information_request": ["labs", "medications"]},
    {"task_id": 2, "task2": {"follow_up_specialties": [...], ...}}
  ],
  "hadm_id": null,
  "noise_level": "clean"
}
```

Returns: `{task_id, hadm_id, total_reward, n_steps, trajectory: [...]}`

### Observability Endpoints

#### GET /health

```json
{
  "status": "ok",
  "ready": true,
  "uptime_seconds": 3600.5,
  "episodes_available": 233,
  "env_active": true,
  "total_episodes_run": 1042,
  "current_task": 1,
  "current_hadm_id": 27703517
}
```

#### GET /metrics

Returns per-route latency percentiles (p50/p95/p99), per-task reward stats, JSON parse success rate, revision count, complexity distribution.

#### GET /history?n=10

Returns last N completed episodes with full reward breakdowns (max 50).

#### GET /state

Debug endpoint — returns ground truth: `current_task_id`, `current_hadm_id`, `step_num`, `last_reward`, `ground_truth: {true_disposition, active_medications, ...}`

### Metadata Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /tasks` | All 4 tasks with schemas, scoring weights, difficulty |
| `GET /episodes?sample=5` | Total count + sample hadm_ids |
| `GET /episodes/by_complexity` | All hadm_ids grouped by easy/medium/hard |
| `GET /complexity/{hadm_id}` | Complexity tier for a specific admission |
| `GET /docs` | Interactive Swagger UI |

### Request Tracking

Every request gets an `X-Request-ID` header (8-char UUID) and `X-Response-Time` (ms).  Server logs: `method path status latency req_id`.

---

## 12. End-to-End Data Flow

```
1. Server startup
   ├── EpisodeBuilder loads all MIMIC CSV tables (~2s)
   ├── Classifies all 233 admissions into easy/medium/hard
   └── Initializes MIMICDischargeEnv with 4 task graders

2. POST /reset (task_id=2, noise_level="clean", curriculum_mode="medium_only")
   ├── Sample hadm_id from medium tier pool (109 patients)
   ├── EpisodeBuilder.get_episode(hadm_id) → builds episode dict
   ├── Apply noise masking (clean → no change)
   ├── _build_observation() → Task 2: show only demographics + 5 diagnoses
   └── Return Observation

3. POST /step (information_request: ["labs", "medications", "microbiology"])
   ├── Update _revealed_info: unlock lab_flags, pharmacy_active/stopped, microbiology
   ├── _build_observation() → now includes labs, meds, organisms
   └── Return updated Observation (reward=0, done=False)

4. POST /step (task2: {specialties, medications, instructions})
   ├── CarePlanGrader.grade(action, episode)
   │   ├── _specialty_f1() → compare predicted vs ICD-derived expected
   │   ├── _medication_f1_and_halluc() → stem-match against pharmacy_active
   │   ├── _instruction_quality() → score 6 categories
   │   ├── _discontinue_accuracy() → against pharmacy_stopped
   │   ├── _ghost_specialty_penalty() → unsupported specialties
   │   └── Apply step 2 efficiency (1.0×)
   └── Return StepResult (reward=0.62, done=True, partial_signals={...})

5. Training (GRPO)
   ├── build_seed_dataset() → 220 resets of medium_only, store hadm_id
   ├── GRPOTrainer generates 8 completions per prompt
   ├── reward_fn() → reset to same hadm_id, score each completion
   ├── GRPO computes group-relative advantages
   ├── Backprop through LoRA weights only
   └── Checkpoint after every 50 steps
```

---

## 13. curl Test Reference

### Health check
```bash
curl http://localhost:7860/health
```

### Task 1 — Disposition
```bash
# Reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "noise_level": "clean"}' | jq .

# Submit
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "task1": {"disposition": "home_with_services", "reasoning": "Patient requires wound care."}}' | jq .
```

### Task 2 — Care Plan (2-step)
```bash
# Reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 2, "noise_level": "clean", "curriculum_mode": "medium_only"}' | jq .

# Step 1: Request labs + medications
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 2, "information_request": ["labs", "medications", "microbiology"]}' | jq .

# Step 2: Submit care plan
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 2,
    "task2": {
      "follow_up_specialties": ["Cardiology", "Nephrology"],
      "medications_to_continue": ["Lisinopril 10mg", "Metoprolol 25mg"],
      "medications_to_discontinue": ["Heparin drip"],
      "key_instructions": [
        "Weigh daily; call if > 2 lbs gain in 24 hours",
        "Low-sodium diet < 2g/day",
        "Take Lisinopril every morning",
        "Follow up with Cardiology within 1 week",
        "Return to ED if chest pain or worsening shortness of breath"
      ]
    }
  }' | jq .
```

### Task 3 — Discharge Note
```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 3}' | jq .

curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": 3,
    "task3": {
      "discharge_note": "PRINCIPAL DIAGNOSIS:\n1. Nonrheumatic aortic valve stenosis\n\nBRIEF HOSPITAL COURSE:\nThe patient was admitted for 12.8 days and underwent percutaneous aortic valve replacement...\n\nKEY PROCEDURES PERFORMED:\nPercutaneous aortic valve replacement (TAVR).\n\nDISCHARGE CONDITION:\nStable, ambulating independently.\n\nDISCHARGE DISPOSITION:\nThe patient was discharged home with home health services.\n\nDISCHARGE MEDICATIONS:\nAspirin 81mg daily, Clopidogrel 75mg daily.\n\nFOLLOW-UP INSTRUCTIONS:\nFollow up with Cardiology within 1 week. Return to ED if chest pain, shortness of breath, or fever > 38.5C."
    }
  }' | jq .
```

### Task 4 — ICU Workflow (first 3 steps)
```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "curriculum_mode": "hard_only"}' | jq .

# Step 1: Triage
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "task4": {"triage_level": "icu"}}' | jq .

# Step 2: Labs + consults
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "task4": {"priority_labs": ["CBC", "BMP", "Troponin"], "priority_consults": ["Cardiology"]}}' | jq .

# Step 3: Interventions
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "task4": {"interventions": ["mechanical ventilation", "arterial line"]}}' | jq .
```

### Metadata
```bash
curl http://localhost:7860/tasks | jq .
curl http://localhost:7860/episodes/by_complexity | jq .
curl http://localhost:7860/complexity/27703517 | jq .
curl "http://localhost:7860/history?n=5" | jq .
curl http://localhost:7860/metrics | jq .
```

---

## Thresholds Quick Reference

| Component | Parameter | Value |
|-----------|-----------|-------|
| Task 1 | Exact match | 1.00 |
| Task 1 | Broad group | 0.50 |
| Task 1 | Adjacent | 0.25 |
| Task 1 | Reasoning bonus | 0.05 |
| Task 1 | Min reasoning length | 20 chars |
| Task 2 | Specialty F1 weight | 0.35 |
| Task 2 | Medication F1 weight | 0.25 |
| Task 2 | Instruction quality weight | 0.25 |
| Task 2 | Discontinue accuracy weight | 0.15 |
| Task 2 | Max hallucination penalty | 0.10 |
| Task 2 | Max ghost specialty penalty | 0.10 |
| Task 2 | Efficiency: step 1–2 | 1.00× |
| Task 2 | Efficiency: step 3 | 0.85× |
| Task 2 | Efficiency: step 4+ | 0.70× |
| Task 2 | Max predicted specialties | 8 |
| Task 3 | Diagnosis coverage weight | 0.30 |
| Task 3 | Disposition accuracy weight | 0.20 |
| Task 3 | Medication F1 weight | 0.20 |
| Task 3 | LOS accuracy weight | 0.15 |
| Task 3 | Structure score weight | 0.10 |
| Task 3 | Information density weight | 0.05 |
| Task 3 | Max hallucination penalty | 0.15 |
| Task 3 | Follow-up structure penalty | 0.05 |
| Task 3 | Min word count for structure | 100 |
| Task 3 | Keyword density anti-stuff | 0.08 |
| Task 3 | Min sentence word count | 5 |
| Task 3 | LOS tolerance (fraction) | ±25% |
| Task 4 | Step 2 max | 0.15 |
| Task 4 | Step 3 max | 0.15 |
| Task 4 | Note weight at step 10 | 0.60 |
| Task 4 | Shaping weight at step 10 | 0.40 |
| Task 4 | Consistency bonus | 0.10 |
| Task 4 | Trajectory bonus | 0.05 |
| Task 4 | Revision cost per revision | 0.02 |
| Task 4 | Max revisions | 2 |
| Task 4 | Specialty overlap for bonus | ≥50% |
| GRPO | KL penalty β | 0.04 |
| GRPO | Top entropy quantile | 0.80 |
| GRPO | Dead-gradient halt (T1) | 80% zeros |
| GRPO | Dead-gradient halt (T2-4) | 90% zeros |
| GRPO | Zero-reward threshold | < 0.12 |
| Curriculum | Easy pool minimum | 10 patients |
| Curriculum | Medium/hard minimum | 25 patients |
| Observation | Max diagnoses shown | 15 |
| Observation | Task 2 initial diagnoses | 5 |
| Observation | Max procedures | 10 |

---

## Citation

Johnson, A. E. W., Pollard, T. J., et al.  
**MIMIC-IV (version 2.2)** — PhysioNet.  
https://physionet.org/content/mimic-iv-demo/2.2/
