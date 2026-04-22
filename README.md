---
title: MIMIC Discharge Planning
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - healthcare
  - clinical-nlp
  - reinforcement-learning
---

# MIMIC Discharge Planning — OpenEnv

A clinically grounded AI evaluation environment where agents analyze structured patient data and generate three essential discharge outputs: **disposition decisions, care plans, and discharge summaries**. Built using patient episodes from the [MIMIC-IV dataset](https://physionet.org/content/mimic-iv-demo/2.2/), this environment was developed for the **Meta × Scaler OpenEnv Hackathon**.

---

## Why This Matters

Discharge planning is one of the most critical and error-prone stages in healthcare delivery. Inadequate discharge decisions contribute to avoidable hospital readmissions, increased healthcare costs (exceeding $26B), and compromised patient safety.

This environment enables AI agents to simulate real clinical responsibilities in a **safe, privacy-preserving, and fully evaluable setting**. Unlike synthetic benchmarks, each task directly mirrors real hospital workflows, allowing meaningful assessment of clinical reasoning, decision-making, and documentation quality.

---

## Real-World Task Alignment

| Task | Clinical Equivalent |
|------|-------------------|
| **Discharge disposition** | Post-hospital placement decisions by case managers and utilization review nurses |
| **Care plan generation** | Discharge orders, medications, and follow-up instructions written by hospitalists |
| **Discharge summary note** | Final clinical documentation prepared by attending physicians |

---

## Environment Overview

```
Episode flow:
  POST /reset  →  observation (patient data)
  POST /step   →  reward (0.0–1.0) + partial signals + done
  GET  /state  →  current episode state + ground truth (debug)
  GET  /health →  server readiness probe
  GET  /tasks  →  task catalogue with scoring formulas
```

The environment uses a real MIMIC-IV patient subset (233 episodes). It can also be run against the full MIMIC-IV dataset by renaming `old_episode_builder.py` → `episode_builder.py` and placing the full CSV tables in the project root (the demo uses a 2-CPU / 8 GB subset due to hackathon constraints).

---

## Project Structure

```
mimic-discharge-env/
├── environment/
│   ├── __init__.py               — exports MIMICDischargeEnv, Action, Observation
│   ├── env.py                    — core MIMICDischargeEnv class (Tasks 1-4)
│   ├── old_episode_builder.py    — MIMIC-IV based episode builder (v3)
│   ├── models.py                 — Pydantic models (Action, Observation, StepResult)
│   └── tasks/
│       ├── task1_disposition.py  — Task 1 grader (disposition, 3-tier scoring)
│       ├── task2_careplan.py     — Task 2 grader (care plan, 4-step gated)
│       ├── task3_note.py         — Task 3 grader (discharge note, anti-stuffing)
│       └── task4_workflow.py     — Task 4 grader (10-step workflow, sparse reward)
├── server/
│   ├── __init__.py
│   └── app.py                    — FastAPI server (v3.1.0)
├── training/
│   ├── rollout_collector.py      — LLM rollout collection → HF Dataset
│   └── train_grpo.py             — GRPO training (Tasks 1–3, curriculum phases)
├── inference.py                  — clinical LLM agent (Tasks 1–4)
├── openenv.yaml                  — OpenEnv manifest (v3.1.0)
├── requirements.txt
└── Dockerfile
```

---

## Tasks

### Task 1 — Discharge Disposition (Easy)

**Objective:** Predict the correct post-discharge setting from 8 canonical categories.

| Choice | Clinical meaning |
|---|---|
| `home` | Fully independent, no professional follow-up needed |
| `home_with_services` | Needs visiting nurse, home PT, or wound care |
| `snf` | Skilled nursing: 24h nursing + IV meds / wound care |
| `rehab` | Inpatient rehab: intensive PT/OT, medically stable |
| `hospice` | Terminal prognosis, comfort-focused care |
| `ama` | Left against medical advice |
| `expired` | Patient died during admission |
| `other` | Transfer to acute hospital or psychiatric unit |

**Scoring:**

| Match level | Score |
|---|---|
| Exact canonical match | 1.00 |
| Same broad group (community / facility / end-of-life) | 0.50 |
| Clinically adjacent (e.g. snf ↔ home_with_services) | 0.25 |
| Wrong group, no adjacency | 0.00 |
| Clinical reasoning bonus | +0.05 |

**Key clinical signals used by the agent:**
- ICD code V667 / Z51.5 (palliative care) → **hospice**
- Secondary malignant neoplasm + DRG mortality=4.0 → **hospice**
- GCS ≤ 5 with terminal diagnosis → **hospice**
- Ventilation hours > 0 or dialysis, non-terminal → **snf**
- Orthopedic fracture with fixation → **rehab**
- Discharge orders finalized + oral-only meds → **home**

---

### Task 2 — Care Plan Recommendation (Medium)

**Objective:** Recommend a complete post-discharge care plan. Uses a 4-step information revelation protocol — the agent starts with minimal data and requests more before submitting.

**Optimal strategy:** Request labs + medications + microbiology on step 1, submit care plan on step 2 (≤2 steps = 1.0× efficiency multiplier).

**Action fields:**

| Field | Description |
|---|---|
| `follow_up_specialties` | Derived from ICD codes: e.g. ICD-9 520–579 → Gastroenterology |
| `medications_to_continue` | EXACT names from `pharmacy_active` list only |
| `medications_to_discontinue` | EXACT names from `pharmacy_stopped` list only |
| `key_instructions` | 5 specific instructions with numeric thresholds |

**Scoring formula:**
```
score = 0.35 × specialty_F1
      + 0.25 × medication_F1
      + 0.25 × instruction_quality
      + 0.15 × discontinue_accuracy
      − hallucination_penalty (max 0.10)
      − ghost_specialty_penalty (max 0.10)
      × step_efficiency_discount
```

**Step efficiency discount:** ≤2 steps = 1.0×, 3 steps = 0.85×, 4 steps = 0.70×.

**Specialty ground truth** is derived from ICD-9 numeric ranges and ICD-10 prefix codes. Recommending a specialty with no supporting ICD code incurs a ghost specialty penalty.

---

### Task 3 — Discharge Note Generation (Hard)

**Objective:** Write a complete clinical discharge summary (minimum 300 words) covering 7 required sections.

**Required sections (in order):**
1. PRINCIPAL DIAGNOSIS
2. BRIEF HOSPITAL COURSE *(must state LOS in days explicitly)*
3. KEY PROCEDURES PERFORMED
4. DISCHARGE CONDITION
5. DISCHARGE DISPOSITION *(exact canonical phrase — one of 6 options)*
6. DISCHARGE MEDICATIONS *(only active discharge drugs from `pharmacy_active`)*
7. FOLLOW-UP INSTRUCTIONS

**Required disposition phrases (use verbatim):**
- `"The patient was discharged home."`
- `"The patient was discharged home with home health services."`
- `"The patient was transferred to a skilled nursing facility."`
- `"The patient was transferred to inpatient rehabilitation."`
- `"The patient was transitioned to hospice care."`
- `"The patient expired during this hospitalization."`

**Scoring formula:**
```
score = 0.30 × diagnosis_coverage      (contextual, anti-keyword-stuffing)
      + 0.20 × disposition_accuracy    (exact phrase required)
      + 0.20 × medication_F1           (pharmacy_active vs. note drugs)
      + 0.15 × LOS_accuracy            (within ±25% of actual)
      + 0.10 × structure_score         (7 sections present)
      + 0.05 × information_density
      − hallucination_penalty (max 0.15)
      − followup_structure_penalty (0.05 if planning finalized but follow-up missing)
```

**Anti-gaming:** Keyword stuffing is detected. Diagnosis keywords must appear in sentences of ≥5 words. A sentence matching ≥3 diagnoses simultaneously is discarded (vague catch-all). Drug hallucination is checked against both `prescriptions` and `emar_summary`.

---

### Task 4 — Admission-to-Discharge Workflow (Very Hard)

**Objective:** Manage an ICU patient across 10 sequential clinical decisions, from triage through final discharge note.

**Note: Task 4 is excluded from the GRPO training curriculum** due to its sparse reward structure (only step 10 returns non-zero reward) making gradient signal too dilute at typical training batch sizes. It is available for evaluation and manual testing.

| Step | Focus | Max contribution |
|------|-------|-----------------|
| 1 | Acuity triage (icu / stepdown / floor) | 0.10 |
| 2 | Priority labs + specialist consults | 0.15 |
| 3 | Interventions (ventilation, dialysis, lines) | 0.15 |
| 4 | High-risk medication identification | 0.10 |
| 5 | Antibiotic stewardship plan | 0.10 |
| 6 | Fluid management strategy | 0.08 |
| 7 | ICU-to-stepdown readiness assessment | 0.08 |
| 8 | Discharge disposition + LOS estimate | 0.10 |
| 9 | Discharge medication reconciliation | 0.10 |
| 10 | Final discharge note (composite reward) | 1.0 cap |

**Step 10 formula:**
```
final = note_score × 0.60 + shaping_avg × 0.40
      + consistency_bonus (0.10 if note disposition matches Step 8)
      + trajectory_bonus  (0.05 if ≥50% specialty overlap with Step 2)
      − revision_cost     (0.02 × revisions_used, max 2)
```

---

## Inference Agent

`inference.py` implements a clinical LLM agent for all 4 tasks. Key design decisions:

**Task 1 — Clinical decision tree (in order):**
1. HOSPICE if: ICD V667/Z51.5, secondary malignant neoplasm + DRG mortality=4.0, or GCS ≤ 5 with terminal diagnosis
2. EXPIRED if patient explicitly died
3. SNF if ventilation/dialysis + non-terminal
4. REHAB for orthopedic fracture with surgical fixation
5. HOME WITH SERVICES if wound care or IV antibiotics at home
6. HOME as default for stable short-stay patients

**Task 2 — Specialty derivation from ICD codes:**
```
ICD-9: 001–139→Infectious Disease, 390–459→Cardiology, 520–579→Gastroenterology,
       580–629→Nephrology, 800–999→Trauma Surgery  (+ others)
ICD-10: A,B→ID, I→Cardiology, K→Gastroenterology, N→Nephrology, etc.
```
Medications use EXACT names from `pharmacy_active` only.

**Task 2 — Info-request protocol:**
On step 0, if meds/labs are empty, automatically requests `["labs", "medications", "microbiology"]` before submitting the care plan. This ensures the care plan has real data and achieves the 1.0× efficiency multiplier (≤2 steps).

**Task 3 — Note requirements enforced in prompt:**
Each top-5 diagnosis must be named by its description keywords. Only `pharmacy_active` exact drug names may be listed. LOS must be stated as a number in days within ±25% of actual.

---

## Training

Training covers **Tasks 1, 2, and 3 only**. Task 4 is excluded — its 10-step sparse reward adds training noise without clean per-step gradient signal.

### Curriculum Phases

| Steps | Task | Noise | Curriculum |
|-------|------|-------|------------|
| 0–999 | 1 (disposition) | `clean` | `easy_only` |
| 1000–2999 | 2 (care plan) | `clean` | `medium_only` |
| 3000+ | 3 (discharge note) | `partial` | `random` |

**Why `clean` for Task 2 (changed from `partial`):** `partial` noise drops 30% of labs and truncates medications — the signal for medication F1 and specialty derivation was near-zero even for correct responses, producing dead gradients.

**Why `partial` for Task 3 (changed from `noisy`):** Full `noisy` mode scrambles diagnosis sequence numbers, making it impossible for the model to learn diagnosis coverage.

### Task 2 Reward Fix

The reward function pre-steps the information request before scoring the model's care plan response. Without this, the environment is scored on an empty-medication step-0 state, causing 100% medication hallucination rate regardless of response quality.

```python
# In reward_fn (train_grpo.py):
if task_id == 2:
    _env_post(env_url, "/step", {
        "task_id": 2,
        "information_request": ["labs", "medications", "microbiology"],
    })
result = _env_post(env_url, "/step", action_dict)
```

The seed dataset is built the same way — seeds are step-1 observations (enriched) so the model trains on prompts that already contain lab and medication data.

### Rollout Collection

```bash
python -m training.rollout_collector \
  --env_url http://localhost:7860 \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --task_id 1 \
  --n_episodes 64 \
  --output_path ./rollouts
```

On parse failure the collector uses `_fallback_action` (deterministic clinical heuristics) instead of a zero-reward noop, ensuring every episode contributes training signal.

### GRPO Training

```bash
python -m training.train_grpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --env_url http://localhost:7860 \
  --output_dir ./checkpoints \
  --max_steps 5000
```

Install [Unsloth](https://github.com/unslothai/unsloth) for 4-bit QLoRA (optional — plain `transformers` + PEFT also supported).

**Dead-gradient guard:** Halts if Task 1 zero-reward rate exceeds 60% over 50 consecutive rollouts.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `age`, `gender` | int, str | Patient demographics |
| `admission_type`, `admission_location` | str | Entry point |
| `hospital_los_days` | float | Length of stay |
| `diagnoses` | list | ICD codes + descriptions, ranked by sequence |
| `icu_stays` | list | ICU unit names and LOS |
| `medications` | list | Drug orders with route and dose (reference) |
| `pharmacy_active` | list | **Drugs active at discharge — use for `medications_to_continue`** |
| `pharmacy_stopped` | list | **Drugs stopped during admission — use for `medications_to_discontinue`** |
| `lab_flags` | list | Abnormal lab results |
| `procedures` | list | ICD procedure descriptions |
| `drg_codes` | list | DRG code + severity/mortality scores |
| `microbiology` | list | Culture organisms → triggers Infectious Disease specialty |
| `icu_procedures` | object | Ventilation hours, dialysis, arterial/central lines |
| `vitals` | list | Heart rate, BP, SpO2, GCS (admission → discharge values) |
| `care_trajectory` | list | Ordered care unit path (ED → MICU → floor) |
| `emar_summary` | list | Medication admin records with `active_at_discharge` flag |
| `discharge_orders` | object | `discharge_planning_finalized` + order types |
| `fluid_balance` | object | Net balance, `fluid_overloaded`, `oliguria` flags |

**Critical fields for scoring:**
- `pharmacy_active` → the ONLY valid source for `medications_to_continue`
- `pharmacy_stopped` → the ONLY valid source for `medications_to_discontinue`
- `vitals.GCS Total` → key end-of-life indicator for Task 1
- `diagnoses[*].icd_code` → specialty derivation for Task 2

---

## Reward Signal

All graders emit **partial signals** alongside the scalar reward.

**Task 1:** `disposition_exact`, `disposition_broad`, `disposition_adjacent`, `reasoning_bonus`

**Task 2:** `specialty_recall`, `specialty_precision`, `specialty_f1`, `medication_f1`, `medication_halluc_rate`, `instruction_quality`, `discontinue_accuracy`, `hallucination_penalty`, `ghost_specialty_penalty`

**Task 3:** `diagnosis_coverage`, `disposition_score`, `medication_f1`, `los_accuracy`, `structure_score`, `information_density`, `hallucination_rate`, `hallucination_penalty`

---

## Setup

### Prerequisites

- Python 3.11+

### Install

```bash
pip install -r requirements.txt
# or
uv sync
```

### Start the server

```bash
python -m server.app
```

### Verify health

```bash
curl http://localhost:7860/health
# → {"status": "ok", "ready": true, "episodes_available": 233, ...}
```

### Run the baseline agent

```bash
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export ENV_URL=http://localhost:7860
python inference.py
```

---

## Docker

```bash
docker build -t mimic-discharge-env .

docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  mimic-discharge-env
```

---

## Baseline Scores

Tested with `meta-llama/Llama-3.1-8B-Instruct` (temperature=0.1):

| Task | Name | Difficulty | Baseline (before fix) | Expected (after fix) | Key improvement |
|---|---|---|---|---|---|
| 1 | discharge-disposition | Easy | 0.05 | ~0.75+ | Hospice/end-of-life detection added |
| 2 | care-plan | Medium | 0.245 | ~0.60+ | Specialty derived from ICD codes; meds from active list only |
| 3 | discharge-note | Hard | 0.329 | ~0.60+ | Diagnosis keywords in prose; exact active med names |

**Why scores were low before:**
- Task 1: Schema had no hospice logic. Terminal cancer patients predicted as "home".
- Task 2: Specialties were guessed generically ("Cardiology, Nephrology") regardless of diagnoses. Medications hallucinated at 80% rate because the model used the orders list instead of `pharmacy_active`.
- Task 3: Generic note template contained no diagnosis keywords and invented drug names.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Readiness probe |
| `/reset` | POST | Start episode: `{"task_id": 1, "hadm_id": null, "noise_level": "clean"}` |
| `/step` | POST | Submit action, receive reward |
| `/state` | GET | Current episode state + ground truth (debug) |
| `/tasks` | GET | Task catalogue with scoring formulas |
| `/metrics` | GET | Request counts, latency, reward stats |
| `/history` | GET | Last N completed episodes |
| `/episodes/by_complexity` | GET | All hadm_ids grouped by easy/medium/hard |
| `/complexity/{hadm_id}` | GET | Complexity tier for a specific admission |
| `/docs` | GET | Interactive Swagger UI |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace API token |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | LLM for inference |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API base |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |
| `PORT` | `7860` | Server port |

---

## Grader Design Notes

All graders are **fully deterministic and programmatic** — no LLM judge, no randomness.

- **Task 1** uses exact substring matching against MIMIC discharge location strings mapped to 8 canonical categories. Three scoring tiers ensure non-binary reward signal.
- **Task 2** uses token-stem F1 for medications. Specialty ground truth is derived from ICD prefix tables + microbiology organism detection + HCPCS keywords. Ghost specialty penalty discourages recommending specialties without supporting diagnoses.
- **Task 3** uses sentence-level keyword matching with anti-stuffing detection. A sentence matching ≥3 diagnoses simultaneously scores zero (catches vague catch-all sentences). Drug hallucination is checked against both prescriptions and eMAR.

## Citation

Johnson, A. E. W., Pollard, T. J., Shen, L., et al.
**MIMIC-IV (version 2.2)**. PhysioNet.
https://physionet.org/content/mimic-iv-demo/2.2/
