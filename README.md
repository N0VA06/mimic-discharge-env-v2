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

A clinically grounded AI evaluation environment where agents analyze structured patient data and generate three essential discharge outputs: **disposition decisions, care plans, and discharge summaries**. Built using patient episodes modeled on the [MIMIC-IV dataset](https://physionet.org/content/mimic-iv-demo/2.2/), this environment was developed for the **Meta × Scaler OpenEnv Hackathon**.

---

## Why This Matters

Discharge planning is one of the most critical and error-prone stages in healthcare delivery. Inadequate discharge decisions contribute to avoidable hospital readmissions, increased healthcare costs (exceeding $26B), and compromised patient safety.

This environment enables AI agents to simulate real clinical responsibilities in a **safe, privacy-presing, and fully evaluable setting**. Unlike synthetic benchmarks, each task directly mirrors real hospital workflows, allowing meaningful assessment of clinical reasoning, decision-making, and documentation quality.

---

## Real-World Task Alignment

Each component of the environment corresponds directly to a role performed by healthcare professionals:

| Task | Clinical Equivalent |
|------|-------------------|
| **Discharge disposition** | Determining post-hospital placement (e.g., home, rehab, SNF) by case managers/utilization review nurses |
| **Care plan generation** | Writing discharge orders, medications, and follow-up instructions by hospitalists |
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
Six patients are included, each with different clinical profiles and discharge destinations. The environment server can also use the original dataset MIMIC Discharge by just changing the name of old_episode_builder.py to episode_builder.py and adding the dataset in the project root directory, currently uses this data (subset of [MIMIC-IV dataset](https://physionet.org/content/mimic-iv-demo/2.2/) is used due to 2 core and 8 ram contraint in the hackathon). 


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
│       ├── task1_disposition.py  — Task 1 grader (disposition)
│       ├── task2_careplan.py     — Task 2 grader (care plan, 4-step gated)
│       ├── task3_note.py         — Task 3 grader (discharge note)
│       └── task4_workflow.py     — Task 4 grader (10-step workflow, sparse reward)
├── server/
│   ├── __init__.py
│   └── app.py                    — FastAPI server (v3.1.0)
├── training/
│   ├── rollout_collector.py      — LLM rollout collection → HF Dataset
│   └── train_grpo.py             — GRPO training with curriculum phases
├── inference.py                  — baseline LLM agent
├── openenv.yaml                  — OpenEnv manifest (v3.1.0)
├── requirements.txt
└── Dockerfile
```

---

## Tasks

### Task 1 — Discharge Disposition (Easy)

**Objective:** Predict the correct post-discharge setting from 8 canonical categories.

**Valid choices:**

| Choice | Clinical meaning |
|---|---|
| `home` | Fully independent, no professional follow-up needed |
| `home_with_services` | Needs visiting nurse, home PT, or wound care |
| `snf` | Skilled nursing: 24h nursing + IV meds / wound VAC |
| `rehab` | Inpatient rehab: intensive PT/OT ≥3h/day, medically stable |
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

**Difficulty:** Easy — clear facility indicators (ventilation hours, dialysis, LOS > 7 days + age > 70) directly imply the answer.

---

### Task 2 — Care Plan Recommendation (Medium)

**Objective:** Recommend a complete post-discharge care plan with three components.

**Action fields:**

| Field | Description |
|---|---|
| `follow_up_specialties` | List of required follow-up specialties |
| `medications_to_continue` | Drugs to continue at home |
| `medications_to_discontinue` | Hospital-only or stopped drugs |
| `key_instructions` | 5 specific patient instructions with thresholds |

**Scoring formula:**
```
score = 0.35 × specialty_F1
      + 0.25 × medication_F1
      + 0.25 × instruction_quality
      + 0.15 × discontinue_accuracy
      − hallucination_penalty (max 0.10)
```

**Specialty ground truth** is derived programmatically from ICD-10 codes, microbiology results, and HCPCS categories. Instruction quality requires specific thresholds — "Weigh daily; call if gain >2 lbs" scores; "Monitor weight" does not.

**Difficulty:** Medium — requires integrating multiple data sources and producing clinically specific (not generic) output.

---

### Task 3 — Discharge Note Generation (Hard)

**Objective:** Write a complete clinical discharge summary (minimum 300 words) covering 7 required sections.

**Required sections (in order):**
1. PRINCIPAL DIAGNOSIS
2. BRIEF HOSPITAL COURSE *(must state LOS in days explicitly)*
3. KEY PROCEDURES PERFORMED
4. DISCHARGE CONDITION
5. DISCHARGE DISPOSITION *(exact canonical phrase)*
6. DISCHARGE MEDICATIONS *(only active discharge drugs)*
7. FOLLOW-UP INSTRUCTIONS *(specific timeframes and warning signs)*

**Scoring formula:**
```
score = 0.30 × diagnosis_coverage      (contextual, anti-keyword-stuffing)
      + 0.20 × disposition_accuracy
      + 0.20 × medication_F1
      + 0.15 × LOS_accuracy
      + 0.10 × structure_score
      + 0.05 × information_density
      − hallucination_penalty (max 0.15)
```

**Anti-gaming:** Keyword stuffing is detected (keywords must appear in sentence context ≥6 words; sentences matching ≥3 diagnoses are discarded). Drug hallucination uses pharmacological suffix matching to detect invented drug names.

**Difficulty:** Hard — requires coherent long-form clinical prose, correct section structure, and accurate medication/LOS recall without hallucination.

---

### Task 4 — Admission-to-Discharge Workflow (Very Hard)

**Objective:** Manage an ICU patient across 10 sequential clinical decisions, from admission triage through final discharge note.

**Sparse reward:** Only Step 10 returns a non-zero reward, making this suitable for GRPO training with delayed credit assignment.

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

**Revision mechanism:** Include `revise_step` + `revision` in any action to correct a prior step (max 2, costs 0.02 each from the final reward).

**Observation gating:** Fields are progressively revealed — labs at step 2, vitals/procedures at step 3, medications at step 4, microbiology at step 5, fluid balance at step 6, full observation at step 8+.

**Episode history:** `episode_history` field in each observation summarises prior step decisions for in-context memory.

**Difficulty:** Very Hard — requires clinical knowledge across the full inpatient stay, coherent multi-step planning, and strong discharge note generation.

---

## Training

### Rollout Collection

```bash
python -m training.rollout_collector \
  --env_url http://localhost:7860 \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --task_id 1 \
  --n_episodes 64 \
  --output_path ./rollouts
```

Saves a HuggingFace Dataset with columns: `prompt`, `response`, `reward`, `partial`, `hadm_id`, `task_id`, `step_num`, `parse_ok`.

### GRPO Training

```bash
python -m training.train_grpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --env_url http://localhost:7860 \
  --output_dir ./checkpoints \
  --max_steps 5000
```

Curriculum phases: Task 1 (steps 0-999, easy, clean) → Task 2 (steps 1000-2999, medium, partial noise) → Task 3 (steps 3000+, random, noisy). Dead-gradient guard halts training if Task 1 zero-reward rate exceeds 60% over 50 consecutive rollouts.

Install [Unsloth](https://github.com/unslothai/unsloth) for 4-bit QLoRA efficiency (optional — plain `transformers` + PEFT also supported).

---

## Observation Space

What the agent receives from `reset()` and `step()`:

| Field | Type | Description |
|---|---|---|
| `task_id` | int | 1, 2, or 3 |
| `subject_id` | int | Patient identifier |
| `hadm_id` | int | Admission identifier |
| `age` | int | Patient age at admission |
| `gender` | str | M / F |
| `admission_type` | str | EMERGENCY / ELECTIVE / URGENT |
| `admission_location` | str | Entry point |
| `insurance` | str | Medicare / Medicaid / Private |
| `hospital_los_days` | float | Length of stay in days |
| `diagnoses` | list | ICD codes with descriptions, ranked by sequence |
| `icu_stays` | list | ICU unit names and LOS per stay |
| `medications` | list | Drug orders with route and dose |
| `pharmacy_active` | list | Drugs active at discharge |
| `pharmacy_stopped` | list | Drugs stopped during admission |
| `lab_flags` | list | Abnormal lab results with H/L flags |
| `procedures` | list | ICD procedure descriptions |
| `drg_codes` | list | DRG code + description + severity/mortality |
| `microbiology` | list | Culture results with organism and sensitivities |
| `icu_procedures` | object | Ventilation hours, dialysis, art-line, CVL |
| `care_trajectory` | list | Ordered care unit path (ED → MICU → floor) |
| `hcpcs_categories` | list | HCPCS service categories |
| `task_description` | str | Natural language task description |
| `action_space_description` | str | JSON schema for the action |
| `max_steps` | int | Maximum steps for this task (1 or 2) |

---

## Action Space

What the agent sends to `step()`:

```json
{
  "task_id": 1,
  "task1": {
    "disposition": "snf",
    "reasoning": "Ventilation 58 hours and LOS 12.7 days require facility-level care."
  }
}
```

```json
{
  "task_id": 2,
  "task2": {
    "follow_up_specialties": ["Primary Care", "Pulmonology", "Infectious Disease"],
    "medications_to_continue": ["Amoxicillin-clavulanate", "Amlodipine"],
    "medications_to_discontinue": ["Piperacillin-tazobactam IV", "Vancomycin IV"],
    "key_instructions": [
      "Follow up with primary care within 5 days of discharge.",
      "Take amoxicillin-clavulanate for the full 5-day course even if feeling better.",
      "Return to ED if temperature exceeds 38.5 C or oxygen saturation falls below 92%.",
      "Use albuterol inhaler as needed for shortness of breath, max 2 puffs every 4 hours.",
      "Avoid smoking and all secondhand smoke exposure."
    ],
    "reasoning": "Sepsis recovery requires infectious disease and pulmonology follow-up."
  }
}
```

```json
{
  "task_id": 3,
  "task3": {
    "discharge_note": "PRINCIPAL DIAGNOSIS: ...\n\nBRIEF HOSPITAL COURSE: ..."
  }
}
```

---

## Reward Signal

All three graders emit **partial signals** alongside the scalar reward, enabling rich reward shaping:

**Task 1 partial signals:**
- `disposition_exact` — 1.0 if exact match
- `disposition_broad` — 0.5 if same broad group
- `disposition_adjacent` — 0.25 if clinically adjacent
- `reasoning_bonus` — 0.05 if reasoning contains clinical keywords

**Task 2 partial signals:**
- `specialty_recall`, `specialty_precision`, `specialty_f1`
- `medication_f1`, `medication_halluc_rate`
- `instruction_quality`
- `discontinue_accuracy`
- `hallucination_penalty`

**Task 3 partial signals:**
- `diagnosis_coverage`, `disposition_score`, `medication_f1`
- `los_accuracy`, `structure_score`, `information_density`
- `hallucination_rate`, `hallucination_penalty`

---

## Setup

### Prerequisites

- Python 3.11+
- pip or uv

### Install dependencies

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv sync
```

### Start the server

```bash
python -m server.app
```

Server starts in <2 seconds and is immediately ready. No dataset download required.

### Verify health

```bash
curl http://localhost:7860/health
# → {"status": "ok", "ready": true, "episodes_available": 6, ...}
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

### Build

```bash
docker build -t mimic-discharge-env .
```

### Run

```bash
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  mimic-discharge-env
```

### Verify

```bash
curl http://localhost:7860/health
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}' | python3 -m json.tool
```

---

## Baseline Scores

Tested with `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router (temperature=0.1, pinned episodes):

| Task | Name | Difficulty | Score | Steps | Pinned Patient |
|---|---|---|---|---|---|
| 1 | discharge-disposition | Easy | 0.55 | 1 | hadm_id=1002 (81yo, sepsis + vent) |
| 2 | care-plan | Medium | 0.68 | 1 | hadm_id=1001 (68yo, heart failure) |
| 3 | discharge-note | Hard | 0.95 | 1 | hadm_id=1006 (71yo, COPD) |
| **Avg** | | | **0.73** | | |

Scores are **reproducible** — the same model, temperature, and pinned hadm_ids will produce identical results across runs.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Readiness probe — returns `{"ready": true}` when ready |
| `/reset` | POST | Start episode: `{"task_id": 1, "hadm_id": null}` |
| `/step` | POST | Submit action, receive reward |
| `/state` | GET | Current episode state + ground truth (debug) |
| `/tasks` | GET | Full task catalogue with scoring formulas |
| `/metrics` | GET | Request counts, latency percentiles, reward stats |
| `/history` | GET | Last N completed episodes |
| `/episodes` | GET | Available episode count and sample hadm_ids |
| `/docs` | GET | Interactive Swagger UI |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | HuggingFace API token (or `OPENAI_API_KEY`) |
| `MODEL_NAME` | No | `meta-llama/Llama-3.1-8B-Instruct` | Model for inference |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API base URL |
| `ENV_URL` | No | `http://localhost:7860` | Environment server URL |
| `PORT` | No | `7860` | Server port |

---

## Grader Design Notes

All graders are **fully deterministic and programmatic** — no LLM judge, no randomness.

- **Task 1** uses exact string matching against MIMIC discharge location strings mapped to 8 canonical categories. Three scoring tiers (exact → broad group → adjacent) ensure non-binary reward signal even for partially correct answers.
- **Task 2** uses token-stemmed F1 matching for both medications and specialties. Specialty ground truth is derived from ICD-10 prefix mapping + microbiology organism detection + HCPCS category keywords.
- **Task 3** uses contextual keyword matching (sentence-level, not document-level) with anti-stuffing detection, pharmacological suffix regex for drug hallucination detection, and logarithmic length scoring to avoid rewarding verbosity over quality.

## Citation

Johnson, A. E. W., Pollard, T. J., Shen, L., et al.  
**MIMIC-IV (version 2.2)**. PhysioNet.  
https://physionet.org/content/mimic-iv-demo/2.2/
# mimic-discharge-env-v2
# mimic-discharge-env-v2
