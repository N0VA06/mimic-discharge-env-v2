# Building a Clinical RL Benchmark on Real Hospital Data

*A deep-dive into MIMIC Discharge Planning — built for the Meta × Scaler OpenEnv Hackathon.*

**[🚀 Live Environment : https://iinovaii-mimic-discharge-env-v2.hf.space/](https://iinovaii-mimic-discharge-env-v2.hf.space/)**

**[🚀 Hugging Face Space : https://huggingface.co/spaces/IINOVAII/mimic-discharge-env-v2/](https://huggingface.co/spaces/IINOVAII/mimic-discharge-env-v2/)**

**Watch if you feel its interesting after the read 🙂**

[![Watch the video](https://img.youtube.com/vi/hKE8422zt2k/0.jpg)](https://youtu.be/hKE8422zt2k?cc_load_policy=1)

---

**20% of Medicare patients return to the hospital within 30 days of discharge.** That's $26 billion a year in preventable readmissions — and most of it traces back to discharge planning done poorly under time pressure.

Physicians must simultaneously decide where a patient goes next, which medications continue, who follows up, and what instructions go in the note. They do this for dozens of patients every day, often with incomplete labs and conflicting signals.

This is exactly the kind of sequential, multi-signal decision problem that reinforcement learning is built for. The catch: most medical RL benchmarks use simulated environments or LLM judges for scoring — both of which introduce noise that makes results hard to trust.

We took a different approach. **Every patient in this benchmark is real. Every grader is deterministic. Every reward decomposes into named clinical signals.**

---

## Why MIMIC-IV?

[MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/) is the gold standard public ICU dataset (Medical Information Mart for Intensive Care) — structured electronic patient records from Beth Israel Deaconess Medical Center. The demo version has 233 complete hospital admissions with full lab results, medication records, ICU procedures, infection test results, hospital billing codes, vital signs, and discharge orders. Can be further expanded to **364,627 real patient admissions by using the full dataset [Full MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)**.

Crucially, every admission has a **known discharge outcome**. That gives us a ground truth for every task in the benchmark.

```
233 real hospital admissions
  ├── 21  easy   — short hospital stay, goes home, straightforward
  ├── 109 medium — home vs home-with-services (visiting nurse) discrimination
  └── 103 hard   — nursing facility / hospice / intensive care / died
```

The easy tier is almost useless for training — all 21 patients discharge home. The medium tier is where the real signal lives. The hard tier covers the complex cases: terminal cancers, multi-day ICU stays, patients who died.

---

## The 4 Tasks

Each task is a harder version of the discharge planning problem. The curriculum trains through all 4 sequentially.

### Task 1 — Discharge Disposition *(Easy · 1 step)*

Predict where a patient goes after leaving the hospital — from their full medical record. Sounds simple. The hard part is distinguishing `home` from `home_with_services` (visiting nurse vs. truly independent), or knowing when signs point to `hospice` (end-of-life care) vs. a nursing facility.

**Key signals the model learns:** Patient on a ventilator → skilled nursing facility. Bone surgery (hip/knee) → rehabilitation center. Terminal diagnosis or very low consciousness level → hospice care.

**Scoring:** Exact match = 1.0. Nearby category (nursing facility ↔ home-with-services) = 0.50. Same broad group = 0.25.

Training result: reward climbs **0.24 → 0.73** over 200 steps. The model goes from "always hospice" (mode collapse) to reading actual clinical features.

---

### Task 2 — Care Plan Recommendation *(Medium · ≤4 steps)*

A gated multi-step task. The agent starts with demographics and top 5 diagnoses only, then must request additional data before submitting a care plan.

The submitted plan needs:
- Which specialist doctors need follow-up appointments (based on diagnoses — e.g. heart condition → cardiologist)
- Which medications to keep taking (from the active prescription list)
- Which medications to stop (from the discontinued prescription list)
- 5 specific home instructions with numeric thresholds

**The efficiency trap:** Asking for data on steps 1, 2, and 3 before submitting on step 4 gets a 0.70× multiplier. Optimal is request on step 1, submit on step 2 (1.0×).

**Scoring:** Weighted across specialist accuracy (35%), medication accuracy (25%), instruction quality (25%), and discontinued medication accuracy (15%) — multiplied by the step efficiency discount.

Training result: stabilizes at **~0.58–0.65**. The model quickly learns the 2-step optimal strategy.

---

### Task 3 — Discharge Note Generation *(Hard · 1 step)*

Write a complete clinical discharge summary (minimum 300 words) with all 7 required sections in order. Quality rules prevent keyword cramming: a sentence listing 3 or more diagnoses at once scores zero — the model must write in proper prose.

```
Required sections:
  1. PRINCIPAL DIAGNOSIS
  2. BRIEF HOSPITAL COURSE  (must state how many days the patient was hospitalized)
  3. KEY PROCEDURES PERFORMED
  4. DISCHARGE CONDITION
  5. DISCHARGE DISPOSITION   (exact phrasing only — 6 options)
  6. DISCHARGE MEDICATIONS   (currently prescribed medications, exact names only)
  7. FOLLOW-UP INSTRUCTIONS
```

**Scoring:** Weighted across diagnosis coverage (30%), discharge destination accuracy (20%), medication accuracy (20%), length-of-stay accuracy (15%), document structure (10%), and information density (5%) — minus a penalty for invented facts.

Training result: **~0.40–0.55** with high variance. Long-form generation is harder to optimize — the reward signal is sparse across a large output space.

---

### Task 4 — ICU Admission-to-Discharge Workflow *(Very Hard · 10 steps)*

The hardest task: sequential clinical decision-making across a full ICU admission. Sparse reward — steps 1–9 return 0. Only step 10 (the final discharge note) returns a score.

| Step | Decision |
|------|----------|
| 1 | How sick is the patient? (Intensive care / step-down ward / regular ward) |
| 2 | Which lab tests and specialist doctors are needed |
| 3 | Key treatments needed: breathing support, kidney dialysis, IV lines |
| 4 | Which medications carry high risk and need close monitoring |
| 5 | Choosing the right antibiotics |
| 6 | Managing IV fluid intake and output |
| 7 | Is the patient ready to move from intensive care to a less intensive ward? |
| 8 | Predicted discharge destination + expected days remaining |
| 9 | Final medication review before discharge |
| 10 | Final discharge note (composite reward) |

The patient's medical record is revealed progressively by step — lab results unlock at step 2, vitals and procedures at step 3, infection cultures at step 5, fluid balance at step 6. The agent genuinely has to make early decisions under uncertainty.

**Step 10 score:**
```
note_score × 0.60
+ average of step 1–9 quality scores × 0.40
+ consistency bonus (0.10 if discharge destination at step 8 matches the final note)
+ trajectory bonus  (0.05 if ≥50% of recommended specialists appear in the final note)
− revision cost     (0.02 per revision, max 2)
```

Training result: builds from **~0.26 → 0.38** over 100 steps. The model transfers note-writing from Task 3 and adapts to the full ICU workflow.

---

## Training with GRPO

We used **Group Relative Policy Optimization (GRPO)** from TRL 0.23 — a policy gradient method well-suited to environments with sparse or delayed rewards.

**Model:** Qwen/Qwen2.5-3B-Instruct with LoRA rank=16 (only ~0.1–0.5% of parameters trained — ~1–3M adapter weights; the 3B base model is frozen throughout).

**Hardware:** Single NVIDIA L4 24 GB — 7 hours total for all 550 steps.

**Curriculum (50-step chunks):**

| Phase | Steps | Task | Pool | Notes |
|-------|-------|------|------|-------|
| 1 | 0–199 | Disposition | medium+easy (130) | Clean signal, short outputs |
| 2 | 200–349 | Care Plan | medium (109) | Multi-step with efficiency discount |
| 3 | 350–449 | Discharge Note | all 233 | Long-form, high-variance reward |
| 4 | 450–549 | ICU Workflow | hard (103) | Sparse reward, 9-step advance |

![Reward Curve](logs/plots/01_reward_curve.png)

![Per-Task Learning Curves](logs/plots/08_per_task_curves.png)

---

## What Actually Went Wrong

Four significant problems surfaced during training. Each has a specific, reproducible fix.

### 1. Hospice Mode Collapse

**Symptom:** Model output `hospice` for every single patient for 42 consecutive steps. Reward frozen at ~0.12.

**Why:** MIMIC-IV's hard tier has many hospice/expired patients. The model found a local maximum early: just always say hospice.

**Fix:** If any single disposition class exceeds 85% of the last 16 rollouts, the seed dataset is rebuilt with forced class diversity — the easy tier (all plain home discharges) is added to break the collapse.

---

### 2. Prompt–Reward Decoupling

**Symptom:** Rewards were near-random regardless of output quality. Loss was moving but reward wasn't.

**Why:** The reward function called `/reset` without pinning to the same patient the model saw in its prompt. The model wrote a note about patient A; the reward scored patient B.

**Fix:** Every seed dataset row now stores `hadm_id` alongside the prompt. The reward function pins `/reset` to that specific patient:

```python
reset_body["hadm_id"] = int(hadm_id[i])
_env_post(env_url, "/reset", reset_body)
# ... then step with the model's actual action
```

Without this fix, training is essentially random. This was the single most impactful bug.

---

### 3. Disposition String Mismatch

**Symptom:** `"home with services"` scored 0.44 instead of 1.0.

**Fix:** Normalize model output to canonical values before the env call:
- `"home with services"` → `"home_with_services"`
- `"skilled nursing"` → `"snf"`

---

### 4. KL Collapse / Low Entropy

**Symptom:** GRPO advantage estimates near zero. No gradient signal.

**Why:** With 8 generations per step, if the policy has low entropy, group-relative advantages are all near zero.

**Fix:**
- `beta=0.04` KL penalty — keeps the policy close to the base model's diversity
- `top_entropy_quantile=0.8` — drops the bottom 20% entropy completions

---

## Results

![Phase Timeline](logs/plots/05_phase_timeline.png)

| Phase | Task | Difficulty | Steps | Result |
|-------|------|------------|-------|--------|
| 1 | Disposition | Easy | 0–199 | 0.24 → **0.73** peak |
| 2 | Care Plan | Medium | 200–349 | **~0.60** stable |
| 3 | Discharge Note | Hard | 350–449 | **~0.45** ± high variance |
| 4 | ICU Workflow | Very Hard | 450–549 | 0.26 → **0.38** |

**Overall mean reward across all 550 steps: 0.468**

### Fine-Tuned vs Baseline

We ran the same 4 tasks against **Llama-3.1-8B-Instruct** as a zero-shot baseline (no fine-tuning, no RL) to understand what the curriculum actually bought.

| Phase | Task | Difficulty | Fine-Tuned (Qwen 3B + GRPO) | Baseline (Llama-3.1-8B-Instruct) |
|-------|------|------------|------------------------------|----------------------------------|
| 1 | Disposition | Easy | **0.24 → 0.73** | ~0.05–0.30 (avg ~0.11) |
| 2 | Care Plan | Medium | **~0.58–0.65** | ~0.60–0.73 (avg ~0.68) |
| 3 | Discharge Note | Hard | **~0.40–0.55** | ~0.43–0.59 (avg ~0.51) |
| 4 | ICU Workflow | Very Hard | **~0.26 → 0.38** | ~0.27–0.65 (avg ~0.46) |

A few things stand out:

**Task 1 is where RL wins most clearly.** The baseline Llama-8B scores only ~0.11 on average — it can't reliably map clinical signals to the exact canonical disposition classes under the strict grader. The fine-tuned Qwen-3B ends up at 0.73 despite being a smaller model, because it learned to read diagnosis codes, mortality risk scores, and ICU flags directly from rollout feedback.

**Tasks 2 and 3 are surprisingly close.** The baseline does well on care plans (0.68) because general reasoning ability transfers well — it can read diagnosis codes and suggest relevant specialists without task-specific training. Discharge note generation (0.51 baseline) also benefits from strong base generation capability.

**Task 4 is nuanced.** The baseline's higher average (~0.46) reflects the composite grader structure: steps 1–9 contribute shaping to the final score, and a capable baseline can produce reasonable individual step outputs. The fine-tuned model's lower average but upward trend (0.26 → 0.38) is a signature of learning under sparse reward — the model is adapting its note quality specifically, not just producing plausible text.

**The headline takeaway:** a 3B model with 7 hours of GRPO training on a single L4 matches or beats an 8B zero-shot model on 3 of 4 tasks. Task 4 is the exception, where a stronger base model's generation quality still wins at the current training budget.

![Reward Histogram](logs/plots/06_reward_histogram.png)

The reward distribution is bimodal: a cluster at 0.24–0.44 (partial credit) and a spread from 0.50–0.80 (good clinical reasoning). The spike at 1.0 is exact matches on Task 1 dispositions. Task 4 adds a mode at 0.30–0.40 from the 0.60× note-score cap in its grader formula.

---

## Scaling Up

The demo dataset has 233 patients. The [Full MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) dataset has over 364,627 patients. The environment architecture scales directly — swap in the full CSV files, rebuild the episode index, and the same curriculum runs with a much richer patient distribution.

---

## Try It

```bash
pip install -r requirements.txt
python -m server.app
pip install unsloth accelerate peft matplotlib datasets # is not included in requirements.txt so that docker file does not inflate
python -m training.train_grpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --env_url http://localhost:7860 \
  --max_steps 550
```

---

## Technical Reference

The sections below document the full internal implementation — grader formulas, scoring edge cases, environment state machine, and GRPO hyperparameters.

---

## System Architecture

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
│  │(MIMIC-IV CSV)│  │T1/T2/T3/T4   │  │Gating / Noise      ││
│  └──────────────┘  └──────────────┘  └────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Data Layer — MIMIC-IV Episode Builder

**File:** `environment/old_episode_builder.py`

### Source Tables

| Table | Usage |
|-------|-------|
| `admissions.csv` | hadm_id, admission type, discharge location |
| `patients.csv` | gender, anchor_age |
| `diagnoses_icd.csv` + `d_icd_diagnoses.csv` | ICD codes, descriptions, sequence numbers |
| `prescriptions.csv` | medication orders |
| `pharmacy.csv` | `pharmacy_active` (discharge meds), `pharmacy_stopped` |
| `labevents.csv` | lab results with abnormal flags |
| `microbiologyevents.csv` | culture organisms |
| `chartevents.csv` | vitals (HR, BP, SpO2, GCS, temp) |
| `procedureevents.csv` | ICU procedures (ventilation hours, dialysis, lines) |
| `inputevents.csv` / `outputevents.csv` | fluid balance |
| `icustays.csv` | ICU unit names and LOS |
| `emar.csv` / `emar_detail.csv` | medication administration records |
| `drgcodes.csv` | DRG codes with severity and mortality scores |
| `transfers.csv` | care trajectory (ED → MICU → floor) |

### Complexity Classification

**Easy:** LOS ≤ 10 days, no ICU, ≤ 12 diagnoses, plain home discharge (21 patients — useless for training)

**Hard:** Any of: LOS > 14 days · ventilation > 24h · SNF/rehab/hospice/expired discharge · resistant organisms (MRSA, VRE, ESBL) · oliguria

**Medium:** Everything in between — 56% home_with_services, 39% home. Real discrimination required.

---

## Observation Space

| Field | Description | Used by |
|-------|-------------|---------|
| `diagnoses` | ICD codes + descriptions (capped at 15) | All tasks |
| `pharmacy_active` | Drugs active at discharge — only valid medication source | T2, T3, T4 |
| `pharmacy_stopped` | Drugs stopped during admission | T2 |
| `lab_flags` | Abnormal lab results (CRITICAL/HIGH/LOW/ABNORMAL) | T2, T4 |
| `icu_procedures` | Ventilation hours, dialysis, arterial/central lines | T1, T4 |
| `vitals` | HR, BP, SpO2, GCS (admission → discharge) | T1, T4 |
| `drg_codes` | DRG severity + mortality scores (1–4) | T1 |
| `microbiology` | Culture organisms + susceptibility | T2, T4 |
| `fluid_balance` | Net balance, `fluid_overloaded`, `oliguria` | T4 |
| `care_trajectory` | ED → MICU → floor path | T4 |
| `emar_summary` | Medication admin records + `active_at_discharge` | T3, T4 |
| `discharge_orders` | Finalized flag + order types | T1 |

### Task 2 Information Gating

Task 2 starts with demographics + top 5 diagnoses only. The agent requests more:

```json
{"task_id": 2, "information_request": ["labs", "medications", "microbiology"]}
```

### Task 4 Progressive Revelation

| Unlocks at step | Fields |
|----------------|--------|
| 1 | Demographics, diagnoses, DRG codes |
| 2 | `lab_flags` |
| 3 | `vitals`, `icu_procedures` |
| 4 | `medications`, `emar_summary` |
| 5 | `microbiology` |
| 6 | `fluid_balance` |
| 7 | `care_trajectory` |
| 8 | `discharge_orders` |

---

## Task 1 — Full Scoring Reference

### Disposition Tier Scoring

| Match level | Score |
|------------|-------|
| Exact canonical match | 1.00 |
| Same broad group | 0.50 |
| Clinically adjacent | 0.25 |
| No match | 0.00 |

**Broad groups:** `community` (home, ama) · `community_plus` (home_with_services) · `facility` (snf, rehab) · `end_of_life` (hospice, expired)

**Adjacency:** home ↔ home_with_services, ama · home_with_services ↔ snf · snf ↔ rehab, other · hospice ↔ snf, home_with_services, expired

### MIMIC Discharge Location Normalization

| Canonical | MIMIC strings |
|-----------|--------------|
| `home_with_services` | HOME HEALTH CARE, HOME WITH SERVICE, ASSISTED LIVING |
| `snf` | SKILLED NURSING FACILITY, LONG TERM CARE, NURSING HOME |
| `rehab` | REHABILITATION, REHAB FACILITY, INPATIENT REHAB |
| `hospice` | HOSPICE-MEDICAL FACILITY, HOSPICE-HOME, COMFORT CARE ONLY |
| `expired` | DIED IN ICU, DIED, EXPIRED, DECEASED |

---

## Task 2 — Full Scoring Reference

```
raw = 0.35 × specialty_F1
    + 0.25 × medication_F1
    + 0.25 × instruction_quality
    + 0.15 × discontinue_accuracy
    − hallucination_penalty   (max 0.10)
    − ghost_specialty_penalty (max 0.10)

final = max(0.0, min(1.0, raw)) × step_efficiency_discount
```

### ICD → Specialty Mappings

**ICD-10 (first letter):** A/B → Infectious Disease · C → Oncology · D → Hematology · E → Endocrinology · F → Psychiatry · G → Neurology · I → Cardiology · J → Pulmonology · K → Gastroenterology · M → Rheumatology · N → Nephrology · S → Trauma Surgery

**ICD-9 (numeric):** 001–139 → Infectious Disease · 140–239 → Oncology · 390–459 → Cardiology · 460–519 → Pulmonology · 520–579 → Gastroenterology · 580–629 → Nephrology

**Microbiology:** staphylococcus, klebsiella, pseudomonas, candida, MRSA, VRE → Infectious Disease

### Ghost Specialty Penalty

Fires when a predicted specialty has no supporting ICD code. Penalty: `min(0.10, ghosts × 0.05)`. Skipped for "primary care", "general medicine", "hospitalist".

---

## Task 3 — Full Scoring Reference

```
raw = 0.30 × diagnosis_coverage
    + 0.20 × disposition_accuracy
    + 0.20 × medication_F1
    + 0.15 × LOS_accuracy
    + 0.10 × structure_score
    + 0.05 × information_density

halluc_penalty   = min(0.15, hallucination_rate × 0.15)
followup_penalty = 0.05  (if discharge_planning_finalized=True but no follow-up section)

final = max(0.0, raw − halluc_penalty − followup_penalty)
```

**Required disposition phrases (verbatim):**

| Disposition | Required phrase |
|-------------|----------------|
| home | "The patient was discharged home." |
| home_with_services | "The patient was discharged home with home health services." |
| snf | "The patient was transferred to a skilled nursing facility." |
| rehab | "The patient was transferred to inpatient rehabilitation." |
| hospice | "The patient was transitioned to hospice care." |
| expired | "The patient expired during this hospitalization." |

**Anti-stuffing rules:**
- Keyword density > 0.08 → coverage halved
- Sentence matching ≥3 diagnoses simultaneously → discarded
- Sentence must be ≥5 words to count
- LOS tolerance: ±25% of actual days

---

## Task 4 — Full Scoring Reference

| Step | Max contribution | Focus |
|------|-----------------|-------|
| 1 | 0.10 | Triage acuity |
| 2 | 0.15 | Lab + consult prioritization |
| 3 | 0.15 | Intervention selection |
| 4 | 0.10 | High-risk medication identification |
| 5 | 0.10 | Antibiotic stewardship |
| 6 | 0.08 | Fluid management |
| 7 | 0.08 | Stepdown readiness |
| 8 | 0.10 | Disposition + LOS forecasting |
| 9 | 0.10 | Medication reconciliation |
| 10 | 1.00 cap | Final note (composite) |

**Step 10 composite:**
```
note_score  = Task3Grader.grade(final_note, episode)
shaping_avg = mean([shaping_log[i] / step_max[i] for i in 1..9])

raw = 0.60 × note_score + 0.40 × shaping_avg

consistency_bonus = +0.10 if step-8 disposition found in final_note
trajectory_bonus  = +0.05 if ≥50% of step-2 specialties appear in note
revision_cost     = −0.02 × revisions_used  (max 2)

final = min(1.0, max(0.0, raw + bonuses − revision_cost))
```

---

## GRPO Training Pipeline

**File:** `training/train_grpo.py`

### Model Configuration

- Model: Qwen/Qwen2.5-3B-Instruct (bfloat16)
- Adapter: LoRA r=16, alpha=32, dropout=0.05, all linear layers
- Max sequence length: 2560 tokens
- 4-bit NF4 quantization when available

### GRPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_generations` | 8 | Minimum for non-degenerate advantage estimation |
| `batch_size` | 2 | L4 24GB VRAM constraint |
| `grad_accum` | 8 | Effective batch = 16 |
| `learning_rate` | 5e-6 | Conservative for small RL budget |
| `beta` | 0.04 | KL penalty — prevents mode collapse |
| `top_entropy_quantile` | 0.8 | Drop bottom 20% entropy outputs |
| `temperature` | 1.3 (T1) / 1.1 (T2) / 0.9 (T3/T4) | Higher early to escape hospice collapse |

### Dead-Gradient Guard

| Task | Halt threshold |
|------|---------------|
| T1 | 80% zero-reward |
| T2–T4 | 90% zero-reward |

Zero defined as reward < 0.12.

---

## 11. Server API Reference

**Base URL:** `http://localhost:7860`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Start episode: `{"task_id": 1, "noise_level": "clean", "hadm_id": null}` |
| POST | `/step` | Submit action → reward + partial scores |
| GET | `/health` | Readiness probe |
| GET | `/state` | Ground truth + episode state (debug only) |
| GET | `/tasks` | Full task catalogue + scoring formulas |
| GET | `/metrics` | Per-route latency, per-task reward stats |
| GET | `/history?n=10` | Last N completed episodes |
| GET | `/episodes/by_complexity` | All hadm_ids grouped by tier |
| GET | `/docs` | Swagger UI |

**Noise levels:** `clean` (full data) · `partial` (30% labs dropped) · `noisy` (scrambled sequences + missing fields)

---

## End-to-End Data Flow

```
1. Server startup
   ├── EpisodeBuilder loads all MIMIC CSV tables (~2s)
   ├── Classifies all 233 admissions into easy/medium/hard
   └── Initializes MIMICDischargeEnv with 4 task graders

2. POST /reset  (task_id=2, noise_level="clean", curriculum_mode="medium_only")
   ├── Sample hadm_id from medium pool (109 patients)
   ├── Build episode dict from CSV joins
   ├── Apply noise masking
   └── Return Observation (demographics + top 5 diagnoses for T2)

3. POST /step  (information_request: ["labs", "medications", "microbiology"])
   ├── Unlock lab_flags, pharmacy_active/stopped, microbiology
   └── Return updated Observation (reward=0, done=False)

4. POST /step  (task2: {specialties, medications, instructions})
   ├── CarePlanGrader.grade(action, episode)
   └── Return StepResult (reward=0.62, done=True, partial_signals={...})

5. Training (GRPO)
   ├── build_seed_dataset() → 220 resets, store hadm_id per row
   ├── GRPOTrainer generates 8 completions per prompt
   ├── reward_fn() → reset to same hadm_id, score each completion
   ├── Compute group-relative advantages
   ├── Backprop through LoRA weights only (3B base frozen)
   └── Checkpoint every 50 steps
```

---

## curl Quick Reference

```bash
# Health
curl http://localhost:7860/health

# Task 1 — Disposition
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "noise_level": "clean"}' | jq .

curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "task1": {"disposition": "home_with_services", "reasoning": "Patient requires wound care."}}' | jq .

# Task 2 — Care Plan (optimal 2-step)
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 2, "noise_level": "clean", "curriculum_mode": "medium_only"}' | jq .

curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 2, "information_request": ["labs", "medications", "microbiology"]}' | jq .

# Task 4 — ICU Workflow (step 1)
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "curriculum_mode": "hard_only"}' | jq .

curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": 4, "task4": {"triage_level": "icu"}}' | jq .
```
---

## Presentation

Presentation: [View slides](https://canva.link/swa5x39lx1vlzwi)

---

## Citation

Johnson, A. E. W., et al.  
**MIMIC-IV (version 3.1)**, PhysioNet.  
https://physionet.org/content/mimiciv/3.1/

Johnson, A., et al.  
**MIMIC-IV Clinical Database Demo (version 2.2)**, PhysioNet.  
https://physionet.org/content/mimic-iv-demo/2.2/
