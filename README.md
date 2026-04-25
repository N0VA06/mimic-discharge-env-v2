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

A clinical RL benchmark where an LLM agent makes real hospital discharge decisions on 233 MIMIC-IV patient records.  Built for the **Meta × Scaler OpenEnv Hackathon**.

---

## The Problem

**Every year ~20% of Medicare patients are readmitted within 30 days — costing $26B and representing systematic failures in discharge planning.**

Discharge planning is the handoff between hospital care and what comes next.  A hospitalist must simultaneously decide:

- **Where** does this patient go? (home, SNF, rehab, hospice, ICU step-down?)
- **What** care continues at home? (which medications, which specialist follows up, what instructions?)
- **How** is this documented? (discharge note that downstream providers will actually rely on)

These decisions happen under extreme time pressure, with incomplete information, for dozens of patients at once.  Wrong calls cause readmissions, medication errors, delayed diagnoses, and in terminal cases — patients not reaching hospice in time.

**This benchmark makes the problem tractable for RL:** every decision has a ground-truth answer in the MIMIC-IV records, all graders are deterministic (no LLM judge), and the reward signal decomposes into clinically meaningful components.

---

## Environment

```
POST /reset  →  patient observation (structured EHR + demographics)
POST /step   →  reward (0–1) + partial component scores + done flag
GET  /health →  readiness probe
GET  /tasks  →  full task catalogue + scoring formulas
GET  /state  →  current episode state + ground truth (debug only)
```

**233 real hospital admissions** from MIMIC-IV demo, split by clinical complexity:

| Tier | Count | Profile |
|------|-------|---------|
| Easy | 21 | Short LOS, no ICU, plain home discharge — trivial, no discrimination needed |
| Medium | 109 | 56% home-with-services / 39% home — requires reading clinical features |
| Hard | 103 | SNF / hospice / ICU / expired — complex trajectories, rare outcomes |

The environment exposes **structured EHR data per patient**: ICD diagnoses, active/stopped medications, lab flags, vitals, ICU procedures (ventilation hours, dialysis), microbiology cultures, DRG severity/mortality codes, care trajectory, and discharge orders.

---

## Tasks

### Task 1 — Discharge Disposition *(1 step)*

Predict one of 8 canonical post-discharge placements from the patient's full EHR.

| Value | Clinical meaning |
|-------|-----------------|
| `home` | Fully independent, no professional follow-up |
| `home_with_services` | Visiting nurse, home PT, IV antibiotics, wound care |
| `snf` | Skilled nursing facility — 24h nursing + ongoing medical needs |
| `rehab` | Inpatient rehabilitation — intensive PT/OT, medically stable |
| `hospice` | Terminal prognosis, comfort-focused care only |
| `ama` | Left against medical advice |
| `expired` | Patient died during this admission |
| `other` | Transfer to acute hospital or psychiatric unit |

**Scoring:** Exact match = 1.0 · Clinically adjacent (e.g. SNF ↔ home-with-services) = 0.50 · Same broad group = 0.25 · Wrong group = 0.0.
Key signals: ICD V667/Z51.5 → hospice · DRG mortality=4 + malignant neoplasm → hospice · ventilation/dialysis → SNF · orthopedic fixation → rehab · GCS ≤ 5 + terminal dx → hospice.

---

### Task 2 — Care Plan Recommendation *(≤4 steps, efficiency-discounted)*

A **gated multi-step** task.  The agent starts with minimal data (demographics + top diagnoses only), then requests additional information before submitting a care plan.

**Optimal strategy:** Request `labs + medications + microbiology` on step 1, submit plan on step 2 — earns a 1.0× efficiency multiplier.  Step 3 = 0.85×.  Step 4 = 0.70×.

The submitted plan must specify:
- **Follow-up specialties** derived from ICD codes (Cardiology from `I` prefix, Nephrology from `N`, Infectious Disease from organisms in microbiology, etc.)
- **Medications to continue** — exact names from `pharmacy_active` only
- **Medications to discontinue** — exact names from `pharmacy_stopped` only
- **5 specific home instructions** with numeric thresholds (e.g. "call if temperature > 38.5°C")

**Scoring:** `0.35 × specialty_F1 + 0.25 × medication_F1 + 0.25 × instruction_quality + 0.15 × discontinue_accuracy − hallucination_penalty − ghost_specialty_penalty × step_efficiency`.

Ghost specialty penalty fires when the agent recommends a specialty with no supporting ICD code.

---

### Task 3 — Discharge Note Generation *(1 step)*

Write a complete clinical discharge summary (minimum 300 words) covering all 7 required sections in order:

1. PRINCIPAL DIAGNOSIS
2. BRIEF HOSPITAL COURSE *(must state LOS in days numerically)*
3. KEY PROCEDURES PERFORMED
4. DISCHARGE CONDITION
5. DISCHARGE DISPOSITION *(exact canonical phrase — 6 options)*
6. DISCHARGE MEDICATIONS *(only `pharmacy_active` drugs by exact name)*
7. FOLLOW-UP INSTRUCTIONS

**Scoring:** `0.30 × diagnosis_coverage + 0.20 × disposition_accuracy + 0.20 × medication_F1 + 0.15 × LOS_accuracy + 0.10 × structure_score + 0.05 × information_density − hallucination_penalty`.

Anti-stuffing: diagnosis keywords must appear in sentences ≥5 words.  A sentence matching ≥3 diagnoses simultaneously scores zero (vague catch-all detection).  Drug names verified against both `prescriptions` and `emar_summary`.

---

### Task 4 — ICU Admission-to-Discharge Workflow *(10 steps, sparse reward)*

Sequential clinical decision-making across a full ICU admission.  Sparse reward: steps 1–9 return 0, only step 10 (final discharge note) returns a score.

| Step | Decision |
|------|----------|
| 1 | Acuity triage: ICU / stepdown / floor |
| 2 | Priority labs + specialist consults |
| 3 | Interventions: ventilation, dialysis, lines |
| 4 | High-risk medication identification |
| 5 | Antibiotic stewardship plan |
| 6 | Fluid management strategy |
| 7 | ICU-to-stepdown readiness + barriers |
| 8 | Predicted disposition + LOS estimate |
| 9 | Discharge medication reconciliation |
| 10 | Final discharge note (composite reward) |

Step 10 score = `note_score × 0.60 + shaping_avg × 0.40 + consistency_bonus (0.10) + trajectory_bonus (0.05) − revision_cost`.

---

## GRPO Training

**Model:** Qwen/Qwen2.5-3B-Instruct · bfloat16 · LoRA r=16  
**Framework:** TRL 0.23 GRPO · 8 generations/step · effective batch 16  
**Hardware:** NVIDIA L4 24 GB · ~35 s/step (T1/T2) · ~55 s/step (T3/T4)

**Curriculum (550 steps, ~7 hours on L4):**

| Phase | Steps | Task | Patient pool | Seed dataset |
|-------|-------|------|-------------|--------------|
| 1 | 0–199 | Disposition | medium_only (109) | 220 samples |
| 2 | 200–349 | Care Plan | medium_only (109) | 220 samples |
| 3 | 350–449 | Discharge Note | random (all 233) | 466 samples |
| 4 | 450–549 | ICU Workflow | hard_only (103) | 210 samples |

Seed dataset auto-scales to 2× the active patient pool so every unique patient appears at least twice per chunk.

---

## Training Results

### Reward Curve

![Reward Curve](logs/plots/01_reward_curve.png)

| Phase | Task | Steps | Result |
|-------|------|-------|--------|
| Phase 1 | Disposition | 0–199 | Rises from **0.24 → 0.73** — model learns HOME vs HOME_WITH_SERVICES from clinical features |
| Phase 2 | Care Plan | 200–349 | Stabilizes at **~0.58–0.65** — specialty + medication F1 both contributing |
| Phase 3 | Discharge Note | 350–435 | **~0.40–0.55** with high variance — long-form generation, harder reward signal |

**Overall mean reward: 0.517** across all tasks and steps.

### Curriculum Phase Timeline

![Phase Timeline](logs/plots/05_phase_timeline.png)

Rolling-50 reward peaks at **0.73 at step ~195** (end of Task 1 phase), dips at the T1→T2 curriculum switch (new task schema), then recovers and holds at ~0.58–0.68 through Task 2.  Task 3 brings a natural dip as the model adapts to long-form generation.

### JSON Parse Rate

![Parse Rate](logs/plots/02_parse_rate.png)

Stays at **≥95% through Tasks 1 and 2**.  Drops to ~85% in Task 3 (discharge notes have longer, more complex JSON) — still well above the 80% target floor.

### Reward Distribution

![Reward Histogram](logs/plots/06_reward_histogram.png)

Bimodal: cluster at 0.24–0.44 (partial credit — adjacent disposition or incomplete note) and a spread from 0.50–0.80 (good clinical reasoning).  Spike at 1.0 = exact matches on Task 1 dispositions.

---

## Hard Problems Solved During Training

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Hospice mode collapse** | Model output `hospice` 100% for 42 steps | Switched Phase 1 from easy_only (21 HOME-only patients) to medium_only (109 patients, diverse outcomes) |
| **Prompt-reward decoupling** | Reward scored a random patient, not the one in the prompt | Store `hadm_id` in seed dataset; reward function pins `/reset` to that specific patient |
| **Disposition string mismatch** | `"home with services"` scored 0.44 instead of 1.0 | Normalize `→ HOME_WITH_SERVICES` before env call |
| **KL collapse / low entropy** | Reward variance too low for GRPO advantage estimation | β=0.04 KL penalty + top_entropy_quantile=0.8 (drop bottom 20% entropy completions) |

---

## Project Structure

```
├── environment/
│   ├── env.py                    — MIMICDischargeEnv core (Tasks 1–4)
│   ├── old_episode_builder.py    — MIMIC-IV episode builder + complexity tiers
│   ├── models.py                 — Pydantic Action / Observation / StepResult
│   └── tasks/
│       ├── task1_disposition.py  — 3-tier scorer (exact / adjacent / wrong)
│       ├── task2_careplan.py     — specialty F1 + medication F1 + instruction grader
│       ├── task3_note.py         — anti-stuffing discharge note grader
│       └── task4_workflow.py     — 10-step sparse reward ICU workflow
├── server/app.py                 — FastAPI server (v3.1.0)
├── training/
│   ├── train_grpo.py             — GRPO curriculum training (all 4 tasks)
│   └── rollout_collector.py      — offline LLM rollout → HF Dataset
├── inference.py                  — clinical LLM agent (Tasks 1–4)
└── openenv.yaml
```

---

## Observation Space

Every patient observation contains:

| Field | Description | Used by |
|-------|-------------|---------|
| `diagnoses` | ICD codes + descriptions, ranked by sequence | All tasks |
| `pharmacy_active` | Drugs active at discharge — **only valid source for medications** | T2, T3, T4 |
| `pharmacy_stopped` | Drugs stopped during admission | T2 |
| `lab_flags` | Abnormal lab results | T2 specialty derivation |
| `icu_procedures` | Ventilation hours, dialysis, arterial/central lines | T1 SNF rule; T4 triage |
| `vitals` | HR, BP, SpO2, GCS (admission → discharge) | T1 end-of-life detection |
| `drg_codes` | DRG severity + mortality scores (1–4) | T1 hospice (mortality=4) |
| `microbiology` | Culture organisms | T2 Infectious Disease specialty |
| `fluid_balance` | Net balance, `fluid_overloaded`, `oliguria` | T4 fluid strategy |
| `care_trajectory` | ED → MICU → floor path | T4 triage context |
| `emar_summary` | Medication admin records + `active_at_discharge` | T3 hallucination check |
| `discharge_orders` | Finalized flag + order types | T1 home indicator |

---

## Setup

```bash
pip install -r requirements.txt
python -m server.app           # start environment on :7860
curl localhost:7860/health     # → {"status":"ok","episodes_available":233}
```

**Train (7-hour L4 budget):**
```bash
python -m training.train_grpo \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --env_url http://localhost:7860 \
  --max_steps 550
```
Install unsloth accelerate trl datasets peft (pip install unsloth accelerate trl datasets peft)
**Collect offline rollouts:**
```bash
python -m training.rollout_collector \
  --task_id 1 --n_episodes 220 \
  --env_url http://localhost:7860 \
  --model_name Qwen/Qwen2.5-3B-Instruct
```

**Run inference agent:**
```bash
export HF_TOKEN=hf_...
python inference.py
```

**Docker:**
```bash
docker build -t mimic-discharge-env .
docker run -p 7860:7860 -e HF_TOKEN=hf_... mimic-discharge-env
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Readiness probe |
| `/reset` | POST | Start episode: `{"task_id": 1, "noise_level": "clean", "hadm_id": null}` |
| `/step` | POST | Submit action → reward + partial scores |
| `/state` | GET | Ground truth + episode state (debug) |
| `/tasks` | GET | Full task catalogue + scoring formulas |
| `/episodes/by_complexity` | GET | All hadm_ids grouped by easy/medium/hard |
| `/docs` | GET | Swagger UI |

**Noise levels:** `clean` (full data) · `partial` (30% labs dropped) · `noisy` (scrambled sequences + missing fields)

---

## Citation

Johnson et al., **MIMIC-IV (v2.2)**, PhysioNet. https://physionet.org/content/mimic-iv-demo/2.2/
