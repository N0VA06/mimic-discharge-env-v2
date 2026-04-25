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

A clinical RL environment where an LLM agent makes real hospital discharge decisions on real patient records (collected by MIT [MIMIC-IV](https://physionet.org/content/mimic-iv-demo/2.2/)). Built for the **Meta × Scaler OpenEnv Hackathon**.

---

## The Problem

**Every year ~20% of Medicare patients are readmitted within 30 days — costing $26B and representing systematic failures in discharge planning.**

Discharge planning is the handoff between hospital care and what comes next.  A attending phyiscian must simultaneously decide:

- Where patients go next (home, SNF, rehab, hospice?)
- What medications continue & who follows up
- What instructions go in the discharge note

All under time pressure with incomplete information.

**This environment is tractable for RL:** every decision has a ground-truth answer in MIMIC-IV, all graders are deterministic, reward decomposes into clinically meaningful signals.

---

## What You Get

| Component | Details |
|-----------|---------|
| **4 tasks** | disposition (easy) → care plan (medium) → discharge note (hard) → full workflow (very hard) |
| **233 episodes** | Stratified by complexity: easy (21) / medium (109) / hard (103) |
| **Structured EHR** | ICD diagnoses, medications, labs, vitals, ICU procedures, microbiology, DRG codes |
| **Deterministic graders** | No randomness; all scoring fully specified + explainable |
| **Partial rewards** | Every task emits component signals (specialty F1, medication F1, etc.) alongside scalar score |

---

## Environment

**233 real hospital admissions** sourced from the MIMIC-IV demo dataset, categorized by clinical complexity. This can be scaled to the **full MIMIC-IV dataset**, which includes over **50,000 patients**.

| Tier | Count | Profile |
|------|-------|---------|
| Easy | 21 | Short LOS, no ICU, plain home discharge — trivial, no discrimination needed |
| Medium | 109 | 56% home-with-services / 39% home — requires reading clinical features |
| Hard | 103 | SNF / hospice / ICU / expired — complex trajectories, rare outcomes |

The environment exposes **structured EHR data per patient**: ICD diagnoses, active/stopped medications, lab flags, vitals, ICU procedures (ventilation hours, dialysis), microbiology cultures, DRG severity/mortality codes, care trajectory, and discharge orders.

```
POST /reset  →  patient observation (structured EHR + demographics)
POST /step   →  reward (0–1) + partial component scores + done flag
GET  /health →  readiness probe
GET  /tasks  →  full task catalogue + scoring formulas
GET  /state  →  current episode state + ground truth (debug only)
```

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

> **Training:** Reward rises **0.24 → 0.73** over steps 0–199 as the model learns to read clinical features (DRG mortality, ventilation hours, ICD codes) to distinguish HOME / HOME\_WITH\_SERVICES / SNF / hospice.

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

> **Training:** Stabilizes at **~0.58–0.65** over steps 200–349 with specialty F1 and medication F1 both contributing. Efficiency multiplier rewards the optimal 2-step strategy (request labs/meds/micro → submit plan).

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

> **Training:** Reaches **~0.40–0.55** over steps 350–449 with high variance — long-form generation creates a harder reward signal. Parse rate drops to ~85% due to complex 7-section JSON structure.

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

> **Training:** Sparse reward (steps 1–9 return 0) makes this the hardest phase. The model transfers note-writing knowledge from Task 3 and adapts to the ICU workflow wrapper. Mean reward builds from **~0.26 → 0.38** over steps 450–549 as parse rate recovers and consistency/trajectory bonuses accumulate.

---

## GRPO Training

**Model:** Qwen/Qwen2.5-3B-Instruct  
**Fine-tuning:** LoRA rank=16 (**only 0.1–0.5% of parameters trained**)  
**Framework:** TRL 0.23 GRPO · 8 generations/step · effective batch 16  
**Hardware:** NVIDIA L4 24 GB · ~35 s/step (T1/T2) · ~55 s/step (T3/T4)

**Curriculum (550 steps, ~7 hours on L4):**

| Phase | Steps | Task | Patient pool | Seed dataset |
|-------|-------|------|-------------|--------------|
| 1 | 0–199 | Disposition | medium+easy (130) | 220 samples |
| 2 | 200–349 | Care Plan | medium (109) | 220 samples |
| 3 | 350–449 | Discharge Note | all (233) | 466 samples |
| 4 | 450–549 | ICU Workflow | hard (103) | 210 samples |

Each phase auto-scales seed dataset to 2× patient pool (every patient appears ≥2× per chunk).

[→ Full training details](explanation.md#training-pipeline)

---

## Training Results

### Reward Curve

![Reward Curve](logs/plots/01_reward_curve.png)

| Phase | Task | Steps | Difficulty | Fine-Tuned Model | Baseline (Llama-3.1-8B-Instruct) |
|-------|------|-------|------------|------------------|----------------------------------|
| Phase 1 | Disposition | 0–199 | Easy | **0.24 → 0.73** — learns HOME vs HOME_WITH_SERVICES from clinical features | **~0.05–0.30 (avg ~0.11)** — struggles with exact class matching under strict reward |
| Phase 2 | Care Plan | 200–349 | Medium | **~0.58–0.65** — stable specialty + medication F1 | **~0.60–0.73 (avg ~0.68)** — strong due to general reasoning ability |
| Phase 3 | Discharge Note | 350–449 | Hard | **~0.40–0.55** — high variance, long-form difficulty | **~0.43–0.59 (avg ~0.51)** — comparable performance |
| Phase 4 | ICU Workflow | 450–549 | Very Hard | **~0.26 → 0.38** — sparse reward, gradual improvement | **~0.27–0.65 (avg ~0.46)** — higher due to shaping + strong base generation |

> **Overall Average:**  
> Fine-tuned: **~0.53–0.54** vs Baseline: **~0.47** → **+13–15% improvement**

### Per-Task Learning Curves

![Per-Task Learning Curves](logs/plots/08_per_task_curves.png)

Each panel shows reward across that task's training steps.  Task 1 has the steepest learning curve (simple classification). Task 4 starts lower due to sparse reward and the 9-step zero-reward advance per episode, but improves steadily as the model adapts the note-writing skills from Task 3.

### Curriculum Phase Timeline

![Phase Timeline](logs/plots/05_phase_timeline.png)

Rolling-50 reward peaks at **0.73 at step ~195** (end of Task 1 phase), dips at each curriculum switch, then recovers. Task 4 shows a shallower dip than Task 2 because the model already knows how to write discharge notes — it only needs to adapt to the ICU workflow wrapper and sparse reward signal.

### JSON Parse Rate

![Parse Rate](logs/plots/02_parse_rate.png)

Stays at **≥95% through Tasks 1 and 2**.  Drops to ~85% in Task 3 (7-section JSON) and briefly to ~75% at the Task 4 switch (new `final_note` wrapper format), recovering to ~90% by step 549.

### Reward Distribution

![Reward Histogram](logs/plots/06_reward_histogram.png)

Bimodal: cluster at 0.24–0.44 (partial credit — adjacent disposition or incomplete note) and a spread from 0.50–0.80 (good clinical reasoning).  Spike at 1.0 = exact matches on Task 1 dispositions. Task 4 adds a mode at 0.30–0.40 reflecting the 0.60× note-score cap in the grader formula.

---

## Challenges Solved During Training

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Hospice mode collapse** | Model output `hospice` 100% for 42 steps | Automatically switches to include class diversity; if only a single class is present, it defaults to including the “easy” category. |
| **Prompt-reward decoupling** | Reward scored a random patient, not the one in the prompt | Store `hadm_id` in seed dataset; reward function pins `/reset` to that specific patient |
| **Disposition string mismatch** | `”home with services”` scored 0.44 instead of 1.0 | Normalize `→ HOME_WITH_SERVICES` before env call |
| **KL collapse / low entropy** | Reward variance too low for GRPO advantage estimation | β=0.04 KL penalty + top_entropy_quantile=0.8 (drop bottom 20% entropy completions) |

[→ Full technical deep-dive](explanation.md)

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
