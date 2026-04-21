from __future__ import annotations
import logging
import random
from typing import Optional, Dict, Any, Set

from .old_episode_builder import EpisodeBuilder
from .models import (
    Action, Observation, StepResult, StateInfo, ResetRequest,
    DiagnosisInfo, ICUStay, LabFlag, Medication,
    MicrobiologyResult, ICUProcedureSummary,
    VitalSign, FluidBalance, EmarMedication, DischargeOrders,
)
from .tasks.task1_disposition import DispositionGrader
from .tasks.task2_careplan    import CarePlanGrader
from .tasks.task3_note        import NoteGrader
from .tasks.task4_workflow    import Task4Grader

logger = logging.getLogger(__name__)

TASK_CONFIG: Dict[int, Dict[str, Any]] = {
    1: {
        "description": (
            "Based on the patient's clinical data, predict the most appropriate "
            "discharge disposition from: home, home_with_services, snf "
            "(skilled nursing facility), rehab, hospice, ama (against medical advice), "
            "expired, other."
        ),
        "action_space": (
            'JSON: {"task_id": 1, "task1": {"disposition": "<choice>", "reasoning": "..."}}\n'
            "Valid dispositions: home | home_with_services | snf | rehab | "
            "hospice | ama | expired | other"
        ),
        "max_steps": 1,
    },
    2: {
        "description": (
            "Recommend a comprehensive post-discharge care plan for this patient. "
            "This is a 4-step information revelation protocol. Initially you receive "
            "demographics, top 5 diagnoses, LOS, and complexity tier. "
            "Request additional data with information_request: "
            "labs | vitals | medications | microbiology | fluid_balance. "
            "Submit your final care plan after gathering sufficient information. "
            "Step efficiency multiplier: ≤2 steps=1.0×, 3 steps=0.85×, 4 steps=0.70×."
        ),
        "action_space": (
            "To request data:\n"
            'JSON: {"task_id": 2, "information_request": ["labs", "vitals", ...]}\n'
            "Valid request values: labs | vitals | medications | microbiology | fluid_balance\n\n"
            "To submit care plan:\n"
            'JSON: {"task_id": 2, "task2": {\n'
            '  "follow_up_specialties": ["Cardiology", ...],\n'
            '  "medications_to_continue": ["metoprolol", ...],\n'
            '  "medications_to_discontinue": ["heparin", ...],\n'
            '  "key_instructions": ["Weigh yourself daily", ...],\n'
            '  "reasoning": "..."\n}}'
        ),
        "max_steps": 4,
        "step_efficiency_discount": {1: 1.0, 2: 1.0, 3: 0.85, 4: 0.70},
    },
    3: {
        "description": (
            "Draft a complete clinical discharge summary note for this patient. "
            "Include: admission diagnosis, brief hospital course, key procedures performed, "
            "discharge condition, discharge disposition, discharge medications, "
            "and follow-up instructions. Aim for at least 300 words."
        ),
        "action_space": (
            'JSON: {"task_id": 3, "task3": {"discharge_note": "FULL TEXT NOTE..."}}'
        ),
        "max_steps": 1,
    },
    4: {
        "description": (
            "10-step admission-to-discharge workflow. You manage an ICU patient from "
            "triage through final discharge note. Each step focuses on one clinical decision. "
            "Step 1: acuity triage | Step 2: priority labs & consults | "
            "Step 3: interventions | Step 4: high-risk medications | "
            "Step 5: antibiotic plan | Step 6: fluid strategy | "
            "Step 7: ICU-to-stepdown readiness | Step 8: disposition & LOS | "
            "Step 9: medication reconciliation | Step 10: discharge note. "
            "Sparse reward: only Step 10 returns a non-zero reward. "
            "Optional: add revise_step + revision to correct a prior step (max 2, -0.02 each)."
        ),
        "action_space": (
            'JSON: {"task_id": 4, "task4": {<step-specific fields>}}\n'
            'Step 1:  {"triage_level": "icu|stepdown|floor"}\n'
            'Step 2:  {"priority_labs": [...], "priority_consults": [...]}\n'
            'Step 3:  {"interventions": [...]}\n'
            'Step 4:  {"high_risk_medications": [...]}\n'
            'Step 5:  {"antibiotic_strategy": "none|targeted|broad|empiric", "antibiotics": [...]}\n'
            'Step 6:  {"fluid_strategy": "restrict_diuresis|aggressive_resuscitation|maintain"}\n'
            'Step 7:  {"ready_for_stepdown": true/false, "barriers": [...]}\n'
            'Step 8:  {"predicted_disposition": "home|snf|...", "los_remaining_days": N}\n'
            'Step 9:  {"medications_to_continue": [...]}\n'
            'Step 10: {"final_note": "FULL DISCHARGE NOTE (>=300 words)..."}\n'
            'Revision: add "revise_step": N, "revision": {corrected fields for step N}'
        ),
        "max_steps": 10,
    },
}

# Categories requestable via information_request for Task 2
_TASK2_GATEABLE: Set[str] = {
    "labs", "vitals", "medications", "microbiology", "fluid_balance"
}
# Always-revealed categories at reset for Task 2
_TASK2_INITIAL: Set[str] = {
    "demographics", "admission_type", "hospital_los_days", "complexity"
}


class MIMICDischargeEnv:

    def __init__(self, data_root: Optional[str] = None) -> None:
        self.builder  = EpisodeBuilder(data_root)
        self._graders = {
            1: DispositionGrader(),
            2: CarePlanGrader(),
            3: NoteGrader(),
            4: Task4Grader(),
        }
        self._episode:           Optional[Dict[str, Any]] = None
        self._task_id:           int   = 1
        self._step_num:          int   = 0
        self._cumulative_reward: float = 0.0
        self._last_reward:       Optional[float] = None
        self._total_episodes:    int   = 0
        self.active:             bool  = False
        self._revealed_info:     Set[str] = set()
        self._noise_level:       str  = "clean"
        self._shaping_log:       Dict[str, Any] = {}
        self._episode_history:   list = []

    def reset(
        self,
        task_id: int = 1,
        hadm_id: Optional[int] = None,
        noise_level: str = "clean",
        curriculum_mode: str = "random",
    ) -> Observation:
        task_id = max(1, min(4, int(task_id)))

        if hadm_id is None:
            hadm_id = self._sample_hadm_id(curriculum_mode)

        self._episode           = self.builder.get_episode(hadm_id, noise_level=noise_level)
        self._task_id           = task_id
        self._step_num          = 0
        self._cumulative_reward = 0.0
        self._last_reward       = None
        self._noise_level       = noise_level
        self._total_episodes   += 1
        self.active             = True
        self._revealed_info     = set(_TASK2_INITIAL)
        self._shaping_log       = {}
        self._episode_history   = []

        logger.info(
            "reset() — task=%d hadm_id=%d subject_id=%d noise=%s curriculum=%s",
            task_id, self._episode["hadm_id"], self._episode["subject_id"],
            noise_level, curriculum_mode,
        )
        return self._build_observation()

    def _sample_hadm_id(self, curriculum_mode: str) -> int:
        ep_count = self._total_episodes

        if curriculum_mode == "easy_only":
            pool = self.builder.sample_by_complexity("easy")
        elif curriculum_mode == "medium_only":
            pool = self.builder.sample_by_complexity("medium")
        elif curriculum_mode == "hard_only":
            pool = self.builder.sample_by_complexity("hard")
        elif curriculum_mode == "progressive":
            if ep_count < 200:
                pool = self.builder.sample_by_complexity("easy")
            elif ep_count < 500:
                pool = self.builder.sample_by_complexity("medium")
            else:
                pool = self.builder.hadm_ids
        else:
            pool = self.builder.hadm_ids

        if not pool:
            pool = self.builder.hadm_ids

        return random.choice(pool)

    def step(self, action: Action) -> StepResult:
        if not self.active or self._episode is None:
            raise RuntimeError("Call reset() before step().")

        self._step_num += 1
        cfg       = TASK_CONFIG[self._task_id]
        max_steps = cfg["max_steps"]

        # ── Task 2: information gating protocol ───────────────────────────────
        if self._task_id == 2:
            # Information request step (takes priority over task2 submission)
            if action.information_request is not None:
                newly_revealed = [
                    r for r in action.information_request
                    if r in _TASK2_GATEABLE
                ]
                self._revealed_info.update(newly_revealed)
                next_obs = self._build_observation()

                if self._step_num >= max_steps:
                    self.active = False
                    return StepResult(
                        observation=next_obs,
                        reward=0.0,
                        done=True,
                        info={
                            "info_revealed":  sorted(self._revealed_info),
                            "warning": "max_steps reached during information request",
                        },
                        partial_signals={"info_request_step": 0.0},
                    )
                return StepResult(
                    observation=next_obs,
                    reward=0.0,
                    done=False,
                    info={
                        "info_revealed":  sorted(self._revealed_info),
                        "newly_revealed": sorted(newly_revealed),
                    },
                    partial_signals={"info_request_step": 0.0},
                )

            # Care plan submission
            if action.task2 is not None:
                reward, partial, _, info = self._graders[2].grade(
                    action, self._episode
                )
                discount_table = cfg.get("step_efficiency_discount", {})
                discount = discount_table.get(
                    min(self._step_num, 4),
                    0.70 if self._step_num > 4 else 1.0,
                )
                reward = round(reward * discount, 4)
                partial["step_efficiency_discount"] = discount

                self._cumulative_reward += reward
                self._last_reward        = reward
                self.active = False
                logger.info(
                    "Task 2 done — steps=%d discount=%.2f reward=%.4f cumulative=%.4f",
                    self._step_num, discount, reward, self._cumulative_reward,
                )
                return StepResult(
                    observation=None,
                    reward=reward,
                    done=True,
                    info=info,
                    partial_signals=partial,
                )

            # Neither info_request nor task2 provided
            if self._step_num >= max_steps:
                self.active = False
                return StepResult(
                    observation=None,
                    reward=0.0,
                    done=True,
                    info={"error": "max_steps reached without task2 submission"},
                    partial_signals={},
                )
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=False,
                info={"hint": "provide information_request or task2 care plan"},
                partial_signals={},
            )

        # ── Task 4: 10-step sparse-reward workflow ────────────────────────────
        if self._task_id == 4:
            reward, partial, done_grader, info = self._graders[4].grade(
                action, self._episode, self._step_num, self._shaping_log
            )
            done = done_grader or (self._step_num >= TASK_CONFIG[4]["max_steps"])

            # Track episode history for observation context
            summary = self._format_action_summary(action, self._step_num)
            self._episode_history.append({
                "step_num":       self._step_num,
                "action_summary": summary,
                "reward":         reward,
            })

            self._cumulative_reward += reward
            self._last_reward        = reward

            if done:
                self.active = False
                logger.info(
                    "Task 4 done — steps=%d reward=%.4f cumulative=%.4f",
                    self._step_num, reward, self._cumulative_reward,
                )

            next_obs = self._build_observation() if not done else None
            return StepResult(
                observation=next_obs,
                reward=round(reward, 4),
                done=done,
                info=info,
                partial_signals={k: float(v) for k, v in partial.items()
                                 if isinstance(v, (int, float))},
            )

        # ── Task 1 and 3: single-step grading ─────────────────────────────────
        reward, partial, done_grader, info = self._graders[self._task_id].grade(
            action, self._episode
        )
        done = done_grader or (self._step_num >= max_steps)

        if self._step_num > max_steps:
            penalty = 0.1
            reward  = max(0.0, reward - penalty)
            partial["excess_step_penalty"] = -penalty

        self._cumulative_reward += reward
        self._last_reward        = reward

        if done:
            self.active = False
            logger.info(
                "Episode done — task=%d steps=%d reward=%.4f cumulative=%.4f",
                self._task_id, self._step_num, reward, self._cumulative_reward,
            )

        next_obs = self._build_observation() if not done else None
        return StepResult(
            observation=next_obs,
            reward=round(reward, 4),
            done=done,
            info=info,
            partial_signals=partial,
        )

    def _format_action_summary(self, action: Action, step_num: int) -> str:
        """One-line summary of the action taken at the given step (for episode history)."""
        if self._task_id == 4 and action.task4:
            t4 = action.task4
            if step_num == 1 and t4.triage_level:
                return f"Triaged as {t4.triage_level}"
            if step_num == 2:
                specs = ", ".join((t4.priority_consults or [])[:3]) or "none"
                return f"Consults: {specs}"
            if step_num == 3:
                ivs = ", ".join((t4.interventions or [])[:3]) or "none"
                return f"Interventions: {ivs}"
            if step_num == 4:
                meds = ", ".join((t4.high_risk_medications or [])[:3]) or "none"
                return f"High-risk meds: {meds}"
            if step_num == 5:
                return f"Antibiotic strategy: {t4.antibiotic_strategy or 'unspecified'}"
            if step_num == 6:
                return f"Fluid strategy: {t4.fluid_strategy or 'unspecified'}"
            if step_num == 7:
                return f"Stepdown ready: {t4.ready_for_stepdown}"
            if step_num == 8:
                return f"Dispo: {t4.predicted_disposition}, LOS est: {t4.los_remaining_days}d"
            if step_num == 9:
                meds = ", ".join((t4.medications_to_continue or [])[:3]) or "none"
                return f"Continue meds: {meds}"
        if self._task_id == 1 and action.task1:
            return f"Disposition: {action.task1.disposition}"
        if self._task_id == 2 and action.information_request:
            return f"Requested: {', '.join(action.information_request)}"
        return f"Step {step_num} action"

    def state(self) -> StateInfo:
        gt: Optional[Dict[str, Any]] = None
        if self._episode:
            gt = {
                "discharge_location": self._episode.get("discharge_location"),
                "hospital_los_days":  round(float(self._episode.get("hospital_los_days") or 0), 2),
                "n_diagnoses":        len(self._episode.get("diagnoses", [])),
                "n_medications":      len(self._episode.get("medications", [])),
                "n_icu_stays":        len(self._episode.get("icu_stays", [])),
                "n_microbiology":     len(self._episode.get("microbiology", [])),
                "ventilation_hours":  self._episode.get("icu_procedure_summary", {}).get(
                    "ventilation_hours", 0
                ),
                "complexity":         self._episode.get("complexity", "medium"),
            }
        return StateInfo(
            current_task_id=self._task_id,
            current_subject_id=int(self._episode["subject_id"]) if self._episode else None,
            current_hadm_id=int(self._episode["hadm_id"])       if self._episode else None,
            step_num=self._step_num,
            total_episodes_run=self._total_episodes,
            last_reward=self._last_reward,
            ground_truth=gt,
            task4_shaping_log=self._shaping_log if self._task_id == 4 else None,
        )

    # ─── Observation builder ──────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        ep      = self._episode
        cfg     = TASK_CONFIG[self._task_id]
        task2   = (self._task_id == 2)
        task4   = (self._task_id == 4)

        # Task 4 progressively reveals more fields per step
        _T4_THRESHOLDS: Dict[str, int] = {
            "labs": 2, "vitals": 3, "icu_procedures": 3, "medications": 4,
            "emar": 4, "microbiology": 5, "fluid_balance": 6,
            "care_trajectory": 7, "discharge_orders": 8,
        }

        def _gate(category: str, value: Any, default: Any) -> Any:
            if task2:
                return value if category in self._revealed_info else default
            if task4:
                threshold = _T4_THRESHOLDS.get(category, 0)
                return value if self._step_num >= threshold else default
            return value

        # ── Always visible ────────────────────────────────────────────────────
        age            = int(ep.get("age", 65) or 65)
        gender         = str(ep.get("gender", ""))
        admission_type = str(ep.get("admission_type", ""))
        admission_loc  = str(ep.get("admission_location", ""))
        insurance      = str(ep.get("insurance", ""))
        language       = str(ep.get("language", "ENGLISH"))
        los            = round(float(ep.get("hospital_los_days", 0) or 0), 2)
        complexity     = str(ep.get("complexity", "medium"))
        noise_level    = str(ep.get("noise_level", "clean"))

        # ── Diagnoses: Task 2 shows top 5 initially ────────────────────────────
        all_dx = [
            DiagnosisInfo(
                icd_code=str(d.get("icd_code", "")),
                icd_version=int(d.get("icd_version", 10) or 10),
                description=str(d.get("long_title", "Unknown")),
                seq_num=int(d.get("seq_num", 99) or 99),
            )
            for d in ep.get("diagnoses", [])[:15]
        ]
        diagnoses = all_dx[:5] if task2 else all_dx

        # ── Gated fields ──────────────────────────────────────────────────────
        icu_stays = _gate("icu_stays", [
            ICUStay(
                stay_id=int(s.get("stay_id", 0) or 0),
                los_days=round(float(s.get("los", 0) or 0), 2),
                first_careunit=str(s.get("first_careunit", "")),
                last_careunit=str(s.get("last_careunit", "")),
            )
            for s in ep.get("icu_stays", [])
        ], [])

        medications = _gate("medications", [
            Medication(
                drug=str(m.get("drug", "")),
                route=str(m["route"]) if m.get("route") else None,
                dose_val_rx=str(m["dose_val_rx"]) if m.get("dose_val_rx") else None,
            )
            for m in ep.get("medications", [])
        ], [])

        lab_flags = _gate("labs", [
            LabFlag(
                label=str(l.get("label", "")),
                flag=str(l.get("flag", "")),
                value=str(l["value"]) if l.get("value") else None,
            )
            for l in ep.get("lab_flags", [])
        ], [])

        procedures = _gate("procedures", [
            str(p.get("long_title") or p.get("icd_code", ""))
            for p in ep.get("procedures", [])[:10]
            if p.get("long_title") or p.get("icd_code")
        ], [])

        drg_codes_list = []
        for d in ep.get("drgcodes", [])[:5]:
            if not d.get("drg_code"):
                continue
            label = f"{d['drg_code']} – {d.get('description', '')}"
            sev   = d.get("drg_severity")
            mort  = d.get("drg_mortality")
            if sev and str(sev) != "nan":
                label += f"  [severity={sev}"
                if mort and str(mort) != "nan":
                    label += f", mortality={mort}"
                label += "]"
            drg_codes_list.append(label)
        drg_codes = _gate("drg_codes", drg_codes_list, [])

        microbiology = _gate("microbiology", [
            MicrobiologyResult(
                specimen=m.get("specimen", "UNKNOWN"),
                organism=m.get("organism", ""),
                resistant_to=m.get("resistant_to", []),
                sensitive_to=m.get("sensitive_to", []),
            )
            for m in ep.get("microbiology", [])
        ], [])

        icu_proc_raw = ep.get("icu_procedure_summary") or {}
        icu_proc_full = ICUProcedureSummary(
            ventilation_hours=float(icu_proc_raw.get("ventilation_hours", 0) or 0),
            has_arterial_line=bool(icu_proc_raw.get("has_arterial_line", False)),
            has_central_line=bool(icu_proc_raw.get("has_central_line",  False)),
            has_dialysis=bool(icu_proc_raw.get("has_dialysis",           False)),
            procedure_names=icu_proc_raw.get("procedure_names",          []),
        )
        icu_procedures = _gate("icu_procedures", icu_proc_full, ICUProcedureSummary())

        weight_kg = _gate("medications", ep.get("weight_kg"), None)
        bmi       = _gate("medications", ep.get("bmi"), None)

        care_trajectory  = _gate("care_trajectory",   ep.get("care_trajectory", []),  [])
        hcpcs_categories = _gate("hcpcs_categories",  ep.get("hcpcs_categories", []), [])
        pharmacy_active  = _gate("medications",        ep.get("pharmacy_active",  []), [])
        pharmacy_stopped = _gate("medications",        ep.get("pharmacy_stopped", []), [])

        # ── v3 gated fields ───────────────────────────────────────────────────
        vitals_dict = ep.get("vitals") or {}
        vitals_full = [
            VitalSign(
                name=name,
                admission_value=v.get("admission_value"),
                discharge_value=v.get("discharge_value"),
                min_value=v.get("min_value"),
                max_value=v.get("max_value"),
                critical_flag=bool(v.get("critical_flag", False)),
            )
            for name, v in vitals_dict.items()
        ]
        vitals = _gate("vitals", vitals_full, [])

        fb_raw = ep.get("fluid_balance") or {}
        fluid_balance_full: Optional[FluidBalance] = None
        if fb_raw and "total_output_ml" in fb_raw:
            try:
                fluid_balance_full = FluidBalance(
                    total_input_ml=float(fb_raw.get("total_input_ml", 0)),
                    total_urine_ml=float(fb_raw.get("total_urine_ml", 0)),
                    total_output_ml=float(fb_raw.get("total_output_ml", 0)),
                    net_balance_ml=float(fb_raw.get("net_balance_ml", 0)),
                    fluid_overloaded=bool(fb_raw.get("fluid_overloaded", False)),
                    oliguria=bool(fb_raw.get("oliguria", False)),
                )
            except Exception:
                pass
        fluid_balance = _gate("fluid_balance", fluid_balance_full, None)

        emar_full = [
            EmarMedication(
                medication=e.get("medication", ""),
                first_given=str(e.get("first_given", "")),
                last_given=str(e.get("last_given", "")),
                total_doses=int(e.get("total_doses", 0)),
                active_at_discharge=bool(e.get("active_at_discharge", False)),
            )
            for e in ep.get("emar_summary", [])
        ]
        emar_summary = _gate("medications" if task2 else "emar", emar_full, [])

        do_raw = ep.get("discharge_orders") or {}
        discharge_orders_full: Optional[DischargeOrders] = None
        if do_raw:
            discharge_orders_full = DischargeOrders(
                discharge_planning_finalized=bool(
                    do_raw.get("discharge_planning_finalized", False)
                ),
                documented_discharge_orders=do_raw.get("documented_discharge_orders", []),
            )
        discharge_orders = _gate("procedures" if task2 else "discharge_orders", discharge_orders_full, None)

        # Episode history: only for Task 4, from step 2 onward
        ep_history = list(self._episode_history) if task4 else []

        return Observation(
            task_id=self._task_id,
            subject_id=int(ep.get("subject_id", 0)),
            hadm_id=int(ep.get("hadm_id", 0)),
            step_num=self._step_num,
            max_steps=cfg["max_steps"],
            age=age,
            gender=gender,
            admission_type=admission_type,
            admission_location=admission_loc,
            insurance=insurance,
            language=language,
            diagnoses=diagnoses,
            icu_stays=icu_stays,
            medications=medications,
            lab_flags=lab_flags,
            procedures=procedures,
            drg_codes=drg_codes,
            hospital_los_days=los,
            microbiology=microbiology,
            weight_kg=weight_kg,
            bmi=bmi,
            care_trajectory=care_trajectory,
            icu_procedures=icu_procedures,
            hcpcs_categories=hcpcs_categories,
            pharmacy_active=pharmacy_active,
            pharmacy_stopped=pharmacy_stopped,
            vitals=vitals,
            fluid_balance=fluid_balance,
            emar_summary=emar_summary,
            discharge_orders=discharge_orders,
            complexity=complexity,
            noise_level=noise_level,
            task_description=cfg["description"],
            action_space_description=cfg["action_space"],
            episode_history=ep_history,
        )
