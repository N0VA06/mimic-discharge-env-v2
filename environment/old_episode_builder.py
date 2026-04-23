from __future__ import annotations

import os
import random
import logging
from typing import Dict, Any, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = os.environ.get(
    "MIMIC_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "mimic-iv-clinical-database-demo-2.2"),
)

_VENTILATION_ITEMS   = {225792, 225794, 224332}
_INVASIVE_LINE_ITEMS = {224263, 220339, 228167}
_DIALYSIS_ITEMS      = {225805, 225809, 225955}

_VITAL_ITEMS: Dict[int, str] = {
    220045: "Heart Rate",
    220179: "Systolic BP",
    220180: "Diastolic BP",
    220277: "SpO2",
    223761: "Temperature F",
    220210: "Respiratory Rate",
    223900: "GCS Total",
}

# Discharge locations that imply high clinical complexity.
# "HOME HEALTH CARE" and "AGAINST ADVICE" removed: home-with-services and AMA
# are medium complexity, not hard — keeping them here shrank the hard pool
# excessively while also leaving medium under-represented.
_HARD_DISCHARGE_LOCATIONS = {
    "SKILLED NURSING FACILITY",
    "REHAB",
    "HOSPICE",
    "DIED",
    "CHRONIC/LONG TERM ACUTE CARE",
    "OTHER FACILITY",
}

# Fragments that indicate plain home (no professional services at home).
# Used by classify_complexity to separate "easy" from "medium" home discharges.
_PLAIN_HOME_FRAGMENTS = {"HOME", "SELF", "DISCHARGED TO HOME", "RETURNED HOME"}
_HOME_SERVICE_FRAGMENTS = {"HEALTH CARE", "WITH SERVICE", "WITH AIDE", "WITH VNA", "ASSISTED"}


class EpisodeBuilder:
    """Loads all MIMIC tables once and serves enriched episode dicts on demand."""

    def __init__(self, data_root: Optional[str] = None) -> None:
        self.data_root = os.path.abspath(data_root or _DEFAULT_ROOT)
        self.hosp    = os.path.join(self.data_root, "hosp")
        self.icu_dir = os.path.join(self.data_root, "icu")

        logger.info("Loading MIMIC tables from %s", self.data_root)
        self._load_tables()
        self._build_episode_index()
        self._build_complexity_index()
        logger.info(
            "Episode builder ready — %d hospitalisations available",
            len(self.hadm_ids),
        )

    # ─── Table loading ────────────────────────────────────────────────────────

    def _csv(self, folder: str, name: str, **read_kwargs) -> pd.DataFrame:
        path = os.path.join(folder, f"{name}.csv")
        if not os.path.exists(path):
            logger.warning("Table not found, skipping: %s", path)
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False, **read_kwargs)
        logger.debug("  Loaded %s: %d rows", name, len(df))
        return df

    def _load_tables(self) -> None:
        h   = self.hosp
        icu = self.icu_dir

        # ── Core tables ───────────────────────────────────────────────────────
        self.admissions    = self._csv(h, "admissions")
        self.patients      = self._csv(h, "patients")
        self.diagnoses     = self._csv(h, "diagnoses_icd")
        self.d_icd_dx      = self._csv(h, "d_icd_diagnoses")
        self.prescriptions = self._csv(h, "prescriptions")
        self.procedures    = self._csv(h, "procedures_icd")
        self.d_icd_px      = self._csv(h, "d_icd_procedures")
        self.drgcodes      = self._csv(h, "drgcodes")
        self.labevents     = self._csv(h, "labevents")
        self.d_labitems    = self._csv(h, "d_labitems")
        self.services      = self._csv(h, "services")
        self.icustays      = self._csv(icu, "icustays")

        # ── v2 tables ─────────────────────────────────────────────────────────
        self.microbiology   = self._csv(h,   "microbiologyevents")
        self.omr            = self._csv(h,   "omr")
        self.transfers      = self._csv(h,   "transfers")
        self.icu_procedures = self._csv(icu, "procedureevents")
        self.hcpcsevents    = self._csv(h,   "hcpcsevents")
        self.pharmacy       = self._csv(h,   "pharmacy")

        # ── v3 tables ─────────────────────────────────────────────────────────
        # chartevents is large; load only needed columns and filter warning==0
        self._chartevents_cache: Optional[pd.DataFrame] = None
        chart_path = os.path.join(icu, "chartevents.csv")
        if os.path.exists(chart_path):
            try:
                ce = pd.read_csv(
                    chart_path,
                    usecols=["subject_id", "hadm_id", "itemid",
                             "charttime", "valuenum", "warning"],
                    low_memory=False,
                )
                ce = ce[ce["warning"] == 0].copy()
                ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")
                self._chartevents_cache = ce
                logger.debug("  Loaded chartevents (warning==0): %d rows", len(ce))
            except Exception as exc:
                logger.warning("Could not load chartevents: %s", exc)

        self._inputevents: pd.DataFrame = self._csv(icu, "inputevents")
        self._outputevents: pd.DataFrame = self._csv(icu, "outputevents")

        self._emar: pd.DataFrame = pd.DataFrame()
        emar_path = os.path.join(h, "emar.csv")
        if os.path.exists(emar_path):
            try:
                self._emar = pd.read_csv(
                    emar_path,
                    usecols=["hadm_id", "charttime", "medication"],
                    low_memory=False,
                )
                self._emar["charttime"] = pd.to_datetime(
                    self._emar["charttime"], errors="coerce"
                )
            except Exception as exc:
                logger.warning("Could not load emar: %s", exc)

        self._poe        = self._csv(h, "poe")
        self._poe_detail = self._csv(h, "poe_detail")

        # ── Enrichments ───────────────────────────────────────────────────────
        if not self.diagnoses.empty and not self.d_icd_dx.empty:
            self.diagnoses = self.diagnoses.merge(
                self.d_icd_dx[["icd_code", "icd_version", "long_title"]],
                on=["icd_code", "icd_version"], how="left",
            )

        if not self.procedures.empty and not self.d_icd_px.empty:
            self.procedures = self.procedures.merge(
                self.d_icd_px[["icd_code", "icd_version", "long_title"]],
                on=["icd_code", "icd_version"], how="left",
            )

        if not self.labevents.empty and not self.d_labitems.empty:
            self.labevents = self.labevents.merge(
                self.d_labitems[["itemid", "label"]], on="itemid", how="left",
            )

        for col in ["admittime", "dischtime"]:
            self.admissions[col] = pd.to_datetime(self.admissions[col], errors="coerce")
        self.admissions["hospital_los_days"] = (
            (self.admissions["dischtime"] - self.admissions["admittime"])
            .dt.total_seconds() / 86_400
        ).clip(lower=0)

        if not self.microbiology.empty and "chartdate" in self.microbiology.columns:
            self.microbiology["chartdate"] = pd.to_datetime(
                self.microbiology["chartdate"], errors="coerce"
            )

        if not self.omr.empty and "chartdate" in self.omr.columns:
            self.omr["chartdate"] = pd.to_datetime(self.omr["chartdate"], errors="coerce")

    # ─── Episode index ────────────────────────────────────────────────────────

    def _build_episode_index(self) -> None:
        pts = self.patients[
            ["subject_id", "anchor_age", "anchor_year", "gender", "dod"]
        ].copy()
        merged = self.admissions.merge(pts, on="subject_id", how="inner")
        merged["age"] = (
            merged["anchor_age"]
            + (merged["admittime"].dt.year - merged["anchor_year"])
        ).clip(18, 91).fillna(65).astype(int)
        merged = merged[
            merged["discharge_location"].notna()
            & (merged["discharge_location"].str.strip() != "")
        ]
        self._episode_base = merged.set_index("hadm_id")
        self.hadm_ids: List[int] = list(self._episode_base.index)

    # ─── Complexity index (built once on init) ────────────────────────────────

    def _build_complexity_index(self) -> None:
        self._complexity_index: Dict[str, List[int]] = {
            "easy": [], "medium": [], "hard": []
        }
        for hadm_id in self.hadm_ids:
            try:
                ep = self._minimal_episode_for_complexity(hadm_id)
                c  = self.classify_complexity(ep)
                self._complexity_index[c].append(hadm_id)
            except Exception:
                self._complexity_index["medium"].append(hadm_id)

    def _minimal_episode_for_complexity(self, hadm_id: int) -> Dict[str, Any]:
        ep = self._episode_base.loc[hadm_id].to_dict()
        ep["hadm_id"] = hadm_id

        dx = (
            self.diagnoses[self.diagnoses["hadm_id"] == hadm_id]
            .sort_values("seq_num")[["icd_code", "seq_num"]]
            .dropna(subset=["icd_code"])
        )
        ep["diagnoses"] = dx.to_dict("records")

        icu = self.icustays[self.icustays["hadm_id"] == hadm_id][
            ["stay_id", "los", "first_careunit", "last_careunit"]
        ]
        ep["icu_stays"] = icu.to_dict("records")

        ep["microbiology"]         = self._get_microbiology(hadm_id)
        ep["fluid_balance"]        = self._get_fluid_balance(hadm_id)
        ep["icu_procedure_summary"] = self._get_icu_procedures(hadm_id)
        return ep

    # ─── Public API ───────────────────────────────────────────────────────────

    def get_episode(
        self,
        hadm_id: Optional[int] = None,
        noise_level: str = "clean",
    ) -> Dict[str, Any]:
        if hadm_id is None:
            hadm_id = random.choice(self.hadm_ids)
        elif hadm_id not in self._episode_base.index:
            raise ValueError(f"hadm_id {hadm_id} not found in the demo dataset")

        ep: Dict[str, Any] = self._episode_base.loc[hadm_id].to_dict()
        ep["hadm_id"]  = hadm_id
        subject_id     = int(ep.get("subject_id", 0))
        admit_time     = ep.get("admittime")
        dischtime      = ep.get("dischtime")

        # ── Diagnoses ─────────────────────────────────────────────────────────
        dx = (
            self.diagnoses[self.diagnoses["hadm_id"] == hadm_id]
            .sort_values("seq_num")
            [["icd_code", "icd_version", "long_title", "seq_num"]]
            .dropna(subset=["icd_code"])
        )
        ep["diagnoses"] = dx.to_dict("records")

        # ── Procedures (ICD) ─────────────────────────────────────────────────
        px = (
            self.procedures[self.procedures["hadm_id"] == hadm_id]
            .sort_values("seq_num")[["icd_code", "long_title"]]
            .dropna(subset=["icd_code"])
        )
        ep["procedures"] = px.head(10).to_dict("records")

        # ── DRG codes ─────────────────────────────────────────────────────────
        drg = self.drgcodes[self.drgcodes["hadm_id"] == hadm_id][
            ["drg_code", "description", "drg_severity", "drg_mortality"]
        ].dropna(subset=["drg_code"])
        ep["drgcodes"] = drg.head(5).to_dict("records")

        # ── ICU stays ─────────────────────────────────────────────────────────
        icu = self.icustays[self.icustays["hadm_id"] == hadm_id][
            ["stay_id", "los", "first_careunit", "last_careunit"]
        ]
        ep["icu_stays"] = icu.to_dict("records")

        # ── Medications (prescriptions — top-10 unique drugs) ─────────────────
        rx_cols = ["drug", "route", "dose_val_rx"]
        if "formulary_drug_cd" in self.prescriptions.columns:
            rx_cols.append("formulary_drug_cd")
        rx = (
            self.prescriptions[self.prescriptions["hadm_id"] == hadm_id]
            [rx_cols]
            .dropna(subset=["drug"])
            .drop_duplicates("drug")
        )
        ep["medications"] = rx.head(10).to_dict("records")

        # ── Pharmacy active/stopped ───────────────────────────────────────────
        if not self.pharmacy.empty and "hadm_id" in self.pharmacy.columns:
            ph = self.pharmacy[self.pharmacy["hadm_id"] == hadm_id]
            if not ph.empty:
                active = (
                    ph[ph["status"].str.lower().str.contains(
                        "active|running|dispens", na=False
                    )]["medication"].dropna().drop_duplicates().head(8).tolist()
                )
                stopped = (
                    ph[ph["status"].str.lower().str.contains(
                        "stop|inactiv|discontinu", na=False
                    )]["medication"].dropna().drop_duplicates().head(8).tolist()
                )
                ep["pharmacy_active"]  = active
                ep["pharmacy_stopped"] = stopped
            else:
                ep["pharmacy_active"]  = []
                ep["pharmacy_stopped"] = []
        else:
            ep["pharmacy_active"]  = []
            ep["pharmacy_stopped"] = []

        # ── Abnormal labs ─────────────────────────────────────────────────────
        labs = self.labevents[
            (self.labevents["hadm_id"] == hadm_id)
            & self.labevents["flag"].notna()
            & (self.labevents["flag"].str.strip() != "")
        ][["label", "flag", "value"]].dropna(subset=["label"]).drop_duplicates("label")
        ep["lab_flags"] = labs.head(10).to_dict("records")

        # ── Clinical services ─────────────────────────────────────────────────
        svc = self.services[self.services["hadm_id"] == hadm_id]
        ep["services"] = svc["curr_service"].dropna().tolist()

        # ── Microbiology ──────────────────────────────────────────────────────
        ep["microbiology"] = self._get_microbiology(hadm_id)

        # ── OMR weight/BMI ────────────────────────────────────────────────────
        ep["weight_kg"], ep["bmi"] = self._get_omr(subject_id, admit_time)

        # ── Care trajectory ───────────────────────────────────────────────────
        ep["care_trajectory"] = self._get_care_trajectory(hadm_id)

        # ── ICU procedures ────────────────────────────────────────────────────
        ep["icu_procedure_summary"] = self._get_icu_procedures(hadm_id)

        # ── HCPCS ─────────────────────────────────────────────────────────────
        ep["hcpcs_categories"] = self._get_hcpcs(hadm_id)

        # ── v3: Vitals ────────────────────────────────────────────────────────
        ep["vitals"] = self._get_vitals(hadm_id)

        # ── v3: Fluid balance ─────────────────────────────────────────────────
        ep["fluid_balance"] = self._get_fluid_balance(hadm_id)

        # ── v3: EMAR summary + private drug set for graders ───────────────────
        emar_sum = self._get_emar_summary(hadm_id, dischtime)
        ep["emar_summary"] = emar_sum
        ep["_emar_drug_set"] = {
            row["medication"].strip().lower()
            for row in emar_sum
            if row.get("medication")
        }

        # ── v3: Discharge orders ──────────────────────────────────────────────
        ep["discharge_orders"] = self._get_discharge_orders(hadm_id)

        # ── Complexity ────────────────────────────────────────────────────────
        ep["complexity"] = self.classify_complexity(ep)

        # ── Stochastic masking ────────────────────────────────────────────────
        if noise_level != "clean":
            ep = self._apply_noise(ep, noise_level, hadm_id)

        ep["noise_level"] = noise_level
        return ep

    # ─── v3 data extractors ───────────────────────────────────────────────────

    def _get_vitals(self, hadm_id: int) -> Dict[str, Any]:
        if self._chartevents_cache is None or self._chartevents_cache.empty:
            return {}

        ce = self._chartevents_cache[
            (self._chartevents_cache["hadm_id"] == hadm_id)
            & (self._chartevents_cache["itemid"].isin(_VITAL_ITEMS.keys()))
            & self._chartevents_cache["valuenum"].notna()
        ]

        if ce.empty:
            return {}

        result: Dict[str, Any] = {}
        for itemid, name in _VITAL_ITEMS.items():
            rows = ce[ce["itemid"] == itemid].sort_values("charttime")
            if rows.empty:
                continue
            vals = rows["valuenum"].dropna().tolist()
            if not vals:
                continue

            admission_val = float(vals[0])
            discharge_val = float(vals[-1])
            min_val       = float(min(vals))
            max_val       = float(max(vals))

            critical = False
            if name == "Heart Rate"         and (max_val > 130 or min_val < 40):
                critical = True
            elif name == "Systolic BP"      and min_val < 90:
                critical = True
            elif name == "SpO2"             and min_val < 90:
                critical = True
            elif name == "Temperature F"    and max_val > 102.2:
                critical = True
            elif name == "Respiratory Rate" and max_val > 30:
                critical = True

            result[name] = {
                "admission_value": round(admission_val, 2),
                "discharge_value": round(discharge_val, 2),
                "min_value":       round(min_val, 2),
                "max_value":       round(max_val, 2),
                "critical_flag":   critical,
            }
        return result

    def _get_fluid_balance(self, hadm_id: int) -> Dict[str, Any]:
        total_input_ml  = 0.0
        total_urine_ml  = 0.0
        total_output_ml = 0.0

        if not self._inputevents.empty and "hadm_id" in self._inputevents.columns:
            inp = self._inputevents[self._inputevents["hadm_id"] == hadm_id]
            if not inp.empty and "amountuom" in inp.columns and "amount" in inp.columns:
                ml_rows = inp[inp["amountuom"].str.lower().str.strip() == "ml"]
                if not ml_rows.empty:
                    total_input_ml = float(ml_rows["amount"].fillna(0).sum())

        if not self._outputevents.empty and "hadm_id" in self._outputevents.columns:
            out = self._outputevents[self._outputevents["hadm_id"] == hadm_id]
            if not out.empty and "value" in out.columns:
                total_output_ml = float(out["value"].fillna(0).sum())
                if "itemid" in out.columns:
                    urine = out[out["itemid"] == 226559]
                    if not urine.empty:
                        total_urine_ml = float(urine["value"].fillna(0).sum())

        net_balance = total_input_ml - total_output_ml
        return {
            "total_input_ml":   round(total_input_ml, 1),
            "total_urine_ml":   round(total_urine_ml, 1),
            "total_output_ml":  round(total_output_ml, 1),
            "net_balance_ml":   round(net_balance, 1),
            "fluid_overloaded": net_balance > 3000,
            "oliguria":         0 < total_urine_ml < 500,
        }

    def _get_emar_summary(
        self, hadm_id: int, dischtime: Any
    ) -> List[Dict[str, Any]]:
        if self._emar.empty:
            return []

        em = self._emar[
            (self._emar["hadm_id"] == hadm_id)
        ].dropna(subset=["medication"])
        if em.empty:
            return []

        try:
            disch_dt = pd.to_datetime(dischtime)
        except Exception:
            disch_dt = None

        results: List[Dict[str, Any]] = []
        for med_name, grp in em.groupby("medication"):
            times = grp["charttime"].dropna().sort_values()
            if times.empty:
                continue
            first_given = str(times.iloc[0])
            last_given  = str(times.iloc[-1])
            total_doses = len(grp)
            active = False
            if disch_dt is not None:
                try:
                    last_dt = pd.to_datetime(last_given)
                    delta   = (disch_dt - last_dt).total_seconds()
                    active  = 0 <= delta <= 86_400
                except Exception:
                    pass
            results.append({
                "medication":          str(med_name).strip(),
                "first_given":         first_given,
                "last_given":          last_given,
                "total_doses":         total_doses,
                "active_at_discharge": active,
            })

        results.sort(key=lambda x: x["last_given"], reverse=True)
        return results[:15]

    def _get_discharge_orders(self, hadm_id: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "discharge_planning_finalized": False,
            "documented_discharge_orders":  [],
        }
        if self._poe_detail.empty:
            return result

        # Join through poe to get hadm_id → poe_id mapping
        pod: pd.DataFrame
        if (
            not self._poe.empty
            and "hadm_id" in self._poe.columns
            and "poe_id" in self._poe.columns
            and "poe_id" in self._poe_detail.columns
        ):
            poe_ids = set(
                self._poe[self._poe["hadm_id"] == hadm_id]["poe_id"].tolist()
            )
            if not poe_ids:
                return result
            pod = self._poe_detail[self._poe_detail["poe_id"].isin(poe_ids)]
        elif "hadm_id" in self._poe_detail.columns:
            pod = self._poe_detail[self._poe_detail["hadm_id"] == hadm_id]
        else:
            return result

        if pod.empty:
            return result

        orders: List[str] = []
        for _, row in pod.iterrows():
            fname = str(row.get("field_name",  "") or "").strip()
            fval  = str(row.get("field_value", "") or "").strip()

            if "discharge planning" in fname.lower() and fval.lower() == "finalized":
                result["discharge_planning_finalized"] = True

            if fval and any(
                kw in fname.lower() for kw in ("discharge", "transfer", "follow")
            ):
                orders.append(fval)

        result["documented_discharge_orders"] = orders
        return result

    # ─── Existing extractors (unchanged) ─────────────────────────────────────

    def _get_microbiology(self, hadm_id: int) -> List[Dict[str, Any]]:
        if self.microbiology.empty:
            return []
        mb = self.microbiology[self.microbiology["hadm_id"] == hadm_id]
        positive = mb[mb["org_name"].notna() & (mb["org_name"].str.strip() != "")]
        if positive.empty:
            return []
        results = []
        for (spec, org), grp in positive.groupby(
            ["spec_type_desc", "org_name"], dropna=False
        ):
            abx_rows = grp[grp["ab_name"].notna() & grp["interpretation"].notna()]
            sensitivities: Dict[str, str] = {}
            for _, row in abx_rows.iterrows():
                ab     = str(row["ab_name"]).strip()
                interp = str(row["interpretation"]).strip()
                if ab and interp:
                    sensitivities[ab] = interp
            results.append({
                "specimen":      str(spec).strip() if pd.notna(spec) else "UNKNOWN",
                "organism":      str(org).strip(),
                "sensitivities": sensitivities,
                "resistant_to":  [ab for ab, i in sensitivities.items() if i == "R"],
                "sensitive_to":  [ab for ab, i in sensitivities.items() if i == "S"],
            })
        return results[:5]

    def _get_omr(
        self, subject_id: int, admit_time: Any
    ) -> tuple[Optional[float], Optional[float]]:
        if self.omr.empty:
            return None, None
        pt_omr = self.omr[self.omr["subject_id"] == subject_id].copy()
        if pt_omr.empty:
            return None, None
        if admit_time is not None and "chartdate" in pt_omr.columns:
            try:
                admit_dt = pd.to_datetime(admit_time)
                past = pt_omr[pt_omr["chartdate"] <= admit_dt]
                if past.empty:
                    past = pt_omr
            except Exception:
                past = pt_omr
        else:
            past = pt_omr

        weight_kg: Optional[float] = None
        bmi:       Optional[float] = None
        for _, row in past.sort_values("chartdate", ascending=False).iterrows():
            name  = str(row.get("result_name", "")).lower()
            value = row.get("result_value")
            if value is None or pd.isna(value):
                continue
            try:
                fval = float(str(value).replace(",", ""))
            except ValueError:
                continue
            if "weight" in name and "kg" in name and weight_kg is None:
                weight_kg = round(fval, 1)
            elif "weight" in name and "lbs" in name and weight_kg is None:
                weight_kg = round(fval * 0.453592, 1)
            elif "bmi" in name and bmi is None:
                bmi = round(fval, 1)
            if weight_kg is not None and bmi is not None:
                break
        return weight_kg, bmi

    def _get_care_trajectory(self, hadm_id: int) -> List[str]:
        if self.transfers.empty:
            return []
        tr = self.transfers[
            (self.transfers["hadm_id"] == hadm_id)
            & self.transfers["careunit"].notna()
        ].copy()
        if tr.empty:
            return []
        if "intime" in tr.columns:
            tr["intime"] = pd.to_datetime(tr["intime"], errors="coerce")
            tr = tr.sort_values("intime")
        units = tr["careunit"].dropna().tolist()
        deduped = [units[0]] if units else []
        for u in units[1:]:
            if u != deduped[-1]:
                deduped.append(u)
        return deduped[:8]

    def _get_icu_procedures(self, hadm_id: int) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "ventilation_hours": 0.0,
            "has_arterial_line": False,
            "has_central_line":  False,
            "has_dialysis":      False,
            "procedure_names":   [],
        }
        if self.icu_procedures.empty:
            return summary
        ip = self.icu_procedures[self.icu_procedures["hadm_id"] == hadm_id]
        if ip.empty:
            return summary
        for _, row in ip.iterrows():
            item_id  = int(row.get("itemid",  0) or 0)
            value    = float(row.get("value",  0) or 0)
            valueuom = str(row.get("valueuom", "")).lower()
            label    = str(row.get("ordercategoryname", "")).strip()
            if item_id in _VENTILATION_ITEMS:
                if "min" in valueuom:
                    summary["ventilation_hours"] += value / 60.0
                elif "hour" in valueuom:
                    summary["ventilation_hours"] += value
            if item_id in _INVASIVE_LINE_ITEMS:
                if item_id == 224263:
                    summary["has_arterial_line"] = True
                else:
                    summary["has_central_line"] = True
            if item_id in _DIALYSIS_ITEMS:
                summary["has_dialysis"] = True
            if label and label not in summary["procedure_names"]:
                summary["procedure_names"].append(label)
        summary["ventilation_hours"] = round(summary["ventilation_hours"], 1)
        return summary

    def _get_hcpcs(self, hadm_id: int) -> List[str]:
        if self.hcpcsevents.empty:
            return []
        hc = self.hcpcsevents[self.hcpcsevents["hadm_id"] == hadm_id]
        if hc.empty:
            return []
        return hc["short_description"].dropna().str.strip().unique().tolist()[:8]

    # ─── Complexity ───────────────────────────────────────────────────────────

    @staticmethod
    def classify_complexity(ep: Dict[str, Any]) -> str:
        loc      = str(ep.get("discharge_location", "") or "").upper().strip()
        los      = float(ep.get("hospital_los_days", 0) or 0)
        icu_stay = ep.get("icu_stays", [])
        diagnoses= ep.get("diagnoses", [])
        micro    = ep.get("microbiology", [])
        fluid    = ep.get("fluid_balance") or {}
        icu_proc = ep.get("icu_procedure_summary") or {}

        # ── EASY ──────────────────────────────────────────────────────────────
        # Plain home discharge (no professional services at home), short-ish
        # stay, no ICU, moderate diagnostic burden.
        # Thresholds tuned so the easy pool has ≥ 25 patients in the demo
        # dataset (needed for tier-based curriculum in train_grpo.py).
        is_plain_home = any(frag in loc for frag in _PLAIN_HOME_FRAGMENTS)
        has_home_services = any(frag in loc for frag in _HOME_SERVICE_FRAGMENTS)
        if (
            is_plain_home
            and not has_home_services
            and los <= 10
            and len(icu_stay) == 0
            and len(diagnoses) <= 12
        ):
            return "easy"

        # ── HARD ──────────────────────────────────────────────────────────────
        hard_loc       = any(h in loc for h in _HARD_DISCHARGE_LOCATIONS)
        hard_vent      = (
            len(icu_stay) > 0
            and float(icu_proc.get("ventilation_hours", 0) or 0) > 24
        )
        hard_resistant = any(len(m.get("resistant_to", [])) > 0 for m in micro)
        oliguria       = bool(fluid.get("oliguria", False))

        if hard_loc or los > 14 or hard_vent or hard_resistant or oliguria:
            return "hard"

        return "medium"

    def sample_by_complexity(
        self, complexity: str, n: Optional[int] = None
    ) -> List[int]:
        pool = list(self._complexity_index.get(complexity, []))
        if n is None:
            return pool
        return random.sample(pool, min(n, len(pool)))

    # ─── Stochastic noise ────────────────────────────────────────────────────

    def _apply_noise(
        self, ep: Dict[str, Any], noise_level: str, hadm_id: int
    ) -> Dict[str, Any]:
        # Seeded per hadm_id+noise_level for reproducibility within an episode
        rng = random.Random(f"{hadm_id}:{noise_level}")
        ep  = dict(ep)

        # ── partial drops (applied for both "partial" and "noisy") ────────────
        lab_flags = list(ep.get("lab_flags", []))
        if lab_flags:
            keep_n = max(0, len(lab_flags) - int(len(lab_flags) * 0.3))
            lab_flags = rng.sample(lab_flags, keep_n)
        ep["lab_flags"] = lab_flags

        if rng.random() < 0.4:
            ep["weight_kg"] = None
        if rng.random() < 0.4:
            ep["bmi"] = None

        meds = list(ep.get("medications", []))
        if meds:
            n_keep = rng.randint(min(5, len(meds)), min(10, len(meds)))
            meds = meds[:n_keep]
        ep["medications"] = meds

        care = list(ep.get("care_trajectory", []))
        if len(care) > 3:
            drop_idx = rng.randint(0, len(care) - 1)
            care = [u for i, u in enumerate(care) if i != drop_idx]
        ep["care_trajectory"] = care

        if noise_level == "noisy":
            diags = list(ep.get("diagnoses", []))
            if diags and rng.random() < 0.3:
                rng.shuffle(diags)
            ep["diagnoses"] = diags

            if rng.random() < 0.25:
                ep["microbiology"] = []

            vitals = dict(ep.get("vitals") or {})
            vital_keys = list(vitals.keys())
            if vital_keys and rng.random() < 0.5:
                for k in rng.sample(vital_keys, min(2, len(vital_keys))):
                    vitals.pop(k, None)
            ep["vitals"] = vitals

            if rng.random() < 0.2:
                ep["fluid_balance"] = {}

            meds_noisy = []
            for m in ep.get("medications", []):
                m = dict(m)
                if rng.random() < 0.15 and m.get("formulary_drug_cd"):
                    m["drug"] = m["formulary_drug_cd"]
                meds_noisy.append(m)
            ep["medications"] = meds_noisy

        return ep

    def sample_hadm_ids(self, n: Optional[int] = None) -> List[int]:
        if n is None:
            return self.hadm_ids
        return random.sample(self.hadm_ids, min(n, len(self.hadm_ids)))
