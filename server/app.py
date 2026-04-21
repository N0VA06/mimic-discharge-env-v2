from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from environment import MIMICDischargeEnv, ResetRequest
from environment.models import Action

# ─── Logging — structured JSON-style ──────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("req_id", "hadm_id", "task_id", "session_id"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger("mimic.server")


# ─── Metrics store ────────────────────────────────────────────────────────────

class _Metrics:
    def __init__(self) -> None:
        self.start_time          = time.time()
        self.request_count:      Dict[str, int]         = defaultdict(int)
        self.error_count:        Dict[str, int]         = defaultdict(int)
        self.latency_ms:         Dict[str, List[float]] = defaultdict(list)
        self.episode_count:      int                    = 0
        self.total_reward:       float                  = 0.0
        self.rewards_by_task:    Dict[int, List[float]] = defaultdict(list)
        self._latency_window     = 500
        # Extended v3.1.0 metrics
        self.json_parse_attempts: int               = 0
        self.json_parse_success:  int               = 0
        self.revision_count:      int               = 0
        self.complexity_counts:   Dict[str, int]    = defaultdict(int)

    def record_request(self, route: str, latency: float, status_code: int) -> None:
        self.request_count[route] += 1
        buf = self.latency_ms[route]
        buf.append(latency)
        if len(buf) > self._latency_window:
            buf.pop(0)
        if status_code >= 400:
            self.error_count[route] += 1

    def record_episode(self, task_id: int, reward: float, complexity: str = "medium") -> None:
        self.episode_count              += 1
        self.total_reward               += reward
        self.rewards_by_task[task_id].append(reward)
        self.complexity_counts[complexity] += 1

    def snapshot(self) -> Dict[str, Any]:
        uptime = round(time.time() - self.start_time, 1)
        routes: Dict[str, Any] = {}
        for route, lats in self.latency_ms.items():
            if lats:
                sorted_l = sorted(lats)
                n        = len(sorted_l)
                routes[route] = {
                    "requests":  self.request_count[route],
                    "errors":    self.error_count[route],
                    "p50_ms":    round(sorted_l[n // 2], 1),
                    "p95_ms":    round(sorted_l[int(n * 0.95)], 1),
                    "p99_ms":    round(sorted_l[int(n * 0.99)], 1),
                    "mean_ms":   round(sum(sorted_l) / n, 1),
                }
        task_stats: Dict[str, Any] = {}
        for tid, rewards in self.rewards_by_task.items():
            if rewards:
                task_stats[f"task_{tid}"] = {
                    "episodes": len(rewards),
                    "avg_reward": round(sum(rewards) / len(rewards), 4),
                    "max_reward": round(max(rewards), 4),
                    "min_reward": round(min(rewards), 4),
                }
        parse_rate = (
            round(self.json_parse_success / self.json_parse_attempts, 4)
            if self.json_parse_attempts > 0 else None
        )
        return {
            "uptime_seconds":         uptime,
            "episode_count":          self.episode_count,
            "total_reward":           round(self.total_reward, 4),
            "avg_reward":             round(
                self.total_reward / self.episode_count, 4
            ) if self.episode_count else 0.0,
            "routes":                 routes,
            "tasks":                  task_stats,
            "json_parse_success_rate": parse_rate,
            "revision_count":         self.revision_count,
            "complexity_distribution": dict(self.complexity_counts),
        }


_metrics = _Metrics()


# ─── Episode history ──────────────────────────────────────────────────────────

class _EpisodeHistory:
    """Ring buffer of the last MAX_HISTORY completed episodes."""
    MAX_HISTORY = 50

    def __init__(self) -> None:
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=self.MAX_HISTORY)

    def push(self, record: Dict[str, Any]) -> None:
        self._buf.appendleft(record)

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return list(self._buf)[:max(1, min(n, self.MAX_HISTORY))]


_history = _EpisodeHistory()


# ─── Environment singleton ────────────────────────────────────────────────────
# Initialised lazily inside lifespan so startup errors surface cleanly.

_env: Optional[MIMICDischargeEnv] = None
_ready: bool = False


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _env, _ready
    logger.info("Starting MIMIC Discharge Planning server …")
    data_root = os.environ.get("MIMIC_DATA_PATH", None)
    try:
        _env   = MIMICDischargeEnv(data_root=data_root)
        _ready = True
        logger.info(
            "Environment ready",
            extra={"episodes": len(_env.builder.hadm_ids)},
        )
    except Exception as exc:
        logger.error("Environment failed to initialise: %s", exc)
        # Server starts but /health will report not-ready
    yield
    logger.info("Server shutting down")
    _ready = False


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MIMIC Discharge Planning — OpenEnv",
    description=(
        "Production RL environment for clinical discharge planning built on MIMIC-IV EHR data.\n\n"
        "**Tasks**\n"
        "- Task 1 (Easy) — Discharge disposition prediction (8 categories)\n"
        "- Task 2 (Medium) — Care plan recommendation (specialties + meds + instructions)\n"
        "- Task 3 (Hard) — Full discharge note generation (≥300 words)\n"
        "- Task 4 (Very Hard) — 10-step admission-to-discharge workflow (sparse reward)\n\n"
        "**Episode flow**: `POST /reset` → `POST /step` (repeat up to max_steps) → done\n\n"
        "All graders are deterministic and run server-side with no LLM judge."
    ),
    version="3.1.0",
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Episode", "description": "Core RL loop: reset and step"},
        {"name": "Observability", "description": "Health, metrics, history, state"},
        {"name": "Meta", "description": "Task descriptions and schema info"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware: request ID + timing ──────────────────────────────────────────

@app.middleware("http")
async def _request_middleware(request: Request, call_next):
    req_id  = str(uuid.uuid4())[:8]
    start   = time.perf_counter()
    request.state.req_id = req_id

    response: Response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    route      = request.url.path
    response.headers["X-Request-ID"]    = req_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

    _metrics.record_request(route, elapsed_ms, response.status_code)
    logger.info(
        "%s %s → %d  (%.1fms)",
        request.method, route, response.status_code, elapsed_ms,
        extra={"req_id": req_id},
    )
    return response


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_env() -> MIMICDischargeEnv:
    if _env is None or not _ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Environment is initialising. Retry in a few seconds.",
            },
        )
    return _env


def _require_active(env: MIMICDischargeEnv) -> None:
    if not env.active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "episode_not_started",
                "message": "No active episode. Call POST /reset first.",
                "hint":    "POST /reset with body {\"task_id\": 1}",
            },
        )


# ─── Core RL routes ───────────────────────────────────────────────────────────

@app.post(
    "/reset",
    tags=["Episode"],
    summary="Start a new episode",
    response_description="Initial observation for the sampled patient",
    responses={
        200: {"description": "Episode started, initial observation returned"},
        400: {"description": "Invalid task_id"},
        503: {"description": "Environment not ready"},
    },
)
async def reset(request: Request, body: ResetRequest = None):
    """
    Start a new episode.

    - **task_id**: `1` = discharge disposition (easy), `2` = care plan (medium), `3` = discharge note (hard)
    - **hadm_id**: optional — pin a specific hospitalisation for reproducibility
    """
    env   = _require_env()
    req   = body or ResetRequest()
    req_id = getattr(request.state, "req_id", "—")

    if req.task_id not in (1, 2, 3, 4):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "invalid_task_id",
                "message": f"task_id must be 1, 2, 3, or 4. Got: {req.task_id}",
            },
        )

    obs = env.reset(
        task_id=req.task_id,
        hadm_id=req.hadm_id,
        noise_level=getattr(req, "noise_level", "clean"),
        curriculum_mode=getattr(req, "curriculum_mode", "random"),
    )
    logger.info(
        "Episode reset",
        extra={
            "req_id":   req_id,
            "task_id":  req.task_id,
            "hadm_id":  obs.hadm_id,
            "subject":  obs.subject_id,
        },
    )
    return obs.model_dump()


@app.post(
    "/step",
    tags=["Episode"],
    summary="Submit an action",
    response_description="Step result: observation, reward, done, partial signals",
    responses={
        200: {"description": "Action processed, step result returned"},
        409: {"description": "No active episode — call /reset first"},
        422: {"description": "Action schema validation failed"},
        503: {"description": "Environment not ready"},
    },
)
async def step(request: Request, action: Action):
    """
    Submit one agent action and receive the environment's response.

    The action envelope must include `task_id` and the matching sub-object
    (`task1`, `task2`, or `task3`). The grader runs synchronously and returns
    partial reward signals alongside the scalar reward.
    """
    env    = _require_env()
    req_id = getattr(request.state, "req_id", "—")
    _require_active(env)

    t0     = time.perf_counter()
    result = env.step(action)
    grade_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Track JSON parse quality
    _metrics.json_parse_attempts += 1
    _metrics.json_parse_success  += 1  # action already passed Pydantic validation

    # Track Task 4 revisions
    if action.task4 and action.task4.revise_step is not None:
        _metrics.revision_count += 1

    if result.done:
        complexity = (env._episode.get("complexity", "medium") if env._episode else "medium")
        _metrics.record_episode(action.task_id, result.reward, complexity)
        _history.push({
            "task_id":         action.task_id,
            "hadm_id":         env._episode.get("hadm_id") if env._episode else None,
            "reward":          result.reward,
            "steps":           env._step_num,
            "complexity":      complexity,
            "partial_signals": result.partial_signals,
            "completed_at":    int(time.time()),
        })

    logger.info(
        "Step completed",
        extra={
            "req_id":    req_id,
            "task_id":   action.task_id,
            "step_num":  env._step_num,
            "reward":    result.reward,
            "done":      result.done,
            "grade_ms":  grade_ms,
        },
    )
    return result.model_dump()


# ─── Observability routes ─────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["Observability"],
    summary="Liveness and readiness probe",
)
async def health():
    """
    Returns server status, environment readiness, and basic episode stats.
    Use this as the liveness probe in your Kubernetes/HF Space deployment.
    """
    if not _ready or _env is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status":  "initialising",
                "ready":   False,
                "message": "Environment is loading MIMIC tables. Retry in a few seconds.",
            },
        )

    uptime = round(time.time() - _metrics.start_time, 1)
    return {
        "status":             "ok",
        "ready":              True,
        "uptime_seconds":     uptime,
        "episodes_available": len(_env.builder.hadm_ids),
        "env_active":         _env.active,
        "total_episodes_run": _env._total_episodes,
        "current_task":       _env._task_id if _env.active else None,
        "current_hadm_id":    (
            int(_env._episode["hadm_id"]) if _env.active and _env._episode else None
        ),
    }


@app.get(
    "/metrics",
    tags=["Observability"],
    summary="Request counts, latency percentiles, and reward statistics",
)
async def metrics():
    """
    Returns Prometheus-style aggregate metrics including:
    - Per-route request counts and latency percentiles (p50 / p95 / p99)
    - Per-task episode counts and reward statistics
    - Server uptime and cumulative reward
    """
    return _metrics.snapshot()


@app.get(
    "/history",
    tags=["Observability"],
    summary="Recent completed episodes",
)
async def history(n: int = 10):
    """
    Returns the last `n` completed episodes (max 50) with reward breakdowns.
    Useful for monitoring agent performance over time without re-running graders.
    """
    if n < 1 or n > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_n", "message": "n must be between 1 and 50"},
        )
    records = _history.recent(n)
    return {
        "count":    len(records),
        "episodes": records,
    }


@app.get(
    "/state",
    tags=["Observability"],
    summary="Current internal environment state (debug)",
)
async def state():
    """
    Exposes ground-truth values for the current episode.
    Useful for grader debugging — not intended for agent use.
    """
    env = _require_env()
    s   = env.state()
    return s.model_dump()


# ─── Meta routes ──────────────────────────────────────────────────────────────

@app.get(
    "/tasks",
    tags=["Meta"],
    summary="Task catalogue with schemas and scoring weights",
)
async def tasks():
    """
    Returns all tasks with their action schemas, scoring formulas,
    and difficulty levels. Useful for agents to self-configure per task.
    """
    return {
        "tasks": [
            {
                "id":           1,
                "name":         "discharge_disposition",
                "difficulty":   "easy",
                "max_steps":    1,
                "description":  "Predict the appropriate discharge disposition (8 categories).",
                "action_schema": {
                    "task_id": 1,
                    "task1": {
                        "disposition": "home|home_with_services|snf|rehab|hospice|ama|expired|other",
                        "reasoning":   "string (optional, up to 30 words)",
                    },
                },
                "scoring": {
                    "exact_match":         1.00,
                    "broad_group_match":   0.50,
                    "adjacent_group":      0.25,
                    "reasoning_bonus":     0.05,
                },
            },
            {
                "id":           2,
                "name":         "care_plan_recommendation",
                "difficulty":   "medium",
                "max_steps":    2,
                "description":  "Recommend follow-up specialties, medications, and patient instructions.",
                "action_schema": {
                    "task_id": 2,
                    "task2": {
                        "follow_up_specialties":      ["string"],
                        "medications_to_continue":    ["string"],
                        "medications_to_discontinue": ["string"],
                        "key_instructions":           ["string"],
                        "reasoning":                  "string (optional)",
                    },
                },
                "scoring": {
                    "specialty_f1_weight":           0.35,
                    "medication_f1_weight":          0.25,
                    "instruction_quality_weight":    0.25,
                    "discontinue_accuracy_weight":   0.15,
                    "hallucination_penalty_max":     0.10,
                },
            },
            {
                "id":           3,
                "name":         "discharge_note_generation",
                "difficulty":   "hard",
                "max_steps":    1,
                "description":  "Draft a complete clinical discharge summary (≥300 words).",
                "action_schema": {
                    "task_id": 3,
                    "task3": {"discharge_note": "string (≥300 words of clinical prose)"},
                },
                "scoring": {
                    "diagnosis_coverage_weight":    0.30,
                    "disposition_accuracy_weight":  0.20,
                    "medication_f1_weight":         0.20,
                    "los_accuracy_weight":          0.15,
                    "structure_score_weight":       0.10,
                    "information_density_weight":   0.05,
                    "hallucination_penalty_max":    0.15,
                    "followup_structure_penalty":   0.05,
                },
            },
            {
                "id":          4,
                "name":        "admission_to_discharge_workflow",
                "difficulty":  "very_hard",
                "max_steps":   10,
                "description": (
                    "10-step admission-to-discharge workflow. Sparse reward: "
                    "only Step 10 returns a non-zero reward for training."
                ),
                "action_schema": {
                    "task_id": 4,
                    "task4": {
                        "step_1":  {"triage_level": "icu|stepdown|floor"},
                        "step_2":  {"priority_labs": ["string"], "priority_consults": ["string"]},
                        "step_3":  {"interventions": ["string"]},
                        "step_4":  {"high_risk_medications": ["string"]},
                        "step_5":  {"antibiotic_strategy": "none|targeted|broad|empiric", "antibiotics": ["string"]},
                        "step_6":  {"fluid_strategy": "restrict_diuresis|aggressive_resuscitation|maintain"},
                        "step_7":  {"ready_for_stepdown": "bool", "barriers": ["string"]},
                        "step_8":  {"predicted_disposition": "home|snf|...", "los_remaining_days": "float"},
                        "step_9":  {"medications_to_continue": ["string"]},
                        "step_10": {"final_note": "string (≥300 words)"},
                        "revision": {"revise_step": "int 1-9", "revision": "object"},
                    },
                },
                "scoring": {
                    "step1_triage_max":      0.10,
                    "step2_labs_consult_max": 0.15,
                    "step3_intervention_max": 0.15,
                    "step4_hr_meds_max":     0.10,
                    "step5_antibiotics_max": 0.10,
                    "step6_fluid_max":       0.08,
                    "step7_readiness_max":   0.08,
                    "step8_dispo_los_max":   0.10,
                    "step9_med_recon_max":   0.10,
                    "step10_note_weight":    0.60,
                    "step10_shaping_weight": 0.40,
                    "consistency_bonus":     0.10,
                    "trajectory_bonus":      0.05,
                    "revision_cost_each":    0.02,
                },
            },
        ]
    }


@app.get(
    "/episodes",
    tags=["Meta"],
    summary="Available episode count and sample hadm_ids",
)
async def episodes(sample: int = 5):
    """
    Returns the total number of available hospitalisations and a random sample
    of hadm_ids that can be pinned via `POST /reset` for reproducible evaluation.
    """
    import random
    env = _require_env()
    ids = env.builder.hadm_ids
    return {
        "total":      len(ids),
        "sample_ids": random.sample(ids, min(sample, len(ids))),
        "hint":       "Pass hadm_id in POST /reset body to pin a specific episode.",
    }


@app.get(
    "/complexity/{hadm_id}",
    tags=["Meta"],
    summary="Complexity tier for a specific hospitalisation",
)
async def complexity_for_hadm(hadm_id: int):
    """
    Returns the complexity tier (easy / medium / hard) for a specific hadm_id.
    Useful for curriculum planning and reproducible evaluation splits.
    """
    env = _require_env()
    if hadm_id not in set(env.builder.hadm_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "hadm_not_found", "hadm_id": hadm_id},
        )
    try:
        ep         = env.builder.get_episode(hadm_id, noise_level="clean")
        complexity = ep.get("complexity", "medium")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(exc)},
        )
    return {"hadm_id": hadm_id, "complexity": complexity}


@app.get(
    "/episodes/by_complexity",
    tags=["Meta"],
    summary="Available hadm_ids grouped by complexity tier",
)
async def episodes_by_complexity():
    """
    Returns all hadm_ids grouped into easy / medium / hard tiers.
    Uses the cached complexity index built at server startup.
    """
    env = _require_env()
    grouped: Dict[str, List[int]] = {"easy": [], "medium": [], "hard": []}
    for tier in ("easy", "medium", "hard"):
        grouped[tier] = env.builder.sample_by_complexity(tier) or []
    totals = {k: len(v) for k, v in grouped.items()}
    return {
        "totals":  totals,
        "grouped": grouped,
    }


@app.post(
    "/rollout",
    tags=["Episode"],
    summary="Run a full episode trajectory with a list of actions",
)
async def rollout(request: Request, body: dict):
    """
    Execute a complete episode by replaying a provided list of actions.

    Request body:
        task_id:     int
        actions:     list[Action]  — one action per step
        hadm_id:     int (optional) — pin episode
        noise_level: str (optional, default "clean")

    Returns the full step-by-step trajectory with rewards and partial signals.
    """
    env    = _require_env()
    req_id = getattr(request.state, "req_id", "—")

    task_id     = int(body.get("task_id", 1))
    actions_raw = body.get("actions", [])
    hadm_id     = body.get("hadm_id")
    noise_level = body.get("noise_level", "clean")

    if task_id not in (1, 2, 3, 4):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_task_id"},
        )

    from environment.models import Action as _Action

    obs = env.reset(
        task_id=task_id,
        hadm_id=hadm_id,
        noise_level=noise_level,
    )

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0

    for i, raw_action in enumerate(actions_raw):
        if not env.active:
            break
        try:
            action = _Action.model_validate(raw_action)
        except Exception as exc:
            trajectory.append({"step": i, "error": str(exc)})
            continue

        result = env.step(action)
        total_reward += result.reward
        trajectory.append({
            "step":            i + 1,
            "reward":          result.reward,
            "done":            result.done,
            "partial_signals": result.partial_signals,
            "info":            result.info,
        })
        if result.done:
            break

    if env.active:
        env.active = False  # forcibly close if fewer actions than max_steps

    logger.info(
        "Rollout completed",
        extra={"req_id": req_id, "task_id": task_id, "steps": len(trajectory)},
    )
    return {
        "task_id":      task_id,
        "hadm_id":      obs.hadm_id,
        "total_reward": round(total_reward, 4),
        "n_steps":      len(trajectory),
        "trajectory":   trajectory,
    }


# ─── Exception handlers ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "req_id", "—")
    logger.error(
        "Unhandled exception: %s", exc,
        extra={"req_id": req_id},
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error":      "internal_server_error",
            "message":    "An unexpected error occurred. Check server logs.",
            "request_id": req_id,
        },
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Start the uvicorn server. Called by openenv runner and __main__ guard."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        log_level="warning",   # suppress uvicorn access logs (we have middleware)
        access_log=False,
    )


if __name__ == "__main__":
    main()