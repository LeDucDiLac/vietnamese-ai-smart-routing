"""FastAPI routing endpoint — the silent Gateway shim (plan §5).

``POST /route`` takes a prompt + the caller's permission groups, runs the
classifier, then the matcher, and returns the chosen model plus the full
analysis and a human-readable reason. Designed as a drop-in: the Gateway calls
this and forwards to whatever ``model_id`` comes back, no client changes.

Layering / dependencies
------------------------
The router half (``router.*``, ``config``) is pure-Python and always importable.
The classifier half needs either the ``ml`` extra (torch) or ``onnxruntime``.
To keep this module importable in a CPU-only / no-ML environment (and for tests),
the classifier is loaded **lazily** from environment variables at startup, and a
deterministic stub is used when no model artifact is configured.

Environment variables
----------------------
- ``VI_ROUTER_ONNX``       path to an INT8 ONNX export (preferred serving path)
- ``VI_ROUTER_BACKBONE``   backbone id for the ONNX tokenizer (default mDeBERTa)
- ``VI_ROUTER_MAX_TOKENS`` tokenizer cap (default 256)
- ``VI_ROUTER_TORCH``      path to a torch checkpoint dir (quality path)
- ``VI_ROUTER_MODEL_SIZE`` config key for the torch checkpoint
- ``VI_ROUTER_LATENCY_CEIL`` optional latency ceiling (ms/1k) for matching

Run:

    uv run --extra serve uvicorn serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from typing import Any

from pydantic import BaseModel, Field

from config import load_complexity, load_label_schema
from router.capability import load_capabilities
from router.match import select_model

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt text to route.")
    user_groups: list[str] | None = Field(
        default=None,
        description="Permission groups the caller holds (None = default user).",
    )
    latency_ceiling_ms_per_1k: float | None = Field(
        default=None,
        description="Optional latency ceiling (ms per 1k tokens) for matching.",
    )


class RouteResponse(BaseModel):
    model_id: str
    task_type: str
    task_type_secondary: str
    task_type_prob: float
    prompt_complexity_score: float
    complexity: dict[str, float]
    required_skill: float
    chosen_skill: float
    est_cost_per_1k: float
    est_latency_ms_per_1k: float
    cleared_bar: bool
    reason: str
    alternatives: list[str]
    classifier_backend: str
    routing_overhead_ms: float


# ---------------------------------------------------------------------------
# Stub classifier — keeps the service importable & testable without ML deps
# ---------------------------------------------------------------------------


class StubClassifier:
    """Deterministic, dependency-free classifier for tests / no-ML deploys.

    Uses cheap text heuristics to produce a plausible NVIDIA-style record. It is
    NOT accurate — it exists so the routing path (matcher, endpoint, eval glue)
    can be exercised end-to-end before a trained ONNX artifact is available.
    """

    backend = "stub"

    def __init__(self) -> None:
        self.schema = load_label_schema()
        self.complexity = load_complexity()

    def predict(self, prompts: list[str]) -> list[dict[str, Any]]:
        records = []
        for text in prompts:
            lower = text.lower()
            # crude task guess
            if any(k in lower for k in ("code", "function", "def ", "python", "lập trình")):
                task = "Code Generation"
            elif any(k in lower for k in ("tóm tắt", "summar")):
                task = "Summarization"
            elif "?" in text or lower.startswith(("what", "why", "how", "vì sao", "tại sao")):
                task = "Open QA"
            else:
                task = "Text Generation"
            # complexity scales with length + question marks (toy heuristic)
            length_norm = min(1.0, len(text) / 800.0)
            dims = {
                "creativity_scope": round(0.3 + 0.4 * length_norm, 3),
                "reasoning": round(0.2 + 0.5 * length_norm, 3),
                "contextual_knowledge": round(0.2 * length_norm, 3),
                "domain_knowledge": round(0.3 * length_norm, 3),
                "constraint_ct": round(min(1.0, text.count("\n") / 10.0), 3),
                "number_of_few_shots": round(min(1.0, text.count("Ví dụ") / 5.0), 3),
            }
            secondary = "Other" if task != "Other" else "Open QA"
            records.append(
                {
                    "task_type_1": task,
                    "task_type_2": secondary,
                    "task_type_prob": 0.75,
                    **dims,
                    "prompt_complexity_score": self.complexity.score(dims),
                }
            )
        return records


def _build_classifier():
    """Pick a classifier backend from env, falling back to the stub.

    Order: ONNX (serving path) -> torch checkpoint (quality path) -> stub.
    """
    onnx_path = os.environ.get("VI_ROUTER_ONNX")
    if onnx_path:
        from classifier.infer import OnnxClassifier

        backbone = os.environ.get(
            "VI_ROUTER_BACKBONE", "microsoft/Multilingual-MiniLM-L12-H384"
        )
        max_tokens = int(os.environ.get("VI_ROUTER_MAX_TOKENS", "256"))
        clf = OnnxClassifier(onnx_path, backbone, max_tokens)
        clf.backend = "onnx"  # type: ignore[attr-defined]
        return clf

    torch_dir = os.environ.get("VI_ROUTER_TORCH")
    if torch_dir:
        from classifier.infer import TorchClassifier

        size = os.environ.get("VI_ROUTER_MODEL_SIZE", "vi-router-quality")
        clf = TorchClassifier(torch_dir, size)
        clf.backend = "torch"  # type: ignore[attr-defined]
        return clf

    return StubClassifier()


# ---------------------------------------------------------------------------
# Core routing function — usable without FastAPI (tests import this directly)
# ---------------------------------------------------------------------------


def route_prompt(
    classifier: Any,
    capabilities: Any,
    req: RouteRequest,
    *,
    default_latency_ceiling: float | None = None,
) -> RouteResponse:
    """Classify ``req.prompt`` then select a model. Pure, no global state."""
    t0 = time.perf_counter()
    analysis = classifier.predict([req.prompt])[0]

    ceiling = (
        req.latency_ceiling_ms_per_1k
        if req.latency_ceiling_ms_per_1k is not None
        else default_latency_ceiling
    )

    decision = select_model(
        capabilities,
        task_type=analysis["task_type_1"],
        complexity_score=analysis["prompt_complexity_score"],
        user_groups=req.user_groups,
        latency_ceiling_ms_per_1k=ceiling,
    )
    overhead_ms = (time.perf_counter() - t0) * 1000.0

    schema = load_label_schema()
    complexity_dims = {
        d: float(analysis.get(d, 0.0)) for d in schema.complexity_dimensions
    }

    return RouteResponse(
        model_id=decision.model_id,
        task_type=analysis["task_type_1"],
        task_type_secondary=analysis["task_type_2"],
        task_type_prob=float(analysis["task_type_prob"]),
        prompt_complexity_score=float(analysis["prompt_complexity_score"]),
        complexity=complexity_dims,
        required_skill=decision.required_skill,
        chosen_skill=decision.chosen_skill,
        est_cost_per_1k=decision.est_cost_per_1k,
        est_latency_ms_per_1k=decision.est_latency_ms_per_1k,
        cleared_bar=decision.cleared_bar,
        reason=decision.reason,
        alternatives=decision.alternatives,
        classifier_backend=getattr(classifier, "backend", "unknown"),
        routing_overhead_ms=round(overhead_ms, 3),
    )


# ---------------------------------------------------------------------------
# FastAPI app — import-guarded so the module loads without the serve extra
# ---------------------------------------------------------------------------


def create_app():
    """Build the FastAPI app. Imported lazily so ``fastapi`` is optional."""
    from fastapi import FastAPI

    app = FastAPI(
        title="Vietnamese AI Smart Router",
        description="Classifies a prompt and routes it to the cheapest capable model.",
        version="0.1.0",
    )

    # Loaded once at startup; cheap for the stub, lazy-heavy for ONNX/torch.
    state: dict[str, Any] = {}
    default_ceiling_env = os.environ.get("VI_ROUTER_LATENCY_CEIL")
    default_ceiling = float(default_ceiling_env) if default_ceiling_env else None

    @app.on_event("startup")
    def _startup() -> None:
        state["classifier"] = _build_classifier()
        state["capabilities"] = load_capabilities()

    @app.get("/health")
    def health() -> dict[str, Any]:
        clf = state.get("classifier")
        return {
            "status": "ok",
            "classifier_backend": getattr(clf, "backend", "uninitialized"),
            "num_models": len(state["capabilities"].models) if "capabilities" in state else 0,
        }

    @app.get("/leaderboard")
    def leaderboard(task_type: str) -> dict[str, Any]:
        rows = state["capabilities"].leaderboard(task_type)
        return {
            "task_type": task_type,
            "ranking": [
                {
                    "model_id": r.model_id,
                    "skill": r.skill,
                    "cost_per_1k_tokens": r.cost_per_1k_tokens,
                    "latency_ms_per_1k": r.latency_ms_per_1k,
                }
                for r in rows
            ],
        }

    @app.post("/route", response_model=RouteResponse)
    def route(req: RouteRequest) -> RouteResponse:
        return route_prompt(
            state["classifier"],
            state["capabilities"],
            req,
            default_latency_ceiling=default_ceiling,
        )

    return app


# Module-level app for `uvicorn serving.api:app`. Guard the import so simply
# importing this module (e.g. to use route_prompt in tests) doesn't require
# fastapi to be installed.
try:  # pragma: no cover - exercised only when serve extra present
    app = create_app()
except ImportError:  # pragma: no cover
    app = None
