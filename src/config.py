"""Shared config loader.

Single entry point for reading the YAML configs in ``configs/``. Every other
module (classifier, router, sim, serving) imports from here so the label schema,
complexity weights and model registry stay in lockstep.

No heavy deps — just pyyaml + pydantic — so this imports fine in a CPU-only,
ML-stack-absent environment (e.g. the router / sim path).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _resolve_repo_root() -> Path:
    """Locate the repo root, tolerating Kaggle / relocated layouts.

    Resolution order:
    1. ``VI_ROUTER_REPO_ROOT`` env var, if set — explicit override for Kaggle,
       where the code may live under ``/kaggle/input/<dataset>`` or
       ``/kaggle/working`` and the cwd is unrelated.
    2. The package's own location (``src/config.py`` -> two parents up). Works
       for a normal checkout and for an editable install.
    3. If that has no ``configs/`` dir, scan ``/kaggle/input/**`` for a unique
       ``configs/label_schema.yaml`` so an uploaded-repo dataset is found with
       no configuration.
    """
    env = os.environ.get("VI_ROUTER_REPO_ROOT")
    if env:
        return Path(env).resolve()

    here = Path(__file__).resolve().parent.parent
    if (here / "configs" / "label_schema.yaml").exists():
        return here

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        hits = sorted(kaggle_input.glob("**/configs/label_schema.yaml"))
        if len(hits) == 1:
            return hits[0].parent.parent

    return here  # leave as-is; loaders raise a clear FileNotFoundError


REPO_ROOT = _resolve_repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of {path}, got {type(data)}")
    return data


def task_type_id(display: str) -> str:
    """Normalize a task-type label to its snake_case id.

    ``"Open QA" -> "open_qa"``. Used to bridge the human-readable labels in
    ``label_schema.yaml`` with the snake_case skill keys in ``sim_models.yaml``.
    """
    return display.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------


class LabelSchema(BaseModel):
    task_types: list[str]
    task_types_vi: dict[str, str] = Field(default_factory=dict)
    complexity_dimensions: list[str]
    complexity_dimensions_vi: dict[str, str] = Field(default_factory=dict)

    @property
    def task_type_ids(self) -> list[str]:
        """snake_case ids, in head-index order."""
        return [task_type_id(t) for t in self.task_types]

    @property
    def num_task_types(self) -> int:
        return len(self.task_types)

    def task_index(self, label: str) -> int:
        """Index of a task type by either display label or snake_case id."""
        tid = task_type_id(label)
        for i, t in enumerate(self.task_types):
            if task_type_id(t) == tid:
                return i
        raise KeyError(f"Unknown task type: {label!r}")


# ---------------------------------------------------------------------------
# Complexity config
# ---------------------------------------------------------------------------


class ComplexityConfig(BaseModel):
    weights_map: dict[str, float]
    divisor_map: dict[str, float]
    target_sizes: dict[str, int]

    @model_validator(mode="after")
    def _check_weights(self) -> ComplexityConfig:
        total = sum(self.weights_map.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights_map must sum to 1.0, got {total}")
        return self

    def score(self, dims: dict[str, float]) -> float:
        """Compute prompt_complexity_score from per-dimension values.

        Each value is divided by its divisor (normalization) then weighted.
        Missing dimensions are treated as 0.
        """
        total = 0.0
        for dim, weight in self.weights_map.items():
            raw = float(dims.get(dim, 0.0))
            divisor = self.divisor_map.get(dim, 1.0) or 1.0
            total += weight * (raw / divisor)
        return total


# ---------------------------------------------------------------------------
# Model registry (sim)
# ---------------------------------------------------------------------------


class ModelProfile(BaseModel):
    id: str
    display_name: str = ""
    cost_per_1k_tokens: float
    latency_ms_per_1k: float
    permissions: list[str] = Field(default_factory=lambda: ["all"])
    skills: dict[str, float] = Field(default_factory=dict)

    def skill_for(self, task: str, floor: float) -> float:
        """Skill score for a task type (display label or id), falling back to floor."""
        return self.skills.get(task_type_id(task), floor)


class ModelRegistry(BaseModel):
    skill_floor: float = 0.40
    models: list[ModelProfile]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> ModelRegistry:
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate model ids in registry: {ids}")
        return self

    def by_id(self, model_id: str) -> ModelProfile:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"Unknown model id: {model_id!r}")


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def load_label_schema(path: str | None = None) -> LabelSchema:
    p = Path(path) if path else CONFIGS_DIR / "label_schema.yaml"
    return LabelSchema(**_read_yaml(p))


@functools.lru_cache(maxsize=None)
def load_complexity(path: str | None = None) -> ComplexityConfig:
    p = Path(path) if path else CONFIGS_DIR / "complexity.yaml"
    return ComplexityConfig(**_read_yaml(p))


@functools.lru_cache(maxsize=None)
def load_model_registry(path: str | None = None) -> ModelRegistry:
    p = Path(path) if path else CONFIGS_DIR / "sim_models.yaml"
    raw = _read_yaml(p)
    skill_floor = raw.get("defaults", {}).get("skill_floor", 0.40)
    return ModelRegistry(skill_floor=skill_floor, models=raw["models"])


@functools.lru_cache(maxsize=None)
def load_model_configs(path: str | None = None) -> dict[str, dict[str, Any]]:
    """Classifier model configs (vi-router-quality / vi-router-fast)."""
    p = Path(path) if path else CONFIGS_DIR / "model.yaml"
    return _read_yaml(p)
