"""Load RouterBench-format data into our replay format (plan §11.1).

RouterBench is a public benchmark of ~30k prompts with **pre-computed** responses
from 11 real LLMs — each record carries the prompt, every model's answer, that
answer's estimated cost, and a correctness/quality score. Because the answers,
costs and scores are already recorded, we can replay any routing policy over the
dataset and read off cost/quality/latency **without running a single model**
(plan §11.4).

This adapter is schema-tolerant: RouterBench has shipped in a couple of column
layouts, and the Vietnamese fixture we build (``vi_response_cache.py``) writes
the same logical schema. Rather than hard-code column names we accept a small
mapping and normalize everything to :class:`ReplayRecord`.

Logical schema we normalize to, per (prompt, model):
    prompt_id, prompt, model_id, cost, quality, [latency]

Pure-Python + stdlib (optionally pandas/datasets if you load a HF parquet). The
core replay math lives in ``tests/eval/simulate.py``; this file only loads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelOutcome:
    """One model's pre-recorded result for one prompt."""

    model_id: str
    cost: float
    quality: float
    latency_ms: float | None = None
    response: str | None = None


@dataclass
class ReplayRecord:
    """All models' outcomes for a single prompt — the unit the harness replays."""

    prompt_id: str
    prompt: str
    outcomes: dict[str, ModelOutcome] = field(default_factory=dict)
    # optional pre-classified labels (if the fixture carries them)
    task_type: str | None = None
    complexity_score: float | None = None

    def best_quality(self) -> ModelOutcome:
        return max(self.outcomes.values(), key=lambda o: o.quality)

    def cheapest(self) -> ModelOutcome:
        return min(self.outcomes.values(), key=lambda o: o.cost)

    def outcome_for(self, model_id: str) -> ModelOutcome | None:
        return self.outcomes.get(model_id)


@dataclass
class ColumnMap:
    """Maps source column names to our logical schema.

    Defaults match the JSONL fixture written by ``vi_response_cache.py``. For the
    real RouterBench parquet, pass the mapping that matches its columns.
    """

    prompt_id: str = "prompt_id"
    prompt: str = "prompt"
    model_id: str = "model_id"
    cost: str = "cost"
    quality: str = "quality"
    latency_ms: str | None = "latency_ms"
    response: str | None = "response"
    task_type: str | None = "task_type"
    complexity_score: str | None = "complexity_score"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_long_format(
    path: str | Path, colmap: ColumnMap | None = None
) -> list[ReplayRecord]:
    """Load a *long* (tidy) table: one row per (prompt, model).

    This is the schema our VN fixture writes and the easiest RouterBench export to
    consume. Rows sharing a ``prompt_id`` are grouped into one :class:`ReplayRecord`.
    """
    cm = colmap or ColumnMap()
    by_prompt: dict[str, ReplayRecord] = {}
    for row in _read_jsonl(Path(path)):
        pid = str(row[cm.prompt_id])
        rec = by_prompt.get(pid)
        if rec is None:
            rec = ReplayRecord(
                prompt_id=pid,
                prompt=str(row[cm.prompt]),
                task_type=(
                    row.get(cm.task_type) if cm.task_type else None
                ),
                complexity_score=(
                    float(row[cm.complexity_score])
                    if cm.complexity_score and row.get(cm.complexity_score) is not None
                    else None
                ),
            )
            by_prompt[pid] = rec
        latency = (
            float(row[cm.latency_ms])
            if cm.latency_ms and row.get(cm.latency_ms) is not None
            else None
        )
        mid = str(row[cm.model_id])
        rec.outcomes[mid] = ModelOutcome(
            model_id=mid,
            cost=float(row[cm.cost]),
            quality=float(row[cm.quality]),
            latency_ms=latency,
            response=(
                row.get(cm.response) if cm.response else None
            ),
        )
    return list(by_prompt.values())


def load_wide_format(
    path: str | Path,
    model_ids: list[str],
    *,
    prompt_key: str = "prompt",
    prompt_id_key: str = "prompt_id",
    cost_suffix: str = "_cost",
    quality_suffix: str = "_score",
    latency_suffix: str = "_latency_ms",
) -> list[ReplayRecord]:
    """Load a *wide* table: one row per prompt, per-model columns.

    RouterBench's published parquet is wide — each model contributes columns like
    ``<model>_score`` and ``<model>_cost``. Pass the list of model ids and the
    column suffixes; everything else is derived.
    """
    records: list[ReplayRecord] = []
    for i, row in enumerate(_read_jsonl(Path(path))):
        pid = str(row.get(prompt_id_key, i))
        rec = ReplayRecord(prompt_id=pid, prompt=str(row.get(prompt_key, "")))
        for mid in model_ids:
            cost_col = f"{mid}{cost_suffix}"
            q_col = f"{mid}{quality_suffix}"
            if cost_col not in row or q_col not in row:
                continue
            lat_col = f"{mid}{latency_suffix}"
            rec.outcomes[mid] = ModelOutcome(
                model_id=mid,
                cost=float(row[cost_col]),
                quality=float(row[q_col]),
                latency_ms=float(row[lat_col]) if lat_col in row else None,
            )
        if rec.outcomes:
            records.append(rec)
    return records


def model_ids_in(records: Iterable[ReplayRecord]) -> list[str]:
    """Union of model ids appearing across all records (for the registry)."""
    ids: set[str] = set()
    for r in records:
        ids.update(r.outcomes.keys())
    return sorted(ids)
