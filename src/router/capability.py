"""Capability profiles + the model leaderboard (plan §5).

A *capability profile* is everything the router needs to know about one model
without ever running it: per-task skill scores, cost, latency, and which
permission groups may use it. Profiles are loaded from ``configs/sim_models.yaml``
(the RouterBench-seeded registry now; the real Viettel model list later — same
schema, see plan §11).

This module is pure-Python (pydantic only) so it imports in the CPU-only / no-ML
serving path. It exposes:

- :func:`load_capabilities` — read the registry into a :class:`CapabilityTable`.
- :class:`CapabilityTable` — lookup + the leaderboard job (success criterion #1).
"""

from __future__ import annotations

from dataclasses import dataclass

from config import ModelProfile, ModelRegistry, load_model_registry, task_type_id


@dataclass(frozen=True)
class LeaderboardRow:
    """One model's standing for a given task type."""

    model_id: str
    display_name: str
    task_type: str
    skill: float
    cost_per_1k_tokens: float
    latency_ms_per_1k: float


class CapabilityTable:
    """In-memory view over the model registry.

    Wraps :class:`config.ModelRegistry` with the queries the router needs:
    skill lookups, cost/latency, and the per-task leaderboard. Everything is
    plain dict/list arithmetic — microsecond-scale, fits the 50 ms budget with
    room to spare (plan §5).
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._by_id = {m.id: m for m in registry.models}

    # -- registry passthroughs ------------------------------------------------

    @property
    def skill_floor(self) -> float:
        return self._registry.skill_floor

    @property
    def models(self) -> list[ModelProfile]:
        return list(self._registry.models)

    def get(self, model_id: str) -> ModelProfile:
        return self._by_id[model_id]

    # -- queries --------------------------------------------------------------

    def skill(self, model_id: str, task_type: str) -> float:
        """Skill score of ``model_id`` for ``task_type`` (display label or id)."""
        return self._by_id[model_id].skill_for(task_type, self.skill_floor)

    def leaderboard(self, task_type: str) -> list[LeaderboardRow]:
        """Models ranked best-skill-first for one task type (success criterion #1).

        This is the "AI leaderboard auto-updated" deliverable: rebuilding the
        ``CapabilityTable`` from a refreshed registry re-ranks everything.
        """
        tid = task_type_id(task_type)
        rows = [
            LeaderboardRow(
                model_id=m.id,
                display_name=m.display_name or m.id,
                task_type=tid,
                skill=m.skill_for(tid, self.skill_floor),
                cost_per_1k_tokens=m.cost_per_1k_tokens,
                latency_ms_per_1k=m.latency_ms_per_1k,
            )
            for m in self._registry.models
        ]
        rows.sort(key=lambda r: (-r.skill, r.cost_per_1k_tokens, r.latency_ms_per_1k))
        return rows

    def full_leaderboard(self) -> dict[str, list[LeaderboardRow]]:
        """Leaderboard for every task type, keyed by task id."""
        # task ids are the union of skill keys across models
        task_ids: set[str] = set()
        for m in self._registry.models:
            task_ids.update(m.skills.keys())
        return {t: self.leaderboard(t) for t in sorted(task_ids)}


def load_capabilities(path: str | None = None) -> CapabilityTable:
    """Load the capability table from the model registry YAML."""
    return CapabilityTable(load_model_registry(path))
