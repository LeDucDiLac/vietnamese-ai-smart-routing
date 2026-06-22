"""Offline routing simulation — the four handover metrics (plan §6, §11.4).

Replays a held-out prompt set through three policies and reports the numbers we
hand over as the success report:

- **always-best**   : every prompt to the highest-quality model (quality ceiling,
                      cost ceiling).
- **always-cheap**  : every prompt to the cheapest model (cost floor, quality floor).
- **router (ours)** : classify -> ``router.match.select_model`` -> read the chosen
                      model's pre-recorded outcome for that prompt.

Because the dataset (RouterBench / the VN cache from ``sim.vi_response_cache``)
carries each model's **pre-computed** cost + quality per prompt, every policy's
cost/quality/latency is pure arithmetic over cached columns — no live model calls
(plan §11.4). We report:

- cost delta vs. always-best        (target >= 30% reduction)
- quality delta vs. always-best     (target <= 3% drop)
- latency delta on simple prompts   (target >= 20% faster)
- routing overhead p50/p95          (target <= 50 ms)

Pure-Python (stdlib only). Runs anywhere, no ML deps:

    python -m tests.eval.simulate \
        --data data/sim/vi_cache.jsonl \
        --out runs/eval/report.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from router.capability import load_capabilities
from router.match import select_model
from sim.routerbench_adapter import ReplayRecord, load_long_format, model_ids_in


# ---------------------------------------------------------------------------
# Percentile helper (no numpy dependency)
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass
class PolicyResult:
    name: str
    n_prompts: int
    total_cost: float
    mean_quality: float
    mean_latency_ms: float
    # latency on "simple" prompts only (complexity below median)
    mean_latency_simple_ms: float


def _evaluate_policy(
    name: str,
    records: list[ReplayRecord],
    pick: Callable[[ReplayRecord], str],
    simple_ids: set[str],
) -> tuple[PolicyResult, list[float]]:
    """Run ``pick`` over every record; aggregate cost/quality/latency.

    Returns the aggregate plus the per-prompt decision-overhead samples (ms).
    """
    total_cost = 0.0
    qualities: list[float] = []
    latencies: list[float] = []
    simple_latencies: list[float] = []
    overheads: list[float] = []

    for rec in records:
        t0 = time.perf_counter()
        chosen_id = pick(rec)
        overheads.append((time.perf_counter() - t0) * 1000.0)

        outcome = rec.outcome_for(chosen_id)
        if outcome is None:
            # chosen model has no recorded outcome for this prompt — skip it
            continue
        total_cost += outcome.cost
        qualities.append(outcome.quality)
        if outcome.latency_ms is not None:
            latencies.append(outcome.latency_ms)
            if rec.prompt_id in simple_ids:
                simple_latencies.append(outcome.latency_ms)

    result = PolicyResult(
        name=name,
        n_prompts=len(records),
        total_cost=round(total_cost, 6),
        mean_quality=round(mean(qualities), 6) if qualities else 0.0,
        mean_latency_ms=round(mean(latencies), 4) if latencies else 0.0,
        mean_latency_simple_ms=round(mean(simple_latencies), 4)
        if simple_latencies
        else 0.0,
    )
    return result, overheads


# ---------------------------------------------------------------------------
# Router pick — classify (or reuse cached labels) then match
# ---------------------------------------------------------------------------


def _make_router_pick(
    capabilities: Any,
    classifier: Any | None,
    *,
    user_groups: list[str] | None,
    latency_ceiling: float | None,
) -> Callable[[ReplayRecord], str]:
    """Build the router policy's per-record pick function.

    If the record already carries ``task_type`` + ``complexity_score`` (the VN
    cache does), we route off those directly — no classifier needed, so the
    harness runs with zero ML deps. Otherwise we require a classifier.
    """

    def pick(rec: ReplayRecord) -> str:
        if rec.task_type is not None and rec.complexity_score is not None:
            task = rec.task_type
            score = rec.complexity_score
        else:
            if classifier is None:
                raise RuntimeError(
                    "record has no cached labels and no classifier was supplied; "
                    "pass --classifier or use a fixture with task_type/complexity_score"
                )
            analysis = classifier.predict([rec.prompt])[0]
            task = analysis["task_type_1"]
            score = analysis["prompt_complexity_score"]

        decision = select_model(
            capabilities,
            task_type=task,
            complexity_score=score,
            user_groups=user_groups,
            latency_ceiling_ms_per_1k=latency_ceiling,
        )
        # The matcher may pick a model id that isn't in this record's registry
        # (e.g. registry richer than the fixture). Fall back to the cheapest
        # recorded outcome that still clears the bar, else any recorded model.
        if decision.model_id in rec.outcomes:
            return decision.model_id
        # try alternatives in order
        for alt in decision.alternatives:
            if alt in rec.outcomes:
                return alt
        return rec.cheapest().model_id

    return pick


# ---------------------------------------------------------------------------
# Top-level simulation
# ---------------------------------------------------------------------------


@dataclass
class SimReport:
    policies: dict[str, dict[str, Any]]
    deltas: dict[str, float]
    routing_overhead_ms: dict[str, float]
    targets: dict[str, Any]
    meta: dict[str, Any]


def simulate(
    data_path: str | Path,
    *,
    user_groups: list[str] | None = None,
    latency_ceiling: float | None = None,
    classifier: Any | None = None,
) -> SimReport:
    records = load_long_format(data_path)
    if not records:
        raise ValueError(f"no records loaded from {data_path}")

    capabilities = load_capabilities()

    # "simple" prompts = complexity below the median (for the latency-win metric).
    scores = [
        r.complexity_score if r.complexity_score is not None else 0.5 for r in records
    ]
    median_score = _percentile(scores, 50)
    simple_ids = {
        r.prompt_id
        for r in records
        if (r.complexity_score if r.complexity_score is not None else 0.5)
        <= median_score
    }

    # baseline picks
    best_pick = lambda rec: rec.best_quality().model_id  # noqa: E731
    cheap_pick = lambda rec: rec.cheapest().model_id  # noqa: E731
    router_pick = _make_router_pick(
        capabilities,
        classifier,
        user_groups=user_groups,
        latency_ceiling=latency_ceiling,
    )

    best_res, _ = _evaluate_policy("always-best", records, best_pick, simple_ids)
    cheap_res, _ = _evaluate_policy("always-cheap", records, cheap_pick, simple_ids)
    router_res, overheads = _evaluate_policy(
        "router", records, router_pick, simple_ids
    )

    # deltas vs. always-best (the quality reference)
    cost_reduction = (
        (best_res.total_cost - router_res.total_cost) / best_res.total_cost
        if best_res.total_cost
        else 0.0
    )
    quality_drop = (
        (best_res.mean_quality - router_res.mean_quality) / best_res.mean_quality
        if best_res.mean_quality
        else 0.0
    )
    # latency win on simple prompts: router vs. always-best
    latency_simple_reduction = (
        (best_res.mean_latency_simple_ms - router_res.mean_latency_simple_ms)
        / best_res.mean_latency_simple_ms
        if best_res.mean_latency_simple_ms
        else 0.0
    )

    deltas = {
        "cost_reduction_vs_best": round(cost_reduction, 4),
        "quality_drop_vs_best": round(quality_drop, 4),
        "latency_reduction_simple_vs_best": round(latency_simple_reduction, 4),
    }
    overhead_stats = {
        "p50_ms": round(_percentile(overheads, 50), 4),
        "p95_ms": round(_percentile(overheads, 95), 4),
        "mean_ms": round(mean(overheads), 4) if overheads else 0.0,
    }
    targets = {
        "cost_reduction_vs_best>=0.30": cost_reduction >= 0.30,
        "quality_drop_vs_best<=0.03": quality_drop <= 0.03,
        "latency_reduction_simple>=0.20": latency_simple_reduction >= 0.20,
        "routing_overhead_p95<=50ms": overhead_stats["p95_ms"] <= 50.0,
    }

    return SimReport(
        policies={
            "always_best": asdict(best_res),
            "always_cheap": asdict(cheap_res),
            "router": asdict(router_res),
        },
        deltas=deltas,
        routing_overhead_ms=overhead_stats,
        targets=targets,
        meta={
            "data_path": str(data_path),
            "n_records": len(records),
            "models": model_ids_in(records),
            "median_complexity": round(median_score, 4),
            "n_simple_prompts": len(simple_ids),
            "user_groups": user_groups,
            "latency_ceiling_ms_per_1k": latency_ceiling,
            "note": "routing_overhead measures decision time only (matcher + "
            "cached-label lookup); add classifier inference for end-to-end.",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay routing policies, emit metrics")
    ap.add_argument("--data", required=True, help="long-format RouterBench/VN-cache JSONL")
    ap.add_argument("--out", default="runs/eval/report.json")
    ap.add_argument(
        "--user-groups", nargs="*", default=None, help="caller permission groups"
    )
    ap.add_argument("--latency-ceiling", type=float, default=None)
    args = ap.parse_args()

    report = simulate(
        args.data,
        user_groups=args.user_groups,
        latency_ceiling=args.latency_ceiling,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
