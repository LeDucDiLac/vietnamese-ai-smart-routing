"""Leaderboard refresh job — success criterion #1 (plan §1, §5).

The "AI leaderboard auto-updated" deliverable. Reads the capability registry
(``configs/sim_models.yaml`` now; the real Viettel model list later) and writes a
ranked leaderboard artifact: per task type, every model ordered best-skill-first
with its cost and latency. Re-running this job after a model version changes (new
skill scores / prices in the registry) regenerates the ranking — that's the
"auto-updated" part.

Two outputs, same data:
- ``leaderboard.json`` — machine-readable, consumed by dashboards / the router's
  capability layer.
- ``leaderboard.md`` — human-readable tables for the handover report.

Pure-Python (no ML deps); runs anywhere in milliseconds.

    python -m router.leaderboard --out runs/leaderboard
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import load_label_schema
from router.capability import CapabilityTable, LeaderboardRow, load_capabilities


def build_leaderboard(
    capabilities: CapabilityTable | None = None,
    registry_path: str | None = None,
) -> dict[str, Any]:
    """Build the full leaderboard payload from the capability registry.

    Keyed by task-type id; each value is the ranked list of models for that task.
    A top-level ``overall`` ranking averages each model's skill across all tasks.
    """
    caps = capabilities or load_capabilities(registry_path)
    schema = load_label_schema()

    per_task: dict[str, list[dict[str, Any]]] = {}
    # rank within each task type the classifier can emit (schema order)
    for tid in schema.task_type_ids:
        per_task[tid] = [_row_to_dict(r, rank) for rank, r in enumerate(caps.leaderboard(tid), 1)]

    # overall: mean skill across the schema's task types, cheapest tie-break
    overall: list[dict[str, Any]] = []
    for m in caps.models:
        skills = [caps.skill(m.id, tid) for tid in schema.task_type_ids]
        overall.append(
            {
                "model_id": m.id,
                "display_name": m.display_name or m.id,
                "mean_skill": round(sum(skills) / len(skills), 4) if skills else 0.0,
                "cost_per_1k_tokens": m.cost_per_1k_tokens,
                "latency_ms_per_1k": m.latency_ms_per_1k,
                "permissions": m.permissions,
            }
        )
    overall.sort(key=lambda r: (-r["mean_skill"], r["cost_per_1k_tokens"], r["latency_ms_per_1k"]))
    for rank, row in enumerate(overall, 1):
        row["rank"] = rank

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_models": len(caps.models),
        "skill_floor": caps.skill_floor,
        "overall": overall,
        "by_task_type": per_task,
    }


def _row_to_dict(r: LeaderboardRow, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "model_id": r.model_id,
        "display_name": r.display_name,
        "skill": round(r.skill, 4),
        "cost_per_1k_tokens": r.cost_per_1k_tokens,
        "latency_ms_per_1k": r.latency_ms_per_1k,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the leaderboard payload as Markdown tables for the handover doc."""
    lines: list[str] = []
    lines.append("# AI Capability Leaderboard")
    lines.append("")
    lines.append(f"_Generated: {payload['generated_at']} — {payload['num_models']} models_")
    lines.append("")
    lines.append("## Overall (mean skill across task types)")
    lines.append("")
    lines.append("| Rank | Model | Mean skill | Cost /1k | Latency ms/1k | Permissions |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for row in payload["overall"]:
        lines.append(
            f"| {row['rank']} | {row['display_name']} | {row['mean_skill']:.3f} | "
            f"${row['cost_per_1k_tokens']:.4f} | {row['latency_ms_per_1k']:.0f} | "
            f"{', '.join(row['permissions'])} |"
        )
    lines.append("")
    lines.append("## Per task type")
    lines.append("")
    for tid, rows in payload["by_task_type"].items():
        lines.append(f"### {tid}")
        lines.append("")
        lines.append("| Rank | Model | Skill | Cost /1k | Latency ms/1k |")
        lines.append("|---:|---|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['display_name']} | {row['skill']:.3f} | "
                f"${row['cost_per_1k_tokens']:.4f} | {row['latency_ms_per_1k']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def refresh(
    out_dir: str | Path,
    *,
    registry_path: str | None = None,
) -> dict[str, str]:
    """Build and write ``leaderboard.json`` + ``leaderboard.md`` to ``out_dir``."""
    payload = build_leaderboard(registry_path=registry_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "leaderboard.json"
    md_path = out / "leaderboard.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"json": json_path.as_posix(), "markdown": md_path.as_posix()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the AI capability leaderboard")
    ap.add_argument("--out", default="runs/leaderboard")
    ap.add_argument("--registry", default=None, help="path to a model registry YAML")
    args = ap.parse_args()
    artifacts = refresh(args.out, registry_path=args.registry)
    print("Leaderboard refreshed:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
