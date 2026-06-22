"""One-time Vietnamese response cache (plan §11.3 caveat).

RouterBench is English-only. To extend the simulation to Vietnamese we take a few
hundred VN prompts, call public model APIs **once**, and cache every answer +
its cost + an LLM-judge quality score to disk in the long RouterBench schema
(``routerbench_adapter.ColumnMap`` defaults). From then on the VN simulation is
100% offline — this file is a permanent fixture, regenerated only if the model
list changes.

Cost is estimated from token counts × per-model price (``configs/sim_models.yaml``)
so the cached ``cost`` column is consistent with the router's own cost model.

Modes:
- **synthetic** (default, no network): fabricates plausible outcomes from each
  model's capability profile so the eval harness has data to run on immediately,
  with zero spend. Quality = the model's profile skill for the prompt's task type
  (+ small noise); cost = profile price × a nominal token count. Deterministic.
- **api**: actually calls an OpenAI-compatible endpoint per (prompt, model) and an
  LLM judge for scoring. Requires ``openai`` + keys. This is the "few-dollar,
  one-time" run; its output is identical in schema to synthetic mode.

    python -m sim.vi_response_cache --prompts data/processed/val.jsonl \
        --out data/sim/vi_cache.jsonl --mode synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from config import (
    load_complexity,
    load_label_schema,
    load_model_registry,
    task_type_id,
)


def _read_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _nominal_tokens(text: str) -> float:
    """Rough token count: ~1.3 tokens/word, in thousands, floored at a small value."""
    words = max(1, len(text.split()))
    # output is usually longer than the prompt; assume ~3x for a generated answer
    return max(0.05, words * 1.3 * 3 / 1000.0)


def build_synthetic(
    prompts: list[dict[str, Any]], seed: int = 23
) -> list[dict[str, Any]]:
    """Fabricate per-(prompt, model) outcomes from capability profiles.

    Quality is the model's profile skill for the prompt's task type with light
    noise; cost is profile price × nominal token count. This gives the harness a
    self-consistent dataset where the router's own cost model and the cached costs
    agree, so cost-savings arithmetic is exact (plan §11.4).
    """
    registry = load_model_registry()
    complexity = load_complexity()
    schema = load_label_schema()
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    for i, p in enumerate(prompts):
        text = p["text"]
        # use the row's task_type if present, else fall back to "other"
        task = p.get("task_type", "other")
        tid = task_type_id(task) if task else "other"
        # complexity score from the row's dims if present
        dims = {d: float(p.get(d, 0.0)) for d in schema.complexity_dimensions}
        cscore = complexity.score(dims)
        ktokens = _nominal_tokens(text)
        for m in registry.models:
            skill = m.skill_for(tid, registry.skill_floor)
            # quality: skill, nudged down a touch by complexity (hard prompts are
            # harder for everyone), plus small noise; clamped to [0,1].
            q = skill - 0.15 * cscore * (1.0 - skill) + rng.uniform(-0.03, 0.03)
            q = max(0.0, min(1.0, q))
            cost = m.cost_per_1k_tokens * ktokens
            rows.append(
                {
                    "prompt_id": str(p.get("id", i)),
                    "prompt": text,
                    "model_id": m.id,
                    "cost": round(cost, 8),
                    "quality": round(q, 4),
                    "latency_ms": round(m.latency_ms_per_1k * ktokens, 2),
                    "task_type": tid,
                    "complexity_score": round(cscore, 4),
                }
            )
    return rows


def build_api(
    prompts: list[dict[str, Any]],
    *,
    judge_model: str = "gpt-4o-mini",
    seed: int = 23,
) -> list[dict[str, Any]]:
    """Call real APIs once per (prompt, model) + an LLM judge for quality.

    The cached output schema is identical to :func:`build_synthetic`, so the
    harness can't tell which produced the fixture. Each model id in the registry
    is treated as an OpenAI-compatible model name; override ``OPENAI_BASE_URL`` to
    target Viettel's gateway. This is the one-time spend described in plan §11.3.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("api mode needs `pip install openai` + OPENAI_API_KEY") from exc

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
    registry = load_model_registry()
    complexity = load_complexity()
    schema = load_label_schema()
    rows: list[dict[str, Any]] = []

    def _judge(prompt: str, answer: str) -> float:
        """LLM-judge quality in [0,1]. Best-effort; defaults to 0.5 on failure."""
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là giám khảo. Chấm điểm chất lượng câu trả lời từ 0 "
                            "đến 100, chỉ trả về một con số."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Câu hỏi:\n{prompt}\n\nCâu trả lời:\n{answer}\n\nĐiểm:",
                    },
                ],
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            num = float("".join(c for c in raw if (c.isdigit() or c == ".")) or "50")
            return max(0.0, min(1.0, num / 100.0))
        except Exception:  # pragma: no cover - network path
            return 0.5

    for i, p in enumerate(prompts):
        text = p["text"]
        tid = task_type_id(p.get("task_type", "other") or "other")
        dims = {d: float(p.get(d, 0.0)) for d in schema.complexity_dimensions}
        cscore = complexity.score(dims)
        for m in registry.models:
            try:
                resp = client.chat.completions.create(
                    model=m.id,
                    messages=[{"role": "user", "content": text}],
                    temperature=0.7,
                )
                answer = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                out_tokens = getattr(usage, "completion_tokens", None) or 0
                ktokens = max(0.05, out_tokens / 1000.0) if out_tokens else _nominal_tokens(text)
            except Exception as exc:  # pragma: no cover - network path
                print(f"  api call failed for {m.id} on prompt {i}: {exc}")
                answer = ""
                ktokens = _nominal_tokens(text)
            quality = _judge(text, answer) if answer else 0.0
            rows.append(
                {
                    "prompt_id": str(p.get("id", i)),
                    "prompt": text,
                    "model_id": m.id,
                    "cost": round(m.cost_per_1k_tokens * ktokens, 8),
                    "quality": round(quality, 4),
                    "latency_ms": round(m.latency_ms_per_1k * ktokens, 2),
                    "response": answer,
                    "task_type": tid,
                    "complexity_score": round(cscore, 4),
                }
            )
    return rows


def build(
    prompts_path: str | Path,
    out_path: str | Path,
    *,
    mode: str = "synthetic",
    limit: int | None = None,
    seed: int = 23,
) -> int:
    prompts = _read_prompts(Path(prompts_path))
    if limit:
        prompts = prompts[:limit]
    if mode == "api":
        rows = build_api(prompts, seed=seed)
    else:
        rows = build_synthetic(prompts, seed=seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the one-time VN response cache")
    ap.add_argument("--prompts", required=True, help="JSONL of prompts (text + task_type)")
    ap.add_argument("--out", default="data/sim/vi_cache.jsonl")
    ap.add_argument("--mode", choices=["synthetic", "api"], default="synthetic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    n = build(args.prompts, args.out, mode=args.mode, limit=args.limit, seed=args.seed)
    print(f"wrote {n} cached outcome rows to {args.out}")


if __name__ == "__main__":
    main()
