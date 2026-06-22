"""Assemble the final processed dataset (plan §3, §4).

Takes labeled rows from any of the three sources — crawl+teacher labels
(``data/raw`` -> labeled), synthetic generation (``data/synthetic``), and the
human-gold set (``data/gold``) — validates them against the label schema,
dedupes, splits, and writes ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` to
``data/processed``.

Rules (plan §3.2):
- **silver** rows (teacher / synthetic labels) -> train only.
- **gold** rows (human-reviewed) -> val + test only, never train.

Pure-Python (stdlib only) so it runs anywhere, including as a Kaggle dataset
build step before training.

    python -m data.build_dataset \
        --silver data/synthetic/synth.jsonl data/raw/labeled.jsonl \
        --gold data/gold/gold.jsonl \
        --out data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from config import LabelSchema, load_label_schema


def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def _dedup_key(text: str) -> str:
    return hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()


def validate_row(row: dict[str, Any], schema: LabelSchema) -> dict[str, Any]:
    """Coerce + validate one labeled row to the canonical schema.

    Raises ``ValueError`` with a clear message on malformed rows so a bad
    teacher/synth output fails loudly at build time, not silently at train time.
    """
    if "text" not in row or not str(row["text"]).strip():
        raise ValueError(f"row missing non-empty 'text': {row!r}")
    if "task_type" not in row:
        raise ValueError(f"row missing 'task_type': {row!r}")

    # validate task_type is a known label/id, and canonicalize to the display
    # label so snake_case (synth) and display-label (teacher) rows don't split
    # into separate strata / report buckets.
    try:
        canon = schema.task_types[schema.task_index(row["task_type"])]
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    clean: dict[str, Any] = {
        "text": str(row["text"]).strip(),
        "task_type": canon,
    }
    for dim in schema.complexity_dimensions:
        val = float(row.get(dim, 0.0))
        # complexity dims are bounded [0,1]
        clean[dim] = max(0.0, min(1.0, val))
    # keep provenance if present (source teacher, agreement, etc.)
    if "source" in row:
        clean["source"] = row["source"]
    return clean


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_and_validate(
    paths: list[str], schema: LabelSchema, *, skip_bad: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_bad = 0
    for p in paths:
        for raw in read_jsonl(p):
            try:
                rows.append(validate_row(raw, schema))
            except ValueError as exc:
                n_bad += 1
                if not skip_bad:
                    raise
                if n_bad <= 5:
                    print(f"  skipping bad row in {p}: {exc}")
    if n_bad:
        print(f"  ({n_bad} bad rows skipped total)")
    return rows


def _stratify_by_task(
    rows: list[dict[str, Any]], schema: LabelSchema
) -> dict[int, list[dict[str, Any]]]:
    """Group rows by task_type class index."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        idx = schema.task_index(r["task_type"])
        buckets.setdefault(idx, []).append(r)
    return buckets


def build(
    silver_paths: list[str],
    gold_paths: list[str],
    out_dir: str | Path,
    *,
    val_frac: float = 0.5,
    silver_eval_frac: float = 0.1,
    seed: int = 13,
    skip_bad: bool = True,
) -> dict[str, int]:
    """Build train/val/test splits.

    Eval rows (val + test) come from **gold** when available (plan §3.2). When
    gold is absent or thin, a stratified ``silver_eval_frac`` of silver is held
    out per task_type so val/test are never empty and every class is represented.
    Held-out silver eval rows are tagged ``"eval_source": "silver_holdout"`` so
    the gold-only purity of eval can be restored later by re-running with gold.

    - ``val_frac``: fraction of the eval pool sent to validation (rest -> test).
    - ``silver_eval_frac``: per-class fraction of silver held out for eval when
      gold doesn't already cover that class. Set 0.0 to disable (gold-only eval).
    """
    schema = load_label_schema()
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    silver = _load_and_validate(silver_paths, schema, skip_bad=skip_bad)
    gold = _load_and_validate(gold_paths, schema, skip_bad=skip_bad)

    # dedupe: gold wins over silver; within a tier, first occurrence wins.
    seen: set[str] = set()
    gold_clean: list[dict[str, Any]] = []
    for r in gold:
        k = _dedup_key(r["text"])
        if k not in seen:
            seen.add(k)
            gold_clean.append(r)
    silver_clean: list[dict[str, Any]] = []
    for r in silver:
        k = _dedup_key(r["text"])
        if k not in seen:
            seen.add(k)
            silver_clean.append(r)

    # Stratified silver holdout for eval, per task_type. Only hold out for a
    # class if gold doesn't already provide eval coverage for it.
    gold_classes = {schema.task_index(r["task_type"]) for r in gold_clean}
    eval_extra: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    if silver_eval_frac > 0.0:
        for idx, bucket in _stratify_by_task(silver_clean, schema).items():
            rng.shuffle(bucket)
            n_hold = 0 if idx in gold_classes else int(len(bucket) * silver_eval_frac)
            for r in bucket[:n_hold]:
                r = dict(r)
                r["eval_source"] = "silver_holdout"
                eval_extra.append(r)
            train_rows.extend(bucket[n_hold:])
    else:
        train_rows = list(silver_clean)

    # Build the eval pool (gold first, then stratified silver holdout) and split.
    eval_pool = gold_clean + eval_extra
    eval_by_task = _stratify_by_task(eval_pool, schema)
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for bucket in eval_by_task.values():
        rng.shuffle(bucket)
        n_val = round(len(bucket) * val_frac)
        val_rows.extend(bucket[:n_val])
        test_rows.extend(bucket[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    silver_clean = train_rows  # train output

    def _write(name: str, rows: list[dict[str, Any]]) -> int:
        path = out / name
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    counts = {
        "train": _write("train.jsonl", silver_clean),
        "val": _write("val.jsonl", val_rows),
        "test": _write("test.jsonl", test_rows),
    }
    # task-type distribution report for sanity (plan §4 balance check)
    def _dist(rows: list[dict[str, Any]]) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in rows:
            d[r["task_type"]] = d.get(r["task_type"], 0) + 1
        return d

    n_silver_holdout = sum(1 for r in val_rows + test_rows if r.get("eval_source") == "silver_holdout")
    (out / "build_report.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "eval_silver_holdout": n_silver_holdout,
                "eval_gold": len(gold_clean),
                "train_task_dist": _dist(silver_clean),
                "val_task_dist": _dist(val_rows),
                "test_task_dist": _dist(test_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble processed train/val/test")
    ap.add_argument("--silver", nargs="*", default=[], help="silver-label JSONL files")
    ap.add_argument("--gold", nargs="*", default=[], help="human-gold JSONL files")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument(
        "--silver-eval-frac",
        type=float,
        default=0.1,
        help="per-class fraction of silver held out for eval when gold is absent (0 = gold-only eval)",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--strict", action="store_true", help="fail on first bad row")
    args = ap.parse_args()

    counts = build(
        args.silver,
        args.gold,
        args.out,
        val_frac=args.val_frac,
        silver_eval_frac=args.silver_eval_frac,
        seed=args.seed,
        skip_bad=not args.strict,
    )
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
