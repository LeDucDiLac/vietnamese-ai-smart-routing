"""Convert v1 dataset (11-class / 6-dim NVIDIA schema) to v1.5 (v2 schema, rules-based).

No LLM calls — deterministic mapping only.  Reads data/processed/{train,val,test}.jsonl
(v1 labels), writes data/processed/v1.5/{train,val,test}.jsonl with v2 task types and
v2 complexity dims derived from v1 values.

Task-type mapping (v1 → v2)
────────────────────────────
  Open QA         → Knowledge Retrieval / QA
  Closed QA       → Knowledge Retrieval / QA
  Classification  → Knowledge Retrieval / QA
  Extraction      → Knowledge Retrieval / QA
  Summarization   → Summarization
  Text Generation → Content Creation
  Rewrite         → Content Creation
  Brainstorming   → Content Creation
  Code Generation → Code
  Chatbot         → Conversation
  Other           → Conversation

Complexity mapping (v1 dims → v2 dims)
───────────────────────────────────────
  reasoning_depth      = v1.reasoning
  domain_knowledge     = max(v1.domain_knowledge, v1.contextual_knowledge)
  instruction_precision = v1.constraint_ct

Dropped v1 dims: creativity_scope, number_of_few_shots.

Usage
─────
    python -m data.convert_v1_to_v15
    python -m data.convert_v1_to_v15 --in-dir data/processed --out-dir data/processed/v1.5
    python -m data.convert_v1_to_v15 --dry-run      # print stats, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_label_schema, task_type_id

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# v1 snake_case id → v2 display label
_TASK_MAP: dict[str, str] = {
    "open_qa":          "Knowledge Retrieval / QA",
    "closed_qa":        "Knowledge Retrieval / QA",
    "classification":   "Knowledge Retrieval / QA",
    "extraction":       "Knowledge Retrieval / QA",
    "summarization":    "Summarization",
    "text_generation":  "Content Creation",
    "rewrite":          "Content Creation",
    "brainstorming":    "Content Creation",
    "code_generation":  "Code",
    "chatbot":          "Conversation",
    "other":            "Conversation",
}


def _map_task_type(v1_label: str) -> str:
    """Convert a v1 task_type label to its v2 equivalent.

    Normalises through task_type_id so casing / spacing differences are handled.
    Raises ValueError for unknown v1 labels so bad data is caught at conversion
    time rather than silently passed through.
    """
    key = task_type_id(v1_label)
    if key not in _TASK_MAP:
        raise ValueError(f"Unknown v1 task_type: {v1_label!r} (id={key!r})")
    return _TASK_MAP[key]


def _map_complexity(row: dict) -> dict[str, float]:
    """Derive the 3 v2 complexity dims from a v1 row's 6 dims."""
    def _f(k: str) -> float:
        return float(row.get(k, 0.0))

    return {
        "reasoning_depth":      _f("reasoning"),
        "domain_knowledge":     max(_f("domain_knowledge"), _f("contextual_knowledge")),
        "instruction_precision": _f("constraint_ct"),
    }


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------

_V2_DIMS = ("reasoning_depth", "domain_knowledge", "instruction_precision")


def convert_row(row: dict) -> dict | None:
    """Convert one v1 row to v1.5.  Returns None and prints a warning on bad input."""
    try:
        v2_task = _map_task_type(row.get("task_type", ""))
    except ValueError as exc:
        print(f"  [WARN] skipping row: {exc}", file=sys.stderr)
        return None

    dims = _map_complexity(row)

    out = {
        "text":    row["text"],
        "task_type": v2_task,
        **dims,
    }
    # Preserve provenance fields if present
    for k in ("label_source", "hf_source", "source"):
        if k in row:
            out[k] = row[k]

    return out


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def convert_split(
    in_path: Path,
    out_path: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, int, Counter, Counter]:
    """Convert one split file.  Returns (total, written, before_counts, after_counts)."""
    rows_in = list(_read_jsonl(in_path))
    before: Counter = Counter(r.get("task_type", "MISSING") for r in rows_in)

    converted = [convert_row(r) for r in rows_in]
    good = [r for r in converted if r is not None]
    after: Counter = Counter(r["task_type"] for r in good)

    if not dry_run:
        n = _write_jsonl(good, out_path)
    else:
        n = len(good)

    return len(rows_in), n, before, after


# ---------------------------------------------------------------------------
# Stats printing
# ---------------------------------------------------------------------------


def _print_stats(split: str, total: int, written: int, before: Counter, after: Counter) -> None:
    skipped = total - written
    print(f"\n{'─'*60}")
    print(f"Split: {split}  ({total} rows → {written} written, {skipped} skipped)")

    print("\n  v1 task_type distribution (before):")
    for label, cnt in sorted(before.items(), key=lambda x: -x[1]):
        print(f"    {label:<30} {cnt:>6}")

    print("\n  v2 task_type distribution (after):")
    for label, cnt in sorted(after.items(), key=lambda x: -x[1]):
        pct = cnt / max(1, written) * 100
        print(f"    {label:<35} {cnt:>6}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert v1 dataset to v1.5 (v2 schema, rules-based)")
    ap.add_argument(
        "--in-dir", default=None,
        help="Directory containing v1 train/val/test.jsonl (default: data/processed)",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="Output directory for v1.5 splits (default: data/processed/v1.5)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print stats only, write nothing")
    args = ap.parse_args()

    # Resolve paths relative to repo root
    from config import DATA_DIR
    in_dir  = Path(args.in_dir)  if args.in_dir  else DATA_DIR / "processed"
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR / "processed" / "v1.5"

    # Validate v2 schema is available
    schema = load_label_schema(version="v2")
    v2_types = set(schema.task_types)
    assert v2_types == {
        "Knowledge Retrieval / QA", "Reasoning / Problem Solving",
        "Summarization", "Content Creation", "Code", "Conversation",
    }, f"Unexpected v2 task types: {v2_types}"

    if args.dry_run:
        print("[DRY RUN] no files will be written")

    splits = ["train", "val", "test"]
    grand_total = grand_written = 0

    for split in splits:
        in_path  = in_dir  / f"{split}.jsonl"
        out_path = out_dir / f"{split}.jsonl"

        if not in_path.exists():
            print(f"  [SKIP] {in_path} not found")
            continue

        total, written, before, after = convert_split(in_path, out_path, dry_run=args.dry_run)
        _print_stats(split, total, written, before, after)
        grand_total   += total
        grand_written += written

    print(f"\n{'═'*60}")
    print(f"Total: {grand_total} rows in → {grand_written} rows out")
    if args.dry_run:
        print("(dry run — nothing written)")
    else:
        print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
