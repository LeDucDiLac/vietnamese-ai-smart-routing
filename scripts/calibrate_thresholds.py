#!/usr/bin/env python3
"""Find optimal tier thresholds for a v2 checkpoint on the routing testset.

Runs inference on the routing testset, collects prompt_complexity_score for every row,
then grid-searches thresholds (t_small, t_mid) that maximise routing accuracy against
oracle_tier labels.

Usage
─────
  python scripts/calibrate_thresholds.py \\
      runs/teachers/vi-router-quality \\
      --schema-version v2 \\
      --testset data/eval/routing_testset.jsonl

  # Use a specific Python (e.g. on H200 without uv)
  .venv/bin/python scripts/calibrate_thresholds.py \\
      runs/teachers/vi-router-quality \\
      --schema-version v2

After running, paste the printed tier_thresholds line into
configs/schemas/v2-complexity.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))


def load_testset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_scores(clf, rows: list[dict], batch_size: int = 64) -> list[float]:
    prompts = [r.get("prompt_text", "") or "" for r in rows]
    scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        preds = clf.predict(batch)
        scores.extend(p["prompt_complexity_score"] for p in preds)
    return scores


def grid_search(
    scores: list[float],
    oracle_tiers: list[str],
    steps: int = 200,
) -> tuple[float, float, float]:
    """Return (t_small, t_mid, best_accuracy)."""
    lo, hi = min(scores), max(scores)
    span = hi - lo
    step = span / steps

    best_acc = -1.0
    best_t1 = 0.35
    best_t2 = 0.65

    candidates = [lo + i * step for i in range(steps + 1)]

    for t1 in candidates:
        for t2 in candidates:
            if t2 <= t1:
                continue
            correct = 0
            for s, oracle in zip(scores, oracle_tiers):
                if s < t1:
                    pred = "small"
                elif s < t2:
                    pred = "mid"
                else:
                    pred = "large"
                if pred == oracle:
                    correct += 1
            acc = correct / len(scores)
            if acc > best_acc:
                best_acc = acc
                best_t1 = t1
                best_t2 = t2

    return best_t1, best_t2, best_acc


def print_distribution(label: str, tiers: list[str]) -> None:
    n = len(tiers)
    for t in ("small", "mid", "large"):
        ct = tiers.count(t)
        print(f"  {t:5s}: {ct:5d} ({ct/n*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate tier thresholds for a v2 checkpoint")
    ap.add_argument("checkpoint", help="Path to checkpoint directory")
    ap.add_argument("--testset", default="data/eval/routing_testset.jsonl")
    ap.add_argument("--schema-version", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200,
                    help="Grid resolution per axis (default: 200 → 200×200 = 40k combos)")
    ap.add_argument("--confirmed-only", action="store_true",
                    help="Use only oracle_confidence=confirmed rows")
    args = ap.parse_args()

    testset_path = Path(args.testset)
    if not testset_path.exists():
        sys.exit(f"Testset not found: {testset_path}")

    print(f"Loading testset: {testset_path}")
    rows = load_testset(testset_path)
    if args.confirmed_only:
        rows = [r for r in rows if r.get("oracle_confidence") == "confirmed"]
    oracle_tiers = [r["oracle_tier"] for r in rows]
    print(f"  {len(rows):,} rows  ({sum(1 for r in rows if r.get('oracle_confidence')=='confirmed'):,} confirmed)")

    print("\nOracle tier distribution:")
    print_distribution("oracle", oracle_tiers)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    from classifier.infer import TorchClassifier
    clf = TorchClassifier(
        args.checkpoint,
        model_size=json.loads((Path(args.checkpoint) / "meta.json").read_text()).get("model_name", "vi-router-quality"),
        schema_version=args.schema_version,
    )
    print(f"  Device: {clf.device}")

    print(f"\nRunning inference on {len(rows):,} prompts (batch={args.batch_size}) …")
    t0 = time.time()
    scores = collect_scores(clf, rows, batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    print(f"\nComplexity score distribution:")
    import statistics
    print(f"  min={min(scores):.4f}  max={max(scores):.4f}")
    print(f"  mean={statistics.mean(scores):.4f}  median={statistics.median(scores):.4f}")
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    for pct in (1, 10, 25, 50, 75, 90, 99):
        idx = min(int(pct / 100 * n), n - 1)
        print(f"  p{pct:02d}: {sorted_scores[idx]:.4f}")

    print(f"\nGrid-searching thresholds ({args.steps}×{args.steps} grid) …")
    t_small, t_mid, best_acc = grid_search(scores, oracle_tiers, steps=args.steps)
    print(f"  Best routing accuracy: {best_acc*100:.1f}%")
    print(f"  Optimal thresholds: small < {t_small:.4f}, mid < {t_mid:.4f}")

    # Show predicted distribution with optimal thresholds
    predicted = []
    for s in scores:
        if s < t_small:
            predicted.append("small")
        elif s < t_mid:
            predicted.append("mid")
        else:
            predicted.append("large")

    print(f"\nPredicted distribution with optimal thresholds:")
    print_distribution("predicted", predicted)

    print(f"\nFor comparison — current v1 thresholds (0.35 / 0.65):")
    old_pred = ["small" if s < 0.35 else ("mid" if s < 0.65 else "large") for s in scores]
    old_acc = sum(p == o for p, o in zip(old_pred, oracle_tiers)) / len(oracle_tiers)
    print_distribution("v1-thresh", old_pred)
    print(f"  Accuracy with v1 thresholds: {old_acc*100:.1f}%")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paste the following into configs/schemas/v2-complexity.yaml:

tier_thresholds: [{t_small:.4f}, {t_mid:.4f}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
