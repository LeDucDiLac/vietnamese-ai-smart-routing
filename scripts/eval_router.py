#!/usr/bin/env python3
"""Evaluate trained router models against the routing testset.

Loads one or more trained checkpoints (PyTorch or ONNX), runs them over
data/eval/routing_testset.jsonl, and reports all routing metrics.

The testset was built by build_routing_testset.py from production logs.
Each row has an oracle_tier label (confirmed or estimated) so Gap@O here
is real — not a proxy — compared to the version in eval_logs.py.

Checkpoint detection
  PyTorch : directory containing model.pt + meta.json + tokenizer/
  ONNX    : path ending in .onnx  (pass --backbone and --max-tokens too)

Usage
─────
  # Evaluate one checkpoint
  python scripts/eval_router.py runs/quality

  # Compare teacher vs student side by side
  python scripts/eval_router.py runs/quality runs/student

  # ONNX export
  python scripts/eval_router.py runs/student/model.onnx \\
      --backbone microsoft/Multilingual-MiniLM-L12-H384

  # Write JSON report
  python scripts/eval_router.py runs/quality --out runs/eval_router

  # Use a custom testset
  python scripts/eval_router.py runs/quality \\
      --testset data/eval/routing_testset.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Cost / quality profiles  (same anchors as eval_logs.py)
# ─────────────────────────────────────────────────────────────────────────────

TIER_COST: dict[str, float] = {   # USD per 1k output tokens
    "small": 0.0006,
    "mid":   0.003,
    "large": 0.020,
}
TIER_QUALITY: dict[str, float] = {
    "small": 0.74,
    "mid":   0.83,
    "large": 0.93,
}
TIER_LATENCY_MS_PER_1K: dict[str, float] = {
    "small": 320,
    "mid":   700,
    "large": 1800,
}
TIERS = ["small", "mid", "large"]
TIER_ORDER = {t: i for i, t in enumerate(TIERS)}


def _token_cost(prompt_tokens: int, completion_tokens: int, tier: str) -> float:
    cpt = TIER_COST[tier]
    return (prompt_tokens * cpt * 0.5 + completion_tokens * cpt) / 1000.0


def _estimated_latency(completion_tokens: int, tier: str) -> float:
    return completion_tokens * TIER_LATENCY_MS_PER_1K[tier] / 1000.0


def _tier_from_complexity(score: float, thresholds: tuple[float, float] = (0.35, 0.65)) -> str:
    if score < thresholds[0]:
        return "small"
    if score < thresholds[1]:
        return "mid"
    return "large"


# ─────────────────────────────────────────────────────────────────────────────
# Testset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestRow:
    prompt_id:          str
    prompt_text:        str
    oracle_tier:        str
    oracle_confidence:  str     # "confirmed" | "estimated"
    actual_tier:        str
    prompt_tokens:      int
    completion_tokens:  int
    complexity_proxy:   float


def load_testset(path: Path) -> list[TestRow]:
    rows: list[TestRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(TestRow(
                prompt_id         = d["prompt_id"],
                prompt_text       = d["prompt_text"],
                oracle_tier       = d["oracle_tier"],
                oracle_confidence = d["oracle_confidence"],
                actual_tier       = d["actual_tier"],
                prompt_tokens     = int(d["prompt_tokens"]),
                completion_tokens = int(d["completion_tokens"]),
                complexity_proxy  = float(d["complexity_proxy"]),
            ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_meta(ckpt_dir: Path) -> dict[str, Any]:
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        return {"model_name": "vi-router-quality"}
    return json.loads(meta_path.read_text())


def load_classifier(model_path: str, backbone: str | None, max_tokens: int,
                    schema_version: str | None = None):
    """Return (classifier, label, meta_dict)."""
    p = Path(model_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    if str(model_path).endswith(".onnx"):
        from classifier.infer import OnnxClassifier
        if not backbone:
            sys.exit("--backbone is required for ONNX models")
        clf = OnnxClassifier(p, backbone=backbone, max_tokens=max_tokens)
        return clf, p.name, {"model_name": p.stem, "backbone": backbone}

    # PyTorch checkpoint directory
    meta = _load_meta(p)
    model_name = meta.get("model_name", "vi-router-quality")
    from classifier.infer import TorchClassifier
    clf = TorchClassifier(p, model_size=model_name, schema_version=schema_version)
    return clf, p.name, meta


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    clf,
    rows: list[TestRow],
    batch_size: int = 32,
    tier_thresholds: tuple[float, float] = (0.35, 0.65),
) -> tuple[list[str], float]:
    """
    Return (predicted_tiers, mean_ms_per_query).

    Runs in batches; times only the model forward pass.
    """
    prompts = [r.prompt_text or "" for r in rows]
    predictions: list[str] = []
    total_ms = 0.0

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        t0 = time.perf_counter()
        preds = clf.predict(batch)
        elapsed = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed
        for p in preds:
            predictions.append(_tier_from_complexity(p["prompt_complexity_score"], tier_thresholds))

    mean_ms = total_ms / len(rows) if rows else 0.0
    return predictions, mean_ms


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    label:              str     # checkpoint name
    model_name:         str
    n_total:            int
    n_confirmed:        int

    # ── Industrial KPIs ──────────────────────────────────────────────────────
    router_latency_ms:      float   # per-query  (target ≤ 50 ms)
    cost_saving_pct:        float   # vs always-large  (target ≥ 30 %)
    latency_reduction_pct:  float   # for queries sent below large  (target ≥ 20 %)
    quality_loss_pct:       float   # vs always-large  (target ≤ 3 %)

    # ── Routing accuracy ─────────────────────────────────────────────────────
    routing_acc_all:        float   # exact tier match, all rows
    routing_acc_confirmed:  float   # exact tier match, confirmed rows only
    underrouting_rate:      float   # predicted cheaper than oracle (quality risk)
    overrouting_rate:       float   # predicted more expensive (cost waste)

    # ── Research: quality / oracle gap ───────────────────────────────────────
    avg_acc:            float   # AvgAcc — mean quality of predicted tier
    gain_over_best:     float   # Gain@B vs always-large
    gap_to_oracle:      float   # Gap@O  (real, not proxy)
    cost_save:          float   # CostSave

    # ── Research: cost-quality trade-off ─────────────────────────────────────
    pgr:        float
    cpt_50:     float
    cpt_80:     float
    aiq:        float

    # ── Research: scenario-specific ──────────────────────────────────────────
    lpm:    float
    hcr:    float
    mpm:    float

    # ── Tier distribution ─────────────────────────────────────────────────────
    predicted_dist: dict


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(int(p * len(s)), len(s) - 1)]


def _aiq_from_tiers(
    rows: list[TestRow],
    predicted: list[str],
) -> float:
    """Sweep fraction-to-large from 0→1; integrate quality vs cost fraction."""
    n = len(rows)
    # Sort by complexity proxy descending (most complex → first to go to large)
    order = sorted(range(n), key=lambda i: rows[i].complexity_proxy, reverse=True)
    points: list[tuple[float, float]] = []
    for frac_pct in range(0, 101, 5):
        frac = frac_pct / 100.0
        n_large = int(frac * n)
        tiers = ["small"] * n
        for i in order[:n_large]:
            tiers[i] = "large"
        q = mean(TIER_QUALITY[t] for t in tiers)
        points.append((frac, q))
    return sum(
        0.5 * (points[i][1] + points[i - 1][1]) * (points[i][0] - points[i - 1][0])
        for i in range(1, len(points))
    )


def compute_eval(
    rows:       list[TestRow],
    predicted:  list[str],
    label:      str,
    model_name: str,
    latency_ms: float,
) -> EvalResult:
    n = len(rows)
    assert len(predicted) == n

    confirmed_idx = [i for i, r in enumerate(rows) if r.oracle_confidence == "confirmed"]

    # ── Per-row derived values ─────────────────────────────────────────────
    costs_pred    = [_token_cost(r.prompt_tokens, r.completion_tokens, predicted[i]) for i, r in enumerate(rows)]
    costs_large   = [_token_cost(r.prompt_tokens, r.completion_tokens, "large")      for r in rows]
    costs_oracle  = [_token_cost(r.prompt_tokens, r.completion_tokens, r.oracle_tier) for r in rows]
    lats_pred     = [_estimated_latency(r.completion_tokens, predicted[i]) for i, r in enumerate(rows)]
    lats_large    = [_estimated_latency(r.completion_tokens, "large")      for r in rows]
    q_pred        = [TIER_QUALITY[predicted[i]]    for i in range(n)]
    q_oracle      = [TIER_QUALITY[r.oracle_tier]   for r in rows]
    q_large       = TIER_QUALITY["large"]
    q_small       = TIER_QUALITY["small"]

    # ── Cost ──────────────────────────────────────────────────────────────
    total_pred  = sum(costs_pred)
    total_large = sum(costs_large)
    cost_save   = 1.0 - total_pred / total_large if total_large else 0.0
    cost_saving_pct = cost_save * 100.0

    # ── Latency ───────────────────────────────────────────────────────────
    downgraded = [i for i, t in enumerate(predicted) if t != "large"]
    if downgraded:
        lat_red_pct = (
            1.0 - mean(lats_pred[i]  for i in downgraded)
                / mean(lats_large[i] for i in downgraded)
        ) * 100.0
    else:
        lat_red_pct = 0.0

    # ── Quality ───────────────────────────────────────────────────────────
    avg_q        = mean(q_pred)
    avg_oracle_q = mean(q_oracle)
    quality_loss_pct = (q_large - avg_q) / q_large * 100.0

    # ── Routing accuracy ──────────────────────────────────────────────────
    exact_all       = sum(1 for i in range(n) if predicted[i] == rows[i].oracle_tier)
    exact_confirmed = sum(1 for i in confirmed_idx if predicted[i] == rows[i].oracle_tier)
    under = sum(
        1 for i in range(n)
        if TIER_ORDER[predicted[i]] < TIER_ORDER[rows[i].oracle_tier]
    )
    over  = sum(
        1 for i in range(n)
        if TIER_ORDER[predicted[i]] > TIER_ORDER[rows[i].oracle_tier]
    )

    # ── Research: gap ─────────────────────────────────────────────────────
    gain_over_best = (avg_q - q_large) / q_large * 100.0
    gap_to_oracle  = (1.0 - avg_q / avg_oracle_q) * 100.0 if avg_oracle_q else 0.0

    # ── PGR / CPT ─────────────────────────────────────────────────────────
    q_range = q_large - q_small
    pgr = (avg_q - q_small) / q_range if q_range else 0.0

    order_by_cpx = sorted(range(n), key=lambda i: rows[i].complexity_proxy, reverse=True)
    pgr_curve: list[tuple[float, float]] = []
    for frac_pct in range(0, 101, 1):
        frac   = frac_pct / 100.0
        n_lrg  = int(frac * n)
        tiers  = ["small"] * n
        for i in order_by_cpx[:n_lrg]:
            tiers[i] = "large"
        q = mean(TIER_QUALITY[t] for t in tiers)
        p = (q - q_small) / q_range if q_range else 0.0
        pgr_curve.append((frac * 100.0, p))

    def _cpt(target: float) -> float:
        for frac_pct, p in pgr_curve:
            if p >= target:
                return frac_pct
        return 100.0

    # ── AIQ ───────────────────────────────────────────────────────────────
    aiq = _aiq_from_tiers(rows, predicted)

    # ── RouterXBench scenario metrics ─────────────────────────────────────
    sorted_by_cost = sorted(range(n), key=lambda i: costs_pred[i])
    n_band = max(1, int(0.20 * n))
    low_idx  = sorted_by_cost[:n_band]
    high_idx = sorted_by_cost[-n_band:]
    mid_idx  = sorted_by_cost[n_band:-n_band] if n_band * 2 < n else sorted_by_cost

    lpm = mean(q_pred[i] for i in low_idx)
    hcr = sum(1 for i in high_idx if predicted[i] == "large") / n_band
    mpm = mean(q_pred[i] for i in mid_idx)

    # ── Tier distribution ─────────────────────────────────────────────────
    dist: dict[str, int] = {t: 0 for t in TIERS}
    for t in predicted:
        dist[t] = dist.get(t, 0) + 1

    return EvalResult(
        label=label,
        model_name=model_name,
        n_total=n,
        n_confirmed=len(confirmed_idx),
        router_latency_ms=latency_ms,
        cost_saving_pct=cost_saving_pct,
        latency_reduction_pct=lat_red_pct,
        quality_loss_pct=quality_loss_pct,
        routing_acc_all=exact_all / n if n else 0.0,
        routing_acc_confirmed=exact_confirmed / len(confirmed_idx) if confirmed_idx else 0.0,
        underrouting_rate=under / n if n else 0.0,
        overrouting_rate=over  / n if n else 0.0,
        avg_acc=avg_q,
        gain_over_best=gain_over_best,
        gap_to_oracle=gap_to_oracle,
        cost_save=cost_save,
        pgr=pgr,
        cpt_50=_cpt(0.50),
        cpt_80=_cpt(0.80),
        aiq=aiq,
        lpm=lpm,
        hcr=hcr,
        mpm=mpm,
        predicted_dist=dist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def _kpi(val: float, target: float, higher_better: bool, unit: str = "%") -> str:
    ok  = val >= target if higher_better else val <= target
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    op  = "≥" if higher_better else "≤"
    return f"{val:7.2f}{unit}  {sym}  (target {op}{target}{unit})"


def print_report(results: list[EvalResult]) -> None:
    print(f"\n{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD}  ROUTER EVALUATION REPORT{RESET}")
    print(f"{BOLD}{'═' * 72}{RESET}")
    print(f"  Models evaluated : {len(results)}")
    if results:
        print(f"  Testset rows     : {results[0].n_total:,}  "
              f"({results[0].n_confirmed:,} confirmed  /  "
              f"{results[0].n_total - results[0].n_confirmed:,} estimated)")

    for res in results:
        print(f"\n{'─' * 72}")
        print(f"  {BOLD}{res.label}{RESET}  [{res.model_name}]")
        print(f"  Predicted tier distribution: " + "  ".join(
            f"{t}:{res.predicted_dist.get(t,0)}" for t in TIERS))

        print(f"\n  {BOLD}Industrial KPIs  (problem statement §3){RESET}")
        print(f"  router latency  : {_kpi(res.router_latency_ms, 50,  higher_better=False, unit=' ms')}")
        print(f"  cost saving     : {_kpi(res.cost_saving_pct,   30,  higher_better=True)}")
        print(f"  latency reduc.  : {_kpi(res.latency_reduction_pct, 20, higher_better=True)}")
        print(f"  quality loss    : {_kpi(res.quality_loss_pct,  3,   higher_better=False)}")

        print(f"\n  {BOLD}Routing accuracy  (vs testset oracle_tier){RESET}")
        print(f"  all rows        : {res.routing_acc_all * 100:.1f}%")
        print(f"  confirmed only  : {res.routing_acc_confirmed * 100:.1f}%"
              f"  {DIM}← higher trust{RESET}")
        print(f"  underrouting    : {res.underrouting_rate * 100:.1f}%"
              f"  {DIM}(predicted cheaper than oracle → quality risk){RESET}")
        print(f"  overrouting     : {res.overrouting_rate  * 100:.1f}%"
              f"  {DIM}(predicted more expensive → cost waste){RESET}")

        print(f"\n  {BOLD}Research: quality / oracle gap{RESET}")
        print(f"  AvgAcc          : {res.avg_acc:.4f}")
        print(f"  Gain@B          : {res.gain_over_best:+.2f}%  vs always-large")
        print(f"  Gap@O           : {res.gap_to_oracle:.2f}%   vs testset oracle  {DIM}(lower = better){RESET}")
        print(f"  CostSave        : {res.cost_save:.4f}  ({res.cost_saving_pct:.1f}%)")

        print(f"\n  {BOLD}Research: cost–quality trade-off{RESET}")
        print(f"  PGR             : {res.pgr:.4f}")
        print(f"  CPT(50%)        : {res.cpt_50:.1f}%  of traffic to large for PGR ≥ 0.50")
        print(f"  CPT(80%)        : {res.cpt_80:.1f}%  of traffic to large for PGR ≥ 0.80")
        print(f"  AIQ             : {res.aiq:.4f}")

        print(f"\n  {BOLD}Research: scenario-specific (RouterXBench){RESET}")
        print(f"  LPM             : {res.lpm:.4f}  low-cost band quality")
        print(f"  HCR             : {res.hcr:.4f}  high-cost band large-model rate")
        print(f"  MPM             : {res.mpm:.4f}  mid-band quality")

    # Side-by-side comparison if multiple models
    if len(results) > 1:
        print(f"\n{'─' * 72}")
        print(f"  {BOLD}COMPARISON{RESET}")
        _compare(results)

    print(f"\n{'═' * 72}\n")


def _compare(results: list[EvalResult]) -> None:
    metrics = [
        ("router_latency_ms",     "router latency (ms)",    False),
        ("cost_saving_pct",       "cost saving (%)",         True),
        ("quality_loss_pct",      "quality loss (%)",        False),
        ("routing_acc_all",       "routing acc (all)",       True),
        ("routing_acc_confirmed", "routing acc (confirmed)", True),
        ("underrouting_rate",     "underrouting rate",       False),
        ("gap_to_oracle",         "Gap@O (%)",               False),
        ("cost_save",             "CostSave",                True),
        ("pgr",                   "PGR",                     True),
        ("aiq",                   "AIQ",                     True),
    ]
    col_w = 18
    header = f"  {'Metric':<28}" + "".join(f"{r.label[:col_w]:>{col_w}}" for r in results)
    print(header)
    print("  " + "─" * (28 + col_w * len(results)))
    for attr, label, higher_better in metrics:
        vals = [getattr(r, attr) for r in results]
        best = max(vals) if higher_better else min(vals)
        row = f"  {label:<28}"
        for v in vals:
            mark = f"{GREEN}*{RESET}" if v == best else " "
            row += f"{v:>{col_w - 1}.3f}{mark}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate trained router checkpoints against the routing testset"
    )
    ap.add_argument(
        "models", nargs="+",
        help="Checkpoint directory (PyTorch) or .onnx file path"
    )
    ap.add_argument(
        "--testset", default="data/eval/routing_testset.jsonl",
        help="Path to routing testset JSONL (default: data/eval/routing_testset.jsonl)"
    )
    ap.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32)"
    )
    ap.add_argument(
        "--backbone", default=None,
        help="Backbone name, required for ONNX models"
    )
    ap.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens for ONNX tokenizer (default: 256)"
    )
    ap.add_argument(
        "--confirmed-only", action="store_true",
        help="Only evaluate on oracle_confidence='confirmed' rows"
    )
    ap.add_argument(
        "--out", default=None,
        help="Write JSON report to this directory"
    )
    ap.add_argument(
        "--schema-version", default=None,
        help="Label schema version (e.g. 'v2'). Overrides meta.json. Required when "
             "meta.json does not record the schema version."
    )
    args = ap.parse_args()

    testset_path = Path(args.testset)
    if not testset_path.exists():
        sys.exit(
            f"Testset not found: {testset_path}\n"
            "Run: python scripts/build_routing_testset.py"
        )

    print(f"Loading testset from {testset_path} …")
    rows = load_testset(testset_path)
    if args.confirmed_only:
        rows = [r for r in rows if r.oracle_confidence == "confirmed"]
        print(f"Filtered to confirmed rows: {len(rows):,}")
    else:
        print(f"Testset: {len(rows):,} rows  "
              f"({sum(1 for r in rows if r.oracle_confidence == 'confirmed'):,} confirmed)")

    if not rows:
        sys.exit("No rows to evaluate.")

    results: list[EvalResult] = []

    for model_path in args.models:
        print(f"\nLoading {model_path} …")
        try:
            clf, label, meta = load_classifier(model_path, args.backbone, args.max_tokens,
                                                schema_version=args.schema_version)
        except Exception as e:
            print(f"  [error] Could not load {model_path}: {e}", file=sys.stderr)
            continue

        thresholds = tuple(clf.complexity.tier_thresholds) if hasattr(clf, "complexity") else (0.35, 0.65)
        print(f"  Tier thresholds: small<{thresholds[0]}, mid<{thresholds[1]}, large≥{thresholds[1]}")
        print(f"  Running inference on {len(rows):,} prompts (batch={args.batch_size}) …")
        predicted, lat_ms = run_inference(clf, rows, batch_size=args.batch_size,
                                          tier_thresholds=thresholds)
        print(f"  Done — {lat_ms:.1f} ms / query")

        res = compute_eval(rows, predicted, label, meta.get("model_name", label), lat_ms)
        results.append(res)

    if not results:
        sys.exit("No models successfully evaluated.")

    print_report(results)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "testset":  str(testset_path),
            "n_rows":   len(rows),
            "results":  [asdict(r) for r in results],
        }
        out_file = out_dir / "eval_router.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"JSON report → {out_file}")


if __name__ == "__main__":
    main()
