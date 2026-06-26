#!/usr/bin/env python3
"""Evaluate routing quality against Viettel AI Gateway production logs.

Input : data/eval/intern_data.csv  (LiteLLM proxy logs, 5k records)
Output: console report  +  optional JSON artifact (--out <dir>)

──────────────────────────────────────────────────────────────────────────────
Metrics reported
──────────────────────────────────────────────────────────────────────────────
Industrial KPIs  (problem statement §3)
  router_latency_ms           Classifier inference time           target ≤ 50 ms
  cost_saving_pct             vs always-best model                target ≥ 30 %
  latency_reduction_pct       for queries routed to cheaper tier  target ≥ 20 %
  quality_loss_pct            proxy skill degradation             target ≤  3 %

Research metrics  (LLMRouterBench / RouterBench / RouteLLM / RouterXBench)
  avg_acc         AvgAcc — mean quality of routed model per query
  gain_over_best  Gain@B — quality delta vs always-best-model
  gap_to_oracle   Gap@O  — headroom vs proxy oracle  (lower = better)
  cost_save       CostSave = 1 − cost_routed / cost_best
  pgr             Performance Gap Recovered (binary small ↔ large split)
  cpt_50/cpt_80   Call-Performance Threshold at 50 % / 80 % PGR
  aiq             Area under cost–quality curve  (RouterBench AIQ)
  lpm             Low-band Performance Mean   (RouterXBench)
  hcr             High-band Call Rate          (RouterXBench)
  mpm             Mid-band Performance Mean    (RouterXBench)

Quality note: no ground-truth quality labels exist in the logs.
  Quality is proxied by the model-tier skill scores in configs/sim_models.yaml.
  Run with --model-path to activate the real vi-router classifier.

Usage
─────
  python scripts/eval_logs.py
  python scripts/eval_logs.py --model-path runs/student/best_model --out runs/eval
  python scripts/eval_logs.py --no-router   # baseline + oracle only
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Optional

csv.field_size_limit(10 ** 7)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Real-model cost / quality profiles
#     (approximate; calibrated to sim_models.yaml tier anchors)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    tier: str               # "small" | "mid" | "large"
    cost_per_1k: float      # USD per 1 k *output* tokens (input billed at 0.5×)
    latency_ms_per_1k: float
    quality: float          # mean skill [0, 1] across all task types


# Cost/latency scaled to sim_models.yaml tier anchors.
# Quality matches sim_models.yaml overall mean-skill per tier.
REAL_MODELS: dict[str, Optional[ModelSpec]] = {
    "openai/gpt-oss-120b":                        ModelSpec("large", 0.020, 1800, 0.93),
    "Qwen/Qwen3.5-122B-A10B-FP8":                 ModelSpec("large", 0.015, 1600, 0.90),
    "Qwen/Qwen3.5-35B-A3B-FP8":                   ModelSpec("mid",   0.003,  700, 0.83),
    "Qwen/Qwen3.5-35B-A3B-FP8-Image":             ModelSpec("mid",   0.003,  700, 0.83),
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8":       ModelSpec("small", 0.0006, 320, 0.74),
    # Not routing targets — skip
    "BAAI/bge-m3":                                None,
    "deepseek-ai/DeepSeek-OCR-2":                 None,
}

# Default model to deploy when a tier is selected
TIER_MODELS: dict[str, str] = {
    "small": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "mid":   "Qwen/Qwen3.5-35B-A3B-FP8",
    "large": "openai/gpt-oss-120b",
}
TIERS = ["small", "mid", "large"]

def spec_for_tier(tier: str) -> ModelSpec:
    return REAL_MODELS[TIER_MODELS[tier]]  # type: ignore[return-value]

BEST_SPEC  = spec_for_tier("large")
SMALL_SPEC = spec_for_tier("small")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Log parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LogRecord:
    request_id:       str
    model:            str
    prompt_tokens:    int
    completion_tokens: int
    latency_ms:       float
    prompt_text:      str       # last user message
    spec:             ModelSpec


def _extract_user_message(raw: str) -> str:
    """Pull the last user-role message from a LiteLLM proxy_server_request string."""
    try:
        req = ast.literal_eval(raw)
        msgs = req.get("messages", [])
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
        # fall back to system prompt
        for msg in msgs:
            if msg.get("role") == "system":
                return (msg.get("content") or "")[:500]
    except Exception:
        pass
    return ""


def parse_logs(csv_path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["call_type"] not in ("acompletion", "aresponses"):
                continue
            spec = REAL_MODELS.get(row["model"])
            if spec is None:
                continue
            try:
                pt  = max(0, int(row["prompt_tokens"] or 0))
                ct  = max(0, int(row["completion_tokens"] or 0))
                lat = float(row["request_duration_ms"] or 0)
            except ValueError:
                continue
            if ct == 0 and lat == 0:
                continue  # failed / empty calls
            records.append(LogRecord(
                request_id=row["request_id"],
                model=row["model"],
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_ms=lat,
                prompt_text=_extract_user_message(row["proxy_server_request"]),
                spec=spec,
            ))
    return records

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-record helpers
# ─────────────────────────────────────────────────────────────────────────────

def token_cost(r: LogRecord, spec: ModelSpec) -> float:
    """Estimated API cost: input tokens billed at 0.5× output price."""
    return (r.prompt_tokens * spec.cost_per_1k * 0.5
            + r.completion_tokens * spec.cost_per_1k) / 1000.0


def estimated_latency(r: LogRecord, spec: ModelSpec) -> float:
    """Model latency proxy: completion_tokens × latency_per_token."""
    return r.completion_tokens * spec.latency_ms_per_1k / 1000.0


def complexity_proxy(r: LogRecord) -> float:
    """
    Token-count heuristic complexity score in [0, 1].

    Long completions signal deep reasoning; large contexts signal rich inputs.
    Used as oracle stand-in when the real classifier is not loaded.
    """
    ct_norm = math.log1p(r.completion_tokens) / math.log1p(8_000)
    pt_norm = math.log1p(r.prompt_tokens)     / math.log1p(50_000)
    return min(1.0, 0.6 * ct_norm + 0.4 * pt_norm)


def tier_for_complexity(score: float) -> str:
    if score < 0.35:
        return "small"
    if score < 0.65:
        return "mid"
    return "large"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Routing scenarios
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name:    str
    tiers:   list[str]  # one tier per record, same order as records


def baseline_scenario(records: list[LogRecord]) -> Scenario:
    """Actual system: use whichever model was called."""
    return Scenario("baseline (actual)", [r.spec.tier for r in records])


def always_best_scenario(records: list[LogRecord]) -> Scenario:
    return Scenario("always-best", ["large"] * len(records))


def oracle_scenario(records: list[LogRecord]) -> Scenario:
    """Proxy oracle: route by token-complexity heuristic."""
    return Scenario("oracle (proxy)", [tier_for_complexity(complexity_proxy(r)) for r in records])


def heuristic_scenario(records: list[LogRecord]) -> Scenario:
    """Same as oracle — separates conceptually when vi-router is also present."""
    return Scenario("heuristic router", [tier_for_complexity(complexity_proxy(r)) for r in records])

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Metric computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricReport:
    scenario: str
    n: int

    # ── Industrial KPIs ──────────────────────────────────────
    router_latency_ms:          float   # per-query inference time (0 if no model)
    cost_saving_pct:            float   # vs always-best
    latency_reduction_pct:      float   # for queries downgraded from large
    quality_loss_pct:           float   # vs always-best (positive = worse)

    # ── Cost ─────────────────────────────────────────────────
    total_cost_usd:             float
    cost_per_request_usd:       float
    cost_save:                  float   # CostSave ∈ [0,1]

    # ── Latency ──────────────────────────────────────────────
    latency_p50_ms:             float
    latency_p95_ms:             float

    # ── Research: quality / oracle gap ───────────────────────
    avg_acc:                    float   # AvgAcc
    gain_over_best:             float   # Gain@B  (negative = quality sacrifice)
    gap_to_oracle:              float   # Gap@O   (lower = better)

    # ── Research: cost-quality trade-off ─────────────────────
    pgr:                        float   # Performance Gap Recovered
    cpt_50:                     float   # % strong calls for PGR ≥ 50 %
    cpt_80:                     float   # % strong calls for PGR ≥ 80 %
    aiq:                        float   # Area under cost-quality curve

    # ── Research: scenario-specific (RouterXBench) ────────────
    lpm:                        float   # Low-band Performance Mean
    hcr:                        float   # High-band Call Rate
    mpm:                        float   # Mid-band Performance Mean

    # ── Tier distribution ─────────────────────────────────────
    tier_dist:                  dict    # {"small": n, "mid": n, "large": n}


def _percentile(sorted_vals: list[float], p: float) -> float:
    idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _aiq_sweep(records: list[LogRecord], oracle: Scenario) -> float:
    """
    RouterBench AIQ: area under the cost–quality curve.

    x-axis  = fraction of traffic sent to 'large' model  (0 → 1)
    y-axis  = mean quality at that operating point
    Oracle tiers determine which queries go to large vs small/mid.
    We sweep by adjusting the fraction threshold.
    """
    n = len(records)
    complexity_order = sorted(range(n), key=lambda i: complexity_proxy(records[i]), reverse=True)

    points: list[tuple[float, float]] = []
    for frac_pct in range(0, 101, 5):
        frac = frac_pct / 100.0
        n_large = int(frac * n)
        per_q = ["small"] * n
        for i in complexity_order[:n_large]:
            per_q[i] = "large"
        q = mean(spec_for_tier(t).quality for t in per_q)
        points.append((frac, q))

    # trapezoidal integration over [0, 1]
    aiq = sum(
        0.5 * (points[i][1] + points[i - 1][1]) * (points[i][0] - points[i - 1][0])
        for i in range(1, len(points))
    )
    return aiq


def compute_metrics(
    records:       list[LogRecord],
    scenario:      Scenario,
    oracle:        Scenario,
    router_lat_ms: float = 0.0,
) -> MetricReport:
    n = len(records)

    routed_specs  = [spec_for_tier(t) for t in scenario.tiers]
    oracle_specs  = [spec_for_tier(t) for t in oracle.tiers]

    # ── Cost ─────────────────────────────────────────────────────────────────
    costs_routed = [token_cost(r, s) for r, s in zip(records, routed_specs)]
    costs_best   = [token_cost(r, BEST_SPEC) for r in records]

    total_cost  = sum(costs_routed)
    total_best  = sum(costs_best)
    cost_save   = 1.0 - total_cost / total_best if total_best else 0.0
    cost_saving_pct = cost_save * 100.0

    # ── Latency ──────────────────────────────────────────────────────────────
    lats_routed = [estimated_latency(r, s) for r, s in zip(records, routed_specs)]
    lats_best   = [estimated_latency(r, BEST_SPEC) for r in records]
    sorted_lat  = sorted(lats_routed)
    p50 = _percentile(sorted_lat, 0.50)
    p95 = _percentile(sorted_lat, 0.95)

    # Latency reduction: queries the router sends to small/mid instead of large
    downgraded = [i for i, t in enumerate(scenario.tiers) if t != "large"]
    if downgraded:
        lat_red_pct = (
            1.0 - mean(lats_routed[i] for i in downgraded)
                / mean(lats_best[i]   for i in downgraded)
        ) * 100.0
    else:
        lat_red_pct = 0.0

    # ── Quality (proxy) ───────────────────────────────────────────────────────
    q_routed = [s.quality for s in routed_specs]
    q_oracle  = [s.quality for s in oracle_specs]

    avg_q      = mean(q_routed)
    avg_best   = BEST_SPEC.quality
    avg_oracle = mean(q_oracle)

    quality_loss_pct = (avg_best - avg_q) / avg_best * 100.0

    # ── Research: oracle gap ──────────────────────────────────────────────────
    avg_acc       = avg_q
    gain_over_best = (avg_q - avg_best) / avg_best * 100.0   # negative = quality sacrifice
    gap_to_oracle  = (1.0 - avg_q / avg_oracle) * 100.0 if avg_oracle else 0.0

    # ── Research: PGR / CPT ───────────────────────────────────────────────────
    q_weak = SMALL_SPEC.quality
    q_strong = BEST_SPEC.quality
    q_range = q_strong - q_weak
    pgr = (avg_q - q_weak) / q_range if q_range else 0.0

    # CPT(k%): minimum % of large calls needed to reach PGR ≥ k
    complexity_order = sorted(range(n), key=lambda i: complexity_proxy(records[i]), reverse=True)
    pgr_curve: list[tuple[float, float]] = []
    for frac_pct in range(0, 101, 1):
        frac   = frac_pct / 100.0
        n_lrg  = int(frac * n)
        tiers  = ["small"] * n
        for i in complexity_order[:n_lrg]:
            tiers[i] = "large"
        q = mean(spec_for_tier(t).quality for t in tiers)
        p = (q - q_weak) / q_range if q_range else 0.0
        pgr_curve.append((frac * 100.0, p))

    def find_cpt(target: float) -> float:
        for frac_pct, p in pgr_curve:
            if p >= target:
                return frac_pct
        return 100.0

    cpt_50 = find_cpt(0.50)
    cpt_80 = find_cpt(0.80)

    # ── AIQ ──────────────────────────────────────────────────────────────────
    aiq = _aiq_sweep(records, oracle)

    # ── Scenario-specific (RouterXBench) ─────────────────────────────────────
    # Sort record indices by their routed cost (ascending)
    sorted_by_cost = sorted(range(n), key=lambda i: costs_routed[i])
    n_band = max(1, int(0.20 * n))

    # LPM — bottom 20 % cost band: mean quality
    low_idx = sorted_by_cost[:n_band]
    lpm = mean(q_routed[i] for i in low_idx)

    # HCR — top 20 % cost band: fraction routed to large (complex queries handled correctly)
    high_idx = sorted_by_cost[-n_band:]
    hcr = sum(1 for i in high_idx if scenario.tiers[i] == "large") / n_band

    # MPM — middle 60 %: mean quality in the trade-off band
    mid_idx = sorted_by_cost[n_band:-n_band] if n_band * 2 < n else sorted_by_cost
    mpm = mean(q_routed[i] for i in mid_idx)

    # ── Tier distribution ─────────────────────────────────────────────────────
    tier_dist: dict[str, int] = {t: 0 for t in TIERS}
    for t in scenario.tiers:
        tier_dist[t] = tier_dist.get(t, 0) + 1

    return MetricReport(
        scenario=scenario.name,
        n=n,
        router_latency_ms=router_lat_ms,
        cost_saving_pct=cost_saving_pct,
        latency_reduction_pct=lat_red_pct,
        quality_loss_pct=quality_loss_pct,
        total_cost_usd=total_cost,
        cost_per_request_usd=total_cost / n if n else 0.0,
        cost_save=cost_save,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        avg_acc=avg_q,
        gain_over_best=gain_over_best,
        gap_to_oracle=gap_to_oracle,
        pgr=pgr,
        cpt_50=cpt_50,
        cpt_80=cpt_80,
        aiq=aiq,
        lpm=lpm,
        hcr=hcr,
        mpm=mpm,
        tier_dist=tier_dist,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Optional: real vi-router classifier
# ─────────────────────────────────────────────────────────────────────────────

def vi_router_scenario(
    records: list[LogRecord], model_path: str
) -> tuple[Optional[Scenario], float]:
    """
    Load the trained classifier and score every prompt.
    Returns (scenario, mean_ms_per_query) or (None, 0) if unavailable.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from classifier.infer import TorchClassifier
        clf = TorchClassifier(model_path)
    except Exception as exc:
        print(f"[warn] Could not load vi-router from {model_path!r}: {exc}", file=sys.stderr)
        return None, 0.0

    prompts = [r.prompt_text or "" for r in records]
    t0 = time.perf_counter()
    preds = clf.predict(prompts)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mean_ms = elapsed_ms / len(records) if records else 0.0

    tiers = [tier_for_complexity(p["prompt_complexity_score"]) for p in preds]
    return Scenario("vi-router", tiers), mean_ms

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Report rendering
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def _kpi(value: float, target: float, higher_better: bool, unit: str = "%") -> str:
    ok  = value >= target if higher_better else value <= target
    sym = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    op  = "≥" if higher_better else "≤"
    return f"{value:7.1f}{unit}  {sym}  (target {op}{target}{unit})"


def print_report(
    records:       list[LogRecord],
    reports:       list[MetricReport],
    oracle_report: MetricReport,
) -> None:
    n = len(records)
    # actual tier distribution
    actual_dist: dict[str, int] = {t: 0 for t in TIERS}
    for r in records:
        actual_dist[r.spec.tier] += 1

    print(f"\n{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD}  AI SMART ROUTING — LOG-BASED EVALUATION{RESET}")
    print(f"{BOLD}{'═' * 72}{RESET}")
    print(f"  Records  : {n:,}  (acompletion + aresponses, non-empty)")
    print(f"  Models   : {len(set(r.model for r in records))}")
    print(f"  Current routing distribution:")
    for t in TIERS:
        c = actual_dist[t]
        print(f"    {t:6s}  {c:4d}  ({c/n*100:.1f}%)")

    print(f"\n  {BOLD}Oracle ceiling (proxy){RESET}")
    print(f"    cost saving vs best : {oracle_report.cost_saving_pct:.1f}%")
    print(f"    quality loss        : {abs(oracle_report.quality_loss_pct):.2f}%")
    print(f"    tier dist           : " + "  ".join(
        f"{t}:{oracle_report.tier_dist.get(t,0)}" for t in TIERS))

    for rpt in reports:
        print(f"\n{'─' * 72}")
        print(f"  {BOLD}Scenario: {rpt.scenario.upper()}{RESET}")
        print(f"  Tier distribution: " + "  ".join(
            f"{t}:{rpt.tier_dist.get(t,0)}" for t in TIERS))

        print(f"\n  {BOLD}Industrial KPIs  (problem statement §3){RESET}")
        if rpt.router_latency_ms > 0:
            print(f"  router latency  : {_kpi(rpt.router_latency_ms, 50, higher_better=False, unit=' ms')}")
        else:
            print(f"  router latency  : — (not measured)")
        print(f"  cost saving     : {_kpi(rpt.cost_saving_pct, 30, higher_better=True)}")
        print(f"  latency reduc.  : {_kpi(rpt.latency_reduction_pct, 20, higher_better=True)}")
        print(f"  quality loss    : {_kpi(abs(rpt.quality_loss_pct), 3, higher_better=False)}")

        print(f"\n  {BOLD}Cost{RESET}")
        print(f"  total cost      : ${rpt.total_cost_usd:.4f}")
        print(f"  per request     : ${rpt.cost_per_request_usd:.6f}")
        print(f"  CostSave        : {rpt.cost_save:.4f}  ({rpt.cost_saving_pct:.1f}% cheaper than always-best)")

        print(f"\n  {BOLD}Latency (estimated from completion tokens){RESET}")
        print(f"  P50             : {rpt.latency_p50_ms:,.0f} ms")
        print(f"  P95             : {rpt.latency_p95_ms:,.0f} ms")
        print(f"  reduction (↓)   : {rpt.latency_reduction_pct:.1f}%  for simple queries")

        print(f"\n  {BOLD}Research: Quality / Oracle gap{RESET}")
        print(f"  AvgAcc          : {rpt.avg_acc:.4f}")
        print(f"  Gain@B          : {rpt.gain_over_best:+.2f}%   (vs always-best; negative = quality tradeoff)")
        print(f"  Gap@O           : {rpt.gap_to_oracle:.2f}%   headroom vs proxy oracle (lower = better)")

        print(f"\n  {BOLD}Research: Cost–quality trade-off{RESET}")
        print(f"  PGR             : {rpt.pgr:.4f}   (fraction of small→large quality gap recovered)")
        print(f"  CPT(50%)        : {rpt.cpt_50:.1f}%   of traffic to large model for PGR ≥ 0.50")
        print(f"  CPT(80%)        : {rpt.cpt_80:.1f}%   of traffic to large model for PGR ≥ 0.80")
        print(f"  AIQ             : {rpt.aiq:.4f}   (area under cost–quality curve)")

        print(f"\n  {BOLD}Research: Scenario-specific (RouterXBench){RESET}")
        print(f"  LPM             : {rpt.lpm:.4f}   low-cost band quality  (bottom 20%)")
        print(f"  HCR             : {rpt.hcr:.4f}   high-cost band large-model rate  (top 20%)")
        print(f"  MPM             : {rpt.mpm:.4f}   mid-band quality  (middle 60%)")

    print(f"\n{'═' * 72}")
    print(f"  NOTE: Quality is proxied by tier skill scores from sim_models.yaml.")
    print(f"  For ground-truth evaluation, add quality labels to the log dataset")
    print(f"  or run with --model-path to enable the real vi-router classifier.")
    print(f"{'═' * 72}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Routing evaluation from intern_data.csv logs")
    ap.add_argument("--csv",        default="data/eval/intern_data.csv")
    ap.add_argument("--model-path", default=None,
                    help="Path to trained vi-router checkpoint for real classifier scoring")
    ap.add_argument("--out",        default=None,
                    help="Directory to write eval_logs.json report")
    ap.add_argument("--no-router",  action="store_true",
                    help="Skip vi-router inference (baseline + oracle only)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found")

    print(f"Loading logs from {csv_path} …")
    records = parse_logs(csv_path)
    if not records:
        sys.exit("No valid records found in CSV.")
    print(f"Parsed {len(records):,} valid records.")

    baseline  = baseline_scenario(records)
    oracle    = oracle_scenario(records)
    heuristic = heuristic_scenario(records)

    router_lat_ms = 0.0
    scenarios = [baseline, heuristic]

    if args.model_path and not args.no_router:
        vi_scen, router_lat_ms = vi_router_scenario(records, args.model_path)
        if vi_scen:
            scenarios.append(vi_scen)

    oracle_report = compute_metrics(records, oracle, oracle)
    reports = [
        compute_metrics(records, s, oracle, router_lat_ms if s.name == "vi-router" else 0.0)
        for s in scenarios
    ]

    print_report(records, reports, oracle_report)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_records":     len(records),
            "oracle":        asdict(oracle_report),
            "scenarios":     [asdict(r) for r in reports],
        }
        out_file = out_dir / "eval_logs.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"JSON report → {out_file}")


if __name__ == "__main__":
    main()
