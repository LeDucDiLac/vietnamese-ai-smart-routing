#!/usr/bin/env python3
"""Build a routing evaluation testset from production logs.

We cannot replay queries against all model tiers (no API access), so we use
two principles to assign oracle_tier labels from what the logs already contain:

  Monotonicity
    If tier X answered correctly → all more expensive tiers would too.
    If tier X failed → all cheaper tiers also fail.

  Complexity proxy
    For queries where only the large model ran but succeeded, we estimate
    whether a cheaper tier could have handled it using token-based complexity.

Oracle confidence levels
  confirmed       Cheaper model actually ran and succeeded — oracle ≤ actual tier.
                  Also: mid failed → large is confirmed necessary.
  estimated       Large ran and succeeded; cheaper oracle is estimated from
                  complexity proxy. Less reliable but covers most of the data.

Output
  data/eval/routing_testset.csv    — flat CSV, one row per query
  data/eval/routing_testset.jsonl  — same data as JSON lines

Each row
  prompt_id, prompt_text, actual_tier, actual_quality,
  oracle_tier, oracle_confidence,
  complexity_proxy, prompt_tokens, completion_tokens, latency_ms,
  task_fingerprint (hash of first 80 chars of system prompt)

Usage
  python scripts/build_routing_testset.py
  python scripts/build_routing_testset.py --csv data/eval/intern_data.csv --out data/eval
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

csv.field_size_limit(10 ** 7)

# ─────────────────────────────────────────────────────────────────────────────
# Model → tier
# ─────────────────────────────────────────────────────────────────────────────

TIER_MAP: dict[str, str] = {
    "openai/gpt-oss-120b":                        "large",
    "Qwen/Qwen3.5-122B-A10B-FP8":                 "large",
    "Qwen/Qwen3.5-35B-A3B-FP8":                   "mid",
    "Qwen/Qwen3.5-35B-A3B-FP8-Image":             "mid",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8":       "small",
}

TIER_ORDER = {"small": 0, "mid": 1, "large": 2}

# ─────────────────────────────────────────────────────────────────────────────
# Parse helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_messages(raw_request: str) -> tuple[str, str]:
    """Return (system_prompt, last_user_message) from a LiteLLM request string."""
    system = ""
    user   = ""
    try:
        req  = ast.literal_eval(raw_request)
        msgs = req.get("messages", [])
        for m in msgs:
            if m.get("role") == "system":
                c = m.get("content") or ""
                system = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict)
                )
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content") or ""
                user = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict)
                )
                break
    except Exception:
        pass
    return system, user


def _extract_content(raw_response: str) -> str:
    """Extract assistant content from a LiteLLM response string."""
    try:
        resp = ast.literal_eval(raw_response)
        return resp["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Quality assessment  (no ground-truth labels — uses response structure)
# ─────────────────────────────────────────────────────────────────────────────

_REFUSAL_VI = re.compile(
    r"(tôi không thể|xin lỗi,?\s+tôi|không thể (giúp|hỗ trợ|thực hiện)|"
    r"vượt quá khả năng)",
    re.IGNORECASE,
)
_REFUSAL_EN = re.compile(
    r"\b(i('m| am) (sorry|unable)|i cannot|i can'?t help|as an ai)\b",
    re.IGNORECASE,
)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", re.DOTALL)


def _is_valid_json(text: str) -> bool:
    """True if text (or the first JSON block in it) parses cleanly."""
    try:
        json.loads(text)
        return True
    except Exception:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            json.loads(m.group(1))
            return True
        except Exception:
            pass
    return False


def assess_quality(
    content: str,
    completion_tokens: int,
    system_prompt: str,
) -> int:
    """
    Return 1 (success) or 0 (failure) for a model response.

    Rules (in order):
    1. Empty / near-empty  → 0
    2. Refusal phrase       → 0
    3. System prompt expects JSON and content is NOT valid JSON → 0
    4. Everything else      → 1

    Note: {"index": N} with 7 tokens is intentional and valid for
    index-classification tasks — do not penalise short JSON outputs.
    """
    if completion_tokens == 0 or not content.strip():
        return 0

    if _REFUSAL_VI.search(content) or _REFUSAL_EN.search(content):
        return 0

    expects_json = (
        "json" in system_prompt.lower()
        or re.search(r"\{.*\}", system_prompt[:400])
        or "format_final_json_response" in system_prompt
    )
    if expects_json and not _is_valid_json(content):
        # Accept short index responses even if top-level JSON check fails
        if not re.fullmatch(r'\s*\{"index"\s*:\s*-?\d+\}\s*', content):
            return 0

    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Complexity proxy
# ─────────────────────────────────────────────────────────────────────────────

def complexity_proxy(prompt_tokens: int, completion_tokens: int) -> float:
    ct_norm = math.log1p(completion_tokens) / math.log1p(8_000)
    pt_norm = math.log1p(prompt_tokens)     / math.log1p(50_000)
    return min(1.0, 0.6 * ct_norm + 0.4 * pt_norm)


def _tier_from_complexity(score: float) -> str:
    if score < 0.35:
        return "small"
    if score < 0.65:
        return "mid"
    return "large"


# ─────────────────────────────────────────────────────────────────────────────
# Oracle assignment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OracleLabel:
    oracle_tier:        str             # "small" | "mid" | "large"
    oracle_confidence:  str             # "confirmed" | "estimated"
    include:            bool            # False → exclude from testset
    exclude_reason:     str = ""


def assign_oracle(
    actual_tier:    str,
    actual_quality: int,
    cpx:            float,
) -> OracleLabel:
    """
    Assign oracle_tier using monotonicity + complexity proxy.

    Cases
    ─────
    actual_tier=small, quality=1
        → oracle=small  (confirmed: cheapest model succeeded)

    actual_tier=small, quality=0
        → oracle unknown (could be mid or large) → exclude

    actual_tier=mid, quality=1
        → mid works; estimate if small also would:
          complexity < 0.35 → oracle=small (estimated)
          else              → oracle=mid   (confirmed lower bound)

    actual_tier=mid, quality=0
        → by monotonicity, large is necessary → oracle=large (confirmed)

    actual_tier=large, quality=1
        → estimate cheapest tier from complexity proxy (estimated)

    actual_tier=large, quality=0
        → task impossible at any tier → exclude
    """
    if actual_tier == "small":
        if actual_quality == 1:
            return OracleLabel("small", "confirmed", True)
        else:
            return OracleLabel("", "", False, "small failed — oracle unknown (mid or large)")

    if actual_tier == "mid":
        if actual_quality == 1:
            if cpx < 0.35:
                return OracleLabel("small", "estimated", True)
            return OracleLabel("mid", "confirmed", True)
        else:
            # mid failed → large required (monotonicity)
            return OracleLabel("large", "confirmed", True)

    if actual_tier == "large":
        if actual_quality == 0:
            return OracleLabel("", "", False, "large failed — task impossible")
        # quality=1 on large: estimate cheaper oracle
        oracle = _tier_from_complexity(cpx)
        return OracleLabel(oracle, "estimated", True)

    return OracleLabel("", "", False, f"unknown tier {actual_tier!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main build loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestRow:
    prompt_id:          str
    prompt_text:        str
    task_fingerprint:   str     # first-80-char hash of system prompt
    actual_model:       str
    actual_tier:        str
    actual_quality:     int
    oracle_tier:        str
    oracle_confidence:  str     # "confirmed" | "estimated"
    complexity_proxy:   float
    prompt_tokens:      int
    completion_tokens:  int
    latency_ms:         float


def build_testset(csv_path: Path) -> tuple[list[TestRow], dict]:
    rows: list[TestRow] = []
    stats = {
        "total_parsed":     0,
        "skipped_tier":     0,
        "skipped_quality":  0,
        "excluded":         0,
        "included":         0,
        "by_confidence":    {"confirmed": 0, "estimated": 0},
        "by_oracle_tier":   {"small": 0, "mid": 0, "large": 0},
        "by_actual_tier":   {"small": 0, "mid": 0, "large": 0},
        "quality_zero":     0,
    }

    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["call_type"] not in ("acompletion", "aresponses"):
                continue

            tier = TIER_MAP.get(row["model"])
            if tier is None:
                stats["skipped_tier"] += 1
                continue

            try:
                pt  = max(0, int(row["prompt_tokens"]  or 0))
                ct  = max(0, int(row["completion_tokens"] or 0))
                lat = float(row["request_duration_ms"]  or 0)
            except ValueError:
                continue

            stats["total_parsed"] += 1
            stats["by_actual_tier"][tier] = stats["by_actual_tier"].get(tier, 0) + 1

            system, user = _parse_messages(row["proxy_server_request"])
            content      = _extract_content(row["response"])
            quality      = assess_quality(content, ct, system)

            if quality == 0:
                stats["quality_zero"] += 1

            cpx    = complexity_proxy(pt, ct)
            oracle = assign_oracle(tier, quality, cpx)

            if not oracle.include:
                stats["excluded"] += 1
                continue

            task_fp = hashlib.md5(system[:80].encode()).hexdigest()[:8]

            rows.append(TestRow(
                prompt_id         = row["request_id"],
                prompt_text       = user,
                task_fingerprint  = task_fp,
                actual_model      = row["model"],
                actual_tier       = tier,
                actual_quality    = quality,
                oracle_tier       = oracle.oracle_tier,
                oracle_confidence = oracle.oracle_confidence,
                complexity_proxy  = round(cpx, 4),
                prompt_tokens     = pt,
                completion_tokens = ct,
                latency_ms        = lat,
            ))

            stats["included"] += 1
            stats["by_confidence"][oracle.oracle_confidence] = (
                stats["by_confidence"].get(oracle.oracle_confidence, 0) + 1
            )
            stats["by_oracle_tier"][oracle.oracle_tier] = (
                stats["by_oracle_tier"].get(oracle.oracle_tier, 0) + 1
            )

    return rows, stats


# ─────────────────────────────────────────────────────────────────────────────
# Output + report
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "prompt_id", "prompt_text", "task_fingerprint",
    "actual_model", "actual_tier", "actual_quality",
    "oracle_tier", "oracle_confidence",
    "complexity_proxy", "prompt_tokens", "completion_tokens", "latency_ms",
]


def write_outputs(rows: list[TestRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = out_dir / "routing_testset.csv"
    jsonl_path = out_dir / "routing_testset.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(f"  CSV   → {csv_path}")
    print(f"  JSONL → {jsonl_path}")


def print_report(rows: list[TestRow], stats: dict) -> None:
    n = len(rows)
    print("\n" + "═" * 64)
    print("  ROUTING TESTSET BUILD REPORT")
    print("═" * 64)
    print(f"  Parsed records      : {stats['total_parsed']:,}")
    print(f"  Skipped (no tier)   : {stats['skipped_tier']:,}")
    print(f"  Quality=0 records   : {stats['quality_zero']:,}")
    print(f"  Excluded            : {stats['excluded']:,}  (ambiguous oracle)")
    print(f"  Included in testset : {n:,}")
    print()

    print("  Actual tier distribution:")
    for t, c in sorted(stats["by_actual_tier"].items()):
        print(f"    {t:6s}  {c:5d}")

    print()
    print("  Oracle tier distribution:")
    for t in ("small", "mid", "large"):
        c = stats["by_oracle_tier"].get(t, 0)
        pct = c / n * 100 if n else 0
        print(f"    {t:6s}  {c:5d}  ({pct:.1f}%)")

    print()
    print("  Label confidence:")
    conf = stats["by_confidence"]
    for k in ("confirmed", "estimated"):
        c = conf.get(k, 0)
        pct = c / n * 100 if n else 0
        print(f"    {k:12s}  {c:5d}  ({pct:.1f}%)")

    print()
    print("  How to use this testset")
    print("  ───────────────────────")
    print("  1. Run your router on each `prompt_text`")
    print("     → get `predicted_tier` per row")
    print("  2. Compare predicted_tier vs oracle_tier:")
    print("     - routing_accuracy  = mean(predicted == oracle)")
    print("     - cost_error        = cost(predicted) - cost(oracle)  per row")
    print("     - quality_miss_rate = mean(predicted cheaper than oracle)")
    print("  3. Filter to oracle_confidence='confirmed' for high-trust evaluation")
    print("     Filter to oracle_confidence='estimated' for broader coverage")
    print()
    print("  Limitations")
    print("  ───────────")
    print("  - Estimated labels use token-count complexity proxy, not semantic")
    print("    understanding. The real classifier may disagree.")
    print("  - Quality is assessed from response structure (JSON validity,")
    print("    refusal detection) — not from task-specific correctness.")
    print("  - Mid/large queries where quality=0 are excluded if tier=small")
    print("    (oracle unknown — could be mid or large).")
    print("═" * 64 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/eval/intern_data.csv")
    ap.add_argument("--out", default="data/eval")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: {csv_path} not found")

    print(f"Building routing testset from {csv_path} …")
    rows, stats = build_testset(csv_path)

    write_outputs(rows, Path(args.out))
    print_report(rows, stats)


if __name__ == "__main__":
    main()
