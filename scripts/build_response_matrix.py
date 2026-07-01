#!/usr/bin/env python3
"""build_response_matrix.py — multi-model replay → *measured* routing oracle.

The routing testset from build_routing_testset.py has one response per prompt
(whatever prod actually called), so its oracle_tier is 73% *estimated* from a
complexity proxy. This script closes that gap: it replays every prompt through
every routing candidate on a local GPU, scores each response with the SAME
structural check as Phase 11, and derives an oracle that is 100% *measured*.

Three stages (extract + assemble are CPU-only; replay needs a GPU + vLLM):

  # 1. extract replay jobs from one or more logs     (CPU — run anywhere)
  python scripts/build_response_matrix.py extract \
      --csv data/logs/*.csv --out data/eval/matrix

  # 2. replay one candidate model                   (H200 — once per model, resumable)
  python scripts/build_response_matrix.py replay \
      --jobs data/eval/matrix/jobs.jsonl \
      --model openai/gpt-oss-120b --out data/eval/matrix

  # 3. score + assemble the measured oracle testset (CPU)
  python scripts/build_response_matrix.py assemble \
      --jobs data/eval/matrix/jobs.jsonl \
      --responses data/eval/matrix \
      --out data/eval/routing_testset_measured.jsonl

The stage-3 output is drop-in for eval_router.py:
  python scripts/eval_router.py runs/quality \
      --testset data/eval/routing_testset_measured.jsonl
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

# Reuse the Phase 11 structural scorer + parsers so the measured oracle is
# consistent with the existing (estimated) testset — no scoring drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_routing_testset import (  # noqa: E402
    _extract_content,
    assess_quality,
    complexity_proxy,
)

# ─────────────────────────────────────────────────────────────────────────────
# Routing candidate registry — the 4 text-completion models in the netflow logs
# ─────────────────────────────────────────────────────────────────────────────
MODEL_TIER: dict[str, str] = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": "small",
    "Qwen/Qwen3.5-35B-A3B-FP8":             "mid",
    "openai/gpt-oss-120b":                  "large",
    "Qwen/Qwen3.5-122B-A10B-FP8":           "large",
}
TIER_ORDER = {"small": 0, "mid": 1, "large": 2}
TIERS_CHEAP_FIRST = ["small", "mid", "large"]

csv.field_size_limit(1 << 24)  # proxy_server_request blobs run to ~400 KB


# ─────────────────────────────────────────────────────────────────────────────
# Request parsing (proxy_server_request is a python-dict repr, not JSON)
# ─────────────────────────────────────────────────────────────────────────────
def parse_request(raw: str) -> tuple[list, str, str, dict]:
    """Return (messages, system_prompt, last_user_text, sampling_params)."""
    req = ast.literal_eval(raw)
    msgs = req.get("messages", []) or []

    def _text(c) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return ""

    system = next((_text(m.get("content")) for m in msgs if m.get("role") == "system"), "")
    user = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
    sampling = {
        "temperature": req.get("temperature", 1.0),
        "top_p": req.get("top_p", 1.0),
        "max_tokens": req.get("max_completion_tokens") or req.get("max_tokens") or 4096,
    }
    return msgs, system, user, sampling


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def _iter_jsonl(path: Path):
    """Yield parsed records from a JSONL file, one per '\\n'-delimited line.

    Deliberately iterates the file object (splits on '\\n' only) instead of
    str.splitlines(), which ALSO splits on U+2028/U+2029/U+0085. Those appear in
    scraped text and json.dumps(ensure_ascii=False) writes them raw inside string
    values, so splitlines() would tear one record into invalid fragments.
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — extract replay jobs
# ─────────────────────────────────────────────────────────────────────────────
def cmd_extract(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = out_dir / "jobs.jsonl"

    n_total = n_kept = n_dup = 0
    tier_counts: dict[str, int] = {}
    seen: set[str] = set()  # dedup request_ids across (possibly overlapping) files
    with jobs_path.open("w", encoding="utf-8") as out:
        for csv_path in args.csv:
            n_file = 0
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    n_total += 1
                    model = (row.get("model") or "").strip()
                    if row.get("call_type") != "acompletion" or model not in MODEL_TIER:
                        continue
                    pid = row["request_id"]
                    if pid in seen:
                        n_dup += 1
                        continue
                    try:
                        msgs, system, user, sampling = parse_request(row["proxy_server_request"])
                    except Exception:
                        continue
                    if not msgs:
                        continue
                    seen.add(pid)
                    tier = MODEL_TIER[model]
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    n_kept += 1
                    n_file += 1
                    out.write(json.dumps({
                        "prompt_id": pid,
                        "prompt_text": user,
                        "system_prompt": system,
                        "messages": msgs,
                        "sampling": sampling,
                        "prompt_tokens": int(float(row.get("prompt_tokens") or 0)),
                        "completion_tokens": int(float(row.get("completion_tokens") or 0)),
                        "actual_model": model,
                        "actual_tier": tier,
                        # keep prod's own response so assemble can reuse that cell for free
                        "actual_response": _extract_content(row.get("response") or ""),
                    }, ensure_ascii=False) + "\n")
            print(f"  {csv_path}: +{n_file} jobs")

    print(f"extract: scanned {n_total} rows across {len(args.csv)} file(s) "
          f"→ {n_kept} replay jobs ({n_dup} duplicate request_ids skipped)")
    print(f"  actual-tier distribution: {dict(sorted(tier_counts.items(), key=lambda kv: TIER_ORDER[kv[0]]))}")
    print(f"  wrote {jobs_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — replay one model on GPU (vLLM).  Untested without a GPU by design.
# ─────────────────────────────────────────────────────────────────────────────
def cmd_replay(args) -> None:
    from vllm import LLM, SamplingParams  # imported lazily: only needed on the GPU box
    from tqdm import tqdm

    model_id = args.model
    if model_id not in MODEL_TIER:
        sys.exit(f"replay: {model_id!r} is not a routing candidate ({list(MODEL_TIER)})")
    tier = MODEL_TIER[model_id]

    jobs = list(_iter_jsonl(Path(args.jobs)))

    out_path = Path(args.out) / f"responses_{_safe_name(model_id)}.jsonl"
    done: set[str] = set()
    if out_path.exists():  # resume — skip prompts already generated
        for r in _iter_jsonl(out_path):
            done.add(r["prompt_id"])
    todo = [j for j in jobs if j["prompt_id"] not in done]
    print(f"replay: {model_id} ({tier}) — {len(todo)} to do, {len(done)} already done", flush=True)
    if not todo:
        return

    llm = LLM(
        model=model_id,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,   # skip torch.compile/CUDA-graph capture (fast start-up)
        trust_remote_code=True,
    )

    # Generate in chunks and flush each to disk, so a crash mid-model resumes from
    # the last completed chunk instead of losing the whole run. One tqdm bar tracks
    # overall prompt progress (vLLM's own bar is suppressed to avoid one per chunk).
    bs = args.batch
    with out_path.open("a", encoding="utf-8") as out, \
         tqdm(total=len(todo), desc=f"{tier}:{_safe_name(model_id)}",
              unit="prompt", mininterval=15) as bar:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            conversations = [j["messages"] for j in chunk]
            sampling = [
                SamplingParams(
                    temperature=float(j["sampling"]["temperature"]),
                    top_p=float(j["sampling"]["top_p"]),
                    max_tokens=int(j["sampling"]["max_tokens"]),
                )
                for j in chunk
            ]
            outputs = llm.chat(conversations, sampling, use_tqdm=False)
            for job, res in zip(chunk, outputs):
                gen = res.outputs[0]
                out.write(json.dumps({
                    "prompt_id": job["prompt_id"],
                    "model": model_id,
                    "tier": tier,
                    "response": gen.text,
                    "completion_tokens": len(gen.token_ids),
                }, ensure_ascii=False) + "\n")
            out.flush()          # checkpoint this chunk
            bar.update(len(chunk))
    print(f"replay: wrote {len(todo)} responses → {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — score every cell + derive the measured oracle
# ─────────────────────────────────────────────────────────────────────────────
def load_jobs(path: Path) -> dict[str, dict]:
    return {j["prompt_id"]: j for j in _iter_jsonl(path)}


def score_cell(content: str, completion_tokens: int, system_prompt: str) -> int:
    """Structural quality of one response — Phase 11's assess_quality verbatim."""
    return assess_quality(content, completion_tokens, system_prompt)


def oracle_from_tiers(tier_ok: dict[str, bool]) -> str | None:
    """Cheapest tier with a structurally-valid response; None if all failed."""
    for tier in TIERS_CHEAP_FIRST:
        if tier_ok.get(tier):
            return tier
    return None


def cmd_assemble(args) -> None:
    jobs = load_jobs(Path(args.jobs))
    resp_dir = Path(args.responses)

    # matrix[prompt_id][tier] = 1/0  (a tier passes if any of its models passed)
    matrix: dict[str, dict[str, int]] = {pid: {} for pid in jobs}

    resp_files = sorted(resp_dir.glob("responses_*.jsonl"))
    if not resp_files:
        sys.exit(f"assemble: no responses_*.jsonl found in {resp_dir} — run stage 2 first")
    for rf in resp_files:
        for r in _iter_jsonl(rf):
            job = jobs.get(r["prompt_id"])
            if job is None:
                continue
            q = score_cell(r["response"], int(r["completion_tokens"]), job["system_prompt"])
            tier = r["tier"]
            matrix[r["prompt_id"]][tier] = max(matrix[r["prompt_id"]].get(tier, 0), q)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_excluded = 0
    oracle_dist: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for pid, job in jobs.items():
            tier_ok = {t: bool(v) for t, v in matrix[pid].items()}
            oracle = oracle_from_tiers(tier_ok)
            if oracle is None:  # no tier produced a valid response → oracle undefined
                n_excluded += 1
                continue
            oracle_dist[oracle] = oracle_dist.get(oracle, 0) + 1
            n_written += 1
            out.write(json.dumps({
                "prompt_id": pid,
                "prompt_text": job["prompt_text"],
                "oracle_tier": oracle,
                "oracle_confidence": "confirmed",  # measured by real replay, not estimated
                "actual_tier": job["actual_tier"],
                "prompt_tokens": job["prompt_tokens"],
                "completion_tokens": job["completion_tokens"],
                "complexity_proxy": complexity_proxy(job["prompt_tokens"], job["completion_tokens"]),
            }, ensure_ascii=False) + "\n")

    print(f"assemble: {n_written} rows written, {n_excluded} excluded (all tiers failed)")
    print(f"  measured oracle distribution: {dict(sorted(oracle_dist.items(), key=lambda kv: TIER_ORDER[kv[0]]))}")
    print(f"  wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="extract replay jobs from one or more LiteLLM log CSVs")
    pe.add_argument("--csv", nargs="+", default=["data/eval/intern_data.csv"],
                    help="one or more log CSVs (globs work: data/logs/*.csv); request_ids deduped across files")
    pe.add_argument("--out", default="data/eval/matrix")
    pe.set_defaults(func=cmd_extract)

    pr = sub.add_parser("replay", help="replay one candidate model on GPU (vLLM)")
    pr.add_argument("--jobs", default="data/eval/matrix/jobs.jsonl")
    pr.add_argument("--model", required=True)
    pr.add_argument("--out", default="data/eval/matrix")
    pr.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    pr.add_argument("--gpu-mem-util", type=float, default=0.90)
    pr.add_argument("--batch", type=int, default=512,
                    help="prompts per generation chunk; each chunk is flushed to disk (resume granularity)")
    pr.add_argument("--eager", action="store_true",
                    help="enforce_eager: skip torch.compile/CUDA-graph capture — starts generating in ~1 min "
                         "instead of a multi-minute compile (slightly lower throughput; good for one-shot batch)")
    pr.add_argument("--max-model-len", type=int, default=32768,
                    help="context window; prompts here run up to ~98k tokens — raise if you don't want the long tail truncated (costs KV-cache VRAM, tight on the 122B)")
    pr.set_defaults(func=cmd_replay)

    pa = sub.add_parser("assemble", help="score responses + write measured oracle testset")
    pa.add_argument("--jobs", default="data/eval/matrix/jobs.jsonl")
    pa.add_argument("--responses", default="data/eval/matrix")
    pa.add_argument("--out", default="data/eval/routing_testset_measured.jsonl")
    pa.set_defaults(func=cmd_assemble)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
