#!/usr/bin/env bash
# run_replay.sh — replay every routing candidate over the extracted jobs to build
# the response matrix. Runs the models cheap→expensive, one at a time (GPU freed
# between models), and keeps going if one model fails so you still get the rest.
#
# Usage (from anywhere — the script cd's to the repo root itself):
#     bash scripts/run_replay.sh [JOBS] [OUT]
#
# Override the interpreter / knobs via env vars:
#     PYTHON=/home/leduc/ai-smart-routing/.venv/bin/python \
#     GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=32768 BATCH=512 \
#     bash scripts/run_replay.sh
#
# Detached run that survives disconnect (see progress in the log):
#     nohup bash scripts/run_replay.sh > data/eval/matrix/replay.log 2>&1 &
#     tail -f data/eval/matrix/replay.log
#
# Safe to re-run: replay skips prompt_ids already written per model.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

JOBS="${1:-data/eval/matrix/jobs.jsonl}"
OUT="${2:-data/eval/matrix}"
PYTHON="${PYTHON:-python}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BATCH="${BATCH:-512}"
EAGER_FLAG=""; [ -n "${EAGER:-}" ] && EAGER_FLAG="--eager"   # EAGER=1 skips torch.compile
export PYTHONUNBUFFERED=1   # stream vLLM's load/compile output to the log LIVE — else it
                            # block-buffers and the multi-minute torch.compile looks frozen

# HuggingFace auth: resolve HF_TOKEN (from .claude/settings.local.json / env / .env)
# so the 30–120 GB FP8 shard downloads aren't unauthenticated + rate-limited — that
# stall looks like a freeze right after "Starting to load model". hf_transfer speeds
# the download when the `hf_transfer` package is installed.
source "$REPO_DIR/scripts/resolve_hf_token.sh"
# fast parallel HF downloads when the backend is installed (prefetch_models.sh installs it)
if [ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]; then
  "$PYTHON" -c 'import hf_transfer' 2>/dev/null && HF_HUB_ENABLE_HF_TRANSFER=1 || HF_HUB_ENABLE_HF_TRANSFER=0
fi
export HF_HUB_ENABLE_HF_TRANSFER

# Cheap→expensive; the GPU is freed between models, so a fresh vLLM starts once per
# model and pays a full torch.compile each time (kept ON for throughput — EAGER=1 is
# only an escape hatch and is NOT used here).
# The two Qwen3.5 MoE models need a vLLM build that registers their arch
# (Qwen3_5MoeForCausalLM / Qwen3_5MoeForConditionalGeneration); vLLM 0.24 does NOT.
# Verify before trusting their rows:
#   .venv-replay/bin/python -c "from vllm.model_executor.models.registry import \
#     ModelRegistry as R; print(R.get_supported_archs())"
# run_replay continues past a model that fails to load, so leaving them in is safe.
MODELS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"   # small — Qwen3 MoE, works on vLLM 0.24
  "Qwen/Qwen3.5-35B-A3B-FP8"               # mid   — Qwen3.5 MoE, needs newer vLLM
  "openai/gpt-oss-120b"                     # large — works on vLLM 0.24
  "Qwen/Qwen3.5-122B-A10B-FP8"             # large — Qwen3.5 MoE, needs newer vLLM + likely --tp 2
)

if [ ! -f "$JOBS" ]; then
  echo "ERROR: jobs file not found: $JOBS  (run the extract stage first)" >&2
  exit 1
fi
mkdir -p "$OUT"
echo $$ > "$OUT/replay.pid"        # record own PID so launch/watch/stop can find us
trap 'rm -f "$OUT/replay.pid"' EXIT   # clear it on normal exit/SIGTERM so no stale pidfile lingers
echo "repo=$REPO_DIR"
echo "jobs=$JOBS  out=$OUT  batch=$BATCH  gpu_mem_util=$GPU_MEM_UTIL  max_model_len=$MAX_MODEL_LEN"
echo "python=$("$PYTHON" -c 'import sys; print(sys.executable)')"

for M in "${MODELS[@]}"; do
  # Per-model resource overrides (fall back to the globals). The 122B is ~122 GB of
  # FP8 weights on a single 141 GB H200 — shrink context + batch and push mem-util
  # up so the KV cache has any room at all. TP stays 1 (only one GPU), so this is a
  # best-effort squeeze; it may still OOM (a real 122B pass wants a 2nd H200, --tp 2).
  MML="$MAX_MODEL_LEN"; GMU="$GPU_MEM_UTIL"; BS="$BATCH"
  case "$M" in
    *122B*) MML=8192; GMU=0.96; BS=32 ;;
  esac
  echo "===== $(date +%H:%M:%S)  $M  (max_model_len=$MML gpu_mem_util=$GMU batch=$BS) ====="
  stdbuf -oL -eL "$PYTHON" -u scripts/build_response_matrix.py replay \
      --jobs "$JOBS" --model "$M" --out "$OUT" \
      --gpu-mem-util "$GMU" --max-model-len "$MML" --batch "$BS" $EAGER_FLAG \
      || echo "!!!!! $M FAILED (continuing) !!!!!"
done

echo "===== ALL DONE $(date +%H:%M:%S) ====="
