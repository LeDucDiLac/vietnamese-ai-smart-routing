#!/usr/bin/env bash
# prefetch_models.sh — pre-download the replay models into the HF cache with a
# VISIBLE, resumable progress bar and hf_transfer (fast parallel download). Run this
# BEFORE launch_replay.sh. vLLM's own "Starting to load model" step is silent — with
# a slow single-stream download it sits there for a long time with GPU at ~0 MiB and
# no log output, which looks exactly like a freeze. Prefetching makes the download
# observable (and much faster), so the later vLLM load is seconds from cache.
#
#   bash scripts/prefetch_models.sh                 # all replay models
#   bash scripts/prefetch_models.sh Qwen/Qwen3-30B-A3B-Instruct-2507-FP8   # just one
#
# Downloads land in $HF_HOME (default ~/.cache/huggingface). The full set is ~250 GB
# (30B + 35B + 120B + 122B). If that disk is small, point it at a big volume first:
#   export HF_HOME=/home/jovyan/work/leduc/.hf-cache
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON="${PYTHON:-$REPO_DIR/.venv-replay/bin/python}"
source "$REPO_DIR/scripts/resolve_hf_token.sh"

# fast parallel downloads: install the backend if missing, then turn it on
"$PYTHON" -c 'import hf_transfer' 2>/dev/null || { echo "installing hf_transfer…"; uv pip install --python "$PYTHON" hf_transfer; }
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    "Qwen/Qwen3.5-35B-A3B-FP8"
    "openai/gpt-oss-120b"
    "Qwen/Qwen3.5-122B-A10B-FP8"
  )
fi

CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$CACHE"
echo "HF cache: $CACHE   free: $(df -h "$CACHE" | awk 'NR==2{print $4}')"

for M in "${MODELS[@]}"; do
  echo "===== $(date +%H:%M:%S)  $M ====="
  # Retry with resume: HF's CDN drops big shards mid-stream ("peer closed connection").
  # snapshot_download resumes from the partial file, so re-running just continues.
  ok=0
  for attempt in $(seq 1 8); do
    if "$PYTHON" - "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("cached ->", snapshot_download(sys.argv[1], max_workers=4))
PY
    then ok=1; break; fi
    echo "  attempt $attempt failed (connection drop?) — resuming in 5s…"
    sleep 5
  done
  [ "$ok" = 1 ] || echo "!!!!! $M still failing after 8 attempts — check access/gating/disk above !!!!!"
done
echo "===== prefetch done $(date +%H:%M:%S)   cache now: $(du -sh "$CACHE/hub" 2>/dev/null | cut -f1) ====="
