#!/usr/bin/env bash
# Full teacher training + evaluation pipeline — sequential, versioned, MLflow-tracked.
# Counterpart of run_full_distillation.sh, for (re)training the teacher backbones.
#
# For each teacher, in sequence (one at a time → fits the 12GB H200 partition):
#   1. train the quality model              (AMP, logged to MLflow)
#   2. evaluate immediately on BOTH the production logs and the routing eval set
#      via eval_all_teachers.py (eval_logs.py + eval_router.py), logged to MLflow.
#
# VERSIONING: every invocation gets a RUN id (timestamp by default, or set a
# semantic one). All checkpoints go under runs/teachers/<RUN>/ and MLflow runs are
# named "<RUN>/<model>" + tagged pipeline_run=<RUN>, so retrains never clobber.
#
# Launch detached:
#   RUN=t-v2-e3 nohup bash scripts/run_full_teacher_training.sh &
#   tail -f runs/teachers/<RUN>/pipeline.log
#   mlflow ui --backend-store-uri ./mlruns          # filter by tag pipeline_run=<RUN>
#
# 12GB VRAM note: the 300M+ backbones are tight at seq 512. If you OOM, set
#   GRADIENT_CHECKPOINTING=1   (halves activation memory; ~20% slower)
#   ADAFACTOR=1                (≈10× less optimizer memory than AdamW)
#   BATCH=8                    (smaller batch)
# bge-m3 (568M) in particular will likely need GRADIENT_CHECKPOINTING=1 here.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export VI_ROUTER_REPO_ROOT="$REPO"

# ── config (override via env) ────────────────────────────────────────────────
PYTHON="${PYTHON:-uv run --extra ml python}"
DATA_ROOT="${DATA_ROOT:-data/processed/v2}"      # v2 schema dataset
OUT_ROOT="${OUT_ROOT:-runs/teachers}"
RUN="${RUN:-$(date +%Y%m%d-%H%M%S)}"
SCHEMA="${SCHEMA:-v2}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-16}"
LR="${LR:-2e-5}"
MAX_STEPS="${MAX_STEPS:-}"                        # cap steps/teacher (smoke-test the wiring)
TEACHERS="${TEACHERS:-}"                          # space-separated subset, else all 4
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-}"   # set to 1 to enable
ADAFACTOR="${ADAFACTOR:-}"                        # set to 1 to enable
CSV="${CSV:-data/eval/intern_data.csv}"
TESTSET="${TESTSET:-data/eval/routing_testset.jsonl}"
MLFLOW_EXP="${MLFLOW_EXP:-vi-smart-routing}"
MLFLOW_URI="${MLFLOW_URI:-$REPO/mlruns}"

RUN_DIR="$OUT_ROOT/$RUN"
mkdir -p "$RUN_DIR"
export MLFLOW_TRACKING_URI="$MLFLOW_URI"
# Newer MLflow gates the ./mlruns file store behind this opt-out; we use file store.
export MLFLOW_ALLOW_FILE_STORE=true
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

# teacher model_name keys (configs/model.yaml) — trained in this order
ALL_TEACHERS=(
  vi-router-quality          # mDeBERTa-v3-base
  vi-router-quality-mmbert   # mmBERT-base
  vi-router-quality-granite  # Granite-311m
  vi-router-quality-bgem3    # BGE-M3 568M (heaviest — needs GRADIENT_CHECKPOINTING on 12GB)
)

echo "=================================================================="
echo " TEACHER TRAINING PIPELINE  |  RUN=$RUN  |  start $(date)"
echo " artifacts : $RUN_DIR"
echo " data=$DATA_ROOT  schema=$SCHEMA  epochs=$EPOCHS  batch=$BATCH  lr=$LR"
echo " grad_ckpt=${GRADIENT_CHECKPOINTING:-off}  adafactor=${ADAFACTOR:-off}"
echo " MLflow    : experiment=$MLFLOW_EXP  uri=$MLFLOW_URI  (tag pipeline_run=$RUN)"
echo " python    : $PYTHON"
echo "=================================================================="

for teacher in "${ALL_TEACHERS[@]}"; do
  # Optional subset filter (TEACHERS="a b").
  if [[ -n "$TEACHERS" && " $TEACHERS " != *" $teacher "* ]]; then continue; fi

  tdir="$RUN_DIR/$teacher"

  echo; echo "##################################################################"
  echo "### [$(date)] TRAIN  $teacher"
  echo "##################################################################"
  $PYTHON -m classifier.train \
    --model "$teacher" --out "$tdir" \
    --data "$DATA_ROOT" --schema-version "$SCHEMA" \
    --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
    --run-name "$RUN" ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
    ${GRADIENT_CHECKPOINTING:+--gradient-checkpointing} ${ADAFACTOR:+--adafactor} \
    --mlflow-experiment "$MLFLOW_EXP" --mlflow-tracking-uri "$MLFLOW_URI"

  echo; echo "### [$(date)] EVALUATE  $teacher  (production logs + routing eval set)"
  $PYTHON scripts/eval_all_teachers.py \
    --checkpoints-root "$RUN_DIR" --models "$teacher" \
    --csv "$CSV" --testset "$TESTSET" --schema-version "$SCHEMA" \
    --run-name "$RUN" --python "$PYTHON" \
    --mlflow-experiment "$MLFLOW_EXP" --mlflow-tracking-uri "$MLFLOW_URI"
done

echo; echo "=== TEACHER PIPELINE DONE $(date) — RUN=$RUN — checkpoints in $RUN_DIR ==="
echo "    Distill from them with:  TEACHERS_ROOT=$RUN_DIR bash scripts/run_full_distillation.sh"
