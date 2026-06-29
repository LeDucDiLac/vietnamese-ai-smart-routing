#!/usr/bin/env bash
# Full student distillation + evaluation pipeline — sequential, MLflow-tracked.
#
# For each student, in sequence (one at a time → fits the 12GB H200 partition):
#   1. distill from its v2 teacher          (AMP, logged to MLflow)
#   2. export to INT8 ONNX
#   3. evaluate immediately on BOTH:
#        - the production logs   (eval_logs.py  → data/eval/intern_data.csv)
#        - the eval set          (eval_router.py → routing_testset.jsonl, torch + INT8 ONNX)
#      also logged to MLflow.
#
# Launch detached (survives logout):
#   nohup bash scripts/run_full_distillation.sh > runs/students/pipeline.log 2>&1 &
#   tail -f runs/students/pipeline.log
#   mlflow ui --backend-store-uri ./mlruns      # to watch metrics live
#
# Override any setting via env, e.g.:
#   BATCH=24 EPOCHS=4 PYTHON=".venv/bin/python" nohup bash scripts/run_full_distillation.sh ...

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export VI_ROUTER_REPO_ROOT="$REPO"

# ── config (override via env) ────────────────────────────────────────────────
PYTHON="${PYTHON:-uv run --extra ml python}"
DATA_ROOT="${DATA_ROOT:-data/processed/v2}"      # v2 teachers ⇒ v2 data
TEACHERS_ROOT="${TEACHERS_ROOT:-runs/teachers}"
OUT_ROOT="${OUT_ROOT:-runs/students}"
SCHEMA="${SCHEMA:-v2}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-32}"                              # 32 ⇒ ~9GB peak (granite), fits 12GB
TEMP="${TEMP:-2.0}"
ALPHA="${ALPHA:-0.5}"
CSV="${CSV:-data/eval/intern_data.csv}"          # production logs
TESTSET="${TESTSET:-data/eval/routing_testset.jsonl}"
MLFLOW_EXP="${MLFLOW_EXP:-vi-smart-routing}"
MLFLOW_URI="${MLFLOW_URI:-$REPO/mlruns}"

# teacher_dir : student_name  (run in this order)
PAIRS=(
  "vi-router-quality-granite:vi-router-fast-granite"   # v2 accuracy/MAE winner
  "vi-router-quality-mmbert:vi-router-fast-mmbert"     # v2 test-F1 winner
  "vi-router-quality:vi-router-fast"                   # mDeBERTa → MiniLM baseline
)

mkdir -p "$OUT_ROOT"
export MLFLOW_TRACKING_URI="$MLFLOW_URI"

echo "=================================================================="
echo " STUDENT DISTILLATION PIPELINE  |  start $(date)"
echo " data=$DATA_ROOT  schema=$SCHEMA  epochs=$EPOCHS  batch=$BATCH"
echo " MLflow: experiment=$MLFLOW_EXP  uri=$MLFLOW_URI"
echo " python: $PYTHON"
echo "=================================================================="

for pair in "${PAIRS[@]}"; do
  teacher="${pair%%:*}"
  student="${pair##*:}"
  tdir="$TEACHERS_ROOT/$teacher"
  sdir="$OUT_ROOT/$student"

  if [[ ! -f "$tdir/model.pt" ]]; then
    echo; echo "### SKIP $student — teacher checkpoint missing at $tdir"; continue
  fi

  echo; echo "##################################################################"
  echo "### [$(date)] DISTILL  $student   <-   $teacher"
  echo "##################################################################"
  $PYTHON -m classifier.distill \
    --teacher "$tdir" --out "$sdir" --student "$student" \
    --data "$DATA_ROOT" --schema-version "$SCHEMA" \
    --epochs "$EPOCHS" --batch-size "$BATCH" \
    --temperature "$TEMP" --alpha "$ALPHA" \
    --mlflow-experiment "$MLFLOW_EXP" --mlflow-tracking-uri "$MLFLOW_URI"

  echo; echo "### [$(date)] EXPORT  $student  ->  INT8 ONNX"
  if ! $PYTHON -m classifier.export_onnx \
        --checkpoint "$sdir/model.pt" --model-name "$student" \
        --out-dir "$sdir/onnx" --schema-version "$SCHEMA"; then
    echo "### WARN: export failed for $student — eval will use the torch path only"
  fi

  echo; echo "### [$(date)] EVALUATE  $student  (production logs + routing eval set)"
  $PYTHON scripts/eval_all_students.py \
    --students-root "$OUT_ROOT" --teachers-root "$TEACHERS_ROOT" \
    --students "$student" --csv "$CSV" --testset "$TESTSET" \
    --schema-version "$SCHEMA" --python "$PYTHON" \
    --mlflow-experiment "$MLFLOW_EXP" --mlflow-tracking-uri "$MLFLOW_URI"
done

echo; echo "##################################################################"
echo "### [$(date)] FINAL comparison across all students"
echo "##################################################################"
$PYTHON scripts/eval_all_students.py \
  --students-root "$OUT_ROOT" --teachers-root "$TEACHERS_ROOT" \
  --csv "$CSV" --testset "$TESTSET" --schema-version "$SCHEMA" \
  --python "$PYTHON" --run-name final \
  --mlflow-experiment "$MLFLOW_EXP" --mlflow-tracking-uri "$MLFLOW_URI" || true

echo; echo "=== PIPELINE DONE $(date) — checkpoints in $OUT_ROOT, metrics in MLflow ($MLFLOW_EXP) ==="
