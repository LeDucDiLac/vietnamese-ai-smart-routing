#!/usr/bin/env python3
"""Evaluate all 4 teacher model checkpoints and print a comparison table.

Runs eval_logs.py (production log KPIs) and eval_router.py (routing testset)
for each model found under --checkpoints-root, then prints a side-by-side table
and saves a combined report.

Usage
─────
  # Evaluate all models in runs/teachers/
  python scripts/eval_all_teachers.py --checkpoints-root runs/teachers/

  # Evaluate specific models only
  python scripts/eval_all_teachers.py --checkpoints-root runs/teachers/ \\
      --models vi-router-quality vi-router-quality-mmbert

  # Custom eval data paths
  python scripts/eval_all_teachers.py --checkpoints-root runs/teachers/ \\
      --csv data/eval/intern_data.csv \\
      --testset data/eval/routing_testset.jsonl

H200 quick-start (after git pull):
  python3 scripts/eval_all_teachers.py \\
      --checkpoints-root runs/teachers/ \\
      --python .venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

TEACHER_MODELS = [
    "vi-router-quality",
    "vi-router-quality-mmbert",
    "vi-router-quality-bgem3",
    "vi-router-quality-granite",
]


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VI_ROUTER_REPO_ROOT"] = str(REPO)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _fmt(v: object, decimals: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)


def _log_to_mlflow(
    model_name: str,
    result: dict,
    out_dir: Path,
    mlflow_experiment: str,
    mlflow_tracking_uri: str | None,
) -> None:
    if not _MLFLOW:
        return
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"{model_name}-eval"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("eval_type", "routing_simulation")

        kpis = result.get("eval_logs", {}).get("kpis", result.get("eval_logs", {}))
        logs_metrics = {
            "cost_saving_pct": kpis.get("cost_saving_pct"),
            "quality_loss_pct": kpis.get("quality_loss_pct"),
            "latency_reduction_pct": kpis.get("latency_reduction_pct"),
            "router_latency_ms": kpis.get("router_latency_ms"),
        }
        router = result.get("eval_router", {})
        router_metrics = {
            "avg_acc": router.get("avg_acc"),
            "gap_to_oracle": router.get("gap_to_oracle"),
            "cost_save": router.get("cost_save"),
            "pgr": router.get("pgr"),
            "aiq": router.get("aiq"),
        }
        all_metrics = {k: float(v) for k, v in {**logs_metrics, **router_metrics}.items() if v is not None}
        if all_metrics:
            mlflow.log_metrics(all_metrics)

        for artifact_name in ("eval_logs/report.json", "eval_router/report.json"):
            p = out_dir / artifact_name
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path=artifact_name.split("/")[0])


def run_eval(
    model_name: str,
    ckpt_dir: Path,
    out_dir: Path,
    csv_path: str,
    testset_path: str,
    python_cmd: list[str],
    env: dict[str, str],
    schema_version: str | None = None,
    mlflow_experiment: str = "vi-smart-routing",
    mlflow_tracking_uri: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"model_name": model_name}

    if not ckpt_dir.exists():
        print(f"  [SKIP] {model_name} — checkpoint not found at {ckpt_dir}", flush=True)
        result["error"] = "checkpoint not found"
        return result

    # --- eval_logs.py ---
    logs_out = out_dir / "eval_logs"
    cmd_logs = [
        *python_cmd, str(REPO / "scripts" / "eval_logs.py"),
        "--csv", csv_path,
        "--model-path", str(ckpt_dir),
        "--out", str(logs_out),
    ]
    t0 = time.time()
    print(f"  [{model_name}] running eval_logs ...", flush=True)
    proc = subprocess.run(cmd_logs, env=env, cwd=str(REPO),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [FAIL] {model_name} eval_logs exit={proc.returncode}", flush=True)
        print(proc.stderr[-1000:] if proc.stderr else "", flush=True)
        result["eval_logs_error"] = f"exit {proc.returncode}"
    else:
        report_path = logs_out / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            result["eval_logs"] = report
            _print_logs_summary(model_name, report)

    # --- eval_router.py ---
    router_out = out_dir / "eval_router"
    cmd_router = [
        *python_cmd, str(REPO / "scripts" / "eval_router.py"),
        str(ckpt_dir),
        "--testset", testset_path,
        "--out", str(router_out),
        *( ["--schema-version", schema_version] if schema_version else []),
    ]
    print(f"  [{model_name}] running eval_router ...", flush=True)
    proc2 = subprocess.run(cmd_router, env=env, cwd=str(REPO),
                           capture_output=True, text=True)
    if proc2.returncode != 0:
        print(f"  [FAIL] {model_name} eval_router exit={proc2.returncode}", flush=True)
        print(proc2.stderr[-1000:] if proc2.stderr else "", flush=True)
        result["eval_router_error"] = f"exit {proc2.returncode}"
    else:
        router_report = router_out / "report.json"
        if router_report.exists():
            result["eval_router"] = json.loads(router_report.read_text())

    result["elapsed_s"] = time.time() - t0
    _log_to_mlflow(model_name, result, out_dir, mlflow_experiment, mlflow_tracking_uri)
    return result


def _print_logs_summary(model_name: str, report: dict) -> None:
    kpis = report.get("kpis", report)
    print(
        f"    cost_saving={_fmt(kpis.get('cost_saving_pct'), 1)}% "
        f"quality_loss={_fmt(kpis.get('quality_loss_pct'), 1)}% "
        f"latency_reduction={_fmt(kpis.get('latency_reduction_pct'), 1)}% "
        f"router_latency={_fmt(kpis.get('router_latency_ms'), 2)}ms",
        flush=True,
    )


def print_table(results: list[dict]) -> None:
    headers = [
        "Model", "Cost Save%", "Qual Loss%", "Lat Reduce%", "Router ms",
        "Avg Acc", "Gap@O", "Time(m)",
    ]

    rows = []
    for r in results:
        name = r.get("model_name", "?")
        if "error" in r:
            rows.append([name, r["error"], "—", "—", "—", "—", "—", "—"])
            continue
        kpis = r.get("eval_logs", {}).get("kpis", r.get("eval_logs", {}))
        router = r.get("eval_router", {})
        rows.append([
            name,
            _fmt(kpis.get("cost_saving_pct"), 1),
            _fmt(kpis.get("quality_loss_pct"), 1),
            _fmt(kpis.get("latency_reduction_pct"), 1),
            _fmt(kpis.get("router_latency_ms"), 2),
            _fmt(router.get("avg_acc")),
            _fmt(router.get("gap_to_oracle")),
            _fmt(r.get("elapsed_s", 0) / 60, 1),
        ])

    col_w = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cells[j].ljust(col_w[j]) for j in range(len(headers)))

    sep = "-+-".join("-" * w for w in col_w)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate all teacher model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--checkpoints-root", required=True,
                    help="Directory containing per-model checkpoint subdirs (e.g. runs/teachers/)")
    ap.add_argument("--out-root", default=None,
                    help="Output root for eval reports (default: <checkpoints-root>/eval/)")
    ap.add_argument("--models", nargs="+", default=TEACHER_MODELS, metavar="MODEL",
                    help="Which models to evaluate (default: all 4)")
    ap.add_argument("--csv", default="data/eval/intern_data.csv",
                    help="Path to intern_data.csv for eval_logs.py")
    ap.add_argument("--testset", default="data/eval/routing_testset.jsonl",
                    help="Path to routing_testset.jsonl for eval_router.py")
    ap.add_argument("--python", default=None, metavar="CMD",
                    help="Python interpreter (default: uv run --extra ml python)")
    ap.add_argument("--schema-version", default=None, metavar="VERSION",
                    help="Label schema version to force (e.g. 'v2'). Use when meta.json "
                         "does not record the version.")
    ap.add_argument("--mlflow-experiment", default="vi-smart-routing",
                    help="MLflow experiment name (default: vi-smart-routing)")
    ap.add_argument("--mlflow-tracking-uri", default=None,
                    help="MLflow tracking URI (default: local ./mlruns)")
    args = ap.parse_args()

    ckpts_root = Path(args.checkpoints_root)
    out_root = Path(args.out_root) if args.out_root else ckpts_root / "eval"
    python_cmd = args.python.split() if args.python else ["uv", "run", "--extra", "ml", "python"]
    env = _build_env()

    print(f"Checkpoints : {ckpts_root}")
    print(f"Output      : {out_root}")
    print(f"Models      : {args.models}")
    print(f"CSV         : {args.csv}")
    print(f"Testset     : {args.testset}")
    print()

    results = []
    for model_name in args.models:
        ckpt_dir = ckpts_root / model_name
        out_dir = out_root / model_name
        print(f"=== {model_name} ===", flush=True)
        r = run_eval(model_name, ckpt_dir, out_dir, args.csv, args.testset, python_cmd, env,
                     schema_version=args.schema_version,
                     mlflow_experiment=args.mlflow_experiment,
                     mlflow_tracking_uri=args.mlflow_tracking_uri)
        results.append(r)
        print()

    report_path = out_root / "teacher_eval_comparison.json"
    out_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n=== Teacher Eval Comparison ===\n")
    print_table(results)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
