#!/usr/bin/env python3
"""Evaluate all distilled students: classification parity + routing + latency.

For each student under --students-root, this runs eval_router.py twice:

  1. on the PyTorch checkpoint dir  → routing / industrial KPIs
  2. on the INT8 ONNX export        → the CPU serving path, incl. router_latency_ms
                                       (the ≤50ms SLA number)

It also reads classification parity (macro-F1 / acc / complexity MAE on val+test,
written into each student's meta.json by distill.py) and compares it against the
matching teacher's meta.json under --teachers-root, so the parity table shows the
student-vs-teacher delta.

Usage
─────
  python scripts/eval_all_students.py \
      --students-root runs/students --teachers-root runs/teachers

  # ONNX path needs the export present: runs/students/<name>/onnx/model.int8.onnx
"""

from __future__ import annotations

import argparse
import json
import math
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

# (student model_name, teacher dir under --teachers-root)
PAIRS = [
    ("vi-router-fast-granite", "vi-router-quality-granite"),
    ("vi-router-fast-mmbert", "vi-router-quality-mmbert"),
    ("vi-router-fast", "vi-router-quality"),
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
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "—"
    return f"{f:.{decimals}f}"


def _run_eval_router(
    model_path: str,
    out_dir: Path,
    testset: str,
    python_cmd: list[str],
    env: dict[str, str],
    *,
    backbone: str | None = None,
    max_tokens: int | None = None,
    schema_version: str | None = None,
) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *python_cmd, str(REPO / "scripts" / "eval_router.py"),
        model_path,
        "--testset", testset,
        "--out", str(out_dir),
    ]
    if backbone:
        cmd += ["--backbone", backbone]
    if max_tokens:
        cmd += ["--max-tokens", str(max_tokens)]
    if schema_version:
        cmd += ["--schema-version", schema_version]
    proc = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    eval_router FAIL exit={proc.returncode}", flush=True)
        print((proc.stderr or "")[-800:], flush=True)
        return None
    report = out_dir / "eval_router.json"
    if not report.exists():
        return None
    results = json.loads(report.read_text()).get("results", [])
    return results[0] if results else None


def _run_eval_logs(
    ckpt_dir: str,
    out_dir: Path,
    csv: str,
    python_cmd: list[str],
    env: dict[str, str],
) -> dict | None:
    """Production-log KPIs via eval_logs.py (schema auto-detected from meta.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *python_cmd, str(REPO / "scripts" / "eval_logs.py"),
        "--csv", csv,
        "--model-path", ckpt_dir,
        "--out", str(out_dir),
    ]
    proc = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    eval_logs FAIL exit={proc.returncode}", flush=True)
        print((proc.stderr or "")[-800:], flush=True)
        return None
    report = out_dir / "eval_logs.json"
    if not report.exists():
        return None
    scenarios = json.loads(report.read_text()).get("scenarios", [])
    # Pick the vi-router scenario (the real classifier), not baseline/heuristic.
    for s in scenarios:
        if s.get("scenario") == "vi-router":
            return s
    return scenarios[-1] if scenarios else None


def _parity(student_meta: dict, teacher_meta: dict) -> dict:
    """Student test metrics + delta vs the teacher's own test metrics."""
    def g(m: dict, k: str) -> float:
        v = m.get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    s_f1 = g(student_meta, "test_task_macro_f1")
    t_f1 = g(teacher_meta, "test_task_macro_f1")
    s_acc = g(student_meta, "test_task_top1_acc")
    s_mae = g(student_meta, "test_complexity_mae")
    t_mae = g(teacher_meta, "test_complexity_mae")
    return {
        "test_f1": s_f1,
        "test_acc": s_acc,
        "test_mae": s_mae,
        "delta_f1_vs_teacher": (s_f1 - t_f1) if not (math.isnan(s_f1) or math.isnan(t_f1)) else math.nan,
        "delta_mae_vs_teacher": (s_mae - t_mae) if not (math.isnan(s_mae) or math.isnan(t_mae)) else math.nan,
    }


def _log_to_mlflow(student: str, result: dict, experiment: str, tracking_uri: str | None,
                   run_tag: str | None = None) -> None:
    if not _MLFLOW:
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    label = f"{run_tag}/{student}-eval" if run_tag else f"{student}-eval"
    with mlflow.start_run(run_name=label):
        mlflow.log_param("model_name", student)
        if run_tag:
            mlflow.set_tag("pipeline_run", run_tag)
        metrics: dict[str, float] = {}
        onnx = result.get("onnx") or {}
        torch_r = result.get("torch") or {}
        logs = result.get("eval_logs") or {}
        for src, r in (("onnx", onnx), ("torch", torch_r)):
            for k in ("router_latency_ms", "cost_saving_pct", "quality_loss_pct",
                      "latency_reduction_pct", "avg_acc", "gap_to_oracle", "pgr"):
                if r.get(k) is not None:
                    metrics[f"{src}_{k}"] = float(r[k])
        for k in ("cost_saving_pct", "quality_loss_pct", "latency_reduction_pct", "router_latency_ms"):
            if logs.get(k) is not None:
                metrics[f"logs_{k}"] = float(logs[k])
        for k, v in result.get("parity", {}).items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                metrics[k] = float(v)
        if metrics:
            mlflow.log_metrics(metrics)


def run_eval(
    student: str,
    teacher: str,
    students_root: Path,
    teachers_root: Path,
    out_dir: Path,
    testset: str,
    csv: str,
    python_cmd: list[str],
    env: dict[str, str],
    *,
    schema_version: str | None,
    experiment: str,
    tracking_uri: str | None,
    run_tag: str | None = None,
    log_mlflow: bool = True,
) -> dict:
    result: dict = {"model_name": student}
    ckpt_dir = students_root / student
    if not (ckpt_dir / "model.pt").exists():
        print(f"  [SKIP] {student} — no checkpoint at {ckpt_dir}", flush=True)
        result["error"] = "checkpoint not found"
        return result

    student_meta = json.loads((ckpt_dir / "meta.json").read_text())
    backbone = student_meta.get("backbone")
    max_tokens = student_meta.get("max_tokens")
    # Resolve schema: explicit flag wins, else read what distill recorded in meta.
    if not schema_version:
        sv = student_meta.get("schema_version")
        schema_version = sv if sv and sv != "default" else None
    t0 = time.time()

    # 1) production-log KPIs (eval_logs.py)
    print(f"  [{student}] eval_logs (production logs) ...", flush=True)
    logs = _run_eval_logs(str(ckpt_dir), out_dir / "eval_logs", csv, python_cmd, env)
    result["eval_logs"] = logs
    if logs:
        print(f"    logs: cost_save={_fmt(logs.get('cost_saving_pct'), 1)}% "
              f"quality_loss={_fmt(logs.get('quality_loss_pct'), 1)}% "
              f"router_latency={_fmt(logs.get('router_latency_ms'), 2)}ms", flush=True)

    # 2) routing KPIs on the torch checkpoint
    print(f"  [{student}] eval_router (torch) ...", flush=True)
    result["torch"] = _run_eval_router(
        str(ckpt_dir), out_dir / "torch", testset, python_cmd, env,
        schema_version=schema_version,
    )

    # 3) serving-path KPIs + latency on the INT8 ONNX
    onnx_path = ckpt_dir / "onnx" / "model.int8.onnx"
    if onnx_path.exists():
        print(f"  [{student}] eval_router (int8 onnx) ...", flush=True)
        result["onnx"] = _run_eval_router(
            str(onnx_path), out_dir / "onnx", testset, python_cmd, env,
            backbone=backbone, max_tokens=max_tokens, schema_version=schema_version,
        )
    else:
        print(f"  [{student}] no INT8 ONNX at {onnx_path} — run distill --export first", flush=True)
        result["onnx"] = None

    # 4) classification parity vs teacher
    teacher_meta_path = teachers_root / teacher / "meta.json"
    teacher_meta = json.loads(teacher_meta_path.read_text()) if teacher_meta_path.exists() else {}
    result["parity"] = _parity(student_meta, teacher_meta)

    result["elapsed_s"] = time.time() - t0
    if log_mlflow:
        _log_to_mlflow(student, result, experiment, tracking_uri, run_tag=run_tag)
    return result


def print_table(results: list[dict]) -> None:
    headers = [
        "Student", "Test F1", "ΔF1 vs T", "Test MAE", "Router ms", "Cost Save%",
        "Qual Loss%", "Avg Acc", "Time(m)",
    ]
    rows: list[list[str]] = []
    for r in results:
        name = r.get("model_name", "?")
        if "error" in r:
            rows.append([name, r["error"], "—", "—", "—", "—", "—", "—", "—"])
            continue
        parity = r.get("parity", {})
        onnx = r.get("onnx") or {}
        torch_r = r.get("torch") or {}
        # latency from the ONNX (serving) path; KPIs prefer ONNX, fall back to torch
        kpi = onnx if onnx else torch_r
        rows.append([
            name,
            _fmt(parity.get("test_f1")),
            _fmt(parity.get("delta_f1_vs_teacher")),
            _fmt(parity.get("test_mae")),
            _fmt(onnx.get("router_latency_ms") if onnx else torch_r.get("router_latency_ms"), 2),
            _fmt(kpi.get("cost_saving_pct"), 1),
            _fmt(kpi.get("quality_loss_pct"), 1),
            _fmt(kpi.get("avg_acc")),
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
        description="Evaluate all distilled students (parity + routing + latency)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--students-root", default="runs/students",
                    help="Root dir of student checkpoints (default: runs/students)")
    ap.add_argument("--teachers-root", default="runs/teachers",
                    help="Root dir of teacher checkpoints, for parity baseline (default: runs/teachers)")
    ap.add_argument("--out-root", default=None,
                    help="Output root for eval reports (default: <student dir>/eval-<run-name>/)")
    ap.add_argument("--run-name", default="baseline",
                    help="Tag appended to per-student eval output dir (default: baseline)")
    ap.add_argument("--students", nargs="+", default=None, metavar="STUDENT",
                    help="Which students to evaluate (default: all 3)")
    ap.add_argument("--testset", default="data/eval/routing_testset.jsonl",
                    help="routing testset for eval_router.py (the 'eval set')")
    ap.add_argument("--csv", default="data/eval/intern_data.csv",
                    help="production logs CSV for eval_logs.py (the 'logs')")
    ap.add_argument("--python", default=None, metavar="CMD",
                    help="Python interpreter (default: uv run --extra ml python)")
    ap.add_argument("--schema-version", default=None, metavar="VERSION")
    ap.add_argument("--mlflow-experiment", default="vi-smart-routing")
    ap.add_argument("--mlflow-tracking-uri", default=None)
    ap.add_argument("--no-mlflow", action="store_true",
                    help="skip MLflow logging (for an aggregation-only comparison pass)")
    args = ap.parse_args()

    students_root = Path(args.students_root)
    teachers_root = Path(args.teachers_root)
    python_cmd = args.python.split() if args.python else ["uv", "run", "--extra", "ml", "python"]
    env = _build_env()
    run_tag = args.run_name

    pairs = PAIRS
    if args.students:
        wanted = set(args.students)
        pairs = [(s, t) for (s, t) in PAIRS if s in wanted]
        if not pairs:
            sys.exit(f"no known students match {args.students}; choices: {[s for s, _ in PAIRS]}")

    print(f"Students root : {students_root}")
    print(f"Teachers root : {teachers_root}")
    print(f"Run name      : {run_tag}")
    print(f"Eval set      : {args.testset}")
    print(f"Logs CSV      : {args.csv}\n")

    results = []
    for student, teacher in pairs:
        ckpt_dir = students_root / student
        out_dir = (Path(args.out_root) / student) if args.out_root else (ckpt_dir / f"eval-{run_tag}")
        print(f"=== {student}  (teacher: {teacher}) ===", flush=True)
        r = run_eval(
            student, teacher, students_root, teachers_root, out_dir, args.testset, args.csv,
            python_cmd, env, schema_version=args.schema_version,
            experiment=args.mlflow_experiment, tracking_uri=args.mlflow_tracking_uri,
            run_tag=run_tag, log_mlflow=not args.no_mlflow,
        )
        results.append(r)
        print()

    comparison_dir = Path(args.out_root) if args.out_root else students_root
    comparison_dir.mkdir(parents=True, exist_ok=True)
    report_path = comparison_dir / f"student-eval-comparison-{run_tag}.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n=== Student Eval Comparison ===\n")
    print_table(results)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
