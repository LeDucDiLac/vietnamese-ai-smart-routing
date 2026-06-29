#!/usr/bin/env python3
"""Distill all student models from their pre-trained teachers on a GPU server.

Mirrors train_all_teachers.py, but for the distillation phase. Each student is
distilled from its matching teacher checkpoint (the v2 winners trained on the
H200, sitting under --teachers-root). After distillation, each student is
optionally exported to INT8 ONNX for the CPU serving path (--export).

Prerequisites on the H200:
  1. Teacher checkpoints present under --teachers-root, one dir per teacher:
       runs/teachers/vi-router-quality-granite/{model.pt,meta.json,tokenizer/}
       runs/teachers/vi-router-quality-mmbert/...
       runs/teachers/vi-router-quality/...           (mDeBERTa v1)
  2. Dataset (train/val/test.jsonl) under --data-root.

Run:
       python scripts/distill_all_students.py \
           --teachers-root runs/teachers --data-root ~/data/ai-smart-routing/ --export

Each student distills in its own subprocess sharing the GPU. Logs land under
--out-root/<student_name>.log. A comparison table is printed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# (teacher dir under --teachers-root, student model_name in configs/model.yaml)
PAIRS = [
    ("vi-router-quality-granite", "vi-router-fast-granite"),  # Granite 311m → 97m (v2 winner)
    ("vi-router-quality-mmbert", "vi-router-fast-mmbert"),    # mmBERT base → small (v2 winner)
    ("vi-router-quality", "vi-router-fast"),                  # mDeBERTa → MiniLM (v1 baseline)
]


def _check_data(data_root: Path) -> None:
    missing = [f for f in ("train.jsonl", "val.jsonl", "test.jsonl") if not (data_root / f).exists()]
    if missing:
        print(f"ERROR: missing files in {data_root}: {missing}", file=sys.stderr)
        sys.exit(1)
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
        n = sum(1 for line in (data_root / fname).open(encoding="utf-8") if line.strip())
        print(f"  {fname}: {n:,} rows")


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VI_ROUTER_REPO_ROOT"] = str(REPO)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _runner(python_cmd: list[str] | None) -> list[str]:
    return python_cmd if python_cmd else ["uv", "run", "--extra", "ml", "python"]


def _make_cmd(
    teacher_dir: str,
    student_name: str,
    teachers_root: Path,
    data_root: Path,
    out_root: Path,
    *,
    epochs: int,
    batch_size: int,
    temperature: float,
    alpha: float,
    max_steps: int | None,
    no_pretrained: bool,
    schema_version: str | None,
    python_cmd: list[str] | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str | None,
) -> tuple[list[str], Path, Path]:
    out_dir = out_root / student_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"{student_name}.log"
    cmd = [
        *_runner(python_cmd), "-m", "classifier.distill",
        "--teacher", str(teachers_root / teacher_dir),
        "--out", str(out_dir),
        "--student", student_name,
        "--data", str(data_root),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--temperature", str(temperature),
        "--alpha", str(alpha),
        "--mlflow-experiment", mlflow_experiment,
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if no_pretrained:
        cmd += ["--no-pretrained"]
    if schema_version:
        cmd += ["--schema-version", schema_version]
    if mlflow_tracking_uri:
        cmd += ["--mlflow-tracking-uri", mlflow_tracking_uri]
    return cmd, log_path, out_dir


def _launch_one(student_name: str, cmd: list[str], log_path: Path, env: dict) -> tuple:
    log_fh = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, env=env, cwd=str(REPO), stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"  [PID {proc.pid:>6}] {student_name}  ->  {log_path.name}", flush=True)
    return (student_name, proc, log_fh, log_path, time.time())


def _collect(entry: tuple, out_root: Path) -> dict:
    student_name, proc, log_fh, log_path, t0 = entry
    log_fh.close()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [FAIL] {student_name}  exit={proc.returncode}  ({elapsed / 60:.1f}m)  log: {log_path.name}", flush=True)
        return {"model_name": student_name, "error": f"exit {proc.returncode}", "elapsed_s": elapsed}
    print(f"  [DONE] {student_name}  ({elapsed / 60:.1f}m)", flush=True)
    meta_path = out_root / student_name / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["elapsed_s"] = elapsed
        return meta
    return {"model_name": student_name, "error": "no meta.json", "elapsed_s": elapsed}


def _export_one(student_name: str, out_root: Path, env: dict, python_cmd: list[str] | None,
                schema_version: str | None) -> None:
    out_dir = out_root / student_name
    ckpt = out_dir / "model.pt"
    if not ckpt.exists():
        print(f"  [SKIP export] {student_name} — no model.pt", flush=True)
        return
    onnx_dir = out_dir / "onnx"
    cmd = [
        *_runner(python_cmd), "-m", "classifier.export_onnx",
        "--checkpoint", str(ckpt),
        "--model-name", student_name,
        "--out-dir", str(onnx_dir),
    ]
    if schema_version:
        cmd += ["--schema-version", schema_version]
    log_path = out_root / f"{student_name}.export.log"
    with log_path.open("w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, env=env, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
    status = "DONE" if rc == 0 else f"FAIL exit={rc}"
    print(f"  [export {status}] {student_name}  ->  {onnx_dir}  (log: {log_path.name})", flush=True)


def run_all(
    pairs: list[tuple[str, str]],
    teachers_root: Path,
    data_root: Path,
    out_root: Path,
    *,
    epochs: int,
    batch_size: int,
    temperature: float,
    alpha: float,
    max_steps: int | None,
    no_pretrained: bool,
    schema_version: str | None,
    max_parallel: int,
    python_cmd: list[str] | None,
    mlflow_experiment: str,
    mlflow_tracking_uri: str | None,
) -> list[dict]:
    env = _build_env()
    queue: list[tuple] = []
    results: list[dict] = []
    for teacher_dir, student_name in pairs:
        tdir = teachers_root / teacher_dir
        if not (tdir / "model.pt").exists() or not (tdir / "meta.json").exists():
            print(f"  [SKIP] {student_name} — teacher checkpoint missing at {tdir}", flush=True)
            results.append({"model_name": student_name, "error": "teacher not found"})
            continue
        cmd, log_path, _ = _make_cmd(
            teacher_dir, student_name, teachers_root, data_root, out_root,
            epochs=epochs, batch_size=batch_size, temperature=temperature, alpha=alpha,
            max_steps=max_steps, no_pretrained=no_pretrained, schema_version=schema_version,
            python_cmd=python_cmd,
            mlflow_experiment=mlflow_experiment, mlflow_tracking_uri=mlflow_tracking_uri,
        )
        queue.append((student_name, cmd, log_path))

    running: list[tuple] = []
    t_start = time.time()
    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                student_name, cmd, log_path = queue.pop(0)
                running.append(_launch_one(student_name, cmd, log_path, env))

            time.sleep(15)

            still_running = []
            for entry in running:
                if entry[1].poll() is None:
                    still_running.append(entry)
                else:
                    results.append(_collect(entry, out_root))
            running = still_running

            if running:
                names = ", ".join(e[0] for e in running)
                queued = ", ".join(j[0] for j in queue)
                status = f"running: {names}"
                if queued:
                    status += f"  |  queued: {queued}"
                print(f"  [{(time.time() - t_start) / 60:.0f}m] {status}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — terminating...", file=sys.stderr, flush=True)
        for student_name, proc, log_fh, _, _ in running:
            proc.terminate()
            log_fh.close()
            print(f"  killed {student_name} (PID {proc.pid})", file=sys.stderr)
        sys.exit(1)

    return results


def _fmt_float(value: object, decimals: int = 4) -> str:
    try:
        return f"{float(value):.{decimals}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"


def print_table(results: list[dict]) -> None:
    def sort_key(r: dict) -> float:
        return float(r.get("task_macro_f1", -1))

    sorted_results = sorted(results, key=sort_key, reverse=True)
    headers = ["Student", "Val F1", "Val Acc", "Test F1", "Test Acc", "Cplx MAE", "Time (m)"]
    rows: list[list[str]] = []
    for r in sorted_results:
        name = r.get("model_name", "?")
        if "error" in r:
            rows.append([name, "ERROR", r["error"], "—", "—", "—", _fmt_float(r.get("elapsed_s", 0) / 60, 1)])
        else:
            rows.append([
                name,
                _fmt_float(r.get("task_macro_f1")),
                _fmt_float(r.get("task_top1_acc")),
                _fmt_float(r.get("test_task_macro_f1")),
                _fmt_float(r.get("test_task_top1_acc")),
                _fmt_float(r.get("complexity_mae")),
                _fmt_float(r.get("elapsed_s", 0) / 60, 1),
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
        description="Distill all student models from their teachers on a GPU server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--data-root", required=True, help="Dir with train/val/test.jsonl")
    ap.add_argument("--teachers-root", default="runs/teachers", help="Root dir of teacher checkpoints (default: runs/teachers)")
    ap.add_argument("--out-root", default="runs/students", help="Root output dir for students (default: runs/students)")
    ap.add_argument(
        "--students", nargs="+", default=None, metavar="STUDENT",
        help="Which student model_names to distill (default: all 3 pairs)",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64, help="Per-student batch size (students are small; default 64)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="distillation vs hard-label weight")
    ap.add_argument("--max-steps", type=int, default=None, help="Cap steps per student (smoke test)")
    ap.add_argument("--max-parallel", type=int, default=3, metavar="N", help="max students running at once (default 3)")
    ap.add_argument("--export", action="store_true", help="export each student to INT8 ONNX after distillation")
    ap.add_argument("--schema-version", default=None, metavar="VERSION",
                    help="label schema version, e.g. 'v2'. Must match the teachers (the v2 winners use 'v2').")
    ap.add_argument("--python", default=None, metavar="CMD", help="Python interpreter, e.g. 'python3' (default: uv run --extra ml python)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip backbone download (smoke test only)")
    ap.add_argument("--mlflow-experiment", default="vi-smart-routing")
    ap.add_argument("--mlflow-tracking-uri", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    teachers_root = Path(args.teachers_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = PAIRS
    if args.students:
        wanted = set(args.students)
        pairs = [(t, s) for (t, s) in PAIRS if s in wanted]
        if not pairs:
            sys.exit(f"no known students match {args.students}; choices: {[s for _, s in PAIRS]}")

    print(f"\nData root     : {data_root}")
    print(f"Teachers root : {teachers_root}")
    print(f"Out root      : {out_root}")
    print(f"Pairs         : {[(t, s) for t, s in pairs]}")
    print(f"Epochs        : {args.epochs}  |  Batch: {args.batch_size}  |  T={args.temperature}  alpha={args.alpha}")
    print(f"Schema        : {args.schema_version or 'default (v1)'}")
    print(f"Parallel      : {args.max_parallel}  |  Export ONNX: {args.export}")

    print("\nValidating dataset...")
    _check_data(data_root)

    python_cmd = args.python.split() if args.python else None
    print(f"\nLaunching {len(pairs)} distillation jobs (max {args.max_parallel} concurrent)...")
    results = run_all(
        pairs, teachers_root, data_root, out_root,
        epochs=args.epochs, batch_size=args.batch_size, temperature=args.temperature,
        alpha=args.alpha, max_steps=args.max_steps, no_pretrained=args.no_pretrained,
        schema_version=args.schema_version,
        max_parallel=args.max_parallel, python_cmd=python_cmd,
        mlflow_experiment=args.mlflow_experiment, mlflow_tracking_uri=args.mlflow_tracking_uri,
    )

    if args.export:
        print("\nExporting students to INT8 ONNX...")
        env = _build_env()
        for r in results:
            if "error" not in r:
                _export_one(r["model_name"], out_root, env, python_cmd, args.schema_version)

    comp_path = out_root / "student_comparison.json"
    comp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nComparison saved -> {comp_path}")
    print("\n=== Student Model Comparison ===\n")
    print_table(results)
    print()


if __name__ == "__main__":
    main()
