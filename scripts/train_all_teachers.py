#!/usr/bin/env python3
"""Train all 4 teacher models in parallel on an H200 GPU server.

Prerequisites on the H200:
  1. Clone the repo:
       git clone https://github.com/LeDucDiLac/vietnamese-ai-smart-routing.git
       cd vietnamese-ai-smart-routing

  2. Download the dataset (uploaded from the local machine via upload_dataset.py):
       kaggle datasets download duckgotsick/ai-smart-routing-dataset --unzip -p ~/data/ai-smart-routing/

  3. Run this script:
       python scripts/train_all_teachers.py --data-root ~/data/ai-smart-routing/

Each model trains in its own subprocess sharing the same GPU. Logs are saved to
--out-root/<model_name>.log. A comparison table is printed and saved at the end.

H200 tips:
  --batch-size 64    safe for all 4 models simultaneously on 141GB VRAM
  --epochs 5         more epochs if time allows (baseline was 3)
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

TEACHER_MODELS = [
    "vi-router-quality",         # mDeBERTa-v3-base 278M — current baseline
    "vi-router-quality-mmbert",  # mmBERT-base 276M — RoPE, 2× faster
    "vi-router-quality-bgem3",   # BGE-M3 568M — MTEB leader
    "vi-router-quality-granite", # Granite 311M — ModernBERT, 32k ctx
]


def _pip_install() -> None:
    pkgs = [
        "transformers>=4.40",
        "sentencepiece>=0.2",
        "datasets>=2.19",
        "huggingface-hub>=0.23",
        "pydantic>=2",
        "pyyaml",
        "tqdm",
    ]
    print(f"[install] pip install {' '.join(pkgs)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _check_data(data_root: Path) -> None:
    missing = [f for f in ("train.jsonl", "val.jsonl", "test.jsonl") if not (data_root / f).exists()]
    if missing:
        print(f"ERROR: missing files in {data_root}: {missing}", file=sys.stderr)
        print("Download the dataset first:", file=sys.stderr)
        print("  kaggle datasets download duckgotsick/ai-smart-routing-dataset --unzip -p <data-root>", file=sys.stderr)
        sys.exit(1)
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
        n = sum(1 for line in (data_root / fname).open(encoding="utf-8") if line.strip())
        print(f"  {fname}: {n:,} rows")


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VI_ROUTER_REPO_ROOT"] = str(REPO)
    # Prepend src/ so `from classifier.train import ...` and `from config import ...` resolve.
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def launch_all(
    models: list[str],
    data_root: Path,
    out_root: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_steps: int | None,
    schema_version: str | None,
    no_pretrained: bool = False,
) -> list[tuple]:
    """Start one subprocess per model. Returns list of (name, proc, log_fh, log_path, out_dir, t0)."""
    env = _build_env()
    launched = []

    for model_name in models:
        out_dir = out_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_root / f"{model_name}.log"

        cmd = [
            "uv", "run", "--extra", "ml",
            "python", "-m", "classifier.train",
            "--model", model_name,
            "--data", str(data_root),
            "--out", str(out_dir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
        ]
        if max_steps is not None:
            cmd += ["--max-steps", str(max_steps)]
        if schema_version is not None:
            cmd += ["--schema-version", schema_version]
        if no_pretrained:
            cmd += ["--no-pretrained"]

        log_fh = log_path.open("w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(REPO),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        print(f"  [PID {proc.pid:>6}] {model_name}  ->  {log_path.name}", flush=True)
        launched.append((model_name, proc, log_fh, log_path, out_dir, time.time()))

    return launched


def wait_all(launched: list[tuple]) -> list[dict]:
    """Poll until every process finishes. Returns list of meta dicts."""
    results: list[dict] = []
    pending = list(launched)
    t_start = time.time()

    try:
        while pending:
            time.sleep(15)
            still_pending = []
            for entry in pending:
                model_name, proc, log_fh, log_path, out_dir, t0 = entry
                retcode = proc.poll()
                if retcode is None:
                    still_pending.append(entry)
                    continue

                log_fh.close()
                elapsed = time.time() - t0

                if retcode != 0:
                    print(f"  [FAIL] {model_name}  exit={retcode}  ({elapsed / 60:.1f}m)  log: {log_path.name}", flush=True)
                    results.append({"model_name": model_name, "error": f"exit {retcode}", "elapsed_s": elapsed})
                else:
                    print(f"  [DONE] {model_name}  ({elapsed / 60:.1f}m)", flush=True)
                    meta_path = out_dir / "meta.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta["elapsed_s"] = elapsed
                        results.append(meta)
                    else:
                        results.append({"model_name": model_name, "error": "no meta.json", "elapsed_s": elapsed})

            if still_pending:
                names = ", ".join(e[0] for e in still_pending)
                print(f"  [{(time.time() - t_start) / 60:.0f}m elapsed] running: {names}", flush=True)

            pending = still_pending

    except KeyboardInterrupt:
        print("\nInterrupted — terminating all training processes...", file=sys.stderr, flush=True)
        for model_name, proc, log_fh, _, _, _ in pending:
            proc.terminate()
            log_fh.close()
            print(f"  killed {model_name} (PID {proc.pid})", file=sys.stderr)
        sys.exit(1)

    return results


def _fmt_float(value: object, decimals: int = 4) -> str:
    try:
        return f"{float(value):.{decimals}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"


def print_table(results: list[dict]) -> None:
    # Sort by val macro-F1 descending; errors last
    def sort_key(r: dict) -> float:
        return float(r.get("task_macro_f1", -1))

    sorted_results = sorted(results, key=sort_key, reverse=True)

    headers = [
        "Model", "Val F1", "Val Acc", "Test F1", "Test Acc", "Cplx MAE", "Time (m)",
    ]

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
        description="Train all teacher models in parallel on a GPU server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--data-root", required=True,
        help="Directory containing train.jsonl / val.jsonl / test.jsonl",
    )
    ap.add_argument(
        "--out-root", default="runs/teachers",
        help="Root output directory for checkpoints and logs (default: runs/teachers)",
    )
    ap.add_argument(
        "--models", nargs="+", default=TEACHER_MODELS,
        metavar="MODEL",
        help="Which teacher models to train (default: all 4)",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--batch-size", type=int, default=32,
        help="Per-model batch size — 32 is safe for 4 concurrent jobs on H200 (default: 32)",
    )
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=None, help="Cap steps per model (smoke test)")
    ap.add_argument(
        "--schema-version", default=None, metavar="VERSION",
        help="Label schema version, e.g. 'v2' → configs/schemas/v2.yaml. Required when training on v2 data.",
    )
    ap.add_argument("--no-pretrained", action="store_true", help="skip backbone weight download (smoke test only)")
    ap.add_argument("--install", action="store_true", help="pip-install ML deps before training")
    args = ap.parse_args()

    if args.install:
        _pip_install()

    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\nData root : {data_root}")
    print(f"Out root  : {out_root}")
    print(f"Models    : {args.models}")
    print(f"Epochs    : {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    if args.max_steps:
        print(f"Max steps : {args.max_steps}")

    print("\nValidating dataset...")
    _check_data(data_root)

    if args.schema_version:
        print(f"Schema:   v{args.schema_version} (configs/schemas/{args.schema_version}.yaml)")
    else:
        print("Schema:   default (configs/label_schema.yaml)")

    print(f"\nLaunching {len(args.models)} training jobs in parallel...")
    launched = launch_all(
        args.models, data_root, out_root,
        args.epochs, args.batch_size, args.lr, args.max_steps,
        args.schema_version,
        no_pretrained=args.no_pretrained,
    )

    print("\nAll jobs running — polling every 15s...")
    results = wait_all(launched)

    # Save comparison JSON
    comp_path = out_root / "teacher_comparison.json"
    comp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nComparison saved -> {comp_path}")
    print("\n=== Teacher Model Comparison ===\n")
    print_table(results)
    print()


if __name__ == "__main__":
    main()
