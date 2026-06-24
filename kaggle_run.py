#!/usr/bin/env python
"""Kaggle entrypoint for the GPU pipeline (train / distill / export).

All GPU-bound work runs on Kaggle, not locally (per project decision). This
single script is the thing you call from a Kaggle notebook cell. It:

1. **bootstraps imports** — puts ``src/`` on ``sys.path`` and pins
   ``VI_ROUTER_REPO_ROOT`` so ``from config import ...`` / ``from classifier
   ...`` resolve no matter the cwd or where the repo dataset is mounted;
2. **optionally installs** the ``ml`` deps (``--install``) — Kaggle images carry
   torch but not always the exact transformers/onnx versions we need;
3. **runs the pipeline stages** you ask for, in order.

Typical Kaggle usage (one notebook cell)
-----------------------------------------
    # repo uploaded as a Kaggle dataset at /kaggle/input/ai-smart-routing
    !cp -r /kaggle/input/ai-smart-routing /kaggle/working/repo
    %cd /kaggle/working/repo
    !python kaggle_run.py --install --steps all --epochs 3

Or run individual stages:
    !python kaggle_run.py --steps synth dataset           # CPU, no GPU needed
    !python kaggle_run.py --steps train distill export     # the GPU stages

Outputs land under ``--out-root`` (default ``/kaggle/working/runs`` on Kaggle,
``runs/`` locally) so they persist in the notebook's working dir for download.

Every stage is also runnable standalone via ``python -m <module>``; this script
just wires them together with Kaggle-friendly defaults and path handling.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make the package importable + pin the repo root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Pin repo root so config.py finds configs/ regardless of cwd / Kaggle mount.
os.environ.setdefault("VI_ROUTER_REPO_ROOT", str(REPO_ROOT))


ON_KAGGLE = Path("/kaggle/working").is_dir()
DEFAULT_OUT_ROOT = "/kaggle/working/runs" if ON_KAGGLE else "runs"
DEFAULT_DATA_ROOT = "/kaggle/working/data" if ON_KAGGLE else "data"

ALL_STEPS = ["synth", "dataset", "train", "distill", "export", "leaderboard", "simulate"]


def _log(msg: str) -> None:
    print(f"\n=== [kaggle_run] {msg} ===", flush=True)


def _pip_install_ml() -> None:
    """Install the ``ml`` extra into the Kaggle kernel.

    Kaggle images ship torch already; we only top up the rest so we don't fight
    the pre-installed CUDA torch build.
    """
    pkgs = [
        "transformers>=4.40",
        "sentencepiece>=0.2",
        "onnx>=1.16",
        "onnxruntime>=1.18",
        "datasets>=2.19",
        "huggingface-hub>=0.23",
    ]
    _log(f"pip install {len(pkgs)} ml deps")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _device_report() -> None:
    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "cpu"
        _log(f"torch {torch.__version__} | cuda={cuda} | device={name}")
    except ImportError:
        _log("torch not installed yet (run with --install, or --steps synth/dataset only)")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def step_synth(args: argparse.Namespace) -> None:
    """Generate synthetic VN prompts (CPU, no deps)."""
    from data.synth_gen import generate

    out = Path(args.data_root) / "synthetic" / "synth.jsonl"
    n = generate(out, per_cell=args.per_cell, mode=args.synth_mode)
    _log(f"synth: wrote {n} rows -> {out}")


def step_dataset(args: argparse.Namespace) -> None:
    """Assemble train/val/test from silver (+ gold if present)."""
    from data.build_dataset import build

    data_root = Path(args.data_root)
    silver = [str(data_root / "synthetic" / "synth.jsonl")]
    # include teacher-labeled crawl output if it exists
    labeled = data_root / "raw" / "labeled.jsonl"
    if labeled.exists():
        silver.append(str(labeled))
    gold_path = data_root / "gold" / "gold.jsonl"
    gold = [str(gold_path)] if gold_path.exists() else []

    out = data_root / "processed"
    counts = build(silver, gold, out)
    _log(f"dataset: {counts} -> {out}")
    if counts["train"] == 0:
        raise RuntimeError("no training rows produced — run --steps synth first")


def step_train(args: argparse.Namespace) -> None:
    """Train vi-router-quality (GPU)."""
    from classifier.train import resolve_data_dir, train

    data_dir = resolve_data_dir(Path(args.data_root) / "processed")
    out = Path(args.out_root) / "quality"
    meta = train(
        "vi-router-quality",
        data_dir,
        out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
    )
    _log(f"train: {meta} -> {out}")


def step_distill(args: argparse.Namespace) -> None:
    """Distill quality -> vi-router-fast student (GPU)."""
    from classifier.distill import distill
    from classifier.train import resolve_data_dir

    data_dir = resolve_data_dir(Path(args.data_root) / "processed")
    teacher = Path(args.out_root) / "quality"
    out = Path(args.out_root) / "fast"
    meta = distill(
        teacher,
        out,
        data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size * 2,
        max_steps=args.max_steps,
    )
    _log(f"distill: {meta} -> {out}")


def step_export(args: argparse.Namespace) -> None:
    """Export the fast student to ONNX + INT8 (CPU-friendly)."""
    from classifier.export_onnx import export

    fast_dir = Path(args.out_root) / "fast"
    ckpt = fast_dir / "model.pt"
    if not ckpt.exists():
        raise RuntimeError(f"no student checkpoint at {ckpt}; run --steps distill first")
    out = Path(args.out_root) / "onnx"
    artifacts = export(str(ckpt), "vi-router-fast", str(out), quantize=True)
    _log(f"export: {artifacts}")


def step_leaderboard(args: argparse.Namespace) -> None:
    """Refresh the AI capability leaderboard (CPU, no GPU). Success criterion #1."""
    from router.leaderboard import refresh

    out = Path(args.out_root) / "leaderboard"
    artifacts = refresh(out)
    _log(f"leaderboard: {artifacts}")


def step_simulate(args: argparse.Namespace) -> None:
    """Run the offline routing eval (CPU, no GPU). Builds a VN cache if missing."""
    from sim.vi_response_cache import build as build_cache
    from tests.eval.simulate import simulate
    from dataclasses import asdict
    import json

    from classifier.train import resolve_data_dir

    data_root = Path(args.data_root)
    data_dir = resolve_data_dir(data_root / "processed")
    val = data_dir / "val.jsonl"
    train_jsonl = data_dir / "train.jsonl"
    # prefer val; fall back to train so the smoke path always has prompts
    prompts = val if val.exists() and val.stat().st_size > 0 else train_jsonl
    cache = Path(args.out_root) / "sim" / "vi_cache.jsonl"
    build_cache(prompts, cache, mode="synthetic", limit=args.sim_limit)

    # Default to a caller that can reach every model (premium + engineering),
    # so the cost/quality tradeoff is measured against the full registry rather
    # than being skewed by permission filtering. Override with --user-groups.
    groups = args.user_groups if args.user_groups is not None else ["premium", "engineering"]
    report = simulate(cache, user_groups=groups)
    out = Path(args.out_root) / "eval" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    _log(f"simulate: targets={report.targets} -> {out}")


STEP_FUNCS = {
    "synth": step_synth,
    "dataset": step_dataset,
    "train": step_train,
    "distill": step_distill,
    "export": step_export,
    "leaderboard": step_leaderboard,
    "simulate": step_simulate,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle GPU pipeline runner")
    ap.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help="stages to run, in order; 'all' = " + " ".join(ALL_STEPS),
    )
    ap.add_argument("--install", action="store_true", help="pip-install ml deps first")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    # synth / dataset
    ap.add_argument("--per-cell", type=int, default=60, help="synth rows per task x tier")
    ap.add_argument("--synth-mode", choices=["template", "llm"], default="template")
    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke test)")
    # simulate
    ap.add_argument("--sim-limit", type=int, default=None)
    ap.add_argument("--user-groups", nargs="*", default=None)
    args = ap.parse_args()

    steps = ALL_STEPS if "all" in args.steps else args.steps
    unknown = [s for s in steps if s not in STEP_FUNCS]
    if unknown:
        ap.error(f"unknown steps: {unknown}; choose from {ALL_STEPS}")

    if args.install:
        _pip_install_ml()
    _device_report()

    _log(f"running steps: {steps}")
    for step in steps:
        STEP_FUNCS[step](args)
    _log("done")


if __name__ == "__main__":
    main()
