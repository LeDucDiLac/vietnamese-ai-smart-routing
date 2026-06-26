"""Training loop for the multi-head classifier (plan §4).

Multi-task objective: cross-entropy on ``task_type`` + SmoothL1 on each of the 6
complexity regression heads, summed with per-task weights. Reads the processed
dataset produced by ``src/data/build_dataset.py``.

Heavy ML deps (torch/transformers) — only runs with the ``ml`` extra installed.
Invoke as a module:

    uv run --extra ml python -m classifier.train \
        --model vi-router-quality \
        --data data/processed --out runs/quality

The loop is intentionally framework-light (plain torch) so it runs on CPU for a
smoke test and scales to GPU without code changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from classifier.model import CustomModel, ModelSpec
from classifier.tokenization import PromptTokenizer
from config import (
    LabelSchema,
    load_complexity,
    load_label_schema,
    load_model_configs,
)


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve the processed-data directory, tolerating Kaggle layouts.

    On Kaggle the dataset is mounted read-only under ``/kaggle/input/<name>/``.
    If the given path has no ``train.jsonl`` but a unique match exists under
    ``/kaggle/input``, use that instead so the same command line works locally
    and on Kaggle without edits.
    """
    p = Path(data_dir)
    if (p / "train.jsonl").exists():
        return p
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        hits = sorted(kaggle_input.glob("**/train.jsonl"))
        if len(hits) == 1:
            return hits[0].parent
    return p  # leave as-is; PromptDataset will raise a clear error


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Example:
    text: str
    task_type: str  # display label or id
    dims: dict[str, float]


class PromptDataset(Dataset):
    """Reads a JSONL file of labeled prompts.

    Each line: {"text": ..., "task_type": ..., "creativity_scope": ..., ...}
    """

    def __init__(self, path: str | Path, schema: LabelSchema):
        self.schema = schema
        self.examples: list[Example] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.examples.append(
                    Example(
                        text=row["text"],
                        task_type=row["task_type"],
                        dims={
                            d: float(row.get(d, 0.0))
                            for d in schema.complexity_dimensions
                        },
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def make_collate(tokenizer: PromptTokenizer, schema: LabelSchema):
    dims = schema.complexity_dimensions

    def collate(batch: list[Example]) -> dict[str, Any]:
        enc = tokenizer([e.text for e in batch])
        task_idx = torch.tensor(
            [schema.task_index(e.task_type) for e in batch], dtype=torch.long
        )
        dim_targets = {
            d: torch.tensor([[e.dims[d]] for e in batch], dtype=torch.float)
            for d in dims
        }
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "task_idx": task_idx,
            "dim_targets": dim_targets,
        }

    return collate


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class MultiTaskLoss(nn.Module):
    """CE(task_type) + sum_d w_reg * SmoothL1(dim_d)."""

    def __init__(
        self,
        schema: LabelSchema,
        *,
        reg_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.schema = schema
        self.reg_weight = reg_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.reg = nn.SmoothL1Loss()

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        task_loss = self.ce(outputs["task_type"], batch["task_idx"])
        reg_total = outputs["task_type"].new_zeros(())
        for dim in self.schema.complexity_dimensions:
            reg_total = reg_total + self.reg(outputs[dim], batch["dim_targets"][dim])
        loss = task_loss + self.reg_weight * reg_total
        parts = {
            "task_loss": float(task_loss.item()),
            "reg_loss": float(reg_total.item()),
            "loss": float(loss.item()),
        }
        return loss, parts


def compute_class_weights(ds: PromptDataset, schema: LabelSchema) -> torch.Tensor:
    """Inverse-frequency class weights for the rare task types (plan §4)."""
    counts = Counter(schema.task_index(e.task_type) for e in ds.examples)
    n = len(schema.task_types)
    total = sum(counts.values())
    weights = torch.ones(n)
    for i in range(n):
        c = counts.get(i, 0)
        weights[i] = total / (n * c) if c > 0 else 1.0
    return weights


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: CustomModel,
    dl: DataLoader,
    schema: LabelSchema,
    device: str,
) -> dict[str, float]:
    """task_type macro-F1 + top-2 accuracy, and mean MAE over complexity dims."""
    model.eval()
    n_classes = len(schema.task_types)
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    top1_correct = 0
    top2_correct = 0
    total = 0
    mae_sum = 0.0
    mae_count = 0

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        task_idx = batch["task_idx"].to(device)
        out = model(input_ids, attention_mask)
        logits = out["task_type"]
        k = min(2, logits.shape[-1])
        top = torch.topk(logits, k=k, dim=-1).indices
        pred = top[:, 0]
        for i in range(task_idx.shape[0]):
            gold = int(task_idx[i].item())
            p = int(pred[i].item())
            total += 1
            if p == gold:
                top1_correct += 1
                tp[gold] += 1
            else:
                fp[p] += 1
                fn[gold] += 1
            if gold in {int(x) for x in top[i].tolist()}:
                top2_correct += 1
        for dim in schema.complexity_dimensions:
            tgt = batch["dim_targets"][dim].to(device)
            mae_sum += float((out[dim] - tgt).abs().sum().item())
            mae_count += tgt.numel()

    f1s = []
    for c in range(n_classes):
        denom_p = tp[c] + fp[c]
        denom_r = tp[c] + fn[c]
        if denom_p == 0 and denom_r == 0:
            continue  # class absent from this split — skip, don't penalize
        prec = tp[c] / denom_p if denom_p else 0.0
        rec = tp[c] / denom_r if denom_r else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)

    return {
        "task_macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "task_top1_acc": top1_correct / total if total else 0.0,
        "task_top2_acc": top2_correct / total if total else 0.0,
        "complexity_mae": mae_sum / mae_count if mae_count else 0.0,
    }


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train(
    model_name: str,
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    reg_weight: float = 1.0,
    device: str | None = None,
    pretrained: bool = True,
    max_steps: int | None = None,
    schema_version: str | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device == "cuda":
        # Disable cuDNN flash-attention graph executor — it fails under memory pressure.
        # The math backend is slower but stable on all CUDA/cuDNN versions.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        import os as _os
        _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    schema = load_label_schema(version=schema_version)
    # Load the matching complexity config: configs/schemas/<ver>-complexity.yaml if versioned.
    from config import CONFIGS_DIR
    complexity_path = (
        str(CONFIGS_DIR / "schemas" / f"{schema_version}-complexity.yaml")
        if schema_version
        else None
    )
    complexity = load_complexity(path=complexity_path)
    cfg = load_model_configs()[model_name]
    spec = ModelSpec.from_config(cfg)

    tokenizer = PromptTokenizer(spec.backbone, spec.max_tokens)
    collate = make_collate(tokenizer, schema)

    train_ds = PromptDataset(data_dir / "train.jsonl", schema)
    class_weights = compute_class_weights(train_ds, schema).to(device)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    val_path = data_dir / "val.jsonl"
    val_dl: DataLoader | None = None
    if val_path.exists():
        val_ds = PromptDataset(val_path, schema)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = CustomModel(spec, schema, complexity, pretrained=pretrained).float().to(device)
    criterion = MultiTaskLoss(schema, reg_weight=reg_weight, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    use_amp = torch.cuda.is_available() and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: list[dict[str, Any]] = []
    step = 0
    total_batches = len(train_dl)
    total_steps = min(epochs * total_batches, max_steps) if max_steps else epochs * total_batches
    t_train_start = time.time()
    is_interactive = sys.stdout.isatty()
    log_every = 10  # print a structured line every N steps when writing to a log file

    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    def _log(msg: str) -> None:
        print(f"[{_ts()}] {msg}", flush=True)

    _log(f"START  model={model_name}  device={device}  epochs={epochs}  "
         f"batches/epoch={total_batches}  total_steps={total_steps}  bs={batch_size}")

    model.train()
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch", ncols=100,
                     disable=not is_interactive)
    for epoch in epoch_bar:
        _log(f"EPOCH {epoch + 1}/{epochs} begin")
        batch_bar = tqdm(
            train_dl,
            desc=f"  Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=True,
            ncols=100,
            disable=not is_interactive,
        )
        for batch_idx, batch in enumerate(batch_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch["task_idx"] = batch["task_idx"].to(device)
            batch["dim_targets"] = {
                k: v.to(device) for k, v in batch["dim_targets"].items()
            }
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                loss, parts = criterion(outputs, batch)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if is_interactive:
                batch_bar.set_postfix(loss=f"{parts['loss']:.4f}")
            elif step % log_every == 0 or step == 0:
                elapsed = time.time() - t_train_start
                eta_s = (elapsed / (step + 1)) * (total_steps - step - 1)
                _log(
                    f"  E{epoch + 1}/{epochs} | batch {batch_idx + 1}/{total_batches} | "
                    f"step {step + 1}/{total_steps} | "
                    f"loss={parts['loss']:.4f} task={parts['task_loss']:.4f} reg={parts['reg_loss']:.4f} | "
                    f"elapsed={elapsed / 60:.1f}m eta={eta_s / 60:.1f}m"
                )

            if step % 20 == 0:
                parts["epoch"] = epoch
                parts["step"] = step
                history.append(parts)
            step += 1
            if max_steps is not None and step >= max_steps:
                break

        # Validation loss pass
        val_loss = math.nan
        if val_dl is not None:
            _log(f"EPOCH {epoch + 1}/{epochs} val loss pass...")
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            val_bar = tqdm(val_dl, desc="  Val", unit="batch", leave=False, ncols=100,
                           disable=not is_interactive)
            with torch.no_grad():
                for vbatch in val_bar:
                    vbatch["input_ids"] = vbatch["input_ids"].to(device)
                    vbatch["attention_mask"] = vbatch["attention_mask"].to(device)
                    vbatch["task_idx"] = vbatch["task_idx"].to(device)
                    vbatch["dim_targets"] = {
                        k: v.to(device) for k, v in vbatch["dim_targets"].items()
                    }
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        vout = model(vbatch["input_ids"], vbatch["attention_mask"])
                        vloss, _ = criterion(vout, vbatch)
                    val_loss_sum += vloss.item()
                    val_steps += 1
                    if is_interactive:
                        val_bar.set_postfix(loss=f"{val_loss_sum / val_steps:.4f}")
            val_loss = val_loss_sum / val_steps if val_steps else math.nan
            model.train()

        elapsed = time.time() - t_train_start
        steps_remaining = total_steps - step
        eta_s = (elapsed / step * steps_remaining) if step > 0 else 0.0
        last_loss = history[-1]["loss"] if history else math.nan
        if is_interactive:
            epoch_bar.set_postfix(
                loss=f"{last_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                eta=f"{eta_s / 60:.1f}m",
            )
        _log(
            f"EPOCH {epoch + 1}/{epochs} done | "
            f"loss={last_loss:.4f} val_loss={val_loss:.4f} | "
            f"elapsed={elapsed / 60:.1f}m eta={eta_s / 60:.1f}m"
        )
        if history:
            history[-1]["val_loss"] = val_loss
        if max_steps is not None and step >= max_steps:
            break

    # Final validation metrics
    val_metrics: dict[str, Any] = {}
    if val_dl is not None:
        val_metrics = evaluate(model, val_dl, schema, device)
        val_metrics["val_loss"] = history[-1].get("val_loss", math.nan) if history else math.nan

    # Final test metrics (if test.jsonl is present alongside train/val)
    test_metrics: dict[str, Any] = {}
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = PromptDataset(test_path, schema)
        test_dl_eval = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        test_metrics = {f"test_{k}": v for k, v in evaluate(model, test_dl_eval, schema, device).items()}

    # Persist weights + the spec needed to rebuild for inference/distill.
    ckpt = out_dir / "model.pt"
    torch.save(model.state_dict(), ckpt)
    tokenizer.save(str(out_dir / "tokenizer"))
    meta = {
        "model_name": model_name,
        "backbone": spec.backbone,
        "max_tokens": spec.max_tokens,
        "epochs": epochs,
        "steps": step,
        "final_loss": history[-1]["loss"] if history else math.nan,
        **val_metrics,
        **test_metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(h) for h in history), encoding="utf-8"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the VN prompt classifier")
    ap.add_argument("--model", default="vi-router-quality")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/quality")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--reg-weight", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="build backbone from config only (fast smoke test, no download)",
    )
    ap.add_argument(
        "--schema-version",
        default=None,
        help="label schema version to use, e.g. 'v2' → configs/schemas/v2.yaml (default: configs/label_schema.yaml)",
    )
    args = ap.parse_args()

    meta = train(
        args.model,
        args.data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg_weight=args.reg_weight,
        max_steps=args.max_steps,
        pretrained=not args.no_pretrained,
        schema_version=args.schema_version,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
