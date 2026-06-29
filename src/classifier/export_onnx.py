"""Export a trained CustomModel to ONNX + INT8 dynamic quantization.

This is the bridge from the PyTorch quality/student model to the artifact that
serves the <=50ms CPU path (plan §4, §2). We:

  1. trace the multi-head model to ONNX (dynamic batch + sequence axes),
  2. apply ONNX Runtime dynamic INT8 quantization,
  3. verify PyTorch vs ONNX parity within tolerance.

Torch + onnx + onnxruntime live in the ``ml`` extra. Run on Kaggle/host after
training; the produced ``.onnx`` is what ``src/serving`` loads (onnxruntime only,
no torch needed at serve time).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from classifier.model import CustomModel, ModelSpec
from classifier.tokenization import PromptTokenizer
from config import load_complexity, load_label_schema, load_model_configs


def export(
    checkpoint: str,
    model_name: str,
    out_dir: str,
    *,
    quantize: bool = True,
    opset: int = 17,
    schema_version: str | None = None,
) -> dict[str, str]:
    """Export ``checkpoint`` (a CustomModel state_dict) to ONNX (+INT8).

    ``schema_version`` must match how the checkpoint was trained — a v2 student
    has a 6-class task head + 3 complexity dims, so the model and ONNX output
    names differ from the v1 default.

    Returns a dict of produced artifact paths.
    """
    schema = load_label_schema(version=schema_version)
    if schema_version:
        from config import CONFIGS_DIR
        complexity = load_complexity(
            path=str(CONFIGS_DIR / "schemas" / f"{schema_version}-complexity.yaml")
        )
    else:
        complexity = load_complexity()
    model_cfg = load_model_configs()[model_name]
    spec = ModelSpec.from_config(model_cfg)

    model = CustomModel(spec, schema, complexity, pretrained=False)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp32_path = out / "model.onnx"

    # Dummy inputs for tracing.
    tok = PromptTokenizer(spec.backbone, spec.max_tokens)
    enc = tok(["Xin chào, đây là một prompt mẫu để export ONNX."])
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    output_names = ["task_type"] + list(schema.complexity_dimensions)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        fp32_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    artifacts = {"fp32": fp32_path.as_posix()}

    # Save the tokenizer alongside so serving is self-contained.
    tok.save(out.as_posix())

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out / "model.int8.onnx"
        quantize_dynamic(
            model_input=fp32_path.as_posix(),
            model_output=int8_path.as_posix(),
            weight_type=QuantType.QInt8,
        )
        artifacts["int8"] = int8_path.as_posix()

    # Parity check (fp32 ONNX vs torch).
    _verify_parity(model, fp32_path.as_posix(), input_ids, attention_mask, output_names)

    return artifacts


def _verify_parity(
    model: CustomModel,
    onnx_path: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_names: list[str],
    tol: float = 1e-3,
) -> None:
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = model(input_ids, attention_mask)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(
        output_names,
        {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": attention_mask.numpy().astype(np.int64),
        },
    )
    for name, arr in zip(output_names, onnx_out):
        diff = float(np.abs(torch_out[name].numpy() - arr).max())
        if diff > tol:
            raise AssertionError(f"ONNX parity failed for {name!r}: max diff {diff} > {tol}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export CustomModel to ONNX + INT8")
    ap.add_argument("--checkpoint", required=True, help="path to trained state_dict (.pt)")
    ap.add_argument("--model-name", default="vi-router-fast", help="key in configs/model.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--schema-version", default=None,
                    help="label schema version, e.g. 'v2'. Must match the checkpoint.")
    args = ap.parse_args()

    artifacts = export(
        args.checkpoint,
        args.model_name,
        args.out_dir,
        quantize=not args.no_quantize,
        schema_version=args.schema_version,
    )
    print("Exported artifacts:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
