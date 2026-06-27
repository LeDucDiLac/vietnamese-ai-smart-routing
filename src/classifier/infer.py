"""Inference wrappers for the classifier.

Two backends behind one interface:

- :class:`TorchClassifier` — loads a trained ``CustomModel`` checkpoint (quality
  path, or fast path before ONNX export). Needs the ``ml`` extra.
- :class:`OnnxClassifier` — loads the INT8 ONNX export and runs on CPU via
  onnxruntime. This is the <=50ms serving path (plan §5). Needs ``onnxruntime``.

Both return the same NVIDIA-style dict so the router doesn't care which ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from config import ComplexityConfig, LabelSchema, load_complexity, load_label_schema

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _schema_from_meta(
    checkpoint_dir: Path,
    schema_version: str | None = None,
) -> tuple[LabelSchema, ComplexityConfig]:
    """Return (schema, complexity) for a checkpoint dir.

    Priority: explicit schema_version arg > meta.json > default (v1).
    """
    version = schema_version
    if not version:
        meta_path = checkpoint_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            v = meta.get("schema_version")
            if v and v != "default":
                version = v
    schema = load_label_schema(version=version)
    if version:
        complexity_path = str(_CONFIGS_DIR / "schemas" / f"{version}-complexity.yaml")
        complexity = load_complexity(path=complexity_path)
    else:
        complexity = load_complexity()
    return schema, complexity


class Classifier(Protocol):
    """Common interface: text(s) in, NVIDIA-style prediction dict(s) out."""

    def predict(self, prompts: list[str]) -> list[dict[str, Any]]: ...


def _scores_to_record(
    task_logits: list[float],
    dim_values: dict[str, float],
    schema: LabelSchema,
    complexity: ComplexityConfig,
) -> dict[str, Any]:
    """Shared post-processing: logits + dim regressions -> NVIDIA-style record."""
    # softmax over task logits
    import math

    m = max(task_logits)
    exps = [math.exp(x - m) for x in task_logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    idx1 = order[0]
    idx2 = order[1] if len(order) > 1 else order[0]

    return {
        "task_type_1": schema.task_types[idx1],
        "task_type_2": schema.task_types[idx2],
        "task_type_prob": probs[idx1],
        **dim_values,
        "prompt_complexity_score": complexity.score(dim_values),
    }


class TorchClassifier:
    """Trained PyTorch ``CustomModel`` for inference (quality path)."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_size: str = "vi-router-quality",
        schema_version: str | None = None,
        device: str | None = None,
    ):
        import torch

        from config import load_model_configs
        from classifier.model import CustomModel, ModelSpec
        from classifier.tokenization import build_tokenizer

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.schema, self.complexity = _schema_from_meta(Path(checkpoint_dir), schema_version)
        spec = ModelSpec.from_config(load_model_configs()[model_size])
        self.spec = spec
        # Prefer the saved tokenizer in the checkpoint dir to avoid HF network calls.
        local_tok = Path(checkpoint_dir) / "tokenizer"
        tok_source = str(local_tok) if local_tok.exists() else spec.backbone
        self.tokenizer = build_tokenizer(tok_source, spec.max_tokens)

        self.model = CustomModel(spec, self.schema, self.complexity, pretrained=False)
        state = torch.load(Path(checkpoint_dir) / "model.pt", map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.float()  # backbone (e.g. Granite) may emit BF16; keep everything in fp32
        self.model.eval()

    def predict(self, prompts: list[str]) -> list[dict[str, Any]]:
        enc = self.tokenizer(prompts)
        ids = self._torch.tensor(enc["input_ids"]).to(self.device)
        mask = self._torch.tensor(enc["attention_mask"]).to(self.device)
        return self.model.predict(ids, mask)


class OnnxClassifier:
    """INT8 ONNX export for the CPU <=50ms serving path."""

    def __init__(
        self,
        onnx_path: str | Path,
        backbone: str,
        max_tokens: int = 256,
    ):
        import onnxruntime as ort

        from classifier.tokenization import build_tokenizer

        self.schema = load_label_schema()
        self.complexity = load_complexity()
        self.tokenizer = build_tokenizer(backbone, max_tokens)
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, prompts: list[str]) -> list[dict[str, Any]]:
        import numpy as np

        enc = self.tokenizer(prompts)
        feeds = {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
        }
        outputs = self.session.run(self._output_names, feeds)
        named = dict(zip(self._output_names, outputs))

        records: list[dict[str, Any]] = []
        task_logits_batch = named["task_type"]
        for i in range(task_logits_batch.shape[0]):
            dim_values = {
                dim: float(named[dim][i].reshape(-1)[0])
                for dim in self.schema.complexity_dimensions
            }
            records.append(
                _scores_to_record(
                    list(map(float, task_logits_batch[i].reshape(-1))),
                    dim_values,
                    self.schema,
                    self.complexity,
                )
            )
        return records
