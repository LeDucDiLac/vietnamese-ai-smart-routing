"""Tokenization helper shared by training, inference and ONNX export.

Wraps the HF tokenizer for the chosen backbone. mDeBERTa-v3 uses a SentencePiece
tokenizer (the ``sentencepiece`` dep in the ml extra), so no word-segmentation
step is required for Vietnamese — unlike PhoBERT, which is one reason we picked
mDeBERTa (see plan §2).
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


class PromptTokenizer:
    def __init__(self, backbone: str, max_tokens: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.max_tokens = max_tokens

    def __call__(self, texts: list[str] | str, **kwargs: Any):
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(
            texts,
            padding=kwargs.pop("padding", True),
            truncation=True,
            max_length=self.max_tokens,
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )

    def save(self, path: str) -> None:
        self.tokenizer.save_pretrained(path)


def build_tokenizer(backbone: str, max_tokens: int = 512) -> PromptTokenizer:
    return PromptTokenizer(backbone, max_tokens)
