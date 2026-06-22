"""Data-pipeline tests: synthetic generation + dataset assembly + validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import load_label_schema
from data.build_dataset import build, validate_row
from data.synth_gen import generate


def test_validate_row_accepts_good_row():
    schema = load_label_schema()
    row = {"text": "Tóm tắt bài viết.", "task_type": "Summarization", "reasoning": 0.4}
    clean = validate_row(row, schema)
    assert clean["task_type"] == "Summarization"
    assert clean["reasoning"] == pytest.approx(0.4)
    # missing dims default to 0
    assert clean["creativity_scope"] == 0.0


def test_validate_row_rejects_bad_task_type():
    schema = load_label_schema()
    with pytest.raises(ValueError):
        validate_row({"text": "hi", "task_type": "NotARealTask"}, schema)


def test_validate_row_rejects_empty_text():
    schema = load_label_schema()
    with pytest.raises(ValueError):
        validate_row({"text": "  ", "task_type": "Open QA"}, schema)


def test_validate_row_clamps_dims():
    schema = load_label_schema()
    clean = validate_row(
        {"text": "x", "task_type": "Open QA", "reasoning": 5.0}, schema
    )
    assert clean["reasoning"] == 1.0


def test_synth_gen_produces_all_task_types(tmp_path: Path):
    out = tmp_path / "synth.jsonl"
    n = generate(out, per_cell=2)
    schema = load_label_schema()
    assert n > 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    produced = {r["task_type"] for r in rows}
    # every task type id should appear
    assert produced == set(schema.task_type_ids)
    # 11 tasks x 3 tiers x 2 per cell
    assert n == 11 * 3 * 2


def test_build_dataset_silver_to_train_gold_to_valtest(tmp_path: Path):
    silver = tmp_path / "silver.jsonl"
    gold = tmp_path / "gold.jsonl"
    silver.write_text(
        "\n".join(
            json.dumps({"text": f"prompt silver {i}", "task_type": "Open QA"})
            for i in range(10)
        )
    )
    gold.write_text(
        "\n".join(
            json.dumps({"text": f"prompt gold {i}", "task_type": "Summarization"})
            for i in range(4)
        )
    )
    out = tmp_path / "processed"
    # gold-only eval: disable silver holdout so all silver -> train.
    counts = build([str(silver)], [str(gold)], out, val_frac=0.5, silver_eval_frac=0.0)
    assert counts["train"] == 10
    assert counts["val"] + counts["test"] == 4
    assert (out / "train.jsonl").exists()
    assert (out / "build_report.json").exists()


def test_build_dataset_silver_holdout_when_no_gold(tmp_path: Path):
    """With no gold, a stratified slice of silver feeds val/test (never empty)."""
    silver = tmp_path / "silver.jsonl"
    silver.write_text(
        "\n".join(
            json.dumps({"text": f"p {tt} {i}", "task_type": tt})
            for tt in ("Open QA", "Summarization")
            for i in range(100)
        )
    )
    out = tmp_path / "processed"
    counts = build([str(silver)], [], out, val_frac=0.5, silver_eval_frac=0.1)
    # 200 silver, 10% per class held out for eval -> 20 eval, 180 train.
    assert counts["train"] == 180
    assert counts["val"] + counts["test"] == 20
    assert counts["val"] > 0 and counts["test"] > 0


def test_build_dataset_dedups_gold_over_silver(tmp_path: Path):
    dup_text = "duplicate prompt text"
    silver = tmp_path / "silver.jsonl"
    gold = tmp_path / "gold.jsonl"
    silver.write_text(json.dumps({"text": dup_text, "task_type": "Open QA"}))
    gold.write_text(json.dumps({"text": dup_text, "task_type": "Open QA"}))
    out = tmp_path / "processed"
    counts = build([str(silver)], [str(gold)], out, val_frac=1.0)
    # gold wins; silver duplicate dropped -> train empty, val has the row
    assert counts["train"] == 0
    assert counts["val"] == 1
