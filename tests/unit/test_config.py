"""Config-loader tests: label schema, complexity scoring, model registry."""

from __future__ import annotations

import pytest

from config import (
    load_complexity,
    load_label_schema,
    load_model_registry,
    task_type_id,
)


def test_task_type_id_normalizes():
    assert task_type_id("Open QA") == "open_qa"
    assert task_type_id("Code Generation") == "code_generation"
    assert task_type_id("open_qa") == "open_qa"


def test_label_schema_parity():
    schema = load_label_schema()
    # NVIDIA parity: 11 task types, 6 complexity dims (plan §10 Q5)
    assert schema.num_task_types == 11
    assert len(schema.complexity_dimensions) == 6


def test_task_index_by_label_and_id():
    schema = load_label_schema()
    i = schema.task_index("Open QA")
    assert schema.task_index("open_qa") == i
    assert schema.task_types[i] == "Open QA"


def test_task_index_unknown_raises():
    schema = load_label_schema()
    with pytest.raises(KeyError):
        schema.task_index("Nonexistent Task")


def test_complexity_weights_sum_to_one():
    cx = load_complexity()
    assert abs(sum(cx.weights_map.values()) - 1.0) < 1e-6


def test_complexity_score_matches_nvidia_formula():
    cx = load_complexity()
    dims = {
        "creativity_scope": 1.0,
        "reasoning": 1.0,
        "constraint_ct": 1.0,
        "domain_knowledge": 1.0,
        "contextual_knowledge": 1.0,
        "number_of_few_shots": 1.0,
    }
    # all dims at 1.0, divisors 1.0 -> score == sum of weights == 1.0
    assert abs(cx.score(dims) - 1.0) < 1e-6
    # all zero -> 0
    assert cx.score({d: 0.0 for d in dims}) == 0.0


def test_complexity_score_missing_dims_treated_zero():
    cx = load_complexity()
    assert cx.score({"creativity_scope": 1.0}) == pytest.approx(0.35)


def test_model_registry_unique_ids_and_lookup():
    reg = load_model_registry()
    ids = [m.id for m in reg.models]
    assert len(ids) == len(set(ids))
    first = reg.models[0]
    assert reg.by_id(first.id).id == first.id


def test_model_skill_floor_fallback():
    reg = load_model_registry()
    m = reg.models[0]
    # an unknown task falls back to the floor
    assert m.skill_for("definitely_not_a_task", reg.skill_floor) == reg.skill_floor
