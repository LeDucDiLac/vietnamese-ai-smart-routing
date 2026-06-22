"""Serving + sim/eval tests — the pure-Python routing path, no ML deps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router.capability import load_capabilities
from serving.api import RouteRequest, StubClassifier, route_prompt
from sim.routellm_router import ViComplexityScorer, calibrate_threshold
from sim.vi_response_cache import build_synthetic


def test_stub_classifier_returns_nvidia_shape():
    clf = StubClassifier()
    rec = clf.predict(["Viết một hàm Python để sắp xếp danh sách."])[0]
    assert rec["task_type_1"] == "Code Generation"
    assert 0.0 <= rec["prompt_complexity_score"] <= 1.0
    assert "reasoning" in rec


def test_route_prompt_end_to_end_with_stub():
    clf = StubClassifier()
    caps = load_capabilities()
    req = RouteRequest(prompt="Xin chào, bạn khỏe không?", user_groups=None)
    resp = route_prompt(clf, caps, req)
    assert resp.model_id in {m.id for m in caps.models}
    assert resp.classifier_backend == "stub"
    assert resp.routing_overhead_ms >= 0.0


def test_route_prompt_respects_permissions():
    clf = StubClassifier()
    caps = load_capabilities()
    # a default user (no groups) can only get "all"-permitted models
    req = RouteRequest(prompt="x" * 2000, user_groups=None)
    resp = route_prompt(clf, caps, req)
    chosen = caps.get(resp.model_id)
    assert "all" in chosen.permissions


def test_complexity_scorer_wraps_classifier():
    scorer = ViComplexityScorer(StubClassifier())
    score = scorer.score("Một câu hỏi đơn giản?")
    assert 0.0 <= score <= 1.0
    batch = scorer.score_batch(["a", "b", "c"])
    assert len(batch) == 3


def test_calibrate_threshold_quantile():
    scores = [i / 100 for i in range(101)]  # 0.00 .. 1.00
    # send top 30% strong -> threshold ~0.70
    thr = calibrate_threshold(scores, 0.30)
    assert thr == pytest.approx(0.70, abs=0.02)
    # edges
    assert calibrate_threshold(scores, 0.0) == float("inf")
    assert calibrate_threshold(scores, 1.0) == float("-inf")


def test_simulate_end_to_end(tmp_path: Path):
    """Synthetic VN cache -> simulate -> four metrics computed."""
    from tests.eval.simulate import simulate

    # build a small labeled prompt set spanning complexity
    prompts = []
    for i in range(40):
        hard = i % 2 == 0
        prompts.append(
            {
                "id": i,
                "text": f"prompt {i} " + ("phân tích sâu " * 5 if hard else "ngắn"),
                "task_type": "Open QA",
                "reasoning": 0.9 if hard else 0.1,
                "creativity_scope": 0.8 if hard else 0.1,
            }
        )
    rows = build_synthetic(prompts)
    cache = tmp_path / "vi_cache.jsonl"
    cache.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )

    report = simulate(cache, user_groups=["premium"])
    # router cost should not exceed always-best cost
    assert (
        report.policies["router"]["total_cost"]
        <= report.policies["always_best"]["total_cost"] + 1e-9
    )
    # the four target keys are present
    assert set(report.targets) == {
        "cost_reduction_vs_best>=0.30",
        "quality_drop_vs_best<=0.03",
        "latency_reduction_simple>=0.20",
        "routing_overhead_p95<=50ms",
    }
    # decision overhead is microseconds — must clear the 50ms budget easily
    assert report.routing_overhead_ms["p95_ms"] <= 50.0
