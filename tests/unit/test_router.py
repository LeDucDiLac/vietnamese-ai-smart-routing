"""Router tests: permission filter, capability leaderboard, matching algorithm."""

from __future__ import annotations

import pytest

from config import ModelProfile
from router.capability import load_capabilities
from router.match import RouteDecision, required_skill_for, select_model
from router.policy import filter_permitted, is_permitted


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def _model(mid: str, perms: list[str]) -> ModelProfile:
    return ModelProfile(
        id=mid,
        cost_per_1k_tokens=0.001,
        latency_ms_per_1k=100,
        permissions=perms,
    )


def test_is_permitted_open_to_all():
    m = _model("m", ["all"])
    assert is_permitted(m, [])
    assert is_permitted(m, ["anything"])


def test_is_permitted_requires_group_intersection():
    m = _model("m", ["premium"])
    assert not is_permitted(m, [])
    assert not is_permitted(m, ["engineering"])
    assert is_permitted(m, ["premium"])
    assert is_permitted(m, ["engineering", "premium"])


def test_filter_permitted_none_groups():
    models = [_model("open", ["all"]), _model("locked", ["premium"])]
    out = filter_permitted(models, None)
    assert [m.id for m in out] == ["open"]


# ---------------------------------------------------------------------------
# required_skill_for
# ---------------------------------------------------------------------------


def test_required_skill_monotonic():
    assert required_skill_for(0.0) == pytest.approx(0.55)
    assert required_skill_for(1.0) == pytest.approx(0.92)
    assert required_skill_for(0.0) < required_skill_for(0.5) < required_skill_for(1.0)


def test_required_skill_clamps():
    assert required_skill_for(-5.0) == pytest.approx(0.55)
    assert required_skill_for(5.0) == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# select_model (against the seeded registry)
# ---------------------------------------------------------------------------


def test_simple_prompt_routes_cheap():
    caps = load_capabilities()
    # low complexity, default user -> cheapest qualifying model
    decision = select_model(caps, "Summarization", 0.05, user_groups=None)
    assert isinstance(decision, RouteDecision)
    assert decision.cleared_bar
    # cheapest model in the seed registry is tiny-fast
    assert decision.model_id == "tiny-fast"


def test_hard_prompt_routes_strong_when_permitted():
    caps = load_capabilities()
    decision = select_model(
        caps, "Code Generation", 0.95, user_groups=["premium"]
    )
    # high complexity code gen needs a strong model; tiny-fast (code 0.40) can't clear
    assert decision.model_id in {"mid-strong", "giant-smart"}


def test_permission_filter_blocks_premium_models():
    caps = load_capabilities()
    # default user can't reach premium/engineering models even for a hard prompt
    decision = select_model(caps, "Code Generation", 0.95, user_groups=None)
    chosen = caps.get(decision.model_id)
    assert "all" in chosen.permissions


def test_fallback_when_no_model_clears_bar():
    caps = load_capabilities()
    # default user, max complexity: open models may not clear the high bar ->
    # fallback to the most capable permitted model, flagged.
    decision = select_model(caps, "Code Generation", 1.0, user_groups=None)
    assert not decision.cleared_bar
    # fallback picks the highest-skill permitted (all) model
    chosen = caps.get(decision.model_id)
    assert "all" in chosen.permissions


def test_unknown_user_groups_still_get_open_models():
    caps = load_capabilities()
    decision = select_model(caps, "Open QA", 0.1, user_groups=["nonsense_group"])
    chosen = caps.get(decision.model_id)
    assert "all" in chosen.permissions


# ---------------------------------------------------------------------------
# capability leaderboard
# ---------------------------------------------------------------------------


def test_leaderboard_sorted_by_skill():
    caps = load_capabilities()
    rows = caps.leaderboard("code_generation")
    skills = [r.skill for r in rows]
    assert skills == sorted(skills, reverse=True)
    # giant-smart is best at code in the seed registry
    assert rows[0].model_id == "giant-smart"


def test_full_leaderboard_covers_all_tasks():
    caps = load_capabilities()
    full = caps.full_leaderboard()
    assert "open_qa" in full
    assert "code_generation" in full
    # each task ranks all 4 seed models
    assert all(len(rows) == len(caps.models) for rows in full.values())


# ---------------------------------------------------------------------------
# leaderboard refresh job (success criterion #1)
# ---------------------------------------------------------------------------


def test_build_leaderboard_payload():
    from router.leaderboard import build_leaderboard

    payload = build_leaderboard()
    assert payload["num_models"] == 4
    # overall ranking is sorted by mean skill desc, ranks are 1..N contiguous
    overall = payload["overall"]
    means = [r["mean_skill"] for r in overall]
    assert means == sorted(means, reverse=True)
    assert [r["rank"] for r in overall] == list(range(1, len(overall) + 1))
    # giant-smart has the highest mean skill in the seed registry
    assert overall[0]["model_id"] == "giant-smart"
    # by_task_type covers every schema task and ranks all models per task
    from config import load_label_schema

    schema = load_label_schema()
    for tid in schema.task_type_ids:
        assert tid in payload["by_task_type"]
        assert len(payload["by_task_type"][tid]) == 4


def test_leaderboard_refresh_writes_artifacts(tmp_path):
    from router.leaderboard import refresh

    artifacts = refresh(tmp_path)
    assert artifacts["json"].endswith("leaderboard.json")
    assert artifacts["markdown"].endswith("leaderboard.md")
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "leaderboard.md").exists()
    md = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "# AI Capability Leaderboard" in md
    assert "## Per task type" in md
