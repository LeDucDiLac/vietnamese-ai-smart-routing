"""Constrained model selection — the auto-matching algorithm (plan §5).

Given a classified prompt ``(task_type, prompt_complexity_score)`` and the
requesting user's permission groups, pick the **cheapest** model that still
clears the quality bar for that task, subject to a latency ceiling. This is the
"Thuật toán tự động ghép đôi" deliverable and the lever behind the cost/latency
wins (plan §1, §6).

Pure-Python (no ML, no torch). Decision cost is a handful of dict lookups +
a sort over the (small) model list — microseconds, well inside the 50 ms budget
once the classifier has run.

Selection logic
---------------
1. **Permission filter** (``router.policy``): drop models the user can't use.
2. **Quality bar**: map ``prompt_complexity_score`` (0..1) to a minimum required
   skill for the task. Higher complexity demands a more capable model. Among the
   permitted models, keep those whose skill for this task >= the required bar.
3. **Latency ceiling**: drop models whose estimated latency exceeds the budget
   (if one is supplied).
4. **Pick**: cheapest survivor; tie-break on latency, then higher skill.
5. **Fallback**: if no model clears the bar (e.g. a very hard prompt, or a thin
   permission set), pick the most capable permitted model so the request is
   never dropped — and flag it in the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import task_type_id
from router.capability import CapabilityTable
from router.policy import filter_permitted


@dataclass
class RouteDecision:
    """The router's pick plus the reasoning, for transparency at the Gateway."""

    model_id: str
    task_type: str
    complexity_score: float
    required_skill: float
    chosen_skill: float
    est_cost_per_1k: float
    est_latency_ms_per_1k: float
    reason: str
    candidates_considered: int
    cleared_bar: bool = True
    alternatives: list[str] = field(default_factory=list)


def required_skill_for(
    complexity_score: float,
    *,
    min_bar: float = 0.55,
    max_bar: float = 0.92,
) -> float:
    """Map a complexity score in [0,1] to the minimum skill a model must have.

    Linear ramp: a trivial prompt (score 0) only needs ``min_bar`` skill, so the
    cheapest model usually qualifies; a maximal prompt (score 1) needs ``max_bar``,
    forcing selection toward the strongest models. The constants are the policy
    knob for the cost/quality tradeoff (plan §6) and can be calibrated against the
    RouteLLM cost/quality curve (plan §11).
    """
    c = max(0.0, min(1.0, float(complexity_score)))
    return min_bar + (max_bar - min_bar) * c


def select_model(
    capabilities: CapabilityTable,
    task_type: str,
    complexity_score: float,
    *,
    user_groups: list[str] | None = None,
    latency_ceiling_ms_per_1k: float | None = None,
    min_bar: float = 0.55,
    max_bar: float = 0.92,
) -> RouteDecision:
    """Pick the cheapest permitted model that clears the quality bar.

    See module docstring for the full selection logic.
    """
    tid = task_type_id(task_type)
    required = required_skill_for(complexity_score, min_bar=min_bar, max_bar=max_bar)

    # 1. permission filter
    permitted = filter_permitted(capabilities.models, user_groups)
    if not permitted:
        raise ValueError(
            f"No models permitted for user groups {user_groups!r}; "
            "check configs/sim_models.yaml permissions."
        )

    # annotate each permitted model with its skill for this task
    scored = [
        (m, capabilities.skill(m.id, tid))
        for m in permitted
    ]

    # 2. quality bar
    qualified = [(m, s) for (m, s) in scored if s >= required]

    # 3. latency ceiling (applied only to qualified set)
    if latency_ceiling_ms_per_1k is not None:
        within = [
            (m, s) for (m, s) in qualified
            if m.latency_ms_per_1k <= latency_ceiling_ms_per_1k
        ]
        # if the ceiling wipes out every qualified model, keep qualified set
        # rather than dropping to fallback — latency is a soft preference here.
        if within:
            qualified = within

    cleared = bool(qualified)
    if cleared:
        # 4. cheapest survivor; tie-break on latency then higher skill
        pool = qualified
        pool.sort(
            key=lambda ms: (
                ms[0].cost_per_1k_tokens,
                ms[0].latency_ms_per_1k,
                -ms[1],
            )
        )
        reason = (
            f"cheapest model clearing skill bar {required:.2f} for {tid} "
            f"(complexity {complexity_score:.2f})"
        )
    else:
        # 5. fallback: most capable permitted model
        pool = sorted(scored, key=lambda ms: (-ms[1], ms[0].cost_per_1k_tokens))
        reason = (
            f"no model clears skill bar {required:.2f} for {tid}; "
            f"falling back to most capable permitted model"
        )

    chosen, chosen_skill = pool[0]
    alternatives = [m.id for (m, _s) in pool[1:4]]

    return RouteDecision(
        model_id=chosen.id,
        task_type=tid,
        complexity_score=float(complexity_score),
        required_skill=required,
        chosen_skill=chosen_skill,
        est_cost_per_1k=chosen.cost_per_1k_tokens,
        est_latency_ms_per_1k=chosen.latency_ms_per_1k,
        reason=reason,
        candidates_considered=len(permitted),
        cleared_bar=cleared,
        alternatives=alternatives,
    )
