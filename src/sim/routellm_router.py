"""Register the vi-router classifier as a custom RouteLLM router (plan §11.2).

RouteLLM (``pip install "routellm[serve,eval]"``) is a framework for serving and
evaluating routers that direct between a strong/expensive and a weak/cheap model.
It exposes an extensible router interface: a router implements a scoring hook that
returns, per prompt, a number in [0,1] — RouteLLM calibrates a threshold over that
score so that "route X% of traffic to the strong model" maps to a concrete cutoff
(plan §11.2). That threshold knob is exactly our cost/quality lever (plan §6).

Our classifier already emits ``prompt_complexity_score`` in [0,1]; that *is* the
routing signal. This module adapts it to RouteLLM's interface.

Two layers, on purpose:

- :class:`ViComplexityScorer` — pure-Python, no RouteLLM dependency. Wraps any
  object with a ``predict(list[str]) -> list[record]`` method (the stub, torch, or
  ONNX classifier) and returns the complexity score. Our own harness
  (``tests/eval/simulate.py``) uses this directly, so the four success metrics can
  be produced **without** installing RouteLLM at all.
- :func:`build_routellm_router` — only imported if RouteLLM is installed; subclasses
  its ``Router`` base and delegates to the scorer. This is the integration point
  for anyone who wants to drive our classifier from RouteLLM's serve/eval CLIs.
"""

from __future__ import annotations

from typing import Any, Protocol


class _Predicts(Protocol):
    def predict(self, prompts: list[str]) -> list[dict[str, Any]]: ...


class ViComplexityScorer:
    """Turn a classifier into a [0,1] routing score (high = send to strong model).

    The score is the prompt's ``prompt_complexity_score``. RouteLLM's convention is
    that a higher score routes to the *strong* model, which matches us: harder
    prompts need the more capable (expensive) model.
    """

    def __init__(self, classifier: _Predicts):
        self.classifier = classifier

    def score(self, prompt: str) -> float:
        rec = self.classifier.predict([prompt])[0]
        return float(rec["prompt_complexity_score"])

    def score_batch(self, prompts: list[str]) -> list[float]:
        recs = self.classifier.predict(prompts)
        return [float(r["prompt_complexity_score"]) for r in recs]

    def analyze(self, prompt: str) -> dict[str, Any]:
        """Full NVIDIA-style record (task type + dims + score)."""
        return self.classifier.predict([prompt])[0]


def calibrate_threshold(
    scores: list[float], strong_fraction: float
) -> float:
    """Solve for the threshold that sends ``strong_fraction`` of traffic to strong.

    This mirrors RouteLLM's calibration (plan §11.2) but needs no dependency: sort
    the scores and pick the cutoff at the (1 - fraction) quantile. Prompts scoring
    **above** the returned threshold go to the strong model.

    ``strong_fraction=0.3`` => threshold at the 70th percentile => the top 30% most
    complex prompts route strong.
    """
    if not 0.0 <= strong_fraction <= 1.0:
        raise ValueError(f"strong_fraction must be in [0,1], got {strong_fraction}")
    if not scores:
        return 0.5
    if strong_fraction <= 0.0:
        return float("inf")  # nothing routes strong
    if strong_fraction >= 1.0:
        return float("-inf")  # everything routes strong
    ordered = sorted(scores)
    # index of the (1 - fraction) quantile
    idx = int(round((1.0 - strong_fraction) * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def build_routellm_router(classifier: _Predicts):  # pragma: no cover - optional dep
    """Build a RouteLLM ``Router`` subclass backed by our classifier.

    Only call this when ``routellm`` is installed. Returns an *instance* registered
    against RouteLLM's expected interface. Kept import-guarded so the rest of the
    sim layer (and tests) never require RouteLLM.
    """
    try:
        from routellm.routers.routers import Router
    except ImportError as exc:
        raise RuntimeError(
            'RouteLLM not installed. Run: pip install "routellm[serve,eval]"'
        ) from exc

    scorer = ViComplexityScorer(classifier)

    class ViRouter(Router):
        """RouteLLM router that scores by our VN classifier's complexity output."""

        def calculate_strong_win_rate(self, prompt: str) -> float:  # noqa: D401
            # RouteLLM expects: probability the *strong* model is needed.
            # Our complexity score is already that signal in [0,1].
            return scorer.score(prompt)

    return ViRouter()
