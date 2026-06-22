"""Permission filtering — phân quyền (plan §1, §5).

Before the matcher ranks anything, it drops models the requesting user (or their
department) isn't allowed to use. A model lists the permission groups that may
reach it (``permissions: [engineering, premium]``); a request carries the
groups the caller holds. The model is permitted iff the two sets intersect, or
the model is open to ``all``.

Pure-Python, no deps beyond the registry types.
"""

from __future__ import annotations

from collections.abc import Iterable

from config import ModelProfile

# A model with this group in its permission list is reachable by everyone.
OPEN_GROUP = "all"


def is_permitted(model: ModelProfile, user_groups: Iterable[str]) -> bool:
    """True if a caller holding ``user_groups`` may use ``model``.

    A model open to ``all`` is always permitted. Otherwise the caller needs at
    least one group in common with the model's permission list.
    """
    allowed = set(model.permissions)
    if OPEN_GROUP in allowed:
        return True
    return bool(allowed & set(user_groups))


def filter_permitted(
    models: Iterable[ModelProfile], user_groups: Iterable[str] | None
) -> list[ModelProfile]:
    """Return only the models the caller may use.

    ``user_groups=None`` is treated as a caller holding no special groups — they
    still see every ``all`` model, matching the common "anonymous/default user"
    case.
    """
    groups = list(user_groups) if user_groups else []
    return [m for m in models if is_permitted(m, groups)]
