"""Shared feature-leakage guard for the PA3030 training/tuning/forward pipeline.

Why this exists
---------------
Every regional training script defines a local ``EXCLUDE_COLS`` set used to
drop forbidden columns before model fitting. Nothing currently fails the
build if a forbidden column slips through (typo in EXCLUDE_COLS, schema
change, stale cached feature list in a deployment artifact). This module
turns that convention into a hard, shared check so paper-grade artifacts
stay leak-free.

Usage
-----
Call ``check_feature_denylist`` immediately after the model's feature list
is finalised and *before* any ``fit`` / ``Dataset`` / inference batch loop:

    from scripts.regions.shared.training.feature_guard import check_feature_denylist
    check_feature_denylist(feature_cols, context="south_america/lgbm/training")

The call is O(n) over ``feature_cols``; cost is negligible compared to data
loading. ``context`` is a free-form label that is included in the error
message so the failing call site is obvious in logs.

Year handling
-------------
``year`` is in ``FORBIDDEN_COLS`` because the entire pipeline currently uses
``year`` only for filtering and time-weighting (``compute_year_weights``),
never as a model feature. If a future workstream deliberately wants to learn
on ``year``, that call site can pass an explicit ``forbidden`` set with
``year`` removed; the default keeps the historical invariant.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence


# Exact column names that must never appear as model features.
# Includes raw protection labels, target columns (both annual hazard and the
# legacy 5-year window), spatial identifiers, time identifiers, and the W3
# country helper columns.
FORBIDDEN_COLS: frozenset[str] = frozenset({
    # Raw / lagged protection status — the label itself.
    "WDPA",
    "WDPA_b1",
    "WDPA_prev",
    # Targets (annual hazard + legacy 5-year window).
    "transition_01",
    "transition_01_win5",
    "first_transition_year",
    # Spatial / temporal identifiers.
    "x",
    "y",
    "row",
    "col",
    "year",
    # W3 country helper columns (introduced by feature_engineering for the
    # country PA momentum aggregation; must not leak into the feature list).
    "country_id",
    "country_iso3",
})


# Patterns that catch any column name that looks target/leakage-derived even
# if the exact name is not in FORBIDDEN_COLS (e.g. a future ``transition_01_win10``
# or ``WDPA_buffered`` would still be blocked).
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^transition_01(_|$)"),
    re.compile(r"^WDPA(_|$)", re.IGNORECASE),
    re.compile(r"^first_transition"),
    re.compile(r"_target$"),
    re.compile(r"^target_"),
)


class FeatureLeakageError(ValueError):
    """Raised when a forbidden column survives into the model feature list."""


def check_feature_denylist(
    feature_cols: Iterable[str],
    *,
    context: str,
    forbidden: Iterable[str] | None = None,
    patterns: Iterable[re.Pattern[str]] | None = None,
) -> None:
    """Raise ``FeatureLeakageError`` if any forbidden column is in ``feature_cols``.

    Parameters
    ----------
    feature_cols : iterable of str
        The finalised model feature list, after EXCLUDE_COLS filtering.
    context : str
        Short human label identifying the call site (e.g.
        ``"south_america/lgbm/training"`` or ``"forward/predict_core"``).
        Included verbatim in the error message.
    forbidden : iterable of str, optional
        Override for the exact denylist. Defaults to ``FORBIDDEN_COLS``.
    patterns : iterable of compiled regex, optional
        Override for the regex denylist. Defaults to ``FORBIDDEN_PATTERNS``.
    """
    feature_list: Sequence[str] = list(feature_cols)
    forbidden_set: frozenset[str] = (
        frozenset(forbidden) if forbidden is not None else FORBIDDEN_COLS
    )
    pattern_list: Sequence[re.Pattern[str]] = (
        tuple(patterns) if patterns is not None else FORBIDDEN_PATTERNS
    )

    exact_hits: list[str] = [c for c in feature_list if c in forbidden_set]
    pattern_hits: list[str] = [
        c for c in feature_list
        if c not in forbidden_set
        and any(p.search(c) for p in pattern_list)
    ]

    if not exact_hits and not pattern_hits:
        return

    parts = [f"Feature leakage detected in context '{context}':"]
    if exact_hits:
        parts.append(f"  exact denylist matches ({len(exact_hits)}): {sorted(exact_hits)}")
    if pattern_hits:
        parts.append(f"  pattern denylist matches ({len(pattern_hits)}): {sorted(pattern_hits)}")
    parts.append(
        "  -> these columns must not appear as model features. "
        "Fix the EXCLUDE_COLS upstream or pass an explicit `forbidden=` override "
        "if you have a deliberate reason."
    )
    raise FeatureLeakageError("\n".join(parts))
