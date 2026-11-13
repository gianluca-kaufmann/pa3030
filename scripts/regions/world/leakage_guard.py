"""
Leakage-prevention contract for the panel-building pipeline.

Why this exists: CLAUDE.md claims "dynamic features (dist_to_pa_px, pa_density_*,
cum_protected_frac) are computed from the cumulative PA state at the start of each
panel year" — true today, but that was only ever a documentation claim, never
checked by code. PA3030 (the now-frozen thesis project) enforces its equivalent
rule with a hard EXCLUDE column list checked at split time. This module is the
same idea here: turn the prose into an assertion that fails loudly the moment a
future edit breaks the invariant, instead of silently producing a leaky panel
that only gets noticed because a result looks suspiciously good.

Import and use from build_dataset.py (or any other panel-building script):

    from scripts.regions.world.leakage_guard import (
        assert_dynamic_state_is_prior_only, assert_no_forbidden_columns,
    )
"""

# Features that must be recomputed fresh, per panel year t, strictly from PA
# state designated BEFORE t. If any of these were ever computed from state that
# includes year t or later, the model would be told the answer before predicting
# it — this is the single most consequential invariant in the pipeline.
DYNAMIC_FEATURES = frozenset({
    "dist_to_pa_px", "pa_density_10km", "pa_density_50km", "cum_protected_frac",
    "years_since_forest_loss", "had_recent_forest_loss",
})

# Features computed from the SAME calendar year's satellite/annual composite as
# the panel year being predicted (gee_lc_year, embed_velocity_*). Not a proven
# leak — an annual land-cover/embedding composite is unlikely to already reflect
# a designation decision made within that same year — but it's a real,
# undischarged assumption, not a verified one. Flagged here as documentation so
# it isn't silently forgotten; not code-enforced because there's no cheap way to
# check "does this satellite composite reflect the designation event" without
# ground truth on compositing dates per pixel.
#
# aef_mean_* used to belong on this list too, but for a worse reason than a same-
# year assumption: build_dataset.py silently reused the static 2017 AEF mosaic for
# every year without a real annual export, so 100% of BC's 2001-2014 training rows
# were fed a satellite composite up to 16 years in their own future, labeled as
# contemporaneous (found in the 2026-08-09 data audit). Fixed directly in
# build_dataset.py: aef_mean_* is now NaN (median-imputed, same as any other
# genuinely missing value) for any year without a real annual export, instead of
# reusing a future/past snapshot. No longer belongs in this "assumption" list —
# it's a resolved leak, not an open one.
YEAR_MATCHED_RISK_FEATURES = frozenset({
    "gee_lc_year", "embed_velocity_mag", "embed_velocity_cos",
})

# Columns that must NEVER appear in a built panel — anything that could only be
# known after the outcome (label) is determined, or that directly encodes future
# PA state. Hard block: if any of these show up, the build is wrong, full stop.
FORBIDDEN_FEATURES = frozenset({
    "pa_year", "year_designated", "designated_after_t", "future_protected_frac",
    "next_designation_year", "total_area_ever_protected", "designated_in_t",
})


def assert_no_forbidden_columns(columns, context=""):
    """Raise if any FORBIDDEN_FEATURES column made it into a built panel.
    `columns` may be a DataFrame's .columns, a parquet schema's .names, or any
    iterable of column-name strings."""
    leaked = FORBIDDEN_FEATURES & set(columns)
    if leaked:
        where = f" ({context})" if context else ""
        raise RuntimeError(
            f"Leakage guard failed{where}: forbidden column(s) present in panel: "
            f"{sorted(leaked)}. These can only be known after the outcome and must "
            f"never be built into a training panel."
        )


def assert_dynamic_state_is_prior_only(designated_before_t, pa_year, t, context=""):
    """Raise if the 'prior PA state' mask used to compute this year's dynamic
    features (dist_to_pa_px, pa_density_*, cum_protected_frac, ...) includes any
    pixel actually designated in year t or later.

    This is the literal invariant DYNAMIC_FEATURES depends on for temporal
    correctness. Checking it here means a future refactor that accidentally
    changes `designated_before_t`'s definition (e.g. `<=` instead of `<`) fails
    immediately and loudly, instead of quietly leaking future PA state into
    every dynamic feature for the rest of the run.

    designated_before_t / pa_year: same-shape boolean/int arrays as used in
    build_dataset.py's per-year loop. t: the panel year being built.
    """
    bad = designated_before_t & (pa_year >= t) & (pa_year > 0)
    n_bad = int(bad.sum())
    if n_bad:
        where = f" ({context})" if context else ""
        raise RuntimeError(
            f"Leakage guard failed{where}: {n_bad} pixel(s) marked 'designated "
            f"before {t}' were actually designated in {t} or later. Dynamic "
            f"features for this year would leak future PA state."
        )
