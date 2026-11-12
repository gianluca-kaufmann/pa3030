import numpy as np
import pandas as pd
import pytest

from scripts.regions.shared.forward.scenario_allocation import (
    add_scenario_selections,
    national_allocation_table,
    select_by_area,
)


def make_pixels(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "proba": rng.random(n),
        "area_km2": rng.uniform(0.4, 1.0, n),
        "iso_a3": rng.choice(["AAA", "BBB", "CCC"], n, p=[0.5, 0.3, 0.2]),
    })


def make_baseline(df):
    """AAA at 10% coverage, BBB at 25%, CCC already at 40%."""
    unprotected = df.groupby("iso_a3")["area_km2"].sum()
    coverage = {"AAA": 0.10, "BBB": 0.25, "CCC": 0.40}
    stats = []
    for iso, free_km2 in unprotected.items():
        total = free_km2 / (1 - coverage[iso])
        stats.append({"iso_a3": iso, "total_km2": total, "protected_2024_km2": total - free_km2})
    total = sum(s["total_km2"] for s in stats)
    protected = sum(s["protected_2024_km2"] for s in stats)
    return {
        "country_stats": stats,
        "bau_km2": 50.0,
        "km2_needed_for_moderate": ((protected / total + 0.30) / 2) * total - protected,
        "km2_needed_for_30pct": 0.30 * total - protected,
    }


def old_cutoff_selection(proba, area, target):
    """The single-threshold rule this module replaces (results_core.compute_scenario_cutoffs)."""
    order = np.argsort(-proba, kind="stable")
    cum = np.cumsum(area[order])
    idx = min(np.searchsorted(cum, target, side="left"), len(proba) - 1)
    return proba >= proba[order][idx]


def test_regional_pool_reproduces_old_cutoff_rule():
    df = make_pixels()
    target = 0.2 * df["area_km2"].sum()
    new = select_by_area(df["proba"].values, df["area_km2"].values, target)
    old = old_cutoff_selection(df["proba"].values, df["area_km2"].values, target)
    assert np.array_equal(new, old)


def test_zero_target_selects_nothing():
    df = make_pixels()
    assert not select_by_area(df["proba"].values, df["area_km2"].values, 0.0).any()


def test_national_quota_closes_each_gap_by_less_than_one_pixel():
    df = make_pixels()
    baseline = make_baseline(df)
    df = add_scenario_selections(df, baseline, "national_quota", "proba")
    table = national_allocation_table(df, baseline).set_index("iso_a3")

    for iso in ["AAA", "BBB"]:
        over = table.loc[iso, "full_selected_km2"] - table.loc[iso, "full_needed_km2"]
        assert 0.0 <= over < df["area_km2"].max()
        assert table.loc[iso, "projected_pct_30x30"] == pytest.approx(0.30, abs=1e-3)
    assert table.loc["CCC", "full_selected_km2"] == 0.0


def test_national_quota_ranks_within_country():
    df = make_pixels()
    df = add_scenario_selections(df, make_baseline(df), "national_quota", "proba")
    for _, g in df[df["iso_a3"] != "CCC"].groupby("iso_a3"):
        assert g.loc[g["full_sel"], "proba"].min() >= g.loc[~g["full_sel"], "proba"].max()


def test_regional_pool_gives_uneven_country_coverage():
    """The pooled rule is what produced countries far above and far below 30%."""
    df = make_pixels()
    df.loc[df["iso_a3"] == "AAA", "proba"] *= 0.2
    baseline = make_baseline(df)
    df = add_scenario_selections(df, baseline, "regional_pool", "proba")
    table = national_allocation_table(df, baseline).set_index("iso_a3")
    assert table.loc["AAA", "projected_pct_30x30"] < 0.30
    assert table.loc["CCC", "projected_pct_30x30"] > 0.40


def test_shortfall_when_country_lacks_unprotected_land():
    df = make_pixels()
    baseline = make_baseline(df)
    for s in baseline["country_stats"]:
        if s["iso_a3"] == "AAA":
            s["total_km2"] *= 10
    df = add_scenario_selections(df, baseline, "national_quota", "proba")
    table = national_allocation_table(df, baseline).set_index("iso_a3")
    assert df.loc[df["iso_a3"] == "AAA", "full_sel"].all()
    assert table.loc["AAA", "full_shortfall_km2"] > 0.0


def test_unknown_allocation_raises():
    df = make_pixels()
    with pytest.raises(ValueError):
        add_scenario_selections(df, make_baseline(df), "global", "proba")
