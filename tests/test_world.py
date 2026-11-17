import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts.regions.world import paths, wdpa_event_ledger
from scripts.regions.world.leakage_guard import (
    assert_dynamic_state_is_prior_only,
    assert_no_forbidden_columns,
)
from scripts.regions.world.vif_prune import iterative_prune
from scripts.regions.world.wdpa_event_ledger import (
    baseline_overlap,
    classify,
    transient_gaps,
)


def test_country_year_tables_are_tracked_inputs():
    for path in [paths.GOVERNANCE_PATH, paths.FINANCE_PATH, paths.REDD_PATH]:
        assert path.exists(), path
        assert "country" in pd.read_csv(path, nrows=1).columns


def test_forbidden_column_raises():
    assert_no_forbidden_columns(["dist_to_pa_px", "ndvi"])
    with pytest.raises(RuntimeError):
        assert_no_forbidden_columns(["ndvi", "pa_year"])


def test_prior_state_mask_must_exclude_year_t():
    pa_year = np.array([0, 2005, 2010, 2012])
    assert_dynamic_state_is_prior_only(pa_year.astype(bool) & (pa_year < 2010), pa_year, 2010)
    with pytest.raises(RuntimeError):
        assert_dynamic_state_is_prior_only(pa_year.astype(bool) & (pa_year <= 2010), pa_year, 2010)


def test_vif_prune_drops_a_collinear_column():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=500), "a_copy": a + rng.normal(scale=0.01, size=500)})
    vifs, dropped = iterative_prune(X, threshold=10)
    assert len(dropped) == 1 and dropped[0][0] in {"a", "a_copy"}
    assert "b" in vifs


def test_transient_gaps_need_presence_on_both_sides():
    presence = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], bool)
    assert transient_gaps(presence).tolist() == [[False, True, True, False], [False] * 4, [False] * 4]


def test_ledger_classes_follow_the_protocol():
    rows = pd.DataFrame({
        "STATUS_YR": [2022, 2022, 2022, 2022, 2020, 2015, 2025, 0, 2015],
        "change": ["new"] * 8 + ["category_change"],
        "prior_record": [False, True, False, False, False, False, False, False, True],
        "baseline_site": [False, False, True, False, False, False, False, False, True],
        "kind": ["polygons"] * 3 + ["points"] + ["polygons"] * 5,
    })
    out = classify(rows)
    assert out.event_class.tolist() == [1, 4, 4, 5, 3, 2, 5, 5, 4]
    assert out.reason.tolist()[-2:] == ["no_status_year", "category_change"]


def test_baseline_overlap_counts_shared_baseline_area_once(monkeypatch):
    crs = "ESRI:54009"
    sites = gpd.GeoDataFrame(geometry=[box(0, 0, 10_000, 10_000), box(50_000, 0, 60_000, 10_000)], crs=crs)
    baseline = gpd.GeoDataFrame(geometry=[box(-5_000, -5_000, 5_000, 15_000), box(0, -5_000, 5_000, 15_000)], crs=crs)
    assert np.allclose(baseline_overlap(sites, baseline), [0.5, 0.0])
    monkeypatch.setattr(wdpa_event_ledger, "OVERLAP_BATCH", 1)
    assert np.allclose(baseline_overlap(sites, baseline), [0.5, 0.0])
