import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Polygon, box

from scripts.regions.world import paths, wdpa_event_ledger
from scripts.regions.world.leakage_guard import (
    assert_dynamic_state_is_prior_only,
    assert_no_forbidden_columns,
)
from scripts.regions.world.vif_prune import iterative_prune
from scripts.regions.world.wdpa_event_ledger import (
    classify,
    largest_baseline_overlap,
    parcel_lineage,
    split_antimeridian,
    suffix_rank,
    transient_gaps,
    union_baseline_share,
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


def test_largest_baseline_overlap_picks_the_most_covering_polygon(monkeypatch):
    crs = "ESRI:54009"
    two_parts = box(0, 0, 10_000, 10_000).union(box(20_000, 0, 30_000, 10_000))
    sites = gpd.GeoDataFrame(geometry=[box(0, 0, 10_000, 10_000), box(50_000, 0, 60_000, 10_000), two_parts], crs=crs)
    baseline = gpd.GeoDataFrame({"WDPA_PID": ["a", "b", "c"]},
                                geometry=[box(-5_000, -5_000, 2_000, 15_000), box(0, -5_000, 5_000, 15_000),
                                          box(21_000, 1_000, 29_000, 9_000)], crs=crs)
    for batch in [20_000, 1]:
        monkeypatch.setattr(wdpa_event_ledger, "OVERLAP_BATCH", batch)
        out = largest_baseline_overlap(sites, baseline)
        assert out.largest_baseline_pid[0] == "b" and pd.isna(out.largest_baseline_pid[1]) and out.largest_baseline_pid[2] == "c"
        assert np.allclose(out.largest_baseline_share, [0.5, 0.0, 64 / 200])


def test_split_antimeridian_keeps_a_crossing_polygon_near_180_degrees():
    crossing = Polygon([(179, 0), (-179, 0), (-179, 1), (179, 1)])
    split = split_antimeridian([crossing, box(10, 0, 11, 1)])[0]
    assert shapely.area(split) == pytest.approx(2.0)
    assert shapely.bounds(split)[0] == pytest.approx(-180) and shapely.bounds(split)[2] == pytest.approx(180)
    assert not shapely.intersects(split, box(-170, 0, 170, 1))
    past = split_antimeridian([box(178, 0, 182, 1)])[0]
    assert shapely.area(past) == pytest.approx(4.0) and shapely.bounds(past)[2] <= 180
    assert shapely.intersects(past, box(-180, 0, -179, 1))
    sites = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(179.5, 0.2, 179.9, 0.8)], crs="EPSG:4326")
    out = largest_baseline_overlap(sites, gpd.GeoDataFrame({"WDPA_PID": ["x"]}, geometry=[crossing], crs="EPSG:4326"))
    assert out.largest_baseline_share.tolist() == pytest.approx([0.0, 1.0])


def test_suffix_rank_maps_letters_and_digits_to_the_same_position():
    assert [suffix_rank(p) for p in ["5_A", "5_B", "5_Z", "5_AA", "5_1", "5_27", "5"]] == [1, 2, 26, 27, 1, 27, None]


def test_parcel_lineage_matches_renamed_parcels_one_to_one():
    cols = ["WDPAID", "WDPA_PID", "kind", "IUCN_CAT", "AREA"]
    vanished = pd.DataFrame([[1, "1_A", "polygons", "Ia", 10.0], [1, "1_B", "polygons", "Ia", 10.0],
                             [2, "2_A", "polygons", "II", 5.0], [3, "3", "polygons", "IV", 7.0]], columns=cols)
    appearing = pd.DataFrame([[1, "1_1", "polygons", "Ia", 10.0], [1, "1_2", "polygons", "Ia", 10.05],
                              [2, "2_1", "polygons", "II", 9.0], [3, "33", "polygons", "IV", 7.0]], columns=cols)
    out = parcel_lineage(vanished, appearing).sort_values("successor").reset_index(drop=True)
    # Same-site twins resolve by suffix position, an area change beyond the tolerance stays unmatched.
    assert out.to_dict("list") == {"successor": ["1_1", "1_2", "33"], "predecessor": ["1_A", "1_B", "3"]}


def test_union_baseline_share_counts_overlapping_baselines_once():
    crs = "ESRI:54009"
    sites = gpd.GeoDataFrame(geometry=[box(0, 0, 10_000, 10_000), box(50_000, 0, 60_000, 10_000)], crs=crs)
    baseline = gpd.GeoDataFrame(geometry=[box(0, 0, 6_000, 10_000), box(4_000, 0, 8_000, 10_000)], crs=crs)
    assert union_baseline_share(sites, baseline).tolist() == pytest.approx([0.8, 0.0], abs=1e-6)
