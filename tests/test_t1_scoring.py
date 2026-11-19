import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box
from sklearn.metrics import average_precision_score

from scripts.regions.world.t1_scoring import (
    INELIGIBLE,
    NEGATIVE,
    OUTCOMES,
    POSITIVE,
    REMOVED,
    PooledRanking,
    bootstrap_draws,
    cells_of,
    check_protocol,
    country_bootstrap,
    historical_rates,
    historical_records,
    join_on_keys,
    label,
    new_land_flags,
    part_hits,
    positive_events,
    primary_points,
    ranking_metrics,
    uncertain_events,
    unique_area_by_country,
)

STRICT, BROAD = OUTCOMES["IaIb"], OUTCOMES["I-VI"]


def events(**cols):
    base = {"expansion_role": "new_site", "DESIG_TYPE": "National", "IUCN_CAT_outcome": "II", "IUCN_CAT": "II",
            "category_timing": "recorded_in_window", "event_class": 1, "reason": "designated_in_window", "STATUS_YR": 2022}
    n = max(len(v) for v in cols.values())
    return pd.DataFrame({k: cols.get(k, [v] * n) for k, v in base.items()})


def test_frozen_protocol_hash_is_checked(tmp_path):
    changed = tmp_path / "t1_protocol.md"
    changed.write_text("edited")
    with pytest.raises(ValueError, match="sha256"):
        check_protocol(changed)


def test_key_join_matches_through_float_drift_and_never_drops_cells():
    a = pd.DataFrame({"ISO3": ["CAN", "CAN"], "long": [-79.04609713735253, 10.0], "lat": [82.84276240805451, 1.0], "s": [0.1, 0.2]})
    b = a[["ISO3", "long", "lat"]].assign(long=a.long + 2.8e-14, t=[1.0, 2.0])
    out = join_on_keys(a, b, n_expected=2)
    assert out.t.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="unmatched"):
        join_on_keys(a, b.iloc[:1])
    with pytest.raises(ValueError, match="duplicate"):
        join_on_keys(a, pd.concat([b, b.iloc[:1]]))
    with pytest.raises(ValueError, match="expected"):
        join_on_keys(a, b, n_expected=3)


def test_post_window_records_stay_negative_and_international_is_uncertain():
    ev = events(expansion_role=["none", "none", "new_site", "new_site"], event_class=[5, 5, 1, 1],
                reason=["designated_after_window", "no_status_year", "designated_in_window", "designated_in_window"],
                DESIG_TYPE=["National", "National", "International", "National"])
    assert positive_events(ev, BROAD).tolist() == [False, False, False, True]
    assert uncertain_events(ev, BROAD).tolist() == [False, True, True, False]
    assert uncertain_events(ev, BROAD, post_window_uncertain=True).tolist() == [True, True, True, False]


def test_assigned_later_is_uncertain_and_category_column_can_switch():
    ev = events(category_timing=["assigned_later", "recorded_after_window"], IUCN_CAT_outcome=["II", "II"], IUCN_CAT=["Ia", "II"])
    assert positive_events(ev, BROAD).tolist() == [False, True]
    assert uncertain_events(ev, BROAD).tolist() == [True, False]
    assert positive_events(ev.assign(category_timing="recorded_in_window"), STRICT, "IUCN_CAT").tolist() == [True, False]


def test_point_masks_are_outcome_and_year_specific():
    pts = pd.DataFrame({"source": ["event"] * 5 + ["baseline"], "IUCN_CAT_outcome": ["Ia", "III", "III", "III", "III", "Ia"],
                        "STATUS_YR": [2022, 2022, 1987, 0, 2025, 2022]})
    assert primary_points(pts, STRICT).tolist() == [True, False, False, False, False, False]
    assert primary_points(pts, BROAD).tolist() == [True, True, False, True, False, False]


def test_labels_remove_only_eligible_cells_and_never_positive_ones():
    population = np.array([True, True, True, False, True])
    positive = np.array([True, False, False, True, False])
    uncertain = np.array([True, True, False, True, False])
    dropped = np.array([False, False, False, False, True])
    assert label(population, positive, uncertain, dropped).tolist() == [POSITIVE, REMOVED, NEGATIVE, INELIGIBLE, REMOVED]


def test_new_land_clipping_with_old_protected_land_and_an_extension():
    # Two 10 km cells side by side. The baseline covers the whole of cell 0. An extension covers cell 0 and reaches
    # 2 km into cell 1. An upgrade lies inside the baseline in cell 0.
    squares = np.array([box(0, 0, 10_000, 10_000), box(10_000, 0, 20_000, 10_000)])
    tree = shapely.STRtree(squares)
    extension, upgrade = box(0, 0, 12_000, 10_000), box(500, 500, 9_500, 9_500)
    hits, parts, _ = part_hits(tree, [extension, upgrade])
    base_in_cell = pd.Series({0: box(0, 0, 10_000, 10_000)})
    hits["new_land"], _ = new_land_flags(hits, parts, squares, base_in_cell)
    flags = hits.set_index(["owner", "cell"]).new_land
    assert not flags[(0, 0)] and flags[(0, 1)]
    assert not flags[(1, 0)] and (1, 1) not in flags.index
    sample = np.ones(2, bool)
    pos = np.array([True, True])
    new = hits[hits.new_land]
    assert label(sample, cells_of(new, pos, 2), np.zeros(2, bool)).tolist() == [NEGATIVE, POSITIVE]
    assert label(sample, cells_of(hits, pos, 2), np.zeros(2, bool)).tolist() == [POSITIVE, POSITIVE]


def test_collection_parts_are_queried_like_the_whole():
    squares = np.array([box(0, 0, 1, 1), box(5, 5, 6, 6), box(9, 9, 10, 10)])
    coll = shapely.GeometryCollection([box(0.5, 0.5, 2, 2), shapely.MultiPolygon([box(5.5, 5.5, 7, 7)])])
    hits, _, _ = part_hits(shapely.STRtree(squares), [coll])
    assert sorted(hits.cell.unique()) == sorted(shapely.STRtree(squares).query(coll, predicate="intersects"))


def test_ap_treats_ties_as_one_threshold_like_sklearn():
    rng = np.random.default_rng(0)
    country = np.repeat([0, 1, 2], 200)
    score = rng.integers(0, 15, 600).astype(float)
    y = (rng.random(600) < 0.1).astype(float)
    m = ranking_metrics(country, score, y)
    for c in range(3):
        sel = country == c
        assert m.ap[c] == pytest.approx(average_precision_score(y[sel], score[sel]))
        assert m.prevalence[c] == pytest.approx(y[sel].mean())


def test_recall_gives_fractional_credit_for_the_tie_at_the_budget():
    # 20 cells, budget 5% = 1 cell. The top score is shared by 2 cells, one positive: expected recall 0.5 of 1 positive.
    score = np.r_[1.0, 1.0, np.zeros(18)]
    y = np.r_[1.0, np.zeros(19)]
    m = ranking_metrics(np.zeros(20, int), score, y, budgets={"recall5": 0.05})
    assert m.recall5[0] == pytest.approx(0.5)
    random = ranking_metrics(np.zeros(20, int), np.zeros(20), y, budgets={"recall5": 0.05})
    assert random.ap[0] == pytest.approx(0.05) and random.recall5[0] == pytest.approx(0.05)


def test_countries_without_positives_get_no_metric():
    m = ranking_metrics(np.array([0, 0, 1, 1]), np.array([2.0, 1.0, 2.0, 1.0]), np.array([1.0, 0, 0, 0]))
    assert m.ap[0] == 1.0 and np.isnan(m.ap[1]) and np.isnan(m.recall5[1])


def test_pooled_reweighting_equals_duplicating_whole_countries():
    rng = np.random.default_rng(1)
    country = np.repeat([0, 1, 2], 50)
    score = rng.random(150).round(2)
    y = (rng.random(150) < 0.2).astype(float)
    mult = np.array([2.0, 0.0, 1.0])
    reweighted = PooledRanking(country, score, y, 3).metrics(mult)
    keep = np.r_[np.flatnonzero(country == 0), np.flatnonzero(country == 0), np.flatnonzero(country == 2)]
    duplicated = PooledRanking(country[keep], score[keep], y[keep], 3).metrics()
    assert reweighted["ap"] == pytest.approx(duplicated["ap"])
    assert reweighted["recall5"] == pytest.approx(duplicated["recall5"])
    assert duplicated["ap"] == pytest.approx(average_precision_score(y[keep], score[keep]))


def test_bootstrap_resamples_countries_and_is_paired():
    draws = bootstrap_draws(3, n_boot=200, seed=5)
    assert draws.shape == (200, 3) and draws.max() < 3
    diff = np.array([0.1, 0.2, 0.6])
    est, lo, hi = country_bootstrap(diff, draws)
    assert est == pytest.approx(0.3)
    assert lo >= diff.min() and hi <= diff.max()
    assert country_bootstrap(np.zeros(3), draws) == (0.0, 0.0, 0.0)
    assert np.array_equal(draws, bootstrap_draws(3, n_boot=200, seed=5))


def rate_records():
    return pd.DataFrame({"WDPAID": [1, 1, 1, 2, 3, 4], "WDPA_PID": ["1_A", "1_A", "1_B", "2", "3", "4"],
                         "ISO3": ["AAA", "AAA", "AAA", "AAA;BBB", "BBB", "BBB"], "IUCN_CAT": ["Ia", "Ia", "II", "Ib", "Ia", "V"],
                         "AREA": [10.0, 10.0, 5.0, 4.0, 2.0, 50.0]})


def test_historical_records_count_parcels_once_and_sites_by_wdpaid():
    names = np.array(["AAA", "BBB"])
    strict = historical_records(rate_records(), ["Ia", "Ib"])
    rates = historical_rates(strict, np.zeros(2), np.array([1, 1]), names)
    assert rates["rate_area_reported"].tolist() == [14.0, 6.0]  # duplicated 1_A once, 2 listed in both countries
    assert rates["rate_sites"].tolist() == [2.0, 2.0]
    broad = historical_records(rate_records(), ["Ia", "Ib", "II", "V"])
    rates = historical_rates(broad, np.zeros(2), np.array([1, 1]), names)
    assert rates["rate_area_reported"].tolist() == [19.0, 56.0]
    assert rates["rate_sites"].tolist() == [2.0, 3.0]  # site 1 has two parcels, counted once


def test_rate_scores_depend_only_on_the_cutoff_population_not_on_labels():
    names = np.array(["AAA", "BBB"])
    country = np.array([0, 0, 0, 0, 1, 1])
    population = np.array([True, True, True, False, True, True])
    records = historical_records(rate_records(), ["Ia", "Ib"])
    denominator = np.bincount(country[population], minlength=2)
    first = historical_rates(records, np.array([3.0, 1.0]), denominator, names)
    # any 2021-2024 labelling of the same population leaves the scores unchanged
    for labels in [np.array([0, 1, 2, 3, 0, 2]), np.array([2, 2, 0, 3, 1, 1])]:
        assert population.sum() == (labels != 3).sum()
        again = historical_rates(records, np.array([3.0, 1.0]), denominator, names)
        for k in first:
            assert np.array_equal(first[k], again[k])
    assert first["rate_area_unique"].tolist() == [1.0, 0.5]


def test_unique_area_counts_overlapping_polygons_once():
    cell = box(0, 0, 10_000, 10_000)
    a, b = box(0, 0, 6_000, 10_000), box(4_000, 0, 10_000, 10_000)  # 60 + 60 km2, overlap 20 km2
    pieces = shapely.intersection(np.array([a, b, a]), cell)
    km2 = unique_area_by_country(pieces, np.array([0, 0, 1]), np.array([0, 1]), 2)
    assert km2 == pytest.approx([100.0, 60.0])
