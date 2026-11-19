"""T1 scoring under docs/t1_protocol.md version 1.1: within-country ranking of the cells that received new
protection during 2021 to 2024 by the frozen Mouillot et al. (2024) scores.

1. Guards before any score is evaluated: the protocol's sha256, a one-to-one join of the two score files on
   ISO3 and coordinates rounded to KEY_DECIMALS, 258,725 background cells with no missing score, and the
   frozen v1.1 label counts (EXPECTED).
2. Cells: 10 km Mollweide squares around the stored centres. Cell hits of every baseline polygon, event
   polygon (established and window-end footprints) and point record are computed once, part by part.
3. Labels per variant (primary and the non-refit sensitivities): positive, negative, removed or ineligible.
   Every variant takes events as positive only for their new land, the part outside the cutoff baseline. In
   the expansion population every eligible cell lies outside the baseline. In the sample population cells
   overlapping the baseline are positive or removed only if the event's land in the cell is not covered by it.
4. Metrics per country (AP, prevalence, AP over prevalence, Recall@5% and @1% with fractional credit for ties
   at the budget), equal-weight country summaries, pooled AP and Recall@5%, the cutoff-only baselines, and a
   paired country-clustered bootstrap of every model-minus-baseline difference.

Choices the protocol leaves open, fixed here before any score was evaluated (also in the manifest, CHOICES):
see CHOICES.

Inputs: the ledger outputs (DATA_DIR/wdpa_event_ledger/all), the Mouillot package data directory, and
DATA_DIR/t1_scoring/inputs (point_records.parquet, governance_202610.csv, d1_absence.csv).
Outputs in DATA_DIR/t1_scoring: labels.parquet, population.csv, country_metrics.csv, summary.csv,
bootstrap.csv, concentration.csv, manifest.json.

Usage:
    python -m scripts.regions.world.t1_scoring --package <Mouillot data dir>
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyreadr
import scipy.sparse as sp
import shapely
from pyproj import Transformer

from scripts.regions.world import paths
from scripts.regions.world.wdpa_event_ledger import EQUAL_AREA_CRS, IUCN_CATEGORIES, PATCH, QUALIFYING, equal_area

PROTOCOL_PATH = paths.REPO_ROOT / "docs" / "t1_protocol.md"
PROTOCOL_SHA256 = "6e610efc650773ad4edc8b7a86b77555009c55f21181ae1ae6c37ea18a06d980"
LEDGER_DIR = paths.DATA_DIR / "wdpa_event_ledger" / "all"
OUT_DIR = paths.DATA_DIR / "t1_scoring"
INPUT_DIR = OUT_DIR / "inputs"
SCORE_FILES = {"corr.rf.response": "res_prob_terrestrial.Rdata", "PredictedProtection": "final_predictions_terrestrial.Rdata"}
MODELS = list(SCORE_FILES)
N_BACKGROUND = 258_725
KEY_DECIMALS = 6
CELL_M = 10_000
OUTCOMES = {"IaIb": ["Ia", "Ib"], "I-VI": IUCN_CATEGORIES}
POSITIVE_ROLES = ["new_site", "candidate_extension"]
UNCERTAIN_ROLES = ["uncertain_extension", "undated_change"]
PRIMARY_TIMING = ["recorded_in_window", "recorded_after_window"]
GOVERNMENT = ["Federal or national ministry or agency", "Sub-national ministry or agency", "Government-delegated management"]
RECONSTRUCTED = sorted(PATCH)
WINDOW_RELEASES = ("202101", "202412")
BUDGETS = {"recall5": 0.05, "recall1": 0.01}
MIN_ELIGIBLE = 100
MIN_POSITIVE = {"min5": 5, "min1": 1}
COVER_MIN = 0.10
NEW_LAND_MIN_M2 = 1.0
RATE_YEARS = (2015, 2019)
N_BOOT = 1000
SEED = 42
NEGATIVE, POSITIVE, REMOVED, INELIGIBLE = 0, 1, 2, 3
EXPECTED = {"excluded": 18_133, "IaIb": {"positive": 98, "removed": 171, "eligible": 240_421},
            "I-VI": {"positive": 1_559, "removed": 3_145, "eligible": 237_447}}
CHOICES = [
    "Cells reached only through events a sensitivity drops (6, 10, 11b, 14, D1 bound) are removed, not relabelled negative, "
    "since the land was still designated. Events a sensitivity places outside the window (8: STATUS_YR 2024) leave their "
    "cells negative, as post-window events do in the primary.",
    "Sensitivity 5: a cell whose positive new land covers less than 10% of it is negative unless an uncertain record "
    "touches it, then removed.",
    "Sensitivity 8 keeps the footprints up to 202412, since no footprint at the end of 2023 was extracted.",
    "Sensitivity 11 is reported as 11a (category at the observation release) and 11b (events whose category was "
    "recorded after the window removed).",
    "Sensitivity 12 is reported as 12a (post-2024 events removed), 12b (cells containing a baseline point or a primary "
    "event point removed) and 12c (cells touched by their reported-area circles removed).",
    "Sensitivity 10: government governance is GOV_TYPE federal or national, sub-national, or government-delegated in "
    "release 202610. Positive events absent from 202610 have no governance and are dropped.",
    "Recall budgets are fractional (q times the number of eligible cells), with fractional credit for the tie at the "
    "budget, the expected recall under random ordering.",
    "Random ranking is a constant score: under the tie rule its AP is the prevalence and its Recall@q is q.",
    "Distance baseline: distance from the cell centre to the nearest cutoff baseline polygon in Mollweide (0 inside). "
    "Baseline points are not used. Sensitivity 9 recomputes it without the four reconstructed countries' polygons.",
    "Environment and accessibility baseline: the mean of within-country average ranks of tt (higher first), hf "
    "(lower first) and Altitude (higher first).",
    "Between-country baseline, pooled metrics only, built from cutoff baseline records of the outcome's categories with "
    "STATUS Designated, Inscribed or Established, STATUS_YR 2015 to 2019, terrestrial or coastal. Three forms: "
    "rate_area_reported, the sum of reported AREA (km2) over distinct parcels (WDPA_PID), polygons and points, which can "
    "overlap and so is not unique area; rate_sites, the number of distinct sites (WDPAID); rate_area_unique, the area "
    "(km2) of the union of those polygons inside the country's sampled 10 km cells (points have no footprint and are "
    "left out). A parcel listing several ISO3 counts for each in the reported and site forms. Each is divided by the "
    "number of the country's cells in the variant's population (expansion population, or every cell in sensitivity 2), "
    "which is fixed at the cutoff and does not depend on any 2021 to 2024 label. The unit is per sampled at-risk 10 km "
    "cell: the stored cells are a sample of each country's land, so this is not a rate per country area.",
    "Correction dated 2026-11-19, made after the first results (commit 520be4f) and recorded as such: the first run divided "
    "by the variant's eligible cells, which depend on 2021 to 2024 removals, counted parcel rows as sites, and counted 9 "
    "duplicated Chilean parcel rows (Kawesqar, II and IV) more than once. Those pooled rate results are superseded and "
    "kept in their own output directory. No label rule, score or within-country baseline changed.",
    "Sample population: new land in a baseline-overlapping cell needs more than 1 m2 outside the baseline (numerical "
    "tolerance, not a tuned threshold). A point record in such a cell counts only if it lies outside the baseline.",
    "Site counts per country are distinct WDPAIDs of positive events touching the country's positive cells. Each "
    "positive cell is assigned to the event parcel with the largest area in it. The 10 largest events are the 10 sites "
    "(WDPAID) with most assigned positive cells.",
    "Pooled metrics and their bootstrap use every country with eligible cells.",
]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_protocol(path=PROTOCOL_PATH, expected=PROTOCOL_SHA256):
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"{path} sha256 {digest} is not the frozen v1.1 {expected}")
    return digest


def cell_keys(frame):
    """ISO3 and coordinates rounded to KEY_DECIMALS, the stable cell key (float serialisation drifts by ~1e-14)."""
    return pd.DataFrame({"ISO3": frame.ISO3.astype(str).to_numpy(), "long_key": frame.long.round(KEY_DECIMALS).to_numpy(),
                         "lat_key": frame.lat.round(KEY_DECIMALS).to_numpy()}, index=frame.index)


def join_on_keys(left, right, n_expected=None):
    """One-to-one outer join on cell keys. Raises on duplicate keys, unmatched rows or a wrong count, never drops cells."""
    keys = ["ISO3", "long_key", "lat_key"]
    a = pd.concat([cell_keys(left), left.drop(columns=["ISO3", "long", "lat"], errors="ignore")], axis=1)
    b = pd.concat([cell_keys(right), right.drop(columns=["ISO3", "long", "lat"], errors="ignore")], axis=1)
    for name, f in [("left", a), ("right", b)]:
        if f.duplicated(keys).any():
            raise ValueError(f"duplicate cell keys in {name}")
    out = a.merge(b, on=keys, how="outer", validate="one_to_one", indicator=True)
    if (out["_merge"] != "both").any():
        raise ValueError(f"{(out['_merge'] != 'both').sum()} cells unmatched in the key join")
    if n_expected is not None and len(out) != n_expected:
        raise ValueError(f"{len(out)} matched cells, expected {n_expected}")
    return out.drop(columns="_merge")


def load_cells(package):
    """Background cells with both frozen scores and the baseline predictors, checked for complete coverage."""
    frames = {}
    for score, file in SCORE_FILES.items():
        r = pyreadr.read_r(str(Path(package) / file)).popitem()[1]
        r = r.loc[:, ~r.columns.duplicated()]
        r = r[r.IUCN_CAT == "NONE"]
        cols = ["ISO3", "long", "lat", score] + (["tt", "hf", "Altitude"] if score == MODELS[0] else [])
        frames[score] = r[cols].reset_index(drop=True)
    first = frames[MODELS[0]]
    cells = join_on_keys(first, frames[MODELS[1]], N_BACKGROUND)
    coords = cell_keys(first).assign(long=first.long.to_numpy(), lat=first.lat.to_numpy())
    cells = cells.merge(coords, on=["ISO3", "long_key", "lat_key"], validate="one_to_one")
    if cells[MODELS + ["tt", "hf", "Altitude"]].isna().any().any():
        raise ValueError("missing score or predictor values")
    return cells.sort_values(["ISO3", "long_key", "lat_key"]).reset_index(drop=True)


def cell_squares(cells):
    x, y = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True).transform(cells.long.to_numpy(), cells.lat.to_numpy())
    return shapely.box(x - CELL_M / 2, y - CELL_M / 2, x + CELL_M / 2, y + CELL_M / 2), shapely.points(x, y)


def all_parts(geoms):
    """Every part of each geometry (collections and multi-geometries split twice), with its owner position."""
    parts, owner = shapely.get_parts(geoms, return_index=True)
    parts, inner = shapely.get_parts(parts, return_index=True)
    return parts, owner[inner]


def part_hits(tree, geoms):
    """(owner, cell, part) rows of intersecting pairs. Querying by part keeps GEOS on its indexed path."""
    parts, owner = all_parts(shapely.make_valid(np.asarray(geoms, dtype=object)))
    k, cell = tree.query(parts, predicate="intersects")
    return pd.DataFrame({"owner": owner[k], "cell": cell, "part": k}), parts, owner


def to_equal_area(geoms):
    geoms = np.asarray(geoms, dtype=object)
    keep = np.array([g is not None and not g.is_empty for g in geoms], bool)
    out = np.full(len(geoms), None, dtype=object)
    if keep.any():
        out[keep] = equal_area(gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms[keep], crs="EPSG:4326")))
    return out


def new_land_flags(hits, parts, squares, base_in_cell):
    """True for each hit row whose part has more than NEW_LAND_MIN_M2 in the cell outside the cutoff baseline, and the
    number of pairs computed on the 1 m grid. base_in_cell maps a cell position to the union of baseline polygons in
    it (cells absent have no baseline)."""
    flags = np.ones(len(hits), bool)
    inside = hits.cell.isin(base_in_cell.index).to_numpy()
    n_grid = 0
    if inside.any():
        h = hits[inside]
        clip = shapely.make_valid(shapely.intersection(parts[h.part.to_numpy()], squares[h.cell.to_numpy()]))
        base = shapely.make_valid(base_in_cell.loc[h.cell].to_numpy())
        area, n_grid = new_land_area(clip, base)
        flags[inside] = area > NEW_LAND_MIN_M2
    return flags, n_grid


def new_land_area(clip, base, chunk=5_000):
    """Area of clip outside base, element-wise, and how many pairs needed the fallback. Where GEOS cannot overlay a pair
    (degenerate slivers), that pair is recomputed on a 1 m precision grid, far below the cell size."""
    out, n_grid = np.empty(len(clip)), 0
    for lo in range(0, len(clip), chunk):
        a, b = clip[lo:lo + chunk], base[lo:lo + chunk]
        try:
            out[lo:lo + chunk] = shapely.area(shapely.difference(a, b))
        except shapely.errors.GEOSException:
            for k in range(len(a)):
                try:
                    out[lo + k] = shapely.area(shapely.difference(a[k], b[k]))
                except shapely.errors.GEOSException:
                    out[lo + k] = shapely.area(shapely.difference(a[k], b[k], grid_size=1.0))
                    n_grid += 1
    return out, n_grid


def positive_events(ev, cats, cat_col="IUCN_CAT_outcome"):
    return (ev[cat_col].isin(cats) & ev.expansion_role.isin(POSITIVE_ROLES) & (ev.DESIG_TYPE != "International")
            & ev.category_timing.isin(PRIMARY_TIMING)).to_numpy()


def uncertain_events(ev, cats, cat_col="IUCN_CAT_outcome", post_window_uncertain=False):
    in_cat = ev[cat_col].isin(cats).to_numpy()
    not_positive = ev.expansion_role.isin(POSITIVE_ROLES).to_numpy() & ~positive_events(ev, cats, cat_col)
    unresolved = ev.event_class.isin([3, 5]).to_numpy()
    if not post_window_uncertain:
        unresolved &= (ev.reason != "designated_after_window").to_numpy()
    return in_cat & (not_positive | ev.expansion_role.isin(UNCERTAIN_ROLES).to_numpy() | unresolved)


def primary_points(points, cats):
    """Event point records whose cell is removed: the outcome's categories, STATUS_YR 2020 to 2024 or missing."""
    year = pd.to_numeric(points.STATUS_YR, errors="coerce").fillna(0)
    return ((points.source == "event") & points.IUCN_CAT_outcome.isin(cats) & ((year == 0) | year.between(2020, 2024))).to_numpy()


def cells_of(hits, event_mask, n_cells):
    hit = np.zeros(n_cells, bool)
    hit[hits.cell.to_numpy()[event_mask[hits.owner.to_numpy()]]] = True
    return hit


def label(population, positive, uncertain, dropped=None):
    """Cell labels: positive where touched by a positive event, removed where touched only by uncertain records or
    dropped events, negative elsewhere in the population, ineligible outside it."""
    out = np.full(len(population), INELIGIBLE, np.int8)
    out[population] = NEGATIVE
    rem = population & ~positive & (uncertain | (dropped if dropped is not None else False))
    out[rem] = REMOVED
    out[population & positive] = POSITIVE
    return out


def tie_groups(country, score):
    """Order cells by country, then score descending, and number the tie groups (consecutive equal scores of one country)."""
    order = np.lexsort((-score, country))
    c, s = country[order], score[order]
    new = np.ones(len(order), bool)
    new[1:] = (c[1:] != c[:-1]) | (s[1:] != s[:-1])
    return order, np.cumsum(new) - 1, c


def ranking_metrics(country, score, y, budgets=BUDGETS, weight=None):
    """AP and Recall@q per country (country codes 0..C-1), ties as one threshold and fractional credit at the budget.
    Returns a DataFrame indexed by country code with n, positives, prevalence, ap, ap_over_prevalence and the recalls."""
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    order, group, gc = tie_groups(country, score)
    yw, ww = (y * w)[order], w[order]
    n_groups = group[-1] + 1 if len(group) else 0
    tp_g = np.bincount(group, yw, n_groups)
    n_g = np.bincount(group, ww, n_groups)
    country_g = np.zeros(n_groups, int)
    country_g[group] = gc
    return _metrics_from_groups(country_g, tp_g, n_g, budgets)


def _metrics_from_groups(country_g, tp_g, n_g, budgets):
    n_c = np.bincount(country_g, n_g)
    p_c = np.bincount(country_g, tp_g)
    start = np.r_[0, np.flatnonzero(np.diff(country_g)) + 1]
    cum_tp = np.cumsum(tp_g)
    cum_n = np.cumsum(n_g)
    off_tp = np.repeat(np.r_[0, cum_tp[start[1:] - 1]], np.diff(np.r_[start, len(tp_g)]))
    off_n = np.repeat(np.r_[0, cum_n[start[1:] - 1]], np.diff(np.r_[start, len(tp_g)]))
    ctp, cn = cum_tp - off_tp, cum_n - off_n
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = np.bincount(country_g, tp_g * np.where(cn > 0, ctp / cn, 0), len(n_c)) / p_c
        out = pd.DataFrame({"n": n_c, "positives": p_c, "prevalence": p_c / n_c, "ap": ap})
        out["ap_over_prevalence"] = out.ap / out.prevalence
        for name, q in budgets.items():
            frac = np.clip((q * n_c[country_g] - (cn - n_g)) / n_g, 0, 1)
            out[name] = np.bincount(country_g, frac * tp_g, len(n_c)) / p_c
    out.loc[out.positives == 0, ["ap", "ap_over_prevalence"] + list(budgets)] = np.nan
    return out


class PooledRanking:
    """Pooled AP and Recall@q with one global budget, reweighted by country multiplicities for the bootstrap."""

    def __init__(self, country, score, y, n_countries, budgets=None):
        self.budgets = budgets or {"recall5": BUDGETS["recall5"]}
        order = np.argsort(-score, kind="stable")
        s = score[order]
        group = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
        g, c = group, country[order]
        shape = (group[-1] + 1, n_countries)
        self.tp = sp.csr_matrix((y[order].astype(float), (g, c)), shape=shape)
        self.n = sp.csr_matrix((np.ones(len(g)), (g, c)), shape=shape)

    def metrics(self, multiplicity=None):
        """Dict of ap and the recalls for country multiplicities (None: every country once)."""
        m = np.ones(self.n.shape[1]) if multiplicity is None else multiplicity
        tp_g, n_g = self.tp @ m, self.n @ m
        keep = n_g > 0
        tp_g, n_g = tp_g[keep], n_g[keep]
        p, cn = tp_g.sum(), np.cumsum(n_g)
        if p == 0:
            return {"ap": np.nan, **{k: np.nan for k in self.budgets}}
        out = {"ap": float((tp_g * np.cumsum(tp_g) / cn).sum() / p)}
        for name, q in self.budgets.items():
            out[name] = float((np.clip((q * cn[-1] - (cn - n_g)) / n_g, 0, 1) * tp_g).sum() / p)
        return out


def historical_records(rec, cats):
    """Cutoff baseline records of the outcome's categories (already restricted to RATE_YEARS and qualifying status), one
    row per distinct parcel and listed ISO3."""
    r = rec[rec.IUCN_CAT.isin(cats)].drop_duplicates("WDPA_PID")
    return r.assign(ISO3=r.ISO3.astype(str).str.split(";")).explode("ISO3")


def unique_area_by_country(pieces, piece_cell, cell_country, n_countries):
    """km2 per country of the union of polygon pieces already clipped to their cell, so overlaps count once."""
    if len(pieces) == 0:
        return np.zeros(n_countries)
    per_cell = pd.Series(list(pieces)).groupby(np.asarray(piece_cell)).agg(lambda g: shapely.area(shapely.union_all(g.to_numpy())))
    return np.bincount(cell_country[per_cell.index.to_numpy()], weights=per_cell.to_numpy() / 1e6, minlength=n_countries)


def historical_rates(records, unique_km2, denominator, country_names):
    """Per-country rate scores: reported area, distinct sites and unique area, each per population cell."""
    reported = records.groupby("ISO3").AREA.sum().reindex(country_names, fill_value=0).to_numpy()
    sites = records.groupby("ISO3").WDPAID.nunique().reindex(country_names, fill_value=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return {name: np.nan_to_num(v / denominator) for name, v in
                [("rate_area_reported", reported), ("rate_sites", sites), ("rate_area_unique", np.asarray(unique_km2, float))]}


def within_country_rank(country, values):
    return pd.Series(values).groupby(country).rank(method="average", pct=True).to_numpy()


def bootstrap_draws(n_countries, n_boot=N_BOOT, seed=SEED):
    """Country indices drawn with replacement, one row per replicate. Cells are never resampled on their own."""
    return np.random.default_rng(seed).integers(0, n_countries, size=(n_boot, n_countries))


def country_bootstrap(diff, draws):
    """Mean country difference, and its percentile 95% interval over country resamples (paired: diff is per country)."""
    reps = diff[draws].mean(axis=1)
    return diff.mean(), np.percentile(reps, 2.5), np.percentile(reps, 97.5)


# ------------------------------------------------------------------ run


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def log(start, message):
    print(f"[{time.time() - start:6.0f} s] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, help="Mouillot et al. (2024) package data directory")
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory (inputs are always read from INPUT_DIR)")
    args = parser.parse_args()
    out_dir = Path(args.out)
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": str(PROTOCOL_PATH), "protocol_sha256": check_protocol(), "code_commit": git_commit(),
                "choices": CHOICES, "inputs": {}}
    log(start, "protocol hash verified")

    cells = load_cells(args.package)
    n = len(cells)
    log(start, f"{n} background cells, both scores matched one to one, no missing values")
    country_names = np.array(sorted(cells.ISO3.unique()))
    country = np.searchsorted(country_names, cells.ISO3.to_numpy())
    squares, centres = cell_squares(cells)
    tree = shapely.STRtree(squares)

    base = gpd.read_parquet(LEDGER_DIR / "cutoff_baseline_polygons.parquet",
                            columns=["ISO3", "IUCN_CAT", "STATUS", "STATUS_YR", "MARINE", "geometry"])
    base_geom = np.asarray(equal_area(base))
    base_hits, base_parts, base_owner = part_hits(tree, base_geom)
    excluded = np.zeros(n, bool)
    excluded[base_hits.cell] = True
    any_iucn = np.zeros(n, bool)
    any_iucn[base_hits.cell[base.IUCN_CAT.isin(IUCN_CATEGORIES).to_numpy()[base_hits.owner]]] = True
    if excluded.sum() != EXPECTED["excluded"]:
        raise ValueError(f"{excluded.sum()} excluded cells, frozen count {EXPECTED['excluded']}")
    expansion = ~excluded
    log(start, f"expansion population: {expansion.sum()} cells, {excluded.sum()} excluded")
    clip = shapely.intersection(base_parts[base_hits.part.to_numpy()], squares[base_hits.cell.to_numpy()])
    base_in_cell = pd.Series(clip).groupby(base_hits.cell.to_numpy()).agg(lambda g: shapely.union_all(g.to_numpy()))

    base_tree = shapely.STRtree(base_parts)
    _, dist = base_tree.query_nearest(centres, return_distance=True, all_matches=False)
    base_iso = base.ISO3.astype(str).to_numpy()[base_owner]
    keep = ~pd.Series(base_iso).str.contains("|".join(RECONSTRUCTED)).to_numpy()
    _, dist_no_rec = shapely.STRtree(base_parts[keep]).query_nearest(centres, return_distance=True, all_matches=False)
    log(start, "distance baseline computed")

    ev = gpd.read_parquet(LEDGER_DIR / "events.parquet", columns=[
        "WDPAID", "WDPA_PID", "release_pid", "ISO3", "event_class", "reason", "expansion_role", "DESIG_TYPE", "IUCN_CAT",
        "IUCN_CAT_outcome", "category_timing", "STATUS_YR", "geometry", "geometry_window_end"])
    ev = ev[ev.IUCN_CAT_outcome.isin(IUCN_CATEGORIES) | ev.IUCN_CAT.isin(IUCN_CATEGORIES)].reset_index(drop=True)
    gov = pd.read_csv(INPUT_DIR / "governance_202610.csv", encoding="utf-8-sig", dtype=str).drop_duplicates("WDPA_PID")
    ev["GOV_TYPE"] = ev.release_pid.map(gov.set_index("WDPA_PID").GOV_TYPE)
    d1 = pd.read_csv(INPUT_DIR / "d1_absence.csv", encoding="utf-8-sig", dtype=str)
    ev["d1_unresolved"] = ev.WDPA_PID.isin(d1.loc[d1.absence_type == "unresolved", "WDPA_PID"])
    hits, parts, _ = part_hits(tree, to_equal_area(ev.geometry.values))
    hits["new_land"], n_grid = new_land_flags(hits, parts, squares, base_in_cell)
    manifest["new_land_pairs_on_1m_grid"] = n_grid
    role = ev.expansion_role.isin(POSITIVE_ROLES).to_numpy()
    end_hits, _, _ = part_hits(tree, to_equal_area(np.where(role, ev.geometry_window_end.values, None)))
    log(start, f"{len(ev)} events, {len(hits)} part-cell hits, {(~hits.new_land).sum()} without new land, {n_grid} on the 1 m grid")

    points = gpd.read_parquet(INPUT_DIR / "point_records.parquet")
    p_idx, p_cell = tree.query(points.geometry.values, predicate="intersects")
    in_base = np.zeros(len(points), bool)
    in_base[base_tree.query(points.geometry.values, predicate="intersects")[0]] = True
    radius = np.sqrt(points.AREA.fillna(0).clip(lower=0).to_numpy() * 1e6 / np.pi)
    c_idx, c_cell = tree.query(shapely.buffer(points.geometry.values, radius), predicate="intersects")

    def point_cells(mask, circles=False, new_land_only=False):
        idx, cell = (c_idx, c_cell) if circles else (p_idx, p_cell)
        m = mask & (~in_base if new_land_only else True)
        hit = np.zeros(n, bool)
        hit[cell[m[idx]]] = True
        return hit

    for name in ["point_records.parquet", "governance_202610.csv", "d1_absence.csv"]:
        manifest["inputs"][str(INPUT_DIR / name)] = sha256(INPUT_DIR / name)
    for name in ["cutoff_baseline_polygons.parquet", "cutoff_baseline.parquet", "events.parquet", "completeness.csv"]:
        manifest["inputs"][str(LEDGER_DIR / name)] = sha256(LEDGER_DIR / name)
    for file in SCORE_FILES.values():
        manifest["inputs"][str(Path(args.package) / file)] = sha256(Path(args.package) / file)
    log(start, "input hashes recorded")

    # areas of positive-role parts in eligible cells, for assignment and sensitivity 5
    ph = hits[role[hits.owner.to_numpy()]].copy()
    ph["piece"] = shapely.intersection(parts[ph.part.to_numpy()], squares[ph.cell.to_numpy()])
    ph["area"] = shapely.area(ph.piece.to_numpy())
    pair_area = ph.groupby(["owner", "cell"]).area.sum().reset_index()

    comp = pd.read_csv(LEDGER_DIR / "completeness.csv", encoding="utf-8-sig", dtype={"release": str})
    w = comp[comp.release.between(*WINDOW_RELEASES) & (comp.n_transient > 0)]
    transient = sorted(w.ISO3.unique())
    status_yr = pd.to_numeric(ev.STATUS_YR, errors="coerce").fillna(0).to_numpy()

    labels, notes, positive_masks, populations = {}, {}, {}, {}
    for outcome, cats in OUTCOMES.items():
        pos_ev = positive_events(ev, cats)
        unc_ev = uncertain_events(ev, cats)
        pts = primary_points(points, cats)

        def lab(pos_mask=pos_ev, unc_mask=unc_ev, population=expansion, pos_hits=None, point_mask=pts, extra_unc=None,
                drop_mask=None, new_land_only=False):
            h = hits[hits.new_land] if new_land_only else hits
            positive = cells_of(h if pos_hits is None else pos_hits, pos_mask, n)
            uncertain = cells_of(h, unc_mask, n) | point_cells(point_mask, new_land_only=new_land_only)
            if extra_unc is not None:
                uncertain |= extra_unc
            dropped = cells_of(h, drop_mask, n) if drop_mask is not None else None
            return label(population, positive, uncertain, dropped)

        primary = lab()
        got = {"positive": int((primary == POSITIVE).sum()), "removed": int((primary == REMOVED).sum()),
               "eligible": int(np.isin(primary, [NEGATIVE, POSITIVE]).sum())}
        if got != EXPECTED[outcome]:
            raise ValueError(f"{outcome} primary counts {got} differ from the frozen v1.1 counts {EXPECTED[outcome]}")
        labels[(outcome, "primary")] = primary

        # assignment of positive cells to the parcel with the largest area, and the 10 largest sites
        pa = pair_area[pos_ev[pair_area.owner.to_numpy()] & (primary[pair_area.cell.to_numpy()] == POSITIVE)]
        assigned = pa.loc[pa.groupby("cell").area.idxmax()]
        site_cells = ev.WDPAID.to_numpy()[assigned.owner.to_numpy()]
        top10 = pd.Series(site_cells).value_counts().head(10)
        notes[(outcome, "top10")] = top10
        notes[(outcome, "assigned")] = assigned
        top_ev = ev.WDPAID.isin(top10.index).to_numpy()

        cls3 = (ev.event_class == 3).to_numpy() & ev.IUCN_CAT_outcome.isin(cats).to_numpy()
        cls3_pos = cls3 & (ev.DESIG_TYPE != "International").to_numpy() & ev.category_timing.isin(PRIMARY_TIMING).to_numpy()
        cls24 = (ev.event_class.isin([2, 4]).to_numpy() & ev.IUCN_CAT_outcome.isin(cats).to_numpy()
                 & (ev.DESIG_TYPE != "International").to_numpy() & ev.category_timing.isin(PRIMARY_TIMING).to_numpy())
        in_pos = ph[pos_ev[ph.owner.to_numpy()] & (primary[ph.cell.to_numpy()] == POSITIVE)]
        covered = in_pos.groupby("cell").piece.agg(lambda g: shapely.area(shapely.union_all(g.to_numpy())))
        low_cover = np.zeros(n, bool)
        low_cover[covered.index[covered.to_numpy() < COVER_MIN * CELL_M ** 2]] = True
        gov_ok = ev.GOV_TYPE.isin(GOVERNMENT).to_numpy()
        after = (ev.category_timing == "recorded_after_window").to_numpy()
        ext = (ev.expansion_role == "candidate_extension").to_numpy()
        post = (ev.reason == "designated_after_window").to_numpy() & ev.IUCN_CAT_outcome.isin(cats).to_numpy()
        base_pt = (points.source == "baseline").to_numpy()
        drop_rec = ~np.isin(country, np.flatnonzero(np.isin(country_names, RECONSTRUCTED)))
        drop_tr = ~np.isin(country, np.flatnonzero(np.isin(country_names, transient)))
        p5 = primary.copy()
        p5[(p5 == POSITIVE) & low_cover] = NEGATIVE
        p5[(p5 == NEGATIVE) & low_cover & (cells_of(hits, unc_ev, n) | point_cells(pts))] = REMOVED

        alt_pos = {"s03a_class3_positive": pos_ev | cls3_pos, "s04_class2_class4_events": pos_ev | cls24,
                   "s06_drop_top10": pos_ev & ~top_ev, "s08_window_2021_2023": pos_ev & (status_yr <= 2023),
                   "s10_government_only": pos_ev & gov_ok, "s11a_category_at_observation": positive_events(ev, cats, "IUCN_CAT"),
                   "s11b_drop_recorded_after_window": pos_ev & ~after, "s14_new_sites_only": pos_ev & ~ext,
                   "bound_d1_unresolved_removed": pos_ev & ~ev.d1_unresolved.to_numpy()}
        variants = {
            "s02_sample_population": lab(population=np.ones(n, bool), new_land_only=True),
            "s03a_class3_positive": lab(pos_mask=pos_ev | cls3_pos, unc_mask=unc_ev & ~cls3_pos),
            "s03b_class3_negative": lab(unc_mask=unc_ev & ~cls3),
            "s04_class2_class4_events": lab(pos_mask=pos_ev | cls24, unc_mask=unc_ev & ~cls24),
            "s05_cover10": p5,
            "s06_drop_top10": lab(pos_mask=pos_ev & ~top_ev, drop_mask=pos_ev & top_ev),
            "s07_drop_transient_countries": lab(population=expansion & drop_tr),
            "s08_window_2021_2023": lab(pos_mask=pos_ev & (status_yr <= 2023)),
            "s09_drop_reconstructed_countries": lab(population=expansion & drop_rec),
            "s10_government_only": lab(pos_mask=pos_ev & gov_ok, drop_mask=pos_ev & ~gov_ok),
            "s11a_category_at_observation": lab(pos_mask=positive_events(ev, cats, "IUCN_CAT"),
                                                unc_mask=uncertain_events(ev, cats, "IUCN_CAT")),
            "s11b_drop_recorded_after_window": lab(pos_mask=pos_ev & ~after, drop_mask=pos_ev & after),
            "s12a_post2024_removed": lab(unc_mask=unc_ev | post),
            "s12b_point_cells_removed": lab(point_mask=pts | base_pt),
            "s12c_point_circles_removed": lab(extra_unc=point_cells(pts | base_pt, circles=True)),
            "s13_window_end_footprint": lab(pos_hits=end_hits),
            "s14_new_sites_only": lab(pos_mask=pos_ev & ~ext, drop_mask=pos_ev & ext),
            "bound_removed_positive": np.where(primary == REMOVED, POSITIVE, primary).astype(np.int8),
            "bound_removed_negative": np.where(primary == REMOVED, NEGATIVE, primary).astype(np.int8),
            "bound_d1_unresolved_removed": lab(pos_mask=pos_ev & ~ev.d1_unresolved.to_numpy(),
                                               drop_mask=pos_ev & ev.d1_unresolved.to_numpy()),
        }
        positive_masks[(outcome, "primary")] = pos_ev
        populations[(outcome, "primary")] = expansion
        for name in variants:
            populations[(outcome, name)] = {"s02_sample_population": np.ones(n, bool), "s07_drop_transient_countries": expansion & drop_tr,
                                            "s09_drop_reconstructed_countries": expansion & drop_rec}.get(name, expansion)
        for name, v in variants.items():
            labels[(outcome, name)] = v
            positive_masks[(outcome, name)] = alt_pos.get(name, pos_ev)
        log(start, f"{outcome}: {len(variants) + 1} label variants")

    label_table = cells[["ISO3", "long_key", "lat_key"]].copy()
    for (outcome, name), v in labels.items():
        label_table[f"{outcome}|{name}"] = v
    label_table.to_parquet(out_dir / "labels.parquet")
    pop = pd.DataFrame({"ISO3": cells.ISO3, "cells": 1, "excluded": excluded, "excluded_iucn": any_iucn})
    pop.groupby("ISO3").sum().reset_index().to_csv(out_dir / "population.csv", index=False, encoding="utf-8-sig")

    # scores and baselines
    env = (within_country_rank(country, cells.tt.to_numpy()) + within_country_rank(country, -cells.hf.to_numpy())
           + within_country_rank(country, cells.Altitude.to_numpy())) / 3
    within_scores = {m: cells[m].to_numpy(float) for m in MODELS}
    within_scores |= {"random": np.zeros(n), "distance": -dist, "environment": env}
    rec = pd.read_parquet(LEDGER_DIR / "cutoff_baseline.parquet", columns=["ISO3", "STATUS", "STATUS_YR", "IUCN_CAT", "MARINE", "AREA"])
    rec = rec[pd.to_numeric(rec.STATUS_YR, errors="coerce").between(*RATE_YEARS) & rec.MARINE.astype(str).isin(["0", "1"])
              & rec.STATUS.isin(QUALIFYING)]
    rate_poly = (pd.to_numeric(base.STATUS_YR, errors="coerce").between(*RATE_YEARS) & base.MARINE.astype(str).isin(["0", "1"])
                 & base.STATUS.isin(QUALIFYING)).to_numpy()
    hist = {}
    for outcome, cats in OUTCOMES.items():
        keep_rows = (rate_poly & base.IUCN_CAT.isin(cats).to_numpy())[base_hits.owner.to_numpy()]
        hist[outcome] = (historical_records(rec, cats),
                         unique_area_by_country(clip[keep_rows], base_hits.cell.to_numpy()[keep_rows], country, len(country_names)))
    log(start, "historical rate numerators computed")

    rows, summary, boot, conc = [], [], [], []
    for (outcome, variant), lab_v in labels.items():
        cats = OUTCOMES[outcome]
        elig = np.isin(lab_v, [NEGATIVE, POSITIVE])
        y = (lab_v == POSITIVE).astype(float)
        scores = dict(within_scores)
        if variant == "s09_drop_reconstructed_countries":
            scores["distance"] = -dist_no_rec
        n_elig_c = np.bincount(country[elig], minlength=len(country_names))
        n_pop_c = np.bincount(country[populations[(outcome, variant)]], minlength=len(country_names))
        pooled_extra = {k: v[country] for k, v in historical_rates(*hist[outcome], n_pop_c, country_names).items()}
        pos_ev = positive_masks[(outcome, variant)]
        pos_cells = np.flatnonzero(lab_v == POSITIVE)
        hh = hits[hits.cell.isin(pos_cells) & pos_ev[hits.owner.to_numpy()]]
        n_sites = pd.Series(ev.WDPAID.to_numpy()[hh.owner]).groupby(country[hh.cell.to_numpy()]).nunique()
        per = {}
        for s_name, s in scores.items():
            m = ranking_metrics(country[elig], s[elig], y[elig])
            m = m.reindex(range(len(country_names)))
            m["ISO3"] = country_names
            m["n_sites"] = n_sites.reindex(range(len(country_names)), fill_value=0).to_numpy()
            m["n_removed"] = np.bincount(country[lab_v == REMOVED], minlength=len(country_names))
            m["estimable"] = (m.n >= MIN_ELIGIBLE) & (m.positives >= MIN_POSITIVE["min5"])
            m["estimable_min1"] = (m.n >= MIN_ELIGIBLE) & (m.positives >= MIN_POSITIVE["min1"])
            m = m[m.n.fillna(0) > 0].assign(outcome=outcome, variant=variant, score=s_name)
            per[s_name] = m
            rows.append(m)
        pooled = {s_name: PooledRanking(country[elig], s[elig], y[elig], len(country_names))
                  for s_name, s in (scores | pooled_extra).items()}
        pooled_point = {k: v.metrics() for k, v in pooled.items()}
        present = np.flatnonzero(n_elig_c > 0)
        draws_all = bootstrap_draws(len(present))
        mult = np.zeros((N_BOOT, len(country_names)))
        for b in range(N_BOOT):
            mult[b] = np.bincount(present[draws_all[b]], minlength=len(country_names))
        pooled_reps = {}
        for k, v in pooled.items():
            reps = [v.metrics(mult[b]) for b in range(N_BOOT)]
            pooled_reps[k] = np.array([[r["ap"], r["recall5"]] for r in reps])
        for set_name, col in [("min5", "estimable"), ("min1", "estimable_min1")]:
            for s_name, m in per.items():
                sel = m[m[col]]
                summary.append({"outcome": outcome, "variant": variant, "country_set": set_name, "score": s_name,
                                "n_countries": len(sel), **{k: sel[k].mean() for k in ["prevalence", "ap", "ap_over_prevalence", "recall5", "recall1"]},
                                "pooled_ap": pooled_point[s_name]["ap"], "pooled_recall5": pooled_point[s_name]["recall5"]})
            for s_name in pooled_extra:
                summary.append({"outcome": outcome, "variant": variant, "country_set": set_name, "score": s_name,
                                "pooled_ap": pooled_point[s_name]["ap"], "pooled_recall5": pooled_point[s_name]["recall5"]})
            for model in MODELS:
                sel = per[model][col].to_numpy()
                draws = bootstrap_draws(sel.sum())
                for base_name in ["random", "distance", "environment"]:
                    for metric in ["ap", "ap_over_prevalence", "recall5", "recall1"]:
                        d = per[model][metric].to_numpy()[sel] - per[base_name][metric].to_numpy()[sel]
                        est, lo, hi = country_bootstrap(d, draws) if len(d) else (np.nan,) * 3
                        boot.append({"outcome": outcome, "variant": variant, "country_set": set_name, "model": model,
                                     "baseline": base_name, "metric": metric, "n_countries": int(sel.sum()),
                                     "difference": est, "ci_low": lo, "ci_high": hi})
        for model in MODELS:
            for base_name in ["random", "distance", "environment", *pooled_extra]:
                for k, metric in enumerate(["ap", "recall5"]):
                    d = pooled_reps[model][:, k] - pooled_reps[base_name][:, k]
                    boot.append({"outcome": outcome, "variant": variant, "country_set": "pooled", "model": model,
                                 "baseline": base_name, "metric": f"pooled_{metric}", "n_countries": len(present),
                                 "difference": pooled_point[model][metric] - pooled_point[base_name][metric],
                                 "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
        m0 = per[MODELS[0]]
        share = m0.positives / m0.positives.sum()
        conc.append({"outcome": outcome, "variant": variant, "positive_cells": int(y[elig].sum()),
                     "countries_with_positive": int((m0.positives > 0).sum()), "effective_countries": 1 / (share ** 2).sum(),
                     "n_sites": int(pd.Series(ev.WDPAID.to_numpy()[hh.owner]).nunique()),
                     **({"top10_sites": ";".join(map(str, notes[(outcome, "top10")].index)),
                         "top10_share": notes[(outcome, "top10")].sum() / max(int(y[elig].sum()), 1)} if variant == "primary" else {})})
        log(start, f"{outcome} {variant}: metrics and bootstrap done")

    pd.concat(rows).to_csv(out_dir / "country_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boot).to_csv(out_dir / "bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conc).to_csv(out_dir / "concentration.csv", index=False, encoding="utf-8-sig")
    manifest |= {"transient_countries": transient, "reconstructed_countries": RECONSTRUCTED, "n_boot": N_BOOT, "seed": SEED,
                 "outputs": {p.name: sha256(p) for p in sorted(out_dir.glob("*.*")) if p.name != "manifest.json"},
                 "seconds": round(time.time() - start)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(start, "done")


if __name__ == "__main__":
    main()
