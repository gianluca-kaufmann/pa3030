"""
Compute GEE-side zonal-statistic covariates for WDPA polygons fetched by
fetch_wdpa_global.py. Reads every regions/global/wdpa_raw/attrs_<iso3>.parquet
checkpoint, re-filters the live WCMC/WDPA/current/polygons collection by SITE_ID
to recover the real geometry (never downloaded locally — reduceRegions runs
server-side), and computes a decision-relevant covariate subset, now close to
parity with Q1's pixel feature set plus a few additions Q1 never had
(nightlights, climate normals).

SESSION 55 REDESIGN — batch export, not interactive getInfo() calls. Sessions
53-54 called reduceRegions().getInfo() interactively, once per (bucket, ~150-
polygon chunk, per country) — hundreds of small synchronous calls, each paying
GEE's full per-call setup cost (rebuilding a 30y climate composite, rasterizing
the entire global WDPA collection for PA-density, an NDVI year-composite) from
scratch, because GEE has no cross-call memoization for that. Real cost on the
24-country pilot: ~17h estimated, entirely dominated by 5 large countries, not
by total row count (see git history / STATUS.md session 54 "Cost model" for the
measurements that led here). The fix: reduceRegions() is bucketed by 5-year
STATUS_YR floor across ALL countries at once (not per-country), and each
bucket's result is submitted as ONE Earth Engine batch export
(ee.batch.Export.table.toAsset), not many interactive calls. A batch task pays
its setup cost once regardless of feature count and — unlike interactive
getInfo() concurrency, which session 54 measured actively backfiring from
server-side contention — multiple batch tasks are what GEE's batch backend is
built to run concurrently. Real A/B: 212 polygons cost 269s interactively
(cold) vs. 98s as a batch export; 7 polygons cost 58.6s interactively vs. 56s
batch (batch's fixed cost floor is about the same, but its marginal per-
polygon cost is roughly 5x cheaper: ~0.2s/polygon vs. ~1.0s/polygon). This
turns ~17h estimated into roughly the size of the largest single bucket
(~10,579 polygons, session 55's combined-24-country distribution) if all
buckets' tasks run concurrently, which they do here (all buckets submitted up
front, then polled together).

Per-country output files (regions/global/wdpa_raw/zonal_<iso3>.parquet) are
preserved for compatibility with build_wdpa_global_dataset.py — the batch/
bucket-level computation is an internal implementation detail; results are
split back out by country after being read back from the completed EE assets.

Resumability: each bucket's two exports (main covariates, PA-density buffers)
are checked against existing EE assets before submitting — a bucket already
exported is skipped, not resubmitted, so an interrupted run just needs to be
re-launched. A batch task that fails (e.g. a pathological geometry, mirroring
the France "memory limit exceeded" case from session 53) is retried as two
half-size sub-exports, recursively, same halving pattern the old interactive
code used for exceptions.

Covariates (directional aggregation — min/max, not just mean — for distance-type
features, per STATUS.md's "Hard-won lessons: aggregating pixels/area to a
polygon can destroy signal"):
  elev_mean, slope_mean            SRTM, mean across the polygon
  tree_frac_mean/_min/_max         Hansen treecover2000, masked to "still forested
                                    as of year t" for t>=2000; for t<2000 the mask
                                    is skipped (2000 baseline used as-is) and
                                    covariate_year_proxy=True is set
  human_mod_mean                   CSP/HM/GlobalHumanModification (gHM band)
  species_rich_mean                RESOLVE/ECOREGIONS/2017 NNH score (naturalness
                                    index — same proxy/naming Q1's covariate_stack
                                    already uses, see CLAUDE.md band 27)
  access_min                       travel-time-to-cities (Malaria Atlas Project,
                                    successor to the deprecated Oxford asset) —
                                    min = nearest access point in the polygon,
                                    directional since this is a distance-type feature
  pa_density_10km, pa_density_50km mean coverage by OTHER WDPA polygons with
                                    STATUS_YR < t, in a 10km/50km buffer around
                                    this polygon's own geometry — mirrors Q1's
                                    pixel-level pa_density_* at the historically
                                    correct per-polygon cutoff
  perimeter_km, compactness        from geometry().perimeter()/.area(), same
                                    formula as build_polygon_dataset.py
  (area_km2 is NOT recomputed here — GIS_AREA from fetch_wdpa_global.py is
  already authoritative WDPA-reported area)

  eco_id                           RESOLVE/ECOREGIONS/2017 ECO_ID (~850 classes,
                                    same asset already loaded for species_rich) —
                                    categorical biome/ecoregion identity, taken via
                                    mode (not mean/min/max, which are meaningless on
                                    a categorical code)
  clim_tmax_c, clim_tmin_c,        IDAHO_EPSCOR/TERRACLIMATE, 1991-2020 climate
  clim_precip_mm                     normal (a fixed 30y baseline, not "as of year
                                    t" — this is a land-opportunity-cost covariate,
                                    not a causal timing variable, so a normal is the
                                    right precision level, same reasoning as
                                    PA_DENSITY_BUCKET_YEARS's fuzz tradeoff below).
                                    tmmx/tmmn are degC*10 in the raw asset (divided
                                    by 10 here); pr is mean monthly mm (multiplied
                                    by 12 for an annual-equivalent total — an
                                    approximation, not a true annual sum, documented
                                    not silent)
  nightlights_mean                 SRUNet harmonized DMSP/VIIRS-like nighttime
                                    lights (projects/sat-io/open-datasets/
                                    srunet-npp-viirs-ntl, GEE community catalog, no
                                    login — verified live session 54), annual
                                    1992-2023. Economic-activity proxy for pillar
                                    1's money-driven-siting claim. Only starts
                                    1992 — buckets before that reuse the 1992 image
                                    (covariate_year_proxy=True already covers any
                                    bucket <2000, a superset of <1992). Masked/
                                    below-detection-threshold pixels return null,
                                    not 0 — verified live against Suva, Fiji (real
                                    value) vs. a rural Viti Levu point (null) —
                                    treat null as "no detectable light" (fill 0),
                                    not as missing data, at the build-dataset stage.
  forest_loss_frac_5yr_pre         Hansen lossyear: fraction of the polygon that
                                    lost tree cover in the 5 years strictly before
                                    bucket floor t — "was this land already being
                                    cleared right before designation," feeds
                                    pillar 2's paper-park baseline. Only meaningful
                                    once the full lookback window is >=2000 (Hansen's
                                    own start); the bucket=2000 case (window
                                    1995-2000) reads as a false 0, not flagged
                                    separately — a documented single-bucket edge
                                    case, not silently accepted for the whole
                                    pre-2000 range (covariate_year_proxy still
                                    covers bucket<2000 for tree_frac/nightlights/NDVI).
  forest_loss_frac_5yr_post,       Session 57 addition, pillar 2 (paper-park /
  forest_loss_frac_5yr_post_ring   avoided-deforestation outcome, STATUS.md
                                    "Active: Question 2"). `_post` mirrors `_pre`
                                    exactly but looks forward: fraction of the
                                    POLYGON that lost tree cover in [bucket t,
                                    t+5) — "did protection actually hold."
                                    `_post_ring` is the counterfactual: the same
                                    window's loss fraction over a 0-10km ring
                                    immediately outside the polygon (buffer(10000)
                                    minus the polygon itself, geometry simplified
                                    to SCALE first, same fix session 55 needed for
                                    the pa_density buffers), masked to exclude
                                    ring pixels already inside another WDPA
                                    polygon as of bucket t (reuses _pa_density_image,
                                    so this costs no extra compute beyond one more
                                    reduceRegions call) — an "unprotected
                                    neighboring land" counterfactual, not a full
                                    matched/synthetic control (that refinement is
                                    a documented future step, not built here).
                                    avoided_deforestation = _post_ring - _post is
                                    the outcome metric; NOT computed here, left to
                                    whoever builds the pillar 2 analysis script.
                                    Hansen v1_13's lossyear covers through 2025
                                    (verified live session 57: max pixel value 25
                                    sampled over a real Amazon region) — a bucket's
                                    post-window is only FULLY covered if
                                    t+5<=2026, i.e. t<=2021; every row gets
                                    `forest_loss_post_window_complete` (bool) so a
                                    clean-window filter (recommended: STATUS_YR
                                    <=2019, bucket floor <=2015, comfortable
                                    margin) doesn't have to be re-derived by hand.
  ndvi_mean, ndvi_cv                MODIS/061/MOD13A1 NDVI, calendar-year mean and
                                    coefficient of variation for bucket floor t —
                                    same asset/derivation Q1's covariate_stack used
                                    (gee_ndvi_mean/_cv). Only starts 2000 (MODIS
                                    launch); buckets before that reuse the 2000
                                    image, covered by covariate_year_proxy.

Distance to coast is deferred (no ready no-login global GEE asset identified
session 53) — the marine_frac computed in fetch_wdpa_global.py stands in.
Landcover class/built_frac (ESA WorldCover) and GBIF point-occurrence species
richness were considered session 54 and deliberately left out: human_mod +
tree_frac + eco_id already cover the human-footprint/land-cover-identity signal
without adding a redundant categorical band, and NNH (already included) is a
better siting-relevant biodiversity proxy than raw richness, which is confounded
by sampling effort in more-accessible areas.

Sentinel-value reminder for whoever builds the future GMM regime-clustering step
(STATUS.md "Hard-won lessons"): check pa_density_10km/50km and access_min for a
suspiciously-repeated constant before fitting any unsupervised model on them —
this is the exact failure mode that hit competing_risks_hazard.py's GMM once
already (a "not found" sentinel mistaken for a real tight cluster).

Output: regions/global/wdpa_raw/zonal_<iso3>.parquet, keyed by SITE_ID.

Run (after fetch_wdpa_global.py):
    python3 scripts/regions/world/compute_wdpa_zonal_stats.py
"""

import argparse
import math
import os, sys, time
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from scripts.regions.world import paths as world_paths  # noqa: E402

RAW_DIR = str(world_paths.RAW_DIR)

ACCESSIBILITY_ASSET = "projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
HUMAN_MOD_ASSET = "CSP/HM/GlobalHumanModification"
ECOREGIONS_ASSET = "RESOLVE/ECOREGIONS/2017"
SRTM_ASSET = "USGS/SRTMGL1_003"
WDPA_ASSET = "WCMC/WDPA/current/polygons"
TERRACLIMATE_ASSET = "IDAHO_EPSCOR/TERRACLIMATE"
NIGHTLIGHTS_ASSET = "projects/sat-io/open-datasets/srunet-npp-viirs-ntl"  # GEE community catalog, no login (verified session 54)
NDVI_ASSET = "MODIS/061/MOD13A1"

CLIM_START, CLIM_END = "1991-01-01", "2021-01-01"  # 30y normal, see docstring
NIGHTLIGHTS_START_YEAR = 1992
NDVI_START_YEAR = 2000
HANSEN_LOSSYEAR_MAX_ABS = 2025  # verified live session 57: max lossyear pixel value
# sampled at 25 over a real Amazon region -> 2000+25=2025 is Hansen v1_13's real ceiling

SCALE = 1000  # meters, matches the rest of the pipeline's 1km convention
PA_DENSITY_BUCKET_YEARS = 5   # see _pa_density_bucket_cutoff docstring
RING_BUFFER_M = 10000  # pillar 2's "neighboring unprotected land" counterfactual ring

EE_PROJECT = None  # set from the required --project argument in main()
# v2 (session 57): new schema (forest_loss_frac_5yr_post[_ring] added to the main/ring
# exports) -- a fresh asset folder, not a bump-in-place, so this session's Phase 4
# full-scale run recomputes EVERY bucket (old 24-country pilot included) with the new
# bands rather than silently skipping already-cached v1 buckets that lack them.
ASSET_FOLDER = None
# Session 58 addition: --project/--only-buckets/--out-suffix let a second script
# instance run against a second GEE Cloud project (its own separate compute quota,
# see STATUS.md "Recommended path" #2) on a disjoint subset of buckets in parallel
# with the primary run. ASSET_FOLDER/EE_PROJECT are reassigned in main() before any
# of the functions below run -- they read the module global at call time, not import
# time, so this is safe. --out-suffix keeps this instance's per-country parquet
# outputs from colliding with the primary run's; merge them by concatenation
# afterward (disjoint SITE_IDs, since buckets are disjoint).
PAGE_SIZE = 5000  # readback pagination -- getInfo() truncates large FeatureCollections

_pa_density_image_cache = {}


def _static_image():
    """Year-independent composite: elevation, slope, human modification, species
    richness proxy, accessibility, ecoregion identity, 30y climate normal. Built
    once, reused across all buckets."""
    import ee
    srtm = ee.Image(SRTM_ASSET)
    slope = ee.Terrain.slope(srtm)
    ghm = ee.ImageCollection(HUMAN_MOD_ASSET).first().select("gHM")
    ecoregions = ee.FeatureCollection(ECOREGIONS_ASSET)
    nnh_img = ecoregions.reduceToImage(properties=["NNH"], reducer=ee.Reducer.first())
    eco_id_img = ecoregions.reduceToImage(properties=["ECO_ID"], reducer=ee.Reducer.first())
    access = ee.Image(ACCESSIBILITY_ASSET).select("accessibility")

    tc = (ee.ImageCollection(TERRACLIMATE_ASSET).filterDate(CLIM_START, CLIM_END)
          .select(["tmmx", "tmmn", "pr"]).mean())
    clim_tmax = tc.select("tmmx").divide(10).rename("clim_tmax_c")
    clim_tmin = tc.select("tmmn").divide(10).rename("clim_tmin_c")
    clim_precip = tc.select("pr").multiply(12).rename("clim_precip_mm")

    return (srtm.rename("elevation")
            .addBands(slope.rename("slope"))
            .addBands(ghm.rename("human_mod"))
            .addBands(nnh_img.rename("species_rich"))
            .addBands(eco_id_img.rename("eco_id"))
            .addBands(access.rename("access"))
            .addBands(clim_tmax).addBands(clim_tmin).addBands(clim_precip))


def _tree_frac_image(t):
    """Hansen tree fraction 'as of year t': treecover2000, masked to 0 where loss
    happened before t. Years < 2000 skip the loss mask (2000 baseline as-is —
    caller sets covariate_year_proxy=True for those)."""
    import ee
    hansen = ee.Image(HANSEN_ASSET)
    tree2000 = hansen.select("treecover2000").divide(100.0)
    if t < 2000:
        return tree2000.rename("tree_frac")
    lossyear = hansen.select("lossyear")  # 0 = no loss; 1..N = years since 2000
    loss_before_t = lossyear.gt(0).And(lossyear.add(2000).lt(t))
    return tree2000.where(loss_before_t, 0).rename("tree_frac")


def _dynamic_bucket_image(t):
    """Per-bucket time-varying covariates beyond tree_frac: nightlights 'as of
    bucket t' (floored to NIGHTLIGHTS_START_YEAR), recent forest-loss fraction
    (5y lookback strictly before t), and NDVI mean/CV 'as of bucket t' (floored
    to NDVI_START_YEAR)."""
    import ee
    tree = _tree_frac_image(t)

    ntl_year = max(t, NIGHTLIGHTS_START_YEAR)
    night = (ee.ImageCollection(NIGHTLIGHTS_ASSET)
             .filter(ee.Filter.stringContains("system:index", str(ntl_year)))
             .first().select("b1").rename("nightlights"))

    hansen = ee.Image(HANSEN_ASSET)
    lossyear = hansen.select("lossyear")
    loss_abs = lossyear.add(2000)
    recent_loss = (loss_abs.gte(t - 5).And(loss_abs.lt(t)).And(lossyear.gt(0))
                   .rename("forest_loss_frac_5yr_pre"))
    post_loss = (loss_abs.gte(t).And(loss_abs.lt(t + 5)).And(lossyear.gt(0))
                 .rename("forest_loss_frac_5yr_post"))

    ndvi_year = max(t, NDVI_START_YEAR)
    ndvi_coll = (ee.ImageCollection(NDVI_ASSET)
                 .filterDate(f"{ndvi_year}-01-01", f"{ndvi_year}-12-31").select("NDVI"))
    ndvi_mean = ndvi_coll.mean().multiply(0.0001).rename("ndvi_mean")
    ndvi_std = ndvi_coll.reduce(ee.Reducer.stdDev()).multiply(0.0001)
    ndvi_cv = ndvi_std.divide(ndvi_mean.max(1e-6)).rename("ndvi_cv")

    return (tree.addBands(night).addBands(recent_loss).addBands(post_loss)
            .addBands(ndvi_mean).addBands(ndvi_cv))


def _post_loss_ring_image(t):
    """Pillar 2 counterfactual band: post-designation (window [t, t+5)) forest-loss,
    masked to pixels NOT already inside another WDPA polygon as of bucket t (reuses
    _pa_density_image's paint() raster -- see module docstring's
    forest_loss_frac_5yr_post_ring entry). Reduced over a ring geometry (buffer minus
    the polygon itself) by the caller, not here -- this only builds the per-pixel
    image, mirroring _pa_density_image's image/geometry split."""
    import ee
    hansen = ee.Image(HANSEN_ASSET)
    lossyear = hansen.select("lossyear")
    loss_abs = lossyear.add(2000)
    post_loss = loss_abs.gte(t).And(loss_abs.lt(t + 5)).And(lossyear.gt(0))
    pa_img = _pa_density_image(t)
    return post_loss.updateMask(pa_img.eq(0)).rename("forest_loss_frac_5yr_post_ring")


def _pa_density_bucket_cutoff(t):
    return (t // PA_DENSITY_BUCKET_YEARS) * PA_DENSITY_BUCKET_YEARS


def _pa_density_image(t):
    """Binary presence raster of WDPA polygons with STATUS_YR < bucket_floor(t)."""
    import ee
    bucket = _pa_density_bucket_cutoff(t)
    if bucket in _pa_density_image_cache:
        return _pa_density_image_cache[bucket]
    wdpa = ee.FeatureCollection(WDPA_ASSET).filter(ee.Filter.lt("STATUS_YR", bucket))
    # paint() burns 1 into pixels covered by any feature onto a base image that is
    # already 0 (unmasked) everywhere else --
    # far cheaper than reduceToImage (which was timing out / hitting the memory ceiling
    # on every bucket, session 55) AND fixes a latent correctness bug: reduceToImage
    # leaves non-PA pixels MASKED rather than 0, and reduceRegions' mean reducer drops
    # masked pixels from the average -- so the old code's pa_density would compute close
    # to 1.0 wherever a buffer touched ANY protected pixel, regardless of true coverage.
    img = ee.Image(0).byte().paint(wdpa, 1).rename("pa_present")
    _pa_density_image_cache[bucket] = img
    return img


def _main_fc(site_ids):
    import ee
    fc = ee.FeatureCollection(WDPA_ASSET).filter(ee.Filter.inList("SITE_ID", [int(s) for s in site_ids]))

    def per_feature(f):
        geom = f.geometry()
        perim = geom.perimeter(1)
        area = geom.area(1)
        compactness = ee.Number(4 * 3.14159265).multiply(area).divide(perim.pow(2).max(1e-9))
        return f.set({"_perimeter_km": perim.divide(1000), "_compactness": compactness})

    return fc.map(per_feature)


def _asset_exists(asset_id):
    import ee
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def _submit_export(collection, asset_id, description):
    import ee
    task = ee.batch.Export.table.toAsset(collection=collection, description=description, assetId=asset_id)
    task.start()
    return task


def _submit_bucket(bucket, site_ids, static_img, tag="", depth=0, max_depth=6):
    """Submits (or skips, if already exported) the main + buffer + ring exports for
    one bucket's polygon set. Returns a list of job dicts (usually one, more if a
    previous attempt failed and this bucket had to be halved -- tag/depth make
    each half's asset IDs unique so retries never collide)."""
    import ee
    suffix = f"_{tag}" if tag else ""
    main_asset = f"{ASSET_FOLDER}/bucket_{bucket}_main{suffix}"
    buf_asset = f"{ASSET_FOLDER}/bucket_{bucket}_buf{suffix}"
    ring_asset = f"{ASSET_FOLDER}/bucket_{bucket}_ring{suffix}"

    main_done = _asset_exists(main_asset)
    buf_done = _asset_exists(buf_asset)
    ring_done = _asset_exists(ring_asset)
    if main_done and buf_done and ring_done:
        return [{"bucket": bucket, "site_ids": site_ids, "main_asset": main_asset, "buf_asset": buf_asset,
                  "ring_asset": ring_asset, "main_task": None, "buf_task": None, "ring_task": None,
                  "state": "PRE-CACHED"}]

    # Each side (main / buf / ring) is checked and (re)submitted independently -- a
    # bucket whose main export already succeeded but whose buf/ring export
    # failed/is missing must not recompute or resubmit main (asset-already-exists
    # collision, and wasted compute).
    main_task = None
    if not main_done:
        dynamic_img = _dynamic_bucket_image(bucket)
        fc = _main_fc(site_ids)
        combo = static_img.addBands(dynamic_img)
        # tileScale=8: same fix already applied to buf/ring below (session 55) -- at
        # session-58 scale (208 countries vs the 26-country pilot this was never hit
        # against before) main started throwing the identical "User memory limit
        # exceeded" error buf/ring were fixed for; it just never needed the mitigation
        # at pilot scale.
        reduced = combo.reduceRegions(collection=fc, reducer=ee.Reducer.mean().combine(
            ee.Reducer.min(), sharedInputs=True).combine(ee.Reducer.max(), sharedInputs=True
            ).combine(ee.Reducer.mode(), sharedInputs=True), scale=SCALE, tileScale=8)
        main_task = _submit_export(reduced, main_asset, f"wdpa_main_{bucket}{suffix}")

    # Raw FC (no _main_fc per-feature perimeter/compactness map -- unused here, just
    # wasted compute) with geometry simplified to SCALE before buffering: some WDPA
    # polygons (large countries) have thousands of vertices, and buffering+reduceRegions
    # at full vertex detail across 10k+ features was the actual "Computation timed out"
    # cause (confirmed: paint()+tileScale alone did not fix it, persisted even after
    # halving the batch -- pointing at vector-op cost, not raster sampling cost). Built
    # lazily (no cost until an export below actually triggers evaluation) so it's
    # available to both the buf and ring exports without duplicating the fix.
    raw_fc = (ee.FeatureCollection(WDPA_ASSET)
              .filter(ee.Filter.inList("SITE_ID", [int(s) for s in site_ids]))
              .map(lambda f: f.setGeometry(f.geometry().simplify(SCALE))))

    buf_task = None
    if not buf_done:
        pa_img = _pa_density_image(bucket)
        buf10_fc = raw_fc.map(lambda f: ee.Feature(f.geometry().buffer(10000)).copyProperties(f, ["SITE_ID"]).set("_buf", 10))
        buf50_fc = raw_fc.map(lambda f: ee.Feature(f.geometry().buffer(50000)).copyProperties(f, ["SITE_ID"]).set("_buf", 50))
        # tileScale=8: pa_img rasterizes the ENTIRE global WDPA collection (reduceToImage,
        # no spatial pre-filter) -- this reduceRegions call was hitting GEE's "User memory
        # limit exceeded" on every single bucket regardless of batch size (confirmed live,
        # session 55: full-size AND half-size batches both failed identically), so the
        # bottleneck is per-polygon evaluation cost, not batch size. tileScale trades
        # compute time for memory by tiling the computation -- the documented GEE fix for
        # exactly this error.
        pa_reduced = pa_img.reduceRegions(collection=buf10_fc.merge(buf50_fc), reducer=ee.Reducer.mean(),
                                           scale=SCALE, tileScale=8)
        buf_task = _submit_export(pa_reduced, buf_asset, f"wdpa_buf_{bucket}{suffix}")

    ring_task = None
    if not ring_done:
        # Session 57, pillar 2: 0-10km ring (buffer minus the polygon itself) as the
        # "neighboring unprotected land" counterfactual for forest_loss_frac_5yr_post.
        # Same simplify-before-buffer/tileScale mitigations as the buf export above --
        # this is the same class of vector op at the same scale, so it's exposed to the
        # identical session-55 cost failure mode.
        ring_img = _post_loss_ring_image(bucket)
        ring_fc = raw_fc.map(lambda f: ee.Feature(
            f.geometry().buffer(RING_BUFFER_M).difference(f.geometry(), ee.ErrorMargin(30))
        ).copyProperties(f, ["SITE_ID"]))
        ring_reduced = ring_img.reduceRegions(collection=ring_fc, reducer=ee.Reducer.mean(),
                                               scale=SCALE, tileScale=8)
        ring_task = _submit_export(ring_reduced, ring_asset, f"wdpa_ring_{bucket}{suffix}")

    return [{"bucket": bucket, "site_ids": site_ids, "main_asset": main_asset, "buf_asset": buf_asset,
             "ring_asset": ring_asset, "main_task": main_task, "buf_task": buf_task, "ring_task": ring_task,
             "state": "SUBMITTED", "depth": depth, "tag": tag}]


def _poll_and_retry(jobs, static_img, poll_interval=15, max_depth=6):
    """Polls all submitted jobs to completion. A FAILED task is halved and
    resubmitted as two smaller jobs (mirrors the old interactive halving
    retry) -- appended back into the polling loop. Returns the final list of
    jobs, all either PRE-CACHED or COMPLETED (a job that keeps failing down to
    a single polygon is dropped with a warning, matching the old behavior)."""
    pending = [j for j in jobs if j["state"] == "SUBMITTED"]
    finished = [j for j in jobs if j["state"] == "PRE-CACHED"]

    while pending:
        time.sleep(poll_interval)
        still_pending = []
        for j in pending:
            # main_task/buf_task/ring_task is None when that side's asset already
            # existed at submission time (independent per-side resumability) --
            # treat as COMPLETED.
            m_state = j["main_task"].status()["state"] if j["main_task"] else "COMPLETED"
            b_state = j["buf_task"].status()["state"] if j["buf_task"] else "COMPLETED"
            r_state = j["ring_task"].status()["state"] if j.get("ring_task") else "COMPLETED"
            if m_state in ("READY", "RUNNING") or b_state in ("READY", "RUNNING") or r_state in ("READY", "RUNNING"):
                still_pending.append(j)
                continue
            if m_state == "COMPLETED" and b_state == "COMPLETED" and r_state == "COMPLETED":
                print(f"  bucket {j['bucket']}{('_' + j['tag']) if j.get('tag') else ''}: exports completed ({len(j['site_ids'])} polygons)")
                finished.append(j)
            else:
                print(f"  bucket {j['bucket']}{('_' + j['tag']) if j.get('tag') else ''}: FAILED (main={m_state}, buf={b_state}, ring={r_state})")
                site_ids = j["site_ids"]
                depth = j.get("depth", 0)
                if len(site_ids) <= 1 or depth >= max_depth:
                    print(f"        WARNING: giving up on SITE_IDs {site_ids}")
                    continue
                mid = len(site_ids) // 2
                for half_tag, half_ids in (("a", site_ids[:mid]), ("b", site_ids[mid:])):
                    new_tag = f"{j.get('tag', '')}{half_tag}{depth}"
                    still_pending.extend(_submit_bucket(j["bucket"], half_ids, static_img, tag=new_tag, depth=depth + 1))
        pending = still_pending

    return finished


def _read_fc_paginated(asset_id):
    import ee
    fc = ee.FeatureCollection(asset_id)
    size = fc.size().getInfo()
    features = []
    for start in range(0, size, PAGE_SIZE):
        features.extend(fc.toList(PAGE_SIZE, start).getInfo())
    return features


def _extract_main_row(p):
    return {
        "elev_mean": p.get("elevation_mean"), "slope_mean": p.get("slope_mean"),
        "tree_frac_mean": p.get("tree_frac_mean"), "tree_frac_min": p.get("tree_frac_min"),
        "tree_frac_max": p.get("tree_frac_max"),
        "human_mod_mean": p.get("human_mod_mean"), "species_rich_mean": p.get("species_rich_mean"),
        "access_min": p.get("access_min"),
        "eco_id": p.get("eco_id_mode"),
        "clim_tmax_c": p.get("clim_tmax_c_mean"), "clim_tmin_c": p.get("clim_tmin_c_mean"),
        "clim_precip_mm": p.get("clim_precip_mm_mean"),
        "nightlights_mean": p.get("nightlights_mean"),
        "forest_loss_frac_5yr_pre": p.get("forest_loss_frac_5yr_pre_mean"),
        "forest_loss_frac_5yr_post": p.get("forest_loss_frac_5yr_post_mean"),
        "ndvi_mean": p.get("ndvi_mean_mean"), "ndvi_cv": p.get("ndvi_cv_mean"),
        "perimeter_km": p.get("_perimeter_km"), "compactness": p.get("_compactness"),
    }


def main():
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Earth Engine Cloud project id that runs the exports")
    parser.add_argument("--only-buckets", default=None,
                         help="comma-separated bucket-floor years to process, e.g. 2010,2015,2020 (default: all)")
    parser.add_argument("--countries", default=None,
                         help="comma-separated ISO3 codes to restrict to, e.g. the 26-country pilot "
                              "(default: all attrs_*.parquet files in RAW_DIR -- as of session 58 that's "
                              "234 countries, since the full-world attrs fetch and the pilot's zonal-stats "
                              "run share RAW_DIR; pass this explicitly to match the pilot's scope)")
    parser.add_argument("--out-suffix", default="",
                         help="suffix for output parquet filenames, e.g. _partB (default: none, primary run)")
    args = parser.parse_args()

    global EE_PROJECT, ASSET_FOLDER
    if args.project:
        EE_PROJECT = args.project
        ASSET_FOLDER = f"projects/{EE_PROJECT}/assets/wdpa_zonal_batch_v2"
    only_buckets = None
    if args.only_buckets:
        only_buckets = {int(b) for b in args.only_buckets.split(",")}

    ee.Initialize(project=EE_PROJECT)

    try:
        ee.data.createAsset({"type": "FOLDER"}, ASSET_FOLDER)
    except Exception:
        pass  # already exists

    countries_filter = None
    if args.countries:
        countries_filter = {c.strip().upper() for c in args.countries.split(",")}

    attrs_files = [f for f in os.listdir(RAW_DIR) if f.startswith("attrs_") and f.endswith(".parquet")]
    if countries_filter is not None:
        attrs_files = [f for f in attrs_files if f[len("attrs_"):-len(".parquet")].upper() in countries_filter]
    dfs = []
    for fname in attrs_files:
        iso3 = fname[len("attrs_"):-len(".parquet")]
        d = pd.read_parquet(os.path.join(RAW_DIR, fname))
        d["_country"] = iso3
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["STATUS_YR"] = pd.to_numeric(all_df["STATUS_YR"], errors="coerce")
    all_df["_bucket"] = (all_df["STATUS_YR"] // PA_DENSITY_BUCKET_YEARS * PA_DENSITY_BUCKET_YEARS).astype(int)
    if only_buckets is not None:
        all_df = all_df[all_df["_bucket"].isin(only_buckets)]
    print(f"=== {len(all_df)} designation events, {len(attrs_files)} countries, "
          f"{all_df['_bucket'].nunique()} buckets, project={EE_PROJECT} ===\n")

    static_img = _static_image()

    # Pre-chunk each bucket's site_ids to ~A-pilot scale (session 58: A's own
    # 26-country/~23k-event batch was the largest size shown to mostly work) instead
    # of submitting the full country-set and discovering the workable size only via
    # repeated failed-then-halved attempts -- at C's 208-country scale that discovery
    # process was burning real compute on doomed oversized attempts before ever
    # reaching a viable batch (session 58, ~2026-08-25 11:00 postmortem). Each chunk
    # gets its own tag so asset names never collide across chunks.
    CHUNK_EVENT_TARGET = 18000
    jobs = []
    for bucket, grp in all_df.groupby("_bucket"):
        bucket = int(bucket)
        site_ids_all = grp["SITE_ID"].tolist()
        if len(site_ids_all) <= CHUNK_EVENT_TARGET:
            chunks = [site_ids_all]
        else:
            n_chunks = math.ceil(len(site_ids_all) / CHUNK_EVENT_TARGET)
            chunk_size = math.ceil(len(site_ids_all) / n_chunks)
            chunks = [site_ids_all[i:i + chunk_size] for i in range(0, len(site_ids_all), chunk_size)]
        for i, site_ids in enumerate(chunks):
            tag = f"c{i}" if len(chunks) > 1 else ""
            j = _submit_bucket(bucket, site_ids, static_img, tag=tag)
            state = j[0]["state"]
            print(f"  bucket {bucket} chunk {i + 1}/{len(chunks)}: {len(site_ids)} polygons -- {state}")
            jobs.extend(j)

    print("\n=== waiting for exports ===")
    finished = _poll_and_retry(jobs, static_img)

    print("\n=== reading back results ===")
    all_results = {}
    for j in finished:
        main_features = _read_fc_paginated(j["main_asset"])
        for f in main_features:
            p = f["properties"]
            sid = p["SITE_ID"]
            all_results.setdefault(sid, {"SITE_ID": sid}).update(_extract_main_row(p))
        pa10, pa50 = {}, {}
        for f in _read_fc_paginated(j["buf_asset"]):
            p = f["properties"]
            (pa10 if p["_buf"] == 10 else pa50)[p["SITE_ID"]] = p.get("mean")
        for sid in set(list(pa10.keys()) + list(pa50.keys())):
            all_results.setdefault(sid, {"SITE_ID": sid})
            all_results[sid]["pa_density_10km"] = pa10.get(sid)
            all_results[sid]["pa_density_50km"] = pa50.get(sid)
        for f in _read_fc_paginated(j["ring_asset"]):
            p = f["properties"]
            sid = p["SITE_ID"]
            all_results.setdefault(sid, {"SITE_ID": sid})
            all_results[sid]["forest_loss_frac_5yr_post_ring"] = p.get("mean")
        print(f"  bucket {j['bucket']}: read back {len(main_features)} rows")

    year_by_id = dict(zip(all_df["SITE_ID"], all_df["STATUS_YR"]))
    country_by_id = dict(zip(all_df["SITE_ID"], all_df["_country"]))
    bucket_by_id = dict(zip(all_df["SITE_ID"], all_df["_bucket"]))

    rows_by_country = {}
    for sid, vals in all_results.items():
        row = dict(vals)
        row["covariate_year_proxy"] = bool(year_by_id.get(sid, 9999) < 2000)
        bucket = bucket_by_id.get(sid, 9999)
        # See module docstring's forest_loss_frac_5yr_post_ring entry: Hansen v1_13's
        # lossyear covers through 2025, so a bucket's post-window [t, t+5) is only
        # fully covered if t+5<=2026.
        row["forest_loss_post_window_complete"] = bool(bucket + 5 <= HANSEN_LOSSYEAR_MAX_ABS + 1)
        country = country_by_id.get(sid)
        rows_by_country.setdefault(country, []).append(row)

    for country, rows in rows_by_country.items():
        out_path = os.path.join(RAW_DIR, f"zonal_{country}{args.out_suffix}.parquet")
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(rows)} rows)")

    print("\nDone. Next: scripts/regions/world/build_wdpa_global_dataset.py")


if __name__ == "__main__":
    main()
