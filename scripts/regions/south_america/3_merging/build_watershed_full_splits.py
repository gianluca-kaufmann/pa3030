#!/usr/bin/env python3
"""Aggregate full SA pixel splits to HydroSHEDS L7 catchment level.

Reads the three SA pixel parquets (train / earlystop / test), assigns each
pixel to a catchment via the catchment_id_sa.tif raster, and aggregates to
(country_id, year, catchment_id) level using directional aggregation:

  min  → distance features  (closer = more attractive)
  max  → biodiversity / value features  (highest pixel drives designation)
  mean → everything else
  std  → elevation_b1 → elev_std  (terrain ruggedness within catchment)

The output is a single parquet covering ALL years 2001-2024.
model1_LGBM_stage2_watershed.py loads this and does the cross-event split
internally (so train/earlystop/test are NOT kept separate here).

Memory approach: read each source parquet in batches; immediately aggregate
each batch to catchment level; accumulate partial catchment-level aggregates;
do a final re-aggregation at the end. The catchment-level data is tiny
(~100K-1M rows) vs the pixel data (220M+ rows).

Input:
  $SCRATCH/data/south_america/ml/main/train.parquet        (~30 GB)
  $SCRATCH/data/south_america/ml/main/earlystop.parquet
  $SCRATCH/data/south_america/ml/main/test.parquet
  $SCRATCH/data/south_america/ready/hydroshed/catchment_id_sa.tif

Output:
  $SCRATCH/data/south_america/ml/watershed/watershed_full_sa.parquet

Usage (from repo root):
  python scripts/regions/south_america/3_merging/build_watershed_full_splits.py

SLURM: see slurm/south_america/watershed_build.slurm
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRATCH = os.environ.get("SCRATCH", "")
if not _SCRATCH:
    sys.exit("SCRATCH env var not set. Run on Euler or set SCRATCH manually.")

SCRATCH = Path(_SCRATCH)

PIXEL_PARQUETS = [
    SCRATCH / "data/south_america/ml/main/train.parquet",
    SCRATCH / "data/south_america/ml/main/earlystop.parquet",
    SCRATCH / "data/south_america/ml/main/test.parquet",
]
CATCHMENT_TIF = SCRATCH / "data/south_america/ready/hydroshed/catchment_id_sa.tif"
OUT_DIR = SCRATCH / "data/south_america/ml/watershed"
OUT_PATH = OUT_DIR / "watershed_full_sa.parquet"

BATCH_SIZE = 1_000_000  # pixels per read batch

# Columns excluded from aggregation (identifiers, labels, leakage)
EXCLUDE = frozenset({
    "year", "x", "y", "row", "col", "country_id",
    "transition_01", "WDPA_prev",
    "WDPA_b1", "WDPA_b2", "WDPA",
    "country_iso3",
})

# Distance features: use min (closest pixel in catchment → best access signal)
MIN_FEATURES = frozenset({
    "dist_indigenous", "dist_oil_gas", "dist_powerplant",
    "dist_redd_km", "dist_road", "dist_wdpa",
})

# Biodiversity / value features: use max (highest-value pixel drives designation)
MAX_FEATURES = frozenset({
    "GSN_b1", "GSN_b1_smooth16", "GSN_b1_smooth64",
    "GSN_b2", "GSN_b2_smooth16", "GSN_b2_smooth64",
    "GSN_b3", "GSN_b3_smooth16", "GSN_b3_smooth64",
    "GSN_b4", "GSN_b4_smooth16", "GSN_b4_smooth64",
    "GSN_b5", "GSN_b5_smooth16", "GSN_b5_smooth64",
    "agb_tonne_ha",
    "NDVI", "NDVI_smooth16", "NDVI_smooth64", "NDVI_trend5",
    "patch_mean_gsn_b2", "patch_pa_adjacency_frac", "patch_designation_lag1",
})

# Elevation col for std (terrain ruggedness structural feature)
ELEV_COL = "elevation_b1"

GROUP_COLS = ["country_id", "year", "catchment_id"]


def _load_catchment_raster() -> np.ndarray:
    if not CATCHMENT_TIF.exists():
        sys.exit(
            f"Catchment raster not found: {CATCHMENT_TIF}\n"
            "Upload catchment_id_sa.tif from local machine to Euler:\n"
            f"  rsync data/south_america/ready/hydroshed/catchment_id_sa.tif "
            f"euler:{CATCHMENT_TIF}"
        )
    with rasterio.open(CATCHMENT_TIF) as src:
        arr = src.read(1).astype(np.int32)
    n_unique = int(np.unique(arr[arr > 0]).size)
    print(f"Catchment raster: {arr.shape}  unique catchments: {n_unique:,}")
    return arr


def _aggregate_batch(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_cols: list[str],
    max_cols: list[str],
    mean_cols: list[str],
    has_elev: bool,
) -> pd.DataFrame:
    """Aggregate one batch of pixels to catchment level.

    Returns a partial catchment-level DataFrame with columns:
      GROUP_COLS + mean cols (as _sum, _cnt) + min cols + max cols
      + elev_sum, elev_sq_sum, elev_cnt (for std computation)
      + n_pos_pixels, n_total_pixels, WDPA_prev_sum
    """
    parts: list[pd.DataFrame] = []

    # Labels
    label = df.groupby(GROUP_COLS).agg(
        n_pos_pixels=("transition_01", "sum"),
        n_total_pixels=("transition_01", "count"),
        WDPA_prev_sum=("WDPA_prev", "sum"),
    )
    parts.append(label)

    # Mean aggregation: store sum + count to allow combining across batches
    if mean_cols:
        s = df.groupby(GROUP_COLS)[mean_cols].sum().add_suffix("__sum")
        c = df.groupby(GROUP_COLS)[mean_cols].count().add_suffix("__cnt")
        parts.extend([s, c])

    # Min aggregation
    if min_cols:
        parts.append(df.groupby(GROUP_COLS)[min_cols].min().add_suffix("__min"))

    # Max aggregation
    if max_cols:
        parts.append(df.groupby(GROUP_COLS)[max_cols].max().add_suffix("__max"))

    # Elevation sum, sum-of-squares, count for std computation
    if has_elev and ELEV_COL in df.columns:
        elev = df.groupby(GROUP_COLS)[ELEV_COL].agg(
            elev__sum="sum",
            elev__sq_sum=lambda x: (x ** 2).sum(),
            elev__cnt="count",
        )
        parts.append(elev)

    return pd.concat(parts, axis=1).reset_index()


def _combine_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine partial catchment-level aggregates into final values."""
    if not partials:
        return pd.DataFrame()

    combined = pd.concat(partials, ignore_index=True)

    # Identify column groups
    sum_cols = [c for c in combined.columns if c.endswith("__sum") and not c.startswith("elev__")]
    cnt_cols = [c for c in combined.columns if c.endswith("__cnt") and not c.startswith("elev__")]
    min_cols = [c for c in combined.columns if c.endswith("__min")]
    max_cols = [c for c in combined.columns if c.endswith("__max")]
    has_elev = "elev__sum" in combined.columns

    agg_dict: dict[str, object] = {
        "n_pos_pixels":   "sum",
        "n_total_pixels": "sum",
        "WDPA_prev_sum":  "sum",
    }
    for c in sum_cols:
        agg_dict[c] = "sum"
    for c in cnt_cols:
        agg_dict[c] = "sum"
    for c in min_cols:
        agg_dict[c] = "min"
    for c in max_cols:
        agg_dict[c] = "max"
    if has_elev:
        agg_dict["elev__sum"]    = "sum"
        agg_dict["elev__sq_sum"] = "sum"
        agg_dict["elev__cnt"]    = "sum"

    result = combined.groupby(GROUP_COLS).agg(agg_dict).reset_index()

    # Reconstruct mean columns from sum/count
    for s_col in sum_cols:
        feat = s_col[:-5]   # strip __sum
        c_col = feat + "__cnt"
        mean_col = feat      # final name = original feature name
        cnt = result[c_col].replace(0, np.nan)
        result[mean_col] = result[s_col] / cnt
        result.drop(columns=[s_col, c_col], inplace=True)

    # Strip __min / __max suffixes
    for m in min_cols:
        result.rename(columns={m: m[:-5]}, inplace=True)   # strip __min
    for m in max_cols:
        result.rename(columns={m: m[:-5]}, inplace=True)   # strip __max

    # Compute elev_std from running sum / sq_sum / count
    if has_elev:
        n = result["elev__cnt"].replace(0, np.nan)
        mean_e = result["elev__sum"] / n
        var_e  = result["elev__sq_sum"] / n - mean_e ** 2
        result["elev_std"] = np.sqrt(var_e.clip(lower=0)).fillna(0)
        result.drop(columns=["elev__sum", "elev__sq_sum", "elev__cnt"], inplace=True)

    # Derived label columns
    result["transition_01"]   = (result["n_pos_pixels"] > 0).astype(np.uint8)
    result["n_pos_pixels"]    = result["n_pos_pixels"].astype(np.int32)
    result["n_total_pixels"]  = result["n_total_pixels"].astype(np.int32)
    result["WDPA_prev_frac"]  = result["WDPA_prev_sum"] / result["n_total_pixels"].clip(lower=1)
    result.drop(columns=["WDPA_prev_sum"], inplace=True)

    return result


def _process_parquet(
    path: Path,
    cid_arr: np.ndarray,
    feature_cols: list[str],
    min_cols: list[str],
    max_cols: list[str],
    mean_cols: list[str],
    has_elev: bool,
    load_cols: list[str],
) -> list[pd.DataFrame]:
    """Stream one parquet file in batches; return partial catchment-level aggregates."""
    partials: list[pd.DataFrame] = []
    n_total = n_no_catch = 0

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=load_cols):
        df = batch.to_pandas()
        n_total += len(df)
        df = df[df["WDPA_prev"] == 0].copy()
        if df.empty:
            continue

        rows = df["row"].to_numpy(dtype=np.int64)
        cols = df["col"].to_numpy(dtype=np.int64)
        h, w = cid_arr.shape
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        cids = np.zeros(len(df), dtype=np.int32)
        cids[valid] = cid_arr[rows[valid], cols[valid]]
        df["catchment_id"] = cids
        no_catch = int((cids == 0).sum())
        n_no_catch += no_catch
        df = df[df["catchment_id"] > 0]
        if df.empty:
            continue

        partial = _aggregate_batch(df, feature_cols, min_cols, max_cols, mean_cols, has_elev)
        partials.append(partial)

        if n_total % 10_000_000 < BATCH_SIZE:
            print(f"    {n_total:,} rows read  no-catchment={n_no_catch:,}  "
                  f"partial groups so far: {sum(len(p) for p in partials):,}")

    print(f"  Done: {n_total:,} pixel rows  →  {n_no_catch:,} outside catchments  "
          f"→  {sum(len(p) for p in partials):,} partial catchment rows")
    return partials


def main() -> None:
    t0 = time.time()
    for p in PIXEL_PARQUETS:
        if not p.exists():
            sys.exit(f"Missing pixel parquet: {p}")

    print("Loading catchment raster into memory...")
    cid_arr = _load_catchment_raster()

    # Discover feature columns from first parquet schema
    schema_names = set(pq.ParquetFile(PIXEL_PARQUETS[0]).schema_arrow.names)
    feature_cols = [
        c for c in schema_names
        if c not in EXCLUDE
        and c != "catchment_id"
        and c not in ("row", "col")
    ]
    # Filter to only numeric-looking cols (will be validated at read time)
    # Keep only those present; exclude non-numeric at aggregation time.
    # We identify float/int columns by reading a tiny sample.
    sample = pq.read_table(
        PIXEL_PARQUETS[0],
        columns=list(feature_cols)[:1]  # just to check — we'll use schema types
    )
    schema = pq.ParquetFile(PIXEL_PARQUETS[0]).schema_arrow
    import pyarrow as pa
    feature_cols = [
        c for c in feature_cols
        if c in {schema.field(i).name for i in range(len(schema))}
        and (pa.types.is_integer(schema.field(c).type) or pa.types.is_floating(schema.field(c).type))
    ]

    min_cols  = [c for c in feature_cols if c in MIN_FEATURES]
    max_cols  = [c for c in feature_cols if c in MAX_FEATURES]
    has_elev  = ELEV_COL in feature_cols
    mean_cols = [
        c for c in feature_cols
        if c not in MIN_FEATURES and c not in MAX_FEATURES and c != ELEV_COL
    ]

    print(f"\nFeature columns: {len(feature_cols)} total")
    print(f"  mean={len(mean_cols)}  min={len(min_cols)}  max={len(max_cols)}  "
          f"elev_std={'yes' if has_elev else 'no'}")

    load_cols = list(dict.fromkeys(
        ["country_id", "year", "row", "col", "transition_01", "WDPA_prev"] + feature_cols
    ))
    load_cols = [c for c in load_cols if c in schema_names]

    all_partials: list[pd.DataFrame] = []
    for parquet_path in PIXEL_PARQUETS:
        print(f"\nProcessing: {parquet_path.name}")
        partials = _process_parquet(
            parquet_path, cid_arr, feature_cols,
            min_cols, max_cols, mean_cols, has_elev, load_cols,
        )
        all_partials.extend(partials)
        del partials
        gc.collect()

    print(f"\nCombining {len(all_partials)} partial aggregates...")
    del cid_arr
    gc.collect()
    result = _combine_partials(all_partials)
    del all_partials
    gc.collect()

    n_rows    = len(result)
    n_events  = result.groupby(["country_id", "year"]).ngroups
    n_pos     = int(result["transition_01"].sum())
    total_pos_px = int(result["n_pos_pixels"].sum())

    print(f"\nCatchment-year rows:    {n_rows:,}")
    print(f"(country, year) groups: {n_events}")
    print(f"Positive catchments:    {n_pos:,} ({100*n_pos/n_rows:.2f}%)")
    print(f"Total positive pixels:  {total_pos_px:,}")
    print(f"Mean catchment size:    {result['n_total_pixels'].mean():.1f} pixels")

    # Minimal per-country summary
    country_summary = result.groupby("country_id").agg(
        n_rows=("transition_01", "count"),
        n_pos_catchments=("transition_01", "sum"),
        n_pos_pixels=("n_pos_pixels", "sum"),
    )
    print(f"\nPer-country catchment summary:")
    for cid, row in country_summary.iterrows():
        print(f"  country {cid:>2}: {int(row.n_rows):>7,} rows  "
              f"{int(row.n_pos_catchments):>6,} pos catchments  "
              f"{int(row.n_pos_pixels):>10,} pos pixels")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nSaved → {OUT_PATH}  ({size_mb:.0f} MB)")
    print(f"Columns: {len(result.columns)}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"\nNext step:")
    print(f"  sbatch slurm/south_america/watershed_ce_w.slurm")


if __name__ == "__main__":
    main()
