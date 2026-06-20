"""Aggregate pixel mini-sample to HydroSHEDS L7 catchment level.

For each (country_id, year, catchment_id) group, computes:
  - mean of all feature columns
  - transition_01: 1 if any pixel in the catchment was designated
  - n_pos_pixels:  count of designated pixels (for pixel-level Recall@5% evaluation)
  - n_total_pixels: total unprotected pixels in this catchment-year
  - WDPA_prev_frac: fraction of pixels already protected (should be ~0 in mini-sample)

Input:
  data/south_america/mini_sample_v2.parquet          (15.4M pixel rows)
  data/south_america/ready/hydroshed/catchment_id_sa.tif

Output:
  data/south_america/watershed_mini_sample.parquet   (catchment-level rows)

Usage:
  python scripts/regions/south_america/3_merging/build_watershed_mini_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PIXEL_PATH = PROJECT_ROOT / "data/south_america/mini_sample_v2.parquet"
CATCHMENT_TIF = PROJECT_ROOT / "data/south_america/ready/hydroshed/catchment_id_sa.tif"
OUT_PATH = PROJECT_ROOT / "data/south_america/watershed_mini_sample.parquet"

# Columns excluded from feature averaging (identifiers, labels, leakage)
EXCLUDE_FROM_FEATURES = frozenset({
    "year", "x", "y", "row", "col", "country_id",
    "transition_01", "WDPA_prev",
    "WDPA_b1", "WDPA_b2", "WDPA",  # leakage (see STAGE2_EXCLUDE_COLS)
    "country_iso3",
})


def main() -> None:
    for p in (PIXEL_PATH, CATCHMENT_TIF):
        if not p.exists():
            sys.exit(f"Not found: {p}\n"
                     "Run hydroshed_rasterise.py first." if "catchment" in p.name
                     else f"Not found: {p}")

    print("Loading catchment raster...")
    with rasterio.open(CATCHMENT_TIF) as src:
        cid_arr = src.read(1)  # int32, shape (nrows, ncols)
    n_unique = int(np.unique(cid_arr[cid_arr > 0]).size)
    print(f"  {n_unique:,} unique catchments  grid: {cid_arr.shape}")

    print(f"Loading pixel mini-sample: {PIXEL_PATH}")
    df = pd.read_parquet(PIXEL_PATH)
    # Drop any internal pyarrow fragment columns
    df = df[[c for c in df.columns if not c.startswith("__")]]
    print(f"  {len(df):,} rows  {len(df.columns)} columns")

    # Assign catchment_id via vectorised (row, col) lookup
    rows = df["row"].values.astype(np.int32)
    cols = df["col"].values.astype(np.int32)
    df["catchment_id"] = cid_arr[rows, cols].astype(np.int32)

    n_nocatch = int((df["catchment_id"] == 0).sum())
    if n_nocatch > 0:
        print(f"  Dropping {n_nocatch:,} pixels outside any catchment")
        df = df[df["catchment_id"] > 0].copy()

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and c != "catchment_id"
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("__")
    ]
    print(f"  Feature columns to aggregate: {len(feature_cols)}")

    group_cols = ["country_id", "year", "catchment_id"]
    print("Aggregating to catchment level (this may take ~1 min)...")

    feat_agg = df.groupby(group_cols)[feature_cols].mean()

    label_agg = df.groupby(group_cols).agg(
        n_pos_pixels=("transition_01", "sum"),
        n_total_pixels=("transition_01", "count"),
        WDPA_prev_frac=("WDPA_prev", "mean"),
    )

    result = feat_agg.join(label_agg).reset_index()
    result["transition_01"] = (result["n_pos_pixels"] > 0).astype(np.uint8)
    result["n_pos_pixels"] = result["n_pos_pixels"].astype(np.int32)
    result["n_total_pixels"] = result["n_total_pixels"].astype(np.int32)

    n_rows = len(result)
    n_events = result.groupby(["country_id", "year"]).ngroups
    n_pos = int(result["transition_01"].sum())
    total_pos_px = int(result["n_pos_pixels"].sum())
    print(f"\nCatchment-year rows:    {n_rows:,}")
    print(f"(country, year) groups: {n_events}")
    print(f"Positive catchments:    {n_pos:,} ({100*n_pos/n_rows:.2f}%)")
    print(f"Total positive pixels:  {total_pos_px:,}")
    print(f"Mean catchment size:    {result['n_total_pixels'].mean():.1f} pixels")
    print(f"Mean pos pixels/event:  "
          f"{result[result['transition_01']==1]['n_pos_pixels'].mean():.1f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved → {OUT_PATH}")
    print(f"Columns: {len(result.columns)}  "
          f"(features={len(feature_cols)}, meta=country_id/year/catchment_id, "
          f"labels=transition_01/n_pos_pixels/n_total_pixels/WDPA_prev_frac)")


if __name__ == "__main__":
    main()
