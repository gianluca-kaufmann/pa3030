#!/usr/bin/env python3
"""B4 (E3): Aggregate mini_sample_v2 pixels to HydroSHEDS L7 catchment features.

Each row in the output represents one (country_id, year, catchment_id) triplet.
Continuous features are mean-aggregated across the pixels in that catchment-event.
The target label is:
  - catchment_is_positive (0/1): any pixel in catchment was designated
  - n_positive_pixels: count of designated pixels (used for recall evaluation)
  - n_total_pixels: total pixels in catchment (denominator)

Prerequisites:
  1. mini_sample_v2.parquet at data/south_america/ml/mini_sample_v2.parquet
     (rsync from Euler if needed)
  2. hydrosheds_l7_sa.tif at data/south_america/ready/HydroSHEDS/hydrosheds_l7_sa.tif
     (run hydrosheds_rasterise.py first)

Usage:
  python scripts/regions/south_america/3_merging/build_watershed_sample.py

Output:
  data/south_america/watershed_sample.parquet

Columns:
  country_id, year, catchment_id        ← group keys
  n_total_pixels, n_positive_pixels     ← recall evaluation
  catchment_is_positive                  ← binary label for LambdaRank
  <feature>_mean for each pixel feature  ← training features
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

_ROOT          = Path(__file__).resolve().parents[4]
MINI_PATH      = _ROOT / "data/south_america/ml/mini_sample_v2.parquet"
HYDRO_TIF      = _ROOT / "data/south_america/ready/HydroSHEDS/hydrosheds_l7_sa.tif"
OUTPUT_PATH    = _ROOT / "data/south_america/watershed_sample.parquet"
BATCH_SIZE     = 500_000

# Columns to exclude from feature means (same logic as STAGE2_EXCLUDE_COLS)
NON_FEATURE_COLS = frozenset({
    "transition_01", "transition_01_win5",
    "WDPA_b1", "WDPA_b2", "WDPA_prev", "WDPA",
    "x", "y", "row", "col",
    "year", "country_id", "country_iso3",
    "log_patch_size_km2", "patch_designation_lag1",  # patch-level features (meaningless at catchment)
    "patch_mean_gsn_b2", "patch_pa_adjacency_frac",
})


def _load_hydro_raster() -> np.ndarray:
    """Load catchment_id raster into memory (int32 array)."""
    with rasterio.open(HYDRO_TIF) as src:
        arr = src.read(1)  # int32, 0 = no catchment
    print(f"HydroSHEDS raster: {arr.shape}  "
          f"n_assigned={int((arr > 0).sum()):,}  "
          f"unique_catchments={len(np.unique(arr[arr > 0])):,}")
    return arr


def _sample_catchment_ids(hydro: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = hydro.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.zeros(len(rows), dtype=np.int32)
    if valid.any():
        out[valid] = hydro[rows[valid], cols[valid]]
    return out


def main() -> None:
    for p, name in [(MINI_PATH, "mini_sample_v2"), (HYDRO_TIF, "hydrosheds_l7_sa.tif")]:
        if not p.exists():
            sys.exit(
                f"{name} not found: {p}\n"
                + ("Run build_mini_sample_v2.py on Euler and rsync." if "mini" in name
                   else "Run hydrosheds_rasterise.py first.")
            )

    print(f"Loading HydroSHEDS raster...")
    hydro = _load_hydro_raster()

    # Identify feature columns from schema
    schema = pq.ParquetFile(MINI_PATH).schema_arrow
    all_cols = schema.names
    feature_cols = [
        c for c in all_cols
        if c not in NON_FEATURE_COLS
        and (pa.types.is_integer(schema.field(c).type) or pa.types.is_floating(schema.field(c).type))
    ]
    print(f"Pixel features to aggregate: {len(feature_cols)}")

    load_cols = list(dict.fromkeys(
        ["country_id", "year", "row", "col", "transition_01", "WDPA_prev"] + feature_cols
    ))
    load_cols = [c for c in load_cols if c in all_cols]

    # ── Pass: read mini-sample in batches, assign catchment_id, accumulate ──
    # We accumulate sums + counts in a dict: (country_id, year, catchment_id) → arrays
    # For efficiency, collect everything in a list and pandas-aggregate at the end.

    print(f"\nReading {MINI_PATH.name} and assigning catchment IDs...")
    chunks: list[pd.DataFrame] = []
    n_total = 0
    n_nocatch = 0

    pf = pq.ParquetFile(MINI_PATH)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=load_cols):
        df = batch.to_pandas()
        n_total += len(df)

        # Only unprotected pixels (same as model training)
        df = df[df["WDPA_prev"] == 0].copy()
        if df.empty:
            continue

        rows_arr = df["row"].to_numpy(dtype=np.int64)
        cols_arr = df["col"].to_numpy(dtype=np.int64)
        catch_ids = _sample_catchment_ids(hydro, rows_arr, cols_arr)

        df["catchment_id"] = catch_ids
        n_nocatch += int((catch_ids == 0).sum())

        # Drop pixels with no catchment assignment (ocean/edge)
        df = df[df["catchment_id"] > 0]
        if df.empty:
            continue

        chunks.append(df)

        if n_total % 2_000_000 < BATCH_SIZE:
            print(f"  {n_total:,} rows read…")

    print(f"\nRead complete: {n_total:,} total rows  |  no-catchment pixels: {n_nocatch:,}")

    if not chunks:
        sys.exit("No pixels survived catchment assignment.")

    print("Concatenating and aggregating to catchment level...")
    df_all = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  Combined: {len(df_all):,} rows  |  unique catchments: {df_all['catchment_id'].nunique():,}")

    # ── Aggregate ──
    group_keys = ["country_id", "year", "catchment_id"]

    agg_dict: dict[str, object] = {
        "transition_01": ["sum", "count"],
    }
    for fc in feature_cols:
        if fc in df_all.columns:
            agg_dict[fc] = "mean"

    print("  Running groupby aggregation...")
    agg = df_all.groupby(group_keys, as_index=False).agg(agg_dict)

    # Flatten MultiIndex columns
    flat_cols: list[str] = []
    for col in agg.columns:
        if isinstance(col, tuple):
            parent, func = col
            if parent in group_keys:
                flat_cols.append(parent)
            elif func == "sum":
                flat_cols.append(f"n_positive_pixels" if parent == "transition_01" else f"{parent}_sum")
            elif func == "count":
                flat_cols.append("n_total_pixels" if parent == "transition_01" else f"{parent}_count")
            else:
                flat_cols.append(f"{parent}_mean")
        else:
            flat_cols.append(col)
    agg.columns = flat_cols

    # Binary label: catchment has any positive pixel
    agg["catchment_is_positive"] = (agg["n_positive_pixels"] > 0).astype(np.int8)

    # Stats
    n_catchments = len(agg)
    n_positive   = int(agg["catchment_is_positive"].sum())
    n_events     = agg.groupby(["country_id", "year"]).ngroups
    print(f"\nWatershed sample stats:")
    print(f"  Catchment-events: {n_catchments:,}")
    print(f"  Positive catchments: {n_positive:,}  ({100*n_positive/max(n_catchments,1):.2f}%)")
    print(f"  (country_id, year) events: {n_events}")
    print(f"  Total positive pixels: {int(agg['n_positive_pixels'].sum()):,}")
    print(f"  Mean pixels/catchment: {agg['n_total_pixels'].mean():.1f}")
    print(f"  Median pixels/catchment: {agg['n_total_pixels'].median():.1f}")

    # Event-level summary
    ev_agg = agg[agg["catchment_is_positive"] > 0].groupby(["country_id", "year"]).agg(
        n_pos_catchments=("catchment_is_positive", "sum"),
        n_all_catchments=("catchment_is_positive", "count"),
        n_positive_pixels=("n_positive_pixels", "sum"),
    ).reset_index()
    meaningful = ev_agg[ev_agg["n_positive_pixels"] >= 200]
    print(f"  Meaningful events (n_pos_px ≥ 200): {len(meaningful)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(agg, preserve_index=False)
    pq.write_table(table, OUTPUT_PATH, compression="snappy")

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\nSaved: {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    meta = {
        "n_catchment_events": n_catchments,
        "n_positive_catchments": n_positive,
        "n_events": n_events,
        "n_meaningful_events": len(meaningful),
        "feature_cols": [f"{fc}_mean" for fc in feature_cols if f"{fc}_mean" in flat_cols],
        "source": str(MINI_PATH),
        "hydrosheds": str(HYDRO_TIF),
    }
    (OUTPUT_PATH.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))

    print(f"\nNext step:")
    print(f"  python scripts/regions/south_america/5_training/"
          f"model1_LGBM_stage2_watershed.py")


if __name__ == "__main__":
    main()
