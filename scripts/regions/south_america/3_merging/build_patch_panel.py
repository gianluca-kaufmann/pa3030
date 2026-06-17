#!/usr/bin/env python3
"""Build patch-level panel for Tier 3 patch-level ranking (H12).

Aggregates pixel-level Stage 2 splits into patch-level panels where each row
is one 8-connected component (patch) of unprotected land (WDPA_prev==0) within
a (country_id, year) group.

A patch is positive if any pixel inside it was designated (transition_01 > 0).
Feature aggregation: min for dist_* columns, mean for everything else.

Output: patch_train.parquet, patch_earlystop.parquet, patch_test.parquet
in the same directory as the input splits.

Usage (Euler, from repo root):
    srun python scripts/regions/south_america/3_merging/build_patch_panel.py

Env vars:
  SCRATCH — Euler scratch path (mandatory on Euler)
  STAGE2_DATA_ROOT — override splits directory
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import label as scipy_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_STRUCT8 = np.ones((3, 3), dtype=np.int32)

# Columns to exclude from feature aggregation (not features, group keys, or leakage)
EXCLUDE_COLS = frozenset({
    "transition_01",
    "transition_01_win5",
    "WDPA", "WDPA_b1", "WDPA_b2", "WDPA_prev",
    "x", "y", "row", "col",
    "year", "country_id", "country_iso3",
})

PA_ADJACENCY_DIST_M = 1500.0


def _resolve_splits_dir() -> Path:
    data_root = os.environ.get("STAGE2_DATA_ROOT")
    if data_root:
        for p in [Path(data_root) / "main", Path(data_root)]:
            if p.exists():
                return p
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "data/south_america/ml/main"
    raise RuntimeError("Set SCRATCH or STAGE2_DATA_ROOT to locate splits directory.")


def _aggregate_year(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute CCs and aggregate pixels to patches for one year.

    Returns a DataFrame with one row per (country_id, year, cc_label),
    with aggregated features and binary transition_01 label.
    """
    rows_arr = df["row"].to_numpy(np.int64)
    cols_arr = df["col"].to_numpy(np.int64)

    # Build a boolean grid covering the bounding box of all unprotected pixels
    min_row = int(rows_arr.min())
    max_row = int(rows_arr.max())
    min_col = int(cols_arr.min())
    max_col = int(cols_arr.max())
    h = max_row - min_row + 1
    w = max_col - min_col + 1

    r_idx = (rows_arr - min_row).astype(np.int64)
    c_idx = (cols_arr - min_col).astype(np.int64)

    grid = np.zeros((h, w), dtype=np.bool_)
    grid[r_idx, c_idx] = True

    labeled, _ = scipy_label(grid, structure=_STRUCT8)
    patch_ids = labeled[r_idx, c_idx].astype(np.int32)  # 1..n_labels
    del grid, labeled
    gc.collect()

    df = df.copy()
    df["_patch_id"] = patch_ids

    # Build aggregation dict: transition_01 is max; dist_* features use min; others use mean.
    # log_patch_size_km2 is always recomputed from CC pixel count (more accurate than H11's
    # version, which came from the merged panel and may differ slightly from the split).
    agg: dict[str, tuple[str, str]] = {
        "transition_01": ("transition_01", "max"),
        "_n_pixels": ("row", "count"),
    }
    for col in feature_cols:
        if col == "log_patch_size_km2":
            continue  # recomputed below from CC count
        agg_fn = "min" if col.startswith("dist_") else "mean"
        agg[col] = (col, agg_fn)

    # If patch_pa_adjacency_frac not in feature_cols (splits without H11), compute it here.
    if "patch_pa_adjacency_frac" not in feature_cols and "dist_wdpa" in df.columns:
        df["_pa_adj"] = (df["dist_wdpa"] <= PA_ADJACENCY_DIST_M).astype(np.float32)
        agg["patch_pa_adjacency_frac"] = ("_pa_adj", "mean")

    patch_df = (
        df.groupby(["country_id", "year", "_patch_id"], sort=False)
        .agg(**agg)
        .reset_index()
    )

    # Recompute log_patch_size_km2 from the actual CC pixel count
    patch_df["log_patch_size_km2"] = np.log(
        np.maximum(patch_df["_n_pixels"].to_numpy(), 1)
    ).astype(np.float32)
    patch_df["patch_n_pixels"] = patch_df["_n_pixels"].astype(np.int32)

    patch_df.drop(columns=["_patch_id", "_n_pixels"], inplace=True)

    # Cast all feature columns to float32
    cast_cols = [c for c in feature_cols if c in patch_df.columns and c != "log_patch_size_km2"]
    for col in cast_cols:
        patch_df[col] = patch_df[col].astype(np.float32)

    return patch_df


def _process_split(split_path: Path, output_path: Path, feature_cols: list[str]) -> None:
    log.info(f"\nProcessing {split_path.name} → {output_path.name}")
    log.info(f"  Size: {split_path.stat().st_size / 1e9:.1f} GB")

    # Discover years in this split
    year_df = pq.read_table(split_path, columns=["year"]).to_pandas()
    years = sorted(year_df["year"].unique())
    del year_df
    gc.collect()
    log.info(f"  Years: {years[0]}–{years[-1]}  ({len(years)} years)")

    read_cols = list(dict.fromkeys(
        ["row", "col", "year", "country_id", "WDPA_prev", "transition_01"] + feature_cols
    ))
    # Remove columns that may not exist (patch H11 features may or may not be present)
    schema_names = set(pq.ParquetFile(split_path).schema_arrow.names)
    read_cols = [c for c in read_cols if c in schema_names]
    actual_feature_cols = [c for c in feature_cols if c in schema_names]

    writer: pq.ParquetWriter | None = None
    t_total = time.time()
    total_patches = 0

    for year in years:
        t0 = time.time()
        df = pq.read_table(
            split_path,
            columns=read_cols,
            filters=[("year", "==", year), ("WDPA_prev", "==", 0)],
        ).to_pandas()

        # Also filter out rows where transition_01 is NaN (right-censored or missing)
        df = df[df["transition_01"].notna()].reset_index(drop=True)

        if df.empty:
            log.warning(f"  Year {year}: no unprotected pixels — skipping")
            continue

        n_pixels = len(df)
        patch_df = _aggregate_year(df, actual_feature_cols)
        del df
        gc.collect()

        n_patches = len(patch_df)
        n_pos = int((patch_df["transition_01"] > 0).sum())
        total_patches += n_patches
        log.info(
            f"  {year}: {n_pixels:,} pixels → {n_patches:,} patches"
            f"  ({n_pos} positive = {100*n_pos/max(n_patches,1):.2f}%)"
            f"  [{time.time()-t0:.0f}s]"
        )

        table = pa.Table.from_pandas(patch_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)
        del patch_df, table
        gc.collect()

    if writer is not None:
        writer.close()

    elapsed = time.time() - t_total
    size_mb = output_path.stat().st_size / 1e6
    log.info(f"  Written: {output_path}  ({size_mb:.0f} MB, {total_patches:,} patches, {elapsed:.0f}s)")


def main() -> None:
    splits_dir = _resolve_splits_dir()
    log.info(f"Splits directory: {splits_dir}")

    # Discover feature columns from train.parquet schema (all numeric, non-excluded)
    train_path = splits_dir / "train.parquet"
    pf = pq.ParquetFile(train_path)
    schema = pf.schema_arrow
    feature_cols = [
        name for name, field in zip(schema.names, schema)
        if name not in EXCLUDE_COLS
        and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
    ]
    log.info(f"Pixel feature columns: {len(feature_cols)}")
    log.info(f"  First 10: {feature_cols[:10]}")

    for split_name in ("train", "earlystop", "test"):
        in_path = splits_dir / f"{split_name}.parquet"
        out_path = splits_dir / f"patch_{split_name}.parquet"
        if not in_path.exists():
            log.warning(f"SKIP: {in_path} not found")
            continue
        _process_split(in_path, out_path, feature_cols)

    log.info("\nAll patch panels built.")


if __name__ == "__main__":
    main()
