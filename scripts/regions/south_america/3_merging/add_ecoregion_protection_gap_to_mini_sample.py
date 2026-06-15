"""Inject ecoregion protection gap columns into mini_sample.parquet.

Unlike static features (KBA, AGB), the ecoregion protection gap is time-varying:
each pixel gets the protection fraction of its ecoregion in the *previous* year.
This script handles the (eco_id, year) join instead of TIF-based (row, col) sampling.

Columns added:
  eco_protection_ratio_lag1  — fraction of pixel's ecoregion protected at year t-1
  eco_protection_gap_lag1    — max(0, 0.30 - eco_protection_ratio_lag1)

Prerequisite:
  python scripts/regions/south_america/2_preprocessing/ecoregion_protection_gap_rasterise.py

Usage:
  python scripts/regions/south_america/3_merging/add_ecoregion_protection_gap_to_mini_sample.py

Then re-split:
  python scripts/regions/south_america/3_merging/prepare_mini_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MINI_SAMPLE_PATH = PROJECT_ROOT / "data/south_america/mini_sample.parquet"
LOOKUP_PATH = PROJECT_ROOT / "data/south_america/ready/eco_protection_gap/eco_protection_gap_lookup.parquet"
ECO_TIF = PROJECT_ROOT / "data/south_america/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif"

NEW_COLS = ["eco_protection_ratio_lag1", "eco_protection_gap_lag1"]


def _sample_eco_ids(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample ecoregion IDs from the GSN TIF at (row, col) positions."""
    with rasterio.open(ECO_TIF) as src:
        h, w = src.height, src.width
        grid = src.read(1)   # uint8 (9172, 8974)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    eco_ids = np.zeros(len(rows), dtype=np.int16)
    eco_ids[valid] = grid[rows[valid], cols[valid]]
    return eco_ids


def main() -> None:
    for path, label in [(MINI_SAMPLE_PATH, "mini_sample"), (LOOKUP_PATH, "lookup")]:
        if not path.exists():
            sys.exit(
                f"{label} not found: {path}\n"
                "Run ecoregion_protection_gap_rasterise.py first."
            )

    # Load lookup: (ecoregion_id, panel_year) → features
    print(f"Loading lookup: {LOOKUP_PATH}")
    lookup = pd.read_parquet(LOOKUP_PATH)
    lookup_idx = lookup.set_index(["ecoregion_id", "panel_year"])
    print(f"  {len(lookup):,} rows, years {lookup['panel_year'].min()}-{lookup['panel_year'].max()}")

    print(f"Reading: {MINI_SAMPLE_PATH}")
    table = pq.read_table(MINI_SAMPLE_PATH)
    print(f"  {len(table):,} rows, {table.num_columns} columns")

    rows_arr = table.column("row").to_numpy(zero_copy_only=False).astype(np.int64)
    cols_arr = table.column("col").to_numpy(zero_copy_only=False).astype(np.int64)
    years_arr = table.column("year").to_numpy(zero_copy_only=False).astype(np.int16)

    print("Sampling ecoregion IDs from GSN TIF...")
    eco_ids = _sample_eco_ids(rows_arr, cols_arr)
    print(f"  Unique ecoregion IDs: {len(np.unique(eco_ids[eco_ids > 0]))}")

    # Build index arrays for the (eco_id, year) join
    keys = list(zip(eco_ids.tolist(), years_arr.tolist()))

    print("Joining lookup by (eco_id, year)...")
    ratio_vals = np.full(len(rows_arr), np.nan, dtype=np.float32)
    gap_vals = np.full(len(rows_arr), np.nan, dtype=np.float32)

    for col_name in NEW_COLS:
        pass  # placeholder — using vectorised join below

    # Vectorised join via pandas merge
    join_df = pd.DataFrame({
        "ecoregion_id": eco_ids,
        "panel_year": years_arr,
        "_idx": np.arange(len(rows_arr), dtype=np.int32),
    })
    merged = join_df.merge(
        lookup,
        left_on=["ecoregion_id", "panel_year"],
        right_on=["ecoregion_id", "panel_year"],
        how="left",
    )
    ratio_vals = merged["eco_protection_ratio_lag1"].to_numpy(dtype=np.float32)
    gap_vals = merged["eco_protection_gap_lag1"].to_numpy(dtype=np.float32)

    n_valid = int(np.isfinite(ratio_vals).sum())
    print(f"  eco_protection_ratio_lag1: {n_valid:,}/{len(ratio_vals):,} valid, "
          f"mean={np.nanmean(ratio_vals):.3f}")
    print(f"  eco_protection_gap_lag1: mean={np.nanmean(gap_vals):.3f}")

    # Drop existing columns (if re-running)
    keep = [c for c in table.column_names if c not in NEW_COLS]
    if len(keep) < table.num_columns:
        dropped = [c for c in NEW_COLS if c in table.column_names]
        print(f"  Replacing existing columns: {dropped}")
        table = table.select(keep)

    table = table.append_column(
        "eco_protection_ratio_lag1",
        pa.array(ratio_vals, type=pa.float32()),
    )
    table = table.append_column(
        "eco_protection_gap_lag1",
        pa.array(gap_vals, type=pa.float32()),
    )

    pq.write_table(table, MINI_SAMPLE_PATH, compression="snappy")
    size_mb = MINI_SAMPLE_PATH.stat().st_size / 1e6
    print(f"\nUpdated: {MINI_SAMPLE_PATH} ({size_mb:.0f} MB, {table.num_columns} columns)")
    print()
    print("Next:")
    print("  python scripts/regions/south_america/3_merging/prepare_mini_splits.py")


if __name__ == "__main__":
    main()
