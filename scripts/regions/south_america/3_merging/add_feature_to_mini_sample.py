"""Inject raster feature bands into mini_sample.parquet for local Phase 1 iteration.

Reads a backbone-aligned GeoTIF, samples each band at the (row, col) positions in
mini_sample.parquet, and writes the result back in-place.  Run after rasterising a
new feature and before model1_phase1_local_tune.

Usage:
  # KBA (2-band: is_kba, dist_kba_km)
  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\
      --tif data/south_america/ready/KBA/kba_sa.tif \\
      --cols is_kba dist_kba_km

  # Single-band raster
  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\
      --tif data/south_america/ready/AGB/agb_sa.tif \\
      --cols agb_tonne_ha

The TIF must share the SA backbone grid (same resolution, origin, and CRS) so that
row/col indices from mini_sample.parquet address the correct pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MINI_SAMPLE_PATH = PROJECT_ROOT / "data/south_america/mini_sample.parquet"


def _sample_raster(tif_path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample all bands of `tif_path` at (row, col) positions.

    Returns float32 array of shape (n_pixels, n_bands).
    Out-of-bounds positions receive NaN.
    """
    with rasterio.open(tif_path) as src:
        n_bands = src.count
        h, w = src.height, src.width
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.full((len(rows), n_bands), np.nan, dtype=np.float32)
        if valid.any():
            data = src.read()            # (n_bands, h, w)
            for b in range(n_bands):
                out[valid, b] = data[b][rows[valid], cols[valid]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject raster bands into mini_sample.parquet")
    parser.add_argument("--tif", required=True, type=Path,
                        help="Path to backbone-aligned GeoTIF (relative to project root)")
    parser.add_argument("--cols", required=True, nargs="+",
                        help="Column names for each band (must match band count)")
    args = parser.parse_args()

    tif_path = args.tif if args.tif.is_absolute() else PROJECT_ROOT / args.tif
    if not tif_path.exists():
        raise FileNotFoundError(f"TIF not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        n_bands = src.count
    if len(args.cols) != n_bands:
        raise ValueError(
            f"TIF has {n_bands} band(s) but {len(args.cols)} column name(s) supplied. "
            f"Provide exactly one name per band."
        )

    if not MINI_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Mini-sample not found: {MINI_SAMPLE_PATH}")

    print(f"Reading: {MINI_SAMPLE_PATH}")
    table = pq.read_table(MINI_SAMPLE_PATH)

    rows = table.column("row").to_numpy(zero_copy_only=False).astype(np.int64)
    cols_arr = table.column("col").to_numpy(zero_copy_only=False).astype(np.int64)

    print(f"Sampling {tif_path.name} at {len(rows):,} positions...")
    values = _sample_raster(tif_path, rows, cols_arr)

    # Drop columns that already exist so we can overwrite them
    existing = set(table.column_names)
    keep_cols = [c for c in table.column_names if c not in args.cols]
    if len(keep_cols) < len(table.column_names):
        replaced = [c for c in args.cols if c in existing]
        print(f"  Replacing existing column(s): {replaced}")
        table = table.select(keep_cols)

    for i, col_name in enumerate(args.cols):
        col_data = values[:, i]
        n_valid = int(np.isfinite(col_data).sum())
        print(f"  {col_name}: {n_valid:,}/{len(col_data):,} valid, "
              f"mean={np.nanmean(col_data):.4f}")
        table = table.append_column(col_name, pa.array(col_data, type=pa.float32()))

    pq.write_table(table, MINI_SAMPLE_PATH, compression="snappy")
    size_mb = MINI_SAMPLE_PATH.stat().st_size / 1e6
    print(f"Updated: {MINI_SAMPLE_PATH} ({size_mb:.0f} MB, {table.num_columns} columns)")


if __name__ == "__main__":
    main()
