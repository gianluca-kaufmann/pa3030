"""Add new feature columns from a raster to the full SA Stage 2 splits on Euler.

Reads each split parquet (train, earlystop, test) from the SA data directory,
samples the raster at every pixel's (row, col) position, appends the new
feature columns, and writes the augmented parquet back.  Run this on Euler before
submitting the Phase 1 tuning + training gate job.

Usage (Euler):
  python scripts/regions/south_america/3_merging/add_feature_to_splits.py \\
      --tif $SCRATCH/data/south_america/ready/KBA/kba_sa.tif \\
      --cols is_kba dist_kba_km

Env vars:
  SCRATCH — Euler scratch path (mandatory on Euler; optional if STAGE2_DATA_ROOT is set)
  STAGE2_DATA_ROOT — override split directory (default: $SCRATCH/data/south_america/ml/main/)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

BATCH_SIZE = 500_000


def _resolve_splits_dir() -> Path:
    data_root = os.environ.get("STAGE2_DATA_ROOT")
    if data_root:
        p = Path(data_root) / "main"
        if p.exists():
            return p
        p = Path(data_root)
        if p.exists():
            return p
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "data/south_america/ml/main"
    raise RuntimeError(
        "Cannot find splits directory. Set SCRATCH or STAGE2_DATA_ROOT."
    )


def _sample_raster(
    tif_path: Path, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Sample all bands of `tif_path` at (row, col) positions → float32 (n, n_bands)."""
    with rasterio.open(tif_path) as src:
        n_bands = src.count
        h, w = src.height, src.width
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.full((len(rows), n_bands), np.nan, dtype=np.float32)
        if valid.any():
            data = src.read()
            for b in range(n_bands):
                out[valid, b] = data[b][rows[valid], cols[valid]]
    return out


def _augment_split(
    in_path: Path, out_path: Path, tif_path: Path, col_names: list[str]
) -> None:
    """Read one parquet split, add raster columns, write augmented version."""
    if not in_path.exists():
        print(f"  SKIP (not found): {in_path.name}")
        return

    print(f"  {in_path.name}  ({in_path.stat().st_size / 1e9:.1f} GB)...")
    pf = pq.ParquetFile(in_path)

    # Existing columns (we'll remove any that conflict with new names)
    existing_cols = pf.schema_arrow.names
    drop_cols = [c for c in col_names if c in existing_cols]
    if drop_cols:
        print(f"    Replacing existing columns: {drop_cols}")

    writer = None
    rows_processed = 0
    with rasterio.open(tif_path) as _:  # pre-open to validate
        pass

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        table = pa.Table.from_batches([batch])

        # Drop conflicting columns
        if drop_cols:
            table = table.select([c for c in table.column_names if c not in drop_cols])

        rows_arr = table.column("row").to_numpy(zero_copy_only=False).astype(np.int64)
        cols_arr = table.column("col").to_numpy(zero_copy_only=False).astype(np.int64)

        values = _sample_raster(tif_path, rows_arr, cols_arr)
        for i, col_name in enumerate(col_names):
            table = table.append_column(col_name, pa.array(values[:, i], type=pa.float32()))

        if writer is None:
            writer = pq.ParquetWriter(
                out_path, table.schema, compression="snappy"
            )
        writer.write_table(table)
        rows_processed += len(table)

    if writer is not None:
        writer.close()
        print(f"    {rows_processed:,} rows written → {out_path.name}")
    else:
        print(f"    WARNING: no rows written for {in_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add raster feature columns to full SA Stage 2 splits"
    )
    parser.add_argument("--tif", required=True, type=Path,
                        help="Path to backbone-aligned raster (bands = new features)")
    parser.add_argument("--cols", required=True, nargs="+",
                        help="Column name for each raster band")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite the original splits (default: write .augmented first)")
    args = parser.parse_args()

    tif_path = args.tif
    if not tif_path.exists():
        sys.exit(f"TIF not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        n_bands = src.count
    if len(args.cols) != n_bands:
        sys.exit(f"TIF has {n_bands} bands but {len(args.cols)} col names given.")

    splits_dir = _resolve_splits_dir()
    print(f"Splits directory: {splits_dir}")
    print(f"TIF: {tif_path}")
    print(f"New columns: {args.cols}")

    for split_name in ("train.parquet", "earlystop.parquet", "test.parquet"):
        in_path = splits_dir / split_name
        if args.inplace:
            tmp_path = splits_dir / (split_name + ".tmp")
            _augment_split(in_path, tmp_path, tif_path, args.cols)
            if tmp_path.exists():
                tmp_path.rename(in_path)
                print(f"    Replaced in-place: {in_path.name}")
        else:
            out_path = splits_dir / (split_name.replace(".parquet", "_augmented.parquet"))
            _augment_split(in_path, out_path, tif_path, args.cols)

    if not args.inplace:
        print()
        print("Augmented splits written with suffix _augmented.parquet.")
        print("Review, then rename or re-run with --inplace to overwrite originals.")


if __name__ == "__main__":
    main()
