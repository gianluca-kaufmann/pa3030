#!/usr/bin/env python3
"""Create Colombia-only dev panel from the SA Stage 2 splits.

Filters train/earlystop/test parquets to a single country (default: Colombia,
country_iso3='COL') and writes Colombia-specific splits to data/dev/south_america/ml/.
The output directory can then be passed as STAGE2_DATA_ROOT to run Stage 2 scripts
on this ~7M-row subset without any code changes.

Usage (on Euler after SA FE job completes):

    python scripts/regions/south_america/dev/create_colombia_panel.py

    # Verify Colombia has ≥5 expansion training groups before using for iteration:
    python scripts/regions/south_america/dev/create_colombia_panel.py --check-only

    # Then run Stage 2 tuning on Colombia dev panel:
    STAGE2_DATA_ROOT=<repo>/data/dev/south_america/ml \\
        python scripts/regions/south_america/5_training/model1_tune_stage2_binary

Design notes
------------
- STAGE2_DATA_ROOT is checked first in resolve_parquet_file, before SCRATCH.
  Set it to the directory containing main/{train,earlystop,test}.parquet.
- Colombia signals are directional only; hyperparameters do NOT transfer to the
  full SA model. Use for Issues D/O architectural comparison only.
- Prerequisite: SA FE job must have completed so train/earlystop/test parquets
  exist at the standard Euler SCRATCH location.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.training.stage2_lgbm_core import (
    BATCH_SIZE,
    TARGET_COL,
    _expansion_groups_from_batches,
    resolve_parquet_file,
)

REGION = "south_america"
SPLITS = [
    ("train.parquet", (2001, 2013)),
    ("earlystop.parquet", (2014, 2016)),
    ("test.parquet", (2017, 2024)),
]
TRAIN_YEARS = (2001, 2013)


def _filter_to_country(
    in_path: Path,
    out_path: Path,
    country_iso3: str | None,
    country_id_val: int | None,
) -> int:
    """Stream in_path, filter to country, write out_path. Returns row count."""
    pf = pq.ParquetFile(in_path)
    schema = pf.schema_arrow
    has_iso3 = "country_iso3" in schema.names
    has_cid = "country_id" in schema.names

    if country_id_val is None and not has_iso3:
        raise ValueError(
            "Panel has no country_iso3 column. Use --country-id <int> to specify"
            " the country_id value directly (check the raster or a sample row)."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        tbl = batch.cast(schema)
        if country_id_val is not None and has_cid:
            mask = pa.compute.equal(tbl.column("country_id"), country_id_val)
        elif has_iso3:
            mask = pa.compute.equal(tbl.column("country_iso3"), country_iso3)
        else:
            raise RuntimeError("No country filter possible.")
        filtered = tbl.filter(mask)
        if filtered.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(out_path, filtered.schema, compression="zstd")
        writer.write_table(filtered)
        total_rows += filtered.num_rows

    if writer is not None:
        writer.close()
    return total_rows


def main(country_iso3: str, country_id_val: int | None, check_only: bool) -> None:
    repo_root = _ROOT
    out_base = repo_root / "data" / "dev" / REGION / "ml" / "main"

    print(f"Colombia dev panel builder")
    print(f"  Country filter: iso3={country_iso3}" + (f", id={country_id_val}" if country_id_val else ""))
    print(f"  Output: {out_base}")
    print()

    # Resolve input split paths
    split_paths: dict[str, Path] = {}
    for filename, _ in SPLITS:
        try:
            p = resolve_parquet_file(REGION, filename)
            split_paths[filename] = p
            print(f"  Found {filename}: {p}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    print()

    # Check expansion groups in training window
    train_path = split_paths["train.parquet"]
    print("Scanning expansion groups in training window (2001–2013)...")
    t0 = time.time()
    expansion_groups = _expansion_groups_from_batches(train_path, REGION, TRAIN_YEARS)
    print(f"  SA total expansion country-years (train): {len(expansion_groups):,} ({time.time()-t0:.1f}s)")

    if check_only:
        print("\n--check-only: skipping write. Run without flag to create the dev panel.")
        return

    # Filter each split
    for filename, year_range in SPLITS:
        in_path = split_paths[filename]
        out_path = out_base / filename
        print(f"Filtering {filename} → {out_path.relative_to(repo_root)}")
        t1 = time.time()
        n_rows = _filter_to_country(in_path, out_path, country_iso3, country_id_val)
        print(f"  Written {n_rows:,} rows ({time.time()-t1:.1f}s)")

    # Verify Colombia expansion groups in training window
    out_train = out_base / "train.parquet"
    print("\nVerifying Colombia expansion groups in training window...")
    col_expansion = _expansion_groups_from_batches(out_train, REGION, TRAIN_YEARS)
    n_exp = len(col_expansion)
    status = "OK" if n_exp >= 5 else "WARNING: fewer than 5 groups — signals may be unreliable"
    print(f"  Colombia expansion country-years (2001–2013): {n_exp} [{status}]")

    print(f"\nDone. Run Stage 2 scripts with:")
    print(f"  STAGE2_DATA_ROOT={out_base} python scripts/regions/south_america/5_training/model1_tune_stage2_binary")
    print(f"  STAGE2_DATA_ROOT={out_base} python scripts/regions/south_america/5_training/model1_LGBM_stage2_binary")


if __name__ == "__main__":
    # pyarrow.compute is needed for the filter; import here to give a clear error
    try:
        import pyarrow.compute  # noqa: F401
    except ImportError:
        print("ERROR: pyarrow.compute not available. Upgrade pyarrow.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Create Colombia-only dev panel from SA splits")
    parser.add_argument("--country-iso3", default="COL", help="ISO3 country code (default: COL)")
    parser.add_argument("--country-id", type=int, default=None, help="Integer country_id from the raster (overrides iso3 filter if country_iso3 column absent)")
    parser.add_argument("--check-only", action="store_true", help="Only report expansion groups; do not write files")
    args = parser.parse_args()

    main(args.country_iso3, args.country_id, args.check_only)
