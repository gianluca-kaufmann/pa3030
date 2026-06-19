#!/usr/bin/env python3
"""Split mini_sample_v2.parquet into local train / earlystop / test splits.

  train.parquet      years 2001-2013  (model training window)
  earlystop.parquet  years 2014-2016  (model early-stop window)
  test.parquet       years 2017-2024  (model test window — local proxy for Euler test)

Unlike the v1 splits, the test split is present locally because v2 sampled from
all three Euler parquets. This makes E1-style gap analysis runnable locally.

Usage:
  python scripts/regions/south_america/3_merging/prepare_mini_splits_v2.py

Output: data/south_america/mini_splits_v2/main/{train,earlystop,test}.parquet
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_ROOT      = Path(__file__).resolve().parents[4]
SRC_PATH   = _ROOT / "data/south_america/mini_sample_v2.parquet"
OUT_DIR    = _ROOT / "data/south_america/mini_splits_v2/main"

SPLITS = {
    "train":     (2001, 2013),
    "earlystop": (2014, 2016),
    "test":      (2017, 2024),
}


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(
            f"mini_sample_v2.parquet not found at {SRC_PATH}\n"
            "Run build_mini_sample_v2.py on Euler, then rsync:\n"
            "  rsync -avz --progress euler:$SCRATCH/data/south_america/ml/mini_sample_v2.parquet \\\n"
            "      data/south_america/mini_sample_v2.parquet"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {SRC_PATH}")
    table = pq.read_table(SRC_PATH)
    print(f"  {len(table):,} rows  {table.num_columns} cols")

    year_col = table.column("year")
    for name, (y_min, y_max) in SPLITS.items():
        mask   = pc.and_(pc.greater_equal(year_col, y_min), pc.less_equal(year_col, y_max))
        split  = table.filter(mask)
        n_pos  = int(pc.sum(pc.cast(split.column("transition_01"), pa.int32())).as_py())
        rate   = n_pos / max(len(split), 1)
        n_grp  = len(
            split.select(["country_id", "year"]).to_pandas().drop_duplicates()
        )
        out    = OUT_DIR / f"{name}.parquet"
        pq.write_table(split, out, compression="snappy")
        print(f"  {name:12s} ({y_min}-{y_max}): {len(split):>9,} rows  "
              f"pos={n_pos:,} ({100*rate:.3f}%)  {n_grp} groups  → {out.name}")

    print(f"\nDone. Set STAGE2_DATA_ROOT=data/south_america/mini_splits_v2 "
          "to use these splits in training scripts.")


if __name__ == "__main__":
    main()
