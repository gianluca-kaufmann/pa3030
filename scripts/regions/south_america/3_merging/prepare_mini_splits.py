"""Split mini_sample.parquet into local train / earlystop splits for Phase 1 tuning.

Writes to data/south_america/mini_splits/main/ so that setting
STAGE2_DATA_ROOT=data/south_america/mini_splits redirects the existing
tuning and training scripts to use local mini-sample data instead of the
full 57 GB SA panel on Euler.

  train.parquet     years 2001-2010  (~2.6M rows)
  earlystop.parquet years 2011-2013  (~0.9M rows)

Re-run this script whenever mini_sample.parquet is updated with new features
(via add_feature_to_mini_sample.py) to propagate the new columns to both splits.

Usage:
  python scripts/regions/south_america/3_merging/prepare_mini_splits.py
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MINI_SAMPLE_PATH = PROJECT_ROOT / "data/south_america/mini_sample.parquet"
OUT_DIR = PROJECT_ROOT / "data/south_america/mini_splits/main"

TRAIN_MAX_YEAR = 2010       # 2001-2010 → train split
ES_MIN_YEAR = 2011          # 2011-2013 → earlystop (proxy validation)
ES_MAX_YEAR = 2013


def main() -> None:
    if not MINI_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Mini-sample not found: {MINI_SAMPLE_PATH}\n"
            "Sync from Euler first: rsync -avz euler:.../mini_sample.parquet "
            "data/south_america/mini_sample.parquet"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {MINI_SAMPLE_PATH}")
    table = pq.read_table(MINI_SAMPLE_PATH)
    print(f"  {len(table):,} rows, {table.num_columns} columns")

    year_col = table.column("year")
    train_mask = pc.less_equal(year_col, TRAIN_MAX_YEAR)
    es_mask = pc.and_(
        pc.greater_equal(year_col, ES_MIN_YEAR),
        pc.less_equal(year_col, ES_MAX_YEAR),
    )

    train = table.filter(train_mask)
    earlystop = table.filter(es_mask)

    train_path = OUT_DIR / "train.parquet"
    es_path = OUT_DIR / "earlystop.parquet"
    pq.write_table(train, train_path, compression="snappy")
    pq.write_table(earlystop, es_path, compression="snappy")

    def _pos_rate(t: pa.Table) -> tuple[int, float]:
        n_pos = pc.sum(pc.cast(t.column("transition_01"), pa.int32())).as_py()
        return n_pos, n_pos / max(len(t), 1)

    tr_n_pos, tr_rate = _pos_rate(train)
    es_n_pos, es_rate = _pos_rate(earlystop)

    print(f"train      (≤{TRAIN_MAX_YEAR}): {len(train):,} rows, "
          f"{tr_n_pos:,} positives ({100*tr_rate:.3f}%) → {train_path}")
    print(f"earlystop ({ES_MIN_YEAR}-{ES_MAX_YEAR}): {len(earlystop):,} rows, "
          f"{es_n_pos:,} positives ({100*es_rate:.3f}%) → {es_path}")
    print()
    print("Ready for local tuning. Run:")
    print("  STAGE2_DATA_ROOT=data/south_america/mini_splits \\")
    print("  python scripts/regions/south_america/4_tuning/model1_phase1_local_tune \\")
    print("    --feature-tag <name>")


if __name__ == "__main__":
    main()
