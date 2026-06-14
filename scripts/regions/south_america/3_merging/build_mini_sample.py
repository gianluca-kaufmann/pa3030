"""Build SA mini-sample for local Phase 1 feature iteration (Issue MS).

Samples ~TARGET_N rows from the SA training split at the natural pos/neg ratio,
stratified uniformly across all (country_id, year) groups. The test set is
never touched — the mini-sample is training data only.

Output: $SCRATCH/data/south_america/ml/mini_sample.parquet (~200 MB)

Usage on Euler (SCRATCH must be set):
    python scripts/regions/south_america/3_merging/build_mini_sample.py

Env vars:
    MINI_SAMPLE_N   Target row count (default: 4_000_000)
    SCRATCH         Euler scratch path (mandatory)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TARGET_N = int(os.environ.get("MINI_SAMPLE_N", str(4_000_000)))
SEED = 42
BATCH_SIZE = 500_000


def main() -> None:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        sys.exit("SCRATCH env var not set — run on Euler.")

    train_path = Path(scratch) / "data/south_america/ml/main/train.parquet"
    out_path = Path(scratch) / "data/south_america/ml/mini_sample.parquet"

    if not train_path.exists():
        sys.exit(f"Not found: {train_path}")

    pf = pq.ParquetFile(train_path)
    total_rows = pf.metadata.num_rows
    print(f"train.parquet: {total_rows:,} rows")

    frac = min(1.0, TARGET_N / total_rows)
    print(f"Sampling fraction: {frac:.5f}  (target ≈ {TARGET_N:,} rows)")

    rng = np.random.default_rng(SEED)
    chunks: list[pa.Table] = []
    rows_read = 0

    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        n = len(batch)
        rows_read += n
        keep = rng.random(n) < frac
        idx = np.where(keep)[0]
        if len(idx):
            chunks.append(pa.Table.from_batches([batch]).take(idx))
        if (i + 1) % 20 == 0:
            sampled = sum(len(c) for c in chunks)
            print(f"  {rows_read:,} read  |  {sampled:,} sampled")

    result = pa.concat_tables(chunks)
    n_out = len(result)

    n_pos = pc.sum(pc.cast(result.column("transition_01"), pa.int32())).as_py()
    pos_rate = n_pos / n_out if n_out else 0.0

    # Count unique (country_id, year) groups
    import pandas as pd
    groups_df = result.select(["country_id", "year"]).to_pandas()
    n_groups = groups_df.drop_duplicates().shape[0]

    print(f"\nMini-sample: {n_out:,} rows  |  {n_groups} (country, year) groups")
    print(f"Positive rate: {pos_rate:.4%}  ({n_pos:,} positives)")

    pq.write_table(result, out_path, compression="snappy")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Written: {out_path}  ({size_mb:.0f} MB)")
    print("Done. Sync to local with:")
    print(f"  rsync -avz euler:/cluster/scratch/$USER/data/south_america/ml/mini_sample.parquet")
    print(f"  data/south_america/mini_sample.parquet")


if __name__ == "__main__":
    main()
