#!/usr/bin/env python3
"""Rebuild the SA local mini-sample covering ALL years (2001-2024) and ALL features.

The original build_mini_sample.py only sampled from the training split (2001-2013),
leaving out the test period. It also had features stripped by the pruned30 experiment.
This v2 script fixes both problems:
  - Samples proportionally from train + earlystop + test (all years 2001-2024)
  - Preserves every feature column without pruning
  - Writes metadata so the split years and feature list are always traceable

Target row counts (env-var overridable):
  MINI_V2_N_TRAIN     rows from train.parquet      (default: 2_000_000)
  MINI_V2_N_EARLYSTOP rows from earlystop.parquet  (default:   500_000)
  MINI_V2_N_TEST      rows from test.parquet        (default: 1_500_000)
  Total target: ~4 000 000 rows

Output: $SCRATCH/data/south_america/ml/mini_sample_v2.parquet

Usage on Euler:
  python scripts/regions/south_america/3_merging/build_mini_sample_v2.py

After running, rsync to local:
  rsync -avz --progress euler:$SCRATCH/data/south_america/ml/mini_sample_v2.parquet \
      data/south_america/mini_sample_v2.parquet

Then regenerate local splits:
  python scripts/regions/south_america/3_merging/prepare_mini_splits_v2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SEED = 42

N_TRAIN     = int(os.environ.get("MINI_V2_N_TRAIN",     str(2_000_000)))
N_EARLYSTOP = int(os.environ.get("MINI_V2_N_EARLYSTOP", str(  500_000)))
N_TEST      = int(os.environ.get("MINI_V2_N_TEST",      str(1_500_000)))
BATCH_SIZE  = 500_000


def _sample_parquet(path: Path, target_n: int, rng: np.random.Generator,
                    label: str) -> pa.Table:
    pf    = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    frac  = min(1.0, target_n / total)
    print(f"  {label}: {total:,} rows → sampling {frac:.4f} (≈{int(total*frac):,})")

    chunks: list[pa.Table] = []
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        n    = len(batch)
        keep = rng.random(n) < frac
        idx  = np.where(keep)[0]
        if len(idx):
            chunks.append(pa.Table.from_batches([batch]).take(idx))

    result = pa.concat_tables(chunks) if chunks else pa.Table.from_batches([])
    n_pos  = int(pc.sum(pc.cast(result.column("transition_01"), pa.int32())).as_py())
    print(f"    sampled {len(result):,} rows  pos={n_pos:,}  ({100*n_pos/max(len(result),1):.3f}%)")
    return result


def main() -> None:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        sys.exit("SCRATCH env var not set — run on Euler.")

    base     = Path(scratch) / "data/south_america/ml/main"
    out_path = Path(scratch) / "data/south_america/ml/mini_sample_v2.parquet"

    splits = {
        "train":     (base / "train.parquet",     N_TRAIN),
        "earlystop": (base / "earlystop.parquet", N_EARLYSTOP),
        "test":      (base / "test.parquet",       N_TEST),
    }
    for label, (path, _) in splits.items():
        if not path.exists():
            sys.exit(f"Not found: {path}")

    rng    = np.random.default_rng(SEED)
    tables: list[pa.Table] = []
    meta: dict = {"splits": {}}

    print("Sampling from all three splits:")
    for label, (path, target) in splits.items():
        tbl = _sample_parquet(path, target, rng, label)
        tables.append(tbl)
        n_pos = int(pc.sum(pc.cast(tbl.column("transition_01"), pa.int32())).as_py())
        meta["splits"][label] = {
            "path":     str(path),
            "sampled":  len(tbl),
            "n_pos":    n_pos,
            "pos_rate": round(n_pos / max(len(tbl), 1), 6),
        }

    result = pa.concat_tables(tables)
    n_total = len(result)
    n_pos   = int(pc.sum(pc.cast(result.column("transition_01"), pa.int32())).as_py())

    import pandas as pd
    years = sorted(result.column("year").to_pylist())
    year_range = (min(years), max(years))

    meta.update({
        "n_rows":       n_total,
        "n_pos":        n_pos,
        "pos_rate":     round(n_pos / n_total, 6),
        "n_columns":    result.num_columns,
        "columns":      result.schema.names,
        "year_range":   year_range,
        "seed":         SEED,
        "targets":      {"train": N_TRAIN, "earlystop": N_EARLYSTOP, "test": N_TEST},
        "note": (
            "v2: samples from all three splits (train+earlystop+test), all years 2001-2024. "
            "No feature pruning. Use prepare_mini_splits_v2.py to create local splits."
        ),
    })

    print(f"\nTotal: {n_total:,} rows  pos={n_pos:,}  ({100*n_pos/n_total:.3f}%)")
    print(f"Year range: {year_range[0]}–{year_range[1]}")
    print(f"Columns: {result.num_columns}")
    print(f"Writing: {out_path}")
    pq.write_table(result, out_path, compression="snappy")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Done: {size_mb:.0f} MB")

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata: {meta_path}")

    print("\nNext steps:")
    print(f"  rsync -avz --progress euler:{out_path} data/south_america/mini_sample_v2.parquet")
    print("  python scripts/regions/south_america/3_merging/prepare_mini_splits_v2.py")


if __name__ == "__main__":
    main()
