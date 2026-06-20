#!/usr/bin/env python3
"""Build SA mini-sample with event-stratified sampling.

WHY THIS EXISTS
---------------
The original (v1) mini-sample was broken in two ways:
  1. Only sampled from train.parquet — missing years 2017-2024 entirely.
  2. Used proportional random sampling across the whole parquet, so most
     expansion events lost almost all positives (e.g. 400 pos → 4 in sample)
     and were filtered out by the n_pos≥200 threshold in cross-event scripts.

DESIGN
------
Event-stratified sampling fixes both problems:

  Pass 1 (fast, 3 cols): read country_id + year + WDPA_prev from all three
    parquets to count n_total WDPA_prev==0 rows per (country_id, year) event.

  Compute per-event fraction: f = min(1.0, N_CAP / n_total_unprotected).
    - N_CAP = 50,000 rows per event keeps total size ~8-12M rows.
    - Fraction is applied equally to positives and negatives, so the neg/pos
      ratio inside each event is preserved → Recall@5% stays meaningful.

  Pass 2 (full, all cols): read all 94 columns in batches, pre-filter to
    WDPA_prev==0 (protected pixels are never used in training), apply the
    per-event fraction via a fast pandas merge, write output incrementally
    via ParquetWriter (no full dataset in RAM at once).

EXPECTED RESULT
---------------
  ~8-12M rows, all 94 columns, all years 2001-2024.
  ~80-100 of the 139 full-scale meaningful events (n_pos≥200) survive.
  Brazil events: ~3000 positives in mini (vs 354k full), same neg/pos ratio.
  Max Recall@5% in mini is similar to full because ratio is preserved.

USAGE
-----
  On Euler:
    python scripts/regions/south_america/3_merging/build_mini_sample_v2.py

  Or via SLURM:
    sbatch slurm/south_america/build_mini_sample_v2.slurm

  After completion, rsync to local and build splits:
    rsync -avz --progress euler:$SCRATCH/data/south_america/ml/mini_sample_v2.parquet \\
        data/south_america/mini_sample_v2.parquet
    python scripts/regions/south_america/3_merging/prepare_mini_splits_v2.py

ENVIRONMENT VARIABLES
---------------------
  N_CAP_PER_EVENT  max rows to sample per (country_id, year)  [default: 50000]
  MINI_V2_SEED     random seed                                  [default: 42]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED        = int(os.environ.get("MINI_V2_SEED",        "42"))
N_CAP       = int(os.environ.get("N_CAP_PER_EVENT",     "50000"))
BATCH_SIZE  = 500_000   # rows per read batch in pass 2


# ---------------------------------------------------------------------------
# Pass 1: count unprotected rows per (country_id, year)
# ---------------------------------------------------------------------------

def _count_unprotected(paths: list[tuple[str, Path]]) -> dict[tuple[int, int], int]:
    """Return {(country_id, year): n_rows_with_WDPA_prev==0} across all parquets."""
    print("Pass 1: counting unprotected rows per event (3 cols only) ...")
    totals: dict[tuple[int, int], int] = defaultdict(int)

    for label, path in paths:
        pf = pq.ParquetFile(path)
        n_batches = 0
        for batch in pf.iter_batches(
            batch_size=1_000_000,
            columns=["country_id", "year", "WDPA_prev"],
        ):
            df = batch.to_pandas()
            df = df[df["WDPA_prev"] == 0]
            if df.empty:
                continue
            counts = df.groupby(["country_id", "year"]).size()
            for (cid, yr), n in counts.items():
                totals[(int(cid), int(yr))] += int(n)
            n_batches += 1

        # one-line summary per parquet
        rows_in_label = sum(
            n for (_, _), n in totals.items()
        )
        print(f"  {label}: done  ({n_batches} batches)")

    print(f"  {len(totals)} unique (country_id, year) events found")
    return dict(totals)


# ---------------------------------------------------------------------------
# Pass 2: event-stratified sampling → output
# ---------------------------------------------------------------------------

def _build_frac_lookup(event_totals: dict[tuple[int, int], int]) -> pd.DataFrame:
    """Return DataFrame with (country_id, year, _frac) for fast merge in pass 2."""
    rows = []
    for (cid, yr), n_total in event_totals.items():
        frac = min(1.0, N_CAP / n_total) if n_total > 0 else 0.0
        rows.append({"country_id": cid, "year": yr, "_frac": frac})
    return pd.DataFrame(rows)


def _sample_and_write(
    paths: list[tuple[str, Path]],
    frac_df: pd.DataFrame,
    out_path: Path,
    rng: np.random.Generator,
) -> dict:
    """Read all parquets, apply event-stratified sampling, write output."""
    print(f"\nPass 2: sampling and writing → {out_path} ...")

    schema = pq.ParquetFile(paths[0][1]).schema_arrow
    writer = pq.ParquetWriter(out_path, schema, compression="snappy")

    stats: dict = {
        "n_rows_read": 0, "n_rows_kept": 0, "n_pos": 0,
        "per_split": {},
    }

    for label, path in paths:
        pf = pq.ParquetFile(path)
        kept_rows = 0
        kept_pos  = 0

        for batch in pf.iter_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            stats["n_rows_read"] += len(df)

            # Pre-filter: drop already-protected pixels (never used in training)
            df = df[df["WDPA_prev"] == 0]
            if df.empty:
                continue

            # Merge per-event sampling fraction
            df = df.merge(
                frac_df,
                on=["country_id", "year"],
                how="left",
            )
            df["_frac"].fillna(0.0, inplace=True)

            # Stochastic per-row keep decision
            keep_mask = rng.random(len(df)) < df["_frac"].values
            sampled = df.loc[keep_mask].drop(columns=["_frac"])
            if sampled.empty:
                continue

            writer.write_table(pa.Table.from_pandas(sampled, preserve_index=False))
            kept_rows += len(sampled)
            kept_pos  += int((sampled["transition_01"] > 0).sum())

        stats["per_split"][label] = {"n_rows": kept_rows, "n_pos": kept_pos}
        stats["n_rows_kept"] += kept_rows
        stats["n_pos"]       += kept_pos
        print(f"  {label}: kept {kept_rows:,} rows  pos={kept_pos:,}")

    writer.close()
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report_events(out_path: Path, min_pos: int = 200) -> dict:
    """Count meaningful events (n_pos≥min_pos) in the output parquet."""
    print(f"\nCounting events with n_pos≥{min_pos} in mini ...")
    pf = pq.ParquetFile(out_path)
    event_pos: dict[tuple[int, int], int] = defaultdict(int)

    for batch in pf.iter_batches(
        batch_size=1_000_000,
        columns=["country_id", "year", "transition_01"],
    ):
        df = batch.to_pandas()
        pos = df[df["transition_01"] > 0]
        counts = pos.groupby(["country_id", "year"]).size()
        for (cid, yr), n in counts.items():
            event_pos[(int(cid), int(yr))] += int(n)

    meaningful = {k: v for k, v in event_pos.items() if v >= min_pos}
    print(f"  Total events with any positive: {len(event_pos)}")
    print(f"  Meaningful events (n_pos≥{min_pos}): {len(meaningful)}")
    if meaningful:
        top5 = sorted(meaningful.items(), key=lambda x: -x[1])[:5]
        print(f"  Top-5 by n_pos: {[(f'({c},{y})', n) for (c,y),n in top5]}")
    return {"total_events": len(event_pos), "meaningful_events": len(meaningful)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        sys.exit("SCRATCH env var not set — run on Euler or set SCRATCH manually.")

    base     = Path(scratch) / "data/south_america/ml/main"
    out_path = Path(scratch) / "data/south_america/ml/mini_sample_v2.parquet"

    paths = [
        ("train",     base / "train.parquet"),
        ("earlystop", base / "earlystop.parquet"),
        ("test",      base / "test.parquet"),
    ]
    for label, p in paths:
        if not p.exists():
            sys.exit(f"Missing: {p}")

    print(f"Building SA mini-sample v2 (event-stratified, N_CAP={N_CAP:,})")
    print(f"  Input: {base}")
    print(f"  Output: {out_path}")
    print(f"  Seed: {SEED}\n")

    rng = np.random.default_rng(SEED)

    # Pass 1
    event_totals = _count_unprotected(paths)
    frac_df      = _build_frac_lookup(event_totals)

    n_capped   = sum(1 for n in event_totals.values() if n > N_CAP)
    n_full     = sum(1 for n in event_totals.values() if n <= N_CAP)
    print(f"  Capped at {N_CAP:,}: {n_capped} events | kept in full: {n_full} events")

    # Pass 2
    stats = _sample_and_write(paths, frac_df, out_path, rng)

    # Report
    event_info = _report_events(out_path)

    size_mb  = out_path.stat().st_size / 1e6
    meta = {
        "version":        "v2-event-stratified",
        "n_cap_per_event": N_CAP,
        "seed":           SEED,
        "n_rows":         stats["n_rows_kept"],
        "n_pos":          stats["n_pos"],
        "pos_rate":       round(stats["n_pos"] / max(stats["n_rows_kept"], 1), 6),
        "size_mb":        round(size_mb, 1),
        "per_split":      stats["per_split"],
        "events_total":   event_info["total_events"],
        "events_meaningful": event_info["meaningful_events"],
        "note": (
            f"Event-stratified: each (country_id,year) capped at {N_CAP} rows "
            "with proportional pos/neg ratio. Pre-filtered to WDPA_prev==0. "
            "All 3 parquets (train+earlystop+test), all years 2001-2024, all columns."
        ),
    }

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  {stats['n_rows_kept']:,} rows  pos={stats['n_pos']:,}  "
          f"({100*meta['pos_rate']:.3f}%)")
    print(f"  Meaningful events in mini: {event_info['meaningful_events']}")
    print(f"  File size: {size_mb:.0f} MB")
    print(f"  Metadata: {meta_path}")
    print(f"\nNext steps:")
    print(f"  rsync -avz --progress euler:{out_path} \\")
    print(f"      data/south_america/mini_sample_v2.parquet")
    print(f"  python scripts/regions/south_america/3_merging/prepare_mini_splits_v2.py")


if __name__ == "__main__":
    main()
