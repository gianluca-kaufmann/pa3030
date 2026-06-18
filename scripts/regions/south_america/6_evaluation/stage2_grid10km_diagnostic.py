#!/usr/bin/env python3
"""P4.1 diagnostic — 10×10 km fixed-grid group-size analysis.

Checks whether aggregating from 1 km pixels to 10×10 km cells gives
viable LambdaRank group sizes (≥50 cells per expansion group).

Design:
  - Load test.parquet (2017-2024 expansion groups) minimal columns
  - Assign each pixel to a 10×10 km cell: cell_row = row // 10, cell_col = col // 10
  - Count unique cells per (country_id, year) group
  - Count positive cells (cells with ≥1 designated pixel) per group
  - Report: distribution of cells per group, viability threshold, positive rate

The ROADMAP decision threshold: viable if all major countries have ≥50 cells
per expansion year. Colombia (~20-50) flagged as potentially marginal.

Output: JSON summary + printed table.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.training.stage2_lgbm_core import (
    BATCH_SIZE,
    TARGET_COL,
    _expansion_groups_from_batches,
)
from scripts.regions.shared.training.utils import get_repo_root
from scripts.regions.shared.country_raster import load_country_raster, country_ids_for_rows

CELL_SIZE = 10  # 10×10 km cells

ID_TO_ISO3 = {
    1: "ARG", 2: "BOL", 3: "BRA", 4: "CHL", 5: "COL", 6: "ECU",
    7: "GUY", 8: "PRY", 9: "PER", 10: "SUR", 11: "URY", 12: "VEN",
}


def main() -> None:
    repo_root = get_repo_root()
    out_dir = repo_root / "outputs" / "south_america" / "results" / "ml_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    region = "south_america"

    # Resolve test parquet
    scratch = os.environ.get("SCRATCH")
    candidates = []
    if scratch:
        candidates.append(Path(scratch) / f"data/{region}/ml/main/test.parquet")
    candidates.append(repo_root / f"outputs/{region}/results/main/test.parquet")
    test_path = next((p for p in candidates if p.exists()), None)
    if test_path is None:
        raise FileNotFoundError("test.parquet not found; run on Euler with $SCRATCH set")
    print(f"Loading: {test_path}")

    schema_names = pq.ParquetFile(test_path).schema_arrow.names
    has_country = "country_id" in schema_names
    raster = None if has_country else load_country_raster(region)

    # First pass: find expansion groups
    print("Scanning for expansion groups (2017-2024)...")
    expansion_groups = _expansion_groups_from_batches(test_path, region, (2017, 2024))
    print(f"  {len(expansion_groups):,} expansion country-years")

    # Second pass: load row, col, year, country_id, transition_01
    cols = [TARGET_COL, "year", "WDPA_prev", "row", "col"]
    if has_country:
        cols.append("country_id")

    records: list[dict] = []
    pf = pq.ParquetFile(test_path)
    n_loaded = 0
    for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        df = df[df[TARGET_COL].notna() & (df["WDPA_prev"] == 0)]
        df = df[(df["year"] >= 2017) & (df["year"] <= 2024)]
        if df.empty:
            continue
        if not has_country:
            df["country_id"] = country_ids_for_rows(df, raster)
        keys = list(zip(df["country_id"].astype(int), df["year"].astype(int)))
        df = df[np.array([k in expansion_groups for k in keys], dtype=bool)]
        if df.empty:
            continue

        # Assign 10×10 km cell IDs
        df["cell_row"] = (df["row"].astype(np.int32) // CELL_SIZE).astype(np.int32)
        df["cell_col"] = (df["col"].astype(np.int32) // CELL_SIZE).astype(np.int32)
        df["is_pos"] = (df[TARGET_COL] > 0).astype(np.int8)

        n_loaded += len(df)
        records.append(df[["country_id", "year", "cell_row", "cell_col", "is_pos"]].copy())
        del df

    print(f"Loaded {n_loaded:,} rows from {len(expansion_groups)} expansion groups")
    all_df = pd.concat(records, ignore_index=True)
    del records

    # Aggregate to 10×10 km cells: each cell is a unique (country_id, year, cell_row, cell_col)
    cell_agg = (
        all_df.groupby(["country_id", "year", "cell_row", "cell_col"])
        .agg(pixel_count=("is_pos", "size"), pos_pixels=("is_pos", "sum"))
        .reset_index()
    )
    cell_agg["is_pos_cell"] = (cell_agg["pos_pixels"] > 0).astype(int)

    # Per-(country_id, year) group stats
    group_stats = (
        cell_agg.groupby(["country_id", "year"])
        .agg(
            n_cells=("cell_row", "size"),
            n_pos_cells=("is_pos_cell", "sum"),
        )
        .reset_index()
    )
    group_stats["pos_cell_rate"] = group_stats["n_pos_cells"] / group_stats["n_cells"]
    group_stats["iso3"] = group_stats["country_id"].map(ID_TO_ISO3).fillna("UNK")

    print("\n=== 10×10 km Cell Count per Expansion Group ===")
    print(f"{'Country':<8} {'Year':<6} {'n_cells':>10} {'n_pos_cells':>12} {'pos_rate':>10}")
    print("-" * 50)
    for _, row in group_stats.sort_values(["country_id", "year"]).iterrows():
        print(
            f"{row['iso3']:<8} {int(row['year']):<6} {int(row['n_cells']):>10,}"
            f" {int(row['n_pos_cells']):>12,} {row['pos_cell_rate']*100:>9.2f}%"
        )

    print("\n=== Summary ===")
    print(f"Total expansion groups:  {len(group_stats)}")
    print(f"Min cells per group:     {group_stats['n_cells'].min():,}  ({group_stats.loc[group_stats['n_cells'].idxmin(), 'iso3']} {group_stats.loc[group_stats['n_cells'].idxmin(), 'year']:.0f})")
    print(f"Max cells per group:     {group_stats['n_cells'].max():,}")
    print(f"Median cells per group:  {group_stats['n_cells'].median():,.0f}")
    print(f"Groups with <50 cells:   {(group_stats['n_cells'] < 50).sum()}")
    print(f"Groups with <10 pos cells: {(group_stats['n_pos_cells'] < 10).sum()}")
    print(f"\nPositive cell rate:")
    print(f"  Overall: {group_stats['n_pos_cells'].sum() / group_stats['n_cells'].sum() * 100:.2f}%")
    print(f"  Median:  {group_stats['pos_cell_rate'].median() * 100:.2f}%")
    print(f"\nVS pixel-level positive rate: 0.22%")
    print(f"→ 10×10 km cells improve positive rate by ~{group_stats['n_pos_cells'].sum() / group_stats['n_cells'].sum() / 0.0022:.1f}×")

    # Verdict
    n_small = int((group_stats["n_cells"] < 50).sum())
    n_few_pos = int((group_stats["n_pos_cells"] < 10).sum())
    if n_small == 0 and n_few_pos == 0:
        verdict = "VIABLE — all groups have ≥50 cells and ≥10 positive cells; proceed with P4.1 rebuild"
    elif n_small <= 3:
        verdict = f"MARGINAL — {n_small} groups have <50 cells; consider dropping tiny groups"
    else:
        verdict = f"NOT VIABLE — {n_small} groups have <50 cells; pixel framing preferred"

    print(f"\nVerdict: {verdict}")

    # Per-country summary
    country_summary = (
        group_stats.groupby("iso3")
        .agg(
            n_groups=("n_cells", "count"),
            min_cells=("n_cells", "min"),
            median_cells=("n_cells", "median"),
            min_pos_cells=("n_pos_cells", "min"),
        )
        .reset_index()
    )
    print("\n=== Per-country summary ===")
    print(country_summary.sort_values("median_cells").to_string(index=False))

    result = {
        "region": region,
        "cell_size_km": CELL_SIZE,
        "n_expansion_groups": len(group_stats),
        "min_cells_per_group": int(group_stats["n_cells"].min()),
        "max_cells_per_group": int(group_stats["n_cells"].max()),
        "median_cells_per_group": float(group_stats["n_cells"].median()),
        "n_groups_below_50_cells": n_small,
        "n_groups_below_10_pos_cells": n_few_pos,
        "overall_pos_cell_rate": float(group_stats["n_pos_cells"].sum() / group_stats["n_cells"].sum()),
        "pixel_pos_rate": 0.0022,
        "verdict": verdict,
        "per_group": group_stats.rename(columns={"n_cells": "total_cells"}).to_dict(orient="records"),
    }

    out_path = out_dir / "stage2_SA_grid10km_diagnostic.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
