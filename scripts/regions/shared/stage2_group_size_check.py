#!/usr/bin/env python3
"""Pre-flight Check A — Stage 2 LambdaRank within-group sample sizes.

ROADMAP decision rule:
  - median group positives >= 5  -> proceed with annual (country_id, year) groups
  - median group positives 2-4   -> aggregate to 2-3 year windows before grouping
  - median group positives < 2   -> escalate; group definition needs rethinking

Computes country_id on-the-fly from country_iso3.tif (no feature_engineering rerun).

Usage:
    python scripts/regions/shared/stage2_group_size_check.py
    python scripts/regions/shared/stage2_group_size_check.py --region south_america

Outputs:
    outputs/data_checks/stage2_group_sizes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Bootstrap repo root before package imports
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "README.md").is_file():
            return parent
    raise RuntimeError("Cannot find repo root")


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pyarrow.parquet as pq

from scripts.regions.shared.country_raster import (
    country_ids_for_rows,
    load_country_raster,
    repo_root,
    resolve_panel_path,
)

REGIONS = ("south_america", "usa", "se_asia")
BATCH_SIZE = 500_000
OUT_PATH = repo_root() / "outputs" / "data_checks" / "stage2_group_sizes.json"


def _decision_rule(median_positives: float) -> str:
    if median_positives >= 5:
        return "proceed_annual_groups"
    if median_positives >= 2:
        return "aggregate_2_3yr_windows"
    return "escalate_group_definition"


def analyze_region(region: str) -> dict:
    panel_path = resolve_panel_path(region)
    raster = load_country_raster(region)

    # (country_id, year) -> [n_rows, n_positives]
    group_stats: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])

    pf = pq.ParquetFile(panel_path)
    cols = ["row", "col", "year", "transition_01"]
    for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        if df.empty:
            continue
        cid = country_ids_for_rows(df, raster)
        years = df["year"].to_numpy()
        trans = df["transition_01"].to_numpy(dtype=np.int64)
        for i in range(len(df)):
            key = (int(cid[i]), int(years[i]))
            group_stats[key][0] += 1
            group_stats[key][1] += int(trans[i])

    positives_per_group = np.array([v[1] for v in group_stats.values()], dtype=np.int64)
    rows_per_group = np.array([v[0] for v in group_stats.values()], dtype=np.int64)
    expansion_groups = positives_per_group > 0

    if len(positives_per_group) == 0:
        return {
            "region": region,
            "panel_path": str(panel_path),
            "error": "no_groups_found",
        }

    median_all = float(np.median(positives_per_group))
    median_expansion = (
        float(np.median(positives_per_group[expansion_groups]))
        if expansion_groups.any()
        else 0.0
    )

    result = {
        "region": region,
        "panel_path": str(panel_path),
        "n_groups_total": int(len(group_stats)),
        "n_groups_with_expansion": int(expansion_groups.sum()),
        "median_positives_all_groups": median_all,
        "median_positives_expansion_groups": median_expansion,
        "mean_positives_expansion_groups": float(positives_per_group[expansion_groups].mean()),
        "n_groups_gte5_positives": int((positives_per_group >= 5).sum()),
        "n_groups_gte10_positives": int((positives_per_group >= 10).sum()),
        "n_expansion_groups_gte5": int((positives_per_group[expansion_groups] >= 5).sum()),
        "n_expansion_groups_gte10": int((positives_per_group[expansion_groups] >= 10).sum()),
        "min_positives_expansion_group": int(positives_per_group[expansion_groups].min())
        if expansion_groups.any()
        else None,
        "max_positives_expansion_group": int(positives_per_group[expansion_groups].max())
        if expansion_groups.any()
        else None,
        "median_rows_per_group": float(np.median(rows_per_group)),
        "decision_rule_expansion_groups": _decision_rule(median_expansion),
        "percentiles_positives_expansion": {
            "p10": float(np.percentile(positives_per_group[expansion_groups], 10)),
            "p25": float(np.percentile(positives_per_group[expansion_groups], 25)),
            "p50": median_expansion,
            "p75": float(np.percentile(positives_per_group[expansion_groups], 75)),
            "p90": float(np.percentile(positives_per_group[expansion_groups], 90)),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 pre-flight Check A")
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        help="Region to analyze (default: all three)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_PATH,
        help="Output JSON path",
    )
    args = parser.parse_args()
    regions = args.region or list(REGIONS)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"regions": {}, "errors": []}

    for region in regions:
        print(f"\n=== Check A: {region} ===")
        try:
            r = analyze_region(region)
            results["regions"][region] = r
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue
            print(f"  panel: {r['panel_path']}")
            print(f"  groups (total / expansion): {r['n_groups_total']:,} / {r['n_groups_with_expansion']:,}")
            print(f"  median positives (expansion groups): {r['median_positives_expansion_groups']:.2f}")
            print(f"  expansion groups with >=5 positives: {r['n_expansion_groups_gte5']:,}")
            print(f"  expansion groups with >=10 positives: {r['n_expansion_groups_gte10']:,}")
            print(f"  DECISION: {r['decision_rule_expansion_groups']}")
        except FileNotFoundError as exc:
            msg = f"{region}: {exc}"
            results["errors"].append(msg)
            print(f"  SKIP: {msg}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nJSON written to {args.output}")

    if results["errors"] and not results["regions"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
