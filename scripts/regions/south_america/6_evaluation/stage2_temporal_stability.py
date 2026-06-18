#!/usr/bin/env python3
"""P5.2 — Temporal stability: per-year Recall@5% on test set (2017-2024).

Loads the most recent model1_lgbm_stage2_scored_*.parquet and computes
Lift@1%, Recall@5%, and NDCG@1% per test year.

Key question: does performance degrade toward 2024? Measures temporal drift.
Uses the same metrics as the full test evaluation for direct comparison.

Usage:
    python stage2_temporal_stability.py [--parquet PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.evaluation.stage2_metrics import (
    ndcg_at_k_within_groups,
    precision_at_k_within_groups,
    recall_at_k_within_groups,
)

OUT_DIR = _ROOT / "outputs" / "south_america" / "results" / "ml_models"
SCORED_GLOB = "model1_lgbm_stage2_scored_*.parquet"

ID_TO_ISO3 = {
    1: "ARG", 2: "BOL", 3: "BRA", 4: "CHL", 5: "COL", 6: "ECU",
    7: "GUY", 8: "PRY", 9: "PER", 10: "SUR", 11: "URY", 12: "VEN",
}


def _metrics_for_df(df: pd.DataFrame) -> dict:
    df = df.sort_values(["country_id", "year"]).reset_index(drop=True)
    group_sizes = df.groupby(["country_id", "year"], sort=True).size().to_numpy(dtype=np.int32)
    y = df["y_true"].to_numpy(dtype=np.float64)
    s = df["y_pred_score"].to_numpy(dtype=np.float64)
    prec1 = precision_at_k_within_groups(y, s, group_sizes, 1.0)
    prec5 = precision_at_k_within_groups(y, s, group_sizes, 5.0)
    base = float((y > 0).mean()) if len(y) else 0.0
    return {
        "lift_at_1pct": prec1 / base if base > 0 else 0.0,
        "lift_at_5pct": prec5 / base if base > 0 else 0.0,
        "recall_at_1pct": recall_at_k_within_groups(y, s, group_sizes, 1.0),
        "recall_at_5pct": recall_at_k_within_groups(y, s, group_sizes, 5.0),
        "ndcg_at_1pct": ndcg_at_k_within_groups(y, s, group_sizes, 1.0),
        "precision_at_1pct": prec1,
        "baseline_rate": base,
        "n_groups": int(len(group_sizes)),
        "n_rows": int(len(df)),
    }


def _find_scored_parquet(override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = sorted(OUT_DIR.glob(SCORED_GLOB), reverse=True)
    candidates = [
        p for p in candidates
        if not any(tag in p.name for tag in ["naive", "within_group", "binary", "patch", "mini"])
    ]
    if not candidates:
        raise FileNotFoundError(f"No scored parquet found in {OUT_DIR}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.2 temporal stability per-year Recall@5%")
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    parquet_path = _find_scored_parquet(args.parquet)
    print(f"Loading: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} rows, years {sorted(df['year'].unique())}")

    has_country = "country_id" in df.columns
    if not has_country:
        df["country_id"] = 0

    print("\n=== Per-year Lift@1% and Recall@5% ===")
    print(f"{'Year':<6} {'n_groups':>9} {'n_rows':>13} {'Lift@1%':>10} {'Recall@5%':>11} {'NDCG@1%':>9}")
    print("-" * 64)

    by_year = {}
    for yr in sorted(df["year"].unique()):
        m = _metrics_for_df(df[df["year"] == yr])
        by_year[int(yr)] = m
        print(
            f"{yr:<6} {m['n_groups']:>9} {m['n_rows']:>13,} "
            f"{m['lift_at_1pct']:>9.2f}× {m['recall_at_5pct']*100:>10.1f}% "
            f"{m['ndcg_at_1pct']:>9.4f}"
        )

    overall = _metrics_for_df(df)
    print("-" * 64)
    print(
        f"{'Overall':<6} {overall['n_groups']:>9} {overall['n_rows']:>13,} "
        f"{overall['lift_at_1pct']:>9.2f}× {overall['recall_at_5pct']*100:>10.1f}% "
        f"{overall['ndcg_at_1pct']:>9.4f}"
    )

    # Trend: is performance degrading toward 2024?
    years_sorted = sorted(by_year.keys())
    lifts = [by_year[y]["lift_at_1pct"] for y in years_sorted]
    recalls = [by_year[y]["recall_at_5pct"] for y in years_sorted]
    if len(years_sorted) >= 3:
        first_half = lifts[:len(lifts)//2]
        second_half = lifts[len(lifts)//2:]
        trend = "DEGRADING" if np.mean(second_half) < np.mean(first_half) * 0.8 else "STABLE"
        print(f"\nTemporal trend (Lift@1%): first-half mean={np.mean(first_half):.2f}× "
              f"vs second-half={np.mean(second_half):.2f}× → {trend}")

    if has_country:
        print("\n=== Per-country Recall@5% (across all test years) ===")
        print(f"{'Country':<8} {'Recall@5%':>11} {'Lift@1%':>10} {'n_groups':>9}")
        print("-" * 42)
        by_country = {}
        for cid in sorted(df["country_id"].unique()):
            m = _metrics_for_df(df[df["country_id"] == cid])
            iso3 = ID_TO_ISO3.get(int(cid), f"ID{cid}")
            by_country[iso3] = m
            print(
                f"{iso3:<8} {m['recall_at_5pct']*100:>10.1f}% "
                f"{m['lift_at_1pct']:>9.2f}× {m['n_groups']:>9}"
            )

    result = {
        "parquet": str(parquet_path),
        "overall": overall,
        "by_year": {str(y): m for y, m in by_year.items()},
        "by_country": by_country if has_country else {},
    }

    out_path = Path(args.out) if args.out else OUT_DIR / "stage2_SA_temporal_stability.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
