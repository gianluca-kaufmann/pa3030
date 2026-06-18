#!/usr/bin/env python3
"""P1.4 — Stage 2 baseline comparison table.

Compares the locked H6+H1b+H5 model against:
  1. Random ranking (theoretical: Lift@1% = 1.0×)
  2. dist_wdpa-only naive baseline (single-feature LambdaRank)
  3. Full model (H6+H1b+H5, all 79 features)

Outputs a LaTeX table and a JSON summary.

Note: the naive baseline scored parquet covers 2017-2019 only.
Comparison is done on the 2017-2019 intersection, plus the full model
alone on 2017-2024 for reference.

Usage:
    python stage2_baselines_comparison.py [--naive PATH] [--full PATH] [--out DIR]
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
    lift_at_k_within_groups,
    precision_at_k_within_groups,
    recall_at_k_within_groups,
    ndcg_at_k_within_groups,
)

OUT_DIR = _ROOT / "outputs" / "south_america" / "results" / "ml_models"
SCORED_GLOB_FULL = "model1_lgbm_stage2_scored_*.parquet"
SCORED_GLOB_NAIVE = "model1_lgbm_stage2_naive_scored_*.parquet"


def _find_parquet(glob: str, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = sorted(OUT_DIR.glob(glob), reverse=True)
    # Exclude variant files
    candidates = [p for p in candidates if "_within_group_" not in p.name
                  and "_binary_" not in p.name and "_patch_" not in p.name
                  and "_mini_" not in p.name]
    if not candidates:
        raise FileNotFoundError(f"No file matching {glob} in {OUT_DIR}")
    return candidates[0]


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


def _random_metrics(df: pd.DataFrame) -> dict:
    """Random ranking: Precision@k% = baseline_rate; Lift@k% = 1.0×."""
    df = df.sort_values(["country_id", "year"]).reset_index(drop=True)
    group_sizes = df.groupby(["country_id", "year"], sort=True).size().to_numpy(dtype=np.int32)
    y = df["y_true"].to_numpy(dtype=np.float64)
    base = float((y > 0).mean()) if len(y) else 0.0
    return {
        "lift_at_1pct": 1.0,
        "lift_at_5pct": 1.0,
        "recall_at_1pct": 0.01,
        "recall_at_5pct": 0.05,
        "ndcg_at_1pct": float("nan"),
        "precision_at_1pct": base,
        "baseline_rate": base,
        "n_groups": int(len(group_sizes)),
        "n_rows": int(len(df)),
    }


def _latex_row(name: str, m: dict) -> str:
    return (
        f"{name} & {m['n_groups']} & {m['baseline_rate']*100:.2f}\\% "
        f"& {m['lift_at_1pct']:.2f}$\\times$ & {m['recall_at_5pct']*100:.1f}\\% "
        f"& {m['ndcg_at_1pct']:.4f} \\\\"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P1.4 Stage 2 baseline comparison")
    parser.add_argument("--naive", default=None, help="Path to naive scored parquet")
    parser.add_argument("--full", default=None, help="Path to full model scored parquet")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    naive_path = _find_parquet(SCORED_GLOB_NAIVE, args.naive)
    full_path = _find_parquet(SCORED_GLOB_FULL, args.full)
    print(f"Full model: {full_path.name}")
    print(f"Naive:      {naive_path.name}")

    df_full = pd.read_parquet(full_path)
    df_naive = pd.read_parquet(naive_path)

    # Find the common (country_id, year) groups
    full_groups = set(zip(df_full["country_id"], df_full["year"]))
    naive_groups = set(zip(df_naive["country_id"], df_naive["year"]))
    common_groups = full_groups & naive_groups
    print(f"\nFull model groups (2017-2024): {len(full_groups)}")
    print(f"Naive groups:                  {len(naive_groups)}")
    print(f"Common groups:                 {len(common_groups)}")

    # Restrict to common groups for head-to-head comparison (pandas merge is vectorized)
    common_df = pd.DataFrame(list(common_groups), columns=["country_id", "year"])
    df_full_common = df_full.merge(common_df, on=["country_id", "year"])
    df_naive_common = df_naive.merge(common_df, on=["country_id", "year"])
    print(f"Full model rows (common groups): {len(df_full_common):,}")
    print(f"Naive rows (common groups):      {len(df_naive_common):,}")

    # Compute metrics
    m_random = _random_metrics(df_naive_common)
    m_naive = _metrics_for_df(df_naive_common)
    m_full_common = _metrics_for_df(df_full_common)
    m_full_all = _metrics_for_df(df_full)

    print("\n=== Baseline Comparison (common groups: 2017-2019) ===")
    for label, m in [
        ("Random ranking", m_random),
        ("dist_wdpa only (naive)", m_naive),
        ("H6+H1b+H5 full model", m_full_common),
    ]:
        print(
            f"  {label:<30} Lift@1%={m['lift_at_1pct']:.2f}×  "
            f"Recall@5%={m['recall_at_5pct']*100:.1f}%  "
            f"NDCG@1%={m['ndcg_at_1pct']:.4f}  n_groups={m['n_groups']}"
        )

    print(f"\n  Full model (all 2017-2024):       Lift@1%={m_full_all['lift_at_1pct']:.2f}×  "
          f"Recall@5%={m_full_all['recall_at_5pct']*100:.1f}%  "
          f"NDCG@1%={m_full_all['ndcg_at_1pct']:.4f}  n_groups={m_full_all['n_groups']}")

    # Relative improvement vs random and naive
    lift_gain_vs_random = m_full_common["lift_at_1pct"] / m_random["lift_at_1pct"]
    lift_gain_vs_naive = m_full_common["lift_at_1pct"] / m_naive["lift_at_1pct"]
    print(f"\n  Full model vs random:  {lift_gain_vs_random:.2f}× improvement in Lift@1%")
    print(f"  Full model vs naive:   {lift_gain_vs_naive:.2f}× improvement in Lift@1%")

    # LaTeX table
    latex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Stage 2 pixel-ranking baseline comparison (South America, 2017--2019, " + str(len(common_groups)) + r" expansion groups)}",
        r"\label{tab:stage2_baselines}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Model & Groups & Pos. rate & Lift@1\% & Recall@5\% & NDCG@1\% \\",
        r"\hline",
        _latex_row("Random ranking", m_random),
        _latex_row("dist\\_wdpa only", m_naive),
        _latex_row("H6+H1b+H5 (79 features)", m_full_common),
        r"\hline",
        _latex_row(r"H6+H1b+H5 (2017--2024$^\dagger$)", m_full_all),
        r"\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item $^\dagger$ Full test period. Naive baseline not available for 2020--2024.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    latex_str = "\n".join(latex)

    result = {
        "naive_path": str(naive_path),
        "full_path": str(full_path),
        "common_groups": len(common_groups),
        "random_ranking_common": m_random,
        "naive_dist_wdpa_common": m_naive,
        "full_model_common": m_full_common,
        "full_model_all_years": m_full_all,
        "lift_gain_vs_random": lift_gain_vs_random,
        "lift_gain_vs_naive": lift_gain_vs_naive,
    }

    json_path = out_dir / "stage2_SA_baselines_comparison.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {json_path}")

    latex_path = out_dir / "stage2_SA_baselines_comparison.tex"
    latex_path.write_text(latex_str)
    print(f"Saved: {latex_path}")
    print("\nLaTeX table:")
    print(latex_str)


if __name__ == "__main__":
    main()
