#!/usr/bin/env python3
"""P5.1 — Bootstrap confidence intervals on Stage 2 test-set metrics.

Resamples test-set (country_id, year) groups 1000 times with replacement and
computes 95% CIs on Lift@1%, Recall@5%, and NDCG@1%.

Group-level bootstrap (not pixel-level) preserves within-group correlations
and is the appropriate unit: our model is evaluated across groups, and the
natural "independent observation" is an expansion event, not a pixel.

Usage:
    python stage2_bootstrap_ci.py [--parquet PATH] [--n-boot 1000] [--seed 42]
    python stage2_bootstrap_ci.py --parquet PATH --out PATH
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
        raise FileNotFoundError(f"No scored parquet in {OUT_DIR}")
    return candidates[0]


def _metrics_one_bootstrap(
    y: np.ndarray,
    s: np.ndarray,
    group_sizes: np.ndarray,
) -> dict[str, float]:
    prec1 = precision_at_k_within_groups(y, s, group_sizes, 1.0)
    prec5 = precision_at_k_within_groups(y, s, group_sizes, 5.0)
    base = float((y > 0).mean()) if len(y) else 0.0
    return {
        "lift_at_1pct": prec1 / base if base > 0 else 0.0,
        "lift_at_5pct": prec5 / base if base > 0 else 0.0,
        "recall_at_1pct": recall_at_k_within_groups(y, s, group_sizes, 1.0),
        "recall_at_5pct": recall_at_k_within_groups(y, s, group_sizes, 5.0),
        "ndcg_at_1pct": ndcg_at_k_within_groups(y, s, group_sizes, 1.0),
    }


def bootstrap_groups(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    """Group-level bootstrap: resample (country_id, year) groups with replacement."""
    rng = np.random.default_rng(seed)
    group_keys = sorted(df.groupby(["country_id", "year"]).groups.keys())
    n_groups = len(group_keys)
    print(f"  {n_groups} groups, {len(df):,} rows, {int((df['y_true'] > 0).sum()):,} positives")

    # Pre-split into per-group arrays (avoids repeated dataframe filtering)
    print("  Pre-splitting groups...", flush=True)
    t0 = time.time()
    df_sorted = df.sort_values(["country_id", "year"]).reset_index(drop=True)
    group_sizes_arr = df_sorted.groupby(["country_id", "year"], sort=True).size().to_numpy(dtype=np.int32)
    y_global = df_sorted["y_true"].to_numpy(dtype=np.float64)
    s_global = df_sorted["y_pred_score"].to_numpy(dtype=np.float64)

    # Split into list of arrays for efficient resampling
    splits_y: list[np.ndarray] = np.split(y_global, np.cumsum(group_sizes_arr)[:-1])
    splits_s: list[np.ndarray] = np.split(s_global, np.cumsum(group_sizes_arr)[:-1])
    print(f"  Pre-split done ({time.time()-t0:.1f}s)", flush=True)

    # Observed metrics on full data
    observed = _metrics_one_bootstrap(y_global, s_global, group_sizes_arr)
    print(f"  Observed: Lift@1%={observed['lift_at_1pct']:.2f}×  "
          f"Recall@5%={observed['recall_at_5pct']*100:.1f}%  "
          f"NDCG@1%={observed['ndcg_at_1pct']:.4f}", flush=True)

    # Bootstrap
    boot_results: dict[str, list[float]] = {k: [] for k in observed}
    t0 = time.time()
    for i in range(n_boot):
        idx = rng.integers(0, n_groups, size=n_groups)
        y_boot = np.concatenate([splits_y[j] for j in idx])
        s_boot = np.concatenate([splits_s[j] for j in idx])
        sz_boot = np.array([len(splits_y[j]) for j in idx], dtype=np.int32)
        m = _metrics_one_bootstrap(y_boot, s_boot, sz_boot)
        for k, v in m.items():
            boot_results[k].append(v)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_boot - i - 1)
            print(f"  Bootstrap {i+1}/{n_boot} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    return observed, boot_results


def ci_summary(observed: dict, boot_results: dict, alpha: float = 0.05) -> dict:
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    summary = {}
    for k, vals in boot_results.items():
        arr = np.array(vals)
        summary[k] = {
            "observed": observed[k],
            "ci_lo": float(np.percentile(arr, lo)),
            "ci_hi": float(np.percentile(arr, hi)),
            "se": float(arr.std()),
            "n_boot": len(arr),
        }
    return summary


def _latex_ci_table(summary: dict) -> str:
    rows = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Stage 2 test-set metrics with 95\% bootstrap confidence intervals "
        r"(group-level resampling, 1000 iterations, South America 2017--2024)}",
        r"\label{tab:stage2_bootstrap_ci}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Metric & Observed & 95\% CI lower & 95\% CI upper \\",
        r"\hline",
    ]
    display = {
        "lift_at_1pct": ("Lift@1\\%", "{:.2f}\\times"),
        "lift_at_5pct": ("Lift@5\\%", "{:.2f}\\times"),
        "recall_at_1pct": ("Recall@1\\%", "{:.1f}\\%"),
        "recall_at_5pct": ("Recall@5\\%", "{:.1f}\\%"),
        "ndcg_at_1pct": ("NDCG@1\\%", "{:.4f}"),
    }
    for k, (label, fmt) in display.items():
        if k not in summary:
            continue
        d = summary[k]
        scale = 100 if "recall" in k else 1.0
        obs = d["observed"] * scale
        lo = d["ci_lo"] * scale
        hi = d["ci_hi"] * scale
        obs_str = fmt.format(obs)
        lo_str = fmt.format(lo)
        hi_str = fmt.format(hi)
        rows.append(f"{label} & {obs_str} & {lo_str} & {hi_str} \\\\")
    rows += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.1 bootstrap CIs on Stage 2 metrics")
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    parquet_path = _find_scored_parquet(args.parquet)
    print(f"Loading: {parquet_path.name}")
    df = pd.read_parquet(parquet_path)
    if "country_id" not in df.columns:
        df["country_id"] = 0
    print(f"  {len(df):,} rows, years {sorted(df['year'].unique())}")

    print(f"\nRunning {args.n_boot} bootstrap iterations (group-level resampling)...")
    observed, boot_results = bootstrap_groups(df, args.n_boot, args.seed)

    summary = ci_summary(observed, boot_results)

    print("\n=== Bootstrap CIs ===")
    print(f"{'Metric':<20} {'Observed':>12} {'95% CI':>22}")
    print("-" * 58)
    for k, d in summary.items():
        scale = 100 if "recall" in k else 1.0
        sfx = "%" if "recall" in k else ("×" if "lift" in k else "")
        print(
            f"{k:<20} {d['observed']*scale:>10.3f}{sfx}  "
            f"[{d['ci_lo']*scale:.3f}{sfx}, {d['ci_hi']*scale:.3f}{sfx}]"
        )

    latex = _latex_ci_table(summary)
    print("\nLaTeX table:\n" + latex)

    out_dir = Path(args.out) if args.out else OUT_DIR
    json_path = out_dir / "stage2_SA_bootstrap_ci.json"
    json_path.write_text(json.dumps({"parquet": str(parquet_path), "metrics": summary}, indent=2))
    print(f"\nSaved: {json_path}")

    latex_path = out_dir / "stage2_SA_bootstrap_ci.tex"
    latex_path.write_text(latex)
    print(f"Saved: {latex_path}")


if __name__ == "__main__":
    main()
