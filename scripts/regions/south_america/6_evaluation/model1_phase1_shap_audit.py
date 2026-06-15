#!/usr/bin/env python3
"""SHAP feature audit for the Phase 1 baseline model.

Loads the saved booster and earlystop split, computes SHAP values,
and produces two outputs:
  outputs/south_america/results/phase1/baseline/shap_importance.png
  outputs/south_america/results/phase1/baseline/shap_beeswarm.png

Usage:
  python scripts/regions/south_america/6_evaluation/model1_phase1_shap_audit.py
  python scripts/regions/south_america/6_evaluation/model1_phase1_shap_audit.py \
      --tag baseline --n-rows 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lightgbm as lgb
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa
import shap

from scripts.regions.shared.training.stage2_lgbm_core import STAGE2_EXCLUDE_COLS, TARGET_COL

PHASE1_DIR = _ROOT / "outputs/south_america/results/phase1"
ES_PATH = _ROOT / "data/south_america/mini_splits/main/earlystop.parquet"


def _load_earlystop(n_rows: int | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    schema = pq.ParquetFile(ES_PATH).schema_arrow
    feature_cols = [
        n for n, f in zip(schema.names, schema)
        if (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))
        and n not in STAGE2_EXCLUDE_COLS
    ]
    tbl = pq.read_table(ES_PATH, columns=feature_cols + [TARGET_COL])
    if n_rows and len(tbl) > n_rows:
        idx = np.random.default_rng(42).choice(len(tbl), n_rows, replace=False)
        idx.sort()
        tbl = tbl.take(idx)
    X = tbl.select(feature_cols).to_pandas().values.astype(np.float32)
    y = tbl.column(TARGET_COL).to_pylist()
    return X, np.array(y, dtype=np.float64), feature_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--n-rows", type=int, default=80000,
                        help="Max earlystop rows to use for SHAP (speed vs accuracy)")
    args = parser.parse_args()

    booster_path = PHASE1_DIR / args.tag / "model1_stage2_lgbm_booster.txt"
    out_dir = PHASE1_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if not booster_path.exists():
        sys.exit(
            f"Booster not found: {booster_path}\n"
            "Run: python scripts/regions/south_america/5_training/model1_phase1_local_train "
            f"--feature-tag {args.tag} --save-model"
        )

    print(f"Loading booster: {booster_path}")
    booster = lgb.Booster(model_file=str(booster_path))

    print(f"Loading earlystop split (max {args.n_rows} rows)...")
    X, y, feature_cols = _load_earlystop(args.n_rows)
    print(f"  {len(X)} rows, {len(feature_cols)} features, {int((y > 0).sum())} positives")

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)  # shape: (n_rows, n_features)

    # --- Plot 1: mean |SHAP| bar ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_n = min(30, len(feature_cols))
    top_idx = order[:top_n]
    top_names = [feature_cols[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.35 + 1))
    ax.barh(range(top_n), top_vals[::-1], color="#2196f3")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} features by SHAP importance — {args.tag}")
    plt.tight_layout()
    p1 = out_dir / "shap_importance.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved → {p1}")

    # --- Plot 2: beeswarm for top 20 features ---
    top20_idx = order[:20]
    top20_names = [feature_cols[i] for i in top20_idx]
    shap20 = shap_values[:, top20_idx]
    X20 = X[:, top20_idx]

    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.flatten()
    for k, (ax, name) in enumerate(zip(axes, top20_names)):
        sv = shap20[:, k]
        xv = X20[:, k]
        pos_mask = y > 0
        ax.scatter(xv[~pos_mask], sv[~pos_mask], s=1, alpha=0.2, c="#aaaaaa", rasterized=True)
        ax.scatter(xv[pos_mask], sv[pos_mask], s=4, alpha=0.7, c="#e53935", rasterized=True)
        ax.axhline(0, lw=0.5, color="k")
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(top20_names):]:
        ax.set_visible(False)
    fig.suptitle(f"SHAP vs feature value — {args.tag}  (red=positive pixels)", fontsize=10)
    plt.tight_layout()
    p2 = out_dir / "shap_beeswarm.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"Saved → {p2}")

    # --- Text summary ---
    print(f"\nTop 15 features by mean |SHAP|:")
    for i, idx in enumerate(order[:15]):
        print(f"  {i+1:2d}. {feature_cols[idx]:<40s}  {mean_abs[idx]:.5f}")

    # --- Positive vs negative SHAP comparison for top 10 ---
    print(f"\nMean SHAP for positive (designated) vs negative pixels — top 10:")
    print(f"  {'Feature':<40s}  {'pos mean SHAP':>14}  {'neg mean SHAP':>14}  {'diff':>10}")
    pos_mask = y > 0
    for idx in order[:10]:
        name = feature_cols[idx]
        sv = shap_values[:, idx]
        pm = sv[pos_mask].mean() if pos_mask.sum() > 0 else float("nan")
        nm = sv[~pos_mask].mean()
        print(f"  {name:<40s}  {pm:>14.5f}  {nm:>14.5f}  {pm-nm:>10.5f}")


if __name__ == "__main__":
    main()
