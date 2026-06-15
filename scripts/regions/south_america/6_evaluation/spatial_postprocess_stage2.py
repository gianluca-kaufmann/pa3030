#!/usr/bin/env python3
"""Spatial post-processing evaluation for Stage 2 LambdaRank.

Loads the most recent full-SA model, rescores the test set while preserving
(row, col) pixel coordinates, then applies 8-neighbour score propagation
and reports Lift@1% + Recall@5% before/after for a sweep of alpha values.

    final_score(pixel) = model_score + alpha * max(model_score of 8 neighbours)

Important: replicates the exact preprocessing of load_stage2_arrays:
  - WDPA_prev == 0 filter (exclude already-protected pixels)
  - Within-group percentile rank normalization of features (model expects this)
  - Sort by (country_id, year) before computing group_sizes

Usage (on Euler from repo root):
    srun python scripts/regions/south_america/6_evaluation/spatial_postprocess_stage2.py
"""

from __future__ import annotations

import gc
import json
import pickle
import sys
import time
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
    _rank_normalize_within_groups,
    resolve_parquet_file,
)
from scripts.regions.shared.evaluation.stage2_metrics import (
    compute_stage2_metrics_from_frame,
)

REGION = "south_america"
MODEL_PREFIX = "model1"
TEST_YEARS = (2017, 2024)
ALPHAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

# Upper bound for SA raster col count — encodes (row, col) as a single int64 key
_MAX_COL = 600_000


def _find_model_pkl() -> Path:
    model_dir = _ROOT / "data" / REGION / "ml" / "models"
    candidates = sorted(
        p for p in model_dir.glob(f"{MODEL_PREFIX}_lgbm_stage2_2*.pkl")
        if "mini_sample" not in p.name and "binary" not in p.name
    )
    if not candidates:
        raise FileNotFoundError(f"No stage2 model pkl in {model_dir}")
    chosen = candidates[-1]
    print(f"Model: {chosen.name}")
    return chosen


def _load_test_data(
    test_path: Path,
    feature_cols: list[str],
    expansion_groups: set[tuple[int, int]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load test rows for expansion groups.

    Replicates load_stage2_arrays preprocessing:
      - WDPA_prev == 0 filter
      - expansion group filter
      - sort by (country_id, year)
      - within-group rank normalization of features

    Returns:
        meta: DataFrame with [row, col, year, country_id, y_true] sorted by group
        X:    float32 feature array, rank-normalized within groups
    """
    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev", "row", "col", "country_id"]
    essential = list(dict.fromkeys(essential))

    meta_frames: list[pd.DataFrame] = []
    feat_arrays: list[np.ndarray] = []

    pf = pq.ParquetFile(test_path)
    t0 = time.time()
    n_batches = 0

    for batch in pf.iter_batches(columns=essential, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        df = df[df[TARGET_COL].notna() & (df["WDPA_prev"] == 0)]
        if df.empty:
            continue
        df = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1])]
        if df.empty:
            continue
        keys = list(zip(df["country_id"].astype(int), df["year"].astype(int)))
        df = df[np.array([k in expansion_groups for k in keys], dtype=bool)]
        if df.empty:
            continue

        meta_frames.append(df[["row", "col", "year", "country_id", TARGET_COL]].copy())
        feat_arrays.append(df[feature_cols].to_numpy(dtype=np.float32))
        del df

        n_batches += 1
        if n_batches % 100 == 0:
            print(f"  {n_batches} batches ({time.time()-t0:.0f}s)...")

    print(f"  {n_batches} batches loaded in {time.time()-t0:.0f}s")

    meta = pd.concat(meta_frames, ignore_index=True)
    del meta_frames
    gc.collect()

    # Sort by (country_id, year) — same as load_stage2_arrays
    order = np.lexsort((meta["year"].to_numpy(), meta["country_id"].to_numpy()))
    meta = meta.iloc[order].reset_index(drop=True)
    meta.rename(columns={TARGET_COL: "y_true"}, inplace=True)

    X_unsorted = np.concatenate(feat_arrays, axis=0)
    del feat_arrays
    gc.collect()
    X = X_unsorted[order]
    del X_unsorted, order
    gc.collect()

    # Within-group rank normalization — model was trained on these, not raw features
    cy_group_sizes = meta.groupby(["country_id", "year"], sort=False).size().to_numpy(dtype=np.int32)
    _rank_normalize_within_groups(X, cy_group_sizes)

    return meta, X


def _propagate_group(
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """8-neighbour score propagation for one (country_id, year) group."""
    keys = rows.astype(np.int64) * _MAX_COL + cols.astype(np.int64)
    score_series = pd.Series(scores, index=pd.Index(keys, dtype=np.int64))

    max_nb = np.full(len(scores), -np.inf)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            n_keys = (rows.astype(np.int64) + dr) * _MAX_COL + (cols.astype(np.int64) + dc)
            n_scores = score_series.reindex(pd.Index(n_keys, dtype=np.int64)).to_numpy(np.float64)
            valid = ~np.isnan(n_scores)
            max_nb = np.where(valid & (n_scores > max_nb), n_scores, max_nb)

    max_nb = np.where(np.isinf(max_nb), 0.0, max_nb)
    return scores + alpha * np.maximum(max_nb, 0.0)


def _apply_propagation(df: pd.DataFrame, raw_scores: np.ndarray, alpha: float) -> np.ndarray:
    """Return propagated scores array (same length/order as df)."""
    if alpha == 0.0:
        return raw_scores.copy()
    out = raw_scores.copy()
    for (cid, yr), grp in df.groupby(["country_id", "year"], sort=False):
        idx = grp.index.values
        out[idx] = _propagate_group(
            grp["row"].to_numpy(np.int32),
            grp["col"].to_numpy(np.int32),
            raw_scores[idx],
            alpha,
        )
    return out


def main() -> None:
    model_pkl = _find_model_pkl()
    with open(model_pkl, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"Features: {len(feature_cols)}, best_iter: {model.best_iteration}")

    test_path = resolve_parquet_file(REGION, "test.parquet")
    print(f"Test: {test_path}")

    print("Finding expansion groups...")
    t0 = time.time()
    expansion_groups = _expansion_groups_from_batches(test_path, REGION, TEST_YEARS)
    print(f"  {len(expansion_groups)} expansion groups ({time.time()-t0:.0f}s)")

    print("Loading & preprocessing test set...")
    meta, X = _load_test_data(test_path, feature_cols, expansion_groups)
    n_pos = int((meta["y_true"] > 0).sum())
    print(f"  {len(meta):,} rows, {n_pos:,} positives")

    print("Scoring...")
    t0 = time.time()
    raw_scores = model.predict(X, num_iteration=model.best_iteration or None).astype(np.float64)
    del X
    gc.collect()
    print(f"  Scored in {time.time()-t0:.0f}s")

    print("\nAlpha sweep:")
    print(f"  {'alpha':>6}  {'Lift@1%':>8}  {'Recall@5%':>10}  {'Recall@1%':>10}")

    results = {}
    for alpha in ALPHAS:
        t0 = time.time()
        scores = _apply_propagation(meta, raw_scores, alpha)
        eval_df = meta[["country_id", "year", "y_true"]].copy()
        eval_df["y_pred_score"] = scores
        m = compute_stage2_metrics_from_frame(eval_df, score_col="y_pred_score", label_col="y_true")
        results[str(alpha)] = m
        print(
            f"  {alpha:>6.1f}  {m['lift_at_1pct_within_groups']:>8.2f}×"
            f"  {m['recall_at_5pct_within_groups']*100:>9.1f}%"
            f"  {m['recall_at_1pct_within_groups']*100:>9.1f}%"
            f"  ({time.time()-t0:.0f}s)"
        )

    out_path = _ROOT / "outputs" / REGION / "results" / "spatial_postprocess_alpha_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": model_pkl.name,
        "n_groups": len(expansion_groups),
        "n_rows": len(meta),
        "n_positives": n_pos,
        "alphas": ALPHAS,
        "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
