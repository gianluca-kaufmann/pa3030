#!/usr/bin/env python3
"""Spatial post-processing — Tier 1 multi-step mean diffusion.

Loads the locked H6+H1b+H5 booster (20260617_011621), scores the SA test set,
then sweeps iterative 8-neighbour mean diffusion:

    s_k[i] += alpha × decay^k × mean(s_{k-1}[neighbours of i])

Reports Lift@1% + Recall@5% for every combination of
  n_steps ∈ {0, 1, 5, 10, 20}  ×  alpha ∈ {0.1, 0.3, 0.5, 1.0, 2.0}
with decay=0.95.  n_steps=0 is the unmodified baseline (sanity check).

The H6+H1b+H5 model was trained with STAGE2_NORMALIZE_WITHIN_GROUPS=0 (H5),
so raw features are used for scoring here — no within-group rank normalization.

Usage (on Euler from repo root):
    srun python scripts/regions/south_america/6_evaluation/spatial_postprocess_stage2.py
"""

from __future__ import annotations

import gc
import json
import os
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
    resolve_parquet_file,
)

REGION = "south_america"
MODEL_PREFIX = "model1"
TEST_YEARS = (2017, 2024)

# Pin the locked baseline booster. If not found, fall back to the most recent pkl.
BOOSTER_TIMESTAMP = "20260617_011621"  # H6+H1b+H5

# H5: model was trained with no within-group rank normalization.
# Raw feature values are used for both training and inference.
NORMALIZE_WITHIN_GROUPS = False

# Diffusion sweep parameters (Tier 1)
N_STEPS_LIST = [0, 1, 5, 10, 20]   # 0 = unmodified baseline
ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0]
DECAY = 0.95

# Encode (row, col) as a single int64 key: key = row * _MAX_COL + col
_MAX_COL = 600_000


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_model_pkl() -> Path:
    model_dir = _ROOT / "data" / REGION / "ml" / "models"
    pinned = model_dir / f"{MODEL_PREFIX}_lgbm_stage2_{BOOSTER_TIMESTAMP}.pkl"
    if pinned.exists():
        print(f"Model (pinned): {pinned.name}")
        return pinned
    # Fallback: most recent pkl that isn't a mini-sample or binary variant
    candidates = sorted(
        p for p in model_dir.glob(f"{MODEL_PREFIX}_lgbm_stage2_2*.pkl")
        if "mini_sample" not in p.name and "binary" not in p.name
    )
    if not candidates:
        raise FileNotFoundError(f"No stage2 model pkl in {model_dir}")
    chosen = candidates[-1]
    print(f"Model (fallback, most recent): {chosen.name}")
    return chosen


# ---------------------------------------------------------------------------
# Test data loading (no rank normalization — H5 model)
# ---------------------------------------------------------------------------

def _load_test_data(
    test_path: Path,
    feature_cols: list[str],
    expansion_groups: set[tuple[int, int]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Batch-load test rows for expansion groups.

    Returns:
        meta: DataFrame [row, col, year, country_id, y_true] sorted by (country_id, year)
        X:    float32 feature matrix, raw (no rank normalization — H5)
    """
    needed = feature_cols + [TARGET_COL, "year", "WDPA_prev", "row", "col", "country_id"]
    needed = list(dict.fromkeys(needed))

    meta_frames: list[pd.DataFrame] = []
    feat_parts: list[np.ndarray] = []

    pf = pq.ParquetFile(test_path)
    t0 = time.time()
    n_batches = 0

    for batch in pf.iter_batches(columns=needed, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        df = df[df[TARGET_COL].notna() & (df["WDPA_prev"] == 0)]
        if df.empty:
            continue
        df = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1])]
        if df.empty:
            continue
        keys = list(zip(df["country_id"].astype(int), df["year"].astype(int)))
        mask = np.array([k in expansion_groups for k in keys], dtype=bool)
        df = df[mask]
        if df.empty:
            continue

        meta_frames.append(df[["row", "col", "year", "country_id", TARGET_COL]].copy())
        feat_parts.append(df[feature_cols].to_numpy(dtype=np.float32))
        del df

        n_batches += 1
        if n_batches % 100 == 0:
            print(f"  {n_batches} batches ({time.time()-t0:.0f}s)...")

    print(f"  {n_batches} batches loaded in {time.time()-t0:.0f}s")

    meta = pd.concat(meta_frames, ignore_index=True)
    del meta_frames
    gc.collect()

    # Sort by (country_id, year) — consistent with training
    order = np.lexsort((meta["year"].to_numpy(), meta["country_id"].to_numpy()))
    meta = meta.iloc[order].reset_index(drop=True)
    meta.rename(columns={TARGET_COL: "y_true"}, inplace=True)

    X = np.concatenate(feat_parts, axis=0)[order]
    del feat_parts
    gc.collect()

    # H5: no within-group rank normalization. Model was trained on raw features.
    # (Contrast with non-H5 models where _rank_normalize_within_groups was applied here.)

    return meta, X


# ---------------------------------------------------------------------------
# Neighbour precomputation (once per group)
# ---------------------------------------------------------------------------

def _precompute_nb(
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute valid 8-neighbour (src, dst) index pairs for pixels in a group.

    Uses pd.Series key lookup to find which neighbours exist within the group
    (pixels outside the group — already protected, different country, etc. — are
    ignored). This is correct: diffusion only spreads within the expansion group.

    Returns:
        all_src:   source pixel indices (into the group array)
        all_dst:   destination pixel indices (into the group array)
        nb_count:  number of valid in-group neighbours per pixel
    """
    n = len(rows)
    keys = rows.astype(np.int64) * _MAX_COL + cols.astype(np.int64)
    key_series = pd.Series(np.arange(n, dtype=np.int64),
                           index=pd.Index(keys, dtype=np.int64))

    srcs: list[np.ndarray] = []
    dsts: list[np.ndarray] = []

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb_keys = (rows.astype(np.int64) + dr) * _MAX_COL + (cols.astype(np.int64) + dc)
            nb_idx = key_series.reindex(pd.Index(nb_keys, dtype=np.int64)).to_numpy(np.float64)
            valid = ~np.isnan(nb_idx)
            if valid.any():
                dsts.append(np.where(valid)[0].astype(np.int64))
                srcs.append(nb_idx[valid].astype(np.int64))

    if srcs:
        all_src = np.concatenate(srcs)
        all_dst = np.concatenate(dsts)
    else:
        all_src = np.empty(0, dtype=np.int64)
        all_dst = np.empty(0, dtype=np.int64)

    nb_count = np.bincount(all_dst.astype(np.intp), minlength=n).astype(np.float64)
    return all_src, all_dst, nb_count


# ---------------------------------------------------------------------------
# Iterative diffusion
# ---------------------------------------------------------------------------

def _diffuse(
    scores: np.ndarray,
    n_steps: int,
    alpha: float,
    decay: float,
    all_src: np.ndarray,
    all_dst: np.ndarray,
    nb_count: np.ndarray,
) -> np.ndarray:
    """Iterative 8-neighbour mean diffusion.

    For each step k from 1 to n_steps:
        mean_nb[i] = mean score of pixel i's valid in-group neighbours (prev step)
        s[i]      += alpha × decay^k × mean_nb[i]

    The increment shrinks geometrically with decay, so influence from distant
    pixels falls off with distance (step count). Seeds — pixels already ranked
    highly by the model — propagate outward across the PA polygon.
    """
    if n_steps == 0 or len(all_src) == 0:
        return scores.copy()

    n = len(scores)
    scores = scores.copy().astype(np.float64)
    valid = nb_count > 0
    safe_count = np.where(valid, nb_count, 1.0)

    for step in range(1, n_steps + 1):
        nb_sum = np.bincount(
            all_dst.astype(np.intp),
            weights=scores[all_src],
            minlength=n,
        )
        mean_nb = np.where(valid, nb_sum / safe_count, 0.0)
        scores += (alpha * decay ** step) * mean_nb

    return scores


# ---------------------------------------------------------------------------
# Per-group metrics helpers
# ---------------------------------------------------------------------------

def _group_prec_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    k1_pct: float = 1.0,
    k5_pct: float = 5.0,
) -> tuple[float, float] | None:
    """Returns (precision@1%, recall@5%) for a single group, or None if no positives."""
    n = len(y_true)
    n_pos = int((y_true > 0).sum())
    if n_pos == 0:
        return None

    k1 = max(1, int(np.ceil(n * k1_pct / 100.0)))
    k5 = max(1, int(np.ceil(n * k5_pct / 100.0)))

    # argpartition is O(n) and sufficient — we only need membership, not order
    top1_idx = np.argpartition(-scores, min(k1, n - 1))[:k1]
    top5_idx = np.argpartition(-scores, min(k5, n - 1))[:k5]

    prec1 = float((y_true[top1_idx] > 0).sum()) / k1
    rec5 = float((y_true[top5_idx] > 0).sum()) / n_pos
    return prec1, rec5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model_pkl = _find_model_pkl()
    with open(model_pkl, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"Features: {len(feature_cols)}, best_iter: {model.best_iteration}")
    print(f"Normalize within groups: {NORMALIZE_WITHIN_GROUPS} (H5 = off)")

    test_path = resolve_parquet_file(REGION, "test.parquet")
    print(f"Test: {test_path}")

    print("Finding expansion groups...")
    t0 = time.time()
    expansion_groups = _expansion_groups_from_batches(test_path, REGION, TEST_YEARS)
    print(f"  {len(expansion_groups)} expansion groups ({time.time()-t0:.0f}s)")

    print("Loading test set (no rank normalization)...")
    meta, X = _load_test_data(test_path, feature_cols, expansion_groups)
    n_pos_total = int((meta["y_true"] > 0).sum())
    global_baseline = float(n_pos_total) / len(meta)
    print(f"  {len(meta):,} rows, {n_pos_total:,} positives  baseline_rate={global_baseline:.5f}")

    print("Scoring...")
    t0 = time.time()
    raw_scores = model.predict(X, num_iteration=model.best_iteration or None).astype(np.float64)
    del X
    gc.collect()
    print(f"  Scored {len(raw_scores):,} rows in {time.time()-t0:.0f}s")

    # -----------------------------------------------------------------------
    # Group-by-group sweep
    # For each group: precompute neighbours once, then apply all combos.
    # Only prec@1% and recall@5% are accumulated (what we need for the paper metrics).
    # -----------------------------------------------------------------------

    all_combos = [(ns, a) for ns in N_STEPS_LIST for a in ALPHAS]
    prec1_acc: dict[tuple, list[float]] = {c: [] for c in all_combos}
    rec5_acc: dict[tuple, list[float]] = {c: [] for c in all_combos}

    n_groups_done = 0
    t_loop = time.time()

    for (cid, yr), grp in meta.groupby(["country_id", "year"], sort=True):
        idx = grp.index.to_numpy()
        yt = meta["y_true"].to_numpy()[idx]

        if (yt > 0).sum() == 0:
            continue

        rows_g = grp["row"].to_numpy(np.int32)
        cols_g = grp["col"].to_numpy(np.int32)
        raw_g = raw_scores[idx]

        all_src, all_dst, nb_count = _precompute_nb(rows_g, cols_g)

        for (n_steps, alpha) in all_combos:
            gs = _diffuse(raw_g, n_steps, alpha, DECAY, all_src, all_dst, nb_count)
            result = _group_prec_recall(yt, gs)
            if result is not None:
                prec1_acc[(n_steps, alpha)].append(result[0])
                rec5_acc[(n_steps, alpha)].append(result[1])

        n_groups_done += 1
        if n_groups_done % 10 == 0:
            elapsed = time.time() - t_loop
            print(f"  {n_groups_done}/{len(expansion_groups)} groups  ({elapsed:.0f}s)")

    print(f"All {n_groups_done} groups processed in {time.time()-t_loop:.0f}s")

    # -----------------------------------------------------------------------
    # Compute macro metrics and print table
    # -----------------------------------------------------------------------

    results: dict[str, dict] = {}
    print(f"\n{'n_steps':>8}  {'alpha':>6}  {'Lift@1%':>9}  {'Recall@5%':>10}")
    print("-" * 42)

    for n_steps in N_STEPS_LIST:
        for alpha in ALPHAS:
            combo = (n_steps, alpha)
            p1_list = prec1_acc[combo]
            r5_list = rec5_acc[combo]
            if not p1_list:
                continue
            macro_prec1 = float(np.mean(p1_list))
            macro_rec5 = float(np.mean(r5_list))
            lift1 = macro_prec1 / global_baseline if global_baseline > 0 else 0.0

            key = f"steps={n_steps}_alpha={alpha}"
            results[key] = {
                "n_steps": n_steps,
                "alpha": alpha,
                "decay": DECAY,
                "lift_at_1pct_within_groups": lift1,
                "recall_at_5pct_within_groups": macro_rec5,
                "macro_precision_at_1pct": macro_prec1,
                "n_groups_evaluated": len(p1_list),
            }
            marker = " ◄ best?" if n_steps > 0 and lift1 > 6.462 else ""
            print(
                f"{n_steps:>8}  {alpha:>6.1f}  {lift1:>8.2f}×  "
                f"{macro_rec5*100:>9.1f}%{marker}"
            )
        if n_steps < N_STEPS_LIST[-1]:
            print()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    out_path = (
        _ROOT / "outputs" / REGION / "results" / "spatial_diffusion_sweep.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "booster": model_pkl.name,
        "normalize_within_groups": NORMALIZE_WITHIN_GROUPS,
        "n_groups": n_groups_done,
        "n_rows": int(len(meta)),
        "n_positives": n_pos_total,
        "global_baseline_rate": global_baseline,
        "decay": DECAY,
        "n_steps_list": N_STEPS_LIST,
        "alphas": ALPHAS,
        "baseline_lift_at_1pct": 6.462,
        "baseline_recall_at_5pct": 0.1572,
        "results": results,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
