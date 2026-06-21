#!/usr/bin/env python3
"""B4 (E3): Stage 2 LambdaRank at HydroSHEDS L7 catchment level — proof of concept.

Instead of ranking individual 1km pixels, this ranks HydroSHEDS L7 catchments
(~10-100 km²) within each (country_id, year) expansion event. PA designations are
contiguous polygons that span 1-5 catchments; predicting the right catchments
automatically recalls all their pixels.

Evaluation:
  For each test event: take top-5% predicted catchments → expand to their pixels
  → compute pixel-level Recall@5% (n_positive_pixels in top catchments /
  total n_positive_pixels in event).

Decision gate:
  Pixel Recall@5% ≥ 0.70 → Track B validated; scale to full SA (B5)
  Pixel Recall@5% ≥ 0.50 → partial fix; investigate; continue B4
  Pixel Recall@5% < 0.50 → catchment unit insufficient → activate Track C

Prerequisites:
  data/south_america/watershed_sample.parquet
  (run build_watershed_sample.py first)

Usage:
  python scripts/regions/south_america/5_training/model1_LGBM_stage2_watershed.py

Env vars:
  WS_NEG_RATIO          max negatives per positive in training  [default: 20]
  WS_SEED               random seed                             [default: 42]
  WS_TEST_RATIO         fraction of events for test             [default: 0.20]
  WS_VAL_RATIO          fraction of train pool for early-stop   [default: 0.10]
  WS_MIN_POS_PIXELS     min positive pixels for event to count  [default: 200]
  WS_NUM_BOOST_ROUND    max LightGBM rounds                     [default: 1000]
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRATCH = os.environ.get("SCRATCH", "")
_WS_PATH_ENV = os.environ.get("WS_PARQUET", "")

if _WS_PATH_ENV:
    WATERSHED_PATH = Path(_WS_PATH_ENV)
elif _SCRATCH:
    WATERSHED_PATH = Path(_SCRATCH) / "data/south_america/ml/watershed/watershed_full_sa.parquet"
else:
    WATERSHED_PATH = _ROOT / "data/south_america/watershed_mini_sample.parquet"

MODEL_DIR = _ROOT / "data/south_america/ml/models"
OUT_DIR   = _ROOT / "outputs/south_america/results/ml_models"

SEED            = int(os.environ.get("WS_SEED",              "42"))
TEST_RATIO      = float(os.environ.get("WS_TEST_RATIO",      "0.20"))
VAL_RATIO       = float(os.environ.get("WS_VAL_RATIO",       "0.10"))
MIN_POS_PIXELS  = int(os.environ.get("WS_MIN_POS_PIXELS",    "200"))
NEG_RATIO       = int(os.environ.get("WS_NEG_RATIO",         "20"))
NUM_BOOST_ROUND = int(os.environ.get("WS_NUM_BOOST_ROUND",   "1000"))
ES_PATIENCE     = int(os.environ.get("WS_ES_PATIENCE",       "50"))

# Columns that are NOT features.
# n_total_pixels and WDPA_prev_frac are intentionally allowed as features
# (catchment area proxy and PA overlap fraction).
NON_FEATURE = frozenset({
    "country_id", "year", "catchment_id",
    "n_pos_pixels", "n_positive_pixels",   # both naming conventions
    "catchment_is_positive",
    "transition_01", "transition_01_win5",
    "WDPA_b1", "WDPA_b2", "WDPA_prev", "WDPA",
    "x", "y", "row", "col",
    "x_mean", "y_mean", "row_mean", "col_mean",
    "country_iso3", "country_iso3_mean",
})


# ---------------------------------------------------------------------------
# Event split
# ---------------------------------------------------------------------------

def _split_events(
    event_counts: dict[tuple[int, int], int],
    rng: np.random.Generator,
) -> tuple[set, set, set]:
    """Country-stratified 80/20 event split with val anchors."""
    meaningful = sorted(event_counts)
    by_country: dict[int, list] = {}
    for ev in meaningful:
        by_country.setdefault(ev[0], []).append(ev)

    test_events: set  = set()
    train_pool: list  = []

    for cid, events in sorted(by_country.items()):
        evs = list(events)
        rng.shuffle(evs)
        n = len(evs)
        n_test = max(1, round(n * TEST_RATIO)) if n >= 2 else 0
        test_events.update(evs[:n_test])
        train_pool.extend(evs[n_test:])

    # Guarantee ≥1 large event in val for a reliable early-stop signal
    rng.shuffle(train_pool)
    anchor_pos = 5000  # min positive pixels to be a val anchor
    anchors    = [e for e in train_pool if event_counts[e] >= anchor_pos]
    smalls     = [e for e in train_pool if event_counts[e] <  anchor_pos]
    rng.shuffle(anchors)
    rng.shuffle(smalls)

    n_val       = max(1, round(len(train_pool) * VAL_RATIO))
    forced_val  = set(anchors[:min(1, len(anchors))])
    n_val_extra = max(0, n_val - len(forced_val))
    val_events  = forced_val | set(smalls[:n_val_extra])
    train_events = set(anchors[min(1, len(anchors)):]) | set(smalls[n_val_extra:])

    return train_events, val_events, test_events


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are valid training features."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in NON_FEATURE]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _filter_events(df: pd.DataFrame, events: set[tuple[int, int]]) -> pd.DataFrame:
    """Fast event filter using merge instead of row-wise apply."""
    ev_df = pd.DataFrame(list(events), columns=["country_id", "year"])
    return df.merge(ev_df, on=["country_id", "year"], how="inner")


def _build_dataset(
    df: pd.DataFrame,
    events: set[tuple[int, int]],
    feature_cols: list[str],
    neg_ratio: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y_label, group_sizes, country_ids, years) for a set of events.
    y_label = catchment_is_positive (binary).
    group_sizes: sorted by (country_id, year), matching LightGBM group ordering.
    """
    subset = _filter_events(df, events).copy()

    if subset.empty:
        raise ValueError(f"No catchments found for {len(events)} events")

    if neg_ratio is not None:
        # Subsample negatives per event to keep dataset size manageable
        keep_parts: list[pd.DataFrame] = []
        for (cid, yr), g in subset.groupby(["country_id", "year"]):
            pos = g[g["catchment_is_positive"] == 1]
            neg = g[g["catchment_is_positive"] == 0]
            n_keep_neg = min(len(neg), len(pos) * neg_ratio)
            if n_keep_neg < len(neg):
                neg = neg.sample(n=n_keep_neg, random_state=int(rng.integers(1e9)))
            keep_parts.append(pd.concat([pos, neg], ignore_index=True))
        subset = pd.concat(keep_parts, ignore_index=True)

    # Sort by (country_id, year) for LightGBM group ordering
    subset = subset.sort_values(["country_id", "year", "catchment_id"]).reset_index(drop=True)

    grp = subset.groupby(["country_id", "year"], sort=False)
    group_sizes = grp.size().values.astype(np.int32)
    cids  = grp["country_id"].first().values.astype(np.int32)
    years = grp["year"].first().values.astype(np.int32)

    X = subset[feature_cols].to_numpy(dtype=np.float32)
    y = subset["catchment_is_positive"].to_numpy(dtype=np.float32)

    return X, y, group_sizes, cids, years


# ---------------------------------------------------------------------------
# Recall@5% callbacks
# ---------------------------------------------------------------------------

class _WatershedEarlyStop:
    """Custom early stopping based on pixel-level Recall@5% via top-5% catchments."""

    def __init__(
        self,
        X_val: np.ndarray,
        val_df: pd.DataFrame,
        patience: int = 50,
    ) -> None:
        self._X_val    = X_val
        self._val_df   = val_df
        self._patience = patience
        self._best_recall  = -1.0
        self._best_iter    = 0
        self._rounds_no_improve = 0
        self._best_model = None

    def __call__(self, env) -> None:
        model = env.model
        scores = model.predict(self._X_val, num_iteration=env.iteration)
        recall = _pixel_recall_at_5pct_catchments(self._val_df, scores)

        if recall > self._best_recall + 1e-6:
            self._best_recall  = recall
            self._best_iter    = env.iteration
            self._rounds_no_improve = 0
        else:
            self._rounds_no_improve += 1

        if env.iteration % 50 == 0:
            print(f"  [ws_es] iter={env.iteration}  val_pixel_recall@5%={recall:.4f}"
                  f"  best={self._best_recall:.4f}@{self._best_iter}"
                  f"  patience={self._rounds_no_improve}/{self._patience}")

        if self._rounds_no_improve >= self._patience:
            model.best_iteration = self._best_iter
            raise lgb.callback.EarlyStopException(self._best_iter, [(0, "ws_recall5", self._best_recall, True)])

    @property
    def best_iteration(self) -> int:
        return self._best_iter


# ---------------------------------------------------------------------------
# Evaluation metric: pixel-level Recall@5% via top-5% catchments
# ---------------------------------------------------------------------------

def _pixel_recall_at_5pct_catchments(
    df: pd.DataFrame,
    scores: np.ndarray,
    k_pct: float = 5.0,
) -> float:
    """
    For each (country_id, year) event:
      - Rank catchments by score descending
      - Take top ceil(k_pct/100 * n_catchments)
      - Pixel Recall = sum(n_positive_pixels in top-k) / sum(n_positive_pixels in event)
    Returns macro-average Recall across events with any positive pixels.
    """
    df = df.copy()
    df["score"] = scores

    recalls: list[float] = []
    for (cid, yr), g in df.groupby(["country_id", "year"]):
        total_pos_px = int(g["n_positive_pixels"].sum())
        if total_pos_px == 0:
            continue
        n_catch = len(g)
        k = max(1, math.ceil(k_pct / 100.0 * n_catch))
        top_k = g.nlargest(k, "score")
        recalled_px = int(top_k["n_positive_pixels"].sum())
        recalls.append(recalled_px / total_pos_px)

    return float(np.mean(recalls)) if recalls else 0.0


def _per_event_metrics(
    df: pd.DataFrame,
    scores: np.ndarray,
    k_pct: float = 5.0,
) -> list[dict]:
    df = df.copy()
    df["score"] = scores

    results: list[dict] = []
    for (cid, yr), g in df.groupby(["country_id", "year"]):
        total_pos_px = int(g["n_positive_pixels"].sum())
        n_pos_catch  = int(g["catchment_is_positive"].sum())
        n_catch      = len(g)
        k = max(1, math.ceil(k_pct / 100.0 * n_catch))

        top_k = g.nlargest(k, "score")
        recalled_px    = int(top_k["n_positive_pixels"].sum())
        pos_in_top_k   = int((top_k["catchment_is_positive"] > 0).sum())
        catch_recall5  = pos_in_top_k / max(n_pos_catch, 1)

        results.append({
            "country_id":        int(cid),
            "year":              int(yr),
            "n_catchments":      n_catch,
            "n_pos_catchments":  n_pos_catch,
            "n_positive_pixels": total_pos_px,
            "k_top_catchments":  k,
            "pixel_recall_at_5pct":    recalled_px / max(total_pos_px, 1),
            "catchment_recall_at_5pct": catch_recall5,
        })
    return results


def _aggregate_metrics(per_event: list[dict]) -> dict:
    if not per_event:
        return {}
    px_recalls   = [e["pixel_recall_at_5pct"]    for e in per_event]
    catch_recalls = [e["catchment_recall_at_5pct"] for e in per_event]
    pos_px        = [e["n_positive_pixels"]        for e in per_event]
    total_pos     = sum(pos_px)

    def weighted(vals: list, ws: list) -> float:
        s = sum(ws)
        return sum(v * w for v, w in zip(vals, ws)) / s if s > 0 else 0.0

    return {
        "macro_pixel_recall_at_5pct":    float(np.mean(px_recalls)),
        "weighted_pixel_recall_at_5pct": weighted(px_recalls, pos_px),
        "macro_catchment_recall_at_5pct": float(np.mean(catch_recalls)),
        "n_events":     len(per_event),
        "total_pos_px": total_pos,
        "pixel_recall_min": float(min(px_recalls)),
        "pixel_recall_max": float(max(px_recalls)),
        "pixel_recall_std": float(np.std(px_recalls)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not WATERSHED_PATH.exists():
        sys.exit(
            f"Watershed sample not found: {WATERSHED_PATH}\n"
            "Run build_watershed_sample.py first."
        )

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    rng = np.random.default_rng(SEED)

    print(f"Loading watershed sample: {WATERSHED_PATH}")
    df = pd.read_parquet(WATERSHED_PATH)

    # Support both column naming conventions:
    #   old builder: catchment_is_positive, n_positive_pixels
    #   new builder: transition_01, n_pos_pixels
    if "catchment_is_positive" not in df.columns and "transition_01" in df.columns:
        df["catchment_is_positive"] = df["transition_01"]
    if "n_positive_pixels" not in df.columns and "n_pos_pixels" in df.columns:
        df["n_positive_pixels"] = df["n_pos_pixels"]

    print(f"  {len(df):,} catchment-events  |  "
          f"positive catchments: {int(df['catchment_is_positive'].sum()):,}  |  "
          f"events: {df.groupby(['country_id','year']).ngroups}")

    feature_cols = _get_feature_cols(df)
    print(f"  Feature columns: {len(feature_cols)}")
    for sf in ("n_total_pixels", "WDPA_prev_frac", "elev_std"):
        status = "✓" if sf in feature_cols else "✗"
        print(f"    {status} {sf}")

    # ── Enumerate meaningful events ──
    ev_stats = df[df["catchment_is_positive"] == 1].groupby(["country_id", "year"])[
        "n_positive_pixels"
    ].sum()
    meaningful = {(int(k[0]), int(k[1])): int(v)
                  for k, v in ev_stats.items() if v >= MIN_POS_PIXELS}
    print(f"\n  Meaningful events (≥{MIN_POS_PIXELS} pos px): {len(meaningful)}")

    train_events, val_events, test_events = _split_events(meaningful, rng)
    print(
        f"  Event split (seed={SEED}): "
        f"train={len(train_events)} | val={len(val_events)} | test={len(test_events)}"
    )
    print(f"  Train events: {sorted(train_events)[:10]}...")
    print(f"  Test  events: {sorted(test_events)}")

    # ── Build datasets ──
    print(f"\nBuilding train dataset (neg_ratio={NEG_RATIO})...")
    X_tr, y_tr, g_tr, cids_tr, yrs_tr = _build_dataset(
        df, train_events, feature_cols, NEG_RATIO, rng
    )
    n_pos_tr = int((y_tr > 0).sum())
    print(f"  {len(y_tr):,} catchments | {n_pos_tr:,} positive | {len(g_tr)} groups")

    # inv_sqrt_npos weights (H1b) — same as pixel model
    group_weights = np.ones(len(g_tr), dtype=np.float32)
    sample_weights = np.ones(len(y_tr), dtype=np.float32)
    offset = 0
    for i, gs in enumerate(g_tr):
        g_pos = int(y_tr[offset:offset + gs].sum())
        w = 1.0 / math.sqrt(max(g_pos, 1))
        sample_weights[offset:offset + gs] = w
        offset += gs

    print(f"\nBuilding val dataset (all pixels, no neg subsampling)...")
    X_val, y_val, g_val, cids_val, yrs_val = _build_dataset(
        df, val_events, feature_cols, None, rng
    )
    val_df_indexed = (
        _filter_events(df, val_events)
        .sort_values(["country_id", "year", "catchment_id"])
        .reset_index(drop=True)
    )
    print(f"  {len(y_val):,} catchments | {int((y_val > 0).sum()):,} positive | {len(g_val)} groups")

    # ── LightGBM ──
    lgb_params = {
        "boosting_type":      "gbdt",
        "objective":          "lambdarank",
        "metric":             "ndcg",
        "lambdarank_truncation_level": 20,
        "eval_at":            [max(1, 500 // 20)],  # rough proxy
        "num_leaves":         63,
        "max_depth":          -1,
        "learning_rate":      0.05,
        "min_child_samples":  10,
        "subsample":          0.8,
        "colsample_bytree":   0.7,
        "reg_alpha":          0.1,
        "reg_lambda":         1.0,
        "verbose":            -1,
        "num_threads":        num_threads,
    }

    train_set = lgb.Dataset(X_tr, label=y_tr, group=g_tr, weight=sample_weights)
    val_set   = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_set)

    es_cb = _WatershedEarlyStop(X_val, val_df_indexed, patience=ES_PATIENCE)

    print(f"\nTraining LambdaRank (watershed, {len(y_tr):,} catchments, "
          f"{len(g_tr)} groups, {len(feature_cols)} features)...")
    t0 = time.time()
    try:
        model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.log_evaluation(50), es_cb],
        )
    except lgb.callback.EarlyStopException:
        model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=es_cb.best_iteration,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.log_evaluation(50)],
        )

    best_iter = getattr(model, "best_iteration", None) or es_cb.best_iteration
    print(f"  Training done: {time.time()-t0:.1f}s  best_iter={best_iter}")

    del X_tr, y_tr, train_set, X_val, y_val, val_set
    gc.collect()

    # ── Val evaluation ──
    val_df_eval = val_df_indexed.copy()
    X_val_eval, *_ = _build_dataset(df, val_events, feature_cols, None, rng)
    val_scores = model.predict(X_val_eval, num_iteration=best_iter)
    val_recall = _pixel_recall_at_5pct_catchments(val_df_eval, val_scores)
    print(f"\nVal pixel Recall@5%: {val_recall:.4f}")

    # ── Test evaluation ──
    print(f"\nBuilding test dataset ({len(test_events)} events)...")
    X_te, y_te, g_te, *_ = _build_dataset(df, test_events, feature_cols, None, rng)
    test_df = (
        _filter_events(df, test_events)
        .sort_values(["country_id", "year", "catchment_id"])
        .reset_index(drop=True)
    )
    test_scores   = model.predict(X_te, num_iteration=best_iter)
    test_per_event = _per_event_metrics(test_df, test_scores)
    test_agg       = _aggregate_metrics(test_per_event)

    macro_recall = test_agg["macro_pixel_recall_at_5pct"]
    print(f"\n{'='*60}")
    print(f"B4 Watershed Test Results ({len(test_events)} events)")
    print("=" * 60)
    print(f"  Macro pixel Recall@5%:    {macro_recall*100:.1f}%")
    print(f"  Weighted pixel Recall@5%: {test_agg['weighted_pixel_recall_at_5pct']*100:.1f}%")
    print(f"  Macro catchment R@5%:     {test_agg['macro_catchment_recall_at_5pct']*100:.1f}%")
    print(f"  min/max/std:  "
          f"{test_agg['pixel_recall_min']*100:.1f}% / "
          f"{test_agg['pixel_recall_max']*100:.1f}% / "
          f"{test_agg['pixel_recall_std']*100:.1f}%")
    print()
    print("  Per-event breakdown (sorted by pixel recall):")
    for ev in sorted(test_per_event, key=lambda x: x["pixel_recall_at_5pct"], reverse=True):
        print(
            f"    cid={ev['country_id']} yr={ev['year']} "
            f"n_catch={ev['n_catchments']:>5}  "
            f"n_pos_px={ev['n_positive_pixels']:>7,}  "
            f"pixel_R@5%={ev['pixel_recall_at_5pct']:.3f}  "
            f"catch_R@5%={ev['catchment_recall_at_5pct']:.3f}"
        )
    print()

    # Decision gate
    if macro_recall >= 0.70:
        gate = "PASS ≥70% → Track B validated; scale to full SA (B5)"
    elif macro_recall >= 0.50:
        gate = "PARTIAL ≥50% → investigate per-event failures; continue B4"
    else:
        gate = "FAIL <50% → catchment unit insufficient → activate Track C"
    print(f"  Decision gate: {gate}")
    print("=" * 60)

    # ── Save ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"model1_lgbm_stage2_watershed_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "best_iteration": best_iter,
            "experiment": "B4_watershed_poc",
        }, f)
    print(f"\nModel saved: {model_path}")

    result = {
        "metadata": {
            "timestamp":       timestamp,
            "experiment":      "B4_watershed_poc",
            "ranking_unit":    "HydroSHEDS_L7_catchment",
            "seed":            SEED,
            "neg_ratio_train": NEG_RATIO,
            "min_pos_pixels":  MIN_POS_PIXELS,
            "n_meaningful_events": len(meaningful),
            "n_train_events":  len(train_events),
            "n_val_events":    len(val_events),
            "n_test_events":   len(test_events),
            "train_events":    sorted(list(train_events)),
            "val_events":      sorted(list(val_events)),
            "test_events":     sorted(list(test_events)),
            "n_features":      len(feature_cols),
            "best_iteration":  best_iter,
            "model_path":      str(model_path),
        },
        "val_pixel_recall_at_5pct": val_recall,
        "test_aggregate":   test_agg,
        "test_per_event":   test_per_event,
        "decision_gate":    gate,
    }

    metrics_path = OUT_DIR / f"stage2_watershed_metrics_{timestamp}.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    print(f"Metrics saved: {metrics_path}")
    print(f"\nAdd to ROADMAP.md Experiment History:")
    print(f"  B4 watershed PoC | pixel R@5%={macro_recall*100:.1f}%  | {gate[:25]}...")

    if macro_recall < 0.50:
        sys.exit(1)


if __name__ == "__main__":
    main()
