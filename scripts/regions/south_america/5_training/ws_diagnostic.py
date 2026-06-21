#!/usr/bin/env python3
"""Diagnostic: is the 20.6% CE.W result a training failure or a fundamental ceiling?

For each of the 29 CE.W test events, computes:
  1. Random-chance baseline (expected pixel recall if catchments selected at random)
  2. Naive baseline: rank by dist_wdpa_min (proximity to existing PAs)
  3. Model recall (from saved CE.W model)
  4. Oracle upper bound (best possible recall in top-5% budget)
  5. Feature separability: mean feature values pos vs neg for zero-recall events
  6. Training coverage: how many train events per country

Also prints LightGBM feature importances (gain).

Usage:
  python scripts/regions/south_america/5_training/ws_diagnostic.py
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRATCH = os.environ.get("SCRATCH", "")

WS_PATH   = (Path(_SCRATCH) / "data/south_america/ml/watershed/watershed_full_sa.parquet"
             if _SCRATCH else _ROOT / "data/south_america/watershed_mini_sample.parquet")
MODEL_PKL = _ROOT / "data/south_america/ml/models/model1_lgbm_stage2_watershed_20260621_133404.pkl"
METRICS_JSON = _ROOT / "outputs/south_america/results/ml_models/stage2_watershed_metrics_20260621_133404.json"

K_PCT = 5.0
MIN_POS_PIXELS = 200


def pixel_recall_topk(g: pd.DataFrame, score_col: str, k: int) -> float:
    total_pos_px = int(g["n_positive_pixels"].sum())
    if total_pos_px == 0:
        return 0.0
    top = g.nlargest(k, score_col)
    return int(top["n_positive_pixels"].sum()) / total_pos_px


def random_expected_recall(n_catch: int, k: int, pos_px_vec: np.ndarray) -> float:
    """Expected pixel recall when k catchments are drawn uniformly at random."""
    total_pos_px = pos_px_vec.sum()
    if total_pos_px == 0 or k == 0:
        return 0.0
    # E[recall] = (k / n_catch) when all catchments equally likely
    # But pixel recall weights by n_positive_pixels per catchment.
    # E[recalled px] = sum_i(pos_px_i * P(catchment_i selected)) = sum_i(pos_px_i * k/n_catch)
    # E[recall] = sum_i(pos_px_i * k/n_catch) / total_pos_px = k/n_catch
    return k / n_catch


def oracle_upper_bound(g: pd.DataFrame, k: int) -> float:
    """Best possible pixel recall: sort positive catchments by pixel count, take as many as fit."""
    total_pos_px = int(g["n_positive_pixels"].sum())
    if total_pos_px == 0 or k == 0:
        return 0.0
    pos_catchments = g[g["catchment_is_positive"] == 1].nlargest(k, "n_positive_pixels")
    return int(pos_catchments["n_positive_pixels"].sum()) / total_pos_px


def main() -> None:
    print(f"Loading watershed parquet: {WS_PATH}")
    df = pd.read_parquet(WS_PATH)

    # Normalise column names
    if "catchment_is_positive" not in df.columns and "transition_01" in df.columns:
        df["catchment_is_positive"] = df["transition_01"]
    if "n_positive_pixels" not in df.columns and "n_pos_pixels" in df.columns:
        df["n_positive_pixels"] = df["n_pos_pixels"]

    print(f"  {len(df):,} catchment-rows | {df['catchment_is_positive'].sum():,} positive")

    # Load model
    print(f"\nLoading model: {MODEL_PKL}")
    with open(MODEL_PKL, "rb") as f:
        pkg = pickle.load(f)
    model: lgb.Booster = pkg["model"]
    feature_cols: list[str] = pkg["feature_cols"]
    best_iter: int = pkg["best_iteration"]
    print(f"  best_iteration={best_iter}  |  {len(feature_cols)} features")

    # Load test events from metrics JSON
    with open(METRICS_JSON) as f:
        metrics = json.load(f)
    train_events = {tuple(e) for e in metrics["metadata"]["train_events"]}
    val_events   = {tuple(e) for e in metrics["metadata"]["val_events"]}
    test_events  = {tuple(e) for e in metrics["metadata"]["test_events"]}
    print(f"\nTest events: {len(test_events)}  |  train: {len(train_events)}  |  val: {len(val_events)}")

    # Meaningful events for coverage check
    ev_stats = df[df["catchment_is_positive"] == 1].groupby(["country_id", "year"])["n_positive_pixels"].sum()
    meaningful_all = {(int(k[0]), int(k[1])): int(v) for k, v in ev_stats.items() if v >= MIN_POS_PIXELS}

    # Subset to test + val events for scoring
    score_events = test_events | val_events
    ev_df = pd.DataFrame(list(score_events), columns=["country_id", "year"])
    sub = df.merge(ev_df, on=["country_id", "year"], how="inner").copy()
    sub = sub.sort_values(["country_id", "year", "catchment_id"]).reset_index(drop=True)

    # Model scores
    X = sub[feature_cols].to_numpy(dtype=np.float32)
    sub["model_score"] = model.predict(X, num_iteration=best_iter)

    # Naive baselines: available distance-to-PA features
    dist_cols = [c for c in df.columns if "dist_wdpa" in c.lower()]
    print(f"\nDistance-to-WDPA columns available: {dist_cols}")
    # prefer _min (closest pixel in catchment to any PA)
    naive_col = next((c for c in dist_cols if "min" in c), dist_cols[0] if dist_cols else None)
    if naive_col:
        # Closer = more likely → negate for ranking
        sub["naive_score"] = -sub[naive_col].fillna(sub[naive_col].max())
        print(f"  Using naive column: {naive_col} (negated, lower dist → higher rank)")
    else:
        sub["naive_score"] = 0.0
        print("  WARNING: no dist_wdpa column found for naive baseline")

    # ── Per-event diagnostics ──
    print("\n" + "="*100)
    print(f"{'Event':<18} {'n_cat':>6} {'n_pos_cat':>9} {'n_pos_px':>10} {'budget':>7} "
          f"{'random%':>8} {'oracle%':>8} {'naive%':>8} {'model%':>8} {'delta':>8}  note")
    print("-"*100)

    zero_recall_events = []
    results = []
    for (cid, yr), g in sub.groupby(["country_id", "year"]):
        if (cid, yr) not in test_events:
            continue
        n_catch = len(g)
        n_pos_catch = int(g["catchment_is_positive"].sum())
        total_pos_px = int(g["n_positive_pixels"].sum())
        k = max(1, math.ceil(K_PCT / 100.0 * n_catch))

        rand_r   = random_expected_recall(n_catch, k, g["n_positive_pixels"].values) * 100
        oracle_r = oracle_upper_bound(g, k) * 100
        naive_r  = pixel_recall_topk(g, "naive_score", k) * 100 if naive_col else 0.0
        model_r  = pixel_recall_topk(g, "model_score", k) * 100

        delta = model_r - naive_r

        # Is this a "reachable" event? If n_pos_catch <= k, a perfect catchment classifier gets 100%
        reachable = n_pos_catch <= k

        note = ""
        if model_r == 0.0 and oracle_r > 0.0:
            note = "TOTAL FAIL"
            zero_recall_events.append((cid, yr))
        elif model_r < rand_r:
            note = "WORSE THAN RANDOM"
        elif model_r < naive_r - 5.0:
            note = "WORSE THAN NAIVE"

        if reachable:
            note += " [all pos fit in budget]" if note else "[all pos fit in budget]"

        print(f"  cid={cid} yr={yr:<4}  {n_catch:>6} {n_pos_catch:>9} {total_pos_px:>10,} {k:>7} "
              f"{rand_r:>7.1f}% {oracle_r:>7.1f}% {naive_r:>7.1f}% {model_r:>7.1f}% {delta:>+7.1f}%  {note}")

        results.append(dict(cid=cid, yr=yr, n_catch=n_catch, n_pos_catch=n_pos_catch,
                            total_pos_px=total_pos_px, k=k,
                            rand_pct=rand_r, oracle_pct=oracle_r,
                            naive_pct=naive_r, model_pct=model_r))

    print("="*100)

    # Aggregate
    if results:
        macro_model  = np.mean([r["model_pct"]  for r in results])
        macro_naive  = np.mean([r["naive_pct"]  for r in results])
        macro_rand   = np.mean([r["rand_pct"]   for r in results])
        macro_oracle = np.mean([r["oracle_pct"] for r in results])
        print(f"\nMacro Recall@5%:  random={macro_rand:.1f}%  naive={macro_naive:.1f}%  "
              f"model={macro_model:.1f}%  oracle={macro_oracle:.1f}%")
        print(f"  Model vs random: {macro_model-macro_rand:+.1f}pp")
        print(f"  Model vs naive:  {macro_model-macro_naive:+.1f}pp")
        print(f"  Oracle ceiling:  {macro_oracle:.1f}%  (limited by budget vs n_pos_catchments)")

    # ── Feature separability for zero-recall events ──
    if zero_recall_events:
        print(f"\n{'='*100}")
        print(f"Feature separability for ZERO-RECALL test events ({len(zero_recall_events)} events)")
        print("(Do positive catchments look different from negatives in these events?)")
        print("="*100)

        top_feats = ["dist_wdpa_min", "dist_wdpa_mean", "gsn_b2_max", "elevation_mean",
                     "agb_tonne_ha_max", "dist_road_min", "population_density_mean",
                     "n_total_pixels", "WDPA_prev_frac", "elev_std"]
        avail_feats = [f for f in top_feats if f in df.columns]
        # Add model top features (gain)
        fi = pd.Series(model.feature_importance(importance_type="gain"),
                       index=feature_cols).sort_values(ascending=False)
        top_model_feats = fi.head(10).index.tolist()
        all_feats = list(dict.fromkeys(avail_feats + top_model_feats))

        for (cid, yr) in zero_recall_events:
            ev_data = df[(df["country_id"] == cid) & (df["year"] == yr)].copy()
            pos = ev_data[ev_data["catchment_is_positive"] == 1]
            neg = ev_data[ev_data["catchment_is_positive"] == 0]
            n_pos_catch = len(pos)
            k = max(1, math.ceil(K_PCT / 100.0 * len(ev_data)))
            print(f"\n  cid={cid} yr={yr}  |  {len(ev_data)} catchments  |  "
                  f"{n_pos_catch} positive  |  k={k} budget")
            print(f"  {'Feature':<35} {'pos_mean':>12} {'neg_mean':>12} {'ratio':>8}  separable?")
            for feat in all_feats:
                if feat not in ev_data.columns:
                    continue
                pos_m = pos[feat].mean()
                neg_m = neg[feat].mean()
                ratio = pos_m / neg_m if abs(neg_m) > 1e-6 else float("nan")
                sep = "YES" if abs(ratio - 1.0) > 0.25 else "no"
                print(f"    {feat:<35} {pos_m:>12.3f} {neg_m:>12.3f} {ratio:>8.3f}  {sep}")

    # ── Training coverage per country ──
    print(f"\n{'='*80}")
    print("Training coverage per country (test events vs training events)")
    print("="*80)
    countries_in_test = sorted({cid for (cid, yr) in test_events})
    for cid in countries_in_test:
        n_train = sum(1 for (c, y) in train_events if c == cid)
        n_val   = sum(1 for (c, y) in val_events   if c == cid)
        n_test  = sum(1 for (c, y) in test_events  if c == cid)
        test_recalls = [r["model_pct"] for r in results if r["cid"] == cid]
        mean_r = np.mean(test_recalls) if test_recalls else 0.0
        print(f"  country {cid:>2}:  train_events={n_train:>3}  val={n_val}  test={n_test}  "
              f"mean_model_R@5%={mean_r:.1f}%")

    # ── Feature importances ──
    print(f"\n{'='*80}")
    print("Top-20 feature importances (gain) from CE.W model")
    print("="*80)
    for feat, imp in fi.head(20).items():
        print(f"  {feat:<40} {imp:>12.1f}")

    # ── Val set diagnosis ──
    print(f"\n{'='*80}")
    print("Val set event summary (what drove early stopping)")
    print("="*80)
    for (cid, yr), g in sub.groupby(["country_id", "year"]):
        if (cid, yr) not in val_events:
            continue
        n_catch = len(g)
        n_pos_catch = int(g["catchment_is_positive"].sum())
        total_pos_px = int(g["n_positive_pixels"].sum())
        k = max(1, math.ceil(K_PCT / 100.0 * n_catch))
        r = pixel_recall_topk(g, "model_score", k) * 100
        oracle_r = oracle_upper_bound(g, k) * 100
        print(f"  cid={cid} yr={yr:<4}  n_catch={n_catch:>5}  n_pos_catch={n_pos_catch:>4}  "
              f"n_pos_px={total_pos_px:>7,}  k={k:>4}  model_R={r:.1f}%  oracle={oracle_r:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
