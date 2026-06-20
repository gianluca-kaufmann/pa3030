"""Score-then-aggregate: watershed Recall@5% using pixel-level LambdaRank scores.

Instead of training a catchment-level model on mean-aggregated features,
this script:
  1. Trains a fresh pixel-level LambdaRank on mini_sample_v2 (train split)
  2. Scores every test pixel
  3. Aggregates pixel scores to HydroSHEDS L7 catchments (MAX and MEAN)
  4. Ranks catchments by aggregated score
  5. Evaluates pixel-level Recall@5%

Hypothesis: max-pixel-score per catchment correctly identifies the catchment
containing the most "designatable" pixels, without the mean-aggregation
signal dilution that hurt model1_watershed_poc.py.

Usage:
  conda run -n pa3030 python scripts/regions/south_america/5_training/model1_watershed_score_aggregate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIXEL_PATH = PROJECT_ROOT / "data/south_america/mini_sample_v2.parquet"
CAT_TIF    = PROJECT_ROOT / "data/south_america/ready/hydroshed/catchment_id_sa.tif"

LGB_MAX = 9_000  # LightGBM 4.6 per-query row limit

NON_FEAT = frozenset({
    "country_id", "year", "transition_01",
    "WDPA_b1", "WDPA_b2", "WDPA", "WDPA_prev",
    "x", "y", "row", "col", "country_iso3",
    # patch-level features: risk of leakage when used at catchment level
    "log_patch_size_km2", "patch_pa_adjacency_frac",
    "patch_mean_gsn_b2", "patch_designation_lag1",
})

PIXEL_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [90],
    "lambdarank_truncation_level": 100,
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "verbose": -1,
    "n_jobs": -1,
}
N_ROUNDS = 200


def _exp_groups(df: pd.DataFrame) -> set[tuple]:
    pos = df[df["transition_01"] > 0]
    return set(zip(pos["country_id"], pos["year"]))


def _filter_to(df: pd.DataFrame, grps: set[tuple]) -> pd.DataFrame:
    mask = pd.Series(list(zip(df["country_id"], df["year"])), index=df.index).isin(grps)
    return df[mask].sort_values(["country_id", "year"]).reset_index(drop=True)


def _cap_groups(df: pd.DataFrame, max_size: int = LGB_MAX) -> pd.DataFrame:
    """Subsample each (country, year) group to ≤ max_size rows.

    Keeps all positives if they fit, otherwise subsamples positives too.
    Fills remaining budget with negatives.
    """
    parts = []
    for _, grp in df.groupby(["country_id", "year"], sort=False):
        if len(grp) <= max_size:
            parts.append(grp)
            continue
        pos = grp[grp["transition_01"] > 0]
        neg = grp[grp["transition_01"] == 0]
        if len(pos) >= max_size:
            # More positives than budget: subsample positives, keep 1 neg
            pos_s = pos.sample(n=max_size - 1, random_state=42)
            parts.append(pd.concat([pos_s, neg.iloc[:1]]))
        else:
            n_neg = max_size - len(pos)
            neg_s = neg.sample(n=min(len(neg), n_neg), random_state=42)
            parts.append(pd.concat([pos, neg_s]))
    return pd.concat(parts).sort_values(["country_id", "year"]).reset_index(drop=True)


def _pixel_recall_via_catchment(
    df_te: pd.DataFrame,
    eg_te: set[tuple],
    score_col: str = "px_score",
    agg_fn: str = "max",
) -> tuple[float, float, int]:
    catch = (
        df_te.groupby(["country_id", "year", "catchment_id"])
        .agg(cs=(score_col, agg_fn), npp=("transition_01", "sum"))
        .reset_index()
    )
    recalls, weights = [], []
    for (cid, yr), grp in catch.groupby(["country_id", "year"]):
        if (cid, yr) not in eg_te:
            continue
        tot = int(grp["npp"].sum())
        if tot == 0:
            continue
        n_top = max(1, int(np.ceil(0.05 * len(grp))))
        cap = int(grp.nlargest(n_top, "cs")["npp"].sum())
        recalls.append(cap / tot)
        weights.append(tot)
    if not recalls:
        return 0.0, 0.0, 0
    macro = float(np.mean(recalls))
    weighted = float(np.average(recalls, weights=weights))
    return macro, weighted, len(recalls)


def main() -> None:
    for p in (PIXEL_PATH, CAT_TIF):
        if not p.exists():
            sys.exit(f"Not found: {p}")

    print(f"Loading pixel data: {PIXEL_PATH}")
    df = pd.read_parquet(PIXEL_PATH)
    df = df[[c for c in df.columns if not c.startswith("__")]]
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEAT and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  {len(df):,} rows  {len(feature_cols)} features")

    print("Assigning catchment IDs...")
    with rasterio.open(CAT_TIF) as src:
        cid_arr = src.read(1)
    df["catchment_id"] = cid_arr[df["row"].values.astype(int), df["col"].values.astype(int)]
    df = df[df["catchment_id"] > 0].copy()

    df_tr = df[(df["year"] >= 2001) & (df["year"] <= 2013)].copy()
    df_te = df[(df["year"] >= 2017) & (df["year"] <= 2024)].copy()

    eg_tr = _exp_groups(df_tr)
    eg_te = _exp_groups(df_te)
    print(f"  Train expansion groups: {len(eg_tr)}  Test: {len(eg_te)}")

    df_tr = _filter_to(df_tr, eg_tr)
    df_te = _filter_to(df_te, eg_te)

    df_tr_capped = _cap_groups(df_tr)
    tr_sizes = (
        df_tr_capped.groupby(["country_id", "year"], sort=False)
        .size().values.astype(np.int32)
    )
    print(f"  Train (capped): {len(df_tr_capped):,} pixels  "
          f"{len(tr_sizes)} groups  max_group={tr_sizes.max()}")

    ds_tr = lgb.Dataset(
        df_tr_capped[feature_cols].values.astype(np.float32),
        label=df_tr_capped["transition_01"].values.astype(float),
        group=tr_sizes,
        feature_name=feature_cols,
        free_raw_data=False,
    )

    print(f"\nTraining pixel LambdaRank ({N_ROUNDS} rounds)...")
    t0 = time.time()
    model = lgb.train(
        PIXEL_PARAMS, ds_tr, num_boost_round=N_ROUNDS,
        callbacks=[lgb.log_evaluation(50)],
    )
    print(f"Done in {time.time()-t0:.1f}s")

    print("\nScoring test pixels...")
    scores = model.predict(df_te[feature_cols].values.astype(np.float32))
    df_te = df_te.copy()
    df_te["px_score"] = scores

    print("\n" + "="*60)
    print("  SCORE-THEN-AGGREGATE: pixel Recall@5% via catchments")
    print("="*60)

    for agg_fn, label in [("max", "MAX pixel score"), ("mean", "MEAN pixel score")]:
        macro, weighted, n_ev = _pixel_recall_via_catchment(df_te, eg_te, agg_fn=agg_fn)
        print(f"  {label:25s}  macro={macro*100:.1f}%  "
              f"weighted={weighted*100:.1f}%  ({n_ev} events)")

    # Per-event breakdown for MAX
    catch = (
        df_te.groupby(["country_id", "year", "catchment_id"])
        .agg(cs=("px_score", "max"), npp=("transition_01", "sum"))
        .reset_index()
    )
    rows = []
    for (cid, yr), grp in catch.groupby(["country_id", "year"]):
        if (cid, yr) not in eg_te:
            continue
        tot = int(grp["npp"].sum())
        if tot == 0:
            continue
        n_top = max(1, int(np.ceil(0.05 * len(grp))))
        cap = int(grp.nlargest(n_top, "cs")["npp"].sum())
        rows.append({"cid": cid, "yr": yr, "n_catch": len(grp), "n_top": n_top,
                     "tot": tot, "cap": cap, "recall": cap / tot})
    rows.sort(key=lambda r: r["recall"], reverse=True)

    print(f"\nPer-event breakdown (MAX):")
    print(f"  {'cid':>4} {'yr':>4} {'n_catch':>7} {'n_top5%':>7} {'tot_px':>8} {'recall':>8}")
    for r in rows:
        print(f"  {r['cid']:>4} {r['yr']:>4} {r['n_catch']:>7} {r['n_top']:>7} "
              f"{r['tot']:>8,} {r['recall']*100:>7.1f}%")

    macro_max = np.mean([r["recall"] for r in rows])
    gate = ("✅ PASS (≥70%)" if macro_max >= 0.70 else
            "⚠️  PARTIAL (50-70%)" if macro_max >= 0.50 else
            "❌ FAIL (<50%)")
    print(f"\nB4 gate (MAX): {gate}")


if __name__ == "__main__":
    main()
