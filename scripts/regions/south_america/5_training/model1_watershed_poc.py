"""Watershed proof-of-concept: LambdaRank on HydroSHEDS L7 catchments.

Trains a LambdaRank model on catchment-level aggregated features and
evaluates PIXEL-LEVEL Recall@5%: for each test expansion event, take
the top-5% catchments by predicted score and count how many positive
pixels they contain.

This is the B4 experiment from ROADMAP.md. The key structural hypothesis:
  PA designations span 1-5 catchments; top-5% of ~100-5000 catchments
  per country-year = 5-250 slots, enough to capture the entire event.

Input:
  data/south_america/watershed_mini_sample.parquet

Year splits (matching pixel model):
  train:     2001-2013
  earlystop: 2014-2016  (used for early stopping signal)
  test:      2017-2024  (evaluation)

Usage:
  python scripts/regions/south_america/5_training/model1_watershed_poc.py

Decision gates (ROADMAP B4):
  Recall@5% >= 70% → Track B validated; proceed to B5 (full SA)
  Recall@5% 50-70% → partial fix; investigate failing events
  Recall@5% < 50%  → catchment unit also insufficient; activate Track C
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb

SAMPLE_PATH = PROJECT_ROOT / "data/south_america/watershed_mini_sample.parquet"

TRAIN_YEARS = (2001, 2013)
ES_YEARS    = (2014, 2016)
TEST_YEARS  = (2017, 2024)

# Columns that are identifiers / labels — never fed as features
NON_FEATURE_COLS = frozenset({
    "country_id", "year", "catchment_id",
    "transition_01", "n_pos_pixels", "n_total_pixels", "WDPA_prev_frac",
    "WDPA_b1", "WDPA_b2", "WDPA", "WDPA_prev",
    "x", "y", "row", "col", "country_iso3",
})

LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],               # NDCG@10 catchments (monitoring only)
    "lambdarank_truncation_level": 50,
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "verbose": -1,
    "n_jobs": -1,
}
N_ROUNDS = 500
EARLY_STOP_ROUNDS = 50


def _filter_years(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return df[(df["year"] >= years[0]) & (df["year"] <= years[1])].copy()


def _expansion_groups(df: pd.DataFrame) -> set[tuple]:
    pos = df[df["transition_01"] > 0]
    return set(zip(pos["country_id"], pos["year"]))


def _filter_to_expansion(df: pd.DataFrame, groups: set[tuple]) -> pd.DataFrame:
    mask = pd.Series(
        list(zip(df["country_id"], df["year"])), index=df.index
    ).isin(groups)
    return df[mask].copy()


def _sort_and_group_sizes(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.sort_values(["country_id", "year"]).reset_index(drop=True)
    sizes = df.groupby(["country_id", "year"], sort=False).size().values
    return df, sizes.astype(np.int32)


def _build_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_sizes: np.ndarray,
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    X = df[feature_cols].values.astype(np.float32)
    y = df["transition_01"].values.astype(np.float32)
    ds = lgb.Dataset(X, label=y, group=group_sizes, feature_name=feature_cols,
                     free_raw_data=False, reference=reference)
    return ds


def compute_pixel_recall_at_5pct(
    df_test: pd.DataFrame,
    scores: np.ndarray,
    expansion_groups: set[tuple],
) -> tuple[float, float, list[dict]]:
    """Pixel-level Recall@5% per expansion event.

    For each (country_id, year) test event:
      - Top-5% catchments by predicted score
      - Sum n_pos_pixels in those catchments
      - Recall = pos_pixels_captured / total_pos_pixels_in_event

    Returns (macro_recall, weighted_recall, per_event_list).
    """
    df_test = df_test.copy()
    df_test["_score"] = scores

    per_event = []
    for (cid, yr), grp in df_test.groupby(["country_id", "year"]):
        if (cid, yr) not in expansion_groups:
            continue
        total_pos_px = int(grp["n_pos_pixels"].sum())
        if total_pos_px == 0:
            continue

        n_top = max(1, int(np.ceil(0.05 * len(grp))))
        top_grp = grp.nlargest(n_top, "_score")
        captured_pos_px = int(top_grp["n_pos_pixels"].sum())
        recall = captured_pos_px / total_pos_px

        n_pos_catchments = int((grp["transition_01"] > 0).sum())
        per_event.append({
            "country_id": int(cid),
            "year": int(yr),
            "n_catchments": len(grp),
            "n_pos_catchments": n_pos_catchments,
            "n_top": n_top,
            "total_pos_pixels": total_pos_px,
            "captured_pos_pixels": captured_pos_px,
            "recall_at_5pct": recall,
        })

    if not per_event:
        return 0.0, 0.0, []

    recalls = np.array([e["recall_at_5pct"] for e in per_event])
    weights = np.array([e["total_pos_pixels"] for e in per_event], dtype=float)
    macro = float(recalls.mean())
    weighted = float(np.average(recalls, weights=weights))
    return macro, weighted, per_event


def main() -> None:
    if not SAMPLE_PATH.exists():
        sys.exit(
            f"Not found: {SAMPLE_PATH}\n"
            "Run build_watershed_mini_sample.py first."
        )

    print(f"Loading: {SAMPLE_PATH}")
    df = pd.read_parquet(SAMPLE_PATH)
    df = df[[c for c in df.columns if not c.startswith("__")]]
    print(f"  {len(df):,} catchment-year rows  {len(df.columns)} columns")

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  Feature columns: {len(feature_cols)}")

    df_train_raw = _filter_years(df, TRAIN_YEARS)
    df_es_raw    = _filter_years(df, ES_YEARS)
    df_test_raw  = _filter_years(df, TEST_YEARS)

    expansion_groups = (
        _expansion_groups(df_train_raw) | _expansion_groups(df_es_raw)
    )
    test_expansion_groups = _expansion_groups(df_test_raw)
    print(f"  Train+ES expansion groups: {len(expansion_groups)}")
    print(f"  Test expansion groups:     {len(test_expansion_groups)}")

    df_train, train_sizes = _sort_and_group_sizes(
        _filter_to_expansion(df_train_raw, expansion_groups)
    )
    df_es, es_sizes = _sort_and_group_sizes(
        _filter_to_expansion(df_es_raw, expansion_groups)
    )
    df_test, test_sizes = _sort_and_group_sizes(
        _filter_to_expansion(df_test_raw, test_expansion_groups)
    )

    print(f"\nDataset sizes:")
    print(f"  train:     {len(df_train):,} catchments  {len(train_sizes)} groups")
    print(f"  earlystop: {len(df_es):,} catchments  {len(es_sizes)} groups")
    print(f"  test:      {len(df_test):,} catchments  {len(test_sizes)} groups")
    print(f"  pos rate train: {df_train['transition_01'].mean()*100:.2f}%")
    print(f"  pos rate test:  {df_test['transition_01'].mean()*100:.2f}%")

    ds_train = _build_dataset(df_train, feature_cols, train_sizes)
    ds_es    = _build_dataset(df_es, feature_cols, es_sizes, reference=ds_train)

    print(f"\nTraining LambdaRank  {N_ROUNDS} rounds  lr={LGB_PARAMS['learning_rate']}  "
          f"num_leaves={LGB_PARAMS['num_leaves']}")
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=True),
    ]
    model = lgb.train(
        LGB_PARAMS,
        ds_train,
        num_boost_round=N_ROUNDS,
        valid_sets=[ds_es],
        valid_names=["earlystop"],
        callbacks=callbacks,
    )
    print(f"Best iteration: {model.best_iteration}")

    # Evaluate pixel Recall@5% on test set
    X_test = df_test[feature_cols].values.astype(np.float32)
    scores = model.predict(X_test, num_iteration=model.best_iteration or None)

    macro, weighted, per_event = compute_pixel_recall_at_5pct(
        df_test, scores, test_expansion_groups
    )

    print(f"\n{'='*60}")
    print(f"  WATERSHED PoC — pixel Recall@5%  ({len(per_event)} test events)")
    print(f"{'='*60}")
    print(f"  Macro Recall@5%:    {macro*100:.1f}%")
    print(f"  Weighted Recall@5%: {weighted*100:.1f}%")
    print(f"{'='*60}")

    # Per-event breakdown
    per_event.sort(key=lambda e: e["recall_at_5pct"], reverse=True)
    print(f"\nPer-event breakdown (sorted by recall):")
    print(f"  {'country':>7}  {'year':>4}  {'n_catch':>7}  "
          f"{'n_pos_c':>7}  {'n_top5%':>7}  {'pos_px':>8}  {'recall':>8}")
    for e in per_event:
        print(f"  {e['country_id']:>7}  {e['year']:>4}  {e['n_catchments']:>7}  "
              f"{e['n_pos_catchments']:>7}  {e['n_top']:>7}  "
              f"{e['total_pos_pixels']:>8,}  {e['recall_at_5pct']*100:>7.1f}%")

    gate = "✅ PASS (≥70%)" if macro >= 0.70 else (
           "⚠️  PARTIAL (50-70%)" if macro >= 0.50 else
           "❌ FAIL (<50%) — consider Track C")
    print(f"\nB4 decision gate: {gate}")

    # Save model for inspection
    out_dir = PROJECT_ROOT / "outputs/south_america/results/watershed"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "watershed_poc_booster.txt"
    model.save_model(str(model_path))
    print(f"Booster saved → {model_path}")


if __name__ == "__main__":
    main()
