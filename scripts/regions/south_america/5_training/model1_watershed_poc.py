"""Watershed proof-of-concept: LambdaRank on HydroSHEDS L7 catchments.

Trains a LambdaRank model on catchment-level aggregated features and
evaluates PIXEL-LEVEL Recall@5%: for each test expansion event, take
the top-5% catchments by predicted score and count how many positive
pixels they contain.

This is the B4 experiment from ROADMAP.md. The key structural hypothesis:
  PA designations span 1-5 catchments; top-5% of ~100-5000 catchments
  per country-year = 5-250 slots, enough to capture the entire event.

Primary evaluation: cross-event 80/20 split (stratified by country).
  Events are split randomly — train and test events come from different
  country-years across all 2001-2024. This is the correct validation
  for a spatial choice model (analogous to SDM cross-validation).

Secondary evaluation (--temporal flag): temporal holdout 2001-2013 →
  2017-2024. Reported as robustness check; suffers from era-based
  concept drift (see ROADMAP.md § Stage 2: The Critical Design Decision).

Input:
  data/south_america/watershed_mini_sample.parquet

Structural features used (in addition to all pixel-aggregated features):
  n_total_pixels  — catchment area proxy
  WDPA_prev_frac  — fraction of catchment already protected (PA adjacency)
  elev_std        — elevation std within catchment (terrain ruggedness)

Usage:
  # Cross-event (primary, default):
  python scripts/regions/south_america/5_training/model1_watershed_poc.py

  # Temporal split (diagnostic only):
  python scripts/regions/south_america/5_training/model1_watershed_poc.py --temporal

Decision gates (ROADMAP B4):
  Recall@5% >= 70% → Track B validated; proceed to B5 (full SA)
  Recall@5% 50-70% → partial fix; improve features then scale
  Recall@5% < 50%  → catchment unit also insufficient; activate Track C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb

SAMPLE_PATH = PROJECT_ROOT / "data/south_america/watershed_mini_sample.parquet"

# Temporal split years (secondary evaluation only)
TRAIN_YEARS = (2001, 2013)
ES_YEARS    = (2014, 2016)
TEST_YEARS  = (2017, 2024)

# Minimum positive catchments per event to include in the split
MIN_POS_CATCHMENTS = 3

# Cross-event split fractions
CE_TEST_FRAC  = 0.20   # 20% of each country's events go to test
CE_VAL_FRAC   = 0.15   # 15% of remaining events go to early-stop val
CE_SEED       = 42

# Columns that are identifiers / labels — never fed as features.
# n_total_pixels and WDPA_prev_frac are intentionally NOT excluded:
# they are catchment-structural signals (area proxy, PA overlap fraction).
NON_FEATURE_COLS = frozenset({
    "country_id", "year", "catchment_id",
    "transition_01", "n_pos_pixels",
    "WDPA_b1", "WDPA_b2", "WDPA", "WDPA_prev",
    "x", "y", "row", "col", "country_iso3",
})

LGB_PARAMS = {
    "boosting_type":              "gbdt",
    "objective":                  "lambdarank",
    "metric":                     "ndcg",
    "eval_at":                    [10],
    "lambdarank_truncation_level": 50,
    "num_leaves":                 64,
    "learning_rate":              0.05,
    "feature_fraction":           0.8,
    "bagging_fraction":           0.8,
    "bagging_freq":               5,
    "min_child_samples":          5,
    "verbose":                    -1,
    "n_jobs":                     -1,
}
N_ROUNDS          = 500
EARLY_STOP_ROUNDS = 50


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _filter_years(df: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return df[(df["year"] >= years[0]) & (df["year"] <= years[1])].copy()


def _expansion_groups(df: pd.DataFrame, min_pos: int = 0) -> set[tuple]:
    pos = df[df["transition_01"] > 0]
    counts = pos.groupby(["country_id", "year"])["transition_01"].sum()
    return set(counts[counts >= min_pos].index.tolist())


def _filter_to_groups(df: pd.DataFrame, groups: set[tuple]) -> pd.DataFrame:
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
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["transition_01"].values.astype(np.float32)
    return lgb.Dataset(
        X, label=y, group=group_sizes,
        feature_name=feature_cols, free_raw_data=False, reference=reference,
    )


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
    """
    df_test = df_test.copy()
    df_test["_score"] = scores

    per_event: list[dict] = []
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
            "country_id":       int(cid),
            "year":             int(yr),
            "n_catchments":     len(grp),
            "n_pos_catchments": n_pos_catchments,
            "n_top":            n_top,
            "total_pos_pixels": total_pos_px,
            "captured_pos_pixels": captured_pos_px,
            "recall_at_5pct":   recall,
        })

    if not per_event:
        return 0.0, 0.0, []

    recalls = np.array([e["recall_at_5pct"] for e in per_event])
    weights = np.array([e["total_pos_pixels"] for e in per_event], dtype=float)
    macro   = float(recalls.mean())
    weighted = float(np.average(recalls, weights=weights))
    return macro, weighted, per_event


def _print_results(per_event: list[dict], macro: float, weighted: float, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  ({len(per_event)} test events)")
    print(f"{'='*60}")
    print(f"  Macro Recall@5%:    {macro*100:.1f}%")
    print(f"  Weighted Recall@5%: {weighted*100:.1f}%")
    print(f"{'='*60}")

    per_event_s = sorted(per_event, key=lambda e: e["recall_at_5pct"], reverse=True)
    print(f"\nPer-event breakdown (sorted by recall):")
    print(f"  {'country':>7}  {'year':>4}  {'n_catch':>7}  "
          f"{'n_pos_c':>7}  {'n_top5%':>7}  {'pos_px':>8}  {'recall':>8}")
    for e in per_event_s:
        print(f"  {e['country_id']:>7}  {e['year']:>4}  {e['n_catchments']:>7}  "
              f"{e['n_pos_catchments']:>7}  {e['n_top']:>7}  "
              f"{e['total_pos_pixels']:>8,}  {e['recall_at_5pct']*100:>7.1f}%")

    gate = ("✅ PASS (≥70%)" if macro >= 0.70 else
            "⚠️  PARTIAL (50-70%)" if macro >= 0.50 else
            "❌ FAIL (<50%) — consider Track C")
    print(f"\nB4 decision gate: {gate}")


# ---------------------------------------------------------------------------
# Cross-event split
# ---------------------------------------------------------------------------

def _split_events_cross(
    df: pd.DataFrame,
    min_pos_catchments: int = MIN_POS_CATCHMENTS,
    test_frac: float = CE_TEST_FRAC,
    val_frac: float = CE_VAL_FRAC,
    seed: int = CE_SEED,
) -> tuple[set[tuple], set[tuple], set[tuple]]:
    """Random stratified split of expansion events into train / val / test.

    Stratified by country: each country contributes test_frac of its events
    to the test set, val_frac of the remainder to early-stop val, the rest
    to training. This ensures representation of diverse geographies in both
    train and test.
    """
    rng = np.random.default_rng(seed)
    qualifying = _expansion_groups(df, min_pos=min_pos_catchments)

    by_country: dict[int, list[tuple]] = {}
    for cid, yr in qualifying:
        by_country.setdefault(cid, []).append((cid, yr))

    train_events: set[tuple] = set()
    val_events:   set[tuple] = set()
    test_events:  set[tuple] = set()

    for cid, events in sorted(by_country.items()):
        n = len(events)
        if n == 0:
            continue
        idx = rng.permutation(n)
        n_test = max(1, round(test_frac * n))
        remaining = n - n_test
        n_val = max(0, round(val_frac * remaining))
        if n_test >= n:
            n_test, n_val = n, 0

        for i in idx[:n_test]:
            test_events.add(events[i])
        for i in idx[n_test:n_test + n_val]:
            val_events.add(events[i])
        for i in idx[n_test + n_val:]:
            train_events.add(events[i])

    return train_events, val_events, test_events


def run_cross_event(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Primary evaluation: random 80/20 cross-event split stratified by country."""
    train_events, val_events, test_events = _split_events_cross(df)

    print(f"\n  Cross-event split  (seed={CE_SEED}, "
          f"test_frac={CE_TEST_FRAC}, val_frac={CE_VAL_FRAC})")
    print(f"  train events: {len(train_events)}   val events: {len(val_events)}   "
          f"test events: {len(test_events)}")

    all_train = train_events | val_events  # full candidate pool
    df_train, tr_sz = _sort_and_group_sizes(_filter_to_groups(df, train_events))
    df_val,   va_sz = _sort_and_group_sizes(_filter_to_groups(df, val_events))
    df_test,  te_sz = _sort_and_group_sizes(_filter_to_groups(df, test_events))

    print(f"\nDataset sizes:")
    print(f"  train:     {len(df_train):,} catchments  {len(tr_sz)} groups")
    print(f"  val (ES):  {len(df_val):,} catchments  {len(va_sz)} groups")
    print(f"  test:      {len(df_test):,} catchments  {len(te_sz)} groups")
    print(f"  pos rate train: {df_train['transition_01'].mean()*100:.2f}%")
    print(f"  pos rate test:  {df_test['transition_01'].mean()*100:.2f}%")

    if len(val_events) == 0 or len(df_val) == 0:
        print("  WARN: no val events — using train as val (early stopping unreliable)")
        df_val, va_sz = df_train, tr_sz

    ds_train = _build_dataset(df_train, feature_cols, tr_sz)
    ds_val   = _build_dataset(df_val, feature_cols, va_sz, reference=ds_train)

    print(f"\nTraining LambdaRank  {N_ROUNDS} rounds  lr={LGB_PARAMS['learning_rate']}  "
          f"num_leaves={LGB_PARAMS['num_leaves']}")
    model = lgb.train(
        LGB_PARAMS, ds_train,
        num_boost_round=N_ROUNDS,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=True),
        ],
    )
    print(f"Best iteration: {model.best_iteration}")

    scores = model.predict(
        df_test[feature_cols].fillna(0).values.astype(np.float32),
        num_iteration=model.best_iteration or None,
    )
    macro, weighted, per_event = compute_pixel_recall_at_5pct(
        df_test, scores, test_events
    )
    _print_results(per_event, macro, weighted, "WATERSHED PoC — Cross-event (primary)")

    _save_model(model, "watershed_poc_cross_event.txt")


# ---------------------------------------------------------------------------
# Temporal split (secondary diagnostic)
# ---------------------------------------------------------------------------

def run_temporal(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Temporal split: train 2001-2013, test 2017-2024. Diagnostic only."""
    df_train_raw = _filter_years(df, TRAIN_YEARS)
    df_es_raw    = _filter_years(df, ES_YEARS)
    df_test_raw  = _filter_years(df, TEST_YEARS)

    train_groups = _expansion_groups(df_train_raw) | _expansion_groups(df_es_raw)
    test_groups  = _expansion_groups(df_test_raw)

    df_train, tr_sz = _sort_and_group_sizes(_filter_to_groups(df_train_raw, train_groups))
    df_es,    es_sz = _sort_and_group_sizes(_filter_to_groups(df_es_raw, train_groups))
    df_test,  te_sz = _sort_and_group_sizes(_filter_to_groups(df_test_raw, test_groups))

    print(f"\nTemporal split: train {TRAIN_YEARS}, ES {ES_YEARS}, test {TEST_YEARS}")
    print(f"\nDataset sizes:")
    print(f"  train:     {len(df_train):,} catchments  {len(tr_sz)} groups")
    print(f"  earlystop: {len(df_es):,} catchments  {len(es_sz)} groups")
    print(f"  test:      {len(df_test):,} catchments  {len(te_sz)} groups")

    ds_train = _build_dataset(df_train, feature_cols, tr_sz)
    ds_es    = _build_dataset(df_es, feature_cols, es_sz, reference=ds_train)

    print(f"\nTraining LambdaRank  {N_ROUNDS} rounds  lr={LGB_PARAMS['learning_rate']}")
    model = lgb.train(
        LGB_PARAMS, ds_train,
        num_boost_round=N_ROUNDS,
        valid_sets=[ds_es],
        valid_names=["earlystop"],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=True),
        ],
    )
    print(f"Best iteration: {model.best_iteration}")

    scores = model.predict(
        df_test[feature_cols].fillna(0).values.astype(np.float32),
        num_iteration=model.best_iteration or None,
    )
    macro, weighted, per_event = compute_pixel_recall_at_5pct(
        df_test, scores, test_groups
    )
    _print_results(per_event, macro, weighted, "WATERSHED PoC — Temporal split (diagnostic)")

    _save_model(model, "watershed_poc_temporal.txt")


def _save_model(model: lgb.Booster, filename: str) -> None:
    out_dir = PROJECT_ROOT / "outputs/south_america/results/watershed"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    model.save_model(str(path))
    print(f"Booster saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B4: Watershed PoC LambdaRank")
    parser.add_argument(
        "--temporal", action="store_true",
        help="Run temporal split (2001-2013 → 2017-2024) as diagnostic. "
             "Default is cross-event split (primary evaluation)."
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both cross-event (primary) and temporal (diagnostic) evaluations."
    )
    args = parser.parse_args()

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
    # Report which structural features are present
    for sf in ("n_total_pixels", "WDPA_prev_frac", "elev_std"):
        status = "✓" if sf in feature_cols else "✗"
        print(f"    {status} {sf}")

    if args.temporal and not args.both:
        run_temporal(df, feature_cols)
    elif args.both:
        run_cross_event(df, feature_cols)
        run_temporal(df, feature_cols)
    else:
        run_cross_event(df, feature_cols)


if __name__ == "__main__":
    main()
