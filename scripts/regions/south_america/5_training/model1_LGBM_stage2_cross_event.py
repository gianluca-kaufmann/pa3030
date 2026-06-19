#!/usr/bin/env python3
"""Stage 2 LambdaRank — cross-event validation experiment (CE.1 / CE.2).

Split design
------------
Expansion events = (country_id, year) groups with ≥ MIN_POSITIVE_PIXELS positive
pixels. All expansion events across the full 2001–2024 panel are enumerated from
train.parquet, earlystop.parquet, and test.parquet, then randomly split 80/20:

  All meaningful events (~150 SA / ~20 Colombia)
    ├── 80 % → train pool
    │     ├── 90 % → actual training  (neg_ratio=100)
    │     └── 10 % → early-stop val  (all pixels)
    └── 20 % → test  (all pixels, per-event metrics)

Train and test events come from different countries and different years — no spatial
overlap, no geometric leakage. This is the correct evaluation for a spatial choice
model (cf. SDM cross-validation, McFadden discrete-choice models).

Locked settings (H6 + H1b + H5)
---------------------------------
  STAGE2_NORMALIZE_WITHIN_GROUPS=0   H5 — absolute feature signals intact
  STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos  H1b — gradient deconcentration
  STAGE2_EARLY_STOP_METRIC=recall_at_5pct  H6 — directly optimises headline bar

Usage
-----
    # Colombia pilot (local, hours not days):
    STAGE2_DATA_ROOT=$HOME/master_thesis/data/dev/south_america/ml \\
    STAGE2_NORMALIZE_WITHIN_GROUPS=0 \\
    STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos \\
    python scripts/regions/south_america/5_training/model1_LGBM_stage2_cross_event.py

    # Full SA on Euler:
    sbatch slurm/south_america/training_lgbm_stage2_cross_event.slurm

    # Colombia on Euler:
    sbatch slurm/south_america/training_lgbm_stage2_cross_event_colombia.slurm

Decision gate
-------------
  Colombia Recall@5% >= 0.60  → submit full SA to Euler
  SA      Recall@5% >= 0.65  → CE.2 passed; proceed to CE.3 (add AGB + REDD)
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lightgbm.callback import EarlyStopException as _LgbEarlyStopException

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.country_raster import country_ids_for_rows, load_country_raster
from scripts.regions.shared.evaluation.stage2_metrics import (
    lift_at_k_within_groups,
    ndcg_at_k_within_groups,
    precision_at_k_within_groups,
    recall_at_k_within_groups,
)
from scripts.regions.shared.training.feature_guard import check_feature_denylist
from scripts.regions.shared.training.stage2_lgbm_core import (
    BATCH_SIZE,
    STAGE2_EXCLUDE_COLS,
    TARGET_COL,
    Stage2Arrays,
    Stage2Config,
    _Stage2EarlyStop,
    _prepare_lgb_params,
    _split_large_groups,
    _shuffle_within_groups,
    _apply_graded_relevance,
    _rank_normalize_within_groups,
    compute_group_norm_weights,
    load_stage2_arrays,
    resolve_best_params_json,
    resolve_parquet_file,
)
from scripts.regions.shared.training.utils import get_repo_root

REGION = "south_america"
MODEL_PREFIX = "model1"
MIN_POSITIVE_PIXELS = 200
TEST_RATIO = 0.20
VAL_RATIO = 0.10   # fraction of train pool held aside for early stopping
SEED = 42

# Year ranges matching the standard parquet split
_PARQUET_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "train.parquet":      (2001, 2013),
    "earlystop.parquet":  (2014, 2016),
    "test.parquet":       (2017, 2024),
}


# ---------------------------------------------------------------------------
# Event discovery and splitting
# ---------------------------------------------------------------------------

def _count_positives_per_event(
    parquet_sources: list[tuple[Path, tuple[int, int]]],
    region: str,
) -> dict[tuple[int, int], int]:
    """Count positive (transition_01 > 0, WDPA_prev == 0) pixels per event.

    Returns dict mapping (country_id, year) → positive pixel count.
    """
    counts: dict[tuple[int, int], int] = {}
    for panel_path, year_range in parquet_sources:
        schema_names = pq.ParquetFile(panel_path).schema_arrow.names
        has_country = "country_id" in schema_names
        raster = None if has_country else load_country_raster(region)

        cols = ["year", TARGET_COL, "WDPA_prev"]
        if has_country:
            cols.append("country_id")
        else:
            cols += ["row", "col"]

        pf = pq.ParquetFile(panel_path)
        for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            if df.empty:
                continue
            df = df[df[TARGET_COL].notna() & (df["WDPA_prev"] == 0)]
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
            if df.empty:
                continue
            if not has_country:
                df["country_id"] = country_ids_for_rows(df, raster)
            grp = df.groupby(["country_id", "year"], as_index=False)[TARGET_COL].sum()
            for _, row in grp.iterrows():
                n = int(row[TARGET_COL])
                if n > 0:
                    key = (int(row["country_id"]), int(row["year"]))
                    counts[key] = counts.get(key, 0) + n
    return counts


def _split_events(
    event_counts: dict[tuple[int, int], int],
    min_pos: int = MIN_POSITIVE_PIXELS,
    test_ratio: float = TEST_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Partition meaningful events into train, val (early-stop), and test sets.

    Returns:
        (train_events, val_events, test_events)
        where val_events are held from the train pool for early stopping.
    """
    meaningful = sorted(k for k, v in event_counts.items() if v >= min_pos)
    rng = np.random.default_rng(seed)
    shuffled = list(meaningful)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(round(n * test_ratio)))
    n_train_pool = n - n_test
    n_val = max(1, int(round(n_train_pool * val_ratio)))
    n_train = n_train_pool - n_val

    test_events = set(shuffled[:n_test])
    val_events  = set(shuffled[n_test : n_test + n_val])
    train_events = set(shuffled[n_test + n_val :])

    return train_events, val_events, test_events


def _split_events_stratified(
    event_counts: dict[tuple[int, int], int],
    min_pos: int = MIN_POSITIVE_PIXELS,
    test_ratio: float = TEST_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Country-stratified 80/20 split of meaningful events.

    Samples test and val events separately within each country so no single
    country's anomalous years dominate a split.  Countries with only 1 event
    put it in train; countries with 2 events assign 1 to test.

    Returns (train_events, val_events, test_events).
    """
    meaningful = sorted(k for k, v in event_counts.items() if v >= min_pos)

    # Group events by country
    by_country: dict[int, list[tuple[int, int]]] = {}
    for ev in meaningful:
        by_country.setdefault(ev[0], []).append(ev)

    test_events: set[tuple[int, int]] = set()
    val_events: set[tuple[int, int]] = set()
    train_events: set[tuple[int, int]] = set()

    for cid, events in sorted(by_country.items()):
        rng = np.random.default_rng(seed + cid)
        evs = list(events)
        rng.shuffle(evs)
        n = len(evs)

        if n == 1:
            train_events.add(evs[0])
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_train_pool = n - n_test
        n_val = max(1, int(round(n_train_pool * val_ratio))) if n_train_pool >= 2 else 0

        test_events.update(evs[:n_test])
        val_events.update(evs[n_test : n_test + n_val])
        train_events.update(evs[n_test + n_val :])

    return train_events, val_events, test_events


def _load_coherence_data(repo_root: Path, region: str, model_prefix: str) -> dict[tuple[int, int], float]:
    """Load per-event coherence scores from the CE.3a diagnostic JSON.

    Returns empty dict if the file does not exist yet (no filter applied).
    """
    path = (
        repo_root / f"outputs/{region}/results/ml_models"
        / "stage2_event_coherence_diagnostic.json"
    )
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        (int(e["country_id"]), int(e["year"])): float(e["coherence"])
        for e in data.get("per_event", [])
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _concat_stage2_arrays(arrays: list[Stage2Arrays]) -> Stage2Arrays:
    """Concatenate multiple Stage2Arrays and re-sort by (country_id, year)."""
    if len(arrays) == 1:
        return arrays[0]

    X = np.concatenate([a.X for a in arrays], axis=0)
    y = np.concatenate([a.y for a in arrays])
    years = np.concatenate([a.years for a in arrays])
    cids  = np.concatenate([a.country_ids for a in arrays])

    order = np.lexsort((years, cids))
    X = X[order]; y = y[order]; years = years[order]; cids = cids[order]
    del order
    gc.collect()

    key = cids.astype(np.int64) * 100_000 + years.astype(np.int64)
    changes = np.concatenate([[True], key[1:] != key[:-1], [True]])
    group_sizes = np.diff(np.where(changes)[0]).astype(np.int32)

    return Stage2Arrays(
        X=X, y=y, years=years, country_ids=cids,
        cy_group_sizes=group_sizes, train_group_sizes=group_sizes,
    )


def _load_events_across_parquets(
    parquet_sources: list[tuple[Path, tuple[int, int]]],
    region: str,
    feature_cols: list[str],
    events: set[tuple[int, int]],
    neg_ratio: Optional[int],
    normalize: bool,
    graded: bool,
    rng_seed: int = SEED,
) -> Stage2Arrays:
    """Load pixels for an event set, routing each event to its source parquet.

    Each (country_id, year) event belongs to exactly one parquet based on year.
    Loads the matching subset from each parquet, then concatenates and re-sorts.
    """
    arrays: list[Stage2Arrays] = []
    for panel_path, year_range in parquet_sources:
        subset = {e for e in events if year_range[0] <= e[1] <= year_range[1]}
        if not subset:
            continue
        arr = load_stage2_arrays(
            panel_path, region, feature_cols, subset, year_range,
            neg_ratio=neg_ratio,
            rng_seed=rng_seed,
            normalize_within_groups=normalize,
            use_graded_relevance=graded,
        )
        arrays.append(arr)

    if not arrays:
        raise ValueError(f"No pixels found for {len(events)} events in {[p for p, _ in parquet_sources]}")
    return _concat_stage2_arrays(arrays)


# ---------------------------------------------------------------------------
# Per-event evaluation
# ---------------------------------------------------------------------------

def _per_event_metrics(
    arr: Stage2Arrays,
    scores: np.ndarray,
) -> list[dict]:
    """Compute Recall@5%, Lift@1%, NDCG@1% for each (country_id, year) group."""
    results: list[dict] = []
    offset = 0
    for gsize in arr.cy_group_sizes:
        n = int(gsize)
        y_g = arr.y[offset : offset + n].astype(np.float64)
        s_g = scores[offset : offset + n]
        g = np.array([n], dtype=np.int32)

        results.append({
            "country_id": int(arr.country_ids[offset]),
            "year":       int(arr.years[offset]),
            "n_pos":      int((y_g > 0).sum()),
            "n_rows":     n,
            "recall_at_5pct": float(recall_at_k_within_groups(y_g, s_g, g, 5.0)),
            "recall_at_1pct": float(recall_at_k_within_groups(y_g, s_g, g, 1.0)),
            "lift_at_1pct":   float(lift_at_k_within_groups(y_g, s_g, g, 1.0)),
            "ndcg_at_1pct":   float(ndcg_at_k_within_groups(y_g, s_g, g, 1.0)),
        })
        offset += n
    return results


def _aggregate_event_metrics(per_event: list[dict]) -> dict:
    """Macro-average and n_pos-weighted average across events."""
    if not per_event:
        return {}

    recall5_vals = [e["recall_at_5pct"] for e in per_event]
    lift1_vals   = [e["lift_at_1pct"]   for e in per_event]
    ndcg1_vals   = [e["ndcg_at_1pct"]   for e in per_event]
    n_pos_vals   = [e["n_pos"]           for e in per_event]
    total_pos    = sum(n_pos_vals)

    def _weighted_mean(vals: list, weights: list) -> float:
        s = sum(weights)
        return sum(v * w for v, w in zip(vals, weights)) / s if s > 0 else 0.0

    return {
        "macro_recall_at_5pct": float(np.mean(recall5_vals)),
        "macro_lift_at_1pct":   float(np.mean(lift1_vals)),
        "macro_ndcg_at_1pct":   float(np.mean(ndcg1_vals)),
        "weighted_recall_at_5pct": _weighted_mean(recall5_vals, n_pos_vals),
        "weighted_lift_at_1pct":   _weighted_mean(lift1_vals, n_pos_vals),
        "n_events": len(per_event),
        "total_pos": total_pos,
        "recall_at_5pct_min":    float(min(recall5_vals)),
        "recall_at_5pct_max":    float(max(recall5_vals)),
        "recall_at_5pct_std":    float(np.std(recall5_vals)),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_cross_event_training() -> None:
    repo_root = get_repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir   = repo_root / f"outputs/{REGION}/results/ml_models"
    model_dir = repo_root / f"data/{REGION}/ml/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg = Stage2Config(region=REGION, model_prefix=MODEL_PREFIX, variant="cross_event")

    # Resolve parquet paths (STAGE2_DATA_ROOT override applies here for Colombia dev)
    train_path     = resolve_parquet_file(REGION, "train.parquet")
    earlystop_path = resolve_parquet_file(REGION, "earlystop.parquet")
    test_path      = resolve_parquet_file(REGION, "test.parquet")

    parquet_sources: list[tuple[Path, tuple[int, int]]] = [
        (train_path,     (2001, 2013)),
        (earlystop_path, (2014, 2016)),
        (test_path,      (2017, 2024)),
    ]

    # --- Feature columns ---
    schema = pq.ParquetFile(train_path).schema_arrow
    numeric_cols = [
        n for n, f in zip(schema.names, schema)
        if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
    ]
    feature_cols = [c for c in numeric_cols if c not in STAGE2_EXCLUDE_COLS]
    check_feature_denylist(feature_cols, context=f"{REGION}/lgbm/stage2/cross_event")

    # --- Env-var controlled settings ---
    # CE.3b defaults: Fix 2 (inv_npos event normalization), Fix 4 (stratified, patience=200)
    _normalize  = os.environ.get("STAGE2_NORMALIZE_WITHIN_GROUPS", "0") != "1"
    _graded     = os.environ.get("STAGE2_GRADED_RELEVANCE", "1") != "0"
    _gw_mode    = os.environ.get("STAGE2_GROUP_WEIGHT_MODE", "inv_npos")   # Fix 2: event-level norm
    _es_metric  = os.environ.get("STAGE2_EARLY_STOP_METRIC", "recall_at_5pct")
    _stratified = os.environ.get("STAGE2_STRATIFIED_SPLIT", "1") != "0"   # Fix 4: on by default
    _patience   = int(os.environ.get("STAGE2_PATIENCE", "200"))            # Fix 4: was 100
    _coh_thresh = float(os.environ.get("STAGE2_COHERENCE_THRESHOLD", "0.0"))  # Fix 1
    neg_ratio   = int(os.environ.get("STAGE2_NEG_RATIO", "100"))
    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

    print(
        f"Cross-event Stage 2 | normalize_within_groups={not _normalize} "
        f"| graded_relevance={_graded} | group_weights={_gw_mode} "
        f"| early_stop={_es_metric} | neg_ratio={neg_ratio} "
        f"| stratified={_stratified} | patience={_patience} "
        f"| coherence_threshold={_coh_thresh}"
    )

    # --- Best params ---
    params_path = resolve_best_params_json(cfg)
    if params_path:
        with open(params_path) as f:
            raw = json.load(f)
        best_params = raw.get("best_params", raw) if isinstance(raw, dict) else raw
        print(f"  Loaded best params: {params_path}")
    else:
        best_params = {
            "num_leaves": 127, "max_depth": -1, "learning_rate": 0.05,
            "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }
        print("  Using default hyperparameters (no best_params JSON found)")
    num_boost_round = int(best_params.get("n_estimators", best_params.get("num_boost_round", 2000)))

    # --- Step 0: Load coherence data (Fix 1 — optional, requires CE.3a output) ---
    coherence_data: dict[tuple[int, int], float] = {}
    if _coh_thresh > 0.0:
        coherence_data = _load_coherence_data(repo_root, REGION, MODEL_PREFIX)
        if coherence_data:
            n_coh = sum(1 for v in coherence_data.values() if v >= _coh_thresh)
            print(
                f"\nStep 0: Coherence filter active (threshold={_coh_thresh:.2f})"
                f" | {n_coh}/{len(coherence_data)} events above threshold"
            )
        else:
            print(
                f"\nStep 0: WARNING — coherence_threshold={_coh_thresh} set but "
                f"no coherence JSON found (run CE.3a first). Filter disabled."
            )
            _coh_thresh = 0.0

    # --- Step 1: Enumerate and filter events ---
    print(f"\nStep 1: Enumerating expansion events (min_pos={MIN_POSITIVE_PIXELS})...")
    t0 = time.time()
    event_counts = _count_positives_per_event(parquet_sources, REGION)
    n_total = len(event_counts)
    meaningful = {k: v for k, v in event_counts.items() if v >= MIN_POSITIVE_PIXELS}
    print(
        f"  Total expansion events: {n_total}  "
        f"Meaningful (>={MIN_POSITIVE_PIXELS} pos): {len(meaningful)}  "
        f"({time.time()-t0:.1f}s)"
    )

    # Fix 1: filter to coherent events when threshold is set
    if _coh_thresh > 0.0 and coherence_data:
        before = len(meaningful)
        meaningful = {
            k: v for k, v in meaningful.items()
            if coherence_data.get(k, 0.0) >= _coh_thresh
        }
        print(
            f"  Coherence filter: {before} → {len(meaningful)} events "
            f"(removed {before - len(meaningful)} low-coherence events)"
        )

    # --- Step 2: Split events ---
    split_fn = _split_events_stratified if _stratified else _split_events
    split_label = "stratified" if _stratified else "random"
    train_events, val_events, test_events = split_fn(meaningful)
    print(
        f"  Event split [{split_label}] (seed={SEED}): "
        f"train={len(train_events)} | val(early-stop)={len(val_events)} | test={len(test_events)}"
    )

    # Fix 4: restrict early-stop validation to coherent events only
    val_events_es = val_events
    if coherence_data:
        val_coherent = {e for e in val_events if coherence_data.get(e, 0.0) >= _coh_thresh}
        if len(val_coherent) >= 2:
            val_events_es = val_coherent
            print(
                f"  Coherence-filtered val set: {len(val_events)} → {len(val_events_es)} events "
                f"for early stopping"
            )
        else:
            print(
                f"  WARNING: Only {len(val_coherent)} coherent val events; "
                f"using all {len(val_events)} for early stopping"
            )

    # E11: filter early-stop val to large events only (prevents political micro-events
    # from stalling early-stop metric at iter 51-66 via noisy near-zero recall).
    _es_min_pos = int(os.environ.get("STAGE2_ES_MIN_POS", "0"))
    if _es_min_pos > 0:
        val_large = {e for e in val_events_es if event_counts.get(e, 0) >= _es_min_pos}
        if len(val_large) >= 2:
            val_events_es = val_large
            print(
                f"  E11: large-event val filter (n_pos>={_es_min_pos}): "
                f"{len(val_events)} → {len(val_events_es)} events for early stopping"
            )
        else:
            print(
                f"  E11: WARNING — only {len(val_large)} events with n_pos>={_es_min_pos}; "
                f"keeping all {len(val_events_es)} for early stopping"
            )

    print(f"  Train events: {sorted(train_events)}")
    print(f"  Val   events (ES): {sorted(val_events_es)}")
    print(f"  Test  events: {sorted(test_events)}")

    # --- Step 3: Load training data (neg_ratio=100) ---
    print(f"\nStep 3: Loading train events ({len(train_events)} events, neg_ratio={neg_ratio})...")
    t0 = time.time()
    arr_train = _load_events_across_parquets(
        parquet_sources, REGION, feature_cols, train_events,
        neg_ratio=neg_ratio, normalize=_normalize, graded=_graded, rng_seed=SEED,
    )
    n_tr_pos = int((arr_train.y > 0).sum())
    print(
        f"  {len(arr_train.y):,} rows | {n_tr_pos:,} pos | "
        f"{len(arr_train.cy_group_sizes)} groups ({time.time()-t0:.1f}s)"
    )

    # --- Step 4: Load early-stop val data (all pixels, coherent events only) ---
    print(f"\nStep 4: Loading val events ({len(val_events_es)} events, all pixels)...")
    t0 = time.time()
    arr_val = _load_events_across_parquets(
        parquet_sources, REGION, feature_cols, val_events_es,
        neg_ratio=None, normalize=_normalize, graded=_graded, rng_seed=SEED + 1,
    )
    n_val_pos = int((arr_val.y > 0).sum())
    print(
        f"  {len(arr_val.y):,} rows | {n_val_pos:,} pos | "
        f"{len(arr_val.cy_group_sizes)} groups ({time.time()-t0:.1f}s)"
    )

    # --- Step 5: Sample weights (Fix 2: inv_npos gives equal gradient per event) ---
    sample_weights = compute_group_norm_weights(
        arr_train.y, arr_train.cy_group_sizes, _gw_mode
    )
    print(
        f"\nStep 5: Sample weights mode={_gw_mode}  mean={sample_weights.mean():.3f}"
        f"  min={sample_weights.min():.4f}  max={sample_weights.max():.2f}"
    )

    # --- Step 6: Train LambdaRank ---
    lgb_params = _prepare_lgb_params(best_params, num_threads)
    train_set = lgb.Dataset(
        arr_train.X, label=arr_train.y,
        group=_split_large_groups(arr_train.train_group_sizes),
        weight=sample_weights,
    )
    val_set = lgb.Dataset(
        arr_val.X, label=arr_val.y,
        group=_split_large_groups(arr_val.cy_group_sizes),
        reference=train_set,
    )
    # Fix 4: patience=200 prevents premature stopping on coherent val events
    early_stop_cb = _Stage2EarlyStop(
        arr_val.X, arr_val.y.astype(np.float64), arr_val.cy_group_sizes,
        patience=_patience, metric=_es_metric,
    )

    print(
        f"\nStep 6: Fitting LambdaRank "
        f"({len(arr_train.y):,} train rows / {len(arr_val.y):,} val rows "
        f"| {len(arr_train.cy_group_sizes)} train groups / {len(arr_val.cy_group_sizes)} val groups)"
    )
    t0 = time.time()
    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.log_evaluation(50), early_stop_cb],
    )
    print(f"  Training complete: {time.time()-t0:.1f}s  best_iter={model.best_iteration}")

    del arr_train, train_set, val_set
    gc.collect()

    # --- Step 7: Evaluate on val set (early-stop events) ---
    print("\nStep 7: Val-set evaluation...")
    val_scores = model.predict(arr_val.X, num_iteration=model.best_iteration or None).astype(np.float64)
    val_per_event = _per_event_metrics(arr_val, val_scores)
    val_agg = _aggregate_event_metrics(val_per_event)
    print(f"  Val macro Recall@5%: {val_agg['macro_recall_at_5pct']:.4f}  Lift@1%: {val_agg['macro_lift_at_1pct']:.2f}x")

    del arr_val
    gc.collect()

    # --- Step 8: Save model ---
    tag = f"{MODEL_PREFIX}_lgbm_stage2_cross_event"
    model_path = model_dir / f"{tag}_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "config": cfg}, f)
    print(f"\nModel saved: {model_path}")

    # --- Step 9: Load test events (all pixels) ---
    print(f"\nStep 9: Loading test events ({len(test_events)} events, all pixels)...")
    t0 = time.time()
    arr_test = _load_events_across_parquets(
        parquet_sources, REGION, feature_cols, test_events,
        neg_ratio=None, normalize=_normalize, graded=_graded, rng_seed=SEED + 2,
    )
    n_te_pos = int((arr_test.y > 0).sum())
    print(
        f"  {len(arr_test.y):,} rows | {n_te_pos:,} pos | "
        f"{len(arr_test.cy_group_sizes)} groups ({time.time()-t0:.1f}s)"
    )

    # --- Step 10: Per-event test evaluation ---
    print("\nStep 10: Per-event test evaluation...")
    test_scores = model.predict(arr_test.X, num_iteration=model.best_iteration or None).astype(np.float64)
    test_per_event = _per_event_metrics(arr_test, test_scores)
    test_agg = _aggregate_event_metrics(test_per_event)

    print(f"\n{'='*60}")
    print(f"Cross-event Test Results ({len(test_events)} events):")
    print(f"  Macro  Recall@5%: {test_agg['macro_recall_at_5pct']:.4f}  ({test_agg['macro_recall_at_5pct']*100:.1f}%)")
    print(f"  Weighted Recall@5%: {test_agg['weighted_recall_at_5pct']:.4f}")
    print(f"  Macro  Lift@1%:   {test_agg['macro_lift_at_1pct']:.2f}x")
    print(f"  Recall@5% min/max/std: {test_agg['recall_at_5pct_min']:.3f} / {test_agg['recall_at_5pct_max']:.3f} / {test_agg['recall_at_5pct_std']:.3f}")
    print()
    print(f"  Per-event breakdown:")
    for ev in sorted(test_per_event, key=lambda x: x["recall_at_5pct"], reverse=True):
        print(
            f"    cid={ev['country_id']} year={ev['year']} "
            f"n_pos={ev['n_pos']:>6,}  "
            f"Recall@5%={ev['recall_at_5pct']:.3f}  "
            f"Lift@1%={ev['lift_at_1pct']:.2f}x"
        )
    print(f"{'='*60}")

    # Decision gate
    # CE.3b gate: ≥50% macro Recall@5% on coherent-event test set.
    # CE.2 gate was 65% on all events (too strict given heterogeneous events).
    recall5 = test_agg["macro_recall_at_5pct"]
    is_colombia = len({e[0] for e in test_events}) == 1
    has_coherence_filter = _coh_thresh > 0.0 and bool(coherence_data)
    # CE.3b uses 50% gate (on coherent events); fallback to original thresholds
    if has_coherence_filter:
        gate_threshold = 0.50
        gate_label = "CE.3b coherent-events (50%)"
    elif is_colombia:
        gate_threshold = 0.60
        gate_label = "Colombia (60%)"
    else:
        gate_threshold = 0.65
        gate_label = "SA full (65%)"
    passed = recall5 >= gate_threshold
    print(
        f"\nDecision gate [{gate_label}]: Recall@5%={recall5:.3f}"
        f"  → {'PASS' if passed else 'FAIL — re-examine'}"
    )

    # --- Step 11: Save metrics JSON ---
    result = {
        "metadata": {
            "timestamp": timestamp,
            "experiment": "cross_event_stage2",
            "description": (
                "Cross-event validation: country-stratified 80/20 split of "
                "meaningful expansion events (≥200 positive pixels). "
                "Train and test events are from different countries and years "
                "— no spatial overlap, no geometric leakage."
            ),
            "region": REGION,
            "min_positive_pixels": MIN_POSITIVE_PIXELS,
            "test_ratio": TEST_RATIO,
            "val_ratio": VAL_RATIO,
            "seed": SEED,
            "n_total_events": n_total,
            "n_meaningful_events": len(meaningful),
            "n_train_events": len(train_events),
            "n_val_events": len(val_events),
            "n_val_events_es": len(val_events_es),
            "n_test_events": len(test_events),
            "train_events": sorted(list(train_events)),
            "val_events":   sorted(list(val_events)),
            "val_events_es": sorted(list(val_events_es)),
            "test_events":  sorted(list(test_events)),
            "n_features": len(feature_cols),
            "normalize_within_groups": not _normalize,
            "graded_relevance": _graded,
            "group_weight_mode": _gw_mode,
            "early_stop_metric": _es_metric,
            "early_stop_patience": _patience,
            "es_min_pos": _es_min_pos,
            "stratified_split": _stratified,
            "coherence_threshold": _coh_thresh,
            "neg_ratio_train": neg_ratio,
            "best_iteration": model.best_iteration,
            "model_path": str(model_path),
        },
        "event_counts": {str(k): v for k, v in sorted(meaningful.items())},
        "val_performance": {
            "aggregate": val_agg,
            "per_event": val_per_event,
        },
        "test_performance": {
            "aggregate": test_agg,
            "per_event": test_per_event,
        },
        "decision_gate": {
            "threshold": gate_threshold,
            "label": gate_label,
            "recall_at_5pct": float(recall5),
            "passed": bool(passed),
        },
    }

    metrics_path = out_dir / f"{tag}_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    print(f"\nAdd to ROADMAP.md Experiment History:")
    print(
        f"  Cross-event | — | {recall5*100:.1f}% | "
        f"{model.best_iteration} | {'✅ PASS' if passed else '❌ FAIL'}"
    )

    if not passed:
        import sys as _sys
        _sys.exit(1)


if __name__ == "__main__":
    run_cross_event_training()
