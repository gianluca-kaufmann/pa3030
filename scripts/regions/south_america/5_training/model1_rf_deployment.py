#!/usr/bin/env python3
"""Stage 0b (RF): Deployment Model Training — South America RandomForest (2001-2019).

Trains RandomForestClassifier on ALL non-censored labeled years (2001-2019) using
hyperparameters loaded from rf_best_params.json.  No CV, no test split.

Key differences from model1_LGBM_deployment.py:
  - Uses RandomForestClassifier (sklearn) instead of LightGBM
  - Imbalance: class_weight="balanced_subsample" — NO scale_pos_weight, NO temporal weighting
  - RF training: rf.fit(X, y) with NO sample weights
  - Negatives optionally capped via MAX_NEG_TRAIN env var (following model1_RF pattern)

Deployment calibration split:
  Fits Platt + isotonic calibrators using an auxiliary model trained on
  2001-2016 that scores 2017-2019 held-out pixels.

Outputs:
  data/south_america/ml/models/model1_rf_deployment_{timestamp}.pkl
      Pickle dict:
        {
          'model':        RandomForestClassifier (deployment model, 2001-2019),
          'platt':        LogisticRegression fitted on aux-model 2017-2019 scores,
          'isotonic':     IsotonicRegression fitted on aux-model 2017-2019 scores,
          'feature_cols': list[str],
          'timestamp':    str,
          'metadata':     dict,
        }
  outputs/south_america/results/forward/model1_rf_deployment_{timestamp}.txt
      Run log.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

from scripts.regions.shared.training.utils import (  # noqa: E402
    Tee,
    compute_metrics,
    compute_precision_at_k,
    extract_features_pyarrow_to_numpy,
    get_repo_root,
    report_memory_usage,
)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
EXCLUDE_COLS = {
    "transition_01", "transition_01_win5", "WDPA_b1", "WDPA_prev",
    "x", "y", "row", "col", "year",
}
TARGET_COL = "transition_01_win5"
WDPA_LAST_YEAR = 2024
LOOKAHEAD_YEARS = 5
LAST_LABEL_YEAR = WDPA_LAST_YEAR - LOOKAHEAD_YEARS  # 2019

DEPLOY_TRAIN_YEARS = (2001, 2019)
AUX_TRAIN_YEARS   = (2001, 2016)
AUX_CALIB_YEARS   = (2017, 2019)

BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "200_000"))
MAX_NEG_TRAIN = int(os.environ.get("MAX_NEG_TRAIN", "0"))
SK_VERBOSE    = int(os.environ.get("SK_VERBOSE", "0"))
# ─────────────────────────────────────────────────────────────────────────────


def get_n_jobs() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n = int(slurm_cpus)
            if n > 0:
                return n
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


N_JOBS = get_n_jobs()


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_parquet(filename: str) -> Path:
    """Locate a split parquet file (prefer $SCRATCH)."""
    repo_root = get_repo_root()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    candidates: list[Path] = []
    if scratch is not None:
        candidates += [
            scratch / f"outputs/south_america/results/main/{filename}",
            scratch / f"data/south_america/ml/main/{filename}",
            scratch / f"outputs/south_america/results/{filename}",
            scratch / f"data/south_america/ml/{filename}",
        ]
    candidates += [
        repo_root / f"outputs/south_america/results/main/{filename}",
        repo_root / f"data/south_america/ml/main/{filename}",
        repo_root / f"outputs/south_america/results/{filename}",
        repo_root / f"data/south_america/ml/{filename}",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"'{filename}' not found. Checked:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def resolve_best_params_json() -> Optional[Path]:
    repo_root = get_repo_root()
    script_dir = Path(__file__).resolve().parent
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    candidates: list[Path] = [script_dir / "rf_best_params.json"]
    if scratch is not None:
        candidates += [
            scratch / "scripts/regions/south_america/5_training/rf_best_params.json",
            scratch / "rf_best_params.json",
        ]
    candidates += [
        repo_root / "scripts/regions/south_america/5_training/rf_best_params.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


# ── Hyperparameter loading ────────────────────────────────────────────────────

def load_best_params(params_path: Optional[Path]) -> Dict[str, Any]:
    """Load + merge RF hyperparameters from JSON; handles flat and nested formats."""
    defaults = {
        "n_estimators": 1000,
        "max_depth": 20,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbose": SK_VERBOSE,
    }

    if params_path is None:
        print("  rf_best_params.json not found — using default guardrails")
        return defaults.copy()

    with open(params_path) as f:
        data = json.load(f)

    def _sanitize(p: Dict[str, Any]) -> Dict[str, Any]:
        return {k.replace("rf__", "", 1) if k.startswith("rf__") else k: v for k, v in p.items()}

    if isinstance(data, dict) and "best_params" in data:
        fixed = _sanitize(data.get("fixed_params", {}))
        best  = _sanitize(data["best_params"])
        params = {**fixed, **best}
    else:
        params = _sanitize(data) if isinstance(data, dict) else {}

    # Guardrails — only fill missing keys
    for k, v in defaults.items():
        if k not in params:
            params[k] = v

    # Always override threading from SLURM + force RF imbalance handling
    params["n_jobs"]        = N_JOBS
    params["random_state"]  = RANDOM_STATE
    params["class_weight"]  = "balanced_subsample"
    params["verbose"]       = SK_VERBOSE
    return params


# ── Data loading ──────────────────────────────────────────────────────────────

def _stream_data(
    paths: List[Path],
    feature_cols: List[str],
    year_range: Tuple[int, int],
    name: str = "",
    max_neg: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Two-pass streaming loader with optional negative cap.

    Returns:
        X: float32 (n_samples, n_features)
        y: int8    (n_samples,)
        n_pos: int
        n_neg: int
    """
    t0 = time.time()
    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev"]

    # Pass 1 — count positives and negatives separately
    print(f"  Pass 1: counting rows ({name}, years {year_range[0]}–{year_range[1]})…")
    n_pos_total = 0
    n_neg_total = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr  = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr  = batch["year"].to_numpy(zero_copy_only=False)
                wp  = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
                mask = (
                    ~null_mask
                    & (wp == 0)
                    & (yr >= year_range[0]) & (yr <= year_range[1])
                )
                if mask.any():
                    tgt_m = (tgt[mask] > 0).astype(np.int8)
                    n_pos_total += int(tgt_m.sum())
                    n_neg_total += int((tgt_m == 0).sum())
                del batch, tgt_arr, tgt, null_mask, yr, wp, mask
        finally:
            del pf

    if max_neg > 0 and n_neg_total > max_neg:
        n_neg_use = max_neg
        print(f"    {n_pos_total:,} pos / {n_neg_total:,} neg → capping neg to {n_neg_use:,}")
    else:
        n_neg_use = n_neg_total

    n_samples = n_pos_total + n_neg_use
    if n_samples == 0:
        raise ValueError(f"No samples in {name} for years {year_range}")
    print(f"    Using {n_samples:,} samples ({n_pos_total:,} pos / {n_neg_use:,} neg)")

    # Pass 2 — reservoir-sample negatives if capped, keep all positives
    X = np.empty((n_samples, len(feature_cols)), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int8)
    idx_pos = 0
    idx_neg = 0  # offset into X[n_pos_total:]

    rng = np.random.RandomState(RANDOM_STATE) if (max_neg > 0 and n_neg_total > max_neg) else None
    neg_seen = 0

    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr  = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr  = batch["year"].to_numpy(zero_copy_only=False)
                wp  = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
                mask = (
                    ~null_mask
                    & (wp == 0)
                    & (yr >= year_range[0]) & (yr <= year_range[1])
                )
                if not mask.any():
                    del batch, tgt_arr, tgt, null_mask, yr, wp, mask
                    continue

                feat_tbl = batch.select(feature_cols)
                X_b = extract_features_pyarrow_to_numpy(feat_tbl, mask)
                y_b = (tgt[mask] > 0).astype(np.int8)

                pos_mask_b = y_b == 1
                neg_mask_b = y_b == 0

                # All positives
                np_b = int(pos_mask_b.sum())
                if np_b > 0:
                    X[idx_pos:idx_pos + np_b] = X_b[pos_mask_b]
                    y[idx_pos:idx_pos + np_b] = 1
                    idx_pos += np_b

                # Negatives — reservoir sample if capping
                X_neg = X_b[neg_mask_b]
                nn_b = len(X_neg)
                if nn_b > 0:
                    if rng is None:
                        # No capping — just append
                        X[n_pos_total + idx_neg: n_pos_total + idx_neg + nn_b] = X_neg
                        y[n_pos_total + idx_neg: n_pos_total + idx_neg + nn_b] = 0
                        idx_neg += nn_b
                    else:
                        # Reservoir sampling (Algorithm R)
                        for i in range(nn_b):
                            neg_seen += 1
                            if idx_neg < n_neg_use:
                                X[n_pos_total + idx_neg] = X_neg[i]
                                y[n_pos_total + idx_neg] = 0
                                idx_neg += 1
                            else:
                                j = rng.randint(0, neg_seen)
                                if j < n_neg_use:
                                    X[n_pos_total + j] = X_neg[i]

                del batch, tgt, yr, wp, mask, feat_tbl, X_b, y_b
        finally:
            del pf
        gc.collect()

    # Trim to actual written rows
    actual = idx_pos + idx_neg
    X = X[:actual]
    y = y[:actual]
    n_pos = int(y.sum())
    n_neg = actual - n_pos
    elapsed = time.time() - t0
    print(f"  Loaded {actual:,} rows in {elapsed:.1f}s ({n_neg:,} neg / {n_pos:,} pos)")
    return X, y, n_pos, n_neg


# ── RF training helper ────────────────────────────────────────────────────────

def _train_rf(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    label: str = "",
) -> RandomForestClassifier:
    """Fit RandomForestClassifier — NO sample weights (class_weight handles imbalance)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"\n  Training RF {label} — {len(y):,} samples ({n_pos:,} pos / {n_neg:,} neg)")
    for k in sorted(params):
        print(f"    {k}: {params[k]}")

    rf = RandomForestClassifier(**params)
    t0 = time.time()
    rf.fit(X, y)
    print(f"  Done in {time.time() - t0:.1f}s")
    del X, y
    gc.collect()
    return rf


# ── Calibration ───────────────────────────────────────────────────────────────

def fit_deployment_calibrators(
    all_paths: List[Path],
    feature_cols: List[str],
    rf_params: Dict[str, Any],
) -> Tuple[LogisticRegression, IsotonicRegression]:
    """Fit Platt + isotonic calibrators using the deployment calibration split.

    Trains an auxiliary RF on AUX_TRAIN_YEARS (2001–2016) with the same
    locked hyperparameters, scores AUX_CALIB_YEARS (2017–2019) held-out
    pixels, then fits calibrators on those out-of-sample predictions.
    """
    print("\n" + "=" * 70)
    print("DEPLOYMENT CALIBRATION SPLIT (RF)")
    print(f"  Auxiliary model train: {AUX_TRAIN_YEARS[0]}–{AUX_TRAIN_YEARS[1]}")
    print(f"  Calibration held-out:  {AUX_CALIB_YEARS[0]}–{AUX_CALIB_YEARS[1]}")
    print("=" * 70)

    # Train auxiliary model on 2001-2016
    X_aux, y_aux, _, _ = _stream_data(
        all_paths, feature_cols, AUX_TRAIN_YEARS, "aux_train", max_neg=MAX_NEG_TRAIN
    )
    aux_rf = _train_rf(X_aux, y_aux, rf_params, "auxiliary calibration model")

    # Score 2017-2019 held-out pixels
    print("\nScoring calibration held-out set…")
    X_cal, y_cal, n_pos_cal, n_neg_cal = _stream_data(
        all_paths, feature_cols, AUX_CALIB_YEARS, "aux_calib"
    )
    y_cal_proba = aux_rf.predict_proba(X_cal)[:, 1].astype(np.float32)
    del aux_rf, X_cal
    gc.collect()

    cal_prauc  = average_precision_score(y_cal, y_cal_proba)
    cal_rocauc = roc_auc_score(y_cal, y_cal_proba)
    p1 = compute_precision_at_k(y_cal, y_cal_proba, 1)
    print(f"  Aux RF on calib set: PR-AUC {cal_prauc:.4f}, ROC-AUC {cal_rocauc:.4f}, P@1% {p1:.4f}")

    # Fit Platt (logistic on logit) calibrator
    print("\nFitting Platt calibrator…")
    eps = 1e-7
    logit_proba = np.log(
        np.clip(y_cal_proba, eps, 1 - eps) / (1 - np.clip(y_cal_proba, eps, 1 - eps))
    ).reshape(-1, 1)
    platt = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="lbfgs")
    platt.fit(logit_proba, y_cal)

    # Fit isotonic calibrator
    print("Fitting isotonic calibrator…")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(y_cal_proba, y_cal)

    # Quick check: Brier scores
    brier_raw   = brier_score_loss(y_cal, y_cal_proba)
    platt_cal   = platt.predict_proba(logit_proba)[:, 1]
    iso_cal     = iso.predict(y_cal_proba)
    brier_platt = brier_score_loss(y_cal, platt_cal)
    brier_iso   = brier_score_loss(y_cal, iso_cal)
    print(f"  Brier score — raw: {brier_raw:.6f}, Platt: {brier_platt:.6f}, isotonic: {brier_iso:.6f}")

    del y_cal, y_cal_proba, logit_proba, platt_cal, iso_cal
    gc.collect()
    return platt, iso


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    start = time.time()
    repo_root = get_repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("MODEL 1 RF — DEPLOYMENT MODEL (2001-2019)")
    print("=" * 70)

    # ── Paths ─────────────────────────────────────────────────────────────────
    train_path     = resolve_parquet("train_win5.parquet")
    earlystop_path = resolve_parquet("earlystop_win5.parquet")
    test_path      = resolve_parquet("test_win5.parquet")
    all_paths      = [train_path, earlystop_path, test_path]
    params_path    = resolve_best_params_json()

    model_dir  = repo_root / "data/south_america/ml/models"
    output_dir = repo_root / "outputs/south_america/results/forward"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  train:     {train_path}")
    print(f"  earlystop: {earlystop_path}")
    print(f"  test:      {test_path}")
    print(f"  params:    {params_path or '(default guardrails)'}")
    print(f"  n_jobs:    {N_JOBS}")
    print(f"  MAX_NEG_TRAIN: {MAX_NEG_TRAIN or '(no cap)'}")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    rf_params = load_best_params(params_path)
    print(f"\n  n_estimators: {rf_params.get('n_estimators')}")

    # ── Determine feature columns from train parquet schema ───────────────────
    schema = pq.ParquetFile(train_path).schema_arrow
    all_numeric = [
        name for name, field in zip(schema.names, schema)
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
    ]
    feature_cols = [c for c in all_numeric if c not in EXCLUDE_COLS]
    print(f"\n  Feature columns: {len(feature_cols)}")

    # ── Step 1: Train deployment model on 2001-2019 ───────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: TRAINING DEPLOYMENT MODEL (2001-2019)")
    print("=" * 70)
    report_memory_usage("before loading deployment training data")

    X_dep, y_dep, dep_pos, dep_neg = _stream_data(
        all_paths, feature_cols, DEPLOY_TRAIN_YEARS, "deployment_train",
        max_neg=MAX_NEG_TRAIN,
    )
    report_memory_usage("after loading deployment training data")

    deploy_model = _train_rf(X_dep, y_dep, rf_params, "deployment model")
    report_memory_usage("after training deployment model")

    # ── Step 2: Fit calibrators via auxiliary model ───────────────────────────
    platt_cal, iso_cal = fit_deployment_calibrators(all_paths, feature_cols, rf_params)
    report_memory_usage("after fitting calibrators")

    # ── Step 3: Save model + calibrators ─────────────────────────────────────
    artifact = {
        "model": deploy_model,
        "platt": platt_cal,
        "isotonic": iso_cal,
        "feature_cols": feature_cols,
        "timestamp": timestamp,
        "metadata": {
            "model_type": "rf",
            "train_years": list(DEPLOY_TRAIN_YEARS),
            "aux_train_years": list(AUX_TRAIN_YEARS),
            "aux_calib_years": list(AUX_CALIB_YEARS),
            "n_estimators": rf_params.get("n_estimators"),
            "n_features": len(feature_cols),
            "deploy_n_pos": dep_pos,
            "deploy_n_neg": dep_neg,
            "target_col": TARGET_COL,
            "params_path": str(params_path) if params_path else None,
            "total_time_s": round(time.time() - start, 1),
        },
    }

    model_path = model_dir / f"model1_rf_deployment_{timestamp}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(artifact, fh, protocol=5)
    print(f"\nSaved deployment artifact: {model_path}")

    # Print summary
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model:           RandomForest deployment (2001-2019)")
    print(f"  Features:        {len(feature_cols)}")
    print(f"  Train samples:   {dep_neg + dep_pos:,} ({dep_pos:,} pos / {dep_neg:,} neg)")
    print(f"  n_estimators:    {rf_params.get('n_estimators')}")
    print(f"  class_weight:    balanced_subsample (no temporal weighting)")
    print(f"  Calibrators:     Platt + isotonic (deployment calibration split)")
    print(f"  Artifact:        {model_path}")
    print(f"  Total time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    repo_root = get_repo_root()
    output_dir = repo_root / "outputs/south_america/results/forward"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"model1_rf_deployment_{timestamp}.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nLog saved: {log_path}")
