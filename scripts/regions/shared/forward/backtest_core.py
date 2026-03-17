#!/usr/bin/env python3
"""Stage 0c: Pseudo-forecast backtesting.

Validates the forward prediction methodology by simulating the analysis at
four historical time points: T ∈ {2013, 2015, 2017, 2019}.

For each origin year T:
  1. Train a historical deployment model on 2001–(T-1) with locked hyperparams.
  2. Score year-T feature rows for WDPA_prev==0 AND WDPA==0 pixels from
     merged_panel_final.parquet.
  3. Reconstruct 5-year window actuals from the WDPA column (years T+1…T+5)
     — NOT from transition_01_win5 (which may be absent or unreliable).
  4. Evaluate: Precision@1/5/10%, Lift@1/5/10%, Forecast Capture Rate.

Model-type support:
  lgbm  — trains lgb.Booster with locked hyperparams; temporal weighting applied.
  rf    — trains RandomForestClassifier with class_weight="balanced_subsample";
          NO temporal weighting (consistent with existing RF training scripts).

Right-censoring:
  Pixels where T+5 > LAST_LABEL_YEAR (2019) are excluded from quantitative
  evaluation.  T=2013 and T=2015 have clean 5-year windows.  T=2017 is
  partial (only 2 years observable).  T=2019 is a consistency check only.

Outputs (per origin T):
  outputs/{region}/results/forward/{model_type}/forward_backtest_T{T}.json
Aggregated (after all origins complete):
  outputs/{region}/results/forward/{model_type}/forward_backtest_results.json
  outputs/{region}/results/forward/{model_type}/forward_backtest_precision_over_time.pdf
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

from scripts.regions.shared.forward.config import (  # noqa: E402
    DATA_SUBDIR,
    MODEL_PREFIX,
    OUTPUTS_SUBDIR,
    get_repo_root,
)
from scripts.regions.shared.training.utils import (  # noqa: E402
    compute_precision_at_k,
    compute_year_weights,
    extract_features_pyarrow_to_numpy,
    get_repo_root as _get_repo_root,
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


# ── Coordinate + rasterization helpers (for false-positive map) ───────────────

def _bt_lonlat(x: np.ndarray, y: np.ndarray):
    """EPSG:3857 → WGS84 lon/lat (degrees)."""
    R = 6378137.0
    return np.degrees(x / R), np.degrees(2.0 * np.arctan(np.exp(y / R)) - np.pi / 2.0)


def _bt_rasterize(lon, lat, resolution, x_lim, y_lim):
    """Count pixels per cell; returns (H, W) float array with NaN for empty."""
    x_bins = np.arange(x_lim[0], x_lim[1] + resolution, resolution)
    y_bins = np.arange(y_lim[0], y_lim[1] + resolution, resolution)
    H, _, _ = np.histogram2d(lat, lon, bins=[y_bins, x_bins])
    H[H == 0] = np.nan
    return H


def create_false_positive_map(
    y_proba: np.ndarray,
    label_5yr: np.ndarray,
    evaluable: np.ndarray,
    xy: dict,
    origin_year: int,
    output_dir: Path,
) -> None:
    """Map true vs false positives in the top-5% risk zone at origin year T.

    Only runs when x/y coordinates are available (requires panel to include
    x, y columns) and when the origin year has a clean 5-year window.
    Outputs: forward_backtest_T{T}_false_positives.pdf/.png
    """
    if not (xy.get("x") is not None and xy.get("y") is not None):
        print(f"  T={origin_year}: skipping FP map — no x/y coords in panel.")
        return

    # Local config import (values correct after runner.py reload)
    from scripts.regions.shared.forward.config import REGION_LABEL, X_LIMITS, Y_LIMITS

    x_arr = xy["x"]
    y_arr = xy["y"]
    lon, lat = _bt_lonlat(x_arr.astype(np.float64), y_arr.astype(np.float64))

    # Top-5% probability threshold
    threshold = float(np.percentile(y_proba, 95))
    top_mask  = y_proba >= threshold

    ev_top  = top_mask & evaluable
    tp_mask = ev_top & (label_5yr == 1)
    fp_mask = ev_top & (label_5yr == 0)
    n_tp, n_fp = int(tp_mask.sum()), int(fp_mask.sum())
    precision = n_tp / max(n_tp + n_fp, 1)
    print(f"\n  T={origin_year} FP map: top-5% → {n_tp:,} TP, {n_fp:,} FP "
          f"(precision={precision:.3f})")

    if n_tp + n_fp == 0:
        print("  Nothing to plot — skipping FP map.")
        return

    res = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f"False Positive Analysis — T={origin_year} Backtest ({REGION_LABEL})\n"
        f"Top-5% risk zone: {n_tp:,} true positives (green) / {n_fp:,} false positives (red)  "
        f"precision={precision:.3f}",
        fontsize=11,
    )

    panel_defs = [
        (tp_mask, "True Positives\n(high-risk → actually protected)", "#2ca02c"),
        (fp_mask, "False Positives\n(high-risk → NOT protected in window)", "#d62728"),
    ]
    for ax, (mask, title, color) in zip(axes, panel_defs):
        if mask.any():
            grid = _bt_rasterize(lon[mask], lat[mask], res, X_LIMITS, Y_LIMITS)
            ax.imshow(
                np.flipud(grid), origin="lower",
                extent=[X_LIMITS[0], X_LIMITS[1], Y_LIMITS[0], Y_LIMITS[1]],
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c", ["white", color]),
                aspect="auto", interpolation="nearest",
            )
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        n_px = int(mask.sum())
        ax.text(0.02, 0.02, f"{n_px:,} pixels", transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()
    stem = output_dir / f"forward_backtest_T{origin_year}_false_positives"
    for ext in ["pdf", "png"]:
        plt.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}.pdf")


ORIGIN_YEARS = [2013, 2015, 2017, 2019]
N_ESTIMATORS_LOCKED_LGBM = 2555   # LGBM backtest locked round count
N_ESTIMATORS_LOCKED_RF   = 500    # RF uses fewer trees for speed
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "200_000"))
FIXED_PARAMS_LGBM = {
    "random_state": RANDOM_STATE, "boosting_type": "gbdt",
    "objective": "binary", "verbose": -1,
}
# ─────────────────────────────────────────────────────────────────────────────


def get_num_threads() -> int:
    s = os.environ.get("SLURM_CPUS_PER_TASK")
    if s:
        try:
            n = int(s)
            if n > 0:
                return n
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


NUM_THREADS = get_num_threads()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = str(NUM_THREADS)


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_panel(data_subdir: str, filename: str = "merged_panel_final.parquet") -> Path:
    repo_root = get_repo_root()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    candidates: list[Path] = []
    if scratch is not None:
        candidates += [
            scratch / f"data/{data_subdir}/ml/{filename}",
            scratch / f"outputs/{data_subdir}/results/{filename}",
            scratch / f"outputs/{data_subdir}/results/main/{filename}",
        ]
    candidates += [
        repo_root / f"data/{data_subdir}/ml/{filename}",
        repo_root / f"outputs/{data_subdir}/results/{filename}",
        repo_root / f"outputs/{data_subdir}/results/main/{filename}",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"'{filename}' not found.\nChecked:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def resolve_split_parquets(data_subdir: str) -> List[Path]:
    """Resolve split parquet files (train + earlystop + test) for training."""
    filenames = ["train_win5.parquet", "earlystop_win5.parquet", "test_win5.parquet"]
    repo_root = get_repo_root()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    paths = []
    for fn in filenames:
        found = None
        for base in ([scratch] if scratch else []) + [repo_root]:
            for sub in [
                f"data/{data_subdir}/ml/main",
                f"outputs/{data_subdir}/results/main",
                f"data/{data_subdir}/ml",
                f"outputs/{data_subdir}/results",
            ]:
                cand = base / sub / fn
                if cand.exists():
                    found = cand
                    break
            if found:
                break
        if found:
            paths.append(found)
    return paths


def resolve_best_params_json(model_prefix: str, data_subdir: str, model_type: str) -> Optional[Path]:
    """Find {model_type}_best_params.json for the given region."""
    repo_root = get_repo_root()
    # The training script lives at scripts/regions/{data_subdir}/5_training/
    training_dir = repo_root / f"scripts/regions/{data_subdir}/5_training"
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    json_name = f"{model_type}_best_params.json"
    candidates = [training_dir / json_name]
    if scratch is not None:
        candidates.append(
            scratch / f"scripts/regions/{data_subdir}/5_training/{json_name}"
        )
    candidates.append(repo_root / f"scripts/regions/{data_subdir}/5_training/{json_name}")
    for c in candidates:
        if c.exists():
            return c
    return None


# ── Hyperparameters ───────────────────────────────────────────────────────────

def load_best_params_lgbm(params_path: Optional[Path], num_threads: int) -> Dict[str, Any]:
    if params_path is None:
        return {
            **FIXED_PARAMS_LGBM,
            "max_depth": 8, "num_leaves": 255,
            "min_child_samples": 50, "subsample": 0.7, "subsample_freq": 1,
            "colsample_bytree": 0.7, "learning_rate": 0.03,
            "reg_alpha": 1.0, "reg_lambda": 1.0,
            "metric": "average_precision", "num_threads": num_threads,
        }
    with open(params_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "best_params" in data:
        params = {**FIXED_PARAMS_LGBM, **data.get("fixed_params", {}), **data["best_params"]}
    else:
        params = {**FIXED_PARAMS_LGBM, **(data if isinstance(data, dict) else {})}
    params["num_threads"] = num_threads
    return params


def load_best_params_rf(params_path: Optional[Path], n_jobs: int) -> Dict[str, Any]:
    defaults = {
        "n_estimators": N_ESTIMATORS_LOCKED_RF,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
    }
    if params_path is None:
        params = defaults.copy()
    else:
        with open(params_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "best_params" in data:
            loaded = {**data.get("fixed_params", {}), **data["best_params"]}
        else:
            loaded = data if isinstance(data, dict) else {}
        params = {**defaults, **loaded}
    # For backtest, use the locked RF estimator count (faster)
    params["n_estimators"] = N_ESTIMATORS_LOCKED_RF
    params["random_state"] = RANDOM_STATE
    params["n_jobs"] = n_jobs
    return params


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_training_data(
    paths: List[Path],
    feature_cols: List[str],
    year_range: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Stream training data (WDPA_prev==0, year in range) from split parquets."""
    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev"]
    n_samples = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr  = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr  = batch["year"].to_numpy(zero_copy_only=False)
                wp  = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
                mask = (~null_mask
                        & (wp == 0)
                        & (yr >= year_range[0]) & (yr <= year_range[1]))
                n_samples += int(mask.sum())
                del batch, tgt_arr, tgt, null_mask, yr, wp, mask
        finally:
            del pf

    if n_samples == 0:
        raise ValueError(f"No training samples for years {year_range}")

    X = np.empty((n_samples, len(feature_cols)), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int8)
    years_arr = np.empty(n_samples, dtype=np.int32)
    idx = 0

    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr  = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr  = batch["year"].to_numpy(zero_copy_only=False)
                wp  = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
                mask = (~null_mask
                        & (wp == 0)
                        & (yr >= year_range[0]) & (yr <= year_range[1]))
                if not mask.any():
                    del batch, tgt_arr, tgt, null_mask, yr, wp, mask
                    continue
                X_b = extract_features_pyarrow_to_numpy(batch.select(feature_cols), mask)
                y_b = (tgt[mask] > 0).astype(np.int8)
                k = len(y_b)
                X[idx:idx + k] = X_b
                y[idx:idx + k] = y_b
                years_arr[idx:idx + k] = yr[mask].astype(np.int32)
                idx += k
                del batch, tgt, yr, wp, mask, X_b, y_b
        finally:
            del pf
        gc.collect()

    n_pos = int(y.sum())
    n_neg = n_samples - n_pos
    return X, y, years_arr, n_pos, n_neg


def load_inference_rows_and_wdpa(
    panel_path: Path,
    feature_cols: List[str],
    origin_year: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load year-T pixels (WDPA_prev==0, WDPA==0) + WDPA values for T+1…T+5."""
    T = origin_year
    window_years = list(range(T + 1, T + LOOKAHEAD_YEARS + 1))
    is_evaluable_flag = (T + LOOKAHEAD_YEARS) <= LAST_LABEL_YEAR

    cols_inf = feature_cols + ["year", "WDPA_prev", "WDPA", "row", "col"]
    schema_cols = set(pq.ParquetFile(panel_path).schema_arrow.names)
    cols_inf = [c for c in cols_inf if c in schema_cols]
    optional_xy = [c for c in ["x", "y"] if c in schema_cols]

    print(f"\nLoading year-T={T} inference rows (WDPA_prev==0 AND WDPA==0)…")
    rows_list, cols_list, X_list = [], [], []
    x_list, y_list = [], []

    pf = pq.ParquetFile(panel_path)
    try:
        read_cols = list(set(cols_inf + optional_xy + feature_cols))
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols, use_threads=True):
            yr   = batch["year"].to_numpy(zero_copy_only=False)
            wp   = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
            wdpa = batch["WDPA"].to_numpy(zero_copy_only=False)
            mask = (yr == T) & (wp == 0) & (wdpa == 0)
            if not mask.any():
                del batch, yr, wp, wdpa, mask
                continue
            rows_list.append(batch["row"].to_numpy(zero_copy_only=False)[mask].astype(np.int32))
            cols_list.append(batch["col"].to_numpy(zero_copy_only=False)[mask].astype(np.int32))
            feat_tbl = batch.select(feature_cols)
            X_list.append(extract_features_pyarrow_to_numpy(feat_tbl, mask))
            if "x" in schema_cols:
                x_list.append(batch["x"].to_numpy(zero_copy_only=False)[mask].astype(np.float32))
            if "y" in schema_cols:
                y_list.append(batch["y"].to_numpy(zero_copy_only=False)[mask].astype(np.float32))
            del batch, yr, wp, wdpa, mask, feat_tbl
    finally:
        del pf

    if not X_list:
        raise ValueError(f"No inference rows found for origin year T={T}")

    X_inf   = np.concatenate(X_list, axis=0)
    rows_arr = np.concatenate(rows_list, axis=0)
    cols_arr = np.concatenate(cols_list, axis=0)
    n_inf = len(rows_arr)
    print(f"  {n_inf:,} inference pixels at T={T}")

    # Build pixel-id → index mapping
    pixel_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows_arr, cols_arr))}
    label_5yr = np.zeros(n_inf, dtype=np.int8)

    years_to_check = [yr for yr in window_years if yr <= LAST_LABEL_YEAR]
    if years_to_check:
        print(f"  Reconstructing 5-year labels from years {years_to_check[0]}–{years_to_check[-1]}…")
        pf2 = pq.ParquetFile(panel_path)
        try:
            for batch in pf2.iter_batches(
                batch_size=BATCH_SIZE, columns=["year", "row", "col", "WDPA"], use_threads=True
            ):
                yr   = batch["year"].to_numpy(zero_copy_only=False)
                wdpa = batch["WDPA"].to_numpy(zero_copy_only=False)
                r_arr = batch["row"].to_numpy(zero_copy_only=False)
                c_arr = batch["col"].to_numpy(zero_copy_only=False)
                window_mask = np.isin(yr, years_to_check) & (wdpa == 1)
                if not window_mask.any():
                    del batch, yr, wdpa, r_arr, c_arr, window_mask
                    continue
                for r, c in zip(r_arr[window_mask], c_arr[window_mask]):
                    pix = (int(r), int(c))
                    if pix in pixel_idx:
                        label_5yr[pixel_idx[pix]] = 1
                del batch, yr, wdpa, r_arr, c_arr, window_mask
        finally:
            del pf2

    evaluable = np.ones(n_inf, dtype=bool) if is_evaluable_flag else np.zeros(n_inf, dtype=bool)
    if not is_evaluable_flag:
        evaluable_years = len(years_to_check)
        print(f"  NOTE: T={T} has only {evaluable_years}/{LOOKAHEAD_YEARS} years observable "
              f"(T+5={T+LOOKAHEAD_YEARS} > LAST_LABEL_YEAR={LAST_LABEL_YEAR})")
        evaluable[:] = (evaluable_years > 0)

    row_col = np.stack([rows_arr, cols_arr], axis=1)
    xy: dict = {}
    if x_list:
        xy["x"] = np.concatenate(x_list)
    if y_list:
        xy["y"] = np.concatenate(y_list)

    return X_inf, row_col, label_5yr, evaluable, xy


# ── Training helpers ──────────────────────────────────────────────────────────

def _train_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    years_arr: np.ndarray,
    best_params: Dict[str, Any],
    n_est: int,
    year_range: Tuple[int, int],
):
    import lightgbm as lgb

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)
    weights = compute_year_weights(years_arr, min_year=year_range[0], max_year=year_range[1])

    p = {**best_params, "is_unbalance": False}
    for k in ("num_boost_round", "n_estimators", "n_jobs", "scale_pos_weight",
              "early_stopping_rounds", "early_stopping",
              "sampling_strategy", "replacement", "bootstrap", "class_weight"):
        p.pop(k, None)
    p["scale_pos_weight"] = spw
    p.setdefault("num_threads", NUM_THREADS)

    ds = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)
    del X, y, years_arr, weights
    gc.collect()

    t0 = time.time()
    model = lgb.train(p, ds, num_boost_round=n_est, callbacks=[lgb.log_evaluation(500)])
    print(f"  Training done in {time.time() - t0:.1f}s")
    del ds
    gc.collect()
    return model


def _train_rf(
    X: np.ndarray,
    y: np.ndarray,
    rf_params: Dict[str, Any],
):
    from sklearn.ensemble import RandomForestClassifier

    # Build final RF params — strip keys not accepted by sklearn
    params = {k: v for k, v in rf_params.items()
              if k not in ("scale_pos_weight", "num_threads", "metric",
                           "boosting_type", "objective", "verbose",
                           "num_boost_round", "learning_rate",
                           "reg_alpha", "reg_lambda",
                           "subsample", "subsample_freq", "colsample_bytree",
                           "num_leaves", "min_child_samples")}
    params["class_weight"] = "balanced_subsample"
    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("n_jobs", NUM_THREADS)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  RF training: {len(y):,} samples ({n_pos:,} pos / {n_neg:,} neg), "
          f"n_estimators={params.get('n_estimators')}")
    for k in sorted(params):
        print(f"    {k}: {params[k]}")

    rf = RandomForestClassifier(**params)
    t0 = time.time()
    rf.fit(X, y)
    print(f"  Training done in {time.time() - t0:.1f}s")
    del X, y
    gc.collect()
    return rf


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_backtest_metrics(y_true: np.ndarray, y_proba: np.ndarray, label: str) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    n = len(y_true)
    n_pos = int(y_true.sum())
    base_rate = n_pos / max(n, 1)

    metrics: Dict[str, Any] = {
        "n_pixels": n,
        "n_pos": n_pos,
        "base_rate": float(base_rate),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if n_pos > 0 else None,
        "pr_auc":  float(average_precision_score(y_true, y_proba)) if n_pos > 0 else None,
    }
    for k in [1, 5, 10]:
        prec = compute_precision_at_k(y_true, y_proba, k)
        lift = prec / max(base_rate, 1e-9)
        metrics[f"precision_at_{k}pct"] = float(prec)
        metrics[f"lift_at_{k}pct"] = float(lift)

    # Forecast Capture Rate: fraction of all positives in top 5% predicted
    n_top5 = max(1, int(n * 0.05))
    top5_idx = np.argsort(y_proba)[-n_top5:]
    fcr = float(y_true[top5_idx].sum()) / max(n_pos, 1)
    metrics["forecast_capture_rate_top5pct"] = fcr

    print(f"\n  {label}")
    print(f"    n={n:,}, pos={n_pos:,} ({base_rate:.4%})")
    if metrics["roc_auc"] is not None:
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")
    for k in [1, 5, 10]:
        print(f"    P@{k}%: {metrics[f'precision_at_{k}pct']:.4f}  Lift: {metrics[f'lift_at_{k}pct']:.2f}x")
    print(f"    Forecast Capture Rate (top 5%): {fcr:.4f}")

    return metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_backtest_precision(all_results: List[Dict], output_dir: Path, model_type: str) -> None:
    clean   = [r for r in all_results if r.get("clean_5yr_window")]
    partial = [r for r in all_results if not r.get("clean_5yr_window") and r.get("metrics")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Pseudo-forecast Backtesting ({model_type.upper()}): Precision@K% and Lift@K%",
        fontsize=13,
    )

    colors = {"1": "#1f77b4", "5": "#ff7f0e", "10": "#2ca02c"}

    for ax, metric_prefix, ylabel in [
        (axes[0], "precision_at", "Precision@K%"),
        (axes[1], "lift_at",      "Lift@K%"),
    ]:
        for k, color in colors.items():
            key = f"{metric_prefix}_{k}pct"
            if clean:
                xs = [r["origin_year"] for r in clean if r.get("metrics") and key in r["metrics"]]
                ys = [r["metrics"][key] for r in clean if r.get("metrics") and key in r["metrics"]]
                ax.plot(xs, ys, marker="o", color=color, label=f"K={k}% (clean)")
            if partial:
                xs_p = [r["origin_year"] for r in partial if r.get("metrics") and key in r["metrics"]]
                ys_p = [r["metrics"][key] for r in partial if r.get("metrics") and key in r["metrics"]]
                ax.plot(xs_p, ys_p, marker="s", color=color, linestyle="--", alpha=0.6,
                        label=f"K={k}% (partial)")

        ax.set_xlabel("Forecast Origin Year (T)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([r["origin_year"] for r in all_results])

    plt.tight_layout()
    out_pdf = output_dir / "forward_backtest_precision_over_time.pdf"
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved backtest plot: {out_pdf}")


# ── Main for one origin year ──────────────────────────────────────────────────

def run_single_origin(
    origin_year: int,
    model_type: str,
    lgbm_params: Optional[Dict[str, Any]],
    rf_params: Optional[Dict[str, Any]],
    feature_cols: List[str],
    split_paths: List[Path],
    panel_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    T = origin_year
    train_range = (2001, T - 1)
    clean_window = (T + LOOKAHEAD_YEARS) <= LAST_LABEL_YEAR

    print("\n" + "=" * 70)
    print(f"BACKTEST [{model_type.upper()}]: origin year T={T}  "
          f"(train 2001–{T-1}, eval {T+1}–{min(T+5, LAST_LABEL_YEAR)})")
    print(f"  Clean 5-year window: {clean_window}")
    print("=" * 70)
    report_memory_usage(f"start T={T}")

    # ── Train historical deployment model ─────────────────────────────────────
    X_tr, y_tr, yr_tr, n_pos_tr, n_neg_tr = load_training_data(
        split_paths, feature_cols, train_range
    )
    print(f"  Training data: {len(y_tr):,} rows ({n_pos_tr:,} pos / {n_neg_tr:,} neg)")

    if model_type == "lgbm":
        n_est = int(lgbm_params.get("n_estimators", N_ESTIMATORS_LOCKED_LGBM))
        model = _train_lgbm(X_tr, y_tr, yr_tr, lgbm_params, n_est, train_range)
    else:  # rf
        model = _train_rf(X_tr, y_tr, rf_params)

    del X_tr, y_tr, yr_tr
    gc.collect()
    report_memory_usage(f"after training T={T}")

    # ── Load year-T inference set + 5-year labels ─────────────────────────────
    X_inf, row_col, label_5yr, evaluable, xy = load_inference_rows_and_wdpa(
        panel_path, feature_cols, T
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    print(f"\nScoring {len(X_inf):,} pixels…")
    if model_type == "lgbm":
        n_est_inf = int(lgbm_params.get("n_estimators", N_ESTIMATORS_LOCKED_LGBM))
        y_proba = model.predict(X_inf, num_iteration=n_est_inf).astype(np.float32)
    else:  # rf
        y_proba = model.predict_proba(X_inf)[:, 1].astype(np.float32)

    del model, X_inf
    gc.collect()

    # ── Evaluate on evaluable pixels ──────────────────────────────────────────
    result: Dict[str, Any] = {
        "origin_year": T,
        "train_years": list(train_range),
        "model_type": model_type,
        "clean_5yr_window": bool(clean_window),
        "n_pixels_scored": int(len(y_proba)),
        "n_evaluable": int(evaluable.sum()),
        "n_pos_evaluable": int(label_5yr[evaluable].sum()),
        "metrics": None,
    }

    if evaluable.sum() > 0 and label_5yr[evaluable].sum() > 0:
        window_label = (
            f"T={T}, window [{T+1}–{min(T+LOOKAHEAD_YEARS, LAST_LABEL_YEAR)}]"
            + ("" if clean_window else " (PARTIAL)")
        )
        metrics = compute_backtest_metrics(label_5yr[evaluable], y_proba[evaluable], window_label)
        result["metrics"] = metrics
    else:
        print(f"  T={T}: no evaluable positive labels — skipping metrics")

    # False-positive spatial analysis (only for clean 5-year windows)
    if clean_window:
        create_false_positive_map(y_proba, label_5yr, evaluable, xy, T, output_dir)

    del y_proba, label_5yr, evaluable, row_col
    gc.collect()

    # Save individual result
    out_path = output_dir / f"forward_backtest_T{T}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_backtest_results(output_dir: Path) -> List[Dict]:
    results = []
    for T in ORIGIN_YEARS:
        p = output_dir / f"forward_backtest_T{T}.json"
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))
    results.sort(key=lambda r: r["origin_year"])
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Re-import config after runner.py reload
    from scripts.regions.shared.forward.config import DATA_SUBDIR, MODEL_PREFIX, OUTPUTS_SUBDIR  # noqa: F401

    model_type = os.environ.get("PA3030_FORWARD_MODEL_TYPE", "lgbm").strip().lower()

    parser = argparse.ArgumentParser(description="Forward prediction backtesting")
    parser.add_argument("--origin-year", type=int, choices=ORIGIN_YEARS, default=None,
                        help="Origin year T to process. Defaults to SLURM_ARRAY_TASK_ID index.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate individual backtest JSONs into summary + plot.")
    # model-type can also be passed via env (preferred when called from runner.py)
    parser.add_argument("--model-type", type=str, choices=["lgbm", "rf"], default=None,
                        help="Model type (overrides PA3030_FORWARD_MODEL_TYPE env var).")
    args = parser.parse_args()

    if args.model_type is not None:
        model_type = args.model_type.strip().lower()

    repo_root   = get_repo_root()
    forward_dir = repo_root / f"outputs/{OUTPUTS_SUBDIR}/results/forward"
    output_dir  = forward_dir / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Aggregate mode ────────────────────────────────────────────────────────
    if args.aggregate:
        print(f"Aggregating backtest results ({model_type.upper()})…")
        all_results = aggregate_backtest_results(output_dir)
        if not all_results:
            print("No backtest JSONs found — run individual origins first.")
            sys.exit(1)
        out_json = output_dir / "forward_backtest_results.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved: {out_json}")
        plot_backtest_precision(all_results, output_dir, model_type)
        return

    # ── Single origin ─────────────────────────────────────────────────────────
    if args.origin_year is not None:
        origin_year = args.origin_year
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        origin_year = ORIGIN_YEARS[task_id % len(ORIGIN_YEARS)]

    print(f"Processing origin year T={origin_year} [{model_type.upper()}]")

    params_path = resolve_best_params_json(MODEL_PREFIX, DATA_SUBDIR, model_type)
    if model_type == "lgbm":
        import lightgbm  # noqa: F401 — ensure available
        best_params_lgbm = load_best_params_lgbm(params_path, NUM_THREADS)
        best_params_rf   = None
    else:
        best_params_lgbm = None
        best_params_rf   = load_best_params_rf(params_path, NUM_THREADS)

    # Resolve data sources
    split_paths = resolve_split_parquets(DATA_SUBDIR)
    panel_path  = resolve_panel(DATA_SUBDIR)
    print(f"  panel:        {panel_path}")
    print(f"  split_paths:  {[str(p) for p in split_paths]}")

    # Feature columns from train parquet schema (or panel fallback)
    source = split_paths[0] if split_paths else panel_path
    schema = pq.ParquetFile(source).schema_arrow
    feature_cols = [
        name for name, fld in zip(schema.names, schema)
        if (pa.types.is_integer(fld.type) or pa.types.is_floating(fld.type))
        and name not in EXCLUDE_COLS
    ]
    print(f"  feature_cols: {len(feature_cols)}")

    run_single_origin(
        origin_year, model_type, best_params_lgbm, best_params_rf,
        feature_cols, split_paths, panel_path, output_dir,
    )


if __name__ == "__main__":
    main()
