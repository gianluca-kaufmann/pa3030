#!/usr/bin/env python3
"""Stage 0b: Deployment Model Training (annual hazard, 2001-2024).

Trains LightGBM on ALL labeled years (2001-2024) under the annual hazard target using the
hyperparameters and n_estimators locked in lgbm_best_params.json.  No CV,
no test split — the full labeled window is used for training.

Key differences from model3_LGBM (evaluation model):
  - TRAIN_YEARS = (2001, 2019)  instead of (2001, 2016)
  - cv_mode = 'none' (always)   — num_boost_round = best_params['n_estimators'] = 2555
  - scale_pos_weight recomputed from data (NOT locked from JSON)
  - No test-set scoring

Deployment calibration split:
  Fits Platt + isotonic calibrators using an auxiliary model trained on
  2001-2016 that scores AUX_CALIB_YEARS held-out pixels.  Because the deployment
  model is trained on all labeled years there is no proper held-out set;
  the auxiliary model provides a practical approximation (see plan for the
  methodological disclosure language).

Outputs:
  data/se_asia/ml/models/model3_lgbm_deployment_{timestamp}.pkl
      (resolve_ml_models_dir — $SCRATCH/data/.../ml/models on Euler when SCRATCH is set)
      Pickle dict:
        {
          'model':      lgb.Booster  (deployment model, 2001-2024 annual hazard),
          'platt':      LogisticRegression fitted on aux-model 2022-2024 scores,
          'isotonic':   IsotonicRegression fitted on aux-model 2022-2024 scores,
          'feature_cols': list[str],
          'timestamp':  str,
          'metadata':   dict,
        }
  outputs/se_asia/results/forward/model3_lgbm_deployment_{timestamp}.txt
      Run log (resolve_forward_dir — $SCRATCH/outputs/.../forward on Euler when set).
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

from scripts.regions.shared.forward.config import resolve_forward_dir, resolve_ml_models_dir  # noqa: E402
from scripts.regions.shared.training.feature_guard import check_feature_denylist  # noqa: E402
from scripts.regions.shared.training.utils import (  # noqa: E402
    WandbRunLogger,
    Tee,
    compute_metrics,
    compute_precision_at_k,
    compute_year_weights,
    extract_features_pyarrow_to_numpy,
    get_repo_root,
    report_memory_usage,
)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
EXCLUDE_COLS = {
    "transition_01", "transition_01_win5", "WDPA_b1", "WDPA_prev", "WDPA",
    "x", "y", "row", "col", "year",
}
TARGET_COL = "transition_01"  # Annual hazard target
WDPA_LAST_YEAR = 2024  # Last year with WDPA data observed in the panel

DEPLOY_TRAIN_YEARS = (2001, WDPA_LAST_YEAR)   # full deployment training window (annual hazard)
AUX_TRAIN_YEARS   = (2001, 2021)              # auxiliary model for calibration
AUX_CALIB_YEARS   = (2022, WDPA_LAST_YEAR)    # calibration held-out set

FIXED_PARAMS = {
    "random_state": RANDOM_STATE,
    "boosting_type": "gbdt",
    "objective": "binary",
    "verbose": -1,
}
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "200_000"))
# MAX_NEG_TRAIN is no longer used for LGBM training (Sequence API removes the need
# to allocate a full feature matrix, eliminating the OOM that prompted capping).
# Kept as a no-op env var so existing SLURM scripts that export it don't break.
MAX_NEG_TRAIN = int(os.environ.get("MAX_NEG_TRAIN", "0"))
# ─────────────────────────────────────────────────────────────────────────────


def get_num_threads() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n = int(slurm_cpus)
            if n > 0:
                return n
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def configure_threading_env(n: int) -> None:
    s = str(max(1, n))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = s


NUM_THREADS = get_num_threads()
configure_threading_env(NUM_THREADS)

import lightgbm as lgb  # noqa: E402  (must come after thread-env config)


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_parquet(filename: str) -> Path:
    """Locate a split parquet file (prefer $SCRATCH)."""
    repo_root = get_repo_root()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    candidates: list[Path] = []
    if scratch is not None:
        candidates += [
            scratch / f"outputs/se_asia/results/main/{filename}",
            scratch / f"data/se_asia/ml/main/{filename}",
            scratch / f"outputs/se_asia/results/{filename}",
            scratch / f"data/se_asia/ml/{filename}",
        ]
    candidates += [
        repo_root / f"outputs/se_asia/results/main/{filename}",
        repo_root / f"data/se_asia/ml/main/{filename}",
        repo_root / f"outputs/se_asia/results/{filename}",
        repo_root / f"data/se_asia/ml/{filename}",
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

    candidates: list[Path] = [script_dir / "lgbm_best_params.json"]
    if scratch is not None:
        candidates += [
            scratch / "scripts/regions/se_asia/5_training/lgbm_best_params.json",
            scratch / "lgbm_best_params.json",
        ]
    candidates += [
        repo_root / "scripts/regions/se_asia/5_training/lgbm_best_params.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


# ── Hyperparameter loading ────────────────────────────────────────────────────

def load_best_params(params_path: Optional[Path]) -> Dict[str, Any]:
    """Load + merge hyperparameters from JSON; pop scale_pos_weight (always recomputed)."""
    if params_path is None:
        print("  lgbm_best_params.json not found — using default guardrails")
        params = {
            **FIXED_PARAMS,
            "max_depth": 8, "num_leaves": 255, "min_child_samples": 50,
            "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7,
            "learning_rate": 0.03, "n_estimators": 3000,
            "reg_alpha": 1.0, "reg_lambda": 1.0,
            "metric": "average_precision", "num_threads": NUM_THREADS,
        }
        return params

    with open(params_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "best_params" in data:
        # Nested format (standard artifact from tuning)
        best = data["best_params"].copy()
        fixed = data.get("fixed_params", {})
        params = {**FIXED_PARAMS, **fixed, **best}
    else:
        params = {**FIXED_PARAMS, **(data.copy() if isinstance(data, dict) else {})}

    # Guardrails — only fill missing keys
    guardrails = {
        "max_depth": 8, "num_leaves": 255, "min_child_samples": 50,
        "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7,
        "learning_rate": 0.03, "n_estimators": 3000,
        "reg_alpha": 1.0, "reg_lambda": 1.0,
        "metric": "average_precision", "num_threads": NUM_THREADS,
    }
    for k, v in guardrails.items():
        if k not in params:
            params[k] = v

    # Always force thread count from SLURM allocation
    params["num_threads"] = NUM_THREADS
    return params


# ── Lazy Sequence loading (LightGBM Sequence API) ─────────────────────────────

@dataclass
class _BatchRecord:
    """Per-batch metadata collected during the label-loading pass."""
    path_idx: int           # index into the paths list
    parquet_batch_idx: int  # sequential batch number within that Parquet file
    mask_bits: np.ndarray   # bit-packed boolean row-filter (uint8)
    mask_len: int           # original (unpacked) boolean mask length
    mask_true_count: int    # number of selected rows in mask


def _pack_mask(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Bit-pack a boolean mask to reduce RAM by ~8x."""
    mask_bool = np.asarray(mask, dtype=np.bool_)
    packed = np.packbits(mask_bool, bitorder="little")
    return packed, int(mask_bool.size), int(mask_bool.sum())


def _unpack_mask(mask_bits: np.ndarray, mask_len: int) -> np.ndarray:
    """Restore boolean mask from packed representation."""
    return np.unpackbits(mask_bits, bitorder="little", count=mask_len).astype(np.bool_)


class _ParquetFeatureSequence(getattr(lgb, "Sequence", object)):
    """LightGBM row-based `lgb.Sequence` backed by lazy Parquet reads."""

    def __init__(
        self,
        batch_records: List[_BatchRecord],
        paths: List[Path],
        feature_cols: List[str],
        batch_size: int,
    ) -> None:
        self._records = batch_records
        self._paths = paths
        self._feature_cols = feature_cols
        self._batch_size = batch_size
        self.batch_size = int(batch_size)

        self._iter_path_idx: Optional[int] = None
        self._iter_pf = None
        self._iter_gen = None
        self._iter_pos: int = 0

        self._n_rows = int(sum(r.mask_true_count for r in self._records))
        self._n_cols = len(self._feature_cols)
        self._row_starts = np.empty(len(self._records) + 1, dtype=np.int64)
        self._row_starts[0] = 0
        csum = 0
        for i, r in enumerate(self._records):
            csum += int(r.mask_true_count)
            self._row_starts[i + 1] = csum

        self._cache_rec_idx: Optional[int] = None
        self._cache_X: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self._n_rows

    def _get_record_features(self, rec_idx: int) -> np.ndarray:
        if self._cache_rec_idx == rec_idx and self._cache_X is not None:
            return self._cache_X

        rec = self._records[rec_idx]

        if (self._iter_path_idx != rec.path_idx
                or self._iter_pos > rec.parquet_batch_idx):
            self._open_file(rec.path_idx)

        while self._iter_pos < rec.parquet_batch_idx:
            next(self._iter_gen)
            self._iter_pos += 1

        batch = next(self._iter_gen)
        self._iter_pos += 1

        feat_tbl = batch.select(self._feature_cols)
        mask = _unpack_mask(rec.mask_bits, rec.mask_len)
        # LightGBM's Sequence path expects float64 ("double") arrays during sampling / init.
        X_rec = extract_features_pyarrow_to_numpy(feat_tbl, mask).astype(np.float64, copy=False)

        self._cache_rec_idx = rec_idx
        self._cache_X = X_rec
        return X_rec

    def __getitem__(self, idx: int | slice | List[int]) -> np.ndarray:
        if isinstance(idx, slice):
            start = 0 if idx.start is None else int(idx.start)
            stop = self._n_rows if idx.stop is None else int(idx.stop)
            step = 1 if idx.step is None else int(idx.step)
            if step != 1:
                raise ValueError("lgb.Sequence requires step=1 for slice access")

            if start < 0:
                start += self._n_rows
            if stop < 0:
                stop += self._n_rows

            start = max(0, start)
            stop = min(self._n_rows, stop)
            if stop <= start:
                return np.empty((0, self._n_cols), dtype=np.float64)

            out = np.empty((stop - start, self._n_cols), dtype=np.float64)

            rec_start = bisect.bisect_right(self._row_starts, start) - 1
            rec_end = bisect.bisect_right(self._row_starts, stop - 1) - 1

            write_pos = 0
            for rec_idx in range(rec_start, rec_end + 1):
                rec_lo = int(self._row_starts[rec_idx])
                X_rec = self._get_record_features(rec_idx)
                lo_in_rec = max(0, start - rec_lo)
                hi_in_rec = min(X_rec.shape[0], stop - rec_lo)
                part = X_rec[lo_in_rec:hi_in_rec]
                k = part.shape[0]
                if k:
                    out[write_pos:write_pos + k] = part
                    write_pos += k

            return out

        if isinstance(idx, list):
            return np.stack([self[int(i)] for i in idx], axis=0)

        row_idx = int(idx)
        if row_idx < 0:
            row_idx += self._n_rows
        if row_idx < 0 or row_idx >= self._n_rows:
            raise IndexError("Sequence index out of range")

        rec_idx = bisect.bisect_right(self._row_starts, row_idx) - 1
        rec_lo = int(self._row_starts[rec_idx])
        local_idx = row_idx - rec_lo
        X_rec = self._get_record_features(rec_idx)
        return X_rec[int(local_idx)]

    def _open_file(self, path_idx: int) -> None:
        self._iter_pf = pq.ParquetFile(self._paths[path_idx])
        self._iter_gen = self._iter_pf.iter_batches(
            batch_size=self._batch_size,
            columns=self._feature_cols,
            use_threads=True,
        )
        self._iter_path_idx = path_idx
        self._iter_pos = 0

    def close(self) -> None:
        """Release open file handles."""
        self._iter_pf = None
        self._iter_gen = None
        self._cache_rec_idx = None
        self._cache_X = None


# ── Data loading ──────────────────────────────────────────────────────────────

def _stream_data(
    paths: List[Path],
    feature_cols: List[str],
    year_range: Tuple[int, int],
    name: str = "",
    max_neg: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """Two-pass loader with optional negative cap (reservoir) and temporal years.

    Pass 1 records uncapped n_pos_total / n_neg_total for scale_pos_weight.
    """
    t0 = time.time()
    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev"]

    print(f"  Pass 1: counting rows ({name}, years {year_range[0]}–{year_range[1]})…")
    n_pos_total = 0
    n_neg_total = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr = batch["year"].to_numpy(zero_copy_only=False)
                wp = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
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

    X = np.empty((n_samples, len(feature_cols)), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int8)
    years_arr = np.empty(n_samples, dtype=np.int32)
    idx_pos = 0
    idx_neg = 0
    rng = np.random.RandomState(RANDOM_STATE) if (max_neg > 0 and n_neg_total > max_neg) else None
    neg_seen = 0
    _mile = -1

    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr = batch["year"].to_numpy(zero_copy_only=False)
                wp = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
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
                yr_sel = yr[mask].astype(np.int32)

                pos_mask_b = y_b == 1
                neg_mask_b = y_b == 0

                np_b = int(pos_mask_b.sum())
                if np_b > 0:
                    X[idx_pos:idx_pos + np_b] = X_b[pos_mask_b]
                    y[idx_pos:idx_pos + np_b] = 1
                    years_arr[idx_pos:idx_pos + np_b] = yr_sel[pos_mask_b]
                    idx_pos += np_b

                X_neg = X_b[neg_mask_b]
                yr_neg = yr_sel[neg_mask_b]
                nn_b = len(X_neg)
                if nn_b > 0:
                    if rng is None:
                        X[n_pos_total + idx_neg: n_pos_total + idx_neg + nn_b] = X_neg
                        y[n_pos_total + idx_neg: n_pos_total + idx_neg + nn_b] = 0
                        years_arr[n_pos_total + idx_neg: n_pos_total + idx_neg + nn_b] = yr_neg
                        idx_neg += nn_b
                    else:
                        for i in range(nn_b):
                            neg_seen += 1
                            if idx_neg < n_neg_use:
                                X[n_pos_total + idx_neg] = X_neg[i]
                                y[n_pos_total + idx_neg] = 0
                                years_arr[n_pos_total + idx_neg] = yr_neg[i]
                                idx_neg += 1
                            else:
                                j = rng.randint(0, neg_seen)
                                if j < n_neg_use:
                                    X[n_pos_total + j] = X_neg[i]
                                    years_arr[n_pos_total + j] = yr_neg[i]

                loaded = idx_pos + idx_neg
                pct = loaded * 100 // n_samples if n_samples > 0 else 0
                ms = pct // 25
                if ms > _mile:
                    _mile = ms
                    print(f"    {pct}% — {loaded:,}/{n_samples:,}")

                del batch, tgt, yr, wp, mask, feat_tbl, X_b, y_b
        finally:
            del pf
        gc.collect()

    actual = idx_pos + idx_neg
    X = X[:actual]
    y = y[:actual]
    years_arr = years_arr[:actual]
    n_pos_ld = int(y.sum())
    n_neg_ld = actual - n_pos_ld
    elapsed = time.time() - t0
    print(f"  Loaded {actual:,} rows in {elapsed:.1f}s ({n_neg_ld:,} neg / {n_pos_ld:,} pos)")
    return X, y, years_arr, n_pos_ld, n_neg_ld, n_pos_total, n_neg_total


def _stream_labels(
    paths: List[Path],
    feature_cols: List[str],
    year_range: Tuple[int, int],
    name: str = "",
) -> Tuple[np.ndarray, np.ndarray, int, int, List[_BatchRecord]]:
    """Two-pass label-only loader — feature matrix is never allocated.

    Pass 1: count filtered rows (for array pre-allocation + scale_pos_weight).
    Pass 2: fill y + years_arr; record per-batch metadata (path, batch index,
            row-filter mask) so that _ParquetFeatureSequence can re-read X
            lazily during lgb.Dataset construction.

    Returns (y, years_arr, n_pos_total, n_neg_total, batch_records).
    Peak RAM: y + years_arr + masks — no feature matrix allocated.
    """
    t0 = time.time()
    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev"]

    print(f"  Pass 1: counting rows ({name}, years {year_range[0]}–{year_range[1]})…")
    n_pos_total = 0
    n_neg_total = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr = batch["year"].to_numpy(zero_copy_only=False)
                wp = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
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

    n_samples = n_pos_total + n_neg_total
    if n_samples == 0:
        raise ValueError(f"No samples in {name} for years {year_range}")
    print(f"    {n_samples:,} samples ({n_pos_total:,} pos / {n_neg_total:,} neg) — full risk set")

    y = np.empty(n_samples, dtype=np.int8)
    years_arr = np.empty(n_samples, dtype=np.int32)
    batch_records: List[_BatchRecord] = []
    row_idx = 0
    _mile = -1

    for path_idx, path in enumerate(paths):
        parquet_batch_idx = 0
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr = batch["year"].to_numpy(zero_copy_only=False)
                wp = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
                mask = (
                    ~null_mask
                    & (wp == 0)
                    & (yr >= year_range[0]) & (yr <= year_range[1])
                )
                if mask.any():
                    n_b = int(mask.sum())
                    y[row_idx:row_idx + n_b] = (tgt[mask] > 0).astype(np.int8)
                    years_arr[row_idx:row_idx + n_b] = yr[mask].astype(np.int32)
                    mask_bits, mask_len, mask_true_count = _pack_mask(mask)
                    batch_records.append(_BatchRecord(
                        path_idx=path_idx,
                        parquet_batch_idx=parquet_batch_idx,
                        mask_bits=mask_bits,
                        mask_len=mask_len,
                        mask_true_count=mask_true_count,
                    ))
                    row_idx += n_b

                    pct = row_idx * 100 // n_samples
                    ms = pct // 25
                    if ms > _mile:
                        _mile = ms
                        print(f"    {pct}% — {row_idx:,}/{n_samples:,}")

                parquet_batch_idx += 1
                del batch, tgt_arr, tgt, null_mask, yr, wp
        finally:
            del pf
        gc.collect()

    elapsed = time.time() - t0
    print(
        f"  Labels indexed in {elapsed:.1f}s — {n_samples:,} rows "
        f"({n_neg_total:,} neg / {n_pos_total:,} pos), "
        f"{len(batch_records)} feature batches queued (X not in RAM)"
    )
    return y, years_arr, n_pos_total, n_neg_total, batch_records


# ── LightGBM training helper ──────────────────────────────────────────────────

def _build_train_params(best_params: Dict[str, Any], scale_pos_weight: float) -> Dict[str, Any]:
    p = {**best_params, "is_unbalance": False}
    for key in ("num_boost_round", "n_estimators", "n_jobs",
                "sampling_strategy", "replacement", "bootstrap", "class_weight",
                "early_stopping_rounds", "early_stopping", "scale_pos_weight"):
        p.pop(key, None)
    p["scale_pos_weight"] = scale_pos_weight
    p.setdefault("num_threads", NUM_THREADS)
    return p


def _train_lgb(
    X: "np.ndarray | _ParquetFeatureSequence",
    y: np.ndarray,
    years_arr: np.ndarray,
    best_params: Dict[str, Any],
    num_boost_round: int,
    min_year: int,
    max_year: int,
    label: str = "",
    scale_pos_weight_override: float | None = None,
) -> lgb.Booster:
    if scale_pos_weight_override is not None:
        spw = float(scale_pos_weight_override)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"  scale_pos_weight = {spw:.4f} (full pass-1 counts; loaded {n_neg:,} neg / {n_pos:,} pos)")
    else:
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        spw = n_neg / max(n_pos, 1)
        print(f"  scale_pos_weight = {spw:.4f} = {n_neg:,} / {n_pos:,}")

    weights = compute_year_weights(years_arr, min_year=min_year, max_year=max_year)
    train_params = _build_train_params(best_params, spw)

    print(f"  Training {label} — {len(y):,} samples, {num_boost_round} rounds")
    for k in sorted(train_params):
        print(f"    {k}: {train_params[k]}")

    ds = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)
    if hasattr(X, "close"):
        X.close()
    del X, y, years_arr, weights
    gc.collect()

    t0 = time.time()
    model = lgb.train(train_params, ds, num_boost_round=num_boost_round,
                      callbacks=[lgb.log_evaluation(200)])
    print(f"  Done in {time.time() - t0:.1f}s")
    del ds
    gc.collect()
    return model


# ── Calibration ───────────────────────────────────────────────────────────────

def fit_deployment_calibrators(
    all_paths: List[Path],
    feature_cols: List[str],
    best_params: Dict[str, Any],
    num_boost_round: int,
) -> Tuple[LogisticRegression, IsotonicRegression]:
    """Fit Platt + isotonic calibrators using the deployment calibration split.

    Trains an auxiliary model on AUX_TRAIN_YEARS (2001–2016) with the same
    locked hyperparameters, scores AUX_CALIB_YEARS (2017–2019) held-out
    pixels, then fits calibrators on those out-of-sample predictions.

    Methodological note: the calibrators are fitted on an *auxiliary* model's
    score distribution, not the deployment model's.  This is an unavoidable
    approximation (see plan §Stage 0b for disclosure language).
    """
    print("\n" + "=" * 70)
    print("DEPLOYMENT CALIBRATION SPLIT")
    print(f"  Auxiliary model train: {AUX_TRAIN_YEARS[0]}–{AUX_TRAIN_YEARS[1]}")
    print(f"  Calibration held-out:  {AUX_CALIB_YEARS[0]}–{AUX_CALIB_YEARS[1]}")
    print("=" * 70)

    # Train auxiliary model on 2001-2016 (lazy-loaded; no X matrix in RAM)
    y_aux, yr_aux, n_pos_aux_f, n_neg_aux_f, aux_records = _stream_labels(
        all_paths, feature_cols, AUX_TRAIN_YEARS, "aux_train"
    )
    X_aux = _ParquetFeatureSequence(aux_records, all_paths, feature_cols, BATCH_SIZE)
    spw_aux = n_neg_aux_f / max(n_pos_aux_f, 1)
    aux_model = _train_lgb(
        X_aux, y_aux, yr_aux, best_params, num_boost_round,
        min_year=AUX_TRAIN_YEARS[0], max_year=AUX_TRAIN_YEARS[1],
        label="auxiliary calibration model",
        scale_pos_weight_override=spw_aux,
    )

    # Score AUX_CALIB_YEARS held-out pixels (small set — full X in RAM is fine)
    print("\nScoring calibration held-out set…")
    X_cal, y_cal, _, n_pos_cal, n_neg_cal, _, _ = _stream_data(
        all_paths, feature_cols, AUX_CALIB_YEARS, "aux_calib", max_neg=0
    )
    y_cal_proba = aux_model.predict(X_cal, num_iteration=num_boost_round).astype(np.float32)
    del aux_model, X_cal
    gc.collect()

    cal_prauc = average_precision_score(y_cal, y_cal_proba)
    cal_rocauc = roc_auc_score(y_cal, y_cal_proba)
    p1 = compute_precision_at_k(y_cal, y_cal_proba, 1)
    print(f"  Aux model on calib set: PR-AUC {cal_prauc:.4f}, ROC-AUC {cal_rocauc:.4f}, P@1% {p1:.4f}")

    # Fit Platt (logistic on logit) calibrator
    print("\nFitting Platt calibrator…")
    eps = 1e-7
    logit_proba = np.log(np.clip(y_cal_proba, eps, 1 - eps) /
                         (1 - np.clip(y_cal_proba, eps, 1 - eps))).reshape(-1, 1)
    platt = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="lbfgs")
    platt.fit(logit_proba, y_cal)

    # Fit isotonic calibrator
    print("Fitting isotonic calibrator…")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(y_cal_proba, y_cal)

    # Quick check: Brier scores
    from sklearn.metrics import brier_score_loss
    brier_raw = brier_score_loss(y_cal, y_cal_proba)
    platt_cal = platt.predict_proba(logit_proba)[:, 1]
    iso_cal = iso.predict(y_cal_proba)
    brier_platt = brier_score_loss(y_cal, platt_cal)
    brier_iso = brier_score_loss(y_cal, iso_cal)
    print(f"  Brier score — raw: {brier_raw:.6f}, Platt: {brier_platt:.6f}, isotonic: {brier_iso:.6f}")

    del y_cal, y_cal_proba, logit_proba, platt_cal, iso_cal
    gc.collect()
    return platt, iso


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    start = time.time()
    repo_root = get_repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wb = WandbRunLogger(
        project="forward",
        run_name=f"deploy_lgbm_se_asia_{timestamp}",
        config={
            "region": "se_asia",
            "model_type": "lgbm",
            "forward_stage": "deployment",
        },
    )
    wb.start()
    wb.log({"deployment/stage": "start"})

    print("=" * 70)
    print(f"MODEL 3 LGBM — DEPLOYMENT MODEL (annual hazard, {DEPLOY_TRAIN_YEARS[0]}-{DEPLOY_TRAIN_YEARS[1]})")
    print("=" * 70)

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Under annual hazard, deployment training spans the full WDPA window
    # (2001-WDPA_LAST_YEAR). The eval-pipeline split files only cover 2001-2019,
    # so we read from merged_panel_final.parquet directly. train.parquet still
    # provides the schema reference (feature column list).
    train_path           = resolve_parquet("train.parquet")
    merged_panel_path    = resolve_parquet("merged_panel_final.parquet")
    all_paths            = [merged_panel_path]
    params_path    = resolve_best_params_json()

    model_dir  = resolve_ml_models_dir(repo_root, "se_asia")
    output_dir = resolve_forward_dir(repo_root, "se_asia")
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  schema reference: {train_path}")
    print(f"  data source:      {merged_panel_path}")
    print(f"  params:    {params_path or '(default guardrails)'}")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    best_params = load_best_params(params_path)
    num_boost_round = int(best_params.get("n_estimators", best_params.get("num_boost_round", 3000)))
    print(f"\n  num_boost_round (locked from JSON): {num_boost_round}")

    # ── Determine feature columns from train parquet schema ───────────────────
    schema = pq.ParquetFile(train_path).schema_arrow
    all_numeric = [
        name for name, field in zip(schema.names, schema)
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
    ]
    feature_cols = [c for c in all_numeric if c not in EXCLUDE_COLS]
    check_feature_denylist(feature_cols, context="se_asia/lgbm/deployment")
    print(f"\n  Feature columns: {len(feature_cols)}")
    wb.log({"deployment/n_features": float(len(feature_cols)), "deployment/stage": "features_ready"})

    # ── Step 1: Train deployment model on full hazard window ─────────────────
    print("\n" + "=" * 70)
    print(f"STEP 1: TRAINING DEPLOYMENT MODEL ({DEPLOY_TRAIN_YEARS[0]}-{DEPLOY_TRAIN_YEARS[1]})")
    print("=" * 70)
    report_memory_usage("before loading deployment training labels")

    y_dep, yr_dep, dep_pos, dep_neg, dep_records = _stream_labels(
        all_paths, feature_cols, DEPLOY_TRAIN_YEARS, "deployment_train"
    )
    X_dep = _ParquetFeatureSequence(dep_records, all_paths, feature_cols, BATCH_SIZE)
    report_memory_usage("after indexing deployment training data (X not in RAM)")

    deploy_spw = dep_neg / max(dep_pos, 1)
    deploy_model = _train_lgb(
        X_dep, y_dep, yr_dep, best_params, num_boost_round,
        min_year=DEPLOY_TRAIN_YEARS[0], max_year=DEPLOY_TRAIN_YEARS[1],
        label="deployment model",
        scale_pos_weight_override=deploy_spw,
    )
    report_memory_usage("after training deployment model")
    wb.log(
        {
            "deployment/deploy_n_pos": float(dep_pos),
            "deployment/deploy_n_neg": float(dep_neg),
            "deployment/scale_pos_weight": float(deploy_spw),
            "deployment/stage": "train_done",
        }
    )

    # ── Step 2: Fit calibrators via auxiliary model ───────────────────────────
    platt_cal, iso_cal = fit_deployment_calibrators(
        all_paths, feature_cols, best_params, num_boost_round
    )
    report_memory_usage("after fitting calibrators")
    wb.log({"deployment/stage": "calibration_done"})

    # ── Step 3: Save model + calibrators ─────────────────────────────────────
    artifact = {
        "model": deploy_model,
        "platt": platt_cal,
        "isotonic": iso_cal,
        "feature_cols": feature_cols,
        "timestamp": timestamp,
        "metadata": {
            "train_years": list(DEPLOY_TRAIN_YEARS),
            "aux_train_years": list(AUX_TRAIN_YEARS),
            "aux_calib_years": list(AUX_CALIB_YEARS),
            "num_boost_round": num_boost_round,
            "n_features": len(feature_cols),
            "deploy_n_pos": dep_pos,
            "deploy_n_neg": dep_neg,
            "deploy_n_pos_full": dep_pos,
            "deploy_n_neg_full": dep_neg,
            "deploy_scale_pos_weight": float(deploy_spw),
            "target_col": TARGET_COL,
            "params_path": str(params_path) if params_path else None,
            "total_time_s": round(time.time() - start, 1),
        },
    }

    model_path = model_dir / f"model3_lgbm_deployment_{timestamp}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(artifact, fh, protocol=5)
    print(f"\nSaved deployment artifact: {model_path}")

    meta = artifact["metadata"]
    wb.log(
        {
            "deployment/deploy_n_pos": float(dep_pos),
            "deployment/deploy_n_neg": float(dep_neg),
            "deployment/deploy_n_pos_full": float(dep_pos),
            "deployment/deploy_n_neg_full": float(dep_neg),
            "deployment/scale_pos_weight": float(deploy_spw),
            "deployment/n_features": float(len(feature_cols)),
            "deployment/total_time_s": float(meta["total_time_s"]),
            "deployment/stage": "done",
        }
    )
    wb.finish()

    # Print summary
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model:           LightGBM deployment ({DEPLOY_TRAIN_YEARS[0]}-{DEPLOY_TRAIN_YEARS[1]})")
    print(f"  Features:        {len(feature_cols)}")
    print(f"  Train samples:   {dep_neg + dep_pos:,} ({dep_pos:,} pos / {dep_neg:,} neg)")
    print(f"  num_boost_round: {num_boost_round}")
    print(f"  scale_pos_weight:{deploy_spw:.4f}")
    print(f"  Calibrators:     Platt + isotonic (deployment calibration split)")
    print(f"  Artifact:        {model_path}")
    print(f"  Total time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    repo_root = get_repo_root()
    output_dir = resolve_forward_dir(repo_root, "se_asia")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"model3_lgbm_deployment_{timestamp}.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nLog saved: {log_path}")
