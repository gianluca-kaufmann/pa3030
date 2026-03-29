#!/usr/bin/env python3
"""Stage 0c: Pseudo-forecast backtesting.

Validates the forward prediction methodology by simulating the analysis at
one historical time point: T=2019.

For origin year T=2019:
  1. Train a historical deployment model on 2001–2014 with locked hyperparams.
     (Training cutoff = T − LOOKAHEAD_YEARS = 2019 − 5 = 2014, mirroring the
     real deployment: last training year has a complete 5-yr lookahead window.)
  2. Score year-2019 feature rows for WDPA_prev==0 AND WDPA==0 pixels from
     merged_panel_final.parquet.
  3. Reconstruct 5-year window actuals from the WDPA column (years 2020–2024)
     — NOT from transition_01_win5 (which may be absent or unreliable).
  4. Evaluate: Precision@1/5/10%, Lift@1/5/10%, Forecast Capture Rate.

Methodological alignment with the real deployment (5-year gap in both cases):
  Real forward:    train 2001–2019, score 2024, predict 2025–2029  (gap: 5 yrs)
  Backtest T=2019: train 2001–2014, score 2019, eval    2020–2024  (gap: 5 yrs)

Note on LAST_LABEL_YEAR vs WDPA_LAST_YEAR:
  LAST_LABEL_YEAR=2019 is a *training* right-censoring boundary — it prevents
  model training from using labels whose 5-year lookahead window extends beyond
  the available WDPA data. It does NOT limit what can be evaluated in the
  backtest. For evaluation, the correct bound is WDPA_LAST_YEAR=2024.

Model-type support:
  lgbm  — trains lgb.Booster with locked hyperparams; temporal weighting applied.
  rf    — trains RandomForestClassifier with class_weight="balanced_subsample";
          NO temporal weighting (consistent with existing RF training scripts).

Outputs (per origin T):
  outputs/{region}/results/forward/{model_type}/forward_backtest_T{T}.json
Aggregated (after all origins complete):
  outputs/{region}/results/forward/{model_type}/forward_backtest_results.json
  outputs/{region}/results/forward/{model_type}/forward_backtest_precision_over_time.pdf
"""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
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
    resolve_forward_dir,
)
from scripts.regions.shared.training.utils import (  # noqa: E402
    WandbRunLogger,
    compute_precision_at_k,
    compute_recall_at_k,
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


# ── False-positive map (same visual language as forward/results_core maps) ────

FP_MAP_DPI = 300


def create_false_positive_map(
    y_proba: np.ndarray,
    label_5yr: np.ndarray,
    evaluable: np.ndarray,
    xy: dict,
    origin_year: int,
    output_dir: Path,
    repo_root: Path,
    baseline: Dict[str, Any],
) -> None:
    """Map TP/FP in the top-5% risk zone (Web Mercator, backbone underlay, 1 km raster).

    Only runs when x/y are in the panel (EPSG:3857). Matches forward results maps:
    no country outlines, ``FORWARD_PA_HOLE_COLOR`` holes, ``points_to_raster`` @ 1 km.

    Outputs: forward_backtest_T{T}_false_positives.pdf/.png
    """
    if not (xy.get("x") is not None and xy.get("y") is not None):
        print(f"  T={origin_year}: skipping FP map — no x/y coords in panel.")
        return

    from scripts.regions.shared.forward.config import (
        DATA_SUBDIR,
        FORWARD_PA_HOLE_COLOR,
        OUTPUTS_SUBDIR,
        REGION_LABEL,
    )
    from scripts.regions.shared.forward.results_core import resolve_backbone_path_for_plot
    from scripts.regions.shared.results.boundaries import get_region_boundary
    from scripts.regions.shared.results.results_core import (
        _add_latlon_ticks,
        _plot_backbone_background,
        points_to_raster,
    )

    os.environ["PA3030_RESULTS_REGION"] = OUTPUTS_SUBDIR

    x_arr = xy["x"]
    y_arr = xy["y"]
    xm = x_arr.astype(np.float64)
    ym = y_arr.astype(np.float64)

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

    region_gdf = get_region_boundary(None)
    if region_gdf.crs is None:
        region_gdf = region_gdf.set_crs("EPSG:4326", allow_override=True)
    region_proj = region_gdf.to_crs("EPSG:3857")
    proj_bounds = tuple(region_proj.total_bounds.astype(float))
    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

    proj_width = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    pa = proj_height / max(proj_width, 1e-9)
    panel_w = 7.0
    panel_h = panel_w * pa
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h + 0.9))
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
        used_bb = _plot_backbone_background(
            ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
        )
        if not used_bb:
            region_proj.plot(
                ax=ax, color=FORWARD_PA_HOLE_COLOR, edgecolor="none",
                linewidth=0, zorder=0,
            )
        if mask.any():
            grid, gext = points_to_raster(
                xm[mask], ym[mask],
                np.ones(int(mask.sum()), dtype=np.float32),
                target_resolution=1000.0,
                agg_func="max",
                extent_bounds=proj_bounds,
            )
            cmap_fp = mcolors.LinearSegmentedColormap.from_list("fp", ["#ffffff", color])
            ax.imshow(
                np.where(np.isnan(grid), np.nan, 1.0),
                extent=gext,
                cmap=cmap_fp,
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                interpolation="nearest",
                aspect="equal",
                alpha=0.88,
                zorder=2,
            )
        ax.set_xlim(proj_bounds[0], proj_bounds[2])
        ax.set_ylim(proj_bounds[1], proj_bounds[3])
        ax.set_aspect("equal", adjustable="box")
        _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)
        n_px = int(mask.sum())
        ax.text(
            0.02, 0.02, f"{n_px:,} pixels", transform=ax.transAxes,
            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    plt.tight_layout(pad=0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"forward_backtest_T{origin_year}_false_positives"
    for ext in ["pdf", "png"]:
        plt.savefig(f"{stem}.{ext}", dpi=FP_MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}.pdf")


ORIGIN_YEARS = [2019]
# LGBM: N_EST_BACKTEST_LGBM is the fallback num_boost_round only when n_estimators
#       is absent from lgbm_best_params.json (optional faster runs).
# RF:    n_estimators follow rf_best_params.json like deployment training. Set
#       N_EST_BACKTEST_RF only to force a different tree count (e.g. faster jobs).
N_ESTIMATORS_LOCKED_LGBM = int(os.environ.get("N_EST_BACKTEST_LGBM", "2555"))
DEFAULT_RF_N_ESTIMATORS_BACKTEST = 400
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "200_000"))
# RF negative cap: set via SLURM to avoid OOM (sklearn RF needs full X in RAM).
# 0 = no cap (uncapped, matches LGBM behaviour but risks OOM for large regions).
# Recommended values by region: SA=40_000_000, USA=30_000_000, SE Asia=20_000_000.
MAX_NEG_TRAIN = int(os.environ.get("MAX_NEG_TRAIN", "0"))
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
    """Find {model_type}_best_params.json for the given region.

    Search order matches evaluation training scripts: 5_training, 5_training/tuning/,
    4_tuning, then scratch mirrors (and scratch root fallback).
    """
    _ = model_prefix  # reserved for future per-model filenames
    repo_root = get_repo_root()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    json_name = f"{model_type}_best_params.json"
    training_dir = repo_root / f"scripts/regions/{data_subdir}/5_training"
    tuning_dir = repo_root / f"scripts/regions/{data_subdir}/4_tuning"

    candidates: list[Path] = [
        training_dir / json_name,
        training_dir / "tuning" / json_name,
        tuning_dir / json_name,
    ]
    if scratch is not None:
        candidates.extend(
            [
                scratch / f"scripts/regions/{data_subdir}/5_training/{json_name}",
                scratch / f"scripts/regions/{data_subdir}/5_training/tuning/{json_name}",
                scratch / f"scripts/regions/{data_subdir}/4_tuning/{json_name}",
                scratch / json_name,
            ]
        )

    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c
    return None


# ── Hyperparameters ───────────────────────────────────────────────────────────

def _sanitize_rf_param_keys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Strip sklearn Pipeline-style ``rf__`` prefixes (same as model1_RF / deployment)."""
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith("rf__"):
            out[key.replace("rf__", "", 1)] = value
        else:
            out[key] = value
    return out


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
        "n_estimators": DEFAULT_RF_N_ESTIMATORS_BACKTEST,
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
            loaded = {
                **_sanitize_rf_param_keys(data.get("fixed_params", {})),
                **_sanitize_rf_param_keys(data["best_params"]),
            }
        else:
            loaded = _sanitize_rf_param_keys(data) if isinstance(data, dict) else {}
        params = {**defaults, **loaded}
    params["random_state"] = RANDOM_STATE
    params["n_jobs"] = n_jobs
    env_n_est = os.environ.get("N_EST_BACKTEST_RF")
    if env_n_est is not None and str(env_n_est).strip() != "":
        params["n_estimators"] = int(env_n_est)
    return params


# ── Lazy Sequence loading (LightGBM Sequence API) ─────────────────────────────

@dataclass
class _BatchRecord:
    """Per-batch metadata collected during the label-loading pass."""
    path_idx: int           # index into the paths list
    parquet_batch_idx: int  # sequential batch number within that Parquet file
    mask: np.ndarray        # boolean row-filter (length = raw batch row count)


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

        # Row indexing metadata for LightGBM's row-based Sequence protocol.
        self._row_counts = np.array([int(r.mask.sum()) for r in self._records], dtype=np.int64)
        self._n_rows = int(self._row_counts.sum())
        self._n_cols = len(self._feature_cols)
        self._row_starts = np.empty(len(self._records) + 1, dtype=np.int64)
        self._row_starts[0] = 0
        csum = 0
        for i, cnt in enumerate(self._row_counts):
            csum += int(cnt)
            self._row_starts[i + 1] = csum

        self._cache_rec_idx: Optional[int] = None
        self._cache_X: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self._n_rows

    def _get_record_features(self, rec_idx: int) -> np.ndarray:
        """Load and filter the Parquet batch for a given record index."""
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
        # LightGBM's Sequence path expects float64 ("double") arrays during sampling / init.
        X_rec = extract_features_pyarrow_to_numpy(feat_tbl, rec.mask).astype(np.float64, copy=False)
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
            # Dataset.subset() may request an arbitrary set of rows.
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


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_training_data(
    paths: List[Path],
    feature_cols: List[str],
    year_range: Tuple[int, int],
    max_neg: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Stream training data from split parquets into a full feature matrix.

    Used by the RF path (sklearn cannot use the LightGBM Sequence API).
    For LGBM use _load_training_labels + _ParquetFeatureSequence instead.

    max_neg > 0: reservoir-sample negatives to at most max_neg rows (reduces RAM
    at the cost of a smaller negative sample — appropriate for RF with 128 GB limit).
    max_neg == 0: load all negatives (full risk set, may OOM for large regions).

    Pass 1 counts rows for pre-allocation; Pass 2 fills (positives first, then negatives).
    """
    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev"]
    year_lo, year_hi = year_range

    # ── Pass 1: count positives and negatives for pre-allocation ─────────────
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
                mask = (~null_mask
                        & (wp == 0)
                        & (yr >= year_lo) & (yr <= year_hi))
                if mask.any():
                    n_pos_total += int((tgt[mask] > 0).sum())
                    n_neg_total += int((tgt[mask] == 0).sum())
                del batch, tgt_arr, tgt, null_mask, yr, wp, mask
        finally:
            del pf

    if max_neg > 0 and n_neg_total > max_neg:
        n_neg_use = max_neg
        print(f"  Found {n_pos_total:,} pos + {n_neg_total:,} neg "
              f"→ capping neg to {n_neg_use:,} (MAX_NEG_TRAIN)")
    else:
        n_neg_use = n_neg_total

    n_samples = n_pos_total + n_neg_use
    if n_samples == 0:
        raise ValueError(f"No training samples for years {year_range}")

    print(f"  Using {n_samples:,} samples ({n_pos_total:,} pos / {n_neg_use:,} neg)")

    # ── Pass 2: fill pre-allocated arrays (positives first, then negatives) ──
    X         = np.empty((n_samples, len(feature_cols)), dtype=np.float32)
    y         = np.empty(n_samples, dtype=np.int8)
    years_arr = np.empty(n_samples, dtype=np.int32)
    pos_offset = 0
    neg_offset = n_pos_total
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
                mask = (~null_mask
                        & (wp == 0)
                        & (yr >= year_lo) & (yr <= year_hi))
                if not mask.any():
                    del batch, tgt_arr, tgt, null_mask, yr, wp, mask
                    continue

                pos_final     = mask & (tgt > 0)
                neg_final_all = mask & (tgt == 0)

                if pos_final.any():
                    X_b = extract_features_pyarrow_to_numpy(batch.select(feature_cols), pos_final)
                    k   = len(X_b)
                    X[pos_offset:pos_offset + k]         = X_b
                    y[pos_offset:pos_offset + k]         = 1
                    years_arr[pos_offset:pos_offset + k] = yr[pos_final].astype(np.int32)
                    pos_offset += k

                if neg_final_all.any():
                    X_neg = extract_features_pyarrow_to_numpy(batch.select(feature_cols), neg_final_all)
                    yr_neg = yr[neg_final_all].astype(np.int32)
                    nn_b = len(X_neg)
                    if rng is None:
                        X[neg_offset:neg_offset + nn_b]         = X_neg
                        y[neg_offset:neg_offset + nn_b]         = 0
                        years_arr[neg_offset:neg_offset + nn_b] = yr_neg
                        neg_offset += nn_b
                    else:
                        for i in range(nn_b):
                            neg_seen += 1
                            slot = neg_offset - n_pos_total
                            if slot < n_neg_use:
                                X[neg_offset] = X_neg[i]
                                y[neg_offset] = 0
                                years_arr[neg_offset] = yr_neg[i]
                                neg_offset += 1
                            else:
                                j = rng.randint(0, neg_seen)
                                if j < n_neg_use:
                                    X[n_pos_total + j] = X_neg[i]
                                    years_arr[n_pos_total + j] = yr_neg[i]

                del batch, tgt_arr, tgt, null_mask, yr, wp, mask
        finally:
            del pf
        gc.collect()

    n_filled = pos_offset + (neg_offset - n_pos_total)
    if n_filled < n_samples:
        X         = X[:n_filled]
        y         = y[:n_filled]
        years_arr = years_arr[:n_filled]

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return X, y, years_arr, n_pos, n_neg


def _load_training_labels(
    paths: List[Path],
    feature_cols: List[str],
    year_range: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, int, int, List[_BatchRecord]]:
    """Two-pass label-only loader for the LightGBM path — feature matrix never allocated.

    Pass 1: count filtered rows for pre-allocation + scale_pos_weight.
    Pass 2: fill y + years_arr; collect per-batch metadata so that
            _ParquetFeatureSequence can re-read X lazily during lgb.Dataset construction.

    Returns (y, years_arr, n_pos, n_neg, batch_records).
    Peak RAM: y + years_arr + masks — no feature matrix.
    """
    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev"]
    year_lo, year_hi = year_range

    # ── Pass 1: count ─────────────────────────────────────────────────────────
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
                mask = (~null_mask & (wp == 0) & (yr >= year_lo) & (yr <= year_hi))
                if mask.any():
                    n_pos_total += int((tgt[mask] > 0).sum())
                    n_neg_total += int((tgt[mask] == 0).sum())
                del batch, tgt_arr, tgt, null_mask, yr, wp, mask
        finally:
            del pf

    n_samples = n_pos_total + n_neg_total
    if n_samples == 0:
        raise ValueError(f"No training samples for years {year_range}")
    print(f"  Found {n_pos_total:,} pos + {n_neg_total:,} neg → {n_samples:,} total (full risk set)")

    # ── Pass 2: labels + batch metadata (no X) ────────────────────────────────
    y         = np.empty(n_samples, dtype=np.int8)
    years_arr = np.empty(n_samples, dtype=np.int32)
    batch_records: List[_BatchRecord] = []
    row_idx = 0
    _mile = -1

    for path_idx, path in enumerate(paths):
        parquet_batch_idx = 0
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential, use_threads=True):
                tgt_arr  = batch[TARGET_COL]
                null_mask = tgt_arr.is_null().to_numpy(zero_copy_only=False)
                tgt = tgt_arr.to_numpy(zero_copy_only=False)
                yr  = batch["year"].to_numpy(zero_copy_only=False)
                wp  = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
                mask = (~null_mask & (wp == 0) & (yr >= year_lo) & (yr <= year_hi))
                if mask.any():
                    n_b = int(mask.sum())
                    y[row_idx:row_idx + n_b]         = (tgt[mask] > 0).astype(np.int8)
                    years_arr[row_idx:row_idx + n_b] = yr[mask].astype(np.int32)
                    batch_records.append(_BatchRecord(
                        path_idx=path_idx,
                        parquet_batch_idx=parquet_batch_idx,
                        mask=mask,
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

    print(f"  Labels indexed — {n_samples:,} rows, {len(batch_records)} batches queued (X not in RAM)")
    return y, years_arr, n_pos_total, n_neg_total, batch_records


def load_inference_rows_and_wdpa(
    panel_path: Path,
    feature_cols: List[str],
    origin_year: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load year-T pixels (WDPA_prev==0, WDPA==0) + WDPA values for T+1…T+5."""
    T = origin_year
    window_years = list(range(T + 1, T + LOOKAHEAD_YEARS + 1))
    is_evaluable_flag = (T + LOOKAHEAD_YEARS) <= WDPA_LAST_YEAR

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

    years_to_check = [yr for yr in window_years if yr <= WDPA_LAST_YEAR]
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
              f"(T+5={T+LOOKAHEAD_YEARS} > WDPA_LAST_YEAR={WDPA_LAST_YEAR})")
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
    X: "np.ndarray | _ParquetFeatureSequence",
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
    if hasattr(X, "close"):
        X.close()
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
        prec   = compute_precision_at_k(y_true, y_proba, k)
        recall = compute_recall_at_k(y_true, y_proba, k)
        lift   = prec / max(base_rate, 1e-9)
        metrics[f"precision_at_{k}pct"] = float(prec)
        metrics[f"recall_at_{k}pct"]    = float(recall)
        metrics[f"lift_at_{k}pct"]      = float(lift)

    # Forecast Capture Rate: fraction of all positives in top 5% predicted
    # (kept for backward compatibility — identical to recall_at_5pct)
    n_top5 = max(1, int(n * 0.05))
    top5_idx = np.argsort(y_proba)[-n_top5:]
    fcr = float(y_true[top5_idx].sum()) / max(n_pos, 1)
    metrics["forecast_capture_rate_top5pct"] = fcr

    print(f"\n  {label}")
    print(f"    n={n:,}, pos={n_pos:,} ({base_rate:.4%})")
    if metrics["roc_auc"] is not None:
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")
    for k in [1, 5, 10]:
        print(f"    P@{k}%: {metrics[f'precision_at_{k}pct']:.4f}  "
              f"R@{k}%: {metrics[f'recall_at_{k}pct']:.4f}  "
              f"Lift: {metrics[f'lift_at_{k}pct']:.2f}x")
    print(f"    Forecast Capture Rate (top 5%): {fcr:.4f}")

    return metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_backtest_precision(all_results: List[Dict], output_dir: Path, model_type: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    repo_root: Path,
    wb: WandbRunLogger | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    T = origin_year
    train_range = (2001, T - LOOKAHEAD_YEARS)  # mirrors deployment: train ends at T-5
    clean_window = (T + LOOKAHEAD_YEARS) <= WDPA_LAST_YEAR

    print("\n" + "=" * 70)
    print(f"BACKTEST [{model_type.upper()}]: origin year T={T}  "
          f"(train 2001–{T-LOOKAHEAD_YEARS}, eval {T+1}–{min(T+5, WDPA_LAST_YEAR)})")
    print(f"  Clean 5-year window: {clean_window}")
    print("=" * 70)
    report_memory_usage(f"start T={T}")
    if wb is not None:
        wb.log({"backtest/stage": "train_start", "backtest/origin_year": T})

    # ── Train historical deployment model ─────────────────────────────────────
    if model_type == "lgbm":
        # Lazy-load: build y + years_arr only; X streamed on demand via Sequence API.
        # Peak RAM: ~1.7 GB (labels) + ~26 GB (LightGBM histogram) — no full X matrix.
        y_tr, yr_tr, n_pos_tr, n_neg_tr, bt_records = _load_training_labels(
            split_paths, feature_cols, train_range
        )
        X_tr = _ParquetFeatureSequence(bt_records, split_paths, feature_cols, BATCH_SIZE)
        print(f"  Training data: {n_pos_tr + n_neg_tr:,} rows ({n_pos_tr:,} pos / {n_neg_tr:,} neg)")
        n_est = int(lgbm_params.get("n_estimators", N_ESTIMATORS_LOCKED_LGBM))
        model = _train_lgbm(X_tr, y_tr, yr_tr, lgbm_params, n_est, train_range)
        del y_tr, yr_tr
    else:  # rf — sklearn needs full X in RAM; use MAX_NEG_TRAIN cap to stay within budget
        X_tr, y_tr, yr_tr, n_pos_tr, n_neg_tr = load_training_data(
            split_paths, feature_cols, train_range, max_neg=MAX_NEG_TRAIN
        )
        print(f"  Training data: {len(y_tr):,} rows ({n_pos_tr:,} pos / {n_neg_tr:,} neg)")
        model = _train_rf(X_tr, y_tr, rf_params)
        del X_tr, y_tr, yr_tr

    gc.collect()
    report_memory_usage(f"after training T={T}")
    if wb is not None:
        wb.log({"backtest/stage": "train_done", "backtest/origin_year": T})

    # ── Load year-T inference set + 5-year labels ─────────────────────────────
    X_inf, row_col, label_5yr, evaluable, xy = load_inference_rows_and_wdpa(
        panel_path, feature_cols, T
    )
    if wb is not None:
        wb.log(
            {
                "backtest/stage": "inference_rows_loaded",
                "backtest/origin_year": T,
                "backtest/n_pixels_scored": int(len(X_inf)),
                "backtest/n_evaluable": int(evaluable.sum()),
                "backtest/n_pos_evaluable": int(label_5yr[evaluable].sum()),
            }
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
        if wb is not None:
            _live = {
                "backtest/stage": "metrics_done",
                "backtest/origin_year": T,
            }
            for _k, _v in metrics.items():
                if isinstance(_v, (int, float)):
                    _live[f"backtest/{_k}"] = _v
            wb.log(_live)
    else:
        print(f"  T={T}: no evaluable positive labels — skipping metrics")

    # False-positive spatial analysis (only for clean 5-year windows)
    if clean_window:
        forward_dir = output_dir.parent
        baseline_path = forward_dir / "forward_coverage_baseline.json"
        baseline_bt: Dict[str, Any] = {}
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline_bt = json.load(f)
        else:
            print(
                f"  NOTE: {baseline_path.name} not found — FP map uses repo/scratch backbone paths only",
            )
        create_false_positive_map(
            y_proba, label_5yr, evaluable, xy, T, output_dir, repo_root, baseline_bt,
        )

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
    from scripts.regions.shared.forward.config import (  # noqa: F401
        DATA_SUBDIR,
        MODEL_PREFIX,
        OUTPUTS_SUBDIR,
        resolve_forward_dir,
    )

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
    forward_dir = resolve_forward_dir(repo_root, OUTPUTS_SUBDIR)
    output_dir  = forward_dir / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime as _dt

    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    wb = WandbRunLogger(
        project="forward",
        run_name=f"backtest_{OUTPUTS_SUBDIR}_{model_type}_{_ts}",
        config={
            "region": OUTPUTS_SUBDIR,
            "model_type": model_type,
            "forward_stage": "backtest",
        },
    )
    wb.start()

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
        wb.log(
            {
                "backtest/stage": "aggregate_done",
                "backtest/n_origins_aggregated": len(all_results),
            }
        )
        wb.finish()
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

    result = run_single_origin(
        origin_year, model_type, best_params_lgbm, best_params_rf,
        feature_cols, split_paths, panel_path, output_dir, repo_root, wb=wb,
    )
    _log: dict = {
        "backtest/origin_year":    origin_year,
        "backtest/n_pixels_scored": result.get("n_pixels_scored"),
        "backtest/n_evaluable":    result.get("n_evaluable"),
        "backtest/n_pos_evaluable": result.get("n_pos_evaluable"),
        "backtest/clean_window":   int(result.get("clean_5yr_window", False)),
    }
    if result.get("metrics"):
        for _k, _v in result["metrics"].items():
            if isinstance(_v, (int, float)):
                _log[f"backtest/{_k}"] = _v
    _log["backtest/stage"] = "done"
    wb.log(_log)
    wb.finish()


if __name__ == "__main__":
    main()
