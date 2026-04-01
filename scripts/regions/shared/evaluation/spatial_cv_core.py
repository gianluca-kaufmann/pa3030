"""Shared implementation for all three layers of spatial cross-validation.

Entry points (called by thin region-specific wrappers):
    run_lobo_main(region, model_id)          — Layer 1: LOBO fold
    run_lobo_aggregate_main(region)          — Layer 1: aggregate all fold JSONs
    run_spatial_gen_main(region, model_id)   — Layer 2: biome-stratified metrics
    run_transfer_main()                      — Layer 3: cross-continental SA ↔ USA

Wrappers live at:
    scripts/regions/{south_america,usa,se_asia}/6_evaluation/spatial_CV_1
    scripts/regions/{south_america,usa,se_asia}/6_evaluation/spatial_CV_1_aggregate
    scripts/regions/{south_america,usa,se_asia}/6_evaluation/spatial_CV_2
    scripts/regions/{south_america,se_asia}/6_evaluation/spatial_CV_3

Spatial CV framework overview:
    Layer 1 (LOBO):     train on all biomes except one, evaluate on held-out biome.
                        One SLURM job per biome index; aggregate afterwards.
    Layer 2 (stratified): load existing scored parquets, assign biome labels via
                        row/col lookup in GSN raster, compute per-biome metrics.
    Layer 3 (transfer): train on SA, evaluate on USA; train on USA, evaluate on SA.
                        Restricted to features common to both regions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import wandb
except ImportError:
    wandb = None

try:
    import shap as _shap_lib
except ImportError:
    _shap_lib = None

from scripts.regions.shared.training.utils import (
    Tee,
    get_repo_root,
    report_memory_usage,
    convert_numpy_types,
    extract_features_pyarrow_to_numpy,
    compute_metrics,
    compute_year_weights,
)


# =============================================================================
# Shared constants  (identical for both regions and all layers)
# =============================================================================
RANDOM_STATE     = 42
TARGET_COL       = "transition_01_win5"
LOOKAHEAD_YEARS  = 5
WDPA_LAST_YEAR   = 2024
LAST_LABEL_YEAR  = WDPA_LAST_YEAR - LOOKAHEAD_YEARS   # 2019
TRAIN_YEARS      = (2001, 2016)
EARLYSTOP_YEARS  = (2014, 2016)
TEST_START       = 2017
TEST_END         = min(2019, LAST_LABEL_YEAR)
EXCLUDE_COLS     = {
    'transition_01', 'transition_01_win5', 'WDPA_b1', 'WDPA_prev',
    'x', 'y', 'row', 'col', 'year',
}
FIXED_PARAMS     = {
    'random_state': RANDOM_STATE,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'verbose': -1,
}
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "200000"))
MAX_NEG_SPATIAL  = int(os.environ.get("MAX_NEG_TRAIN", "0"))  # 0 = no cap; set in RF SLURM scripts
MIN_POSITIVES    = 10    # minimum positives in a test biome to consider fold valid
MIN_SAMPLES      = 100   # minimum samples in a biome to compute Layer 2 metrics


# =============================================================================
# Threading  (env vars must be set before lightgbm is imported)
# =============================================================================

def _get_num_threads() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            n = int(slurm)
            if n > 0:
                return n
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _configure_threading_env(n: int) -> None:
    s = str(max(1, n))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = s


NUM_THREADS = _get_num_threads()
_configure_threading_env(NUM_THREADS)

import lightgbm as lgb  # noqa: E402 — must be after thread env vars

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SklearnPipeline
except ImportError:
    RandomForestClassifier = None  # type: ignore[assignment,misc]
    SimpleImputer = None           # type: ignore[assignment,misc]
    SklearnPipeline = None         # type: ignore[assignment,misc]


# =============================================================================
# Path resolution
# =============================================================================

def resolve_parquet(region: str, filename: str) -> Path:
    """Find a parquet data file for a region, checking $SCRATCH first."""
    repo_root = get_repo_root()
    scratch   = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    candidates: list[Path] = []
    if scratch:
        candidates += [
            scratch / f"outputs/{region}/results/main/{filename}",
            scratch / f"data/{region}/ml/main/{filename}",
            scratch / f"outputs/{region}/results/{filename}",
            scratch / f"data/{region}/ml/{filename}",
        ]
    candidates += [
        repo_root / f"outputs/{region}/results/main/{filename}",
        repo_root / f"data/{region}/ml/main/{filename}",
        repo_root / f"outputs/{region}/results/{filename}",
        repo_root / f"data/{region}/ml/{filename}",
    ]
    for c in candidates:
        if c.exists():
            return c
    checked = "\n".join(f"  {c}" for c in candidates)
    raise FileNotFoundError(f"{region}/{filename} not found.\nChecked:\n{checked}")


def resolve_best_params(region: str) -> Optional[Path]:
    """Find lgbm_best_params.json for a region, checking $SCRATCH first."""
    repo_root = get_repo_root()
    scratch   = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    candidates: list[Path] = []
    if scratch:
        candidates += [
            Path(scratch) / f"scripts/regions/{region}/5_training/lgbm_best_params.json",
            Path(scratch) / f"scripts/regions/{region}/4_tuning/lgbm_best_params.json",
        ]
    candidates += [
        repo_root / f"scripts/regions/{region}/5_training/lgbm_best_params.json",
        repo_root / f"scripts/regions/{region}/4_tuning/lgbm_best_params.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_trained_model(region: str, model_id: str, model_type: str) -> Path:
    """Find the most-recently-saved production model file for a region.

    File naming conventions (from the training scripts):
        LGBM: data/{region}/ml/models/{model_id}_lgbm_{timestamp}.pkl
        RF:   data/{region}/ml/models/{model_id}_rf_win5_{timestamp}.joblib

    Checks $SCRATCH first, then repo_root.  Raises FileNotFoundError if none found.
    """
    repo_root = get_repo_root()
    scratch   = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    if model_type == "lgbm":
        pattern = f"{model_id}_lgbm_*.pkl"
    else:
        pattern = f"{model_id}_rf_win5_*.joblib"

    search_dirs: list[Path] = []
    if scratch:
        search_dirs.append(scratch / f"data/{region}/ml/models")
    search_dirs.append(repo_root / f"data/{region}/ml/models")

    for d in search_dirs:
        candidates = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]

    checked = "\n".join(f"  {d / pattern}" for d in search_dirs)
    raise FileNotFoundError(
        f"No trained {model_type.upper()} model found for {region}/{model_id}.\n"
        f"Checked:\n{checked}\n"
        f"Run the step-5 training script first."
    )


def resolve_rf_best_params(region: str) -> Optional[Path]:
    """Find rf_best_params.json for a region, checking $SCRATCH first."""
    repo_root = get_repo_root()
    scratch   = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    candidates: list[Path] = []
    if scratch:
        candidates += [
            Path(scratch) / f"scripts/regions/{region}/5_training/rf_best_params.json",
            Path(scratch) / f"scripts/regions/{region}/4_tuning/rf_best_params.json",
        ]
    candidates += [
        repo_root / f"scripts/regions/{region}/5_training/rf_best_params.json",
        repo_root / f"scripts/regions/{region}/4_tuning/rf_best_params.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_cv_output_dir(region: str, model_type: str = "lgbm") -> Path:
    """Resolve spatial_cv output directory, preferring $SCRATCH.

    Outputs are namespaced by model type (lgbm / rf) so that LOBO fold files
    for different algorithms never mix in the same directory.
    """
    repo_root = get_repo_root()
    scratch   = os.environ.get("SCRATCH")
    subdir    = f"spatial_cv/{model_type}"
    if scratch:
        d = Path(scratch) / f"outputs/{region}/results/{subdir}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = repo_root / f"outputs/{region}/results/{subdir}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Biome raster utilities
# =============================================================================

def load_biome_raster_and_mapping(
    raster_path: Path,
    shapefile_path: Path,
) -> tuple[np.ndarray, dict[int, str], dict[int, str]]:
    """Load the GSN terrestrial ecoregions raster and build integer→name mappings.

    The preprocessing script rasterized ECO_NAME as a sorted categorical:
        raster_value = sorted_index + 1   (1-based; 0 = background/NoData)
    This function replicates that encoding exactly.

    Returns:
        biome_raster  — uint8 ndarray [H, W]
        int_to_eco    — {raster_int: ECO_NAME}
        int_to_biome  — {raster_int: BIOME_NAME}
    """
    if rasterio is None:
        raise ImportError("rasterio is required: pip install rasterio")
    if gpd is None:
        raise ImportError("geopandas is required: pip install geopandas")

    print(f"  Loading biome raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        biome_raster  = src.read(1)
        raster_crs    = src.crs.to_string()
        raster_bounds = src.bounds
    print(f"  Raster shape: {biome_raster.shape}, dtype: {biome_raster.dtype}")

    gdf = gpd.read_file(shapefile_path).to_crs(raster_crs)
    gdf["geometry"] = gdf.geometry.make_valid()

    bbox = box(raster_bounds.left, raster_bounds.bottom,
               raster_bounds.right, raster_bounds.top)
    try:
        gdf = gpd.clip(gdf, bbox)
    except Exception as e:
        print(f"  Warning: clip failed ({e}), using full shapefile")

    if gdf.empty:
        raise ValueError("Shapefile empty after clipping to raster bounds.")

    # Replicate preprocessing encoding: sorted ECO_NAMEs → i+1
    unique_eco = sorted(gdf["ECO_NAME"].dropna().unique())
    eco_to_int  = {name: i + 1 for i, name in enumerate(unique_eco)}
    int_to_eco  = {v: k for k, v in eco_to_int.items()}

    eco_to_biome: dict[str, str] = {}
    for _, row in gdf.drop_duplicates(subset=["ECO_NAME"]).iterrows():
        if pd.notna(row.get("ECO_NAME")) and pd.notna(row.get("BIOME_NAME")):
            eco_to_biome[row["ECO_NAME"]] = str(row["BIOME_NAME"])

    int_to_biome: dict[int, str] = {
        i: eco_to_biome.get(name, "Unknown")
        for name, i in eco_to_int.items()
    }
    print(f"  {len(int_to_eco)} ecoregions → {len(set(int_to_biome.values()))} biomes")
    return biome_raster, int_to_eco, int_to_biome


def build_biome_groups(
    int_to_biome: dict[int, str],
    biome_raster: np.ndarray,
) -> list[tuple[str, list[int]]]:
    """Return sorted list of (biome_name, [raster_int, ...]) for biomes that
    have at least one pixel in the raster."""
    raster_values = set(np.unique(biome_raster).tolist()) - {0}
    groups: dict[str, list[int]] = {}
    for int_val, biome_name in int_to_biome.items():
        if int_val in raster_values:
            groups.setdefault(biome_name, []).append(int_val)
    return sorted(groups.items(), key=lambda x: x[0])


def _biome_raster_paths(region: str) -> tuple[Path, Path]:
    """Return (raster_path, shapefile_path) for a region."""
    repo_root  = get_repo_root()
    raster_path = repo_root / f"data/{region}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif"
    shp_path    = repo_root / "data/shared/GlobalSafetyNet/terrestrial_ecoregions/Terrestrial_ecoregions.shp"
    return raster_path, shp_path


# =============================================================================
# Two-pass data loading — Layer 1 (spatially filtered)
# =============================================================================

def load_data_spatial(
    parquet_paths: list[Path],
    feature_cols: list[str],
    biome_raster: np.ndarray,
    biome_ints: set[int],
    include: bool,
    year_range: tuple[int, int],
    name: str = "",
    max_neg_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Two-pass PyArrow loader with spatial biome mask + year + risk-set filters.

    Args:
        include: True = keep only pixels in biome_ints; False = exclude them.
        max_neg_samples: If > 0, cap negatives at this count via uniform random
            sampling without replacement (all positives are always kept).
            Mirrors the MAX_NEG_TRAIN env-var approach used in production RF
            training scripts; set via MAX_NEG_SPATIAL at module level.

    Returns:
        X (float32), y (int8), years_arr (int32), n_pos, n_neg
    """
    H, W          = biome_raster.shape
    biome_arr_set = np.array(sorted(biome_ints), dtype=np.int32)
    year_lo, year_hi = year_range
    op = "INCLUDE" if include else "EXCLUDE"
    neg_cap_str = f", max_neg={max_neg_samples:,}" if max_neg_samples > 0 else ""
    print(f"\nLoading {name} [{op} biome, years {year_lo}-{year_hi}{neg_cap_str}]...")
    t0 = time.time()

    essential_cols = feature_cols + [TARGET_COL, 'year', 'WDPA_prev', 'row', 'col']

    # ── Pass 1: count positives and negatives separately ────────────────────
    print("  Pass 1: counting rows...")
    n_pos_total = 0
    n_neg_total = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential_cols,
                                          use_threads=True):
                null_mask    = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
                valid_mask   = ~null_mask
                if not valid_mask.any():
                    continue
                target_arr = batch[TARGET_COL].to_numpy(zero_copy_only=False)
                year_arr  = batch['year'].to_numpy(zero_copy_only=False)
                wdpa_arr  = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
                row_arr   = batch['row'].to_numpy(zero_copy_only=False)
                col_arr   = batch['col'].to_numpy(zero_copy_only=False)
                risk_mask = (wdpa_arr == 0)
                year_mask = (year_arr >= year_lo) & (year_arr <= year_hi)
                in_bounds    = (row_arr >= 0) & (row_arr < H) & (col_arr >= 0) & (col_arr < W)
                biome_vals   = np.zeros(len(row_arr), dtype=np.int32)
                biome_vals[in_bounds] = biome_raster[row_arr[in_bounds],
                                                      col_arr[in_bounds]].astype(np.int32)
                spatial_mask = np.isin(biome_vals, biome_arr_set) if include \
                               else ~np.isin(biome_vals, biome_arr_set)
                final = valid_mask & risk_mask & year_mask & spatial_mask
                n_pos_total += int((final & (target_arr > 0)).sum())
                n_neg_total += int((final & (target_arr == 0)).sum())
        finally:
            del pf; gc.collect()

    do_neg_sample = max_neg_samples > 0 and n_neg_total > max_neg_samples
    n_neg_final   = max_neg_samples if do_neg_sample else n_neg_total
    n_total       = n_pos_total + n_neg_final
    print(f"  Found {n_total:,} samples")
    if do_neg_sample:
        print(f"  ({n_pos_total:,} pos, {n_neg_total:,} neg → capping neg at {n_neg_final:,})")
    if n_total == 0:
        raise ValueError(f"No samples found for '{name}'. Check biome index and year range.")

    # ── Pre-generate negative keep indices (exact uniform sample) ───────────
    if do_neg_sample:
        rng      = np.random.default_rng(RANDOM_STATE)
        neg_keep = np.sort(rng.choice(n_neg_total, size=n_neg_final, replace=False))
    else:
        neg_keep = None

    # ── Pass 2: fill pre-allocated arrays ───────────────────────────────────
    print(f"  Pass 2: pre-allocating {n_total:,} × {len(feature_cols)} float32...")
    X          = np.empty((n_total, len(feature_cols)), dtype=np.float32)
    y          = np.empty(n_total, dtype=np.int8)
    years_arr  = np.empty(n_total, dtype=np.int32)
    pos_offset = 0
    neg_offset = n_pos_total   # negatives placed after all positives
    neg_seen   = 0             # global counter across batches (for subsampling)
    batch_num  = 0
    _load_milestone = -1

    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=essential_cols,
                                          use_threads=True):
                batch_num += 1
                null_mask    = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
                valid_mask   = ~null_mask
                if not valid_mask.any():
                    continue
                target_arr = batch[TARGET_COL].to_numpy(zero_copy_only=False)
                year_arr   = batch['year'].to_numpy(zero_copy_only=False)
                wdpa_arr   = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
                row_arr    = batch['row'].to_numpy(zero_copy_only=False)
                col_arr    = batch['col'].to_numpy(zero_copy_only=False)
                risk_mask  = (wdpa_arr == 0)
                year_mask  = (year_arr >= year_lo) & (year_arr <= year_hi)
                in_bounds   = (row_arr >= 0) & (row_arr < H) & (col_arr >= 0) & (col_arr < W)
                biome_vals  = np.zeros(len(row_arr), dtype=np.int32)
                biome_vals[in_bounds] = biome_raster[row_arr[in_bounds],
                                                      col_arr[in_bounds]].astype(np.int32)
                spatial_mask = np.isin(biome_vals, biome_arr_set) if include \
                               else ~np.isin(biome_vals, biome_arr_set)
                final = valid_mask & risk_mask & year_mask & spatial_mask
                if not final.any():
                    continue

                pos_final     = final & (target_arr > 0)
                neg_final_all = final & (target_arr == 0)

                # --- Positives: always keep all ---
                if pos_final.any():
                    Xb_pos = extract_features_pyarrow_to_numpy(
                        batch.select(feature_cols), pos_final)
                    yr_pos = year_arr[pos_final].astype(np.int32)
                    bs_pos = len(yr_pos)
                    X[pos_offset:pos_offset + bs_pos]         = Xb_pos
                    y[pos_offset:pos_offset + bs_pos]         = 1
                    years_arr[pos_offset:pos_offset + bs_pos] = yr_pos
                    pos_offset += bs_pos

                # --- Negatives: optional subsampling ---
                if neg_final_all.any():
                    n_neg_b = int(neg_final_all.sum())
                    if neg_keep is not None:
                        # Vectorised membership test: which global neg indices land in neg_keep
                        batch_idx = np.arange(neg_seen, neg_seen + n_neg_b, dtype=np.int64)
                        lo        = np.searchsorted(neg_keep, batch_idx)
                        lo_c      = np.minimum(lo, len(neg_keep) - 1)
                        keep_local = neg_keep[lo_c] == batch_idx   # bool mask, len = n_neg_b
                        neg_seen  += n_neg_b

                        if keep_local.any():
                            neg_where    = np.where(neg_final_all)[0]
                            kept_global  = neg_where[keep_local]
                            neg_filtered = np.zeros(len(target_arr), dtype=bool)
                            neg_filtered[kept_global] = True
                            Xb_neg = extract_features_pyarrow_to_numpy(
                                batch.select(feature_cols), neg_filtered)
                            n_kept = int(keep_local.sum())
                            X[neg_offset:neg_offset + n_kept]         = Xb_neg
                            y[neg_offset:neg_offset + n_kept]         = 0
                            years_arr[neg_offset:neg_offset + n_kept] = year_arr[neg_filtered].astype(np.int32)
                            neg_offset += n_kept
                    else:
                        # No subsampling — combined extraction path
                        Xb_neg = extract_features_pyarrow_to_numpy(
                            batch.select(feature_cols), neg_final_all)
                        X[neg_offset:neg_offset + n_neg_b]         = Xb_neg
                        y[neg_offset:neg_offset + n_neg_b]         = 0
                        years_arr[neg_offset:neg_offset + n_neg_b] = year_arr[neg_final_all].astype(np.int32)
                        neg_offset += n_neg_b

                if batch_num % 25 == 0:
                    gc.collect()
                total_filled = pos_offset + (neg_offset - n_pos_total)
                _pct = total_filled * 100 // n_total if n_total > 0 else 0
                _ms  = _pct // 25
                if _ms > _load_milestone:
                    _load_milestone = _ms
                    print(f"  {_pct}% — {total_filled:,}/{n_total:,} rows")
                    report_memory_usage(f"  {_pct}%")
        finally:
            del pf; gc.collect()

    n_pos = int(y[:n_total].sum())
    n_neg = int((y[:n_total] == 0).sum())
    print(f"  Loaded in {time.time()-t0:.1f}s — {n_neg:,} neg, {n_pos:,} pos "
          f"(ratio 1:{n_neg/max(n_pos,1):.1f})")
    return X, y, years_arr, n_pos, n_neg


# =============================================================================
# Two-pass data loading — Layer 3 (full region, no spatial mask)
# =============================================================================

def load_region_train(
    train_path: Path,
    earlystop_path: Path,
    feature_cols: list[str],
    year_range: tuple[int, int],
    name: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load training data for a full region (two-pass, no spatial biome mask)."""
    print(f"\nLoading {name} train data (years {year_range[0]}-{year_range[1]})...")
    t0      = time.time()
    paths   = [train_path, earlystop_path]
    ess     = feature_cols + [TARGET_COL, 'year', 'WDPA_prev']
    year_lo, year_hi = year_range

    # Pass 1: count
    print("  Pass 1: counting rows...")
    n_total = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=ess,
                                          use_threads=True):
                nm  = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
                vm  = ~nm
                if not vm.any():
                    continue
                ya  = batch['year'].to_numpy(zero_copy_only=False)
                wa  = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
                n_total += int((vm & (wa == 0) & (ya >= year_lo) & (ya <= year_hi)).sum())
        finally:
            del pf; gc.collect()

    print(f"  Found {n_total:,} samples")
    if n_total == 0:
        raise ValueError(f"No training samples found for '{name}'")

    # Pass 2: fill
    print(f"  Pass 2: pre-allocating {n_total:,} × {len(feature_cols)} float32...")
    X         = np.empty((n_total, len(feature_cols)), dtype=np.float32)
    y         = np.empty(n_total, dtype=np.int8)
    years_arr = np.empty(n_total, dtype=np.int32)
    offset = 0; batch_n = 0; _load_milestone = -1

    for path in paths:
        pf = pq.ParquetFile(path)
        try:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=ess,
                                          use_threads=True):
                batch_n += 1
                nm    = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
                vm    = ~nm
                if not vm.any():
                    continue
                ta    = batch[TARGET_COL].to_numpy(zero_copy_only=False)
                ya    = batch['year'].to_numpy(zero_copy_only=False)
                wa    = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
                final = vm & (wa == 0) & (ya >= year_lo) & (ya <= year_hi)
                if not final.any():
                    continue
                Xb    = extract_features_pyarrow_to_numpy(batch.select(feature_cols), final)
                yb    = (ta[final] > 0).astype(np.int8)
                yrb   = ya[final].astype(np.int32)
                bs    = len(yb)
                X[offset:offset + bs]         = Xb
                y[offset:offset + bs]         = yb
                years_arr[offset:offset + bs] = yrb
                offset += bs
                if batch_n % 25 == 0:
                    gc.collect()
                _pct = offset * 100 // n_total if n_total > 0 else 0
                _ms  = _pct // 25
                if _ms > _load_milestone:
                    _load_milestone = _ms
                    print(f"  {_pct}% — {offset:,}/{n_total:,} rows")
                    report_memory_usage(f"  {_pct}%")
        finally:
            del pf; gc.collect()

    if offset != n_total:
        raise ValueError(f"Mismatch: expected {n_total}, got {offset}")

    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    print(f"  Loaded in {time.time()-t0:.1f}s — {n_neg:,} neg, {n_pos:,} pos")
    return X, y, years_arr, n_pos, n_neg


# =============================================================================
# Parameter loading
# =============================================================================

def load_best_params(path: Optional[Path]) -> tuple[dict[str, Any], int]:
    """Load LightGBM hyperparameters; return (params_dict, num_boost_round)."""
    guardrails: dict[str, Any] = {
        "max_depth": 8, "num_leaves": 255, "min_child_samples": 50,
        "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7,
        "learning_rate": 0.03, "n_estimators": 3000,
        "reg_alpha": 1.0, "reg_lambda": 1.0,
        "metric": "average_precision", "num_threads": NUM_THREADS, "verbose": -1,
    }
    if path is None:
        print("  lgbm_best_params.json not found — using guardrail defaults")
        return {**FIXED_PARAMS, **guardrails}, int(guardrails["n_estimators"])

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and ("fixed_params" in data or "best_params" in data):
        params = {**FIXED_PARAMS,
                  **data.get("fixed_params", {}),
                  **data.get("best_params", {}).copy()}
    else:
        params = {**FIXED_PARAMS, **(data if isinstance(data, dict) else {})}

    for k, v in guardrails.items():
        params.setdefault(k, v)

    n_est = params.get("n_estimators", params.get("num_boost_round",
                                                    guardrails["n_estimators"]))
    return params, int(n_est)


# =============================================================================
# Training — Layer 1 & 3
# =============================================================================

def train_lgbm(
    arrays_holder: list,
    params: dict[str, Any],
    num_boost_round: int,
) -> lgb.Booster:
    """Train LightGBM with fixed num_boost_round (no temporal CV, no early stopping).

    ``arrays_holder`` must be a list ``[X_train, y_train, years_train]``.  The
    function clears the list immediately after extracting the arrays so that the
    caller's references are dropped before ``lgb.Dataset.construct()`` runs.
    This mirrors the pattern used in model1_LGBM: ``del X_train`` in the same
    scope as ``lgb.Dataset(X_train, ..., free_raw_data=True)`` ensures the raw
    89 GB array is freed during Dataset construction rather than held through all
    training rounds, cutting peak RSS by ~90 GB on the SA LOBO folds.
    """
    X_train, y_train, years_train = arrays_holder
    arrays_holder.clear()   # release caller's last references
    del arrays_holder
    gc.collect()

    scale_pos_weight = int((y_train == 0).sum()) / max(int(y_train.sum()), 1)
    print(f"\n  scale_pos_weight = {scale_pos_weight:.4f}")

    train_params = {k: v for k, v in params.items() if k not in
                    ("n_estimators", "num_boost_round", "n_jobs",
                     "sampling_strategy", "replacement", "bootstrap",
                     "class_weight", "early_stopping_rounds", "early_stopping")}
    train_params["scale_pos_weight"] = scale_pos_weight
    train_params["num_threads"]      = NUM_THREADS
    train_params["is_unbalance"]     = False

    weights = compute_year_weights(years_train, min_year=TRAIN_YEARS[0], max_year=TRAIN_YEARS[1])
    print(f"  sample weights: min={weights.min():.3f} max={weights.max():.3f}")

    print(f"  Creating lgb.Dataset ({len(y_train):,} samples)...")
    ds = lgb.Dataset(X_train, label=y_train, weight=weights, free_raw_data=True)
    del X_train, y_train, years_train, weights; gc.collect()
    report_memory_usage("before lgb.train")

    print(f"  Training {num_boost_round} rounds...")
    t0    = time.time()
    model = lgb.train(train_params, ds, num_boost_round=num_boost_round,
                      callbacks=[lgb.log_evaluation(200)])
    print(f"  Training done in {time.time()-t0:.1f}s")
    del ds; gc.collect()
    return model


# =============================================================================
# RF parameter loading and training — Layer 1
# =============================================================================

_RF_DEFAULTS: dict[str, Any] = {
    "n_estimators":      300,
    "max_depth":         None,
    "max_features":      "sqrt",
    "min_samples_leaf":  1,
    "min_samples_split": 2,
    "bootstrap":         True,
    "class_weight":      "balanced_subsample",
    "random_state":      RANDOM_STATE,
    "verbose":           0,
}

_RF_ALLOWED_KEYS = {
    "n_estimators", "max_depth", "max_features", "min_samples_leaf",
    "min_samples_split", "bootstrap", "class_weight", "random_state",
    "min_weight_fraction_leaf", "max_leaf_nodes", "min_impurity_decrease",
    "oob_score", "warm_start", "ccp_alpha", "max_samples",
}


def load_rf_params(path: Optional[Path]) -> tuple[dict[str, Any], int]:
    """Load RandomForest hyperparameters; return (params_dict, n_estimators).

    Expected JSON structure (produced by tuning scripts):
        {"best_params": {...}, "fixed_params": {...}}
    or a flat dict.  Falls back to _RF_DEFAULTS if path is None.
    """
    if path is None:
        print("  rf_best_params.json not found — using defaults")
        params = dict(_RF_DEFAULTS)
        return params, int(params["n_estimators"])

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and ("fixed_params" in data or "best_params" in data):
        params = {**data.get("fixed_params", {}), **data.get("best_params", {}).copy()}
    else:
        params = dict(data) if isinstance(data, dict) else {}

    for k, v in _RF_DEFAULTS.items():
        params.setdefault(k, v)

    n_estimators = int(params.get("n_estimators", _RF_DEFAULTS["n_estimators"]))
    return params, n_estimators


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    n_jobs: int,
) -> Any:
    """Train RandomForestClassifier, with NaN handling only when needed.

    When the training matrix contains no NaN values (the normal case for the
    preprocessed LOBO parquets), the RF is fitted directly — matching the
    pattern used in model1_RF / model2_RF.  This avoids the full-copy that
    SimpleImputer.fit_transform() allocates inside a Pipeline, which would
    temporarily double peak RSS (e.g. 2 × 17 GB with a 50M neg cap, or
    2 × 88 GB without one).

    When NaNs are present a sklearn Pipeline(SimpleImputer → RF) is returned
    instead so that inference-time imputation is still bundled with the model.

    The returned object always supports ``.predict_proba(X)[:, 1]``.
    """
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn is required for RF training: pip install scikit-learn")

    rf_params = {k: v for k, v in params.items() if k in _RF_ALLOWED_KEYS}
    rf_params["n_jobs"] = n_jobs
    rf_params["verbose"] = 0
    n_estimators = int(rf_params.pop("n_estimators", _RF_DEFAULTS["n_estimators"]))

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"\n  Training RandomForestClassifier: {n_estimators} trees, n_jobs={n_jobs}")
    print(f"  class_weight={rf_params.get('class_weight', 'balanced_subsample')}, "
          f"max_depth={rf_params.get('max_depth', 'None')}, "
          f"max_features={rf_params.get('max_features', 'sqrt')}")
    print(f"  Training set: {len(y_train):,} samples ({n_pos:,} pos, {n_neg:,} neg)")

    has_nan = bool(np.isnan(X_train).any())
    print(f"  NaNs in training data: {has_nan}")

    t0 = time.time()
    report_memory_usage("before RF fit")
    if has_nan:
        # Pipeline bundles imputation with the model so predict_proba works on
        # data that may also contain NaNs.
        model_obj = SklearnPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf",      RandomForestClassifier(n_estimators=n_estimators, **rf_params)),
        ])
        model_obj.fit(X_train, y_train)
    else:
        # No NaNs: fit RF directly, same as model1_RF / model2_RF.
        # Avoids the full-matrix copy that SimpleImputer.fit_transform allocates.
        model_obj = RandomForestClassifier(n_estimators=n_estimators, **rf_params)
        model_obj.fit(X_train, y_train)
    print(f"  RF training done in {time.time()-t0:.1f}s")
    report_memory_usage("after RF fit")
    return model_obj


# =============================================================================
# Evaluation on held-out biome — Layer 1
# =============================================================================

def evaluate_on_biome(
    test_path: Path,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_cols: list[str],
    biome_raster: np.ndarray,
    biome_ints: set[int],
    output_dir: Path,
    timestamp: str,
    biome_slug: str,
) -> dict[str, Any]:
    """Score test parquet filtered to held-out biome; compute metrics.

    Args:
        predict_fn: Callable that takes a float32 feature matrix and returns
                    a float32 probability array.  Build it at the call site:
                    LGBM: ``lambda X: model.predict(X, num_iteration=n).astype(np.float32)``
                    RF:   ``lambda X: model.predict_proba(X)[:, 1].astype(np.float32)``
    """
    print(f"\nEvaluating on held-out biome (test years {TEST_START}-{TEST_END})...")
    H, W = biome_raster.shape
    biome_arr_set = np.array(sorted(biome_ints), dtype=np.int32)
    ess   = feature_cols + [TARGET_COL, 'year', 'WDPA_prev', 'row', 'col']

    # Pass 1: count
    print("  Pass 1: counting test rows...")
    n_test = 0
    pf = pq.ParquetFile(test_path)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=ess, use_threads=True):
            nm   = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
            vm   = ~nm
            if vm.any():
                ya  = batch['year'].to_numpy(zero_copy_only=False)
                wa  = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
                ra  = batch['row'].to_numpy(zero_copy_only=False)
                ca  = batch['col'].to_numpy(zero_copy_only=False)
                ib  = (ra >= 0) & (ra < H) & (ca >= 0) & (ca < W)
                bv  = np.zeros(len(ra), dtype=np.int32)
                bv[ib] = biome_raster[ra[ib], ca[ib]].astype(np.int32)
                n_test += int((vm & (wa == 0) &
                               (ya >= TEST_START) & (ya <= TEST_END) &
                               np.isin(bv, biome_arr_set)).sum())
    finally:
        del pf; gc.collect()

    print(f"  Found {n_test:,} test samples")
    if n_test == 0:
        # Diagnostic: count biome rows in test years ignoring WDPA_prev filter
        # to distinguish "all already protected" from "raster coordinate mismatch"
        n_biome_unfiltered = 0
        n_biome_protected  = 0
        pf_diag = pq.ParquetFile(test_path)
        try:
            for batch in pf_diag.iter_batches(
                    batch_size=BATCH_SIZE,
                    columns=['year', 'WDPA_prev', 'row', 'col'],
                    use_threads=True):
                ya = batch['year'].to_numpy(zero_copy_only=False)
                wa = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
                ra = batch['row'].to_numpy(zero_copy_only=False)
                ca = batch['col'].to_numpy(zero_copy_only=False)
                ib = (ra >= 0) & (ra < H) & (ca >= 0) & (ca < W)
                bv = np.zeros(len(ra), dtype=np.int32)
                bv[ib] = biome_raster[ra[ib], ca[ib]].astype(np.int32)
                in_biome = np.isin(bv, biome_arr_set)
                in_years = (ya >= TEST_START) & (ya <= TEST_END)
                mask_raw  = in_biome & in_years
                n_biome_unfiltered += int(mask_raw.sum())
                n_biome_protected  += int((mask_raw & (wa == 1)).sum())
        finally:
            del pf_diag; gc.collect()
        if n_biome_unfiltered == 0:
            print("  DIAGNOSTIC: 0 rows found for this biome in test years even before "
                  "WDPA_prev filter — possible biome raster coordinate mismatch or "
                  "biome truly absent from test parquet.")
        else:
            print(f"  DIAGNOSTIC: {n_biome_unfiltered:,} rows in test years for this biome "
                  f"({n_biome_protected:,} already protected, WDPA_prev=1). "
                  "All eligible pixels are already protected — fold correctly skipped.")
        print(f"  WARNING: No unprotected test samples for held-out biome in test years "
              f"({TEST_START}-{TEST_END}). Fold will be skipped.")
        return None

    # Pass 2: predict and stream scored output
    print("  Pass 2: predicting and writing scored output...")
    y_test       = np.empty(n_test, dtype=np.int8)
    y_proba      = np.empty(n_test, dtype=np.float32)
    scored_path  = output_dir / f"spatial_cv1_scored_{biome_slug}_{timestamp}.parquet"
    writer       = None
    offset       = 0
    batch_n      = 0

    pf = pq.ParquetFile(test_path)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=ess, use_threads=True):
            batch_n += 1
            nm   = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
            vm   = ~nm
            if not vm.any():
                continue
            ta   = batch[TARGET_COL].to_numpy(zero_copy_only=False)
            ya   = batch['year'].to_numpy(zero_copy_only=False)
            wa   = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
            ra   = batch['row'].to_numpy(zero_copy_only=False)
            ca   = batch['col'].to_numpy(zero_copy_only=False)
            ib   = (ra >= 0) & (ra < H) & (ca >= 0) & (ca < W)
            bv   = np.zeros(len(ra), dtype=np.int32)
            bv[ib] = biome_raster[ra[ib], ca[ib]].astype(np.int32)
            final = vm & (wa == 0) & (ya >= TEST_START) & (ya <= TEST_END) \
                    & np.isin(bv, biome_arr_set)
            if not final.any():
                continue
            Xb  = extract_features_pyarrow_to_numpy(batch.select(feature_cols), final)
            yb  = (ta[final] > 0).astype(np.int8)
            pb  = predict_fn(Xb)
            bs  = len(yb)
            y_test[offset:offset + bs]  = yb
            y_proba[offset:offset + bs] = pb
            offset += bs
            rt = pa.table({'row':          ra[final].astype(np.int32),
                           'col':          ca[final].astype(np.int32),
                           'year':         ya[final].astype(np.int32),
                           'y_true':       yb,
                           'y_pred_proba': pb})
            if writer is None:
                writer = pq.ParquetWriter(scored_path, rt.schema)
            writer.write_table(rt)
            if batch_n % 25 == 0:
                gc.collect()
    finally:
        if writer:
            writer.close()
        del pf; gc.collect()

    if offset != n_test:
        raise ValueError(f"Row count mismatch: expected {n_test}, got {offset}")

    metrics = compute_metrics(y_test, y_proba)
    n_pos   = int(y_test.sum())
    n_neg   = int((y_test == 0).sum())
    print(f"\n  Test held-out biome: {n_neg:,} neg, {n_pos:,} pos")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  P@1%: {metrics['precision_at_1pct']:.4f}  "
          f"Lift@1%: {metrics['lift_at_1pct']:.1f}x")
    print(f"  Scored output: {scored_path.name}")
    del y_test, y_proba; gc.collect()

    return {"n_test": n_test, "n_test_pos": n_pos, "n_test_neg": n_neg,
            "test_metrics": metrics, "scored_path": str(scored_path)}


# =============================================================================
# Feature detection — Layers 1 & 3
# =============================================================================

def get_feature_cols(parquet_path: Path) -> list[str]:
    """Return numeric non-excluded column names from a parquet schema."""
    schema  = pq.ParquetFile(parquet_path).schema_arrow
    numeric = [name for name, field in zip(schema.names, schema)
               if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)]
    return [c for c in numeric if c not in EXCLUDE_COLS]


def common_feature_cols(cols_a: list[str], cols_b: list[str]) -> list[str]:
    """Intersection of two feature column lists, preserving order of cols_a."""
    set_b = set(cols_b)
    return [c for c in cols_a if c in set_b]


# =============================================================================
# Transfer scoring — Layer 3
# =============================================================================

def score_region_test(
    test_path: Path,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_cols: list[str],
    output_dir: Path,
    timestamp: str,
    label: str,
) -> dict[str, Any]:
    """Score a full region's test set with a transfer model.

    Args:
        predict_fn: Callable that takes float32 feature matrix and returns
                    float32 probabilities.  Build at call site:
                    LGBM: ``lambda X: model.predict(X, num_iteration=n).astype(np.float32)``
                    RF:   ``lambda X: model.predict_proba(X)[:, 1].astype(np.float32)``
    """
    print(f"\nScoring {label} test set (years {TEST_START}-{TEST_END})...")
    ess = feature_cols + [TARGET_COL, 'year', 'WDPA_prev', 'row', 'col']

    # Pass 1: count
    n_test = 0
    pf = pq.ParquetFile(test_path)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                      columns=[TARGET_COL, 'year', 'WDPA_prev'],
                                      use_threads=True):
            nm = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
            vm = ~nm
            if vm.any():
                ya = batch['year'].to_numpy(zero_copy_only=False)
                wa = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
                n_test += int((vm & (wa == 0) &
                               (ya >= TEST_START) & (ya <= TEST_END)).sum())
    finally:
        del pf; gc.collect()

    print(f"  Found {n_test:,} test samples")
    if n_test == 0:
        raise ValueError(f"No test samples for {label}")

    # Pass 2: predict
    y_test       = np.empty(n_test, dtype=np.int8)
    y_proba      = np.empty(n_test, dtype=np.float32)
    scored_path  = output_dir / f"transfer_{label}_{timestamp}.parquet"
    writer       = None
    offset       = 0
    batch_n      = 0

    pf = pq.ParquetFile(test_path)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=ess, use_threads=True):
            batch_n += 1
            nm   = batch[TARGET_COL].is_null().to_numpy(zero_copy_only=False)
            vm   = ~nm
            if not vm.any():
                continue
            ta   = batch[TARGET_COL].to_numpy(zero_copy_only=False)
            ya   = batch['year'].to_numpy(zero_copy_only=False)
            wa   = batch['WDPA_prev'].to_numpy(zero_copy_only=False)
            ra   = batch['row'].to_numpy(zero_copy_only=False)
            ca   = batch['col'].to_numpy(zero_copy_only=False)
            final = vm & (wa == 0) & (ya >= TEST_START) & (ya <= TEST_END)
            if not final.any():
                continue
            Xb   = extract_features_pyarrow_to_numpy(batch.select(feature_cols), final)
            yb   = (ta[final] > 0).astype(np.int8)
            pb   = predict_fn(Xb)
            bs   = len(yb)
            y_test[offset:offset + bs]  = yb
            y_proba[offset:offset + bs] = pb
            offset += bs
            rt = pa.table({'row': ra[final].astype(np.int32), 'col': ca[final].astype(np.int32),
                           'year': ya[final].astype(np.int32), 'y_true': yb, 'y_pred_proba': pb})
            if writer is None:
                writer = pq.ParquetWriter(scored_path, rt.schema)
            writer.write_table(rt)
            if batch_n % 25 == 0:
                gc.collect()
    finally:
        if writer:
            writer.close()
        del pf; gc.collect()

    if offset != n_test:
        raise ValueError(f"Mismatch: expected {n_test}, got {offset}")

    metrics = compute_metrics(y_test, y_proba)
    n_pos   = int(y_test.sum()); n_neg = int((y_test == 0).sum())
    print(f"  {n_neg:,} neg, {n_pos:,} pos | "
          f"ROC-AUC {metrics['roc_auc']:.4f}  "
          f"PR-AUC {metrics['pr_auc']:.4f}  "
          f"P@1% {metrics['precision_at_1pct']:.4f}")
    del y_test, y_proba; gc.collect()
    return {"n_test": n_test, "n_test_pos": n_pos, "n_test_neg": n_neg,
            "metrics": metrics, "scored_path": str(scored_path)}


# =============================================================================
# Layer 2 utilities — biome-stratified metrics on existing scored parquets
# =============================================================================

def lookup_biomes_for_rows(
    row_array: np.ndarray,
    col_array: np.ndarray,
    biome_raster: np.ndarray,
    int_to_biome: dict[int, str],
) -> np.ndarray:
    """Vectorised (row, col) → BIOME_NAME lookup. Returns object array of strings."""
    H, W  = biome_raster.shape
    valid = (row_array >= 0) & (row_array < H) & (col_array >= 0) & (col_array < W)
    raw   = np.zeros(len(row_array), dtype=np.uint8)
    raw[valid] = biome_raster[row_array[valid], col_array[valid]]
    return pd.Series(raw).map(int_to_biome).fillna("Unknown").to_numpy(dtype=object)


def _compute_region_metrics_layer2(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """Per-biome metric dict for Layer 2 (with MIN_SAMPLES guard)."""
    n     = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    out: dict[str, Any] = {
        "n_samples":    n,
        "n_positives":  n_pos,
        "positive_rate": float(y_true.mean()) if n > 0 else 0.0,
        "neg_pos_ratio": float(n_neg / n_pos) if n_pos > 0 else np.nan,
    }
    if n < MIN_SAMPLES or n_pos < MIN_POSITIVES or len(np.unique(y_true)) < 2:
        out.update(roc_auc=np.nan, pr_auc=np.nan,
                   precision_at_1pct=np.nan,  precision_at_5pct=np.nan,
                   precision_at_10pct=np.nan,
                   recall_at_1pct=np.nan,     recall_at_5pct=np.nan,
                   recall_at_10pct=np.nan)
        return out

    def _p_at_k(k: float) -> float:
        n_top = max(1, int(n * k / 100))
        top_idx = np.argpartition(y_proba, -n_top)[-n_top:]
        return float(y_true[top_idx].sum() / n_top)

    def _r_at_k(k: float) -> float:
        """Recall at top-k%: fraction of all positives captured in top-k% predictions."""
        if n_pos == 0:
            return 0.0
        n_top = max(1, int(n * k / 100))
        top_idx = np.argpartition(y_proba, -n_top)[-n_top:]
        return float(y_true[top_idx].sum() / n_pos)

    try:
        roc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        roc = np.nan
    try:
        pr = float(average_precision_score(y_true, y_proba))
    except ValueError:
        pr = np.nan

    out.update(roc_auc=roc, pr_auc=pr,
               precision_at_1pct=_p_at_k(1.0),
               precision_at_5pct=_p_at_k(5.0),
               precision_at_10pct=_p_at_k(10.0),
               recall_at_1pct=_r_at_k(1.0),
               recall_at_5pct=_r_at_k(5.0),
               recall_at_10pct=_r_at_k(10.0))
    return out


def find_scored_file(
    output_dir: Path,
    algorithm: str,
    model_id: str,
) -> Optional[Path]:
    """Find the most recent scored parquet for a given algorithm.

    Searches both ``output_dir/`` and ``output_dir/main/`` because calibrated
    outputs from the calibration step land in the ``main/`` subdirectory.
    """
    if algorithm == "lgbm":
        pattern = f"{model_id}_lgbm_scored_*.parquet"
    else:
        pattern = f"{model_id}_rf_win5_scored_*.parquet"
    files = list(output_dir.glob(pattern))
    files.extend(output_dir.glob(f"main/{pattern}"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _evaluate_model_layer2(
    scored_path: Path,
    model_label: str,
    biome_raster: np.ndarray,
    int_to_biome: dict[int, str],
    test_year_min: int = TEST_START,
    test_year_max: int = TEST_END,
) -> pd.DataFrame:
    """Load scored parquet, assign biomes, compute per-biome metrics (Layer 2)."""
    print(f"\n  Loading {model_label} scored file: {scored_path.name}")
    t0 = time.time()

    df = pd.read_parquet(scored_path, columns=["row", "col", "year",
                                                "y_true", "y_pred_proba"])
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    df = df[(df["year"] >= test_year_min) & (df["year"] <= test_year_max)].reset_index(drop=True)
    print(f"  After year filter ({test_year_min}-{test_year_max}): {len(df):,} rows")
    if len(df) == 0:
        raise ValueError(f"No rows in test years for {model_label}")

    print("  Looking up biomes via (row, col)...")
    df["biome"] = lookup_biomes_for_rows(
        df["row"].to_numpy(dtype=np.int32),
        df["col"].to_numpy(dtype=np.int32),
        biome_raster, int_to_biome,
    )

    print(f"  Found {df['biome'].nunique()} biomes in test data")
    rows = []
    for biome_name, grp in df.groupby("biome", sort=True):
        m = _compute_region_metrics_layer2(
            grp["y_true"].to_numpy(dtype=np.int8),
            grp["y_pred_proba"].to_numpy(dtype=np.float32),
        )
        m["model"] = model_label
        m["biome"] = biome_name
        rows.append(m)

    # Global row
    m_all = _compute_region_metrics_layer2(
        df["y_true"].to_numpy(dtype=np.int8),
        df["y_pred_proba"].to_numpy(dtype=np.float32),
    )
    m_all["model"] = model_label
    m_all["biome"] = "ALL"
    rows.append(m_all)

    del df; gc.collect()
    result = pd.DataFrame(rows)
    cols = ["model", "biome", "n_samples", "n_positives", "positive_rate",
            "neg_pos_ratio",
            "roc_auc", "pr_auc",
            "precision_at_1pct",  "precision_at_5pct",  "precision_at_10pct",
            "recall_at_1pct",     "recall_at_5pct",     "recall_at_10pct"]
    return result[[c for c in cols if c in result.columns]]


# =============================================================================
# Layer 2 helper — per-biome feature importance
# =============================================================================
# Both LGBM and RF use sklearn permutation importance so results are on the
# same scale and directly comparable.  SHAP is avoided here because for RF
# with 1 000 trees × depth 20 it costs O(n_trees×depth×samples) — hours.
# LGBM global SHAP is still produced by results_core.py.

_SHAP_ROWS_PER_BIOME = 1_000   # max sample size per biome
_SHAP_TOP_K          = 10      # features to report per biome
_PERM_N_REPEATS      = 5       # repeats for permutation importance


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Parse integer environment variable with lower bound and fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return max(minimum, default)
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return max(minimum, default)


# Layer 2 permutation-importance controls
# Default to conservative settings to avoid OOM on large Slurm nodes.
PERM_N_REPEATS = _env_int("SPCV2_PERM_N_REPEATS", _PERM_N_REPEATS, minimum=1)
PERM_N_JOBS = _env_int("SPCV2_PERM_N_JOBS", 1, minimum=1)
SPCV2_ROWS_PER_BIOME = _env_int("SPCV2_ROWS_PER_BIOME", _SHAP_ROWS_PER_BIOME, minimum=100)


def _compute_biome_feature_importance(
    region: str,
    model_id: str,
    model_type: str,
    biome_raster: np.ndarray,
    int_to_biome: dict[int, str],
    n_per_biome: int = SPCV2_ROWS_PER_BIOME,
    top_k: int = _SHAP_TOP_K,
) -> Optional[pd.DataFrame]:
    """Compute per-biome feature importance using the production model.

    Both LGBM and RF use sklearn permutation importance so that results are on
    the same scale (mean PR-AUC drop when a feature is permuted) and directly
    comparable across models.  SHAP is intentionally avoided here: for RF with
    1 000 trees × depth 20 it takes hours; using a different metric for LGBM
    would make the two models incomparable.  LGBM SHAP is still computed
    globally in the main results pipeline (results_core.py).

    Streams the test parquet in chunks, builds a stratified sample of up to
    ``n_per_biome`` rows per biome, then runs permutation importance per biome.
    Returns a DataFrame with the top-``top_k`` features per biome, or ``None``
    if the model/data files are not found.

    Columns: model, biome, rank, feature, importance_score, method, n_samples
      importance_score = mean PR-AUC drop under permutation (sklearn ``average_precision``)
    """
    method_label = "permutation"
    print(f"  {model_type.upper()} model: permutation importance "
          f"(n_repeats={PERM_N_REPEATS}, n_jobs={PERM_N_JOBS}, "
          f"scoring=PR-AUC / average_precision)")

    # ── Load production model ─────────────────────────────────────────────────
    try:
        model_path = resolve_trained_model(region, model_id, model_type)
    except FileNotFoundError as e:
        print(f"  Model not found ({model_type}): {e}")
        return None

    print(f"  Loading {model_type.upper()} model: {model_path.name}")
    if model_type == "lgbm":
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        feature_cols: list[str] = list(model.feature_name())
        # Generic "Column_N" names → resolve from parquet schema
        if feature_cols and feature_cols[0].startswith("Column_") and feature_cols[0][7:].isdigit():
            try:
                tp = resolve_parquet(region, "test_win5.parquet")
                feature_cols = get_feature_cols(tp)
                print(f"  Resolved {len(feature_cols)} feature names from parquet schema")
            except FileNotFoundError:
                print("  Cannot resolve feature names — skipping")
                del model; gc.collect()
                return None
    else:
        try:
            import joblib as _joblib
        except ImportError:
            print("  joblib not available — skipping RF feature importance")
            return None
        model = _joblib.load(model_path)
        if hasattr(model, "feature_names_in_"):
            feature_cols = list(model.feature_names_in_)
        else:
            try:
                tp = resolve_parquet(region, "test_win5.parquet")
                feature_cols = get_feature_cols(tp)
            except FileNotFoundError:
                print("  Cannot resolve feature names — skipping")
                del model; gc.collect()
                return None

    print(f"  {len(feature_cols)} features")

    # ── Find test parquet ─────────────────────────────────────────────────────
    try:
        test_path = resolve_parquet(region, "test_win5.parquet")
    except FileNotFoundError as e:
        print(f"  Test parquet not found: {e}")
        del model; gc.collect()
        return None

    schema_names = set(pq.ParquetFile(test_path).schema_arrow.names)
    actual_feat_cols = [c for c in feature_cols if c in schema_names]
    if not actual_feat_cols:
        print("  No feature columns found in test parquet — skipping")
        del model; gc.collect()
        return None

    if TARGET_COL not in schema_names:
        print(f"  Target column '{TARGET_COL}' not in parquet — skipping")
        del model; gc.collect()
        return None

    load_cols = (
        [TARGET_COL, "row", "col"]
        + [c for c in actual_feat_cols if c not in ("row", "col")]
    )

    # ── Stream test parquet, collect per-biome samples ────────────────────────
    print(f"  Streaming {test_path.name} (≤{n_per_biome} rows/biome)...")
    t0 = time.time()
    x_buffers: dict[str, list[np.ndarray]] = {}
    y_buffers: dict[str, list[np.ndarray]] = {}
    counts:    dict[str, int]              = {}

    pf = pq.ParquetFile(test_path)
    for batch in pf.iter_batches(batch_size=500_000, columns=load_cols):
        chunk = batch.to_pandas()
        biome_arr = lookup_biomes_for_rows(
            chunk["row"].to_numpy(dtype=np.int32),
            chunk["col"].to_numpy(dtype=np.int32),
            biome_raster, int_to_biome,
        )
        chunk["_biome"] = biome_arr
        for biome_name, grp in chunk.groupby("_biome"):
            current = counts.get(biome_name, 0)
            if current >= n_per_biome:
                continue
            needed = n_per_biome - current
            X = grp[actual_feat_cols].to_numpy(dtype=np.float32)
            take = min(needed, len(X))
            x_buffers.setdefault(biome_name, []).append(X[:take])
            y = grp[TARGET_COL].to_numpy(dtype=np.int8)
            y_buffers.setdefault(biome_name, []).append(y[:take])
            counts[biome_name] = current + take
        del chunk; gc.collect()

    print(f"  Sampling done in {time.time()-t0:.1f}s  "
          f"({len(x_buffers)} biomes, {sum(counts.values()):,} total rows)")

    if not x_buffers:
        del model; gc.collect()
        return None

    # ── Permutation importance — same metric for both LGBM and RF ────────────
    try:
        from sklearn.inspection import permutation_importance as _perm_imp
    except ImportError:
        print("  sklearn.inspection not available — skipping feature importance")
        del model; gc.collect()
        return None

    # Avoid nested parallelism: permutation_importance uses joblib parallelism,
    # while RF models may also use internal thread pools. Force single-thread
    # model inference during permutation scoring.
    if model_type == "rf" and hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
            print("  RF prediction threads forced to n_jobs=1 during permutation scoring")
        except Exception as e:
            print(f"  WARNING: could not force RF n_jobs=1 ({e})")

    model_label = model_type.upper()
    rows_out: list[dict[str, Any]] = []

    print(f"  Running permutation importance ({len(x_buffers)} biomes)...")
    t_perm = time.time()
    for biome_name in sorted(x_buffers):
        X_biome = np.vstack(x_buffers[biome_name])
        y_biome = np.concatenate(y_buffers[biome_name]).astype(int)
        n_pos = int(y_biome.sum())
        n_neg = len(y_biome) - n_pos
        if n_pos < 1 or n_neg < 1:
            # PR-AUC / average_precision needs at least one positive and one negative
            print(f"  {biome_name}: need both classes (pos={n_pos}, neg={n_neg}) — skipping")
            continue
        try:
            result = _perm_imp(
                model, X_biome, y_biome,
                n_repeats=PERM_N_REPEATS,
                scoring="average_precision",
                random_state=42,
                n_jobs=PERM_N_JOBS,
            )
            importance = result.importances_mean
            top_idx = np.argsort(importance)[::-1][:top_k]
            for rank_i, feat_idx in enumerate(top_idx, start=1):
                rows_out.append({
                    "model":            model_label,
                    "biome":            biome_name,
                    "rank":             rank_i,
                    "feature":          actual_feat_cols[feat_idx],
                    "importance_score": float(importance[feat_idx]),
                    "method":           method_label,
                    "n_samples":        int(X_biome.shape[0]),
                })
        except Exception as e:
            print(f"  Permutation importance failed for {biome_name}: {e}")
    print(f"  Permutation importance done in {time.time()-t_perm:.1f}s")

    del model, x_buffers, y_buffers; gc.collect()
    n_biomes = len(rows_out) // top_k if top_k else "?"
    print(f"  Feature importance complete: {len(rows_out)} rows ({n_biomes} biomes)")
    return pd.DataFrame(rows_out) if rows_out else None


# =============================================================================
# Layer 3 transfer helper — uses production models, no retraining
# =============================================================================

def _run_transfer_direction(
    source_region: str,
    source_model_id: str,
    target_region: str,
    model_type: str,
    output_dir: Path,
    timestamp: str,
) -> dict[str, Any]:
    """Load the production model for source_region and evaluate on target_region.

    No retraining: the production model (trained in step 5) is loaded directly.
    Feature columns are read from the model itself, guaranteeing that the target
    region's data is extracted in exactly the order the model expects.

    SA and USA share identical 73-feature sets, so cross-scoring is exact.
    """
    label = f"{source_region}_to_{target_region}_{model_type}"
    print("\n" + "=" * 70)
    print(f"TRANSFER ({model_type.upper()}): "
          f"{source_region.upper()} → {target_region.upper()}")
    print("=" * 70)

    # ── Load production model ────────────────────────────────────────────────
    model_path = resolve_trained_model(source_region, source_model_id, model_type)
    print(f"  Loading model: {model_path.name}")

    if model_type == "lgbm":
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        # LightGBM stores exact training column order
        feature_cols = list(model.feature_name())
        # Production models trained on numpy arrays store generic names ("Column_0", ...).
        # Replace with parquet-derived names from the target region (same feature set,
        # same positional order) so that batch.select(feature_cols) works correctly.
        if feature_cols and feature_cols[0].startswith("Column_") and feature_cols[0][7:].isdigit():
            tgt_parquet_cols = get_feature_cols(resolve_parquet(target_region, "train_win5.parquet"))
            if len(tgt_parquet_cols) == len(feature_cols):
                feature_cols = tgt_parquet_cols
                print(f"  NOTE: Replaced generic Column_N names with "
                      f"{len(feature_cols)} parquet-derived feature names")
            else:
                raise ValueError(
                    f"Feature count mismatch: model has {len(feature_cols)} features "
                    f"(generic names), target parquet has {len(tgt_parquet_cols)}. "
                    f"Cannot infer correct column mapping.")
        num_boost_round = model.num_trees()
        predict_fn: Callable[[np.ndarray], np.ndarray] = (
            lambda X, _m=model, _n=num_boost_round:
                _m.predict(X, num_iteration=_n).astype(np.float32)
        )
        model_details = {"num_boost_round": num_boost_round}
    else:  # rf
        try:
            import joblib as _joblib
        except ImportError:
            raise ImportError("joblib is required to load RF models: pip install joblib")
        model = _joblib.load(model_path)
        # sklearn RF stores training feature order in feature_names_in_
        if hasattr(model, "feature_names_in_"):
            feature_cols = list(model.feature_names_in_)
        else:
            # Fallback: infer feature order from both parquet schemas and verify
            # they agree, so cross-continental scoring uses correct column alignment.
            src_train = resolve_parquet(source_region, "train_win5.parquet")
            tgt_train = resolve_parquet(target_region, "train_win5.parquet")
            src_cols = get_feature_cols(src_train)
            tgt_cols = get_feature_cols(tgt_train)
            if src_cols != tgt_cols:
                raise ValueError(
                    f"RF feature_names_in_ not available and source/target parquet "
                    f"schemas differ — cannot safely score cross-continental transfer.\n"
                    f"Source ({source_region}): {src_cols}\n"
                    f"Target ({target_region}): {tgt_cols}"
                )
            feature_cols = src_cols
            print("  WARNING: feature_names_in_ not available — inferred from parquet "
                  "schemas (source and target schemas verified identical)")
        n_estimators = model.n_estimators if hasattr(model, "n_estimators") else "?"
        predict_fn = lambda X, _m=model: _m.predict_proba(X)[:, 1].astype(np.float32)
        model_details = {"n_estimators": n_estimators}

    print(f"  Features: {len(feature_cols)}")
    for k, v in model_details.items():
        print(f"  {k}: {v}")

    # ── Score target region ──────────────────────────────────────────────────
    tgt_test = resolve_parquet(target_region, "test_win5.parquet")
    test_result = score_region_test(
        tgt_test, predict_fn, feature_cols, output_dir, timestamp, label,
    )
    del model; gc.collect()

    return {
        "direction":     f"{source_region} → {target_region}",
        "model_type":    model_type,
        "source_region": source_region,
        "target_region": target_region,
        "n_features":    len(feature_cols),
        "model_path":    str(model_path),
        **model_details,
        "test_data": {
            "n_samples":  test_result["n_test"],
            "n_positive": test_result["n_test_pos"],
            "n_negative": test_result["n_test_neg"],
        },
        "test_metrics": test_result["metrics"],
        "scored_path":  test_result["scored_path"],
    }


# =============================================================================
# Entry point — Layer 1: LOBO fold
# =============================================================================

def run_lobo_main(region: str, model_id: str) -> None:
    """Run one LOBO fold for the given region.  Called by thin wrapper scripts."""
    parser = argparse.ArgumentParser(
        description=f"Leave-One-Biome-Out (LOBO) CV — {region.replace('_', ' ').title()}"
    )
    parser.add_argument("--biome-idx", type=int, default=None,
                        help="0-based biome index to hold out (see --list-biomes)")
    parser.add_argument("--list-biomes", action="store_true",
                        help="Print available biomes and exit")
    parser.add_argument("--model-type", choices=["lgbm", "rf"], default="lgbm",
                        help="Model type to train in each fold (default: lgbm)")
    args = parser.parse_args()

    model_type  = args.model_type
    repo_root   = get_repo_root()
    output_dir  = resolve_cv_output_dir(region, model_type)
    raster_path, shp_path = _biome_raster_paths(region)

    if not raster_path.exists():
        print(f"ERROR: Biome raster not found: {raster_path}")
        print(f"Run: python scripts/regions/{region}/2_preprocessing/gsn_preprocessing")
        sys.exit(1)

    print("Loading biome raster and mapping...")
    biome_raster, _, int_to_biome = load_biome_raster_and_mapping(raster_path, shp_path)
    biome_groups = build_biome_groups(int_to_biome, biome_raster)

    if args.list_biomes:
        print("\nAvailable biomes (--biome-idx values):")
        for idx, (name, ints) in enumerate(biome_groups):
            print(f"  [{idx:2d}] {name}  (ecoregion codes: {ints})")
        sys.exit(0)

    if args.biome_idx is None:
        parser.error("--biome-idx is required (or use --list-biomes)")
    if not (0 <= args.biome_idx < len(biome_groups)):
        print(f"ERROR: --biome-idx must be in [0, {len(biome_groups)-1}]. "
              f"Got: {args.biome_idx}")
        sys.exit(1)

    biome_name, biome_ints_list = biome_groups[args.biome_idx]
    biome_ints = set(biome_ints_list)
    biome_slug = "".join(
        c for c in biome_name.replace(" ", "_").replace("/", "_")
                              .replace("&", "and")[:40]
        if c.isalnum() or c == "_"
    )
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path   = output_dir / f"fold_{args.biome_idx:02d}_{biome_slug}_{timestamp}.txt"
    use_wandb  = False

    tee = Tee(log_path)
    sys.stdout = tee
    try:
        t_global = time.time()
        region_label = region.replace("_", " ").upper()

        print("=" * 70)
        print(f"LOBO SPATIAL CV — {region_label}  "
              f"Fold {args.biome_idx}/{len(biome_groups)-1}")
        print("=" * 70)
        print(f"Model type:       {model_type.upper()}")
        print(f"Held-out biome:   {biome_name}")
        print(f"Biome int codes:  {sorted(biome_ints)}")
        print(f"N biomes total:   {len(biome_groups)}")
        print(f"Train years:      {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}")
        print(f"Test years:       {TEST_START}-{TEST_END}")
        print(f"Output dir:       {output_dir}")

        train_path     = resolve_parquet(region, "train_win5.parquet")
        earlystop_path = resolve_parquet(region, "earlystop_win5.parquet")
        test_path      = resolve_parquet(region, "test_win5.parquet")
        print(f"\nTrain:     {train_path}")
        print(f"Earlystop: {earlystop_path}")
        print(f"Test:      {test_path}")

        if model_type == "lgbm":
            params_path = resolve_best_params(region)
            if params_path:
                print(f"\nLoaded LGBM params: {params_path}")
            else:
                print("\nWARNING: lgbm_best_params.json not found — using guardrail defaults")
            params, num_boost_round = load_best_params(params_path)
            params["num_threads"] = NUM_THREADS
            print(f"num_boost_round: {num_boost_round}")
            _lobo_rounds = int(os.environ.get("LOBO_NUM_BOOST_ROUND", "0"))
            if _lobo_rounds > 0 and _lobo_rounds != num_boost_round:
                print(f"  LOBO override: num_boost_round {num_boost_round} → {_lobo_rounds} "
                      f"(LOBO_NUM_BOOST_ROUND env var; production value preserved elsewhere)")
                num_boost_round = _lobo_rounds
        else:  # rf
            params_path = resolve_rf_best_params(region)
            if params_path:
                print(f"\nLoaded RF params: {params_path}")
            else:
                print("\nWARNING: rf_best_params.json not found — using defaults")
            params, num_boost_round = load_rf_params(params_path)
            print(f"n_estimators: {num_boost_round}")
            _lobo_trees = int(os.environ.get("LOBO_N_ESTIMATORS", "0"))
            if _lobo_trees > 0 and _lobo_trees != num_boost_round:
                print(f"  LOBO override: n_estimators {num_boost_round} → {_lobo_trees} "
                      f"(LOBO_N_ESTIMATORS env var; production value preserved elsewhere)")
                num_boost_round = _lobo_trees
                params["n_estimators"] = _lobo_trees  # train_rf reads from params dict

        schema       = pq.ParquetFile(train_path).schema_arrow
        feature_cols = [
            name for name, field in zip(schema.names, schema)
            if (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
            and name not in EXCLUDE_COLS
        ]
        print(f"\nFeatures: {len(feature_cols)}")
        report_memory_usage("startup")

        if wandb is not None:
            try:
                wandb.init(
                    project="spatial-cv",
                    entity=os.environ.get("WANDB_ENTITY"),
                    name=f"lobo_{region}_{args.biome_idx:02d}_{biome_slug[:20]}",
                    config={
                        "region":          region,
                        "model_id":        model_id,
                        "model_type":      model_type,
                        "biome_idx":       args.biome_idx,
                        "biome_name":      biome_name,
                        "n_biomes":        len(biome_groups),
                        "train_years":     list(TRAIN_YEARS),
                        "test_years":      [TEST_START, TEST_END],
                        "num_boost_round": num_boost_round,
                        "n_features":      len(feature_cols),
                    },
                )
                use_wandb = True
                print("W&B connected")
            except Exception as _err:
                print(f"W&B failed: {_err}")

        print("\n" + "=" * 70)
        print("STEP 1: LOAD TRAINING DATA (held-out biome excluded)")
        print("=" * 70)
        X_train, y_train, years_train, n_pos_tr, n_neg_tr = load_data_spatial(
            [train_path, earlystop_path],
            feature_cols, biome_raster, biome_ints,
            include=False,
            year_range=(TRAIN_YEARS[0], EARLYSTOP_YEARS[1]),
            name="train+earlystop",
            max_neg_samples=MAX_NEG_SPATIAL,
        )
        n_train = len(y_train)
        print(f"  Training set: {n_train:,} rows")
        report_memory_usage("after loading training data")

        print("\n" + "=" * 70)
        print(f"STEP 2: TRAIN {model_type.upper()} (fixed rounds/trees)")
        print("=" * 70)
        if model_type == "lgbm":
            # Transfer ownership via a holder list so the caller's X_train reference
            # is dropped before lgb.Dataset.construct() runs.  After arrays_holder.clear()
            # inside train_lgbm, ds.data is the only live reference; free_raw_data=True
            # then frees the 89 GB array after binning, cutting ~90 GB from peak RSS
            # during training rounds.  RF keeps the full array through fit() so its
            # memory is managed via MAX_NEG_TRAIN in the SLURM scripts.
            _holder = [X_train, y_train, years_train]
            del X_train, y_train, years_train
            gc.collect()
            model = train_lgbm(_holder, params, num_boost_round)
            predict_fn: Callable[[np.ndarray], np.ndarray] = (
                lambda X, _m=model, _n=num_boost_round:
                    _m.predict(X, num_iteration=_n).astype(np.float32)
            )
        else:  # rf
            model = train_rf(X_train, y_train, params, n_jobs=NUM_THREADS)
            predict_fn = lambda X, _m=model: _m.predict_proba(X)[:, 1].astype(np.float32)
            del X_train, y_train, years_train; gc.collect()
        report_memory_usage("after training")

        model_dir  = repo_root / f"data/{region}/ml/models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"lobo_{model_type}_{args.biome_idx:02d}_{biome_slug}_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved: {model_path}")

        print("\n" + "=" * 70)
        print("STEP 3: EVALUATE ON HELD-OUT BIOME")
        print("=" * 70)
        test_results = evaluate_on_biome(
            test_path, predict_fn, feature_cols, biome_raster, biome_ints,
            output_dir, timestamp, biome_slug,
        )
        del model; gc.collect()

        elapsed = time.time() - t_global

        if test_results is None:
            # Biome has no unprotected test pixels in test years — skip evaluation
            fold_result = {
                "fold_idx":      args.biome_idx,
                "biome_name":    biome_name,
                "biome_ints":    sorted(biome_ints),
                "region":        region,
                "model":         f"{model_id}_{model_type}",
                "model_type":    model_type,
                "timestamp":     timestamp,
                "num_boost_round": num_boost_round,
                "train_years":   list(TRAIN_YEARS),
                "test_years":    [TEST_START, TEST_END],
                "n_biome_folds": len(biome_groups),
                "skipped":       True,
                "skip_reason":   "No unprotected test pixels in test years",
                "train_data": {"n_samples": n_train,
                               "n_positive": n_pos_tr,
                               "n_negative": n_neg_tr},
                "test_data":     {"n_samples": 0, "n_positive": 0, "n_negative": 0},
                "timing":        {"total_seconds": elapsed, "total_minutes": elapsed / 60},
                "model_path":    str(model_path),
                "scored_path":   None,
            }
            print(f"\nFold {args.biome_idx} ({biome_name}) SKIPPED — no test samples.")
        else:
            n_pos_test = test_results["n_test_pos"]
            if n_pos_test < MIN_POSITIVES:
                print(f"\nWARNING: Only {n_pos_test} positives in held-out biome "
                      f"(minimum recommended: {MIN_POSITIVES}). Metrics may be unreliable.")

            fold_result = {
                "fold_idx":        args.biome_idx,
                "biome_name":      biome_name,
                "biome_ints":      sorted(biome_ints),
                "region":          region,
                "model":           f"{model_id}_{model_type}",
                "model_type":      model_type,
                "timestamp":       timestamp,
                "num_boost_round": num_boost_round,
                "train_years":     list(TRAIN_YEARS),
                "test_years":      [TEST_START, TEST_END],
                "n_biome_folds":   len(biome_groups),
                "train_data": {"n_samples": n_train,
                               "n_positive": n_pos_tr,
                               "n_negative": n_neg_tr},
                "test_data": test_results["test_metrics"] | {
                    "n_samples":  test_results["n_test"],
                    "n_positive": test_results["n_test_pos"],
                    "n_negative": test_results["n_test_neg"],
                },
                "timing":       {"total_seconds": elapsed, "total_minutes": elapsed / 60},
                "model_path":   str(model_path),
                "scored_path":  test_results["scored_path"],
            }
        fold_result = convert_numpy_types(fold_result)

        json_path = output_dir / f"fold_{args.biome_idx:02d}_{biome_slug}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(fold_result, f, indent=2)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        m = fold_result["test_data"]
        print(f"Biome:     {biome_name}")
        if test_results is not None:
            print(f"Test rows: {test_results['n_test']:,}  ({n_pos_test:,} pos)")
            print(f"ROC-AUC:   {m.get('roc_auc', float('nan')):.4f}")
            print(f"PR-AUC:    {m.get('pr_auc', float('nan')):.4f}")
            print(f"P@1%:      {m.get('precision_at_1pct', float('nan')):.4f}  "
                  f"(Lift {m.get('lift_at_1pct', float('nan')):.1f}x)")
            print(f"P@5%:      {m.get('precision_at_5pct', float('nan')):.4f}")
        else:
            print("Test rows: 0 (fold skipped — no test samples)")
        print(f"Time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print("=" * 70)
        print(f"\nFold results saved: {json_path}")

        if use_wandb:
            wandb.log({
                "test/roc_auc":            m.get("roc_auc",            float("nan")),
                "test/pr_auc":             m.get("pr_auc",             float("nan")),
                "test/precision_at_1pct":  m.get("precision_at_1pct",  float("nan")),
                "test/precision_at_5pct":  m.get("precision_at_5pct",  float("nan")),
                "test/recall_at_1pct":     m.get("recall_at_1pct",     float("nan")),
                "test/recall_at_5pct":     m.get("recall_at_5pct",     float("nan")),
                "test/recall_at_10pct":    m.get("recall_at_10pct",    float("nan")),
                "test/lift_at_1pct":       m.get("lift_at_1pct",       float("nan")),
                "test/brier_score":        m.get("brier_score",        float("nan")),
                "test/n_samples":          fold_result["test_data"].get("n_samples", 0),
                "test/n_positive":         fold_result["test_data"].get("n_positive", 0),
                "train/n_samples":         n_train,
                "train/n_positive":        n_pos_tr,
                "timing/total_seconds":    elapsed,
            })
            wandb.finish()

        print("Done.")

    finally:
        if use_wandb and wandb is not None and wandb.run is not None:
            wandb.finish()
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nLog saved to: {log_path}")


# =============================================================================
# Entry point — Layer 1: aggregate fold JSONs
# =============================================================================

METRIC_COLS = [
    "roc_auc", "pr_auc",
    "precision_at_1pct", "precision_at_5pct", "precision_at_10pct",
    "recall_at_1pct",    "recall_at_5pct",    "recall_at_10pct",
    "lift_at_1pct", "lift_at_5pct", "lift_at_10pct",
    "brier_score",
]


def run_lobo_aggregate_main(region: str) -> None:
    """Aggregate all fold_*.json files for a region into a summary CSV."""
    parser = argparse.ArgumentParser(
        description=f"LOBO aggregate — {region.replace('_', ' ').title()}"
    )
    parser.add_argument("--model-type", choices=["lgbm", "rf"], default="lgbm",
                        help="Model type whose fold JSONs to aggregate (default: lgbm)")
    args      = parser.parse_args()
    model_type = args.model_type
    cv_dir    = resolve_cv_output_dir(region, model_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = cv_dir / f"lobo_aggregate_{timestamp}.txt"

    tee = Tee(log_path)
    sys.stdout = tee
    try:
        t0           = time.time()
        region_label = region.replace("_", " ").upper()

        print("=" * 70)
        print(f"LOBO AGGREGATE — {region_label} ({model_type.upper()})")
        print("=" * 70)
        print(f"Looking for fold JSONs in: {cv_dir}")

        # Collect fold JSONs, keeping newest per fold index
        by_idx: dict[int, Path] = {}
        for p in cv_dir.glob("fold_*.json"):
            try:
                idx = int(p.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if idx not in by_idx or p.stat().st_mtime > by_idx[idx].stat().st_mtime:
                by_idx[idx] = p

        if not by_idx:
            print("ERROR: No fold_*.json files found. "
                  f"Run spatial_CV_1 --biome-idx N for region '{region}' first.")
            sys.exit(1)

        folds = []
        for idx in sorted(by_idx):
            with open(by_idx[idx]) as f:
                folds.append(json.load(f))
            print(f"  Loaded fold {idx:2d}: {by_idx[idx].name}")

        print(f"\nLoaded {len(folds)} fold(s)")

        rows = []
        for fold in folds:
            td = fold.get("test_data", {})
            row: dict[str, Any] = {
                "fold_idx":      fold.get("fold_idx"),
                "biome_name":    fold.get("biome_name"),
                "n_test":        td.get("n_samples"),
                "n_test_pos":    td.get("n_positive"),
                "n_test_neg":    td.get("n_negative"),
                "positive_rate": (td.get("n_positive", 0) /
                                  max(td.get("n_samples", 1), 1)),
                "n_train":       fold.get("train_data", {}).get("n_samples"),
            }
            for m in METRIC_COLS:
                row[m] = td.get(m, np.nan)
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("fold_idx").reset_index(drop=True)

        summary_cols = [c for c in METRIC_COLS if c in df.columns]
        # Exclude folds with fewer than MIN_POSITIVES test positives: their metrics
        # (0.0 for PR-AUC, P@K, lift; NaN for ROC-AUC) would bias the aggregate.
        valid_df = df[pd.to_numeric(df["n_test_pos"], errors="coerce").fillna(0) >= MIN_POSITIVES]
        mean_row: dict[str, Any] = {"fold_idx": "MEAN", "biome_name": "— MEAN —"}
        std_row:  dict[str, Any] = {"fold_idx": "STD",  "biome_name": "— STD —"}
        for c in summary_cols:
            vals = pd.to_numeric(valid_df[c], errors="coerce").dropna()
            mean_row[c] = float(vals.mean()) if len(vals) > 0 else np.nan
            std_row[c]  = float(vals.std())  if len(vals) > 1 else np.nan

        df_out = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        print("\n" + "=" * 70)
        print("PER-BIOME RESULTS")
        print("=" * 70)
        print(f"\n{'Fold':>4}  {'Biome':<48} {'N_test':>8} {'Pos':>6} "
              f"{'ROC-AUC':>8} {'PR-AUC':>8} {'P@1%':>7} {'R@5%':>7} {'Lift@1%':>8}")
        print("-" * 120)
        for _, row in df_out.iterrows():
            idx_ = str(row["fold_idx"])
            bm   = str(row["biome_name"])[:48]
            n    = f"{int(row['n_test']):>8,}"     if pd.notna(row.get("n_test"))    else "     N/A"
            pos  = f"{int(row['n_test_pos']):>6}"  if pd.notna(row.get("n_test_pos")) else "   N/A"
            roc  = f"{row['roc_auc']:.4f}"         if pd.notna(row.get("roc_auc"))   else "   N/A "
            pr   = f"{row['pr_auc']:.4f}"          if pd.notna(row.get("pr_auc"))    else "   N/A "
            p1   = f"{row['precision_at_1pct']:.4f}" if pd.notna(row.get("precision_at_1pct")) else "  N/A "
            r5   = f"{row['recall_at_5pct']:.4f}"  if pd.notna(row.get("recall_at_5pct"))   else "  N/A "
            l1   = f"{row['lift_at_1pct']:.1f}x"  if pd.notna(row.get("lift_at_1pct"))      else "  N/A"
            print(f"{idx_:>4}  {bm:<48} {n} {pos} {roc:>8} {pr:>8} {p1:>7} {r5:>7} {l1:>8}")

        csv_path = cv_dir / f"lobo_summary_{timestamp}.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\nSummary saved: {csv_path}")
        print(f"Total time: {time.time()-t0:.1f}s")
        print("Done.")

    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nLog saved to: {log_path}")


# =============================================================================
# Entry point — Layer 2: biome-stratified metrics on existing scored parquets
# =============================================================================

def run_spatial_gen_main(region: str, model_id: str) -> None:
    """Evaluate model performance stratified by biome using existing scored parquets."""
    repo_root = get_repo_root()
    scratch   = os.environ.get("SCRATCH")

    # Locate scored parquets (prefer $SCRATCH)
    ml_models_dir = (
        Path(scratch) / f"outputs/{region}/results/ml_models"
        if scratch and (Path(scratch) / f"outputs/{region}/results/ml_models").exists()
        else repo_root / f"outputs/{region}/results/ml_models"
    )
    eval_dir = repo_root / f"outputs/{region}/results/spatial_generalisation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    region_label = region.replace("_", " ").upper()
    log_path     = eval_dir / f"spatial_generalisation_{timestamp}.txt"
    use_wandb    = False

    tee = Tee(log_path)
    sys.stdout = tee
    try:
        t_start = time.time()
        print("=" * 70)
        print(f"SPATIAL GENERALISATION — {region_label} (Layer 2: Biome-stratified)")
        print("=" * 70)
        print(f"Test years:  {TEST_START}–{TEST_END}")
        print(f"Output dir:  {eval_dir}")

        print("\n" + "=" * 70)
        print("STEP 1: LOAD BIOME RASTER")
        print("=" * 70)
        raster_path, shp_path = _biome_raster_paths(region)
        if not raster_path.exists():
            raise FileNotFoundError(
                f"Biome raster not found: {raster_path}\n"
                f"Run: python scripts/regions/{region}/2_preprocessing/gsn_preprocessing"
            )
        if not shp_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        biome_raster, _, int_to_biome = load_biome_raster_and_mapping(raster_path, shp_path)

        if wandb is not None:
            try:
                wandb.init(
                    project="spatial-cv",
                    entity=os.environ.get("WANDB_ENTITY"),
                    name=f"cv2_{region}_{model_id}_{timestamp}",
                    config={
                        "region":     region,
                        "model_id":   model_id,
                        "test_years": [TEST_START, TEST_END],
                        "layer":      2,
                    },
                )
                use_wandb = True
                print("W&B connected")
            except Exception as _err:
                print(f"W&B failed: {_err}")

        print("\n" + "=" * 70)
        print("STEP 2: FIND SCORED FILES")
        print("=" * 70)
        lgbm_file = find_scored_file(ml_models_dir, "lgbm", model_id)
        rf_file   = find_scored_file(ml_models_dir, "rf",   model_id)

        if lgbm_file is None:
            print("  WARNING: No LGBM scored file found — skipping LGBM evaluation")
        else:
            print(f"  LGBM: {lgbm_file.name}")
        if rf_file is None:
            print("  WARNING: No RF scored file found — skipping RF evaluation")
        else:
            print(f"  RF:   {rf_file.name}")
        if lgbm_file is None and rf_file is None:
            raise FileNotFoundError(
                f"No scored output files found in {ml_models_dir}\n"
                f"Run the full training pipeline (step 5) first."
            )

        all_results: list[pd.DataFrame] = []
        if lgbm_file is not None:
            print("\n" + "=" * 70)
            print("STEP 3a: EVALUATE LGBM")
            print("=" * 70)
            all_results.append(
                _evaluate_model_layer2(lgbm_file, "LGBM", biome_raster, int_to_biome)
            )
        if rf_file is not None:
            print("\n" + "=" * 70)
            print("STEP 3b: EVALUATE RF")
            print("=" * 70)
            all_results.append(
                _evaluate_model_layer2(rf_file, "RF", biome_raster, int_to_biome)
            )

        results = pd.concat(all_results, ignore_index=True)

        # ── Step 3c: Per-biome permutation feature importance (both models)
        fi_frames: list[pd.DataFrame] = []
        for mt, mfile in (("lgbm", lgbm_file), ("rf", rf_file)):
            if mfile is None:
                continue
            print("\n" + "=" * 70)
            print(f"STEP 3{'a' if mt == 'lgbm' else 'b'} (cont.): "
                  f"{mt.upper()} BIOME FEATURE IMPORTANCE (PERMUTATION)")
            print("=" * 70)
            fi = _compute_biome_feature_importance(
                region, model_id, mt, biome_raster, int_to_biome,
            )
            if fi is not None:
                fi_frames.append(fi)

        print("\n" + "=" * 70)
        print("STEP 4: SAVE RESULTS")
        print("=" * 70)
        csv_path = eval_dir / f"biome_metrics_{timestamp}.csv"
        pq_path  = eval_dir / f"biome_metrics_{timestamp}.parquet"
        results.to_csv(csv_path, index=False)
        results.to_parquet(pq_path, index=False)
        print(f"  CSV:     {csv_path}")
        print(f"  Parquet: {pq_path}")

        fi_csv_path: Optional[Path] = None
        if fi_frames:
            fi_all = pd.concat(fi_frames, ignore_index=True)
            fi_csv_path = eval_dir / f"biome_feature_importance_{timestamp}.csv"
            fi_all.to_csv(fi_csv_path, index=False)
            print(f"  Feature importance: {fi_csv_path}")
        else:
            fi_all = None
            print("  Feature importance: skipped (no model found)")

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        for model_label in results["model"].unique():
            m = results[results["model"] == model_label].copy()
            all_row = m[m["biome"] == "ALL"]
            other   = m[m["biome"] != "ALL"].sort_values("biome")
            m_sorted = pd.concat([all_row, other], ignore_index=True)
            print(f"\n{model_label}:")
            print(f"  {'Biome':<50} {'N':>8} {'Pos':>6} {'Neg:Pos':>9} "
                  f"{'ROC-AUC':>8} {'PR-AUC':>8} {'P@1%':>7}")
            print(f"  {'-'*50} {'-'*8} {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")
            for _, row in m_sorted.iterrows():
                roc  = f"{row['roc_auc']:.4f}"           if pd.notna(row.get('roc_auc'))           else "  N/A  "
                pr   = f"{row['pr_auc']:.4f}"            if pd.notna(row.get('pr_auc'))            else "  N/A  "
                p1   = f"{row['precision_at_1pct']:.4f}" if pd.notna(row.get('precision_at_1pct')) else "  N/A "
                npr  = f"{row['neg_pos_ratio']:.0f}:1"   if pd.notna(row.get('neg_pos_ratio'))     else "  N/A  "
                print(f"  {str(row['biome']):<50} {int(row['n_samples']):>8,} "
                      f"{int(row['n_positives']):>6} {npr:>9} {roc:>8} {pr:>8} {p1:>7}")

        # Print feature importance summary (top 5 per biome)
        if fi_all is not None and len(fi_all) > 0:
            print("\n" + "=" * 70)
            print("BIOME FEATURE IMPORTANCE SUMMARY (top 5 per biome)")
            print("=" * 70)
            for model_label in fi_all["model"].unique():
                fi_m = fi_all[fi_all["model"] == model_label]
                print(f"\n{model_label}:")
                for biome_name in sorted(fi_m["biome"].unique()):
                    top5 = fi_m[fi_m["biome"] == biome_name].sort_values("rank").head(5)
                    feats = ", ".join(top5["feature"].tolist())
                    print(f"  {biome_name:<50} {feats}")

        elapsed = time.time() - t_start
        print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print("=" * 70)

        if use_wandb:
            for _, row in results.iterrows():
                prefix = f"{row['model']}/{row['biome'].replace(' ', '_')}"
                log_dict: dict[str, Any] = {
                    f"{prefix}/n_samples":   row.get("n_samples"),
                    f"{prefix}/n_positives": row.get("n_positives"),
                }
                for metric in ("neg_pos_ratio",
                               "roc_auc", "pr_auc",
                               "precision_at_1pct", "precision_at_5pct", "precision_at_10pct",
                               "recall_at_1pct",    "recall_at_5pct",    "recall_at_10pct"):
                    v = row.get(metric)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        log_dict[f"{prefix}/{metric}"] = v
                wandb.log(log_dict)
            # Log top-1 feature importance per biome as W&B summary
            if fi_all is not None and len(fi_all) > 0:
                fi_top1 = fi_all[fi_all["rank"] == 1]
                for _, fi_row in fi_top1.iterrows():
                    method_prefix = fi_row.get("method", "fi")
                    key = (f"{method_prefix}/{fi_row['model']}/"
                           f"{str(fi_row['biome']).replace(' ', '_')}/top1_feature")
                    wandb.log({key: fi_row["feature"],
                               key.replace("top1_feature", "top1_importance_score"):
                                   fi_row["importance_score"]})
            wandb.log({"timing/total_seconds": elapsed})
            wandb.finish()

        print("Done.")

    finally:
        if use_wandb and wandb is not None and wandb.run is not None:
            wandb.finish()
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nLog saved to: {log_path}")


# =============================================================================
# Entry point — Layer 3: cross-continental transfer SA ↔ USA
# =============================================================================

def run_transfer_main(output_region: str = "south_america") -> None:
    """Cross-continental transfer evaluation: SA ↔ USA, for both LGBM and RF.

    Loads the production models trained in step 5 (no retraining) and scores
    them on the other continent's test set.  SA and USA share identical 73-feature
    sets, so feature columns are taken directly from the loaded model.

    Runs 4 combinations:
        SA-LGBM → USA,  USA-LGBM → SA
        SA-RF   → USA,  USA-RF   → SA
    """
    output_dir = resolve_cv_output_dir(output_region, "transfer")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path   = output_dir / f"transfer_results_{timestamp}.txt"
    use_wandb  = False

    tee = Tee(log_path)
    sys.stdout = tee
    try:
        t_global = time.time()
        print("=" * 70)
        print("CROSS-CONTINENTAL TRANSFER EVALUATION — SA ↔ USA (Layer 3)")
        print("Uses production models from step-5 training — no retraining.")
        print("=" * 70)
        print(f"Test years:   {TEST_START}-{TEST_END}")
        print(f"Output dir:   {output_dir}")

        # ── Sanity check: confirm feature sets are identical ─────────────────
        print("\n" + "=" * 70)
        print("STEP 0: VERIFY FEATURE SETS")
        print("=" * 70)
        sa_train_path  = resolve_parquet("south_america", "train_win5.parquet")
        usa_train_path = resolve_parquet("usa",           "train_win5.parquet")
        sa_cols  = get_feature_cols(sa_train_path)
        usa_cols = get_feature_cols(usa_train_path)
        sa_only  = set(sa_cols)  - set(usa_cols)
        usa_only = set(usa_cols) - set(sa_cols)
        print(f"SA features:  {len(sa_cols)}")
        print(f"USA features: {len(usa_cols)}")
        if sa_only:
            print(f"WARNING — SA-only features ({len(sa_only)}): {sorted(sa_only)}")
        if usa_only:
            print(f"WARNING — USA-only features ({len(usa_only)}): {sorted(usa_only)}")
        if not sa_only and not usa_only:
            print("Feature sets are identical — cross-scoring is exact.")

        if wandb is not None:
            try:
                wandb.init(
                    project="spatial-cv",
                    entity=os.environ.get("WANDB_ENTITY"),
                    name=f"transfer_sa_usa_{timestamp}",
                    config={
                        "n_sa_features":  len(sa_cols),
                        "n_usa_features": len(usa_cols),
                        "test_years":     [TEST_START, TEST_END],
                        "layer":          3,
                    },
                )
                use_wandb = True
                print("W&B connected")
            except Exception as _err:
                print(f"W&B failed: {_err}")

        # ── Run all 4 combinations ───────────────────────────────────────────
        directions = [
            ("south_america", "model1", "usa"),
            ("usa",           "model2", "south_america"),
        ]
        model_types_to_run = ["lgbm", "rf"]

        all_results: list[dict[str, Any]] = []
        for model_type in model_types_to_run:
            for src, src_model_id, tgt in directions:
                try:
                    result = _run_transfer_direction(
                        src, src_model_id, tgt, model_type, output_dir, timestamp
                    )
                    all_results.append(result)
                except FileNotFoundError as e:
                    print(f"\nSKIPPED {src}→{tgt} ({model_type.upper()}): {e}")
                gc.collect()
                report_memory_usage(f"after {src}→{tgt} ({model_type})")

        # ── Print results table ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        def _fmt(m: dict, key: str) -> str:
            v = m.get(key)
            return f"{v:.4f}" if v is not None else "  N/A"

        print(f"\n{'Direction':<32} {'Model':>5}  "
              f"{'ROC-AUC':>8} {'PR-AUC':>8} {'P@1%':>7} {'Lift@1%':>8}")
        print("-" * 75)
        for result in all_results:
            m   = result["test_metrics"]
            dir_str = f"{result['source_region']} → {result['target_region']}"
            print(f"{dir_str:<32} {result['model_type'].upper():>5}  "
                  f"{_fmt(m, 'roc_auc'):>8} {_fmt(m, 'pr_auc'):>8} "
                  f"{_fmt(m, 'precision_at_1pct'):>7} "
                  f"{m.get('lift_at_1pct', float('nan')):.1f}x")

        elapsed = time.time() - t_global
        output_json = {
            "experiment":  "cross_continental_transfer",
            "timestamp":   timestamp,
            "test_years":  [TEST_START, TEST_END],
            "results":     all_results,
            "timing":      {"total_seconds": elapsed, "total_minutes": elapsed / 60},
        }
        output_json = convert_numpy_types(output_json)

        json_path = output_dir / f"transfer_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)

        print(f"\nResults saved: {json_path}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if use_wandb:
            for result in all_results:
                m    = result["test_metrics"]
                pref = f"{result['source_region']}_to_{result['target_region']}_{result['model_type']}"
                wandb.log({
                    f"{pref}/roc_auc":           m.get("roc_auc",           float("nan")),
                    f"{pref}/pr_auc":            m.get("pr_auc",            float("nan")),
                    f"{pref}/precision_at_1pct": m.get("precision_at_1pct", float("nan")),
                    f"{pref}/lift_at_1pct":      m.get("lift_at_1pct",      float("nan")),
                    f"{pref}/precision_at_5pct": m.get("precision_at_5pct", float("nan")),
                    f"{pref}/brier_score":       m.get("brier_score",       float("nan")),
                    f"{pref}/n_test":            result["test_data"]["n_samples"],
                    f"{pref}/n_test_positive":   result["test_data"]["n_positive"],
                })
            wandb.log({"timing/total_seconds": elapsed})
            wandb.finish()

        print("Done.")

    finally:
        if use_wandb and wandb is not None and wandb.run is not None:
            wandb.finish()
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nLog saved to: {log_path}")


__all__ = [
    "run_lobo_main",
    "run_lobo_aggregate_main",
    "run_spatial_gen_main",
    "run_transfer_main",
    # helpers exposed for testing / external use
    "train_lgbm",
    "train_rf",
    "load_best_params",
    "load_rf_params",
    "resolve_best_params",
    "resolve_rf_best_params",
    "resolve_trained_model",
    "resolve_cv_output_dir",
]
