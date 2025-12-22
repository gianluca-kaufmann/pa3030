#!/usr/bin/env python3

"""Model1 LightGBM Hyperparameter Tuning Script

Purpose: Perform hyperparameter tuning for a LightGBM model using
         per-year sampling (keep all positives; cap negatives per year)
         from train.parquet with a strict temporal split
         (train: year <= 2014, val: 2015 <= year <= 2017).

Input:   `train.parquet`
Output:  - lgbm_best_params.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit


class Tee:
    """Write to both stdout and a file."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.stdout.flush()
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


# =============================================================================
# Utility Functions
# =============================================================================


def get_n_jobs() -> int:
    """Get number of jobs from SLURM_CPUS_PER_TASK, returning -1 on missing/invalid values."""
    try:
        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus is None:
            return -1
        return int(cpus)
    except (ValueError, TypeError):
        return -1


# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 42

# Temporal split configuration
TRAIN_YEAR_MAX = 2014  # train_idx = year <= 2014
VAL_YEAR_MIN = 2015  # val_idx = 2015 <= year <= 2017
VAL_YEAR_MAX = 2017
MIN_YEAR = int(os.environ.get("TRANSITION_MIN_YEAR", "2001"))  # drop year 2000 by default

# Per-year negative cap (keep all positives)
MAX_NEG_PER_YEAR = int(os.environ.get("MAX_NEG_PER_YEAR", "100000"))

# Columns to exclude from features
EXCLUDE_COLS = {
    "transition_01",  # Target variable
    "WDPA_b1",  # Leakage
    "WDPA_prev",  # Leakage
    "x",  # Coordinate
    "y",  # Coordinate
    "row",  # Identifier
    "col",  # Identifier
    "year",  # Temporal identifier (kept for splitting)
}

# RandomizedSearchCV configuration
N_ITER_LGBM = 50

# LightGBM parameter grid (base; scale_pos_weight will be extended with auto)
LGBM_PARAM_GRID_BASE = {
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 15, 25],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "scale_pos_weight": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 0.1, 1.0],
}

# LightGBM fixed parameters
LGBM_FIXED_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_jobs": get_n_jobs(),
    "boosting_type": "gbdt",
    "objective": "binary",
    "verbose": -1,
    # Use a high n_estimators value in combination with early stopping
    "n_estimators": 2000,
}


# =============================================================================
# Utility Functions
# =============================================================================


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(
        obj,
        (
            np.integer,
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        try:
            if hasattr(obj, "item"):
                return obj.item()
            else:
                return obj
        except (ValueError, TypeError, AttributeError):
            return str(obj)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get valid feature columns, excluding identifiers and leakage columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [
        col for col in numeric_cols if col.lower() not in {c.lower() for c in EXCLUDE_COLS}
    ]
    return feature_cols


def resolve_train_parquet() -> Path:
    """Locate train.parquet (prefer $SCRATCH if present)."""
    repo_root = Path(__file__).resolve().parents[3]
    scratch_root = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    candidates = []
    if scratch_root is not None:
        candidates.append(scratch_root / "data/ml/train.parquet")
    candidates.append(repo_root / "data/ml/train.parquet")

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError("train.parquet not found in expected locations")


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    start_time = time.time()

    # -------------------------------------------------------------------------
    # Setup paths
    # -------------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[3]
    script_dir = Path(__file__).resolve().parent

    train_path = resolve_train_parquet()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize W&B
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY not found in environment")
    if not wandb_entity:
        print("Warning: WANDB_ENTITY not found in environment (will use default)")

    use_wandb = False
    try:
        print("Initializing Weights & Biases...")
        wandb.init(
            project="ml-hyperparameter-tuning-lgbm",
            entity=wandb_entity,
            name=f"model1_tuning_lgbm_{timestamp}",
            config={
                "random_state": RANDOM_STATE,
                "temporal_split": {
                    "train_year_max": TRAIN_YEAR_MAX,
                    "val_year_min": VAL_YEAR_MIN,
                    "val_year_max": VAL_YEAR_MAX,
                },
                "n_iter_lgbm": N_ITER_LGBM,
                "max_neg_per_year": MAX_NEG_PER_YEAR,
            },
        )
        use_wandb = True
        print("W&B connected\n")
    except Exception as err:
        print(f"W&B initialization failed: {err}\n")

    print("=" * 70)
    print("MODEL1 LIGHTGBM HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"\nInput:  {train_path}")
    print(f"Output: {script_dir}")

    # -------------------------------------------------------------------------
    # Step 1: Load data and apply per-year sampling
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA AND APPLY PER-YEAR SAMPLING")
    print("=" * 70)

    print(f"\nLoading train.parquet...")
    load_start = time.time()
    df = pd.read_parquet(train_path)
    load_time = time.time() - load_start
    print(f"  Loaded {len(df):,} rows in {load_time:.1f}s")

    # Downcast numeric dtypes to reduce memory usage
    print("\nDowncasting numeric dtypes (float64→float32, int64→int32)...")
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = df[col].astype("int32")
    print("  Downcasting completed")

    target_col = "transition_01"

    # Check target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Drop rows with missing target
    df_clean = df.dropna(subset=[target_col])
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        print(f"\nDropped {dropped:,} rows with missing target")

    # Drop year 2000 for transition modeling (avoid treating existing PAs as transitions)
    if "year" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["year"] >= MIN_YEAR].copy()
        removed = before - len(df_clean)
        if removed > 0:
            print(f"\nDropped {removed:,} rows with year < {MIN_YEAR} (TRANSITION_MIN_YEAR)")

    print(f"\nDataset after cleaning: {len(df_clean):,} rows")

    # Per-year sampling: keep all positives, cap negatives per year
    print(
        f"\nApplying per-year sampling: keep all positives, "
        f"cap negatives per year at {MAX_NEG_PER_YEAR:,}"
    )
    sampled_frames = []
    years = sorted(df_clean["year"].unique())
    for year in years:
        year_df = df_clean[df_clean["year"] == year]
        pos_mask = year_df[target_col] > 0
        pos = year_df[pos_mask]
        neg = year_df[~pos_mask]

        if len(neg) > MAX_NEG_PER_YEAR:
            neg = neg.sample(n=MAX_NEG_PER_YEAR, random_state=RANDOM_STATE, replace=False)

        year_sampled = pd.concat([pos, neg], axis=0)
        sampled_frames.append(year_sampled)

        print(
            f"  Year {year}: kept {len(pos):,} positives, "
            f"{len(neg):,} negatives (original negatives: {len(year_df) - len(pos):,})"
        )

    df_sampled = pd.concat(sampled_frames, axis=0)
    df_sampled = df_sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"\nFinal sampled dataset: {len(df_sampled):,} rows")

    # Get feature columns
    feature_cols = get_feature_columns(df_sampled)
    excluded_cols = sorted(EXCLUDE_COLS & set(df_sampled.columns))

    print(f"\nUsing {len(feature_cols)} features")
    print(f"Excluded columns ({len(excluded_cols)}): {excluded_cols}")

    # Target distribution
    pos = (df_sampled[target_col] > 0).sum()
    neg = (df_sampled[target_col] == 0).sum()
    pos_pct = pos / len(df_sampled) * 100

    print("\n" + "-" * 40)
    print("Sampled dataset target distribution:")
    print(f"  No transition (0): {neg:>12,}  ({100 - pos_pct:.3f}%)")
    print(f"  Transition (0→1):  {pos:>12,}  ({pos_pct:.3f}%)")
    print(f"  Class ratio:       1 : {neg / max(pos, 1):.1f}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # Step 2: Create temporal split
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: CREATE TEMPORAL SPLIT")
    print("=" * 70)

    # Create train and validation indices based on year
    train_idx = df_sampled["year"] <= TRAIN_YEAR_MAX
    val_idx = (df_sampled["year"] >= VAL_YEAR_MIN) & (df_sampled["year"] <= VAL_YEAR_MAX)

    df_train = df_sampled[train_idx].copy()
    df_val = df_sampled[val_idx].copy()

    print(f"\nTemporal split configuration:")
    print(f"  Train: year <= {TRAIN_YEAR_MAX}")
    print(f"  Val:   {VAL_YEAR_MIN} <= year <= {VAL_YEAR_MAX}")

    print(f"\nTrain set: {len(df_train):,} rows")
    print(f"Val set:   {len(df_val):,} rows")

    # Check years in each split
    train_years = sorted(df_train["year"].unique())
    val_years = sorted(df_val["year"].unique())
    print(f"\nTrain years: {train_years}")
    print(f"Val years:   {val_years}")

    # Target distribution
    train_pos = (df_train[target_col] > 0).sum()
    train_neg = (df_train[target_col] == 0).sum()
    train_pos_pct = train_pos / len(df_train) * 100

    val_pos = (df_val[target_col] > 0).sum()
    val_neg = (df_val[target_col] == 0).sum()
    val_pos_pct = val_pos / len(df_val) * 100

    print("\n" + "-" * 40)
    print("Train set distribution:")
    print(f"  No transition (0): {train_neg:>12,}  ({100 - train_pos_pct:.3f}%)")
    print(f"  Transition (0→1):  {train_pos:>12,}  ({train_pos_pct:.3f}%)")
    print("\nVal set distribution:")
    print(f"  No transition (0): {val_neg:>12,}  ({100 - val_pos_pct:.3f}%)")
    print(f"  Transition (0→1):  {val_pos:>12,}  ({val_pos_pct:.3f}%)")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # Step 3: Prepare features and target
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: PREPARE FEATURES AND TARGET")
    print("=" * 70)

    # Training set
    y_train = (df_train[target_col] > 0).astype(np.int8)
    X_train = df_train[feature_cols]

    # Validation set
    y_val = (df_val[target_col] > 0).astype(np.int8)
    X_val = df_val[feature_cols]

    print(f"\nTrain feature matrix shape: {X_train.shape}")
    print(f"Train target shape: {y_train.shape}")
    print(f"Val feature matrix shape: {X_val.shape}")
    print(f"Val target shape: {y_val.shape}")

    # Store sizes for summary
    n_train = len(X_train)
    n_val = len(X_val)

    # Combine train and val for RandomizedSearchCV with PredefinedSplit
    # -1 indicates train, 0 indicates test/val
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = np.concatenate([y_train.values, y_val.values])
    split_indices = np.concatenate(
        [
            np.full(n_train, -1),  # -1 for train
            np.zeros(n_val),  # 0 for val/test
        ]
    )
    predefined_split = PredefinedSplit(split_indices)

    print(f"\nCombined dataset for CV: {len(X_combined):,} rows")
    print(f"  Train: {n_train:,} rows (indices -1)")
    print(f"  Val:   {n_val:,} rows (indices 0)")

    # -------------------------------------------------------------------------
    # Step 4: Prepare parameter grid (including auto scale_pos_weight)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: PREPARE PARAMETER GRID")
    print("=" * 70)

    # Compute auto scale_pos_weight from train distribution
    auto_scale_pos_weight = train_neg / max(train_pos, 1)
    print(f"\nAuto scale_pos_weight (train_neg/train_pos): {auto_scale_pos_weight:.3f}")

    lgbm_param_grid = deepcopy(LGBM_PARAM_GRID_BASE)
    spw_values = list(lgbm_param_grid["scale_pos_weight"])
    spw_values.append(float(auto_scale_pos_weight))
    # Remove potential duplicates while preserving order
    seen = set()
    unique_spw_values = []
    for v in spw_values:
        if v not in seen:
            seen.add(v)
            unique_spw_values.append(v)
    lgbm_param_grid["scale_pos_weight"] = unique_spw_values

    print("\nParameter grid:")
    for param, values in lgbm_param_grid.items():
        print(f"  {param}: {values}")

    print("\nFixed parameters:")
    for param, value in LGBM_FIXED_PARAMS.items():
        print(f"  {param}: {value}")

    # -------------------------------------------------------------------------
    # Step 5: LightGBM hyperparameter tuning with early stopping
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: LIGHTGBM HYPERPARAMETER TUNING")
    print("=" * 70)

    print(
        f"\nUsing temporal split: train (year <= {TRAIN_YEAR_MAX}) / "
        f"val ({VAL_YEAR_MIN} <= year <= {VAL_YEAR_MAX})"
    )
    print(f"Sampling {N_ITER_LGBM} parameter combinations with RandomizedSearchCV")

    # Create LightGBM model
    lgbm_model = LGBMClassifier(**LGBM_FIXED_PARAMS)

    # Randomized search with fixed train/val split
    print(f"\nStarting randomized search ({N_ITER_LGBM} iterations) with early stopping...")
    print("This may take a while...\n")

    random_search_lgbm = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=lgbm_param_grid,
        n_iter=N_ITER_LGBM,
        cv=predefined_split,  # Use fixed train/val split
        scoring="average_precision",  # PR-AUC (better for imbalanced data)
        n_jobs=1,  # Avoid nested parallelism
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )

    # Early stopping is configured via fit parameters; eval_set uses the held-out val set
    fit_params_lgbm = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "average_precision",
        "early_stopping_rounds": 100,
        "verbose": False,
    }

    tune_start_lgbm = time.time()
    random_search_lgbm.fit(X_combined, y_combined, **fit_params_lgbm)
    tune_time_lgbm = time.time() - tune_start_lgbm

    # best_score_ is the score on the validation set
    best_val_score_lgbm = random_search_lgbm.best_score_

    print(
        f"\nLightGBM randomized search completed in "
        f"{tune_time_lgbm:.1f}s ({tune_time_lgbm/60:.1f} min)"
    )
    print("\nBest parameters:")
    for param, value in random_search_lgbm.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest val score (PR-AUC): {best_val_score_lgbm:.4f}")

    # Log to wandb
    if use_wandb:
        wandb.log(
            {
                "lgbm/best_val_score": float(best_val_score_lgbm),
                "lgbm/tuning_time_seconds": tune_time_lgbm,
                "lgbm/best_params": random_search_lgbm.best_params_,
                "lgbm/auto_scale_pos_weight": float(auto_scale_pos_weight),
            }
        )

    # Save LGBM best parameters
    lgbm_best_params = {
        "best_params": random_search_lgbm.best_params_,
        "best_val_score": float(best_val_score_lgbm),
        "param_grid": lgbm_param_grid,
        "fixed_params": LGBM_FIXED_PARAMS,
        "n_iter": N_ITER_LGBM,
        "scoring": "average_precision",
        "tuning_time_seconds": tune_time_lgbm,
        "split_info": {
            "train_years": f"year <= {TRAIN_YEAR_MAX}",
            "val_years": f"{VAL_YEAR_MIN} <= year <= {VAL_YEAR_MAX}",
            "n_train": int(n_train),
            "n_val": int(n_val),
            "auto_scale_pos_weight": float(auto_scale_pos_weight),
        },
    }
    lgbm_best_params = convert_numpy_types(lgbm_best_params)

    lgbm_output_path = script_dir / "lgbm_best_params.json"
    with open(lgbm_output_path, "w") as f:
        json.dump(lgbm_best_params, f, indent=2)
    print(f"\nLGBM best parameters saved to: {lgbm_output_path}")

    # Free memory
    del random_search_lgbm, lgbm_model
    gc.collect()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset size (cleaned):        {len(df_clean):,} rows")
    print(f"Dataset size (sampled):        {len(df_sampled):,} rows")
    print(f"Features:                      {len(feature_cols)}")
    print("\nTemporal split:")
    print(f"  Train: year <= {TRAIN_YEAR_MAX}  ({n_train:,} rows)")
    print(f"  Val:   {VAL_YEAR_MIN} <= year <= {VAL_YEAR_MAX}  ({n_val:,} rows)")
    print("\nLightGBM:")
    print(f"  Best val score (PR-AUC):     {lgbm_best_params['best_val_score']:.4f}")
    print(
        f"  Tuning time:                 "
        f"{tune_time_lgbm:.1f}s ({tune_time_lgbm/60:.1f} min)"
    )
    print(f"  Best parameters saved to:    {lgbm_output_path}")
    print(f"\nTotal time:                    {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print("Done.")

    if use_wandb:
        wandb.log(
            {
                "summary/total_time_seconds": total_time,
                "summary/total_time_minutes": total_time / 60,
                "summary/lgbm_best_val_score": float(lgbm_best_params["best_val_score"]),
                "summary/auto_scale_pos_weight": float(auto_scale_pos_weight),
                "status": "success",
            }
        )
        wandb.finish()


if __name__ == "__main__":
    # Set up output file capture
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "outputs/Results/ml_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"model1_tuning_lgbm_{timestamp}.txt"

    tee = Tee(output_file)
    sys.stdout = tee

    try:
        main()
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nOutput saved to: {output_file}")



