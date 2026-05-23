"""
Search-space and fixed-parameter definitions for tuning classifiers.

- Separates model-specific parameter ranges from execution code so tuning modes
  (fast vs paper) are easy to compare and maintain.

Input:
- Tuning mode, random seed/CPU settings, and class-imbalance signal from labels.

Output:
- Fixed estimator parameters, randomized-search grids, Optuna bounds, and
  auto-computed class-weight guidance.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def get_lgbm_fixed_params(random_state: int, n_jobs: int) -> Dict:
    return {
        "random_state": random_state,
        "n_jobs": n_jobs,
        "boosting_type": "gbdt",
        "objective": "binary",
        "verbose": -1,
    }


def get_rf_fixed_params(random_state: int, n_jobs: int) -> Dict:
    return {
        "random_state": random_state,
        "n_jobs": n_jobs,
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "verbose": 0,
    }


def get_lgbm_randomized_space(mode: str, auto_scale_pos_weight: float) -> Dict[str, List]:
    """
    Return the RandomizedSearchCV distribution for LightGBM.

    ``scale_pos_weight`` is intentionally absent: it is computed deterministically
    from the training fold (neg/pos) inside the fitting code, not tuned as a
    free hyperparameter.  Tuning it on a sampled dataset with distorted class
    balance would optimise the wrong regime and invalidate PR-AUC comparisons.
    """
    mode = mode.lower()
    base = {
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 15, 25],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.9],
        "colsample_bynode": [0.5, 0.7, 0.9],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0, 5.0],
        "min_split_gain": [0.0, 0.001, 0.01, 0.1],
        "path_smooth": [0.0, 0.1, 1.0],
    }
    if mode == "paper":
        base.update(
            {
                "num_leaves": [31, 63, 127, 255],
                "max_depth": [-1, 10, 15, 25, 35],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "min_child_samples": [10, 20, 40, 80, 120],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bynode": [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0, 2.0, 5.0],
                "reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
                "min_split_gain": [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
                "path_smooth": [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            }
        )
    return base


def get_rf_randomized_space(mode: str) -> Dict[str, List]:
    mode = mode.lower()
    if mode == "paper":
        return {
            "n_estimators": [200, 400, 800],
            "max_depth": [15, 20, 30, None],
            "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5],
            "min_samples_leaf": [1, 2, 5, 10],
            "min_samples_split": [2, 5, 10, 20],
            "bootstrap": [True],
        }
    return {
        "n_estimators": [200, 500],
        "max_depth": [15, 20, None],
        "max_features": ["sqrt", "log2", 0.3],
        "min_samples_leaf": [2, 5, 10],
    }


def get_lgbm_optuna_bounds(mode: str, auto_scale_pos_weight: float) -> Dict:
    """
    Return Optuna hyperparameter bounds for LightGBM.

    ``scale_pos_weight`` is intentionally absent: it is computed per fold
    inside ``optimize_lgbm_optuna`` as ``n_neg / n_pos``, matching the
    production training convention and keeping evaluation in the correct
    rare-event regime regardless of the sampled class balance.
    """
    if mode.lower() == "paper":
        return {
            "num_leaves": (31, 255),
            "max_depth_choices": [-1, 10, 15, 20, 25, 35],
            "learning_rate": (0.01, 0.12),
            "min_child_samples": (10, 160),
            "subsample": (0.6, 1.0),
            "colsample_bynode": (0.3, 1.0),
            "reg_alpha": (1e-6, 10.0),
            "reg_lambda": (1e-6, 20.0),
            "min_split_gain": (1e-6, 1.0),
            "path_smooth": (1e-3, 20.0),
            "n_estimators": (300, 1500),
        }
    return {
        "num_leaves": (31, 127),
        "max_depth_choices": [-1, 15, 25],
        "learning_rate": (0.02, 0.12),
        "min_child_samples": (20, 120),
        "subsample": (0.7, 1.0),
        "colsample_bynode": (0.4, 1.0),
        "reg_alpha": (1e-6, 5.0),
        "reg_lambda": (1e-6, 10.0),
        "min_split_gain": (1e-6, 0.5),
        "path_smooth": (1e-3, 10.0),
        "n_estimators": (300, 1800),
    }


def get_rf_optuna_bounds(mode: str) -> Dict:
    if mode.lower() == "paper":
        return {
            "n_estimators": (150, 400),
            "max_depth_choices": [None, 15, 20, 25, 30, 40],
            "max_features_choices": ["sqrt", "log2", 0.2, 0.3, 0.5],
            "min_samples_leaf": (1, 15),
            "min_samples_split": (2, 30),
            "bootstrap_choices": [True],
        }
    return {
        "n_estimators": (100, 300),
        "max_depth_choices": [None, 15, 20, 25],
        "max_features_choices": ["sqrt", "log2", 0.3],
        "min_samples_leaf": (1, 10),
        "min_samples_split": (2, 20),
        "bootstrap_choices": [True],
    }


def compute_auto_scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y > 0).sum())
    neg = float((y == 0).sum())
    return float(neg / max(pos, 1.0))


def get_lgbm_stage2_fixed_params(random_state: int, n_jobs: int) -> Dict:
    """Fixed LightGBM params for Stage 2 LambdaRank tuning/training."""
    return {
        "random_state": random_state,
        "n_jobs": n_jobs,
        "boosting_type": "gbdt",
        "objective": "lambdarank",
        "metric": "ndcg",
        "verbose": -1,
        "lambdarank_truncation_level": 100,
        "max_query_size": 2_000_000,
    }


def get_lgbm_stage2_optuna_bounds(mode: str) -> Dict:
    """Optuna bounds for Stage 2 (no scale_pos_weight).

    lambdarank_truncation_level: median k@1% across groups is ~2000 for SA/SEA.
    Training with truncation_level << k@1% starves positives outside the top-K
    of any gradient signal. Range [50, 500] covers 25th–90th percentile of
    group-level k@1% values and is computationally feasible.
    """
    base = get_lgbm_optuna_bounds(mode, auto_scale_pos_weight=1.0)
    base["lambdarank_truncation_level"] = (50, 500)
    base["n_estimators"] = (200, 3000)
    return base
