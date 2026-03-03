"""
Temporal split builders for tuning holdout and rolling cross-validation.

- Keeps time-aware validation logic in one place so tuning remains consistent
  and avoids data leakage across years.

Input:
- A dataframe with a year column and split configuration parameters.

Output:
- Index-based fold tuples and a split-info dictionary for logging/artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_year_max: int = 2014
    val_year_min: int = 2015
    val_year_max: int = 2017
    strategy: str = "rolling"
    rolling_folds: int = 5
    year_col: str = "year"


def _indices_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask)[0]


def build_holdout_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    years = df[cfg.year_col].values
    train_mask = years <= cfg.train_year_max
    val_mask = (years >= cfg.val_year_min) & (years <= cfg.val_year_max)
    train_idx = _indices_from_mask(train_mask)
    val_idx = _indices_from_mask(val_mask)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Holdout split has empty train or validation set.")
    info = {
        "strategy": "holdout",
        "train_years": f"year <= {cfg.train_year_max}",
        "val_years": f"{cfg.val_year_min} <= year <= {cfg.val_year_max}",
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }
    return [(train_idx, val_idx)], info


def build_rolling_splits(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    years = df[cfg.year_col].values
    unique_years = sorted(df[cfg.year_col].unique())
    val_window = cfg.val_year_max - cfg.val_year_min + 1
    end_years = list(range(cfg.train_year_max - (cfg.rolling_folds - 1), cfg.train_year_max + 1))
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_info: List[Dict] = []

    for fold_num, train_end in enumerate(end_years, start=1):
        val_start = train_end + 1
        val_end = train_end + val_window
        train_mask = years <= train_end
        val_mask = (years >= val_start) & (years <= val_end)
        train_idx = _indices_from_mask(train_mask)
        val_idx = _indices_from_mask(val_mask)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
        fold_info.append(
            {
                "fold": fold_num,
                "train_years": f"year <= {train_end}",
                "val_years": f"{val_start} <= year <= {val_end}",
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            }
        )

    if not folds:
        raise ValueError(f"No rolling folds created. Available years: {unique_years}")

    info = {
        "strategy": "rolling",
        "rolling_folds_requested": int(cfg.rolling_folds),
        "rolling_folds_built": int(len(folds)),
        "folds": fold_info,
        "base_train_year_max": int(cfg.train_year_max),
        "base_val_year_min": int(cfg.val_year_min),
        "base_val_year_max": int(cfg.val_year_max),
    }
    return folds, info


def build_splits(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    strategy = cfg.strategy.strip().lower()
    if strategy == "holdout":
        return build_holdout_split(df, cfg)
    if strategy == "rolling":
        return build_rolling_splits(df, cfg)
    raise ValueError(f"Unknown CV strategy '{cfg.strategy}'. Expected one of: holdout, rolling.")

