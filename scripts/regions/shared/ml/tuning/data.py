from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSummary:
    n_rows_raw: int
    n_rows_clean: int
    n_rows_sampled: int
    n_features: int
    n_pos: int
    n_neg: int


def resolve_train_parquet(repo_root: Path, region: str, scratch_root: Path | None = None) -> Path:
    candidates: List[Path] = []
    if scratch_root is not None:
        candidates.append(scratch_root / f"data/{region}/ml/train.parquet")
    candidates.append(repo_root / f"data/{region}/ml/train.parquet")

    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"train.parquet not found for region={region}. Checked: {candidates}")


def get_feature_columns(df: pd.DataFrame, exclude_cols: set[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    exclude_lower = {c.lower() for c in exclude_cols}
    return [col for col in numeric_cols if col.lower() not in exclude_lower]


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64"]).columns
    int_cols = out.select_dtypes(include=["int64"]).columns
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def clean_and_filter_risk_set(
    df: pd.DataFrame, target_col: str, min_year: int, year_col: str = "year"
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    if "WDPA_prev" not in df.columns:
        raise ValueError("WDPA_prev column missing; required for risk-set filtering.")
    if year_col not in df.columns:
        raise ValueError(f"Year column '{year_col}' not found in dataframe")

    clean = df.dropna(subset=[target_col]).copy()
    clean = clean[clean[year_col] >= min_year].copy()
    clean = clean[clean["WDPA_prev"] == 0].copy()
    return clean


def get_negative_cap(
    year_pos_count: int,
    base_cap: int,
    adaptive_enabled: bool,
    target_neg_pos_ratio: float,
    min_cap: int,
    max_cap: int,
) -> int:
    if not adaptive_enabled:
        return base_cap
    adaptive_cap = int(max(min_cap, min(max_cap, year_pos_count * target_neg_pos_ratio)))
    return max(base_cap, adaptive_cap)


def sample_per_year(
    df: pd.DataFrame,
    target_col: str,
    random_state: int,
    base_max_neg_per_year: int,
    adaptive_enabled: bool = False,
    target_neg_pos_ratio: float = 50.0,
    adaptive_min_cap: int = 20_000,
    adaptive_max_cap: int = 500_000,
    year_col: str = "year",
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]]]:
    sampled_frames: List[pd.DataFrame] = []
    stats: Dict[int, Dict[str, int]] = {}

    for year in sorted(df[year_col].unique()):
        year_df = df[df[year_col] == year]
        pos = year_df[year_df[target_col] > 0]
        neg = year_df[year_df[target_col] == 0]
        cap = get_negative_cap(
            year_pos_count=len(pos),
            base_cap=base_max_neg_per_year,
            adaptive_enabled=adaptive_enabled,
            target_neg_pos_ratio=target_neg_pos_ratio,
            min_cap=adaptive_min_cap,
            max_cap=adaptive_max_cap,
        )
        if len(neg) > cap:
            neg = neg.sample(n=cap, random_state=random_state, replace=False)
        sampled = pd.concat([pos, neg], axis=0)
        sampled_frames.append(sampled)
        stats[int(year)] = {
            "pos_kept": int(len(pos)),
            "neg_kept": int(len(neg)),
            "neg_original": int((year_df[target_col] == 0).sum()),
            "neg_cap": int(cap),
        }

    out = pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out, stats


def prepare_tuning_dataset(
    train_path: Path,
    target_col: str,
    min_year: int,
    random_state: int,
    exclude_cols: set[str],
    max_neg_per_year: int,
    adaptive_cap_enabled: bool,
    target_neg_pos_ratio: float,
    adaptive_min_cap: int,
    adaptive_max_cap: int,
) -> Tuple[pd.DataFrame, List[str], DataSummary, Dict[int, Dict[str, int]]]:
    raw = pd.read_parquet(train_path)
    clean = clean_and_filter_risk_set(raw, target_col=target_col, min_year=min_year)
    sampled, year_stats = sample_per_year(
        clean,
        target_col=target_col,
        random_state=random_state,
        base_max_neg_per_year=max_neg_per_year,
        adaptive_enabled=adaptive_cap_enabled,
        target_neg_pos_ratio=target_neg_pos_ratio,
        adaptive_min_cap=adaptive_min_cap,
        adaptive_max_cap=adaptive_max_cap,
    )
    sampled = _downcast_numeric(sampled)
    feature_cols = get_feature_columns(sampled, exclude_cols=exclude_cols)
    pos = int((sampled[target_col] > 0).sum())
    neg = int((sampled[target_col] == 0).sum())
    summary = DataSummary(
        n_rows_raw=int(len(raw)),
        n_rows_clean=int(len(clean)),
        n_rows_sampled=int(len(sampled)),
        n_features=int(len(feature_cols)),
        n_pos=pos,
        n_neg=neg,
    )
    return sampled, feature_cols, summary, year_stats
