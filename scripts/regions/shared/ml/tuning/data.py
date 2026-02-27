"""
Data loading, filtering, and class-balanced sampling for tuning datasets.

- Prepares very large train parquet files into a memory-feasible risk-set sample
  while preserving temporal class structure for robust hyperparameter search.

Input:
- Region train parquet path plus sampling/filter configuration (target, years,
  negative caps, exclusions, random seed).

Output:
- A sampled tuning dataframe, selected feature list, dataset summary, and
  per-year sampling statistics.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DataSummary:
    n_rows_raw: int
    n_rows_clean: int
    n_rows_sampled: int
    n_features: int
    n_pos: int
    n_neg: int


def resolve_train_parquet(
    repo_root: Path,
    region: str,
    scratch_root: Path | None = None,
    split_version: str = "main",
) -> Path:
    """
    Locate the tuning train parquet across split-aware and legacy layouts.

    Search order prioritizes SCRATCH over repo root and requested split over
    fallback splits for robustness.
    """
    split = split_version.strip() or "main"
    split_candidates: List[str] = []
    for item in (split, "main", "robustness", ""):
        if item not in split_candidates:
            split_candidates.append(item)

    base_dirs = [
        f"outputs/{region}/results",
        f"data/{region}/ml",
    ]
    filenames = [
        "train_win5.parquet",
        "train.parquet",
    ]

    roots: List[Path] = []
    if scratch_root is not None:
        roots.append(scratch_root)
    roots.append(repo_root)

    candidates: List[Path] = []
    for root in roots:
        for base_dir in base_dirs:
            for split_dir in split_candidates:
                prefix = f"{base_dir}/{split_dir}" if split_dir else base_dir
                for filename in filenames:
                    candidates.append(root / f"{prefix}/{filename}")

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"Train parquet not found for region={region}, split={split}. Checked: {candidates}"
    )


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
    parquet_file = pq.ParquetFile(train_path)
    schema = parquet_file.schema_arrow

    numeric_cols = [
        name
        for name, field in zip(schema.names, schema)
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
    ]
    feature_cols = [col for col in numeric_cols if col.lower() not in {c.lower() for c in exclude_cols}]
    keep_cols = list(dict.fromkeys(feature_cols + [target_col, "year", "WDPA_prev"]))

    n_rows_raw = int(parquet_file.metadata.num_rows) if parquet_file.metadata is not None else 0

    # Pass 1: gather year-level class counts after risk-set and min-year filtering.
    year_pos_counts: Dict[int, int] = {}
    year_neg_counts: Dict[int, int] = {}
    n_rows_clean = 0
    for batch in parquet_file.iter_batches(batch_size=200_000, columns=[target_col, "year", "WDPA_prev"], use_threads=True):
        batch_table = batch
        target_np = batch_table[target_col].to_numpy(zero_copy_only=False)
        year_np = batch_table["year"].to_numpy(zero_copy_only=False)
        wdpa_prev_np = batch_table["WDPA_prev"].to_numpy(zero_copy_only=False)

        valid_mask = ~batch_table[target_col].is_null().to_numpy(zero_copy_only=False)
        base_mask = valid_mask & (year_np >= min_year) & (wdpa_prev_np == 0)
        if not base_mask.any():
            continue

        y_valid = (target_np[base_mask] > 0).astype(np.int8)
        years_valid = year_np[base_mask].astype(np.int32)
        n_rows_clean += int(len(y_valid))

        unique_years = np.unique(years_valid)
        for year in unique_years:
            year = int(year)
            year_mask = years_valid == year
            pos_count = int(y_valid[year_mask].sum())
            total_count = int(year_mask.sum())
            neg_count = total_count - pos_count
            year_pos_counts[year] = year_pos_counts.get(year, 0) + pos_count
            year_neg_counts[year] = year_neg_counts.get(year, 0) + neg_count

        del batch_table, target_np, year_np, wdpa_prev_np, valid_mask, base_mask, y_valid, years_valid

    if n_rows_clean == 0:
        raise ValueError(f"No rows remain after filtering train_path={train_path}, min_year={min_year}.")

    year_neg_caps: Dict[int, int] = {}
    year_stats: Dict[int, Dict[str, int]] = {}
    for year in sorted(set(year_pos_counts.keys()) | set(year_neg_counts.keys())):
        pos_count = int(year_pos_counts.get(year, 0))
        neg_original = int(year_neg_counts.get(year, 0))
        cap = get_negative_cap(
            year_pos_count=pos_count,
            base_cap=max_neg_per_year,
            adaptive_enabled=adaptive_cap_enabled,
            target_neg_pos_ratio=target_neg_pos_ratio,
            min_cap=adaptive_min_cap,
            max_cap=adaptive_max_cap,
        )
        year_neg_caps[year] = int(cap)
        year_stats[year] = {
            "pos_kept": pos_count,
            "neg_kept": 0,
            "neg_original": neg_original,
            "neg_cap": int(cap),
        }

    # Pass 2: stream rows and keep all positives + bounded random negatives per year.
    parquet_file = pq.ParquetFile(train_path)
    rng = np.random.default_rng(random_state)
    pos_frames: List[pd.DataFrame] = []
    neg_frames_by_year: Dict[int, pd.DataFrame] = {}
    neg_scores_by_year: Dict[int, np.ndarray] = {}

    for batch in parquet_file.iter_batches(batch_size=200_000, columns=keep_cols, use_threads=True):
        batch_table = batch
        target_np = batch_table[target_col].to_numpy(zero_copy_only=False)
        year_np = batch_table["year"].to_numpy(zero_copy_only=False)
        wdpa_prev_np = batch_table["WDPA_prev"].to_numpy(zero_copy_only=False)

        valid_mask = ~batch_table[target_col].is_null().to_numpy(zero_copy_only=False)
        base_mask = valid_mask & (year_np >= min_year) & (wdpa_prev_np == 0)
        if not base_mask.any():
            continue

        target_valid = target_np[base_mask]
        years_valid = year_np[base_mask].astype(np.int32)
        y_bin = (target_valid > 0).astype(np.int8)

        filtered_table = batch_table.filter(pa.array(base_mask))
        filtered_df = filtered_table.to_pandas()
        filtered_df[target_col] = y_bin

        pos_df = filtered_df[y_bin > 0]
        if not pos_df.empty:
            pos_frames.append(pos_df)

        neg_df = filtered_df[y_bin == 0]
        if not neg_df.empty:
            for year in np.unique(neg_df["year"].to_numpy()):
                year_int = int(year)
                cap = int(year_neg_caps.get(year_int, max_neg_per_year))
                if cap <= 0:
                    continue
                year_chunk = neg_df[neg_df["year"] == year_int]
                if year_chunk.empty:
                    continue

                chunk_scores = rng.random(len(year_chunk))
                existing_df = neg_frames_by_year.get(year_int)
                existing_scores = neg_scores_by_year.get(year_int)

                if existing_df is None:
                    if len(year_chunk) <= cap:
                        neg_frames_by_year[year_int] = year_chunk.copy()
                        neg_scores_by_year[year_int] = chunk_scores
                    else:
                        keep_idx = np.argpartition(chunk_scores, cap - 1)[:cap]
                        neg_frames_by_year[year_int] = year_chunk.iloc[keep_idx].copy()
                        neg_scores_by_year[year_int] = chunk_scores[keep_idx]
                    continue

                combined_df = pd.concat([existing_df, year_chunk], axis=0, ignore_index=True)
                combined_scores = np.concatenate([existing_scores, chunk_scores], axis=0)
                keep_n = min(cap, len(combined_df))
                keep_idx = np.argpartition(combined_scores, keep_n - 1)[:keep_n]
                neg_frames_by_year[year_int] = combined_df.iloc[keep_idx].reset_index(drop=True)
                neg_scores_by_year[year_int] = combined_scores[keep_idx]

        del batch_table, target_np, year_np, wdpa_prev_np, valid_mask, base_mask
        del target_valid, years_valid, y_bin, filtered_table, filtered_df, pos_df, neg_df

    sampled_frames: List[pd.DataFrame] = []
    if pos_frames:
        sampled_frames.append(pd.concat(pos_frames, axis=0, ignore_index=True))
    for year in sorted(neg_frames_by_year):
        year_neg_df = neg_frames_by_year[year]
        sampled_frames.append(year_neg_df)
        year_stats[year]["neg_kept"] = int(len(year_neg_df))

    if not sampled_frames:
        raise ValueError("Sampling produced an empty tuning dataset.")

    sampled = pd.concat(sampled_frames, axis=0, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sampled = _downcast_numeric(sampled)
    pos = int((sampled[target_col] > 0).sum())
    neg = int((sampled[target_col] == 0).sum())

    del pos_frames, neg_frames_by_year, neg_scores_by_year, sampled_frames
    gc.collect()

    summary = DataSummary(
        n_rows_raw=n_rows_raw,
        n_rows_clean=int(n_rows_clean),
        n_rows_sampled=int(len(sampled)),
        n_features=int(len(feature_cols)),
        n_pos=pos,
        n_neg=neg,
    )
    return sampled, feature_cols, summary, year_stats
