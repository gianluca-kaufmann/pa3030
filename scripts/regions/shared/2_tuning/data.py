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
from typing import Any, Dict, List, Tuple

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


def _allocate_with_largest_remainder(total: int, weights: List[float], caps: List[int]) -> List[int]:
    if total <= 0 or not weights or not caps:
        return [0 for _ in caps]
    raw = [float(max(w, 0.0)) for w in weights]
    denom = sum(raw)
    if denom <= 0:
        raw = [1.0 for _ in caps]
        denom = float(len(raw))
    quotas = [total * (w / denom) for w in raw]
    alloc = [min(int(np.floor(q)), cap) for q, cap in zip(quotas, caps)]
    remainder = total - sum(alloc)
    if remainder <= 0:
        return alloc
    frac_order = sorted(
        range(len(caps)),
        key=lambda i: (quotas[i] - np.floor(quotas[i]), caps[i] - alloc[i]),
        reverse=True,
    )
    for idx in frac_order:
        if remainder <= 0:
            break
        room = caps[idx] - alloc[idx]
        if room <= 0:
            continue
        take = min(room, remainder)
        alloc[idx] += take
        remainder -= take
    return alloc


def _apply_total_row_budget(
    sampled_df: pd.DataFrame,
    target_col: str,
    year_col: str,
    total_row_budget: int | None,
    random_state: int,
    min_pos_per_year: int,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]], Dict[str, Any]]:
    if total_row_budget is None or total_row_budget <= 0:
        return sampled_df, {}, {"total_row_budget": None, "budget_applied": False}
    if len(sampled_df) <= total_row_budget:
        return sampled_df, {}, {"total_row_budget": int(total_row_budget), "budget_applied": False}

    years = sorted(int(y) for y in sampled_df[year_col].unique())
    sampled_df = sampled_df.copy()
    sampled_df["_y_bin"] = (sampled_df[target_col] > 0).astype(np.int8)

    strata: List[Tuple[int, int]] = []
    counts: Dict[Tuple[int, int], int] = {}
    for year in years:
        for ybin in (1, 0):
            cnt = int(((sampled_df[year_col] == year) & (sampled_df["_y_bin"] == ybin)).sum())
            if cnt > 0:
                key = (year, ybin)
                strata.append(key)
                counts[key] = cnt

    pos_years = [year for year in years if counts.get((year, 1), 0) > 0]
    mandatory_pos: Dict[int, int] = {year: min(counts[(year, 1)], int(min_pos_per_year)) for year in pos_years}
    mandatory_total = int(sum(mandatory_pos.values()))
    warning = None

    if mandatory_total > total_row_budget:
        fallback = {year: 1 for year in pos_years}
        if sum(fallback.values()) > total_row_budget:
            # Budget too small to preserve positives in every positive year.
            sorted_years = sorted(pos_years, key=lambda y: counts[(y, 1)], reverse=True)
            fallback = {year: 0 for year in pos_years}
            for year in sorted_years[: int(total_row_budget)]:
                fallback[year] = 1
            warning = (
                "Total-row budget is smaller than number of positive years; "
                "could not preserve positives for all years."
            )
        else:
            warning = (
                "Total-row budget constrained mandatory per-year positives; "
                "reduced to one positive per positive year."
            )
        mandatory_pos = fallback
        mandatory_total = int(sum(mandatory_pos.values()))

    remaining_budget = int(total_row_budget) - mandatory_total
    strata_caps: List[int] = []
    strata_weights: List[float] = []
    for key in strata:
        year, ybin = key
        if ybin == 1:
            cap = counts[key] - mandatory_pos.get(year, 0)
        else:
            cap = counts[key]
        cap = max(int(cap), 0)
        strata_caps.append(cap)
        strata_weights.append(float(counts[key]))

    alloc_extra = _allocate_with_largest_remainder(
        total=max(remaining_budget, 0),
        weights=strata_weights,
        caps=strata_caps,
    )

    keep_by_stratum: Dict[Tuple[int, int], int] = {}
    for key, extra in zip(strata, alloc_extra):
        year, ybin = key
        base = mandatory_pos.get(year, 0) if ybin == 1 else 0
        keep_by_stratum[key] = int(base + extra)

    # Deterministic sampling inside each (year, class) stratum.
    sampled_out: List[pd.DataFrame] = []
    rng = np.random.default_rng(random_state)
    for key in strata:
        year, ybin = key
        keep_n = int(keep_by_stratum.get(key, 0))
        if keep_n <= 0:
            continue
        stratum_df = sampled_df[(sampled_df[year_col] == year) & (sampled_df["_y_bin"] == ybin)]
        if len(stratum_df) <= keep_n:
            sampled_out.append(stratum_df)
            continue
        chosen_idx = rng.choice(len(stratum_df), size=keep_n, replace=False)
        sampled_out.append(stratum_df.iloc[chosen_idx])

    if not sampled_out:
        raise ValueError("Total-row budgeting produced an empty dataset.")

    out_df = pd.concat(sampled_out, axis=0, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    out_df = out_df.drop(columns=["_y_bin"])

    year_post_stats: Dict[int, Dict[str, int]] = {}
    for year in years:
        year_df = out_df[out_df[year_col] == year]
        year_post_stats[year] = {
            "pos_kept_after_total_budget": int((year_df[target_col] > 0).sum()),
            "neg_kept_after_total_budget": int((year_df[target_col] == 0).sum()),
        }

    diagnostics: Dict[str, Any] = {
        "total_row_budget": int(total_row_budget),
        "budget_applied": True,
        "mandatory_pos_per_year_requested": int(min_pos_per_year),
        "mandatory_pos_total_applied": int(mandatory_total),
    }
    if warning:
        diagnostics["warning"] = warning
        print(f"[WARN] {warning}")
    return out_df, year_post_stats, diagnostics


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
    if year_col not in df.columns:
        raise ValueError(f"Year column '{year_col}' not found in dataframe")

    clean = df.dropna(subset=[target_col]).copy()
    clean = clean[clean[year_col] >= min_year].copy()
    if "WDPA_prev" in clean.columns:
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
    total_row_budget: int | None = None,
    min_pos_per_year_after_budget: int = 25,
    max_pos_per_year: int = 0,
) -> Tuple[pd.DataFrame, List[str], DataSummary, Dict[int, Dict[str, int]], Dict[str, Any], Dict[str, Any]]:
    parquet_file = pq.ParquetFile(train_path)
    schema = parquet_file.schema_arrow

    numeric_cols = [
        name
        for name, field in zip(schema.names, schema)
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
    ]
    feature_cols = [col for col in numeric_cols if col.lower() not in {c.lower() for c in exclude_cols}]
    has_wdpa_prev = "WDPA_prev" in schema.names
    keep_cols = list(dict.fromkeys(feature_cols + [target_col, "year"] + (["WDPA_prev"] if has_wdpa_prev else [])))

    n_rows_raw = int(parquet_file.metadata.num_rows) if parquet_file.metadata is not None else 0

    # Pass 1: gather year-level class counts after risk-set and min-year filtering.
    year_pos_counts: Dict[int, int] = {}
    year_neg_counts: Dict[int, int] = {}
    n_rows_clean = 0
    wdpa_diag: Dict[str, Any] = {"present": bool(has_wdpa_prev)}
    wdpa_unique: set[int] = set()
    wdpa_nonzero_rows = 0
    wdpa_rows_seen = 0
    pass1_cols = [target_col, "year"] + (["WDPA_prev"] if has_wdpa_prev else [])
    for batch in parquet_file.iter_batches(batch_size=200_000, columns=pass1_cols, use_threads=True):
        batch_table = batch
        target_np = batch_table[target_col].to_numpy(zero_copy_only=False)
        year_np = batch_table["year"].to_numpy(zero_copy_only=False)

        valid_mask = ~batch_table[target_col].is_null().to_numpy(zero_copy_only=False)
        if has_wdpa_prev:
            wdpa_prev_np = batch_table["WDPA_prev"].to_numpy(zero_copy_only=False)
            wdpa_valid = wdpa_prev_np[valid_mask]
            wdpa_rows_seen += int(len(wdpa_valid))
            if len(wdpa_valid) > 0:
                wdpa_non_null = wdpa_valid[pd.notnull(wdpa_valid)]
                if len(wdpa_non_null) > 0:
                    wdpa_unique.update(int(v) for v in np.unique(wdpa_non_null))
                wdpa_nonzero_rows += int((wdpa_valid != 0).sum())
            base_mask = valid_mask & (year_np >= min_year) & (wdpa_prev_np == 0)
        else:
            base_mask = valid_mask & (year_np >= min_year)
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

        if has_wdpa_prev:
            del wdpa_prev_np, wdpa_valid
        del batch_table, target_np, year_np, valid_mask, base_mask, y_valid, years_valid

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
        pos_kept_planned = min(pos_count, int(max_pos_per_year)) if max_pos_per_year > 0 else pos_count
        year_stats[year] = {
            "pos_original": pos_count,
            "pos_kept": pos_kept_planned,
            "neg_kept": 0,
            "neg_original": neg_original,
            "neg_cap": int(cap),
            "pos_cap": int(max_pos_per_year) if max_pos_per_year > 0 else None,
        }

    # Pass 2: stream rows, reservoir-sample positives (if capped) and negatives per year.
    parquet_file = pq.ParquetFile(train_path)
    rng = np.random.default_rng(random_state)
    # pos_frames: used when max_pos_per_year == 0 (keep all positives)
    pos_frames: List[pd.DataFrame] = []
    # pos_frames_by_year / pos_scores_by_year: reservoir when max_pos_per_year > 0
    pos_frames_by_year: Dict[int, pd.DataFrame] = {}
    pos_scores_by_year: Dict[int, np.ndarray] = {}
    neg_frames_by_year: Dict[int, pd.DataFrame] = {}
    neg_scores_by_year: Dict[int, np.ndarray] = {}

    for batch in parquet_file.iter_batches(batch_size=200_000, columns=keep_cols, use_threads=True):
        batch_table = batch
        target_np = batch_table[target_col].to_numpy(zero_copy_only=False)
        year_np = batch_table["year"].to_numpy(zero_copy_only=False)

        valid_mask = ~batch_table[target_col].is_null().to_numpy(zero_copy_only=False)
        if has_wdpa_prev:
            wdpa_prev_np = batch_table["WDPA_prev"].to_numpy(zero_copy_only=False)
            base_mask = valid_mask & (year_np >= min_year) & (wdpa_prev_np == 0)
        else:
            base_mask = valid_mask & (year_np >= min_year)
        if not base_mask.any():
            continue

        target_valid = target_np[base_mask]
        y_bin = (target_valid > 0).astype(np.int8)

        filtered_table = batch_table.filter(pa.array(base_mask))
        filtered_df = filtered_table.to_pandas()
        filtered_df[target_col] = y_bin

        pos_df = filtered_df[y_bin > 0]
        if not pos_df.empty:
            if max_pos_per_year <= 0:
                # No cap: keep every positive (original behaviour).
                pos_frames.append(pos_df)
            else:
                # Reservoir-sample positives per year, identical to negative logic.
                for year in np.unique(pos_df["year"].to_numpy()):
                    year_int = int(year)
                    pos_cap = int(max_pos_per_year)
                    year_chunk = pos_df[pos_df["year"] == year_int]
                    if year_chunk.empty:
                        continue
                    chunk_scores = rng.random(len(year_chunk))
                    existing_df = pos_frames_by_year.get(year_int)
                    existing_scores = pos_scores_by_year.get(year_int)
                    if existing_df is None:
                        if len(year_chunk) <= pos_cap:
                            pos_frames_by_year[year_int] = year_chunk.copy()
                            pos_scores_by_year[year_int] = chunk_scores
                        else:
                            keep_idx = np.argpartition(chunk_scores, pos_cap - 1)[:pos_cap]
                            pos_frames_by_year[year_int] = year_chunk.iloc[keep_idx].copy()
                            pos_scores_by_year[year_int] = chunk_scores[keep_idx]
                        continue
                    combined_df = pd.concat([existing_df, year_chunk], axis=0, ignore_index=True)
                    combined_scores = np.concatenate([existing_scores, chunk_scores], axis=0)
                    keep_n = min(pos_cap, len(combined_df))
                    keep_idx = np.argpartition(combined_scores, keep_n - 1)[:keep_n]
                    pos_frames_by_year[year_int] = combined_df.iloc[keep_idx].reset_index(drop=True)
                    pos_scores_by_year[year_int] = combined_scores[keep_idx]

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

        if has_wdpa_prev:
            del wdpa_prev_np
        del batch_table, target_np, year_np, valid_mask, base_mask
        del target_valid, y_bin, filtered_table, filtered_df, pos_df, neg_df

    sampled_frames: List[pd.DataFrame] = []
    if pos_frames:
        # max_pos_per_year == 0: collected as flat list of batch frames
        sampled_frames.append(pd.concat(pos_frames, axis=0, ignore_index=True))
    elif pos_frames_by_year:
        # max_pos_per_year > 0: collected per year with reservoir sampling
        pos_all = pd.concat(list(pos_frames_by_year.values()), axis=0, ignore_index=True)
        sampled_frames.append(pos_all)
        for year, yr_pos_df in pos_frames_by_year.items():
            if year in year_stats:
                year_stats[year]["pos_kept"] = int(len(yr_pos_df))
    for year in sorted(neg_frames_by_year):
        year_neg_df = neg_frames_by_year[year]
        sampled_frames.append(year_neg_df)
        year_stats[year]["neg_kept"] = int(len(year_neg_df))

    if not sampled_frames:
        raise ValueError("Sampling produced an empty tuning dataset.")

    sampled = pd.concat(sampled_frames, axis=0, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sampled, post_budget_stats, budget_diag = _apply_total_row_budget(
        sampled_df=sampled,
        target_col=target_col,
        year_col="year",
        total_row_budget=total_row_budget,
        random_state=random_state,
        min_pos_per_year=min_pos_per_year_after_budget,
    )
    for year, post_stats in post_budget_stats.items():
        if year not in year_stats:
            year_stats[year] = {}
        year_stats[year].update(post_stats)
        if "pos_kept_after_total_budget" in post_stats:
            year_stats[year]["pos_kept"] = int(post_stats["pos_kept_after_total_budget"])
        if "neg_kept_after_total_budget" in post_stats:
            year_stats[year]["neg_kept"] = int(post_stats["neg_kept_after_total_budget"])

    sampled = _downcast_numeric(sampled)
    pos = int((sampled[target_col] > 0).sum())
    neg = int((sampled[target_col] == 0).sum())

    del pos_frames, pos_frames_by_year, pos_scores_by_year, neg_frames_by_year, neg_scores_by_year, sampled_frames
    gc.collect()

    summary = DataSummary(
        n_rows_raw=n_rows_raw,
        n_rows_clean=int(n_rows_clean),
        n_rows_sampled=int(len(sampled)),
        n_features=int(len(feature_cols)),
        n_pos=pos,
        n_neg=neg,
    )
    if has_wdpa_prev:
        wdpa_diag.update(
            {
                "unique_values": sorted(int(v) for v in wdpa_unique),
                "n_rows_checked": int(wdpa_rows_seen),
                "n_rows_nonzero": int(wdpa_nonzero_rows),
            }
        )
        if wdpa_nonzero_rows > 0:
            print(
                "[WARN] WDPA_prev includes non-zero values; risk-set guardrail filter "
                "retained only WDPA_prev == 0 rows."
            )
    else:
        wdpa_diag.update(
            {
                "warning": "WDPA_prev missing; assumed risk-set-only input and skipped WDPA guardrail filter.",
            }
        )
        print("[WARN] WDPA_prev missing; assuming risk-set-only input.")

    spw_actual = float(neg / max(pos, 1))
    pos_cap_msg = f"pos_cap/yr={max_pos_per_year:,}" if max_pos_per_year > 0 else "pos_cap=none"
    print(
        f"[Sampling] Final sampled rows={len(sampled):,} (pos={pos:,}, neg={neg:,}, "
        f"SPW={spw_actual:.1f}) | {pos_cap_msg} | total_budget={budget_diag.get('total_row_budget')}"
    )
    if max_pos_per_year == 0 and pos > neg:
        print(
            f"[WARN] Majority-positive dataset: pos={pos:,} > neg={neg:,} ({100*pos/(pos+neg):.1f}% positive). "
            "Set MAX_POS_PER_YEAR_PAPER (e.g. 5000 for SA) to restore rare-event regime."
        )
    return sampled, feature_cols, summary, year_stats, wdpa_diag, budget_diag
