"""
Stage 2 LightGBM LambdaRank training core.

Conditional geographic selection within country-years that had PA expansion.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.regions.shared.country_raster import (
    country_ids_for_rows,
    load_country_raster,
)
from scripts.regions.shared.evaluation.stage2_metrics import compute_stage2_metrics
from scripts.regions.shared.training.feature_guard import check_feature_denylist
from scripts.regions.shared.training.utils import compute_year_weights, get_repo_root

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "200000"))
TARGET_COL = "transition_01"
GROUP_COLS = ("country_id", "year")
STAGE2_EXCLUDE_COLS = frozenset({
    "transition_01",
    "transition_01_win5",
    "WDPA_b1",
    "WDPA_prev",
    "WDPA",
    "x",
    "y",
    "row",
    "col",
    "year",
    "country_id",
    "country_iso3",
})

# LightGBM 4.6 hardcodes a 10 000-row per-query limit in C++.
# We split any group that exceeds this before building lgb.Dataset.
_LGB_MAX_QUERY = 9_000

FIXED_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "lambdarank",
    "metric": "ndcg",
    "verbose": -1,
    "lambdarank_truncation_level": 100,
    # Early stopping monitors ndcg@k@1% of the split sub-groups (_LGB_MAX_QUERY rows).
    # Default eval_at=[1,2,3,4,5] uses only the single top position — too noisy.
    "eval_at": [max(1, _LGB_MAX_QUERY // 100)],  # = [90] for _LGB_MAX_QUERY=9000
}

# Binary LightGBM (W8): no sub-window constraint, uses scale_pos_weight for imbalance.
# metric=average_precision (PR-AUC) is more informative than AUC under extreme imbalance.
BINARY_FIXED_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "average_precision",
    "verbose": -1,
}


def _split_large_groups(group_sizes: np.ndarray) -> np.ndarray:
    """Split any group larger than _LGB_MAX_QUERY into equal-ish sub-groups."""
    out: list[int] = []
    for s in group_sizes:
        s = int(s)
        if s <= _LGB_MAX_QUERY:
            out.append(s)
        else:
            n = (s + _LGB_MAX_QUERY - 1) // _LGB_MAX_QUERY
            base, rem = divmod(s, n)
            out.extend([base + 1] * rem + [base] * (n - rem))
    return np.array(out, dtype=np.int32)


@dataclass
class Stage2Config:
    region: str
    model_prefix: str  # model1, model2, model3
    train_years: Tuple[int, int] = (2001, 2013)
    earlystop_years: Tuple[int, int] = (2014, 2016)
    test_years: Tuple[int, int] = (2017, 2024)
    random_state: int = 42
    feature_subset: Optional[List[str]] = None  # e.g. ["dist_wdpa"] for naive baseline
    ablation_drop: Optional[str] = None  # drop feature group by name prefix
    variant: str = "full"  # "full" or "naive"


def resolve_parquet_file(region: str, filename: str) -> Path:
    repo_root = get_repo_root()
    scratch_root = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    candidates: list[Path] = []
    if scratch_root is not None:
        candidates.append(scratch_root / f"data/{region}/ml/main/{filename}")
        candidates.append(scratch_root / f"outputs/{region}/results/main/{filename}")
        candidates.append(scratch_root / f"outputs/{region}/results/{filename}")
    candidates.append(repo_root / f"outputs/{region}/results/main/{filename}")
    candidates.append(repo_root / f"outputs/{region}/results/{filename}")
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{filename} not found for region={region}")


def resolve_best_params_json(cfg: Stage2Config) -> Optional[Path]:
    script_dir = get_repo_root() / "scripts" / "regions" / cfg.region / "5_training"
    suffix = "_binary" if cfg.variant == "binary" else ""
    candidates = [
        script_dir / f"{cfg.model_prefix}_stage2_lgbm{suffix}_best_params.json",
        script_dir / "lgbm_best_params.json",
        script_dir / f"{cfg.model_prefix}_lgbm_best_params.json",
    ]
    scratch = os.environ.get("SCRATCH")
    if scratch:
        candidates.insert(0, Path(scratch) / f"scripts/regions/{cfg.region}/5_training/{cfg.model_prefix}_stage2_lgbm{suffix}_best_params.json")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _expansion_groups_from_batches(
    panel_path: Path,
    region: str,
    year_range: Optional[Tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return (country_id, year) keys with at least one positive transition."""
    schema_names = pq.ParquetFile(panel_path).schema_arrow.names
    has_country = "country_id" in schema_names
    raster = None if has_country else load_country_raster(region)

    cols = ["row", "col", "year", "transition_01"]
    if has_country:
        cols.append("country_id")

    pos_by_group: dict[tuple[int, int], int] = {}
    pf = pq.ParquetFile(panel_path)
    for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        if df.empty:
            continue
        if year_range is not None:
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        if df.empty:
            continue
        if not has_country:
            df["country_id"] = country_ids_for_rows(df, raster)
        grouped = df.groupby(["country_id", "year"], as_index=False)["transition_01"].sum()
        for _, row in grouped.iterrows():
            key = (int(row["country_id"]), int(row["year"]))
            if int(row["transition_01"]) > 0:
                pos_by_group[key] = 1
    return set(pos_by_group.keys())


def _compute_event_size_labels(
    rows: np.ndarray,
    cols: np.ndarray,
    pos_mask: np.ndarray,
) -> np.ndarray:
    """Graded relevance (1–4) for one group based on designation event cluster size.

    Uses 4-connected BFS over positive pixels. Negatives receive label 0.
    Relevance scale:
      1 = isolated pixel or event < 10 px
      2 = event 10–199 px
      3 = event 200–4 999 px
      4 = event ≥ 5 000 px

    Scientific rationale: large contiguous designations represent greater policy
    commitment and conservation value than isolated pixels, providing a stronger
    gradient signal for LambdaRank training.
    """
    n = len(rows)
    labels = np.zeros(n, dtype=np.int8)
    pos_idx = np.where(pos_mask)[0]
    if len(pos_idx) == 0:
        return labels

    pos_lookup: dict[tuple[int, int], int] = {
        (int(rows[i]), int(cols[i])): i for i in pos_idx
    }
    assigned = np.full(n, -1, dtype=np.int32)
    comp_members: list[list[int]] = []

    for start in pos_idx:
        if assigned[start] != -1:
            continue
        comp_id = len(comp_members)
        members: list[int] = []
        queue = [start]
        assigned[start] = comp_id
        head = 0
        while head < len(queue):
            i = queue[head]
            head += 1
            members.append(i)
            r, c = int(rows[i]), int(cols[i])
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                j = pos_lookup.get((r + dr, c + dc))
                if j is not None and assigned[j] == -1:
                    assigned[j] = comp_id
                    queue.append(j)
        comp_members.append(members)

    for members in comp_members:
        size = len(members)
        rel = 4 if size >= 5000 else 3 if size >= 200 else 2 if size >= 10 else 1
        for i in members:
            labels[i] = rel

    return labels


def _apply_graded_relevance(
    data: pd.DataFrame,
    group_sizes: np.ndarray,
    target_col: str,
) -> np.ndarray:
    """Compute graded relevance labels for the full sorted DataFrame."""
    y = np.zeros(len(data), dtype=np.int8)
    rows_arr = data["row"].to_numpy(dtype=np.int32)
    cols_arr = data["col"].to_numpy(dtype=np.int32)
    target_arr = data[target_col].to_numpy()
    offset = 0
    for size in group_sizes:
        end = offset + int(size)
        pos_mask = target_arr[offset:end] > 0
        if pos_mask.any():
            y[offset:end] = _compute_event_size_labels(
                rows_arr[offset:end], cols_arr[offset:end], pos_mask
            )
        offset = end
    return y


def _rank_normalize_within_groups(X: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    """Replace each feature with its within-group percentile rank (0–1), in-place.

    LambdaRank learns to rank within query groups. Absolute feature values carry
    between-country variation that is orthogonal to the within-country selection
    task. Normalizing to within-group percentile ranks gives the tree splits
    direct access to relative position information and improves cross-country
    generalization.

    Column-wise processing keeps peak memory to O(group_size) per column
    rather than O(group_size × n_features), making it safe for USA-scale groups
    (~12M rows per country-year).
    """
    offset = 0
    for size in group_sizes:
        end = offset + int(size)
        n = int(size)
        if n <= 1:
            X[offset:end] = 0.5
            offset = end
            continue
        denom = float(n - 1)
        arange = np.arange(n, dtype=np.float32)
        for col in range(X.shape[1]):
            order = np.argsort(X[offset:end, col])  # int64, ~8n bytes
            ranks = np.empty(n, dtype=np.float32)
            ranks[order] = arange
            X[offset:end, col] = ranks / denom
        offset = end
    return X


def _shuffle_within_groups(
    X: np.ndarray,
    y: np.ndarray,
    group_sizes: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Randomly shuffle rows within each group in-place (X column-wise, y directly).

    Must be called AFTER rank normalisation so that relative rank values are
    preserved, but BEFORE sub-window splitting in lgb.Dataset.  The result is that
    each consecutive 9K sub-window is a random sample of the full country-year
    rather than a spatially-contiguous patch.

    Why this matters: panels are spatially sorted (row, col order), so consecutive
    9K windows contain pixels from the same geographic neighbourhood.  LambdaRank
    gradient signals then compare only locally-similar pixels, teaching the model
    "is this pixel better than its 9K spatial neighbours?" — not the test task of
    "is this pixel better than ALL 500K pixels in the country-year?"  With random
    sub-windows, gradients compare pixels drawn from the full feature distribution,
    aligning training with the test evaluation scale.

    country_id and year are constant within every group, so they need not be shuffled.
    X is shuffled column-wise to keep peak memory O(n × 4B) per column rather than
    O(n × n_features × 4B), which is critical for large country-years (USA ~50M rows).
    """
    offset = 0
    for size in group_sizes:
        n = int(size)
        if n > 1:
            perm = rng.permutation(n)
            sl = slice(offset, offset + n)
            y[sl] = y[sl][perm]
            for col_idx in range(X.shape[1]):
                X[sl, col_idx] = X[sl, col_idx][perm]
        offset += n


def load_stage2_arrays(
    panel_path: Path,
    region: str,
    feature_cols: List[str],
    expansion_groups: set[tuple[int, int]],
    year_range: Optional[Tuple[int, int]],
    neg_ratio: Optional[int] = None,
    rng_seed: int = 42,
    normalize_within_groups: bool = True,
    use_graded_relevance: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, years, country_id, group_sizes for expansion country-years only.

    neg_ratio: if set, subsample negatives to neg_ratio * n_positives per group.
               Useful for tuning to avoid OOM on large regions.
    normalize_within_groups: replace each feature with its within-group percentile
               rank (0–1). Aligns feature representation with the within-group
               ranking task and removes between-country scale variation.
    use_graded_relevance: assign relevance 1–4 based on designation event cluster
               size instead of binary 0/1. Large contiguous events get higher
               relevance, providing richer gradient signal for LambdaRank.
    """
    schema_names = pq.ParquetFile(panel_path).schema_arrow.names
    has_country = "country_id" in schema_names
    raster = None if has_country else load_country_raster(region)

    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev", "row", "col"]
    if has_country:
        essential.append("country_id")
    essential = list(dict.fromkeys(essential))

    rng = np.random.default_rng(rng_seed) if neg_ratio is not None else None

    # Keep metadata (5 small columns) and feature arrays (float32) separate during
    # streaming to avoid the 2× peak from pd.concat on a DataFrame that includes all
    # 75 feature columns (~190 GB peak for SA at 203M rows with float32 features).
    # With separation: meta concat is ~4 GB; feature concat peaks at ~2×60 GB = 120 GB.
    meta_cols = ["country_id", "year", TARGET_COL]
    if use_graded_relevance:
        meta_cols += ["row", "col"]

    meta_frames: list[pd.DataFrame] = []
    feat_arrays: list[np.ndarray] = []

    pf = pq.ParquetFile(panel_path)
    for batch in pf.iter_batches(columns=essential, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        if df.empty:
            continue
        df = df[df[TARGET_COL].notna() & (df["WDPA_prev"] == 0)]
        if year_range is not None:
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        if df.empty:
            continue
        if not has_country:
            df["country_id"] = country_ids_for_rows(df, raster)
        keys = list(zip(df["country_id"].astype(int), df["year"].astype(int)))
        df = df[np.array([k in expansion_groups for k in keys], dtype=bool)]
        if df.empty:
            continue
        if neg_ratio is not None:
            pos_mask = (df[TARGET_COL] > 0).to_numpy()
            neg_idx = np.where(~pos_mask)[0]
            pos_idx = np.where(pos_mask)[0]
            n_keep = min(len(neg_idx), len(pos_idx) * neg_ratio)
            if n_keep < len(neg_idx):
                sampled_neg = rng.choice(neg_idx, size=n_keep, replace=False)
                keep = np.sort(np.concatenate([pos_idx, sampled_neg]))
                df = df.iloc[keep]
        meta_frames.append(df[meta_cols].copy())
        feat_arrays.append(df[feature_cols].to_numpy(dtype=np.float32))
        del df

    if not feat_arrays:
        raise ValueError(f"No Stage 2 samples after expansion filter: {panel_path}")

    # Metadata concat: cheap (~4 GB for SA's 203M rows × 5 cols × 4B).
    data = pd.concat(meta_frames, ignore_index=True)
    del meta_frames
    gc.collect()

    # Sort order for (country_id, year): lexsort uses last key as primary.
    order = np.lexsort((data["year"].to_numpy(), data["country_id"].to_numpy()))
    data = data.iloc[order].reset_index(drop=True)

    # Feature concat then sort: peak = 2 × n_rows × n_features × 4B (~120 GB for SA).
    X_unsorted = np.concatenate(feat_arrays, axis=0)
    del feat_arrays
    gc.collect()
    X = X_unsorted[order]
    del X_unsorted, order
    gc.collect()

    group_sizes = data.groupby(["country_id", "year"], sort=False).size().to_numpy(dtype=np.int32)

    if use_graded_relevance:
        y = _apply_graded_relevance(data, group_sizes, TARGET_COL)
    else:
        y = (data[TARGET_COL] > 0).astype(np.int8).to_numpy()

    years = data["year"].to_numpy(dtype=np.int32)
    country_ids = data["country_id"].to_numpy(dtype=np.int32)
    del data
    gc.collect()

    if normalize_within_groups:
        X = _rank_normalize_within_groups(X, group_sizes)

    # Shuffle rows within each group so that consecutive 9K sub-windows (created
    # by _split_large_groups in lgb.Dataset) are random samples of the full
    # country-year rather than spatially-contiguous patches.  Uses a fixed seed
    # offset from rng_seed so it is independent of neg_ratio sampling but still
    # fully reproducible.  See _shuffle_within_groups docstring for the rationale.
    _shuffle_within_groups(X, y, group_sizes, np.random.default_rng(rng_seed + 1337))

    return X, y, years, country_ids, group_sizes


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight = n_neg / n_pos from binary labels."""
    n_pos = int((y > 0).sum())
    n_neg = len(y) - n_pos
    return float(n_neg) / float(max(n_pos, 1))


def _prepare_lgb_params(best_params: Dict[str, Any], num_threads: int) -> Dict[str, Any]:
    params = {**FIXED_PARAMS, **best_params}
    for key in (
        "scale_pos_weight",
        "is_unbalance",
        "class_weight",
        "n_estimators",
        "num_boost_round",
        "n_jobs",
    ):
        params.pop(key, None)
    # Enforce fixed objective/metric — best_params must not override these
    params["objective"] = FIXED_PARAMS["objective"]
    params["metric"] = FIXED_PARAMS["metric"]
    params["random_state"] = params.get("random_state", 42)
    params["num_threads"] = num_threads
    if "lambdarank_truncation_level" not in params:
        params["lambdarank_truncation_level"] = FIXED_PARAMS["lambdarank_truncation_level"]
    # STAGE2_TRUNCATION_LEVEL env var overrides tuned value for post-tuning experiments.
    # Set to ~1500 to cover test k@1% (~2000 for SA/SEA full-data groups).
    override = os.environ.get("STAGE2_TRUNCATION_LEVEL")
    if override:
        params["lambdarank_truncation_level"] = int(override)
    return params


def train_lambdarank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    group_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    group_val: np.ndarray,
    params: Dict[str, Any],
    weights: Optional[np.ndarray],
    num_boost_round: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, group=_split_large_groups(group_train), weight=weights)
    val_set = lgb.Dataset(X_val, label=y_val, group=_split_large_groups(group_val), reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["earlystop"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )


def train_binary_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    weights: Optional[np.ndarray],
    num_boost_round: int,
) -> lgb.Booster:
    """Train binary LightGBM for W8 Stage 2 — no 9K sub-window constraint."""
    y_tr_bin = (y_train > 0).astype(np.int8)
    y_va_bin = (y_val > 0).astype(np.int8)
    train_set = lgb.Dataset(X_train, label=y_tr_bin, weight=weights)
    val_set = lgb.Dataset(X_val, label=y_va_bin, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["earlystop"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )


def run_stage2_training(cfg: Stage2Config, cv_mode: str = "fold3") -> None:
    """Train Stage 2 LambdaRank model and write metrics + scored test parquet."""
    repo_root = get_repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = resolve_parquet_file(cfg.region, "train.parquet")
    earlystop_path = resolve_parquet_file(cfg.region, "earlystop.parquet")
    test_path = resolve_parquet_file(cfg.region, "test.parquet")

    out_dir = repo_root / f"outputs/{cfg.region}/results/ml_models"
    model_dir = repo_root / f"data/{cfg.region}/ml/models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    schema = pq.ParquetFile(train_path).schema_arrow
    numeric_cols = [
        n
        for n, f in zip(schema.names, schema)
        if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
    ]
    feature_cols = [c for c in numeric_cols if c not in STAGE2_EXCLUDE_COLS]
    if cfg.feature_subset is not None:
        missing = [c for c in cfg.feature_subset if c not in feature_cols]
        if missing:
            raise ValueError(f"Feature subset not in panel: {missing}")
        feature_cols = list(cfg.feature_subset)

    if cfg.ablation_drop:
        from scripts.regions.shared.training.stage2_ablation_groups import ABLATION_GROUPS

        prefixes = ABLATION_GROUPS.get(cfg.ablation_drop, [cfg.ablation_drop])
        feature_cols = [
            c for c in feature_cols if not any(p in c for p in prefixes)
        ]

    check_feature_denylist(
        feature_cols,
        context=f"{cfg.region}/lgbm/stage2/{cfg.variant}",
    )

    params_path = resolve_best_params_json(cfg)
    if params_path:
        with open(params_path) as f:
            raw = json.load(f)
        best_params = raw.get("best_params", raw) if isinstance(raw, dict) else raw
    else:
        best_params = {
            "num_leaves": 127,
            "max_depth": -1,
            "learning_rate": 0.05,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
        print("  Using default Stage 2 hyperparameters (no stage2 best params JSON found)")

    num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    num_boost_round = int(best_params.get("n_estimators", best_params.get("num_boost_round", 2000)))

    # Negative-subsampling ratio: binary uses scale_pos_weight instead; LambdaRank must
    # match the tuning ratio (STAGE2_NEG_RATIO=100 in all tuning SLURMs) so hyperparameters
    # are valid for the class-balance regime the final model actually trains in.
    _nr_env = os.environ.get("STAGE2_NEG_RATIO", "100")
    neg_ratio: Optional[int] = (int(_nr_env) if int(_nr_env) > 0 else None) if cfg.variant != "binary" else None

    train_pool_years = (cfg.train_years[0], cfg.earlystop_years[1])
    print(f"Building expansion group index ({train_pool_years})...")
    t0 = time.time()
    expansion_groups = _expansion_groups_from_batches(
        train_path, cfg.region, train_pool_years
    ) | _expansion_groups_from_batches(earlystop_path, cfg.region, train_pool_years)
    print(f"  {len(expansion_groups):,} expansion country-years ({time.time()-t0:.1f}s)")

    print(f"Loading train pool (variant={cfg.variant}, neg_ratio={neg_ratio})...")
    X_tr, y_tr, years_tr, _, g_tr = load_stage2_arrays(
        train_path, cfg.region, feature_cols, expansion_groups, cfg.train_years,
        neg_ratio=neg_ratio,
    )
    X_es, y_es, _, _, g_es = load_stage2_arrays(
        earlystop_path, cfg.region, feature_cols, expansion_groups, cfg.earlystop_years,
        neg_ratio=neg_ratio,
    )

    year_weights = compute_year_weights(
        years_tr, min_year=cfg.train_years[0], max_year=cfg.train_years[1]
    )

    lgb_params: Dict[str, Any] = {}
    binary_params: Dict[str, Any] = {}

    if cfg.variant == "binary":
        # W8: binary LightGBM — no 9K sub-window, scale_pos_weight for imbalance.
        # Hyperparameters are stripped/rebuilt to avoid LambdaRank-only keys.
        scale_pos_weight = _compute_scale_pos_weight(y_tr)
        binary_params = {**BINARY_FIXED_PARAMS}
        for key in (
            "num_leaves", "max_depth", "learning_rate", "min_child_samples",
            "subsample", "colsample_bynode", "colsample_bytree",
            "reg_alpha", "reg_lambda", "min_split_gain", "path_smooth",
        ):
            if key in best_params:
                binary_params[key] = best_params[key]
        binary_params["scale_pos_weight"] = scale_pos_weight
        binary_params["random_state"] = best_params.get("random_state", 42)
        binary_params["num_threads"] = num_threads
        print(
            f"Fitting Binary LightGBM: train {len(y_tr):,} rows / earlystop {len(y_es):,} rows"
            f" | scale_pos_weight={scale_pos_weight:.1f}"
        )
        model = train_binary_lgbm(
            X_tr, y_tr, X_es, y_es, binary_params, year_weights, num_boost_round
        )
    else:
        lgb_params = _prepare_lgb_params(best_params, num_threads)
        print(f"Fitting LambdaRank: train {len(y_tr):,} rows / earlystop {len(y_es):,} rows...")
        model = train_lambdarank(
            X_tr, y_tr, g_tr, X_es, y_es, g_es, lgb_params, year_weights, num_boost_round
        )

    del X_tr, y_tr, g_tr, X_es, y_es, g_es
    gc.collect()

    tag = f"{cfg.model_prefix}_lgbm_stage2"
    if cfg.variant == "naive":
        tag += "_naive"
    elif cfg.variant == "binary":
        tag += "_binary"
    model_path = model_dir / f"{tag}_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "config": cfg}, f)
    print(f"Model saved: {model_path}")

    test_expansion = _expansion_groups_from_batches(test_path, cfg.region, cfg.test_years)
    print("Loading test set (no neg_ratio — full evaluation)...")
    X_te, y_te, years_te, country_ids_te, g_te = load_stage2_arrays(
        test_path, cfg.region, feature_cols, test_expansion, cfg.test_years
    )
    scores = model.predict(X_te, num_iteration=model.best_iteration or None).astype(np.float64)
    test_metrics = compute_stage2_metrics(y_te.astype(np.float64), scores, g_te)
    print(f"Test NDCG@1%: {test_metrics['ndcg_at_1pct']:.4f}")
    print(f"Test concordance: {test_metrics['concordance_index_within_groups']:.4f}")

    scored = pd.DataFrame({
        "y_true": y_te,
        "y_pred_score": scores,
        "year": years_te,
        "country_id": country_ids_te,
    })
    pq.write_table(
        pa.Table.from_pandas(scored, preserve_index=False),
        out_dir / f"{tag}_scored_{timestamp}.parquet",
    )

    saved_params = binary_params if cfg.variant == "binary" else lgb_params
    metrics = {
        "metadata": {
            "timestamp": timestamp,
            "model": "LightGBM_Binary" if cfg.variant == "binary" else "LightGBM_LambdaRank",
            "task": "stage2_geographic_selection",
            "target_column": TARGET_COL,
            "variant": cfg.variant,
            "region": cfg.region,
            "n_features": len(feature_cols),
            "features": feature_cols,
            "group_cols": list(GROUP_COLS),
            "neg_ratio": neg_ratio,
        },
        "temporal_split": {
            "train_years": list(cfg.train_years),
            "earlystop_years": list(cfg.earlystop_years),
            "test_years": list(cfg.test_years),
        },
        "test_performance": test_metrics,
        "model_parameters": saved_params,
        "n_expansion_groups_train_pool": len(expansion_groups),
    }
    metrics_path = out_dir / f"{tag}_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
