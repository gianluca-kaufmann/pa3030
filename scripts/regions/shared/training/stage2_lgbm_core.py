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

FIXED_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "lambdarank",
    "metric": "ndcg",
    "verbose": -1,
    "lambdarank_truncation_level": 5,
}


@dataclass
class Stage2Config:
    region: str
    model_prefix: str  # model1, model2, model3
    train_years: Tuple[int, int] = (2001, 2013)
    earlystop_years: Tuple[int, int] = (2014, 2016)
    test_years: Tuple[int, int] = (2017, 2019)
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
    candidates = [
        script_dir / f"{cfg.model_prefix}_stage2_lgbm_best_params.json",
        script_dir / "lgbm_best_params.json",
        script_dir / f"{cfg.model_prefix}_lgbm_best_params.json",
    ]
    scratch = os.environ.get("SCRATCH")
    if scratch:
        candidates.insert(0, Path(scratch) / f"scripts/regions/{cfg.region}/5_training/{cfg.model_prefix}_stage2_lgbm_best_params.json")
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


def load_stage2_arrays(
    panel_path: Path,
    region: str,
    feature_cols: List[str],
    expansion_groups: set[tuple[int, int]],
    year_range: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, years, country_id, group_sizes for expansion country-years only."""
    schema_names = pq.ParquetFile(panel_path).schema_arrow.names
    has_country = "country_id" in schema_names
    raster = None if has_country else load_country_raster(region)

    essential = feature_cols + [TARGET_COL, "year", "WDPA_prev", "row", "col"]
    if has_country:
        essential.append("country_id")
    essential = list(dict.fromkeys(essential))

    frames: list[pd.DataFrame] = []
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
        frames.append(df[["country_id", "year", TARGET_COL] + feature_cols])

    if not frames:
        raise ValueError(f"No Stage 2 samples after expansion filter: {panel_path}")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["country_id", "year"]).reset_index(drop=True)
    X = data[feature_cols].to_numpy(dtype=np.float32)
    y = (data[TARGET_COL] > 0).astype(np.int8).to_numpy()
    years = data["year"].to_numpy(dtype=np.int32)
    country_ids = data["country_id"].to_numpy(dtype=np.int32)
    group_sizes = data.groupby(["country_id", "year"], sort=False).size().to_numpy(dtype=np.int32)
    return X, y, years, country_ids, group_sizes


def _prepare_lgb_params(best_params: Dict[str, Any], num_threads: int) -> Dict[str, Any]:
    params = {**FIXED_PARAMS, **best_params}
    for key in (
        "scale_pos_weight",
        "is_unbalance",
        "class_weight",
        "n_estimators",
        "num_boost_round",
        "n_jobs",
        "objective",
        "metric",
    ):
        params.pop(key, None)
    params["random_state"] = params.get("random_state", 42)
    params["num_threads"] = num_threads
    if "lambdarank_truncation_level" not in params:
        params["lambdarank_truncation_level"] = 5
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
    train_set = lgb.Dataset(X_train, label=y_train, group=group_train, weight=weights)
    val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)
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
    lgb_params = _prepare_lgb_params(best_params, num_threads)
    num_boost_round = int(best_params.get("n_estimators", best_params.get("num_boost_round", 2000)))

    train_pool_years = (cfg.train_years[0], cfg.earlystop_years[1])
    print(f"Building expansion group index ({train_pool_years})...")
    t0 = time.time()
    expansion_groups = _expansion_groups_from_batches(
        train_path, cfg.region, train_pool_years
    ) | _expansion_groups_from_batches(earlystop_path, cfg.region, train_pool_years)
    print(f"  {len(expansion_groups):,} expansion country-years ({time.time()-t0:.1f}s)")

    print("Loading train pool...")
    X_tr, y_tr, _, _, g_tr = load_stage2_arrays(
        train_path, cfg.region, feature_cols, expansion_groups, cfg.train_years
    )
    X_es, y_es, _, _, g_es = load_stage2_arrays(
        earlystop_path, cfg.region, feature_cols, expansion_groups, cfg.earlystop_years
    )
    print(f"Fitting LambdaRank: train {len(y_tr):,} rows / earlystop {len(y_es):,} rows...")
    model = train_lambdarank(
        X_tr,
        y_tr,
        g_tr,
        X_es,
        y_es,
        g_es,
        lgb_params,
        compute_year_weights(
            np.full(len(y_tr), cfg.train_years[1], dtype=np.int32),
            min_year=2001,
            max_year=cfg.train_years[1],
        ),
        num_boost_round,
    )
    del X_tr, y_tr, g_tr, X_es, y_es, g_es
    gc.collect()

    tag = f"{cfg.model_prefix}_lgbm_stage2"
    if cfg.variant == "naive":
        tag += "_naive"
    model_path = model_dir / f"{tag}_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "config": cfg}, f)
    print(f"Model saved: {model_path}")

    test_expansion = _expansion_groups_from_batches(test_path, cfg.region, cfg.test_years)
    print("Loading test set...")
    X_te, y_te, years_te, _, g_te = load_stage2_arrays(
        test_path, cfg.region, feature_cols, test_expansion, cfg.test_years
    )
    scores = model.predict(X_te, num_iteration=model.best_iteration or None).astype(np.float64)
    test_metrics = compute_stage2_metrics(y_te.astype(np.float64), scores, g_te)
    print(f"Test NDCG@1%: {test_metrics['ndcg_at_1pct']:.4f}")
    print(f"Test concordance: {test_metrics['concordance_index_within_groups']:.4f}")

    scored = pd.DataFrame({"y_true": y_te, "y_pred_score": scores, "year": years_te})
    pq.write_table(
        pa.Table.from_pandas(scored, preserve_index=False),
        out_dir / f"{tag}_scored_{timestamp}.parquet",
    )

    metrics = {
        "metadata": {
            "timestamp": timestamp,
            "model": "LightGBM_LambdaRank",
            "task": "stage2_geographic_selection",
            "target_column": TARGET_COL,
            "variant": cfg.variant,
            "region": cfg.region,
            "n_features": len(feature_cols),
            "features": feature_cols,
            "group_cols": list(GROUP_COLS),
        },
        "temporal_split": {
            "train_years": list(cfg.train_years),
            "earlystop_years": list(cfg.earlystop_years),
            "test_years": list(cfg.test_years),
        },
        "test_performance": test_metrics,
        "model_parameters": lgb_params,
        "n_expansion_groups_train_pool": len(expansion_groups),
    }
    metrics_path = out_dir / f"{tag}_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
