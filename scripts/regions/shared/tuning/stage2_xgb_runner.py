"""Stage 2 XGBoost rank:ndcg tuning — orchestration + Optuna objective.

Issue BB: compare XGBoost rank:ndcg against LightGBM LambdaRank W9a.
XGBoost has no per-query row cap (unlike LightGBM's 9K limit), so no
ecoregion sub-group workaround is needed.

Entry point: run_stage2_xgb_tuning().
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from scripts.regions.shared.evaluation.stage2_metrics import ndcg_at_k_within_groups
from scripts.regions.shared.training.feature_guard import check_feature_denylist
from scripts.regions.shared.training.stage2_xgb_core import XGBNdcg1PctEarlyStopper
from scripts.regions.shared.training.stage2_lgbm_core import (
    STAGE2_EXCLUDE_COLS,
    TARGET_COL,
    _expansion_groups_from_batches,
    load_stage2_arrays,
    resolve_parquet_file,
)
from scripts.regions.shared.training.utils import compute_year_weights, get_repo_root
from scripts.regions.shared.tuning.cv import SplitConfig, build_splits


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _sort_fold(
    X: np.ndarray,
    y: np.ndarray,
    country_id: np.ndarray,
    years: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, group_sizes, years) sorted by (country_id, year) for one fold."""
    Xs = X[indices]
    ys = y[indices]
    cid = country_id[indices]
    yrs = years[indices]
    order = np.lexsort((yrs, cid))
    Xs, ys, cid, yrs = Xs[order], ys[order], cid[order], yrs[order]
    gdf = pd.DataFrame({"country_id": cid, "year": yrs})
    sizes = gdf.groupby(["country_id", "year"], sort=False).size().to_numpy(dtype=np.int32)
    return Xs, ys, sizes, yrs


_XGBNdcg1PctEarlyStopper = XGBNdcg1PctEarlyStopper  # alias for backward compat


def _optimize_xgb_optuna(
    X: np.ndarray,
    y: np.ndarray,
    country_id: np.ndarray,
    years: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    random_state: int,
    n_jobs: int,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Optuna HPO for XGBoost rank:ndcg."""
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(10, n_trials // 10))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {
            "objective": "rank:ndcg",
            "tree_method": "hist",
            "device": "cpu",
            "nthread": n_jobs,
            "seed": random_state,
            "verbosity": 0,
            "eta": trial.suggest_float("eta", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
            "lambdarank_pair_method": "topk",
            "lambdarank_num_pair_per_sample": trial.suggest_int(
                "lambdarank_num_pair_per_sample", 4, 16
            ),
        }
        n_estimators = trial.suggest_int("n_estimators", 200, 1500)

        fold_scores: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            X_tr, y_tr, g_tr, yrs_tr = _sort_fold(X, y, country_id, years, train_idx)
            X_va, y_va, g_va, _ = _sort_fold(X, y, country_id, years, val_idx)

            weights_tr = compute_year_weights(
                yrs_tr, min_year=int(yrs_tr.min()), max_year=int(yrs_tr.max())
            )
            # rank:ndcg requires one weight per query group, not per row.
            # All rows in a (country_id, year) group share the same year weight,
            # so take the first row index of each group.
            group_starts = np.concatenate([[0], np.cumsum(g_tr[:-1])])
            group_weights_tr = weights_tr[group_starts]

            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=group_weights_tr)
            dtrain.set_group(g_tr.tolist())

            dval = xgb.DMatrix(X_va, label=y_va)
            dval.set_group(g_va.tolist())

            early_stopper = _XGBNdcg1PctEarlyStopper(
                dval, y_va, g_va, patience=50, verbose=False
            )
            xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                callbacks=[early_stopper],
                verbose_eval=False,
            )

            fold_scores.append(early_stopper.best_score)
            trial.report(_mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return _mean(fold_scores)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best = study.best_trial
    best_params = dict(best.params)
    records = [
        {
            "best_trial_number": int(best.number),
            "n_trials_completed": int(len(study.trials)),
        }
    ]
    return best_params, float(best.value), records


def run_stage2_xgb_tuning(
    region: str,
    model_prefix: str,
    *,
    n_trials: int = 30,
    output_dir: Optional[Path] = None,
) -> Path:
    """Load data, run XGBoost rank:ndcg Optuna search, save best_params.json."""
    repo_root = get_repo_root()
    script_dir = repo_root / "scripts" / "regions" / region / "5_training"

    if env_out_dir := os.environ.get("STAGE2_OUTPUT_DIR"):
        out_dir = Path(env_out_dir)
    else:
        out_dir = output_dir or script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_parquet_file(region, "train.parquet")
    earlystop_path = resolve_parquet_file(region, "earlystop.parquet")

    schema = pq.ParquetFile(train_path).schema_arrow
    feature_cols = []
    for name in schema.names:
        if name in STAGE2_EXCLUDE_COLS:
            continue
        f = schema.field(name)
        if pa.types.is_integer(f.type) or pa.types.is_floating(f.type):
            feature_cols.append(name)
    check_feature_denylist(feature_cols, context=f"{region}/stage2/xgb_tuning")

    expansion_groups = _expansion_groups_from_batches(
        train_path, region, (2001, 2016)
    ) | _expansion_groups_from_batches(earlystop_path, region, (2001, 2016))

    _nr_env = os.environ.get("STAGE2_NEG_RATIO", "100")
    neg_ratio: Optional[int] = int(_nr_env) if int(_nr_env) > 0 else None

    arr_tr = load_stage2_arrays(
        train_path, region, feature_cols, expansion_groups, (2001, 2013),
        neg_ratio=neg_ratio,
    )
    arr_es = load_stage2_arrays(
        earlystop_path, region, feature_cols, expansion_groups, (2014, 2016),
        neg_ratio=neg_ratio,
    )

    X = np.vstack([arr_tr.X, arr_es.X])
    y = np.concatenate([arr_tr.y, arr_es.y])
    years = np.concatenate([arr_tr.years, arr_es.years])
    country_id = np.concatenate([arr_tr.country_ids, arr_es.country_ids])

    df_index = pd.DataFrame({"year": years, "country_id": country_id})
    cfg = SplitConfig(
        train_year_max=2014, val_year_min=2015, val_year_max=2017,
        strategy="rolling", rolling_folds=3,
    )
    folds, _ = build_splits(df_index, cfg)

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    n_trials = int(os.environ.get("STAGE2_TUNING_TRIALS", str(n_trials)))

    print(
        f"XGBoost stage2 tuning: region={region}, trials={n_trials},"
        f" neg_ratio={neg_ratio}, features={len(feature_cols)}"
    )

    best_params, best_score, records = _optimize_xgb_optuna(
        X, y, country_id, years, folds, n_trials, 42, n_jobs
    )

    artifact = {
        "best_params": best_params,
        "best_val_score": best_score,
        "metric": "ndcg_at_1pct_within_groups (xgboost rank:ndcg)",
        "fixed_params": {
            "objective": "rank:ndcg",
            "lambdarank_pair_method": "topk",
        },
        "metadata": {
            "region": region,
            "model_prefix": model_prefix,
            "variant": "xgboost",
            "target_col": TARGET_COL,
            "feature_count": len(feature_cols),
            "neg_ratio": neg_ratio,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "tuning_records": records,
    }

    out_path = out_dir / f"{model_prefix}_stage2_xgb_best_params.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Stage 2 XGB tuning complete. Best NDCG@1% = {best_score:.4f}")
    print(f"Saved: {out_path}")
    return out_path
