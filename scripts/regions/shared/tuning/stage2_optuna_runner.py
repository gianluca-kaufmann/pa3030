"""
Optuna hyperparameter optimization for Stage 2 LightGBM LambdaRank.

Year-based CV folds keep (country_id, year) groups intact within each fold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna

from scripts.regions.shared.evaluation.stage2_metrics import ndcg_at_k_within_groups
from scripts.regions.shared.training.stage2_lgbm_core import (
    BINARY_FIXED_PARAMS,
    _TrueNdcg1PctEarlyStop,
    _compute_scale_pos_weight,
    _split_large_groups,
)
from scripts.regions.shared.training.utils import compute_year_weights
from scripts.regions.shared.tuning.search_spaces import (
    get_lgbm_stage2_binary_optuna_bounds,
    get_lgbm_stage2_optuna_bounds,
)


def _mean(scores: List[float]) -> float:
    return float(sum(scores) / max(len(scores), 1))


def _sort_subset(
    X: np.ndarray,
    y: np.ndarray,
    country_id: np.ndarray,
    years: np.ndarray,
    indices: np.ndarray,
    eco_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract fold rows, sort by (country_id, year), optionally form eco sub-groups.

    Returns (Xs, ys, cy_group_sizes, train_group_sizes, sorted_years).

    cy_group_sizes: (country_id, year) groups — always used for NDCG evaluation.
    train_group_sizes: (country_id, year, eco_id) sub-groups when eco_ids is
        provided (W9a), otherwise equal to cy_group_sizes.
    sorted_years: year per row after sorting; needed for compute_year_weights.
    """
    import pandas as pd

    Xs = X[indices]
    ys = y[indices]
    cid = country_id[indices]
    yrs = years[indices]

    # Sort by (country_id, year) for cy groups and rank normalization alignment.
    cy_order = np.lexsort((yrs, cid))
    Xs = Xs[cy_order]
    ys = ys[cy_order]
    cid = cid[cy_order]
    yrs = yrs[cy_order]

    gdf = pd.DataFrame({"country_id": cid, "year": yrs})
    cy_group_sizes = gdf.groupby(["country_id", "year"], sort=False).size().to_numpy(dtype=np.int32)

    if eco_ids is not None:
        # W9a: sort within each cy group by eco_id to form eco sub-groups.
        ecos = eco_ids[indices][cy_order]
        eco_order = np.lexsort(
            (ecos.astype(np.int64), yrs.astype(np.int64), cid.astype(np.int64))
        )
        Xs = Xs[eco_order]
        ys = ys[eco_order]
        cid = cid[eco_order]
        yrs = yrs[eco_order]
        ecos = ecos[eco_order]
        # Eco group sizes from consecutive (cy, eco_id) runs.
        key = (
            cid.astype(np.int64) * 1_000_000_000
            + yrs.astype(np.int64) * 100_000
            + (ecos.astype(np.int64) + 32_768)
        )
        changes = np.concatenate([[True], key[1:] != key[:-1], [True]])
        train_group_sizes = np.diff(np.where(changes)[0]).astype(np.int32)
    else:
        train_group_sizes = cy_group_sizes

    return Xs, ys, cy_group_sizes, train_group_sizes, yrs


def optimize_lgbm_stage2_optuna(
    X: np.ndarray,
    y: np.ndarray,
    country_id: np.ndarray,
    years: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    mode: str,
    fixed_params: Dict[str, Any],
    n_trials: int,
    random_state: int,
    eco_ids: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Optuna HPO for Stage 2 LambdaRank.

    eco_ids: when provided (W9a), each CV fold's training lgb.Dataset uses
        (country_id, year, eco_id) sub-groups instead of full cy groups, resolving
        the 9K per-query hard cap.  The NDCG@1% callback always evaluates on full
        (country_id, year) groups for a fair comparison.
    """
    bounds = get_lgbm_stage2_optuna_bounds(mode)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(10, n_trials // 10))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    fold_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", bounds["num_leaves"][0], bounds["num_leaves"][1]),
            "max_depth": trial.suggest_categorical("max_depth", bounds["max_depth_choices"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", bounds["learning_rate"][0], bounds["learning_rate"][1], log=True
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", bounds["min_child_samples"][0], bounds["min_child_samples"][1]
            ),
            "subsample": trial.suggest_float("subsample", bounds["subsample"][0], bounds["subsample"][1]),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", bounds["colsample_bynode"][0], bounds["colsample_bynode"][1]
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", bounds["reg_alpha"][0], bounds["reg_alpha"][1], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", bounds["reg_lambda"][0], bounds["reg_lambda"][1], log=True),
            "min_split_gain": trial.suggest_float(
                "min_split_gain", bounds["min_split_gain"][0], bounds["min_split_gain"][1], log=True
            ),
            "path_smooth": trial.suggest_float(
                "path_smooth", bounds["path_smooth"][0], bounds["path_smooth"][1], log=True
            ),
            "lambdarank_truncation_level": trial.suggest_int(
                "lambdarank_truncation_level",
                bounds["lambdarank_truncation_level"][0],
                bounds["lambdarank_truncation_level"][1],
            ),
            "n_estimators": trial.suggest_int("n_estimators", bounds["n_estimators"][0], bounds["n_estimators"][1]),
        }
        train_params = {**fixed_params, **params}
        for key in ("scale_pos_weight", "is_unbalance", "n_estimators", "n_jobs"):
            train_params.pop(key, None)
        # Enforce fixed objective/metric — trial params must not override these
        train_params["objective"] = fixed_params.get("objective", "lambdarank")
        train_params["metric"] = fixed_params.get("metric", "ndcg")

        fold_scores: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            X_tr, y_tr, _, g_tr, yrs_tr = _sort_subset(X, y, country_id, years, train_idx, eco_ids)
            X_va, y_va, g_va, _, _      = _sort_subset(X, y, country_id, years, val_idx, eco_ids)
            weights_tr = compute_year_weights(
                yrs_tr, min_year=int(yrs_tr.min()), max_year=int(yrs_tr.max())
            )
            # g_tr = eco sub-groups for training (W9a), or cy groups if eco_ids=None.
            # g_va = cy groups for NDCG@1% callback evaluation (always cy).
            dtrain = lgb.Dataset(X_tr, label=y_tr, group=_split_large_groups(g_tr), weight=weights_tr)
            early_stop_cb = _TrueNdcg1PctEarlyStop(X_va, y_va, g_va, patience=50, verbose=False)
            booster = lgb.train(
                train_params,
                dtrain,
                num_boost_round=params["n_estimators"],
                callbacks=[early_stop_cb],
            )
            score = early_stop_cb.best_score
            fold_scores.append(score)
            trial.report(_mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return _mean(fold_scores)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best = study.best_trial
    best_params = dict(best.params)
    for key in ("objective", "metric"):
        best_params.pop(key, None)
    fold_records.append(
        {
            "best_trial_number": int(best.number),
            "n_trials_completed": int(len(study.trials)),
        }
    )
    return best_params, float(best.value), fold_records


def optimize_lgbm_stage2_binary_optuna(
    X: np.ndarray,
    y: np.ndarray,
    country_id: np.ndarray,
    years: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    mode: str,
    n_trials: int,
    random_state: int,
    true_scale_pos_weight: Optional[float] = None,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Optuna HPO for W8 binary LightGBM Stage 2.

    Trains with binary log-loss + scale_pos_weight.  When true_scale_pos_weight is
    provided (Issue O fix), it is used directly for all folds instead of computing
    SPW from the subsampled fold data (which reflects neg_ratio, not the full data
    class imbalance).  Evaluates on NDCG@1% within groups so the optimised metric
    matches the LambdaRank variant for direct comparison.
    Year weights are applied to training folds (matching final training behaviour).
    """
    bounds = get_lgbm_stage2_binary_optuna_bounds(mode)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(10, n_trials // 10))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    fold_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", bounds["num_leaves"][0], bounds["num_leaves"][1]),
            "max_depth": trial.suggest_categorical("max_depth", bounds["max_depth_choices"]),
            "learning_rate": trial.suggest_float("learning_rate", bounds["learning_rate"][0], bounds["learning_rate"][1], log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", bounds["min_child_samples"][0], bounds["min_child_samples"][1]),
            "subsample": trial.suggest_float("subsample", bounds["subsample"][0], bounds["subsample"][1]),
            "colsample_bynode": trial.suggest_float("colsample_bynode", bounds["colsample_bynode"][0], bounds["colsample_bynode"][1]),
            "reg_alpha": trial.suggest_float("reg_alpha", bounds["reg_alpha"][0], bounds["reg_alpha"][1], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", bounds["reg_lambda"][0], bounds["reg_lambda"][1], log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", bounds["min_split_gain"][0], bounds["min_split_gain"][1], log=True),
            "path_smooth": trial.suggest_float("path_smooth", bounds["path_smooth"][0], bounds["path_smooth"][1], log=True),
            "n_estimators": trial.suggest_int("n_estimators", bounds["n_estimators"][0], bounds["n_estimators"][1]),
        }
        train_params = {**BINARY_FIXED_PARAMS, **params}
        for key in ("n_estimators", "n_jobs"):
            train_params.pop(key, None)

        fold_scores: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            # Binary has no 9K sub-window constraint — eco_ids=None, use cy groups.
            X_tr, y_tr, g_tr, _, yrs_tr = _sort_subset(X, y, country_id, years, train_idx)
            X_va, y_va, g_va, _, _      = _sort_subset(X, y, country_id, years, val_idx)
            weights_tr = compute_year_weights(
                yrs_tr, min_year=int(yrs_tr.min()), max_year=int(yrs_tr.max())
            )
            spw = true_scale_pos_weight if true_scale_pos_weight is not None else _compute_scale_pos_weight(y_tr)
            fold_params = {**train_params, "scale_pos_weight": spw}
            y_tr_bin = (y_tr > 0).astype(np.int8)
            dtrain = lgb.Dataset(X_tr, label=y_tr_bin, weight=weights_tr)
            # y_va (graded 0–4) is passed to the callback so NDCG@1% is computed on
            # the same scale as the final test evaluation (not on binary 0/1).
            early_stop_cb = _TrueNdcg1PctEarlyStop(X_va, y_va, g_va, patience=50, verbose=False)
            booster = lgb.train(
                fold_params,
                dtrain,
                num_boost_round=params["n_estimators"],
                callbacks=[early_stop_cb],
            )
            score = early_stop_cb.best_score
            fold_scores.append(score)
            trial.report(_mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return _mean(fold_scores)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    best = study.best_trial
    best_params = dict(best.params)
    fold_records.append(
        {
            "best_trial_number": int(best.number),
            "n_trials_completed": int(len(study.trials)),
        }
    )
    return best_params, float(best.value), fold_records
