"""
Optuna hyperparameter optimization for Stage 2 LightGBM LambdaRank.

Year-based CV folds keep (country_id, year) groups intact within each fold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import optuna

from scripts.regions.shared.evaluation.stage2_metrics import ndcg_at_k_within_groups
from scripts.regions.shared.tuning.search_spaces import get_lgbm_stage2_optuna_bounds


def _mean(scores: List[float]) -> float:
    return float(sum(scores) / max(len(scores), 1))


def _sort_subset(
    X: np.ndarray,
    y: np.ndarray,
    country_id: np.ndarray,
    years: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fold rows and sort by (country_id, year) for LambdaRank groups."""
    Xs = X[indices]
    ys = y[indices]
    cid = country_id[indices]
    yrs = years[indices]
    order = np.lexsort((yrs, cid))
    Xs = Xs[order]
    ys = ys[order]
    cid = cid[order]
    yrs = yrs[order]
    import pandas as pd

    gdf = pd.DataFrame({"country_id": cid, "year": yrs})
    group_sizes = gdf.groupby(["country_id", "year"], sort=False).size().to_numpy(dtype=np.int32)
    return Xs, ys, group_sizes


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
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
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
            X_tr, y_tr, g_tr = _sort_subset(X, y, country_id, years, train_idx)
            X_va, y_va, g_va = _sort_subset(X, y, country_id, years, val_idx)
            dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
            dval = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain)
            booster = lgb.train(
                train_params,
                dtrain,
                num_boost_round=params["n_estimators"],
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            pred = booster.predict(X_va, num_iteration=booster.best_iteration or None)
            score = ndcg_at_k_within_groups(y_va.astype(np.float64), pred, g_va, 1.0)
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
