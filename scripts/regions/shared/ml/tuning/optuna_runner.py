from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from .search_spaces import get_lgbm_optuna_bounds, get_rf_optuna_bounds


def _mean(scores: List[float]) -> float:
    return float(sum(scores) / max(len(scores), 1))


def optimize_lgbm_optuna(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    mode: str,
    fixed_params: Dict[str, Any],
    auto_scale_pos_weight: float,
    n_trials: int,
    random_state: int,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    bounds = get_lgbm_optuna_bounds(mode, auto_scale_pos_weight=auto_scale_pos_weight)

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
            "colsample_bytree": trial.suggest_float("colsample_bytree", bounds["colsample_bytree"][0], bounds["colsample_bytree"][1]),
            "reg_alpha": trial.suggest_float("reg_alpha", bounds["reg_alpha"][0], bounds["reg_alpha"][1], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", bounds["reg_lambda"][0], bounds["reg_lambda"][1], log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", bounds["scale_pos_weight"][0], bounds["scale_pos_weight"][1], log=True),
            "n_estimators": trial.suggest_int("n_estimators", bounds["n_estimators"][0], bounds["n_estimators"][1]),
            "objective": "binary",
            "metric": "average_precision",
        }
        fold_scores: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            clf = LGBMClassifier(**fixed_params, **params)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="average_precision",
                callbacks=[],
            )
            pred = clf.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, pred)
            fold_scores.append(float(score))
            trial.report(_mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return _mean(fold_scores)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    best_params = dict(best.params)
    best_score = float(best.value)
    for k in ("objective", "metric"):
        best_params.pop(k, None)

    fold_records.append(
        {
            "best_trial_number": int(best.number),
            "n_trials_completed": int(len(study.trials)),
            "n_trials_pruned": int(sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)),
        }
    )
    return best_params, best_score, fold_records


def optimize_rf_optuna(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    mode: str,
    fixed_params: Dict[str, Any],
    n_trials: int,
    random_state: int,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    bounds = get_rf_optuna_bounds(mode)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(10, n_trials // 10))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    fold_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", bounds["n_estimators"][0], bounds["n_estimators"][1]),
            "max_depth": trial.suggest_categorical("max_depth", bounds["max_depth_choices"]),
            "max_features": trial.suggest_categorical("max_features", bounds["max_features_choices"]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", bounds["min_samples_leaf"][0], bounds["min_samples_leaf"][1]),
            "min_samples_split": trial.suggest_int("min_samples_split", bounds["min_samples_split"][0], bounds["min_samples_split"][1]),
            "bootstrap": trial.suggest_categorical("bootstrap", bounds["bootstrap_choices"]),
        }
        fold_scores: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            clf = RandomForestClassifier(**fixed_params, **params)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            clf.fit(X_tr, y_tr)
            pred = clf.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, pred)
            fold_scores.append(float(score))
            trial.report(_mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return _mean(fold_scores)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    best_params = dict(best.params)
    best_score = float(best.value)
    fold_records.append(
        {
            "best_trial_number": int(best.number),
            "n_trials_completed": int(len(study.trials)),
            "n_trials_pruned": int(sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)),
        }
    )
    return best_params, best_score, fold_records
