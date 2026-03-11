"""
Main orchestration pipeline for regional hyperparameter tuning.

- Coordinates dataset preparation, temporal CV, optimizer selection, and
  artifact writing in one reproducible flow shared across regions/models.

Input:
- Region/model identifiers, script/output paths, and environment-driven tuning
  configuration (mode, CV, sampling, optimizer, seeds).

Output:
- Best-parameter artifact payloads (canonical + timestamped JSON) and console
  logs summarizing data size, settings, and best validation score.
"""

from __future__ import annotations

import gc
import importlib
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
try:
    import wandb
except ImportError:
    wandb = None

try:
    from optuna.integration.wandb import WeightsAndBiasesCallback
except ImportError:
    WeightsAndBiasesCallback = None

from .artifacts import build_metadata, save_tuning_artifact
from .cv import SplitConfig, build_splits
from .data import DataSummary, prepare_tuning_dataset, resolve_train_parquet
from .search_spaces import (
    compute_auto_scale_pos_weight,
    get_lgbm_fixed_params,
    get_lgbm_randomized_space,
    get_rf_fixed_params,
    get_rf_randomized_space,
)


TARGET_COL = "transition_01_win5"
YEAR_COL = "year"
PR_AUC_SCORER = make_scorer(average_precision_score, response_method="predict_proba")


@dataclass(frozen=True)
class RuntimeConfig:
    region: str
    model: str
    script_dir: Path
    output_dir: Path
    repo_root: Path
    random_state: int
    tuning_mode: str
    optimizer: str
    cv_strategy: str
    train_year_max: int
    val_year_min: int
    val_year_max: int
    rolling_folds: int
    tuning_split_version: str
    min_year: int
    max_neg_per_year_fast: int
    max_neg_per_year_paper: int
    max_pos_per_year_fast: int
    max_pos_per_year_paper: int
    adaptive_cap_enabled: bool
    adaptive_ratio_target: float
    adaptive_min_cap: int
    adaptive_max_cap: int
    n_iter_lgbm_fast: int
    n_iter_lgbm_paper: int
    n_iter_rf_fast: int
    n_iter_rf_paper: int
    total_row_budget_lgbm_fast: int
    total_row_budget_lgbm_paper: int
    total_row_budget_rf_fast: int
    total_row_budget_rf_paper: int
    min_pos_per_year_after_budget: int
    rf_add_missing_indicators: bool
    n_jobs: int


def _get_n_jobs() -> int:
    try:
        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        return int(cpus) if cpus else -1
    except Exception:
        return -1


def _report_memory_usage(label: str) -> None:
    try:
        import psutil

        rss_gb = psutil.Process().memory_info().rss / (1024**3)
        print(f"[Memory] {label}: RSS {rss_gb:.2f} GB")
    except Exception:
        pass


def _resolve_optimizer(requested_optimizer: str) -> tuple[str, str | None]:
    """
    Resolve optimizer choice with dependency-aware fallback.

    If Optuna is requested but unavailable, automatically downgrade to
    randomized search and return a warning message for logs/artifacts.
    """
    if requested_optimizer != "optuna":
        return requested_optimizer, None
    try:
        importlib.import_module("optuna")
        return "optuna", None
    except ModuleNotFoundError as exc:
        if exc.name != "optuna":
            raise
        warning = (
            "Optuna requested but package 'optuna' is not installed. "
            "Falling back to randomized search. "
            "Install with `pip install optuna` or use TUNING_OPTIMIZER=randomized."
        )
        return "randomized", warning


def _init_wandb_tuning(cfg: RuntimeConfig, optimizer_used: str) -> Any | None:
    if wandb is None:
        print("W&B not available (module not installed)")
        return None
    try:
        os.environ.setdefault("WANDB_CONSOLE", "wrap")
        run_name = f"{cfg.region}_{cfg.model}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_family = "model1" if cfg.region == "south_america" else "model2" if cfg.region == "usa" else cfg.region
        run = wandb.init(
            project=f"ml-tuning-{model_family}",
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_name,
            config={
                "region": cfg.region,
                "model": cfg.model,
                "task": "transition_01_win5_hyperparameter_tuning",
                "target_column": TARGET_COL,
                "random_state": cfg.random_state,
                "tuning_mode": cfg.tuning_mode,
                "optimizer_requested": cfg.optimizer,
                "optimizer_used": optimizer_used,
                "cv_strategy": cfg.cv_strategy,
                "rolling_folds": cfg.rolling_folds,
                "split_version": cfg.tuning_split_version,
                "train_year_max": cfg.train_year_max,
                "val_year_min": cfg.val_year_min,
                "val_year_max": cfg.val_year_max,
                "min_year": cfg.min_year,
                "n_jobs": cfg.n_jobs,
            },
        )
        print("W&B connected")
        return run
    except Exception as err:
        print(f"W&B failed: {err}")
        return None


def _build_optuna_wandb_callbacks(wandb_run: Any | None) -> List[Any]:
    """
    Build Optuna callbacks for per-trial W&B logging when a run is active.
    Logs each trial's hyperparameters and val_pr_auc to the existing run.
    """
    if wandb_run is None or WeightsAndBiasesCallback is None:
        return []
    try:
        wandb_kwargs = {"id": wandb.run.id, "resume": "allow"}
        wb_callback = WeightsAndBiasesCallback(
            metric_name="val_pr_auc",
            wandb_kwargs=wandb_kwargs,
        )
        return [wb_callback]
    except Exception as err:
        print(f"W&B Optuna callback skipped: {err}")
        return []


def get_repo_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    current = Path(__file__).resolve()
    for _ in range(12):
        current = current.parent
        if (current / "README.md").exists():
            return current
    raise RuntimeError("Repository root not found. Set PROJECT_ROOT or REPO_ROOT.")


def load_runtime_config(region: str, model: str, script_dir: Path, output_dir: Path) -> RuntimeConfig:
    tuning_mode = os.environ.get("TUNING_MODE", "paper").strip().lower()
    if tuning_mode not in {"fast", "paper"}:
        raise ValueError(f"Invalid TUNING_MODE='{tuning_mode}'. Expected fast or paper.")

    optimizer_default = "randomized" if tuning_mode == "fast" else "optuna"
    cv_default = "holdout" if tuning_mode == "fast" else "rolling"
    optimizer = os.environ.get("TUNING_OPTIMIZER", optimizer_default).strip().lower()
    cv_strategy = os.environ.get("CV_STRATEGY", cv_default).strip().lower()
    if optimizer not in {"randomized", "optuna"}:
        raise ValueError(f"Invalid TUNING_OPTIMIZER='{optimizer}'. Expected randomized or optuna.")
    if cv_strategy not in {"holdout", "rolling"}:
        raise ValueError(f"Invalid CV_STRATEGY='{cv_strategy}'. Expected holdout or rolling.")

    repo_root = get_repo_root()
    n_jobs = _get_n_jobs()

    return RuntimeConfig(
        region=region,
        model=model,
        script_dir=script_dir,
        output_dir=output_dir,
        repo_root=repo_root,
        random_state=int(os.environ.get("TUNING_RANDOM_STATE", "42")),
        tuning_mode=tuning_mode,
        optimizer=optimizer,
        cv_strategy=cv_strategy,
        train_year_max=int(os.environ.get("TRAIN_YEAR_MAX", "2014")),
        val_year_min=int(os.environ.get("VAL_YEAR_MIN", "2015")),
        val_year_max=int(os.environ.get("VAL_YEAR_MAX", "2017")),
        rolling_folds=int(os.environ.get("ROLLING_FOLDS", "5")),
        tuning_split_version=os.environ.get("TUNING_SPLIT_VERSION", "main").strip() or "main",
        min_year=int(os.environ.get("TRANSITION_MIN_YEAR", "2001")),
        max_neg_per_year_fast=int(os.environ.get("MAX_NEG_PER_YEAR_FAST", "50000")),
        max_neg_per_year_paper=int(os.environ.get("MAX_NEG_PER_YEAR_PAPER", "150000")),
        # 0 = no cap (default for small regions like Colombia).
        # Set to e.g. 5000 for SA where the 5-yr lookahead makes positives dominant.
        max_pos_per_year_fast=int(os.environ.get("MAX_POS_PER_YEAR_FAST", "0")),
        max_pos_per_year_paper=int(os.environ.get("MAX_POS_PER_YEAR_PAPER", "0")),
        adaptive_cap_enabled=os.environ.get("ADAPTIVE_NEG_CAP", "0") == "1",
        adaptive_ratio_target=float(os.environ.get("TARGET_NEG_POS_RATIO", "50.0")),
        adaptive_min_cap=int(os.environ.get("ADAPTIVE_NEG_CAP_MIN", "20000")),
        adaptive_max_cap=int(os.environ.get("ADAPTIVE_NEG_CAP_MAX", "500000")),
        n_iter_lgbm_fast=int(os.environ.get("N_ITER_LGBM_FAST", "50")),
        n_iter_lgbm_paper=int(os.environ.get("N_ITER_LGBM_PAPER", "70")),
        n_iter_rf_fast=int(os.environ.get("N_ITER_RF_FAST", "25")),
        n_iter_rf_paper=int(os.environ.get("N_ITER_RF_PAPER", "60")),
        total_row_budget_lgbm_fast=int(os.environ.get("TOTAL_ROW_BUDGET_LGBM_FAST", "1500000")),
        total_row_budget_lgbm_paper=int(os.environ.get("TOTAL_ROW_BUDGET_LGBM_PAPER", "2500000")),
        total_row_budget_rf_fast=int(os.environ.get("TOTAL_ROW_BUDGET_RF_FAST", "300000")),
        total_row_budget_rf_paper=int(os.environ.get("TOTAL_ROW_BUDGET_RF_PAPER", "500000")),
        min_pos_per_year_after_budget=int(os.environ.get("MIN_POS_PER_YEAR_AFTER_BUDGET", "25")),
        rf_add_missing_indicators=os.environ.get("RF_ADD_MISSING_INDICATORS", "1") == "1",
        n_jobs=n_jobs,
    )


def _get_total_row_budget(cfg: RuntimeConfig) -> int:
    if cfg.model == "lgbm":
        return cfg.total_row_budget_lgbm_paper if cfg.tuning_mode == "paper" else cfg.total_row_budget_lgbm_fast
    return cfg.total_row_budget_rf_paper if cfg.tuning_mode == "paper" else cfg.total_row_budget_rf_fast


def _prepare_rf_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    add_missing_indicators: bool,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    X_df = df[feature_cols].copy()
    missing_counts = X_df.isna().sum(axis=0)
    missing_rates = (missing_counts / max(len(X_df), 1)).to_dict()

    impute_values = X_df.median(axis=0, skipna=True)
    impute_values = impute_values.fillna(0.0)
    X_imp = X_df.fillna(impute_values)

    indicator_cols: List[str] = []
    if add_missing_indicators:
        miss_cols = [col for col in feature_cols if int(missing_counts[col]) > 0]
        if miss_cols:
            # Build all indicator columns at once to avoid DataFrame fragmentation
            indicator_data = {
                f"{col}__is_missing": X_df[col].isna().astype(np.int8)
                for col in miss_cols
            }
            indicator_df = pd.DataFrame(indicator_data, index=X_imp.index)
            X_imp = pd.concat([X_imp, indicator_df], axis=1)
            indicator_cols = list(indicator_data.keys())
    indicator_feature_bases = {c.replace("__is_missing", "") for c in indicator_cols}

    used_features = feature_cols + indicator_cols
    X = X_imp[used_features].to_numpy(dtype=np.float32)
    stats = {
        "add_missing_indicators": bool(add_missing_indicators),
        "n_features_input": int(len(feature_cols)),
        "n_missing_indicator_features": int(len(indicator_cols)),
        "n_features_output": int(len(used_features)),
        "features_with_missing": int(sum(1 for c in feature_cols if int(missing_counts[c]) > 0)),
        "per_feature": {
            col: {
                "missing_count": int(missing_counts[col]),
                "missing_rate": float(missing_rates[col]),
                "impute_value": float(impute_values[col]),
                "missing_indicator_added": bool(col in indicator_feature_bases),
            }
            for col in feature_cols
        },
    }
    return X, used_features, stats


def _compute_fold_base_rates(
    folds: List[Tuple[np.ndarray, np.ndarray]], y: np.ndarray
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    fold_rates: List[Dict[str, Any]] = []
    spw_values: List[float] = []
    for idx, (train_idx, val_idx) in enumerate(folds, start=1):
        y_tr = y[train_idx]
        y_va = y[val_idx]
        tr_pos = int((y_tr > 0).sum())
        tr_neg = int((y_tr == 0).sum())
        va_pos = int((y_va > 0).sum())
        va_neg = int((y_va == 0).sum())
        tr_pos_rate = float(tr_pos / max(len(y_tr), 1))
        va_pos_rate = float(va_pos / max(len(y_va), 1))
        spw = float(tr_neg / max(tr_pos, 1))
        spw_values.append(spw)
        fold_rates.append(
            {
                "fold": int(idx),
                "train": {"n_pos": tr_pos, "n_neg": tr_neg, "pos_rate": tr_pos_rate, "scale_pos_weight_auto": spw},
                "val": {"n_pos": va_pos, "n_neg": va_neg, "pos_rate": va_pos_rate},
            }
        )
    spw_spread = {
        "min": float(min(spw_values)) if spw_values else 0.0,
        "max": float(max(spw_values)) if spw_values else 0.0,
        "mean": float(np.mean(spw_values)) if spw_values else 0.0,
    }
    return fold_rates, spw_spread


def _fit_randomized_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    random_state: int,
    n_jobs: int,
    n_iter: int,
    mode: str,
) -> Tuple[Dict[str, Any], float]:
    train_union_idx = np.unique(np.concatenate([tr for tr, _ in folds]))
    auto_spw = compute_auto_scale_pos_weight(y[train_union_idx])
    space = get_lgbm_randomized_space(mode=mode, auto_scale_pos_weight=auto_spw)
    fixed = get_lgbm_fixed_params(random_state=random_state, n_jobs=n_jobs)
    fixed["n_estimators"] = 1000
    # scale_pos_weight is not in the search grid; inject it as a fixed parameter
    # computed from the training union so evaluation stays in the rare-event regime.
    fixed["scale_pos_weight"] = auto_spw
    estimator = LGBMClassifier(**fixed)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=n_iter,
        cv=folds,
        scoring=PR_AUC_SCORER,
        n_jobs=1,
        random_state=random_state,
        verbose=2,
    )
    search.fit(X, y)
    return dict(search.best_params_), float(search.best_score_)


def _fit_randomized_rf(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    random_state: int,
    n_jobs: int,
    n_iter: int,
    mode: str,
) -> Tuple[Dict[str, Any], float]:
    space = get_rf_randomized_space(mode=mode)
    fixed = get_rf_fixed_params(random_state=random_state, n_jobs=n_jobs)
    estimator = RandomForestClassifier(**fixed)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=n_iter,
        cv=folds,
        scoring=PR_AUC_SCORER,
        n_jobs=1,
        random_state=random_state,
        verbose=2,
    )
    search.fit(X, y)
    return dict(search.best_params_), float(search.best_score_)


def _build_artifact_payload(
    cfg: RuntimeConfig,
    optimizer_used: str,
    timestamp: str,
    train_path: Path,
    feature_count: int,
    split_info: Dict[str, Any],
    best_params: Dict[str, Any],
    best_val_score: float,
    fixed_params: Dict[str, Any],
    search_details: Dict[str, Any],
    data_summary: DataSummary,
    year_sampling_stats: Dict[int, Dict[str, int]],
    feature_list_used: List[str],
    sampling_config: Dict[str, Any],
    wdpa_prev_diagnostics: Dict[str, Any],
    fold_base_rates: List[Dict[str, Any]],
    rf_preprocessing: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = build_metadata(
        train_path=train_path,
        target_col=TARGET_COL,
        feature_count=feature_count,
        year_split={
            "train_year_max": cfg.train_year_max,
            "val_year_min": cfg.val_year_min,
            "val_year_max": cfg.val_year_max,
            "min_year": cfg.min_year,
            "split_version": cfg.tuning_split_version,
        },
        timestamp=timestamp,
        mode=cfg.tuning_mode,
        optimizer=optimizer_used,
        cv_strategy=cfg.cv_strategy,
        repo_root=cfg.repo_root,
        seed=cfg.random_state,
    )
    return {
        "best_params": best_params,
        "best_val_score": float(best_val_score),
        "fixed_params": fixed_params,
        "search_details": search_details,
        "split_info": split_info,
        "data_summary": {
            "n_rows_raw": data_summary.n_rows_raw,
            "n_rows_clean": data_summary.n_rows_clean,
            "n_rows_sampled": data_summary.n_rows_sampled,
            "n_features": data_summary.n_features,
            "n_pos": data_summary.n_pos,
            "n_neg": data_summary.n_neg,
        },
        "year_sampling_stats": year_sampling_stats,
        "feature_list_used": feature_list_used,
        "sampling_config": sampling_config,
        "wdpa_prev_diagnostics": wdpa_prev_diagnostics,
        "fold_base_rates": fold_base_rates,
        "rf_preprocessing": rf_preprocessing or {},
        "metadata": metadata,
    }


def run_tuning(region: str, model: str, script_dir: Path, output_dir: Path) -> Dict[str, Any]:
    if model not in {"lgbm", "rf"}:
        raise ValueError(f"Unsupported model '{model}'")

    cfg = load_runtime_config(region=region, model=model, script_dir=script_dir, output_dir=output_dir)
    optimizer_used, optimizer_warning = _resolve_optimizer(cfg.optimizer)
    if optimizer_warning:
        print(f"[WARN] {optimizer_warning}")
    start = time.time()
    wandb_run = _init_wandb_tuning(cfg=cfg, optimizer_used=optimizer_used)

    try:
        max_neg_per_year = cfg.max_neg_per_year_paper if cfg.tuning_mode == "paper" else cfg.max_neg_per_year_fast
        max_pos_per_year = cfg.max_pos_per_year_paper if cfg.tuning_mode == "paper" else cfg.max_pos_per_year_fast
        total_row_budget = _get_total_row_budget(cfg)
        exclude_cols = {"transition_01", "transition_01_win5", "WDPA_b1", "WDPA_prev", "x", "y", "row", "col", YEAR_COL}
        scratch_root = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
        train_path = resolve_train_parquet(
            cfg.repo_root,
            region=region,
            scratch_root=scratch_root,
            split_version=cfg.tuning_split_version,
        )

        df, feature_cols, data_summary, year_stats, wdpa_diag, budget_diag = prepare_tuning_dataset(
            train_path=train_path,
            target_col=TARGET_COL,
            min_year=cfg.min_year,
            random_state=cfg.random_state,
            exclude_cols=exclude_cols,
            max_neg_per_year=max_neg_per_year,
            adaptive_cap_enabled=cfg.adaptive_cap_enabled,
            target_neg_pos_ratio=cfg.adaptive_ratio_target,
            adaptive_min_cap=cfg.adaptive_min_cap,
            adaptive_max_cap=cfg.adaptive_max_cap,
            total_row_budget=total_row_budget,
            min_pos_per_year_after_budget=cfg.min_pos_per_year_after_budget,
            max_pos_per_year=max_pos_per_year,
        )
        _report_memory_usage("after data sampling")

        split_cfg = SplitConfig(
            train_year_max=cfg.train_year_max,
            val_year_min=cfg.val_year_min,
            val_year_max=cfg.val_year_max,
            strategy=cfg.cv_strategy,
            rolling_folds=cfg.rolling_folds,
            year_col=YEAR_COL,
        )
        folds, split_info = build_splits(df=df, cfg=split_cfg)
        _report_memory_usage("after split construction")

        feature_list_used = list(feature_cols)
        rf_preprocessing: Dict[str, Any] | None = None
        if model == "rf":
            X, feature_list_used, rf_preprocessing = _prepare_rf_matrix(
                df=df,
                feature_cols=feature_cols,
                add_missing_indicators=cfg.rf_add_missing_indicators,
            )
        else:
            X = df[feature_cols].to_numpy(dtype=np.float32)
        y = (df[TARGET_COL].to_numpy() > 0).astype(np.int8)
        fold_base_rates, spw_spread = _compute_fold_base_rates(folds=folds, y=y)
        del df
        gc.collect()
        _report_memory_usage("after matrix materialization")

        if model == "lgbm":
            train_union_idx = np.concatenate([tr for tr, _ in folds])
            auto_spw = compute_auto_scale_pos_weight(y[train_union_idx])
            fixed_params = get_lgbm_fixed_params(cfg.random_state, cfg.n_jobs)
            fixed_params["metric"] = "average_precision"

            if optimizer_used == "optuna":
                from .optuna_runner import optimize_lgbm_optuna

                n_trials = cfg.n_iter_lgbm_paper if cfg.tuning_mode == "paper" else cfg.n_iter_lgbm_fast
                optuna_callbacks = _build_optuna_wandb_callbacks(wandb_run)
                best_params, best_val_score, fold_records = optimize_lgbm_optuna(
                    X=X,
                    y=y,
                    folds=folds,
                    mode=cfg.tuning_mode,
                    fixed_params=fixed_params,
                    auto_scale_pos_weight=auto_spw,
                    n_trials=n_trials,
                    random_state=cfg.random_state,
                    feature_names=feature_list_used,
                    callbacks=optuna_callbacks,
                )
                search_details = {
                    "optimizer": "optuna",
                    "requested_optimizer": cfg.optimizer,
                    "fallback_warning": optimizer_warning or "",
                    "n_trials": int(n_trials),
                    "fold_records": fold_records,
                    "auto_scale_pos_weight": float(auto_spw),
                    "fold_pos_rate_spw_spread": spw_spread,
                }
            else:
                n_iter = cfg.n_iter_lgbm_fast if cfg.tuning_mode == "fast" else cfg.n_iter_lgbm_paper
                best_params, best_val_score = _fit_randomized_lgbm(
                    X=X,
                    y=y,
                    folds=folds,
                    random_state=cfg.random_state,
                    n_jobs=cfg.n_jobs,
                    n_iter=n_iter,
                    mode=cfg.tuning_mode,
                )
                search_details = {
                    "optimizer": "randomized",
                    "requested_optimizer": cfg.optimizer,
                    "fallback_warning": optimizer_warning or "",
                    "n_iter": int(n_iter),
                    "auto_scale_pos_weight": float(auto_spw),
                    "fold_pos_rate_spw_spread": spw_spread,
                }

            output_name = "lgbm_best_params.json"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            payload = _build_artifact_payload(
                cfg=cfg,
                optimizer_used=optimizer_used,
                timestamp=timestamp,
                train_path=train_path,
                feature_count=len(feature_cols),
                split_info=split_info,
                best_params=best_params,
                best_val_score=best_val_score,
                fixed_params=fixed_params,
                search_details=search_details,
                data_summary=data_summary,
                year_sampling_stats=year_stats,
                feature_list_used=feature_list_used,
                sampling_config={
                    "max_neg_per_year": int(max_neg_per_year),
                    "max_pos_per_year": int(max_pos_per_year),
                    "adaptive_cap_enabled": bool(cfg.adaptive_cap_enabled),
                    "adaptive_ratio_target": float(cfg.adaptive_ratio_target),
                    "adaptive_min_cap": int(cfg.adaptive_min_cap),
                    "adaptive_max_cap": int(cfg.adaptive_max_cap),
                    "total_row_budget": int(total_row_budget),
                    "min_pos_per_year_after_budget": int(cfg.min_pos_per_year_after_budget),
                    "total_row_budget_diagnostics": budget_diag,
                },
                wdpa_prev_diagnostics=wdpa_diag,
                fold_base_rates=fold_base_rates,
                rf_preprocessing=rf_preprocessing,
            )
        else:
            fixed_params = get_rf_fixed_params(cfg.random_state, cfg.n_jobs)
            if optimizer_used == "optuna":
                from .optuna_runner import optimize_rf_optuna

                n_trials = cfg.n_iter_rf_paper if cfg.tuning_mode == "paper" else cfg.n_iter_rf_fast
                optuna_callbacks = _build_optuna_wandb_callbacks(wandb_run)
                best_params, best_val_score, fold_records = optimize_rf_optuna(
                    X=X,
                    y=y,
                    folds=folds,
                    mode=cfg.tuning_mode,
                    fixed_params=fixed_params,
                    n_trials=n_trials,
                    random_state=cfg.random_state,
                    callbacks=optuna_callbacks,
                )
                search_details = {
                    "optimizer": "optuna",
                    "requested_optimizer": cfg.optimizer,
                    "fallback_warning": optimizer_warning or "",
                    "n_trials": int(n_trials),
                    "fold_records": fold_records,
                    "fold_pos_rate_spw_spread": spw_spread,
                }
            else:
                n_iter = cfg.n_iter_rf_fast if cfg.tuning_mode == "fast" else cfg.n_iter_rf_paper
                best_params, best_val_score = _fit_randomized_rf(
                    X=X,
                    y=y,
                    folds=folds,
                    random_state=cfg.random_state,
                    n_jobs=cfg.n_jobs,
                    n_iter=n_iter,
                    mode=cfg.tuning_mode,
                )
                search_details = {
                    "optimizer": "randomized",
                    "requested_optimizer": cfg.optimizer,
                    "fallback_warning": optimizer_warning or "",
                    "n_iter": int(n_iter),
                    "fold_pos_rate_spw_spread": spw_spread,
                }

            output_name = "rf_best_params.json"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            payload = _build_artifact_payload(
                cfg=cfg,
                optimizer_used=optimizer_used,
                timestamp=timestamp,
                train_path=train_path,
                feature_count=len(feature_cols),
                split_info=split_info,
                best_params=best_params,
                best_val_score=best_val_score,
                fixed_params=fixed_params,
                search_details=search_details,
                data_summary=data_summary,
                year_sampling_stats=year_stats,
                feature_list_used=feature_list_used,
                sampling_config={
                    "max_neg_per_year": int(max_neg_per_year),
                    "max_pos_per_year": int(max_pos_per_year),
                    "adaptive_cap_enabled": bool(cfg.adaptive_cap_enabled),
                    "adaptive_ratio_target": float(cfg.adaptive_ratio_target),
                    "adaptive_min_cap": int(cfg.adaptive_min_cap),
                    "adaptive_max_cap": int(cfg.adaptive_max_cap),
                    "total_row_budget": int(total_row_budget),
                    "min_pos_per_year_after_budget": int(cfg.min_pos_per_year_after_budget),
                    "total_row_budget_diagnostics": budget_diag,
                },
                wdpa_prev_diagnostics=wdpa_diag,
                fold_base_rates=fold_base_rates,
                rf_preprocessing=rf_preprocessing,
            )

        training_dir = script_dir.parent / "5_training"
        if not training_dir.exists():
            # Backward-compatible fallback for alternate layouts.
            training_dir = script_dir.parent / "3_training"
        canonical_output_path = training_dir / output_name
        timestamped_output_path = script_dir / f"{output_name.replace('.json', '')}_{timestamp}.json"
        save_tuning_artifact(payload, canonical_path=canonical_output_path, timestamped_path=timestamped_output_path)

        elapsed = time.time() - start
        print("=" * 72)
        print(f"{region.upper()} {model.upper()} tuning completed in {elapsed/60:.1f} minutes")
        print(
            f"Mode: {cfg.tuning_mode} | Optimizer: {optimizer_used} (requested: {cfg.optimizer}) | CV: {cfg.cv_strategy} | "
            f"Split: {cfg.tuning_split_version}"
        )
        print(f"Rows sampled: {data_summary.n_rows_sampled:,} | Features: {data_summary.n_features}")
        print(f"Best PR-AUC: {payload['best_val_score']:.6f}")
        print(f"Canonical artifact: {canonical_output_path}")
        print(f"Timestamped artifact: {timestamped_output_path}")
        print("=" * 72)

        payload["paths"] = {
            "canonical_output_path": str(canonical_output_path),
            "timestamped_output_path": str(timestamped_output_path),
        }
        if wandb_run is not None:
            wandb_run.summary["best_pr_auc"] = float(payload["best_val_score"])
            wandb_run.summary["rows_sampled"] = int(data_summary.n_rows_sampled)
            wandb_run.summary["n_features"] = int(data_summary.n_features)
            wandb_run.summary["canonical_output_path"] = str(canonical_output_path)
            wandb_run.summary["timestamped_output_path"] = str(timestamped_output_path)
            wandb.log(
                {
                    "best_pr_auc": float(payload["best_val_score"]),
                    "rows_sampled": int(data_summary.n_rows_sampled),
                    "n_features": int(data_summary.n_features),
                    "elapsed_minutes": float(elapsed / 60.0),
                }
            )
        return payload
    finally:
        if wandb_run is not None:
            wandb_run.finish()
