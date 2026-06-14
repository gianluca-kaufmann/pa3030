"""Stage 2 XGBoost rank:ndcg training core.

Counterpart to stage2_lgbm_core.py for the XGBoost engine (Issue BB).
No 9K sub-window constraint, so eco grouping (W9a) is not needed here.
"""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from scripts.regions.shared.evaluation.stage2_metrics import (
    compute_stage2_metrics,
    ndcg_at_k_within_groups,
)
from scripts.regions.shared.training.feature_guard import check_feature_denylist
from scripts.regions.shared.training.stage2_lgbm_core import (
    STAGE2_EXCLUDE_COLS,
    TARGET_COL,
    GROUP_COLS,
    Stage2Config,
    _expansion_groups_from_batches,
    load_stage2_arrays,
    resolve_parquet_file,
)
from scripts.regions.shared.training.utils import compute_year_weights, get_repo_root

FIXED_PARAMS: Dict[str, Any] = {
    "objective": "rank:ndcg",
    "tree_method": "hist",
    "device": "cpu",
    "lambdarank_pair_method": "topk",
    "verbosity": 0,
    "seed": 42,
}


class XGBNdcg1PctEarlyStopper(xgb.callback.TrainingCallback):
    """Early-stop XGBoost on true NDCG@1% within full (country_id, year) groups.

    Mirrors _TrueNdcg1PctEarlyStop from stage2_lgbm_core so both engines
    are stopped and evaluated on the same metric. Tracks best_iter so
    callers can limit prediction to that many trees.
    """

    def __init__(
        self,
        dval: xgb.DMatrix,
        y_val: np.ndarray,
        group_val: np.ndarray,
        patience: int = 100,
        verbose: bool = True,
        log_interval: int = 50,
    ) -> None:
        super().__init__()
        self.dval = dval
        self.y_val = y_val.astype(np.float64)
        self.group_val = group_val
        self.patience = patience
        self.verbose = verbose
        self.log_interval = log_interval
        self.best_score: float = 0.0
        self.best_iter: int = 0

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        pred = model.predict(self.dval)
        score = ndcg_at_k_within_groups(self.y_val, pred, self.group_val, 1.0)
        if score > self.best_score:
            self.best_score = score
            self.best_iter = epoch
        if self.verbose and (epoch + 1) % self.log_interval == 0:
            print(
                f"[{epoch + 1}]  xgb ndcg@1%: {score:.6f}"
                f"  best: {self.best_score:.6f} @ iter {self.best_iter + 1}"
            )
        if epoch - self.best_iter >= self.patience:
            if self.verbose:
                print(
                    f"XGB early stop at iter {epoch + 1}."
                    f" Best: iter {self.best_iter + 1} ({self.best_score:.6f})"
                )
            return True
        return False


def resolve_xgb_best_params_json(cfg: Stage2Config) -> Optional[Path]:
    script_dir = get_repo_root() / "scripts" / "regions" / cfg.region / "5_training"
    candidates: List[Path] = [
        script_dir / f"{cfg.model_prefix}_stage2_xgb_best_params.json",
    ]
    if env_out_dir := os.environ.get("STAGE2_OUTPUT_DIR"):
        candidates.insert(0, Path(env_out_dir) / f"{cfg.model_prefix}_stage2_xgb_best_params.json")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _build_xgb_params(best_params: Dict[str, Any], n_jobs: int) -> Tuple[Dict[str, Any], int]:
    """Merge best_params with FIXED_PARAMS; return (params, num_boost_round)."""
    params = {**FIXED_PARAMS, **best_params}
    num_boost_round = int(params.pop("n_estimators", params.pop("num_boost_round", 1000)))
    params["nthread"] = n_jobs
    # Fixed params must not be overridden by tuning results
    params["objective"] = FIXED_PARAMS["objective"]
    params["lambdarank_pair_method"] = FIXED_PARAMS["lambdarank_pair_method"]
    return params, num_boost_round


def train_xgb_rank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    group_train: np.ndarray,
    year_weights_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    group_val: np.ndarray,
    params: Dict[str, Any],
    num_boost_round: int,
    patience: int = 100,
) -> Tuple[xgb.Booster, int]:
    """Train XGBoost rank:ndcg with true NDCG@1% early stopping.

    Returns (booster, best_iter). Use iteration_range=(0, best_iter+1) when
    predicting to limit inference to the best checkpoint.
    """
    # XGBoost rank:ndcg requires one weight per query group, not per row.
    # All rows in a (country_id, year) group share the same year weight.
    group_starts = np.concatenate([[0], np.cumsum(group_train[:-1])])
    group_weights = year_weights_train[group_starts]

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=group_weights)
    dtrain.set_group(group_train.tolist())

    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(group_val.tolist())

    early_stopper = XGBNdcg1PctEarlyStopper(
        dval, y_val, group_val, patience=patience, verbose=True
    )
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        callbacks=[early_stopper],
    )
    return booster, early_stopper.best_iter


def run_stage2_xgb_training(cfg: Stage2Config) -> None:
    """Train Stage 2 XGBoost model and write metrics + scored test parquet."""
    repo_root = get_repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_path = resolve_parquet_file(cfg.region, "train.parquet")
    earlystop_path = resolve_parquet_file(cfg.region, "earlystop.parquet")
    test_path = resolve_parquet_file(cfg.region, "test.parquet")

    out_dir = repo_root / f"outputs/{cfg.region}/results/ml_models"
    model_dir = repo_root / f"data/{cfg.region}/ml/models"
    # Redirect outputs to STAGE2_OUTPUT_DIR when set (Colombia dev vs SA production)
    if env_out_dir := os.environ.get("STAGE2_OUTPUT_DIR"):
        out_dir = Path(env_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    schema = pq.ParquetFile(train_path).schema_arrow
    feature_cols = [
        n for n, f in zip(schema.names, schema)
        if n not in STAGE2_EXCLUDE_COLS
        and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))
    ]
    check_feature_denylist(feature_cols, context=f"{cfg.region}/xgb/stage2")

    params_path = resolve_xgb_best_params_json(cfg)
    if params_path:
        raw = json.loads(params_path.read_text())
        best_params: Dict[str, Any] = raw.get("best_params", raw)
        print(f"Loaded best params: {params_path}")
    else:
        raise FileNotFoundError(
            f"No XGBoost best_params JSON found for {cfg.region}/{cfg.model_prefix}. "
            "Run tuning first."
        )

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    params, num_boost_round = _build_xgb_params(best_params, n_jobs)

    _nr_env = os.environ.get("STAGE2_NEG_RATIO", "100")
    neg_ratio: Optional[int] = int(_nr_env) if int(_nr_env) > 0 else None

    train_pool_years = (cfg.train_years[0], cfg.earlystop_years[1])
    print(f"Building expansion group index ({train_pool_years})...")
    t0 = time.time()
    expansion_groups = _expansion_groups_from_batches(
        train_path, cfg.region, train_pool_years
    ) | _expansion_groups_from_batches(earlystop_path, cfg.region, train_pool_years)
    print(f"  {len(expansion_groups):,} expansion country-years ({time.time()-t0:.1f}s)")

    print(f"Loading train (neg_ratio={neg_ratio})...")
    arr_tr = load_stage2_arrays(
        train_path, cfg.region, feature_cols, expansion_groups, cfg.train_years,
        neg_ratio=neg_ratio,
    )
    print(f"Loading earlystop (neg_ratio={neg_ratio})...")
    arr_es = load_stage2_arrays(
        earlystop_path, cfg.region, feature_cols, expansion_groups, cfg.earlystop_years,
        neg_ratio=neg_ratio,
    )

    year_weights = compute_year_weights(
        arr_tr.years, min_year=cfg.train_years[0], max_year=cfg.train_years[1]
    )

    print(
        f"Fitting XGBoost rank:ndcg: train {len(arr_tr.y):,} rows"
        f" / earlystop {len(arr_es.y):,} rows"
        f" | n_trees={num_boost_round}, patience=100"
    )
    booster, best_iter = train_xgb_rank(
        arr_tr.X, arr_tr.y, arr_tr.cy_group_sizes, year_weights,
        arr_es.X, arr_es.y, arr_es.cy_group_sizes,
        params, num_boost_round,
    )
    print(f"Training complete. Best iter: {best_iter + 1}")

    del arr_tr, arr_es
    gc.collect()

    tag = f"{cfg.model_prefix}_xgb_stage2"
    model_path = model_dir / f"{tag}_{timestamp}.ubj"
    booster.save_model(str(model_path))
    print(f"Model saved: {model_path}")

    test_expansion = _expansion_groups_from_batches(test_path, cfg.region, cfg.test_years)
    print("Loading test set (full, no neg_ratio)...")
    arr_te = load_stage2_arrays(
        test_path, cfg.region, feature_cols, test_expansion, cfg.test_years
    )
    dtest = xgb.DMatrix(arr_te.X)
    scores = booster.predict(dtest, iteration_range=(0, best_iter + 1)).astype(np.float64)
    test_metrics = compute_stage2_metrics(
        arr_te.y.astype(np.float64), scores, arr_te.cy_group_sizes
    )
    print(f"Test NDCG@1%:   {test_metrics['ndcg_at_1pct']:.4f}")
    print(f"Test Lift@1%:   {test_metrics['lift_at_1pct_within_groups']:.2f}x")
    print(f"Test Lift@5%:   {test_metrics['lift_at_5pct_within_groups']:.2f}x")

    scored = pd.DataFrame({
        "y_true": arr_te.y,
        "y_pred_score": scores,
        "year": arr_te.years,
        "country_id": arr_te.country_ids,
    })
    scored_path = out_dir / f"{tag}_scored_{timestamp}.parquet"
    pq.write_table(pa.Table.from_pandas(scored, preserve_index=False), scored_path)

    metrics = {
        "metadata": {
            "timestamp": timestamp,
            "model": "XGBoost_rank_ndcg",
            "task": "stage2_geographic_selection",
            "target_column": TARGET_COL,
            "region": cfg.region,
            "n_features": len(feature_cols),
            "neg_ratio": neg_ratio,
            "best_iteration": best_iter + 1,
            "num_boost_round": num_boost_round,
        },
        "temporal_split": {
            "train_years": list(cfg.train_years),
            "earlystop_years": list(cfg.earlystop_years),
            "test_years": list(cfg.test_years),
        },
        "test_performance": test_metrics,
        "model_parameters": params,
        "n_expansion_groups_train_pool": len(expansion_groups),
    }
    metrics_path = out_dir / f"{tag}_metrics_{timestamp}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved: {metrics_path}")
