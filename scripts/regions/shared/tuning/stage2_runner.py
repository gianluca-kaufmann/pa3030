"""Orchestrate Stage 2 LambdaRank hyperparameter tuning for a region."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.regions.shared.training.feature_guard import check_feature_denylist
from scripts.regions.shared.training.stage2_lgbm_core import (
    STAGE2_EXCLUDE_COLS,
    TARGET_COL,
    _expansion_groups_from_batches,
    _scan_true_class_counts,
    load_stage2_arrays,
    resolve_parquet_file,
)
from scripts.regions.shared.tuning.cv import SplitConfig, build_splits
from scripts.regions.shared.tuning.search_spaces import get_lgbm_stage2_fixed_params
from scripts.regions.shared.tuning.stage2_optuna_runner import (
    optimize_lgbm_stage2_binary_optuna,
    optimize_lgbm_stage2_optuna,
)
from scripts.regions.shared.training.utils import get_repo_root


def run_stage2_tuning(
    region: str,
    model_prefix: str,
    *,
    n_trials: int = 50,
    mode: str = "fast",
    output_dir: Path | None = None,
    variant: str = "lambdarank",
) -> Path:
    repo_root = get_repo_root()
    script_dir = repo_root / "scripts" / "regions" / region / "5_training"
    # STAGE2_OUTPUT_DIR: redirect best_params.json for dev/Colombia panel runs so
    # Colombia tuning artefacts don't overwrite production SA best_params.
    if env_out_dir := os.environ.get("STAGE2_OUTPUT_DIR"):
        out_dir = Path(env_out_dir)
    else:
        out_dir = output_dir or script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_parquet_file(region, "train.parquet")
    earlystop_path = resolve_parquet_file(region, "earlystop.parquet")

    import pyarrow.parquet as pq

    import pyarrow as pa

    schema = pq.ParquetFile(train_path).schema_arrow
    feature_cols = []
    for name in schema.names:
        if name in STAGE2_EXCLUDE_COLS:
            continue
        f = schema.field(name)
        if pa.types.is_integer(f.type) or pa.types.is_floating(f.type):
            feature_cols.append(name)
    check_feature_denylist(feature_cols, context=f"{region}/stage2/tuning")

    expansion_groups = _expansion_groups_from_batches(
        train_path, region, (2001, 2016)
    ) | _expansion_groups_from_batches(earlystop_path, region, (2001, 2016))

    # Both binary and LambdaRank use STAGE2_NEG_RATIO for memory-safe loading.
    # Binary uses scale_pos_weight for class-balance signalling: the true SPW
    # is pre-scanned from the full data so it reflects the real imbalance (~300–500
    # for SA), not the subsampled ratio (~neg_ratio). LambdaRank must match the
    # tuning neg_ratio so hyperparameters are valid for the training regime.
    _nr_env = os.environ.get("STAGE2_NEG_RATIO", "100")
    neg_ratio: int | None = int(_nr_env) if int(_nr_env) > 0 else None

    # Issue O fix: pre-scan for true class balance before neg_ratio subsampling.
    true_spw: float | None = None
    if variant == "binary":
        print("Pre-scanning train + earlystop for true class counts (Issue O fix)...")
        _n_pos_tr, _n_neg_tr = _scan_true_class_counts(
            train_path, region, expansion_groups, (2001, 2013)
        )
        _n_pos_es, _n_neg_es = _scan_true_class_counts(
            earlystop_path, region, expansion_groups, (2014, 2016)
        )
        _n_pos_total = _n_pos_tr + _n_pos_es
        _n_neg_total = _n_neg_tr + _n_neg_es
        true_spw = float(_n_neg_total) / float(max(_n_pos_total, 1))
        print(
            f"  n_pos={_n_pos_total:,}, n_neg={_n_neg_total:,}, true_SPW={true_spw:.1f}"
            f" (vs subsampled SPW≈{neg_ratio})"
        )

    X_tr, y_tr, years_tr, cid_tr, _ = load_stage2_arrays(
        train_path, region, feature_cols, expansion_groups, (2001, 2013), neg_ratio=neg_ratio
    )
    X_es, y_es, years_es, cid_es, _ = load_stage2_arrays(
        earlystop_path, region, feature_cols, expansion_groups, (2014, 2016), neg_ratio=neg_ratio
    )
    X = np.vstack([X_tr, X_es])
    y = np.concatenate([y_tr, y_es])
    years = np.concatenate([years_tr, years_es])
    country_id = np.concatenate([cid_tr, cid_es])

    df_index = pd.DataFrame({"year": years, "country_id": country_id})
    cfg = SplitConfig(train_year_max=2014, val_year_min=2015, val_year_max=2017, strategy="rolling", rolling_folds=3)
    folds, _ = build_splits(df_index, cfg)

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

    if variant == "binary":
        best_params, best_score, records = optimize_lgbm_stage2_binary_optuna(
            X, y, country_id, years, folds, mode, n_trials, 42,
            true_scale_pos_weight=true_spw,
        )
        metric_name = "ndcg_at_1pct_within_groups (binary)"
        fixed: dict = {}
    else:
        fixed = get_lgbm_stage2_fixed_params(42, n_jobs)
        best_params, best_score, records = optimize_lgbm_stage2_optuna(
            X, y, country_id, years, folds, mode, fixed, n_trials, 42
        )
        metric_name = "ndcg_at_1pct_within_groups"

    artifact = {
        "best_params": best_params,
        "best_val_score": best_score,
        "metric": metric_name,
        "fixed_params": fixed,
        "metadata": {
            "region": region,
            "model_prefix": model_prefix,
            "variant": variant,
            "target_col": TARGET_COL,
            "feature_count": len(feature_cols),
            "neg_ratio": neg_ratio,
            "true_scale_pos_weight": true_spw,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "tuning_records": records,
    }
    suffix = "_binary" if variant == "binary" else ""
    out_path = out_dir / f"{model_prefix}_stage2_lgbm{suffix}_best_params.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Stage 2 tuning complete. Best NDCG@1% = {best_score:.4f}")
    print(f"Saved: {out_path}")
    return out_path
