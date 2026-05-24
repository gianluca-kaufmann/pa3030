#!/usr/bin/env python3
"""Two-stage forward inference: Stage 1 budget × Stage 2 geographic ranking.

Combines country-year expansion estimates with per-pixel LambdaRank scores
to produce forward_scored_2024_twostage.parquet for scenario / map pipelines.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.regions.shared.country_raster import country_ids_for_rows, load_country_raster
from scripts.regions.shared.forward.config import (
    DATA_SUBDIR,
    MODEL_PREFIX,
    OUTPUTS_SUBDIR,
    forward_dir_search_paths,
    get_repo_root,
    ml_models_dir_search_paths,
    resolve_forward_dir,
)
from scripts.regions.shared.forward.predict_core import resolve_features_parquet
from scripts.regions.shared.training.feature_guard import check_feature_denylist
from scripts.regions.shared.training.stage2_lgbm_core import _rank_normalize_within_groups

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "500000"))


def _resolve_stage2_artifact(repo_root: Path, region: str, model_prefix: str) -> Path:
    pattern = f"{model_prefix}_lgbm_stage2_*.pkl"
    for model_dir in ml_models_dir_search_paths(repo_root, f"{region}/ml"):
        cands = sorted(model_dir.glob(pattern), reverse=True)
        if cands:
            return cands[0]
    raise FileNotFoundError(f"No {pattern} found for two-stage forward inference.")


def _resolve_stage1_coefficients(repo_root: Path, region: str) -> dict:
    path = repo_root / f"outputs/{region}/results/ml_models/model1_expansion_coefficients.json"
    if region != "south_america":
        path = repo_root / f"outputs/{region}/results/ml_models/{MODEL_PREFIX}_expansion_coefficients.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 coefficients not found: {path}")
    return json.loads(path.read_text())


def _predict_country_budgets(coeffs: dict, region: str, forecast_year: int = 2025) -> dict[int, int]:
    """Map country_id -> pixel budget (expected new designations).

    Applies the training scaler then the Poisson log-link (exp) to recover
    the expected pixel count from the linear predictor.
    """
    panel_path = get_repo_root() / f"data/{region}/stage1_panel.parquet"
    if not panel_path.exists():
        raster = load_country_raster(region)
        n_countries = int(raster.max())
        return {cid: 100 for cid in range(1, n_countries + 1)}

    cy = pd.read_parquet(panel_path)
    cy = cy[cy["year"] == cy["year"].max()].copy()
    feature_cols = coeffs.get("feature_cols", [])
    coef_map = coeffs.get("coefficients", {})
    intercept = coeffs.get("intercept", 0.0)
    scaler_mean = coeffs.get("scaler_mean", {})
    scaler_scale = coeffs.get("scaler_scale", {})

    mean_arr = np.array([scaler_mean.get(c, 0.0) for c in feature_cols], dtype=np.float64)
    scale_arr = np.array([scaler_scale.get(c, 1.0) for c in feature_cols], dtype=np.float64)
    scale_arr = np.where(scale_arr > 0, scale_arr, 1.0)
    w = np.array([coef_map.get(c, 0.0) for c in feature_cols], dtype=np.float64)

    budgets: dict[int, int] = {}
    for _, row in cy.iterrows():
        cid = int(row["country_id"])
        x_raw = np.array([float(row.get(c, 0)) for c in feature_cols], dtype=np.float64)
        x = (x_raw - mean_arr) / scale_arr
        eta = float(intercept + x @ w) if len(x) else 0.0
        # Poisson log-link: E[y] = exp(eta); clip eta to prevent overflow
        mu = float(np.exp(np.clip(eta, -10.0, 20.0)))
        budgets[cid] = max(0, int(round(mu)))
    return budgets


def run_two_stage_inference(
    region: str | None = None,
    model_prefix: str | None = None,
) -> Path:
    repo_root = get_repo_root()
    region = region or os.environ.get("PA3030_FORWARD_REGION", "south_america")
    model_prefix = model_prefix or MODEL_PREFIX

    features_path = resolve_features_parquet(repo_root, OUTPUTS_SUBDIR)
    artifact_path = _resolve_stage2_artifact(repo_root, region, model_prefix)
    output_dir = resolve_forward_dir(repo_root, OUTPUTS_SUBDIR) / "lgbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "forward_scored_2024_twostage.parquet"

    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    check_feature_denylist(feature_cols, context="forward/two_stage_predict_core")

    coeffs = _resolve_stage1_coefficients(repo_root, region)
    budgets = _predict_country_budgets(coeffs, region)
    raster = load_country_raster(region)

    print(f"Two-stage forward: {features_path}")
    print(f"  Stage 2 artifact: {artifact_path.name}")
    print(f"  Countries with budget > 0: {sum(1 for v in budgets.values() if v > 0)}")

    schema = pq.ParquetFile(features_path).schema_arrow.names
    read_cols = [c for c in ["row", "col", "x", "y"] if c in schema] + feature_cols
    frames: list[pd.DataFrame] = []

    # Load all data first — rank normalisation must be applied per country over
    # the full country pixel set, so predictions cannot be computed per batch.
    pf = pq.ParquetFile(features_path)
    for batch in pf.iter_batches(columns=read_cols, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        df["country_id"] = country_ids_for_rows(df, raster)
        frames.append(df)

    scored = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # Apply within-country rank normalisation then predict.
    # The Stage 2 model was trained on within-(country,year) percentile ranks [0,1].
    # At inference time we have no year, so we normalise within country — the same
    # relative-position logic, applied to the full 2024 pixel set per country.
    scored["stage2_score"] = np.nan
    for cid in sorted(scored["country_id"].unique()):
        mask = scored["country_id"] == cid
        X_cid = scored.loc[mask, feature_cols].to_numpy(dtype=np.float32)
        group_sizes = np.array([len(X_cid)], dtype=np.int32)
        X_cid = _rank_normalize_within_groups(X_cid, group_sizes)
        scored.loc[mask, "stage2_score"] = model.predict(X_cid).astype(np.float64)
    scored["selected_twostage"] = 0
    for cid, budget in budgets.items():
        if budget <= 0:
            continue
        mask = scored["country_id"] == cid
        sub = scored.loc[mask].nlargest(budget, "stage2_score")
        scored.loc[sub.index, "selected_twostage"] = 1

    scored["y_pred_proba_5yr_cumulative"] = (
        scored["stage2_score"] - scored["stage2_score"].min()
    ) / (scored["stage2_score"].max() - scored["stage2_score"].min() + 1e-9)

    pq.write_table(pa.Table.from_pandas(scored, preserve_index=False), out_path)
    print(f"Wrote {len(scored):,} rows → {out_path}")
    print(f"  Selected pixels: {int(scored['selected_twostage'].sum()):,}")
    gc.collect()
    return out_path


def main() -> None:
    run_two_stage_inference()


if __name__ == "__main__":
    main()
