#!/usr/bin/env python3
"""Recover a metrics JSON from an existing calibrated scored parquet.

The step-5 training scripts crashed after writing the scored parquet but before
writing the metrics JSON, so the metrics JSON is absent.  This script reads the
calibrated scored parquet (which contains y_true + y_pred_proba), recomputes all
metrics, and writes a stub metrics JSON in the format expected by model*_results.

Usage:
    python scripts/regions/shared/results/recover_metrics_json.py --region south_america --model-type lgbm
    python scripts/regions/shared/results/recover_metrics_json.py --region south_america --model-type rf
    python scripts/regions/shared/results/recover_metrics_json.py --region usa --model-type lgbm
    python scripts/regions/shared/results/recover_metrics_json.py --region usa --model-type rf
    python scripts/regions/shared/results/recover_metrics_json.py --region se_asia --model-type lgbm
    python scripts/regions/shared/results/recover_metrics_json.py --region se_asia --model-type rf

    # Or provide an explicit scored parquet path:
    python scripts/regions/shared/results/recover_metrics_json.py \
        --region south_america --model-type lgbm \
        --scored-parquet outputs/south_america/results/ml_models/main/model1_lgbm_scored_calibrated_isotonic_20260318_180255.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Bootstrap repo root on sys.path
_here = Path(__file__).resolve()
for _p in (_here.parent, *_here.parents):
    if (_p / "scripts").is_dir():
        sys.path.insert(0, str(_p))
        break

from scripts.regions.shared.training.utils import (
    get_repo_root,
    compute_metrics,
)

REGION_TO_MODEL_ID = {
    "south_america": "model1",
    "usa":           "model2",
    "se_asia":       "model3",
}


def find_latest_calibrated_parquet(ml_models_dir: Path, model_id: str, model_type: str) -> Path:
    """Find the most recent calibrated scored parquet in ml_models_dir/main/."""
    main_dir = ml_models_dir / "main"
    if model_type == "lgbm":
        pattern = f"{model_id}_lgbm_scored_calibrated_*.parquet"
    else:
        pattern = f"{model_id}_rf_scored_calibrated_*.parquet"
    candidates = sorted(main_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No calibrated scored parquet matching '{pattern}' found in {main_dir}.\n"
            f"Download from Euler: scp euler:master_thesis/{main_dir.relative_to(get_repo_root())}/"
            f"{pattern} {main_dir}/"
        )
    return candidates[0]


def load_y_true_and_proba(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read y_true and y_pred_proba from a scored parquet."""
    pf = pq.ParquetFile(parquet_path)
    schema_cols = pf.schema_arrow.names
    if "y_true" not in schema_cols:
        raise ValueError(f"'y_true' column not found in {parquet_path.name}. "
                         f"Columns: {schema_cols}")
    # Prefer the calibrated probability column if present
    if "y_pred_proba_calibrated" in schema_cols:
        proba_col = "y_pred_proba_calibrated"
    elif "y_pred_proba" in schema_cols:
        proba_col = "y_pred_proba"
    else:
        raise ValueError(f"No probability column found in {parquet_path.name}")

    print(f"  Reading {parquet_path.name} ...")
    print(f"  Using probability column: {proba_col}")
    df = pq.read_table(parquet_path, columns=["y_true", proba_col]).to_pandas()
    df = df.dropna(subset=["y_true", proba_col])
    y_true  = df["y_true"].to_numpy(dtype=np.int8)
    y_proba = df[proba_col].to_numpy(dtype=np.float32)
    print(f"  Loaded {len(y_true):,} rows  |  positives: {y_true.sum():,}  "
          f"({y_true.mean()*100:.3f}%)")
    return y_true, y_proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover metrics JSON from calibrated scored parquet.")
    parser.add_argument("--region",        required=True, choices=list(REGION_TO_MODEL_ID))
    parser.add_argument("--model-type",    required=True, choices=["lgbm", "rf"])
    parser.add_argument("--scored-parquet", default=None,
                        help="Explicit path to calibrated scored parquet (auto-discovered if omitted)")
    args = parser.parse_args()

    repo_root    = get_repo_root()
    model_id     = REGION_TO_MODEL_ID[args.region]
    model_type   = args.model_type
    ml_models_dir = repo_root / f"outputs/{args.region}/results/ml_models"

    # ── Find scored parquet ──────────────────────────────────────────────────
    if args.scored_parquet:
        parquet_path = Path(args.scored_parquet).resolve()
        if not parquet_path.exists():
            print(f"ERROR: --scored-parquet not found: {parquet_path}")
            sys.exit(1)
    else:
        parquet_path = find_latest_calibrated_parquet(ml_models_dir, model_id, model_type)
    print(f"Scored parquet: {parquet_path}")

    # ── Load and compute metrics ─────────────────────────────────────────────
    y_true, y_proba = load_y_true_and_proba(parquet_path)
    print("Computing metrics ...")
    test_metrics = compute_metrics(y_true, y_proba)
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"  P@1%:    {test_metrics['precision_at_1pct']:.4f}  ({test_metrics['lift_at_1pct']:.1f}x)")
    print(f"  Brier:   {test_metrics['brier_score']:.5f}")

    # ── Build stub metrics dict ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Try to extract original run timestamp from parquet filename
    import re
    m = re.search(r"(\d{8}_\d{6})", parquet_path.name)
    source_ts = m.group(1) if m else timestamp

    metrics = {
        "metadata": {
            "timestamp":    source_ts,
            "recovered_at": timestamp,
            "source":       str(parquet_path),
            "note":         "Recovered from calibrated scored parquet — original metrics JSON was lost",
            "features":     [],   # unknown; SHAP will be skipped (use --skip-shap)
        },
        "data": {
            "n_test":         int(len(y_true)),
            "test_positives": int(y_true.sum()),
            "test_negatives": int((y_true == 0).sum()),
        },
        "temporal_split": {
            "test_years": [2017, 2019],
        },
        "test_performance": test_metrics,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    if model_type == "lgbm":
        out_name = f"{model_id}_lgbm_metrics_{source_ts}.json"
    else:
        out_name = f"{model_id}_rf_metrics_{source_ts}.json"
    out_path = ml_models_dir / out_name
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("Run results script with --skip-shap (feature_cols not available in recovered JSON).")


if __name__ == "__main__":
    main()
