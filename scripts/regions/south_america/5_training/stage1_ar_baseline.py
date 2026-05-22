#!/usr/bin/env python3
"""Pre-flight Check B — Stage 1 autoregressive baseline (momentum lags only).

Fits a Poisson GLM: pa_expansion_pixels ~ lag1 + lag2 + lag3 at country-year level.
Momentum lags are computed directly from panel aggregates (no feature_engineering rerun).

Metric: D² (deviance-based pseudo-R²) via sklearn model.score().
McFadden pseudo-R² is NOT used: it assumes LL ≤ 0 (logistic), but Poisson LL on
large counts is positive, which inverts the sign convention.

ROADMAP decision rule (D² on expansion counts):
  - >= 0.50  -> full political model likely credible
  - 0.30-0.50 -> Stage 1 illustrative context; paper lead stays on Stage 2
  - < 0.30   -> Stage 1 dominated by noise; supplement or Discussion only

Usage:
    python scripts/regions/south_america/5_training/stage1_ar_baseline.py
    python scripts/regions/south_america/5_training/stage1_ar_baseline.py \\
        --panel PATH_TO_merged_panel_final.parquet

Outputs:
    outputs/data_checks/stage1_ar_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Bootstrap repo root
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "README.md").is_file():
            return parent
    raise RuntimeError("Cannot find repo root")


_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.country_raster import resolve_panel_path
from scripts.regions.shared.stage1_panel import build_country_year_panel

TRAIN_YEARS = (2001, 2013)
OUT_PATH = _ROOT / "outputs" / "data_checks" / "stage1_ar_baseline.json"
REGION = "south_america"


def _decision_rule(pseudo_r2: float) -> str:
    if pseudo_r2 >= 0.50:
        return "credible_macro_model"
    if pseudo_r2 >= 0.30:
        return "illustrative_context"
    return "supplement_or_discussion_only"


def fit_ar_baseline(cy: pd.DataFrame) -> dict:
    """Poisson regression with momentum lags only."""
    feature_cols = [f"pa_momentum_pixels_lag{lag}" for lag in (1, 2, 3)]
    cy = cy.dropna(subset=feature_cols)
    X_raw = cy[feature_cols].to_numpy(dtype=np.float64)
    y = cy["pa_expansion_pixels"].to_numpy(dtype=np.float64)

    # Pixel counts reach ~350k; without scaling exp() overflows in the Poisson log-link
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = PoissonRegressor(alpha=0.0, max_iter=500)
    model.fit(X, y)

    # D² (deviance-based pseudo-R²): correct for Poisson; sklearn model.score() returns this
    pseudo_r2 = float(model.score(X, y))
    mu = np.clip(model.predict(X), 1e-9, None)
    rmse = float(np.sqrt(mean_squared_error(y, mu)))

    return {
        "n_country_years": int(len(cy)),
        "n_countries": int(cy["country_id"].nunique()),
        "year_range": [int(cy["year"].min()), int(cy["year"].max())],
        "pseudo_r2_d2": pseudo_r2,
        "rmse": rmse,
        "mean_expansion_pixels": float(y.mean()),
        "coefficients": {
            name: float(coef)
            for name, coef in zip(feature_cols, model.coef_.ravel())
        },
        "intercept": float(model.intercept_),
        "decision_rule": _decision_rule(pseudo_r2),
        "feature_cols": feature_cols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 AR baseline (Check B)")
    parser.add_argument("--panel", type=Path, default=None, help="merged_panel_final.parquet")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    panel_path = args.panel or resolve_panel_path(REGION)
    print(f"Panel: {panel_path}")
    print(f"Train years: {TRAIN_YEARS}")

    cy = build_country_year_panel(panel_path, REGION)
    print(f"Country-years: {len(cy):,} ({cy['country_id'].nunique()} countries)")

    result = fit_ar_baseline(cy)
    result["region"] = REGION
    result["panel_path"] = str(panel_path)
    result["train_years"] = list(TRAIN_YEARS)

    print(f"D² (deviance pseudo-R²): {result['pseudo_r2_d2']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"DECISION: {result['decision_rule']}")
    print("Coefficients:", result["coefficients"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nJSON written to {args.output}")


if __name__ == "__main__":
    main()
