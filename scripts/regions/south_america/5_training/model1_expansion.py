#!/usr/bin/env python3
"""Stage 1 country-year PA expansion model (Poisson GLM) — South America."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PANEL_PATH = _ROOT / "data" / "south_america" / "stage1_panel.parquet"
OUT_DIR = _ROOT / "outputs" / "south_america" / "results" / "ml_models"

LAG_COLS = ["pa_momentum_pixels_lag1", "pa_momentum_pixels_lag2", "pa_momentum_pixels_lag3"]
POLITICAL_COLS = [
    "v2x_polyarchy",
    "gov_wgi_ge_est",
    "gdp_per_capita",
    "agricultural_land_pct",
    "target_30x30",
    "cbd_meeting_year",
]


def _mcfadden_pseudo_r2(y: np.ndarray, mu: np.ndarray, mu_null: np.ndarray) -> float:
    eps = 1e-9
    ll = np.sum(y * np.log(mu + eps) - mu)
    ll0 = np.sum(y * np.log(mu_null + eps) - mu_null)
    return 1.0 - ll / ll0 if ll0 != 0 else float("nan")


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 panel not found at {PANEL_PATH}. Run stage1_data_builder.py first."
        )
    cy = pd.read_parquet(PANEL_PATH)
    cy = cy[cy["year"].between(2001, 2013)].copy()

    feature_cols = [c for c in LAG_COLS + POLITICAL_COLS if c in cy.columns]
    cy[feature_cols] = cy[feature_cols].fillna(0)

    X = cy[feature_cols].to_numpy(dtype=np.float64)
    y = cy["pa_expansion_pixels"].to_numpy(dtype=np.float64)

    model = PoissonRegressor(alpha=0.0, max_iter=1000)
    model.fit(X, y)
    mu = np.clip(model.predict(X), 1e-9, None)
    mu_null = np.full_like(y, y.mean())
    pseudo_r2 = _mcfadden_pseudo_r2(y, mu, mu_null)
    rmse = float(np.sqrt(mean_squared_error(y, mu)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "region": "south_america",
        "model": "poisson_glm",
        "pseudo_r2_mcfadden": pseudo_r2,
        "rmse": rmse,
        "n_country_years": int(len(cy)),
        "feature_cols": feature_cols,
        "coefficients": {name: float(coef) for name, coef in zip(feature_cols, model.coef_.ravel())},
        "intercept": float(model.intercept_),
    }
    out_path = OUT_DIR / "model1_expansion_coefficients.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Stage 1 Poisson GLM pseudo-R²: {pseudo_r2:.4f}, RMSE: {rmse:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
