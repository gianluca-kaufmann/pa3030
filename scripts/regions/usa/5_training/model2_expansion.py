#!/usr/bin/env python3
"""Stage 1 country-year PA expansion model (Poisson GLM) — USA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PANEL_PATH = _ROOT / "data" / "usa" / "stage1_panel.parquet"
OUT_DIR = _ROOT / "outputs" / "usa" / "results" / "ml_models"

LAG_COLS = ["pa_momentum_pixels_lag1", "pa_momentum_pixels_lag2", "pa_momentum_pixels_lag3"]
POLITICAL_COLS = [
    "v2x_polyarchy",
    "gov_wgi_ge_est",
    "gdp_per_capita",
    "agricultural_land_pct",
    "target_30x30",
    "cbd_meeting_year",
]


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 panel not found at {PANEL_PATH}. Run stage1_data_builder.py first."
        )
    cy = pd.read_parquet(PANEL_PATH)
    cy = cy[cy["year"].between(2001, 2013)].copy()

    feature_cols = [c for c in LAG_COLS + POLITICAL_COLS if c in cy.columns]
    cy[feature_cols] = cy[feature_cols].fillna(0)

    X_raw = cy[feature_cols].to_numpy(dtype=np.float64)
    y = cy["pa_expansion_pixels"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = PoissonRegressor(alpha=0.0, max_iter=1000)
    model.fit(X, y)
    pseudo_r2 = float(model.score(X, y))
    mu = np.clip(model.predict(X), 1e-9, None)
    rmse = float(np.sqrt(mean_squared_error(y, mu)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "region": "usa",
        "model": "poisson_glm",
        "pseudo_r2_d2": pseudo_r2,
        "rmse": rmse,
        "n_country_years": int(len(cy)),
        "feature_cols": feature_cols,
        "coefficients": {name: float(coef) for name, coef in zip(feature_cols, model.coef_.ravel())},
        "intercept": float(model.intercept_),
        "scaler_mean": {name: float(m) for name, m in zip(feature_cols, scaler.mean_)},
        "scaler_scale": {name: float(s) for name, s in zip(feature_cols, scaler.scale_)},
    }
    out_path = OUT_DIR / "model2_expansion_coefficients.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Stage 1 Poisson GLM D²: {pseudo_r2:.4f}, RMSE: {rmse:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
