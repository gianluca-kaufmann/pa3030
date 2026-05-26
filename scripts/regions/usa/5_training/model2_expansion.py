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

TRAIN_YEARS = (2001, 2016)
TEST_YEARS = (2017, 2024)

# NOTE: USA Stage 1 has only 1 country × 16 training years = 16 observations.
# With 14+ features this is severely underdetermined even with ridge regularisation.
# POLITICAL_COLS will be filtered to whatever is non-null in the panel, but interpret
# USA Stage 1 coefficients with extreme caution — the model is essentially
# fitting a time-series trend, not cross-country political variation.
LAG_COLS = [
    "pa_momentum_pixels_lag1",
    "pa_momentum_pixels_lag2",
    "pa_momentum_pixels_lag3",
    "pa_cumsum_lag1_pixels",
]
POLITICAL_COLS = [
    "v2x_polyarchy",
    "v2x_corr",
    "v2cseeorgs",
    "gov_wgi_ge_est",
    "gov_wgi_rl_est",
    "gdp_per_capita",
    "gdp_growth_lag1",
    "agricultural_land_pct",
    "forest_area_pct",    # forest area % of land (WDI AG.LND.FRST.ZS)
    "target_30x30",
    "cbd_meeting_year",
    "years_to_next_election",
    "redd_plus_enrolled",
]


def _d2(model: PoissonRegressor, X: np.ndarray, y: np.ndarray) -> float:
    return float(model.score(X, y))


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 panel not found at {PANEL_PATH}. Run stage1_data_builder.py first."
        )
    cy = pd.read_parquet(PANEL_PATH)

    # USA: 1 country × 16 training years = 16 obs. With 16 features the model is
    # underdetermined even at alpha=10 (numerical overflow, Issue K). Restrict to
    # momentum/saturation features only — treated as time-series trend extrapolation
    # in the paper, not cross-country political evidence.
    feature_cols = [c for c in LAG_COLS if c in cy.columns]
    cy[feature_cols] = cy[feature_cols].fillna(0)

    train = cy[cy["year"].between(*TRAIN_YEARS)].copy()
    test = cy[cy["year"].between(*TEST_YEARS)].copy()

    X_train_raw = train[feature_cols].to_numpy(dtype=np.float64)
    y_train = train["pa_expansion_pixels"].to_numpy(dtype=np.float64)
    X_test_raw = test[feature_cols].to_numpy(dtype=np.float64)
    y_test = test["pa_expansion_pixels"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # alpha=10 (not 0.1): USA has 1 country × 16 training years = 16 obs for 16 features.
    # Weak regularisation causes numerical overflow (Issue K). Interpret coefficients
    # as trend extrapolation only, not cross-country political evidence.
    model = PoissonRegressor(alpha=10.0, max_iter=1000)
    model.fit(X_train, y_train)

    d2_train = _d2(model, X_train, y_train)
    mu_train = np.clip(model.predict(X_train), 1e-9, None)
    rmse_train = float(np.sqrt(mean_squared_error(y_train, mu_train)))

    d2_test: float | None = None
    rmse_test: float | None = None
    if len(test) > 0:
        d2_test = _d2(model, X_test, y_test)
        mu_test = np.clip(model.predict(X_test), 1e-9, None)
        rmse_test = float(np.sqrt(mean_squared_error(y_test, mu_test)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "region": "usa",
        "model": "poisson_glm",
        "alpha": 10.0,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "pseudo_r2_d2_train": d2_train,
        "pseudo_r2_d2_test": d2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "feature_cols": feature_cols,
        "coefficients": {name: float(coef) for name, coef in zip(feature_cols, model.coef_.ravel())},
        "intercept": float(model.intercept_),
        "scaler_mean": {name: float(m) for name, m in zip(feature_cols, scaler.mean_)},
        "scaler_scale": {name: float(s) for name, s in zip(feature_cols, scaler.scale_)},
    }
    out_path = OUT_DIR / "model2_expansion_coefficients.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Stage 1 Poisson GLM — USA")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Train D²: {d2_train:.4f}  RMSE: {rmse_train:.4f}  (n={len(train)})")
    if d2_test is not None:
        print(f"  Test  D²: {d2_test:.4f}  RMSE: {rmse_test:.4f}  (n={len(test)})")
    else:
        print("  Test: no rows in test window")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
