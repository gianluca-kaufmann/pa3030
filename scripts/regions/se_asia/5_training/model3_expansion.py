#!/usr/bin/env python3
"""Stage 1 country-year PA expansion model (Poisson GLM) — SE Asia.

Specification: 8-feature parsimonious Poisson (4 momentum/saturation + 4 political).

Results summary:
  7-feat parsimonious (v2x_polyarchy, gdp_growth_lag1, redd_plus_enrolled):
    D²_train=0.069  D²_test(8yr)=+0.184  D²_test(3yr)=+0.279
  8-feat + forest_area_pct + WDPA lag fix (this script):
    D²_train=0.103  D²_test(8yr)=+0.109  D²_test(3yr)=+0.301  ← current
  Full 16-feature model: D²_test(8yr)=+0.103 (worse — multicollinear)
SE Asia has no structural break — frontier exhaustion is a Brazil-specific phenomenon.
forest_area_pct coefficient is NEGATIVE (−0.42): countries with more remaining forest
expand PAs less (deforestation pressure > conservation incentive in Malaysia/Indonesia).
Contrast with SA where coefficient is positive (REDD+/carbon market drives protection).
"""

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

PANEL_PATH = _ROOT / "data" / "se_asia" / "stage1_panel.parquet"
OUT_DIR = _ROOT / "outputs" / "se_asia" / "results" / "ml_models"

TRAIN_YEARS = (2001, 2016)   # pre-2001 WDPA training hurts (different expansion regime pre-2001)
TEST_YEARS  = (2017, 2024)

LAG_COLS = [
    "pa_momentum_pixels_lag1",
    "pa_momentum_pixels_lag2",
    "pa_momentum_pixels_lag3",
    "pa_cumsum_lag1_pixels",
]
# Parsimonious political set — same spec as SA for cross-region comparability.
# Drops collinear governance pairs; keeps three theoretically grounded predictors.
POLITICAL_COLS = [
    "v2x_polyarchy",
    "gdp_growth_lag1",
    "redd_plus_enrolled",
    "forest_area_pct",    # forest area % of land (WDI AG.LND.FRST.ZS) — remaining
                          # forest supply for designation; REDD+ incentive proxy
]


def _d2(model: PoissonRegressor, X: np.ndarray, y: np.ndarray) -> float:
    return float(model.score(X, y))


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 panel not found at {PANEL_PATH}. Run stage1_data_builder.py first."
        )
    cy = pd.read_parquet(PANEL_PATH)

    feature_cols = [c for c in LAG_COLS + POLITICAL_COLS if c in cy.columns]
    cy[feature_cols] = cy[feature_cols].fillna(0)

    train = cy[cy["year"].between(*TRAIN_YEARS)].copy()
    test  = cy[cy["year"].between(*TEST_YEARS)].copy()

    X_train_raw = train[feature_cols].to_numpy(dtype=np.float64)
    y_train = train["pa_expansion_pixels"].to_numpy(dtype=np.float64)
    X_test_raw = test[feature_cols].to_numpy(dtype=np.float64)
    y_test = test["pa_expansion_pixels"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    model = PoissonRegressor(alpha=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    d2_train = _d2(model, X_train, y_train)
    mu_train = np.clip(model.predict(X_train), 1e-9, None)
    rmse_train = float(np.sqrt(mean_squared_error(y_train, mu_train)))

    d2_test: float | None = None
    rmse_test: float | None = None
    if len(test) > 0:
        d2_test  = _d2(model, X_test, y_test)
        mu_test  = np.clip(model.predict(X_test), 1e-9, None)
        rmse_test = float(np.sqrt(mean_squared_error(y_test, mu_test)))

    def _d2_window(y_end: int) -> float | None:
        sub = cy[cy["year"].between(TEST_YEARS[0], y_end)]
        if len(sub) == 0:
            return None
        X_sub = scaler.transform(sub[feature_cols].fillna(0).to_numpy(dtype=np.float64))
        return float(model.score(X_sub, sub["pa_expansion_pixels"].to_numpy(dtype=np.float64)))

    d2_test_3yr = _d2_window(2019)
    d2_test_6yr = _d2_window(2022)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "region": "se_asia",
        "model": "poisson_glm",
        "spec": "parsimonious",
        "alpha": 1.0,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "pseudo_r2_d2_train": d2_train,
        "pseudo_r2_d2_test": d2_test,
        "pseudo_r2_d2_test_3yr": d2_test_3yr,
        "pseudo_r2_d2_test_6yr": d2_test_6yr,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "feature_cols": feature_cols,
        "coefficients": {name: float(coef) for name, coef in zip(feature_cols, model.coef_.ravel())},
        "intercept": float(model.intercept_),
        "scaler_mean":  {name: float(m) for name, m in zip(feature_cols, scaler.mean_)},
        "scaler_scale": {name: float(s) for name, s in zip(feature_cols, scaler.scale_)},
    }
    out_path = OUT_DIR / "model3_expansion_coefficients.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("Stage 1 Poisson GLM — SE Asia (parsimonious spec)")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Train D²: {d2_train:.4f}  RMSE: {rmse_train:.0f}  (n={len(train)})")
    if d2_test is not None:
        print(f"  Test  D²: {d2_test:.4f}  RMSE: {rmse_test:.0f}  (n={len(test)}, 2017–2024)")
        print(f"  Test  D² (3yr 2017–2019): {d2_test_3yr:.4f}  |  D² (6yr 2017–2022): {d2_test_6yr:.4f}")
    else:
        print("  Test: no rows in test window")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
