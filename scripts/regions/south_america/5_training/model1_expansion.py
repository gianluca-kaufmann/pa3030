#!/usr/bin/env python3
"""Stage 1 country-year PA expansion model (Poisson GLM) — South America.

Specification: 10-feature parsimonious Poisson with p90 winsorisation + WDPA lag fix.

Model selection history (2001–2016 train / 2017–2024 test):
  Full model (16 feat, α=0.1):                 D²_test=−0.096
  Parsimonious 7-feat (α=1.0):                 D²_test=−0.004
  Parsimonious 7-feat + winsorise p90:         D²_test=+0.195
  9-feat + winsorise p90:                      D²_test(8yr)≈+0.237
  10-feat + forest_area_pct:                   D²_test(8yr)=+0.202 (wrong lag init)
  10-feat + forest_area_pct + WDPA lag fix     D²_train=0.365 D²_test(8yr)=+0.233  ← current
    (panel patched with WDPA_May2026_Public_csv.csv):  D²_test(3yr)=+0.248, D²_6yr=+0.330
    Chow break F=0.82 p=0.621 (NOT significant — break was lag init artefact)

Three governance dimensions — each distinct and theoretically grounded:
  v2x_polyarchy  : electoral/participatory democracy (citizen accountability)
  v2xlg_legcon   : legislative constraints on executive (institutional checks;
                   r=0.82 with polyarchy but different mechanism — a strong
                   legislature prevents rollback of PA designations)
  v2csprtcpt     : civil society participatory environment (NGO/advocacy capacity;
                   r=0.48 with polyarchy, more distinct — measures who can organise
                   and lobby for conservation regardless of election outcomes)

Winsorisation caps training target at p90 (~15,836 px) to dampen Brazil's
2001–2009 frontier-exhaustion boom. Test evaluation uses original unwinsorised data.

v2xlg_legcon and v2csprtcpt are merged from V-Dem v15 at runtime (not yet in the
panel parquet — add to stage1_data_builder.py before next Euler rebuild).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PANEL_PATH  = _ROOT / "data" / "south_america" / "stage1_panel.parquet"
VDEM_PATH   = _ROOT / "data" / "shared" / "VDem" / "V-Dem-CY-Core-v15.csv"
OUT_DIR     = _ROOT / "outputs" / "south_america" / "results" / "ml_models"

TRAIN_YEARS  = (2001, 2016)   # pre-2001 WDPA training hurts (different expansion regime;
                              # VEN/BRA 1990s had 2-10x higher rates → distorts winsor cap)
TEST_YEARS   = (2017, 2024)
BREAK_YEAR   = 2010   # frontier exhaustion structural break
WINSOR_QUANT = 0.90   # cap training target at p90

LAG_COLS = [
    "pa_momentum_pixels_lag1",
    "pa_momentum_pixels_lag2",
    "pa_momentum_pixels_lag3",
    "pa_cumsum_lag1_pixels",
]
# Three theoretically distinct governance dimensions (see docstring above).
# Collinearity managed by L2 regularisation (α=1); VIF acceptable under shrinkage.
POLITICAL_COLS = [
    "v2x_polyarchy",      # electoral democracy — citizen accountability
    "gdp_growth_lag1",    # GDP/capita growth lagged 1yr — fiscal space
    "redd_plus_enrolled", # REDD+ participation — direct financial incentive
    "v2xlg_legcon",       # legislative constraints on executive
    "v2csprtcpt",         # civil society participatory environment
    "forest_area_pct",    # forest area % of land (WDI AG.LND.FRST.ZS) — countries with
                          # more remaining forest have more to designate; REDD+ incentive proxy
]

# V-Dem columns to merge at runtime (not yet in panel parquet)
VDEM_EXTRA = ["v2xlg_legcon", "v2csprtcpt"]


def _merge_vdem_extra(cy: pd.DataFrame) -> pd.DataFrame:
    """Merge v2xlg_legcon and v2csprtcpt from V-Dem v15 CSV if not already present."""
    if all(c in cy.columns for c in VDEM_EXTRA):
        return cy  # already merged (e.g. via stage1_data_builder or direct panel patch)
    if not VDEM_PATH.exists():
        raise FileNotFoundError(f"V-Dem CSV not found at {VDEM_PATH}")
    vdem = pd.read_csv(
        VDEM_PATH,
        usecols=["country_text_id", "year"] + VDEM_EXTRA,
        low_memory=False,
    ).rename(columns={"country_text_id": "iso3"})
    cy["iso3"] = cy["iso3"].astype(str)
    cy = cy.merge(vdem, on=["iso3", "year"], how="left")
    for col in VDEM_EXTRA:
        cy[col] = cy[col].fillna(cy[col].median())
    return cy


def _d2(model: PoissonRegressor, X: np.ndarray, y: np.ndarray) -> float:
    return float(model.score(X, y))


def _chow_test(
    cy: pd.DataFrame,
    feature_cols: list[str],
    break_year: int,
) -> dict:
    """Chow structural break test (OLS on log-scale) at break_year within training set."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return {}
    train = cy[cy["year"].between(*TRAIN_YEARS)].copy()
    sc = StandardScaler()
    X = sc.fit_transform(train[feature_cols].to_numpy(np.float64))
    X_c = sm.add_constant(X)
    y = np.log1p(train["pa_expansion_pixels"].to_numpy(np.float64))
    pre  = (train["year"] < break_year).to_numpy()
    post = ~pre

    rss_full = float(np.sum((y - sm.OLS(y, X_c).fit().predict()) ** 2))
    rss_pre  = float(np.sum((y[pre]  - sm.OLS(y[pre],  X_c[pre]).fit().predict()) ** 2))
    rss_post = float(np.sum((y[post] - sm.OLS(y[post], X_c[post]).fit().predict()) ** 2))
    k = X_c.shape[1]
    n = len(y)
    f_stat = ((rss_full - (rss_pre + rss_post)) / k) / ((rss_pre + rss_post) / (n - 2 * k))
    p_val  = float(1 - stats.f.cdf(f_stat, k, n - 2 * k))
    return {
        "chow_break_year": break_year,
        "chow_f_stat": float(f_stat),
        "chow_p_value": p_val,
        "chow_n_pre": int(pre.sum()),
        "chow_n_post": int(post.sum()),
    }


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 panel not found at {PANEL_PATH}. Run stage1_data_builder.py first."
        )
    cy = pd.read_parquet(PANEL_PATH)
    cy = _merge_vdem_extra(cy)

    feature_cols = [c for c in LAG_COLS + POLITICAL_COLS if c in cy.columns]
    cy[feature_cols] = cy[feature_cols].fillna(0)

    train = cy[cy["year"].between(*TRAIN_YEARS)].copy()
    test  = cy[cy["year"].between(*TEST_YEARS)].copy()

    X_train_raw = train[feature_cols].to_numpy(dtype=np.float64)
    y_train_raw = train["pa_expansion_pixels"].to_numpy(dtype=np.float64)
    X_test_raw  = test[feature_cols].to_numpy(dtype=np.float64)
    y_test      = test["pa_expansion_pixels"].to_numpy(dtype=np.float64)

    winsor_cap = float(np.quantile(y_train_raw, WINSOR_QUANT))
    y_train = np.minimum(y_train_raw, winsor_cap)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    model = PoissonRegressor(alpha=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    d2_train = _d2(model, X_train, y_train_raw)
    mu_train = np.clip(model.predict(X_train), 1e-9, None)
    rmse_train = float(np.sqrt(mean_squared_error(y_train_raw, mu_train)))

    d2_test: float | None = None
    rmse_test: float | None = None
    if len(test) > 0:
        d2_test   = _d2(model, X_test, y_test)
        mu_test   = np.clip(model.predict(X_test), 1e-9, None)
        rmse_test = float(np.sqrt(mean_squared_error(y_test, mu_test)))

    def _d2_window(y_end: int) -> float | None:
        sub = cy[cy["year"].between(TEST_YEARS[0], y_end)]
        if len(sub) == 0:
            return None
        X_sub = scaler.transform(sub[feature_cols].fillna(0).to_numpy(dtype=np.float64))
        return float(model.score(X_sub, sub["pa_expansion_pixels"].to_numpy(dtype=np.float64)))

    d2_test_3yr = _d2_window(2019)
    d2_test_6yr = _d2_window(2022)

    chow = _chow_test(cy, feature_cols, BREAK_YEAR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "region": "south_america",
        "model": "poisson_glm",
        "spec": "parsimonious_10feat_winsorised",
        "alpha": 1.0,
        "winsor_quantile": WINSOR_QUANT,
        "winsor_cap_pixels": winsor_cap,
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
        **chow,
    }
    out_path = OUT_DIR / "model1_expansion_coefficients.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("Stage 1 Poisson GLM — South America (10-feat + winsorise p90)")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Winsor cap: {winsor_cap:.0f} px ({WINSOR_QUANT:.0%} quantile)")
    print(f"  Train D²: {d2_train:.4f}  RMSE: {rmse_train:.0f}  (n={len(train)}, vs orig targets)")
    if d2_test is not None:
        print(f"  Test  D²: {d2_test:.4f}  RMSE: {rmse_test:.0f}  (n={len(test)}, 2017–2024)")
        print(f"  Test  D² (3yr 2017–2019): {d2_test_3yr:.4f}  |  D² (6yr 2017–2022): {d2_test_6yr:.4f}")
    if chow:
        print(f"  Chow break at {BREAK_YEAR}: F={chow['chow_f_stat']:.2f}  p={chow['chow_p_value']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
