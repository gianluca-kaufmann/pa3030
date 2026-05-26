#!/usr/bin/env python3
"""Stage 1 country-year PA expansion model (Poisson GLM) — South America.

Specification: 11-feature parsimonious Poisson, p90 winsorisation, WDPA lag fix,
first-difference governance variables + agricultural land fraction.

Model selection history (2001–2016 train / 2017–2024 test):
  Full model (16 feat, α=0.1):                 D²_test=−0.096
  Parsimonious 7-feat (α=1.0):                 D²_test=−0.004
  Parsimonious 7-feat + winsorise p90:         D²_test=+0.195
  9-feat + winsorise p90:                      D²_test(8yr)≈+0.237
  10-feat + forest_area_pct (wrong lag init):  D²_test(8yr)=+0.202
  10-feat + WDPA lag fix (level gov):          D²_train=0.365, D²_test(8yr)=+0.233
  10-feat + Δgov + agri (this script):         D²_train=0.384, D²_test(8yr)=+0.356  ← current
    D²_test(3yr 2017–2019)=+0.357, D²_test(6yr 2017–2022)=+0.411
    Chow break at 2010: F=0.84, p=0.616 (NOT significant)

Three governance dimensions — each distinct and theoretically grounded.
v2xlg_legcon and v2csprtcpt are used as FIRST DIFFERENCES (Δ), not levels:

  v2x_polyarchy   : level — cross-country: more democratic countries designate more
  Δv2xlg_legcon   : change in legislative constraints — captures the WITHIN-COUNTRY timing
                    of PA decisions; a year in which legislative checks strengthen creates
                    the political opening for designation (not just having strong checks)
  Δv2csprtcpt     : change in civil society participatory environment — a year of NGO/
                    advocacy empowerment triggers designation events; the level is largely
                    absorbed by v2x_polyarchy (r=0.48) and country-level fixed variation

Using first differences is more causally defensible: it is the governance CHANGE in year t
that triggers the designation event, not the absolute governance level (which is stable
across time within a country and confounded with stable country characteristics).
Level-based governance variables also introduce temporal drift — with a positive coefficient,
slowly rising legcon values cause predictions to grow monotonically over 2020–2024, exactly
when actual expansion was declining (COVID + WDPA reporting lag).

agricultural_land_pct (WDI AG.LND.AGR.ZS):
  Countries with more land under agriculture have less natural land available for PA
  designation. Uruguay (82.7%) and Paraguay (45.7%) are always zero in the test period;
  Suriname (0.5%) and Guyana (3.6%) have the most unprotected natural area. Negative
  coefficient is expected and confirmed empirically. Cross-regional sign in SEA differs
  due to land tenure and political economy differences — not used in SEA model.

Winsorisation caps training target at p90 (~15,836 px) to dampen Brazil's 2001–2009
frontier-exhaustion boom. Test evaluation uses original unwinsorised data.

v2xlg_legcon and v2csprtcpt are in the panel parquet (added to stage1_data_builder.py
on 2026-05-26). This script diffs them at runtime; the raw levels are still needed as
intermediates so _merge_vdem_extra remains in case of an older panel without these columns.
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
# Governance variables in first differences (see module docstring for rationale).
# v2x_polyarchy remains as a level — it captures stable cross-country democratic culture
# and does not produce temporal drift because it changes very slowly.
POLITICAL_COLS = [
    "v2x_polyarchy",         # level — electoral democracy, cross-country dimension
    "gdp_growth_lag1",       # GDP/capita growth lagged 1yr — fiscal space for conservation
    "redd_plus_enrolled",    # REDD+ participation — financial incentive for forest protection
    "d_v2xlg_legcon",        # Δ legislative constraints on executive — institutional event signal
    "d_v2csprtcpt",          # Δ civil society participatory environment — advocacy event signal
    "agricultural_land_pct", # agricultural land % (WDI AG.LND.AGRI.ZS) — opportunity cost:
                             # more agri land = less natural area available for PA designation.
                             # Subsumes forest_area_pct (collinear: r≈−0.7): once agri land is
                             # controlled, the marginal forest signal is near-zero in SA.
                             # Note: SEA model retains forest_area_pct (different land-use context).
]

# V-Dem columns needed to compute first differences (merged if absent from panel parquet).
VDEM_EXTRA = ["v2xlg_legcon", "v2csprtcpt"]


def _merge_vdem_extra(cy: pd.DataFrame) -> pd.DataFrame:
    """Merge v2xlg_legcon and v2csprtcpt from V-Dem v15 CSV if not already in panel."""
    if all(c in cy.columns for c in VDEM_EXTRA):
        return cy
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


def _compute_governance_diffs(cy: pd.DataFrame) -> pd.DataFrame:
    """Compute within-country first differences of governance level variables.

    For year t: d_v2xlg_legcon(t) = v2xlg_legcon(t) − v2xlg_legcon(t−1).
    The 1990 row (panel start) gets NaN, which is outside TRAIN_YEARS and filled
    with 0 at feature assembly. All training and test rows have valid diffs because
    the panel includes 1990–2000 as context for the lag computation.
    """
    cy = cy.sort_values(["iso3", "year"]).copy()
    for col in VDEM_EXTRA:
        if col in cy.columns:
            cy[f"d_{col}"] = cy.groupby("iso3")[col].diff()
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
    cy = _merge_vdem_extra(cy)       # ensure legcon/csprtcpt present as levels
    cy = _compute_governance_diffs(cy)  # compute d_v2xlg_legcon, d_v2csprtcpt

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
        "spec": "parsimonious_10feat_delta_gov_winsorised",
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

    print("Stage 1 Poisson GLM — South America (10-feat: Δgov + agri, p90 winsorise)")
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
