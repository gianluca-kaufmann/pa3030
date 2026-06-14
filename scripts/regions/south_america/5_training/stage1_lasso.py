#!/usr/bin/env python3
"""Issue CC: LASSO feature selection for Stage 1 Poisson GLM (South America).

Fits a Poisson LASSO path over a log-spaced alpha grid, selects alpha via
leave-one-year-out (LOYO) CV on the training set (2001-2016), and reports
which features survive. Then re-fits the primary Poisson GLM (sklearn, L2
alpha=1) on the LASSO-selected subset and compares D2_7yr to the 12-feat spec.

Saves to outputs/south_america/results/stage1_lasso.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PANEL_PATH = _ROOT / "data" / "south_america" / "stage1_panel.parquet"
VDEM_PATH  = _ROOT / "data" / "shared" / "VDem" / "V-Dem-CY-Core-v15.csv"
OUT_PATH   = _ROOT / "outputs" / "south_america" / "results" / "stage1_lasso.json"

TRAIN_YEARS      = (2001, 2016)
TEST_YEARS_START = 2017
PRIMARY_EVAL_END = 2023
WINSOR_QUANT     = 0.95
DECAY_BREAK_YEAR = 2010
DECAY_WEIGHT_PRE = 0.6

LAG_COLS = [
    "pa_momentum_pixels_lag1",
    "pa_momentum_pixels_lag2",
    "pa_momentum_pixels_lag3",
    "pa_cumsum_lag1_pixels",
]
LAG_MASK = set(LAG_COLS)

# All candidate features (12 primary + CBD as extra candidate)
CANDIDATE_COLS = [
    "pa_momentum_pixels_lag1",
    "pa_momentum_pixels_lag2",
    "pa_momentum_pixels_lag3",
    "pa_cumsum_lag1_pixels",
    "v2x_polyarchy",
    "gdp_growth_lag1",
    "redd_plus_enrolled",
    "d_v2xlg_legcon",
    "d_v2csprtcpt",
    "agricultural_land_pct",
    "d_v2xlg_legcon_lag1",
    "legcon_x_cspart",
    "cbd_meeting_year",
]

PRIMARY_12 = [
    "pa_momentum_pixels_lag1",
    "pa_momentum_pixels_lag2",
    "pa_momentum_pixels_lag3",
    "pa_cumsum_lag1_pixels",
    "v2x_polyarchy",
    "gdp_growth_lag1",
    "redd_plus_enrolled",
    "d_v2xlg_legcon",
    "d_v2csprtcpt",
    "agricultural_land_pct",
    "d_v2xlg_legcon_lag1",
    "legcon_x_cspart",
]

VDEM_EXTRA = ["v2xlg_legcon", "v2csprtcpt"]
VDEM_LEVEL = ["v2x_polyarchy"]

ALPHA_GRID         = np.logspace(-3, 2, 40)
COEF_ZERO_THRESH   = 1e-4


# ── Preprocessing (mirrors model1_expansion.py) ────────────────────────────

def _apply_log1p(X: np.ndarray, cols: list[str]) -> np.ndarray:
    Xt = X.copy()
    for i, col in enumerate(cols):
        if col in LAG_MASK:
            Xt[:, i] = np.log1p(X[:, i])
    return Xt


def _merge_vdem_extra(cy: pd.DataFrame) -> pd.DataFrame:
    needed = [c for c in VDEM_EXTRA + VDEM_LEVEL if c not in cy.columns]
    if not needed:
        return cy
    if not VDEM_PATH.exists():
        raise FileNotFoundError(f"V-Dem CSV not found: {VDEM_PATH}")
    vdem = pd.read_csv(
        VDEM_PATH,
        usecols=["country_text_id", "year"] + needed,
        low_memory=False,
    ).rename(columns={"country_text_id": "iso3"})
    cy["iso3"] = cy["iso3"].astype(str)
    cy = cy.merge(vdem, on=["iso3", "year"], how="left")
    for col in needed:
        cy[col] = cy[col].fillna(cy[col].median())
    return cy


def _compute_governance_diffs(cy: pd.DataFrame) -> pd.DataFrame:
    cy = cy.sort_values(["iso3", "year"]).copy()
    for col in VDEM_EXTRA:
        if col in cy.columns:
            cy[f"d_{col}"] = cy.groupby("iso3")[col].diff()
    if "d_v2xlg_legcon" in cy.columns:
        cy["d_v2xlg_legcon_lag1"] = cy.groupby("iso3")["d_v2xlg_legcon"].shift(1)
    d_leg = cy.get("d_v2xlg_legcon", pd.Series(0.0, index=cy.index)).fillna(0)
    d_csp = cy.get("d_v2csprtcpt",   pd.Series(0.0, index=cy.index)).fillna(0)
    cy["legcon_x_cspart"] = d_leg * d_csp
    return cy


def _decay_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df))
    w[df["year"].values < DECAY_BREAK_YEAR] = DECAY_WEIGHT_PRE
    return w


# ── Core fits ──────────────────────────────────────────────────────────────

def _prepare_train(cy: pd.DataFrame, cols: list[str]) -> tuple:
    """Returns X_sc, y_win, cap, scaler (on full training set)."""
    tr = cy[cy["year"].between(*TRAIN_YEARS)].copy()
    X  = _apply_log1p(tr[cols].fillna(0).to_numpy(np.float64), cols)
    y  = tr["pa_expansion_pixels"].to_numpy(np.float64)
    cap = float(np.quantile(y, WINSOR_QUANT))
    y_w = np.minimum(y, cap)
    sc = StandardScaler()
    return sc.fit_transform(X), y_w, cap, sc, tr


def _poisson_lasso(X_sc: np.ndarray, y: np.ndarray, alpha: float,
                   var_weights: np.ndarray | None = None) -> np.ndarray | None:
    """Statsmodels Poisson GLM with L1 penalty. Returns param vector (intercept first)."""
    X_c = sm.add_constant(X_sc, has_constant="add")
    kw = {}
    if var_weights is not None:
        kw["var_weights"] = var_weights
    try:
        res = sm.GLM(y, X_c, family=sm.families.Poisson(), **kw).fit_regularized(
            method="elastic_net", alpha=alpha, L1_wt=1.0, maxiter=500,
        )
        return np.array(res.params)   # [intercept, coef1, ..., coefK]
    except Exception:
        return None


def _sklearn_glm(cy: pd.DataFrame, cols: list[str]) -> tuple:
    """Primary sklearn Poisson GLM (L2 alpha=1) — same as model1_expansion.py."""
    X_sc, y_w, _, sc, tr = _prepare_train(cy, cols)
    sw = _decay_weights(tr)
    m  = PoissonRegressor(alpha=1.0, max_iter=1000)
    m.fit(X_sc, y_w, sample_weight=sw)
    return m, sc


def _d2(cy: pd.DataFrame, model: PoissonRegressor, sc: StandardScaler,
        cols: list[str], y_end: int) -> float | None:
    sub = cy[cy["year"].between(TEST_YEARS_START, y_end)]
    if len(sub) == 0:
        return None
    X  = sc.transform(_apply_log1p(sub[cols].fillna(0).to_numpy(np.float64), cols))
    y  = sub["pa_expansion_pixels"].to_numpy(np.float64)
    return float(model.score(X, y))


def _poisson_dev(y: np.ndarray, mu: np.ndarray) -> float:
    try:
        return float(mean_poisson_deviance(y, np.clip(mu, 1e-10, None)))
    except Exception:
        return float("inf")


# ── LOYO-CV ───────────────────────────────────────────────────────────────

def _loyo_cv(cy: pd.DataFrame, cols: list[str]) -> dict:
    train_years = list(range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1))
    cv_devs: dict[float, list[float]] = {float(a): [] for a in ALPHA_GRID}

    for left_out in train_years:
        fit_df = cy[cy["year"].between(*TRAIN_YEARS) & (cy["year"] != left_out)].copy()
        val_df = cy[cy["year"] == left_out].copy()
        if len(val_df) == 0:
            continue

        X_fit = _apply_log1p(fit_df[cols].fillna(0).to_numpy(np.float64), cols)
        y_fit = fit_df["pa_expansion_pixels"].to_numpy(np.float64)
        cap   = float(np.quantile(y_fit, WINSOR_QUANT))
        y_fit = np.minimum(y_fit, cap)

        sc = StandardScaler()
        X_sc = sc.fit_transform(X_fit)
        sw   = _decay_weights(fit_df)

        X_val = sc.transform(_apply_log1p(val_df[cols].fillna(0).to_numpy(np.float64), cols))
        y_val = val_df["pa_expansion_pixels"].to_numpy(np.float64)

        for alpha in ALPHA_GRID:
            params = _poisson_lasso(X_sc, y_fit, float(alpha), var_weights=sw)
            if params is None:
                cv_devs[float(alpha)].append(float("inf"))
                continue
            intercept, coefs = params[0], params[1:]
            log_mu = intercept + X_val @ coefs
            mu     = np.exp(np.clip(log_mu, -20, 20))
            cv_devs[float(alpha)].append(_poisson_dev(y_val, mu))

    mean_devs = {
        a: float(np.mean(v)) if v else float("inf")
        for a, v in cv_devs.items()
    }
    best_alpha = float(min(mean_devs, key=mean_devs.get))
    return {
        "alpha_grid": [round(float(a), 6) for a in ALPHA_GRID],
        "mean_cv_deviance": {f"{a:.6f}": round(v, 6) for a, v in mean_devs.items()},
        "best_alpha": round(best_alpha, 6),
        "best_cv_deviance": round(mean_devs[best_alpha], 6),
        "n_loyo_folds": len(train_years),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    cy = pd.read_parquet(PANEL_PATH)
    cy = _merge_vdem_extra(cy)
    cy = _compute_governance_diffs(cy)

    cols = [c for c in CANDIDATE_COLS if c in cy.columns]
    cy[cols] = cy[cols].fillna(0)
    print(f"Candidates ({len(cols)}): {cols}")
    print(f"LOYO-CV: {len(ALPHA_GRID)} alphas × {TRAIN_YEARS[1]-TRAIN_YEARS[0]+1} folds ...")

    cv = _loyo_cv(cy, cols)
    best_alpha = cv["best_alpha"]
    print(f"  Best alpha: {best_alpha:.4f}  CV deviance: {cv['best_cv_deviance']:.4f}")

    # LASSO on full training set at best alpha
    X_sc_full, y_w_full, _, _, tr_full = _prepare_train(cy, cols)
    sw_full = _decay_weights(tr_full)
    params_full = _poisson_lasso(X_sc_full, y_w_full, best_alpha, var_weights=sw_full)
    if params_full is None:
        raise RuntimeError("LASSO fit on full training set failed.")

    coef_dict = {col: float(c) for col, c in zip(cols, params_full[1:])}
    selected  = [c for c, v in coef_dict.items() if abs(v) > COEF_ZERO_THRESH]
    zeroed    = [c for c, v in coef_dict.items() if abs(v) <= COEF_ZERO_THRESH]

    print(f"\nLASSO (alpha={best_alpha:.4f}):")
    print(f"  Surviving ({len(selected)}): {selected}")
    print(f"  Zeroed    ({len(zeroed)}):   {zeroed}")

    # Baseline: primary 12-feat spec
    primary_12 = [c for c in PRIMARY_12 if c in cy.columns]
    m12, sc12 = _sklearn_glm(cy, primary_12)
    d2_12_7yr = _d2(cy, m12, sc12, primary_12, PRIMARY_EVAL_END)
    d2_12_3yr = _d2(cy, m12, sc12, primary_12, 2019)

    out: dict = {
        "candidate_features": cols,
        "n_candidates": len(cols),
        "loyo_cv": cv,
        "selected_alpha": best_alpha,
        "coef_zero_threshold": COEF_ZERO_THRESH,
        "lasso_coefs": {k: round(v, 6) for k, v in coef_dict.items()},
        "selected_features": selected,
        "zeroed_features": zeroed,
        "n_selected": len(selected),
        "primary_12_features": primary_12,
        "d2_7yr_12feat_PRIMARY": round(d2_12_7yr, 4) if d2_12_7yr is not None else None,
        "d2_3yr_12feat": round(d2_12_3yr, 4) if d2_12_3yr is not None else None,
    }

    # Refit sklearn GLM on LASSO-selected features if different from 12-feat
    if selected and set(selected) != set(primary_12):
        m_sel, sc_sel = _sklearn_glm(cy, selected)
        d2_sel_7yr = _d2(cy, m_sel, sc_sel, selected, PRIMARY_EVAL_END)
        d2_sel_3yr = _d2(cy, m_sel, sc_sel, selected, 2019)
        out["d2_7yr_lasso_selected"] = round(d2_sel_7yr, 4) if d2_sel_7yr is not None else None
        out["d2_3yr_lasso_selected"] = round(d2_sel_3yr, 4) if d2_sel_3yr is not None else None
        print(f"\nRefit on LASSO-selected ({len(selected)} features):")
        print(f"  D2_7yr = {d2_sel_7yr:.4f}  (12-feat: {d2_12_7yr:.4f})")
        print(f"  D2_3yr = {d2_sel_3yr:.4f}  (12-feat: {d2_12_3yr:.4f})")
    else:
        out["d2_7yr_lasso_selected"] = out["d2_7yr_12feat_PRIMARY"]
        out["d2_3yr_lasso_selected"] = out["d2_3yr_12feat"]
        print("\nLASSO-selected matches primary 12-feat spec.")

    print(f"\n12-feat baseline: D2_7yr = {d2_12_7yr:.4f}  D2_3yr = {d2_12_3yr:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
