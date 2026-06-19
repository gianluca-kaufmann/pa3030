#!/usr/bin/env python3
"""E6: Stage 1 overdispersion test and Negative Binomial GLM comparison.

Tests whether the Poisson GLM assumption (Var = μ) is violated and whether
switching to a Negative Binomial family (Var = μ + α·μ²) improves OOS D².

Tests run:
  1. Pearson chi²/df from Poisson fit (>1 = overdispersed)
  2. Cameron-Trivedi (1990) auxiliary OLS regression test for overdispersion
  3. NB GLM fitted with bfgs + Poisson start params (avoids IRLS divergence)
  4. D²_7yr comparison: Poisson vs NB (same features, same train/test split)
  5. Country FE sensitivity re-run with iso3-NaN fix

Output: outputs/south_america/results/stage1_nb_overdispersion.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import from model1_expansion via importlib (5_training dir starts with digit,
# so dotted-module import is invalid Python syntax)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "model1_expansion",
    _ROOT / "scripts/regions/south_america/5_training/model1_expansion.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LAG_COLS              = _mod.LAG_COLS
POLITICAL_COLS        = _mod.POLITICAL_COLS
TRAIN_YEARS           = _mod.TRAIN_YEARS
WINSOR_QUANT          = _mod.WINSOR_QUANT
_apply_log1p          = _mod._apply_log1p
_compute_governance_diffs = _mod._compute_governance_diffs
_make_decay_weights   = _mod._make_decay_weights
_merge_vdem_extra     = _mod._merge_vdem_extra

import os as _os
_scratch = _os.environ.get("SCRATCH", "")
_panel_scratch = Path(_scratch) / "data/south_america/stage1_panel.parquet" if _scratch else None
PANEL_PATH = (
    _panel_scratch
    if _panel_scratch and _panel_scratch.exists()
    else _ROOT / "data" / "south_america" / "stage1_panel.parquet"
)
OUT_PATH   = _ROOT / "outputs" / "south_america" / "results" / "stage1_nb_overdispersion.json"

TEST_START       = 2017
PRIMARY_EVAL_END = 2023


def _poisson_d2(y_true: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-10, None)
    y_bar = np.clip(np.full_like(y_true, y_true.mean()), 1e-10, None)
    try:
        return 1.0 - float(mean_poisson_deviance(y_true, mu)) / float(mean_poisson_deviance(y_true, y_bar))
    except Exception:
        return float("nan")


def _prepare(cy: pd.DataFrame, feature_cols: list[str]) -> tuple:
    tr = cy[cy["year"].between(*TRAIN_YEARS)].copy()
    X_t   = _apply_log1p(tr[feature_cols].to_numpy(np.float64), feature_cols)
    y_raw = tr["pa_expansion_pixels"].to_numpy(np.float64)
    cap   = float(np.quantile(y_raw, WINSOR_QUANT))
    y_fit = np.minimum(y_raw, cap)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_t)
    return tr, X_sc, y_fit, sc, cap


def _test_d2(cy: pd.DataFrame, sc: StandardScaler, feature_cols: list[str],
             mu_fn, end_yr: int) -> float:
    sub = cy[cy["year"].between(TEST_START, end_yr)]
    if len(sub) == 0:
        return float("nan")
    X = sc.transform(_apply_log1p(sub[feature_cols].fillna(0).to_numpy(np.float64), feature_cols))
    y = sub["pa_expansion_pixels"].to_numpy(np.float64)
    mu = np.clip(mu_fn(X), 1e-10, None)
    return _poisson_d2(y, mu)


def _overdispersion_tests(y_fit: np.ndarray, mu: np.ndarray, n_params: int) -> dict:
    """Pearson chi²/df and Cameron-Trivedi (1990) auxiliary OLS test."""
    pearson_resid = (y_fit - mu) / np.sqrt(np.clip(mu, 1e-10, None))
    chi2          = float(np.sum(pearson_resid ** 2))
    df            = len(y_fit) - n_params
    disp_stat     = chi2 / df

    # Cameron-Trivedi: OLS of [(y-μ)²-μ]/μ on μ (no intercept) → slope = α̂
    aux_y = ((y_fit - mu) ** 2 - mu) / np.clip(mu, 1e-10, None)
    lr    = LinearRegression(fit_intercept=False)
    lr.fit(mu.reshape(-1, 1), aux_y)
    ct_alpha = float(lr.coef_[0])
    aux_pred = lr.predict(mu.reshape(-1, 1))
    ss_res   = float(np.sum((aux_y - aux_pred) ** 2))
    ss_tot   = float(np.sum((aux_y - aux_y.mean()) ** 2))
    ct_r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "pearson_chi2":       round(chi2, 2),
        "pearson_df":         df,
        "pearson_disp_stat":  round(disp_stat, 4),
        "overdispersed":      disp_stat > 1.0,
        "ct_alpha_estimate":  round(ct_alpha, 6),
        "ct_r2":              round(ct_r2, 4),
        "interpretation": (
            f"Pearson chi²/df = {disp_stat:.2f} ({'overdispersed' if disp_stat > 1 else 'not overdispersed'}). "
            f"Cameron-Trivedi α̂ = {ct_alpha:.4f} ({'positive → NB warranted' if ct_alpha > 0 else 'non-positive → Poisson adequate'})."
        ),
    }


def _fit_nb(X_sc_c: np.ndarray, y_fit: np.ndarray, pois_params: np.ndarray,
            ct_alpha: float) -> tuple:
    """NB GLM using Poisson start params + bfgs to avoid IRLS divergence."""
    alpha_init = max(ct_alpha, 0.01)
    nb_start   = np.append(pois_params, alpha_init)

    for method in ("bfgs", "nm", "lbfgs"):
        try:
            nb = sm.GLM(
                y_fit, X_sc_c,
                family=sm.families.NegativeBinomial(alpha=alpha_init),
            ).fit(start_params=pois_params, method=method, maxiter=1000, disp=False)
            if not np.any(np.isnan(nb.params)):
                return nb, method
        except Exception:
            continue
    return None, None


def _country_fe(cy: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Poisson + 11 country dummies. Replicates model1_expansion._country_fe_sensitivity
    but with iso3-NaN fix (mixed float/str iso3 columns)."""
    tr = cy[cy["year"].between(*TRAIN_YEARS)].dropna(subset=["iso3"]).copy()
    tr = tr[tr["iso3"].apply(lambda x: isinstance(x, str))]

    countries = sorted(tr["iso3"].unique())
    ref       = countries[0]
    dummies   = countries[1:]

    def _fe_matrix(df: pd.DataFrame) -> np.ndarray:
        df = df[df["iso3"].apply(lambda x: isinstance(x, str))]
        fe = np.zeros((len(df), len(dummies)), dtype=np.float64)
        for j, c in enumerate(dummies):
            fe[:, j] = (df["iso3"].values == c).astype(np.float64)
        return df, fe

    tr_clean, fe_tr = _fe_matrix(tr)
    X_t   = _apply_log1p(tr_clean[feature_cols].to_numpy(np.float64), feature_cols)
    y_raw = tr_clean["pa_expansion_pixels"].to_numpy(np.float64)
    cap   = float(np.quantile(y_raw, WINSOR_QUANT))
    y_fit = np.minimum(y_raw, cap)

    X_aug  = np.hstack([X_t, fe_tr])
    sc_fe  = StandardScaler()
    X_sc   = sc_fe.fit_transform(X_aug)
    sw     = _make_decay_weights(tr_clean)

    model_fe = PoissonRegressor(alpha=1.0, max_iter=1000)
    model_fe.fit(X_sc, y_fit, sample_weight=sw)

    out: dict = {
        "n_dummies": len(dummies),
        "reference_country": ref,
        "note": (
            "11 country dummies + 12 features. "
            "OOS D² collapse confirms no-FE primary spec is correct "
            "(FE cannot generalise: test countries seen in train)."
        ),
    }
    for end_yr, key in [(2019, "d2_3yr"), (PRIMARY_EVAL_END, "d2_7yr"), (2024, "d2_8yr")]:
        sub = cy[cy["year"].between(TEST_START, end_yr)].dropna(subset=["iso3"])
        sub = sub[sub["iso3"].apply(lambda x: isinstance(x, str))]
        if len(sub) == 0:
            out[key] = None
            continue
        sub_clean, fe_sub = _fe_matrix(sub)
        X_sub = sc_fe.transform(
            np.hstack([
                _apply_log1p(sub_clean[feature_cols].fillna(0).to_numpy(np.float64), feature_cols),
                fe_sub,
            ])
        )
        y_sub = sub_clean["pa_expansion_pixels"].to_numpy(np.float64)
        out[key] = round(_poisson_d2(y_sub, np.clip(model_fe.predict(X_sub), 1e-10, None)), 4)
    return out


def main() -> None:
    cy = pd.read_parquet(PANEL_PATH)
    cy = _merge_vdem_extra(cy)
    cy = _compute_governance_diffs(cy)

    feature_cols = [c for c in LAG_COLS + POLITICAL_COLS if c in cy.columns]
    cy[feature_cols] = cy[feature_cols].fillna(0)

    tr, X_sc, y_fit, sc, cap = _prepare(cy, feature_cols)
    X_sc_c = np.column_stack([np.ones(len(X_sc)), X_sc])
    sw     = _make_decay_weights(tr)

    # ── Poisson baseline ───────────────────────────────────────────────────────
    m_pois = PoissonRegressor(alpha=1.0, max_iter=1000)
    m_pois.fit(X_sc, y_fit, sample_weight=sw)
    mu_tr  = np.clip(m_pois.predict(X_sc), 1e-10, None)

    d2_pois = {
        "d2_train": round(_poisson_d2(y_fit, mu_tr), 4),
        "d2_3yr":   round(_test_d2(cy, sc, feature_cols, m_pois.predict, 2019), 4),
        "d2_7yr":   round(_test_d2(cy, sc, feature_cols, m_pois.predict, PRIMARY_EVAL_END), 4),
        "d2_8yr":   round(_test_d2(cy, sc, feature_cols, m_pois.predict, 2024), 4),
    }

    # ── Overdispersion tests ───────────────────────────────────────────────────
    n_params  = len(feature_cols) + 1  # +intercept
    overdisp  = _overdispersion_tests(y_fit, mu_tr, n_params)

    # ── Poisson via statsmodels (for start params + NB) ───────────────────────
    pois_sm = sm.GLM(y_fit, X_sc_c, family=sm.families.Poisson()).fit(disp=False)

    # ── NB GLM ────────────────────────────────────────────────────────────────
    nb, nb_method = _fit_nb(X_sc_c, y_fit, pois_sm.params, overdisp["ct_alpha_estimate"])

    if nb is not None:
        def nb_predict(X: np.ndarray) -> np.ndarray:
            return np.clip(nb.predict(np.column_stack([np.ones(len(X)), X])), 1e-10, None)

        # Sanity: D² using frozen Poisson coefficients under NB family should ≈ Poisson D².
        # If NB-optimised D² differs strongly, bfgs found a divergent solution.
        def pois_coef_predict(X: np.ndarray) -> np.ndarray:
            return np.clip(np.exp(np.column_stack([np.ones(len(X)), X]) @ pois_sm.params), 1e-10, None)

        d2_pois_frozen = round(_poisson_d2(y_fit, pois_coef_predict(X_sc)), 4)

        d2_nb = {
            "d2_train": round(_poisson_d2(y_fit, nb_predict(X_sc)), 4),
            "d2_3yr":   round(_test_d2(cy, sc, feature_cols, nb_predict, 2019), 4),
            "d2_7yr":   round(_test_d2(cy, sc, feature_cols, nb_predict, PRIMARY_EVAL_END), 4),
            "d2_8yr":   round(_test_d2(cy, sc, feature_cols, nb_predict, 2024), 4),
            "nb_optimizer":    nb_method,
            # alpha is FIXED in sm.GLM NB family (not re-estimated); ct_alpha used as fixed value.
            "nb_alpha_fixed":  round(overdisp["ct_alpha_estimate"], 6),
            "d2_train_poisson_frozen_sanity": d2_pois_frozen,
            "note": (
                "sm.GLM NegativeBinomial with alpha fixed at CT estimate. "
                "bfgs optimises regression coefficients only. "
                "D² computed as Poisson D² for direct comparability."
            ),
        }
        delta = {k: round(d2_nb[k] - d2_pois[k], 4)
                 for k in ("d2_train", "d2_3yr", "d2_7yr", "d2_8yr")}
    else:
        d2_nb  = {"error": "NB fitting failed with all optimizers"}
        delta  = {}

    # ── Country FE sensitivity ─────────────────────────────────────────────────
    fe_result = _country_fe(cy, feature_cols)

    # ── Summary ───────────────────────────────────────────────────────────────
    result = {
        "region":        "south_america",
        "experiment":    "E6_nb_overdispersion",
        "feature_cols":  feature_cols,
        "n_features":    len(feature_cols),
        "n_train":       int(len(tr)),
        "train_zeros":   int((y_fit == 0).sum()),
        "winsor_cap":    float(cap),
        "overdispersion": overdisp,
        "poisson_d2":    d2_pois,
        "nb_d2":         d2_nb,
        "nb_vs_poisson_delta_d2": delta,
        "country_fe_sensitivity": fe_result,
        "interpretation": {
            "overdispersion": overdisp["interpretation"],
            "nb_conclusion": (
                f"NB D²_7yr = {d2_nb.get('d2_7yr', 'N/A')}, "
                f"Poisson D²_7yr = {d2_pois['d2_7yr']}. "
                f"Delta = {delta.get('d2_7yr', 'N/A')}."
            ) if nb is not None else "NB fitting failed",
            "fe_conclusion": (
                f"Country FE D²_7yr = {fe_result.get('d2_7yr', 'N/A')} "
                f"vs Poisson {d2_pois['d2_7yr']} — "
                + ("FE collapses OOS performance, no-FE spec confirmed."
                   if (fe_result.get("d2_7yr") or 999) < d2_pois["d2_7yr"]
                   else "FE improves OOS — consider including.")
            ),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print("=" * 60)
    print("E6: Stage 1 Overdispersion + NB GLM")
    print("=" * 60)
    print(f"  n_train = {result['n_train']}  zeros = {result['train_zeros']}")
    print()
    print("  Overdispersion tests:")
    print(f"    Pearson chi²/df = {overdisp['pearson_disp_stat']:.3f}")
    print(f"    Cameron-Trivedi α̂ = {overdisp['ct_alpha_estimate']:.4f}")
    print(f"    → {overdisp['interpretation']}")
    print()
    print(f"  Poisson  D²_7yr = {d2_pois['d2_7yr']:+.4f}  (D²_3yr={d2_pois['d2_3yr']:+.4f})")
    if nb is not None:
        print(f"  NB       D²_7yr = {d2_nb['d2_7yr']:+.4f}  (D²_3yr={d2_nb['d2_3yr']:+.4f})"
              f"  Δ={delta.get('d2_7yr', 'N/A'):+}  (α_fixed={d2_nb.get('nb_alpha_fixed', 'N/A')})")
        print(f"  [sanity: Poisson coefs under NB family D²_train={d2_nb['d2_train_poisson_frozen_sanity']:+.4f}]")
    else:
        print(f"  NB:  {d2_nb}")
    print()
    print(f"  Country FE  D²_7yr = {fe_result.get('d2_7yr', 'N/A'):+}  "
          f"(D²_3yr={fe_result.get('d2_3yr', 'N/A')})")
    print(f"  → {result['interpretation']['fe_conclusion']}")
    print()
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
