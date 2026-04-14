#!/usr/bin/env python3
"""Regenerate SHAP summary plots for all three regions with renamed feature labels.

Usage:
    python scripts/regen_shap_summary.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gc
import pickle
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── Feature rename mapping ────────────────────────────────────────────────────
SHAP_FEATURE_RENAME = {
    "GSN_b2_smooth16":         "Biodiversity Priority (16 km)",
    "GSN_b2_smooth64":         "Biodiversity Priority (64 km)",
    "GSN_b2":                  "Biodiversity Priority",
    "GSN_b3_smooth16":         "Intact Wilderness (16 km)",
    "GSN_b3_smooth64":         "Intact Wilderness (64 km)",
    "GSN_b3":                  "Intact Wilderness",
    "policy_b4":               "Rule of Law (WGI)",
    "policy_b2":               "Liberal Democracy (V-Dem)",
    "policy_b3":               "Govt Effectiveness (WGI)",
    "WorldClim_b7":            "Temp. Annual Range",
    "WorldClim_b12":           "Annual Precipitation (mm)",
    "WorldClim_b13":           "Precip. Wettest Month (mm)",
    "WorldClim_b14":           "Precip. Driest Month (mm)",
    "WorldClim_b17":           "Precipitation Seasonality",
    "dist_indigenous":         "Dist. to Indigenous Land (km)",
    "dist_wdpa":               "Dist. to Existing PA (km)",
    "dist_road":               "Dist. to Road (km)",
    "dist_oil_gas":            "Dist. to Oil/Gas Infra. (km)",
    "dist_powerplant":         "Dist. to Power Plant (km)",
    "WDPA":                    "Existing PA Coverage",
    "WDPA_prev":               "Existing PA Coverage",
    "economic_value_lag1":     "Agric. Land Value (lag 1yr)",
    "elevation_b2_smooth16":   "Elevation (16 km smooth)",
    "elevation_b2_smooth64":   "Elevation (64 km smooth)",
    "elevation_b2":            "Elevation (m)",
    "HNTL_smooth64":           "Night-time Lights (64 km)",
    "HNTL_b1":                 "Night-time Lights",
    "HNTL_b1_smooth64":        "Night-time Lights (64 km)",
    "GPW":                     "Population Density",
    "GPW_b1":                  "Population Density",
    "landcover_b1":            "Land Cover (MODIS)",
    "deforestation_b1":        "Annual Deforestation",
    "wildfire_b1":             "Wildfire Occurrence",
}

REGIONS = [
    {
        "name":       "south_america",
        "model_pkl":  REPO_ROOT / "data/south_america/ml/models/model1_lgbm_deployment_20260330_015217.pkl",
        "test_parq":  REPO_ROOT / "data/south_america/ml/main/test_win5.parquet",
        "out_dir":    REPO_ROOT / "outputs/south_america/results/model1_lgbm",
    },
]

N_SAMPLES = 500


def regen_shap(cfg: dict) -> None:
    name      = cfg["name"]
    model_pkl = cfg["model_pkl"]
    test_parq = cfg["test_parq"]
    out_dir   = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Region: {name}")
    print(f"{'='*60}")

    if not model_pkl.exists():
        print(f"  Model not found: {model_pkl}  — skipping")
        return
    if not test_parq.exists():
        print(f"  Test parquet not found: {test_parq}  — skipping")
        return

    # Load model
    print(f"  Loading model: {model_pkl.name}")
    with open(model_pkl, "rb") as f:
        model_data = pickle.load(f)

    # Deployment pkl is a dict with keys: model, platt, isotonic, feature_cols, etc.
    if isinstance(model_data, dict):
        model        = model_data["model"]          # actual LightGBM Booster
        feature_cols = list(model_data["feature_cols"])
        print(f"  Feature cols from deployment pkl: {len(feature_cols)}")
    else:
        model = model_data
        # Determine feature columns from parquet schema
        pf_schema = pq.ParquetFile(test_parq)
        all_cols = pf_schema.schema_arrow.names
        EXCLUDE = {
            "year", "row", "col", "x", "y",
            "transition_01_win5", "transition_01_win3",
            "WDPA_prev", "WDPA", "wdpa",
        }
        if hasattr(model, "feature_name_"):
            feature_cols = model.feature_name_
        elif hasattr(model, "feature_names_in_"):
            feature_cols = list(model.feature_names_in_)
        else:
            feature_cols = [c for c in all_cols if c not in EXCLUDE]
        print(f"  Feature cols: {len(feature_cols)}")

    # Sample N_SAMPLES rows from test parquet
    print(f"  Sampling {N_SAMPLES} rows from test parquet...")
    rng = np.random.default_rng(42)
    pf = pq.ParquetFile(test_parq)
    n_rg = pf.num_row_groups
    n_feat = len(feature_cols)
    reservoir = np.zeros((N_SAMPLES, n_feat), dtype=np.float32)
    fill = 0
    seen = 0
    for i in range(n_rg):
        chunk = pf.read_row_group(i).select(feature_cols).to_pandas().astype(np.float32).values
        for row in chunk:
            seen += 1
            if fill < N_SAMPLES:
                reservoir[fill] = row
                fill += 1
            else:
                r = rng.integers(0, seen)
                if r < N_SAMPLES:
                    reservoir[r] = row
        del chunk
        gc.collect()

    X = reservoir[:fill]
    print(f"  Sample shape: {X.shape}")

    # Compute SHAP
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X, check_additivity=False)

    # Extract positive class
    if isinstance(sv, list):
        sv_pos = sv[1] if len(sv) > 1 else sv[0]
    elif sv.ndim == 3:
        sv_pos = sv[:, :, 1 if sv.shape[-1] > 1 else 0]
    else:
        sv_pos = sv

    # Build display names
    display_names = [SHAP_FEATURE_RENAME.get(f, f) for f in feature_cols]

    # Plot
    print("  Generating SHAP summary plot...")
    # shap.summary_plot manages its own figure size via `plot_size`; any prior
    # plt.figure(figsize=...) call is overridden.  Pass plot_size as a (w, h)
    # tuple — both values must be real numbers (None crashes matplotlib).
    # 11 × 8 gives noticeably more horizontal room for the beeswarm dots while
    # keeping a sensible aspect ratio in the manuscript.
    try:
        shap.summary_plot(
            sv_pos, X,
            feature_names=display_names,
            show=False, plot_type="dot",
            plot_size=(11, 8),
            color_bar_label="Feature value (blue = low, red = high)",
        )
    except TypeError:
        shap.summary_plot(
            sv_pos, X,
            feature_names=display_names,
            show=False, plot_type="dot",
            plot_size=(11, 8),
        )
    plt.tight_layout()

    out_path = out_dir / "shap_summary.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    for cfg in REGIONS:
        regen_shap(cfg)
    print("\nDone.")
