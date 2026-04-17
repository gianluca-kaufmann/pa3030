#!/usr/bin/env python3
"""Shared results reporting core used by regional thin wrappers."""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import zlib
from pathlib import Path
from typing import Dict, Any, Optional, Sequence

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import interp1d, griddata

from .boundaries import (
    detect_and_reproject_coordinates,
    get_region_boundary,
    get_world_countries,
    validate_coordinates,
)
from .config import (
    CALIBRATE_SCRIPT,
    DEFAULT_FUTURE_YEARS_STR,
    HOTSPOT_REGIONS,
    MAP_INSET_SIDE,
    MAP_LEGEND_FONTSIZE as MAP_LEGEND_FONTSIZE_CFG,
    MAP_LEGEND_MARKERSIZE as MAP_LEGEND_MARKERSIZE_CFG,
    MAP_STATS_FONTSIZE,
    MODEL_ID,
    MODEL_LABEL,
    PROBABILITY_MAP_DISPLAY_GAMMA,
    PROBABILITY_MAP_PERCENTILE_MIN,
    PROBABILITY_MAP_PERCENTILE_MAX,
    PROBABILITY_MAP_TRANSFORMATION,
    REGION_LABEL,
    REGION_SLUG,
    RISK_MAP_MISS_COLOR,
    RISK_MAP_OVERLAP_COLOR,
    RISK_MAP_PREDICTED_COLOR,
    X_LIMITS,
    Y_LIMITS,
    get_repo_root,
)
from .io import (
    derive_test_years,
    find_backbone_path,
    find_latest_file,
    find_latest_file_in_dirs,
    load_future_pa_establishments,
    load_metrics_json,
    load_scored_parquet,
    load_wdpa_from_test_parquet,
    resolve_future_parquet,
    resolve_parquet_file,
)

# Try to import shap - if not available, raise clear error when SHAP analysis is requested
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Optional Weights & Biases integration for streaming logs and metrics
try:
    import wandb
except ImportError:
    wandb = None


# =============================================================================
# VISUALIZATION CONSTANTS - Ensures consistent styling across all models
# =============================================================================

# Map figure dimensions (inches)
MAP_FIGSIZE = (14, 11)

# Data aspect ratio (height/width in degrees) - figure dimensions will match this exactly
MAP_EXTENT_ASPECT = (Y_LIMITS[1] - Y_LIMITS[0]) / (X_LIMITS[1] - X_LIMITS[0])

# Map resolution for rasterization (degrees, approximately 1 km)
MAP_RESOLUTION = 0.009

# Map DPI for output files
MAP_DPI = 600

# Font sizes
FONTSIZE_LABEL = 12  # Axis labels
FONTSIZE_TITLE = 15  # Map titles
FONTSIZE_DESCRIPTION = 8  # Description text boxes
FONTSIZE_LEGEND = 10  # Legend text
FONTSIZE_STATS = 9  # Statistics/time period text

# Risk map color scheme
# Category colors using viridis colormap samples
# Compute viridis colors at module load to ensure accuracy while maintaining consistency
# Use modern matplotlib API to avoid deprecation warnings (matplotlib 3.5+)
try:
    # Modern API: matplotlib.colormaps registry (matplotlib 3.5+)
    _viridis_cmap = matplotlib.colormaps['viridis']
except (AttributeError, KeyError, TypeError):
    try:
        # Fallback: plt.get_cmap (also deprecated but works in 3.5-3.7)
        _viridis_cmap = plt.get_cmap('viridis')
    except (AttributeError, TypeError):
        # Final fallback: cm.get_cmap (deprecated in 3.7+)
        import matplotlib.cm as cm
        _viridis_cmap = cm.get_cmap('viridis')
RISK_MAP_COLORS = {
    'none': _viridis_cmap(0.0),  # viridis(0.0) - darkest purple
    'predicted_only': _viridis_cmap(0.35),  # viridis(0.35) - blue-green
    'observed_only': (0.0, 1.0, 1.0, 1.0),  # bright cyan
    'overlap': (1.0, 0.0, 0.0, 1.0),  # bright red
    'protected': '#2E8B57'  # sea green for protected areas
}

# Risk map transparency settings
RISK_MAP_ALPHA_BACKGROUND = 0.15  # Background category transparency
RISK_MAP_ALPHA_PROTECTED = 0.6  # Protected areas overlay transparency

# Existing PA holes color (pixels absent from scored parquet = already protected before test period)
# Must contrast clearly with both the unprotected background (#F0F0F0) and the white ocean.
PA_HOLE_COLOR = '#D4D4D4'  # Lighter gray — distinguishable from #F0F0F0 (unprotected) and white (ocean)

# Probability map color scheme
PROBABILITY_MAP_COLORMAP = 'plasma'  # Colormap: dark purple -> yellow

# Probability map color scaling — imported from config (region-specific).
# SA: percentile_min=25, percentile_max=98, transformation='sqrt'
# USA: percentile_min=0, percentile_max=99.9, transformation='log'
# (log is needed for USA because isotonic calibration collapses >98% of
#  probabilities to the same floor value, making linear/sqrt normalization fail)

# PR curve figure dimensions
PR_CURVE_FIGSIZE = (8, 6)
PR_CURVE_DPI = 300
CUMULATIVE_GAINS_MAX_PLOT_POINTS = 20000


















def create_metrics_table(metrics_data: Dict[str, Any], output_dir: Path, model_type: str, df: Optional[pd.DataFrame] = None, extra_metrics: Optional[Dict[str, float]] = None) -> None:
    """Create metrics table in CSV and LaTeX formats.
    
    Args:
        metrics_data: Dictionary containing metrics from JSON
        output_dir: Output directory for tables
        model_type: Model type (lgbm, rf)
        df: Optional DataFrame with y_true and y_pred_proba for computing Brier score
        extra_metrics: Optional dict of additional metrics (e.g. future_capture_rate, combined_recall)
    """
    print("\n" + "=" * 70)
    print("CREATING METRICS TABLE")
    print("=" * 70)
    
    # Try test_performance first, then fall back to metrics, then empty dict
    test_perf = metrics_data.get("test_performance", {})
    if not test_perf:
        test_perf = metrics_data.get("metrics", {})
    
    # Extract metrics
    metrics_list = [
        ("ROC AUC", test_perf.get("roc_auc", np.nan)),
        ("PR AUC", test_perf.get("pr_auc", np.nan)),
        ("Precision @ 1%",  test_perf.get("precision_at_1pct",  np.nan)),
        ("Precision @ 5%",  test_perf.get("precision_at_5pct",  np.nan)),
        ("Precision @ 10%", test_perf.get("precision_at_10pct", np.nan)),
        # Recall at top-k%: fraction of all PA designation events in the test
        # set captured when screening the top k% of predicted-risk pixels.
        # A random model achieves recall ≈ k/100 (e.g. 0.05 at k=5).
        ("Recall @ 1%",  test_perf.get("recall_at_1pct",  np.nan)),
        ("Recall @ 5%",  test_perf.get("recall_at_5pct",  np.nan)),
        ("Recall @ 10%", test_perf.get("recall_at_10pct", np.nan)),
        ("Baseline Rate", test_perf.get("baseline_rate", np.nan)),
        ("Lift @ 1%",  test_perf.get("lift_at_1pct",  np.nan)),
        ("Lift @ 5%",  test_perf.get("lift_at_5pct",  np.nan)),
        ("Lift @ 10%", test_perf.get("lift_at_10pct", np.nan)),
    ]
    
    # Add Brier score, ECE, MCE if DataFrame is provided
    brier_score = np.nan
    brier_score_calibrated = np.nan
    if df is not None and 'y_true' in df.columns and 'y_pred_proba' in df.columns:
        try:
            brier_score = brier_score_loss(df['y_true'], df['y_pred_proba'])
            metrics_list.append(("Brier Score", brier_score))
            print(f"  Computed Brier Score: {brier_score:.6f}")
            
            # ECE and MCE (uniform bins, n=10)
            y_true_arr = df['y_true'].values
            y_pred_arr = np.clip(df['y_pred_proba'].values.astype(np.float64), 0.0, 1.0 - 1e-9)
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_pred_arr, bin_edges[1:], right=False)
            ece, mce = 0.0, 0.0
            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() > 0:
                    bin_conf = y_pred_arr[mask].mean()
                    bin_acc = y_true_arr[mask].mean()
                    ece += (mask.sum() / len(y_true_arr)) * abs(bin_conf - bin_acc)
                    mce = max(mce, abs(bin_conf - bin_acc))
            metrics_list.append(("ECE (Expected Calibration Error)", ece))
            metrics_list.append(("MCE (Maximum Calibration Error)", mce))
            print(f"  Computed ECE: {ece:.6f}, MCE: {mce:.6f}")
            
            # If calibrated probabilities available, compute Brier score for both
            if 'y_pred_proba_calibrated' in df.columns and 'y_pred_proba_uncalibrated' in df.columns:
                brier_score_uncalibrated = brier_score_loss(df['y_true'], df['y_pred_proba_uncalibrated'])
                brier_score_calibrated = brier_score_loss(df['y_true'], df['y_pred_proba_calibrated'])
                metrics_list.append(("Brier Score (Uncalibrated)", brier_score_uncalibrated))
                metrics_list.append(("Brier Score (Calibrated)", brier_score_calibrated))
                print(f"  Computed Brier Score (Uncalibrated): {brier_score_uncalibrated:.6f}")
                print(f"  Computed Brier Score (Calibrated): {brier_score_calibrated:.6f}")
        except Exception as e:
            print(f"  Warning: Could not compute Brier score / ECE / MCE: {e}")
    
    # Add extra metrics (e.g. Future Capture Rate, Combined Recall from temporal validation)
    if extra_metrics:
        display_names = {"future_capture_rate": "Future Capture Rate (%)", "combined_recall": "Combined Recall (%)"}
        for name, val in extra_metrics.items():
            metrics_list.append((display_names.get(name, name.replace("_", " ").title()), val))
    
    metrics_dict = {
        "Metric": [m[0] for m in metrics_list],
        "Value": [m[1] for m in metrics_list]
    }
    
    df_metrics = pd.DataFrame(metrics_dict)
    
    # Format values for display
    def format_value(val):
        if pd.isna(val):
            return "N/A"
        if isinstance(val, (int, float)):
            if abs(val) < 0.01:
                return f"{val:.6f}"
            elif abs(val) < 1:
                return f"{val:.4f}"
            elif abs(val) < 100:
                return f"{val:.2f}"
            else:
                return f"{val:.1f}"
        return str(val)
    
    df_metrics['Value'] = df_metrics['Value'].apply(format_value)
    
    # Save CSV
    csv_path = output_dir / f"metrics_table.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    
    # Create LaTeX table using f-strings for cleaner code
    latex_path = output_dir / f"metrics_table.tex"
    
    # Escape LaTeX special characters
    def escape_latex(text: str) -> str:
        return text.replace('_', '\\_').replace('%', '\\%')
    
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]
    
    for _, row in df_metrics.iterrows():
        metric = escape_latex(row['Metric'])
        value = row['Value']
        latex_lines.append(f"{metric} & {value} \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{MODEL_LABEL} ({model_type.upper()}) Performance Metrics}}",
        f"\\label{{tab:{MODEL_ID}_{model_type}_metrics}}",
        "\\end{table}",
    ])
    
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines) + '\n')
    
    print(f"Saved LaTeX: {latex_path}")


# ── False positive spatial map ────────────────────────────────────────────────

_NTL_CANDIDATES_R = [
    "ntl", "NTL", "ntl_mean", "VIIRS_ntl", "nighttime_lights",
    "lights_mean", "BU_ntl", "viirs_ntl",
]
_POP_CANDIDATES_R = [
    "pop_density", "population_density", "pop", "GPW_pop", "population",
    "pop_dens", "GPW",
]


def _detect_col_r(candidates: list, available: set) -> Optional[str]:
    for c in candidates:
        if c in available:
            return c
    avail_lower = {a.lower(): a for a in available}
    for c in candidates:
        if c.lower() in avail_lower:
            return avail_lower[c.lower()]
    return None


def create_false_positive_map(
    df: pd.DataFrame,
    output_dir: Path,
    model_type: str,
    region_gdf=None,
) -> None:
    """2-panel map: true positives (green) vs false positives (red) in top-5% risk zone.

    Uses the test-set ground truth labels directly — no backtesting required.
    Outputs: false_positive_map.pdf/.png
    """
    print("\n" + "=" * 70)
    print("FALSE POSITIVE SPATIAL MAP")
    print("=" * 70)

    if "y_true" not in df.columns or "y_pred_proba" not in df.columns:
        print("  Skipping — y_true or y_pred_proba not in scored parquet.")
        return
    if "x" not in df.columns or "y" not in df.columns:
        print("  Skipping — x/y coordinates not available.")
        return

    from scripts.regions.shared.results.config import REGION_LABEL, X_LIMITS, Y_LIMITS

    y_proba = df["y_pred_proba"].values
    y_true  = df["y_true"].values
    threshold = float(np.percentile(y_proba, 95))
    top_mask  = y_proba >= threshold
    tp_mask   = top_mask & (y_true == 1)
    fp_mask   = top_mask & (y_true == 0)

    n_tp      = int(tp_mask.sum())
    n_fp      = int(fp_mask.sum())
    precision = n_tp / max(n_tp + n_fp, 1)
    print(f"  Top-5% zone: {n_tp:,} TP (green), {n_fp:,} FP (red)  precision={precision:.3f}")

    if n_tp + n_fp == 0:
        print("  Nothing to plot — skipping.")
        return

    x_all = df["x"].values
    y_all = df["y"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f"False Positive Analysis — Test Set ({REGION_LABEL}, {model_type.upper()})\n"
        f"Top-5% risk zone: {n_tp:,} true positives / {n_fp:,} false positives  "
        f"precision={precision:.3f}",
        fontsize=11,
    )

    for ax, mask, title, color in [
        (axes[0], tp_mask,
         "True Positives\n(high-risk → actually became protected)", "#2ca02c"),
        (axes[1], fp_mask,
         "False Positives\n(high-risk → NOT protected in test window)", "#d62728"),
    ]:
        if mask.any():
            grid, extent = points_to_raster(
                x_all[mask], y_all[mask], np.ones(mask.sum()),
                target_resolution=0.1, agg_func="max",
            )
            ax.imshow(
                np.flipud(grid), origin="upper", extent=extent,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c", ["white", color]),
                aspect="auto", interpolation="nearest",
            )
        if region_gdf is not None:
            try:
                region_gdf.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=5)
            except Exception:
                pass
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.text(0.02, 0.02, f"{int(mask.sum()):,} pixels",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        p = output_dir / f"false_positive_map.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── Economic exposure ──────────────────────────────────────────────────────────

def create_economic_exposure(
    df: pd.DataFrame,
    test_parquet_path: Optional[Path],
    output_dir: Path,
    model_type: str,
) -> pd.DataFrame:
    """Quantify economic proxy values (NTL, population) across risk zones.

    Joins the test split parquet (which has all feature columns) to the scored
    parquet on (row, col) to retrieve nighttime lights and population density.
    Computes summary statistics for four zones:
      - All test pixels (baseline)
      - Top 10% by predicted probability  (matches existing risk_map_10)
      - Top 5%  by predicted probability  (matches existing risk_map_5)
      - Top 1%  by predicted probability  (matches existing risk_map_1)

    Outputs:
      economic_exposure.csv
      economic_exposure_map.pdf/.png  (if NTL column found)
    """
    print("\n" + "=" * 70)
    print("ECONOMIC EXPOSURE ANALYSIS")
    print("=" * 70)

    if test_parquet_path is None or not test_parquet_path.exists():
        print(f"  Skipping — test parquet not found: {test_parquet_path}")
        return pd.DataFrame()
    if "row" not in df.columns or "col" not in df.columns:
        print("  Skipping — row/col not in scored parquet (needed for join).")
        return pd.DataFrame()

    import pyarrow.parquet as _pq

    schema_names = set(_pq.ParquetFile(test_parquet_path).schema_arrow.names)
    ntl_col = _detect_col_r(_NTL_CANDIDATES_R, schema_names)
    pop_col = _detect_col_r(_POP_CANDIDATES_R, schema_names)
    print(f"  NTL column:        {ntl_col or 'not found'}")
    print(f"  Population column: {pop_col or 'not found'}")

    proxy_cols = [c for c in [ntl_col, pop_col] if c is not None]
    if not proxy_cols:
        print("  No economic proxy columns found — skipping.")
        return pd.DataFrame()

    feat_df = _pq.read_table(
        test_parquet_path, columns=["row", "col"] + proxy_cols
    ).to_pandas()

    merged = df[["row", "col", "y_pred_proba", "x", "y"]].merge(
        feat_df, on=["row", "col"], how="left"
    )
    del feat_df

    top10_mask = merged["y_pred_proba"] >= float(np.percentile(merged["y_pred_proba"], 90))
    top5_mask  = merged["y_pred_proba"] >= float(np.percentile(merged["y_pred_proba"], 95))
    top1_mask  = merged["y_pred_proba"] >= float(np.percentile(merged["y_pred_proba"], 99))

    zones = [
        ("All test pixels (baseline)", np.ones(len(merged), dtype=bool)),
        ("Top 10% risk",               top10_mask),
        ("Top 5% risk",                top5_mask),
        ("Top 1% risk",                top1_mask),
    ]

    rows = []
    for zone_name, mask in zones:
        sub = merged[mask]
        row: Dict = {
            "zone":     zone_name,
            "n_pixels": int(mask.sum()),
        }
        for col in proxy_cols:
            vals = sub[col].dropna()
            row[f"{col}_mean"]   = round(float(vals.mean()),   3) if len(vals) else None
            row[f"{col}_median"] = round(float(vals.median()), 3) if len(vals) else None
            row[f"{col}_p75"]    = round(float(np.percentile(vals, 75)), 3) if len(vals) else None
        rows.append(row)

    exposure_df = pd.DataFrame(rows)
    csv_path = output_dir / f"economic_exposure_{model_type}.csv"
    exposure_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(exposure_df.to_string(index=False))

    # Map: top-1% risk zone coloured by NTL intensity
    if ntl_col is not None and "x" in merged.columns:
        from scripts.regions.shared.results.config import X_LIMITS, Y_LIMITS, REGION_LABEL
        sub1 = merged[top1_mask].dropna(subset=[ntl_col])
        if len(sub1) > 0:
            grid, extent = points_to_raster(
                sub1["x"].values, sub1["y"].values, sub1[ntl_col].values,
                target_resolution=0.1, agg_func="mean",
            )
            fig, ax = plt.subplots(figsize=(12, 9))
            im = ax.imshow(
                np.flipud(grid), origin="upper", extent=extent,
                cmap="plasma", aspect="auto", interpolation="nearest",
            )
            ax.set_xlim(X_LIMITS)
            ax.set_ylim(Y_LIMITS)
            ax.set_title(
                f"Economic Exposure in Top-1% Risk Zone\n"
                f"Top-1% predicted pixels coloured by {ntl_col} ({REGION_LABEL}, {model_type.upper()})",
                fontsize=11,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label(f"{ntl_col} (mean per 0.1° cell)", fontsize=9)
            plt.tight_layout()
            for ext in ["pdf", "png"]:
                p = output_dir / f"economic_exposure_map_{model_type}.{ext}"
                plt.savefig(p, dpi=300, bbox_inches="tight")
                print(f"  Saved: {p}")
            plt.close(fig)

    return exposure_df


# ── Hotspot zoom-in maps ───────────────────────────────────────────────────────

def create_hotspot_maps(
    df: pd.DataFrame,
    output_dir: Path,
    model_type: str,
    region_gdf=None,
) -> None:
    """2-row × N-column figure with zoomed probability and risk maps per hotspot.

    Uses HOTSPOT_REGIONS from config (list of (label, lon_min, lon_max, lat_min, lat_max)).
    Top row: continuous P(protection) probability map at 0.02° resolution.
    Bottom row: top-1% risk zone (binary) in the same hotspot box.

    Output: hotspot_maps_{model_type}.pdf/.png
    """
    print("\n" + "=" * 70)
    print("HOTSPOT ZOOM-IN MAPS")
    print("=" * 70)

    from scripts.regions.shared.results.config import HOTSPOT_REGIONS, REGION_LABEL

    if not HOTSPOT_REGIONS:
        print("  No hotspot regions defined in config — skipping.")
        return
    if "x" not in df.columns or "y" not in df.columns:
        print("  Skipping — x/y not available.")
        return

    x_all   = df["x"].values
    y_all   = df["y"].values
    p_all   = df["y_pred_proba"].values
    top1_thr = float(np.percentile(p_all, 99))
    top1_mask = p_all >= top1_thr

    n = len(HOTSPOT_REGIONS)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 12))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Hotspot Zoom-ins — {REGION_LABEL} ({model_type.upper()})\n"
        "Top row: P(protection) probability | Bottom row: top-1% risk zone",
        fontsize=12,
    )

    for col_idx, (label, x0, x1, y0, y1) in enumerate(HOTSPOT_REGIONS):
        in_box = (x_all >= x0) & (x_all <= x1) & (y_all >= y0) & (y_all <= y1)
        if not in_box.any():
            print(f"  WARNING: no pixels in hotspot '{label}' — check coordinates.")
            for ri in range(2):
                axes[ri, col_idx].set_visible(False)
            continue

        print(f"  {label}: {in_box.sum():,} pixels")
        x_h  = x_all[in_box]
        y_h  = y_all[in_box]
        p_h  = p_all[in_box]

        # Row 0: continuous probability (sqrt-stretched)
        ax0 = axes[0, col_idx]
        grid_p, extent_p = points_to_raster(
            x_h, y_h, np.sqrt(np.clip(p_h, 0, 1)),
            target_resolution=0.02, agg_func="max",
        )
        im = ax0.imshow(
            np.flipud(grid_p), origin="upper", extent=extent_p,
            cmap="plasma", vmin=0, vmax=1, aspect="auto", interpolation="nearest",
        )
        if region_gdf is not None:
            try:
                region_gdf.boundary.plot(ax=ax0, color="black", linewidth=0.4, zorder=5)
            except Exception:
                pass
        ax0.set_xlim(x0, x1)
        ax0.set_ylim(y0, y1)
        ax0.set_title(label, fontsize=10)
        if col_idx == 0:
            ax0.set_ylabel("P(protection) [√-stretch]", fontsize=8)
        plt.colorbar(im, ax=ax0, fraction=0.04, pad=0.03)

        # Row 1: top-1% risk zone (binary)
        ax1 = axes[1, col_idx]
        top1_in = top1_mask & in_box
        if top1_in.any():
            grid_b, extent_b = points_to_raster(
                x_all[top1_in], y_all[top1_in], np.ones(top1_in.sum()),
                target_resolution=0.02, agg_func="max",
            )
            ax1.imshow(
                np.flipud(grid_b), origin="upper", extent=extent_b,
                cmap=matplotlib.colors.ListedColormap(["#1a78c2"]),
                vmin=0, vmax=1, aspect="auto", interpolation="nearest", alpha=0.85,
            )
        if region_gdf is not None:
            try:
                region_gdf.boundary.plot(ax=ax1, color="black", linewidth=0.4, zorder=5)
            except Exception:
                pass
        ax1.set_xlim(x0, x1)
        ax1.set_ylim(y0, y1)
        ax1.text(0.02, 0.02, f"{int(top1_in.sum()):,} top-1% pixels",
                 transform=ax1.transAxes, fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        if col_idx == 0:
            ax1.set_ylabel("Top-1% risk zone", fontsize=8)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = output_dir / f"hotspot_maps_{model_type}.{ext}"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def create_comparison_table(
    output_dir: Path,
    model_type: str,
    repo_root: Path,
    scratch_root: Optional[Path],
) -> None:
    """Side-by-side performance comparison table for all configured regions.

    Reads the already-written metrics_table.csv for each region and produces
    a two-column PDF table + LaTeX file.  Gracefully skips if neither or only
    one region's CSV is found (but still writes whatever is available).
    """
    print("\n" + "=" * 70)
    print("CREATING CROSS-REGION COMPARISON TABLE")
    print("=" * 70)

    # Canonical metrics rows and display order
    ROWS = [
        ("ROC AUC",              "ROC AUC"),
        ("PR AUC",               "PR AUC"),
        ("Precision @ 1%",       "Precision @ 1\\%"),
        ("Precision @ 5%",       "Precision @ 5\\%"),
        ("Lift @ 1%",            "Lift @ 1\\%"),
        ("Lift @ 5%",            "Lift @ 5\\%"),
        ("Baseline Rate",        "Baseline Rate"),
        ("Brier Score",          "Brier Score"),
        ("ECE (Expected Calibration Error)", "ECE"),
    ]

    # Region configs: (slug, model_id, display_label)
    REGIONS = [
        ("south_america", "model1", "Model 1 (SA)"),
        ("usa",           "model2", "Model 2 (USA)"),
        ("se_asia",       "model3", "Model 3 (SEA)"),
    ]

    def find_metrics_csv(region_slug: str, model_id: str) -> Optional[Path]:
        candidates = []
        if scratch_root:
            candidates.append(
                scratch_root / f"outputs/{region_slug}/results/{model_id}_{model_type}/metrics_table.csv"
            )
        candidates.append(
            repo_root / f"outputs/{region_slug}/results/{model_id}_{model_type}/metrics_table.csv"
        )
        for c in candidates:
            if c.exists():
                return c
        return None

    # Load each region's CSV into a dict {metric_name: value_str}
    region_data = {}
    for slug, mid, label in REGIONS:
        csv_path = find_metrics_csv(slug, mid)
        if csv_path:
            df_m = pd.read_csv(csv_path)
            region_data[label] = dict(zip(df_m['Metric'], df_m['Value'].astype(str)))
            print(f"  Loaded metrics for {label}: {csv_path}")
        else:
            print(f"  Metrics CSV not found for {label} — column will show N/A")
            region_data[label] = {}

    found_labels = [label for _, _, label in REGIONS if region_data[label]]
    if not found_labels:
        print("  No metrics CSVs found for either region; skipping comparison table.")
        return

    # Build table data
    col_labels = [label for _, _, label in REGIONS]
    table_rows = []
    for csv_key, _ in ROWS:
        row = [csv_key]
        for label in col_labels:
            row.append(region_data[label].get(csv_key, "N/A"))
        table_rows.append(row)

    # ── PDF table ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, len(table_rows) * 0.55 + 1.5))
    ax.axis('off')

    display_row_labels = [display for _, display in ROWS]
    cell_text = [r[1:] for r in table_rows]
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=display_row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#4A90D9')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    # Style row label column (alternating shading)
    for i in range(1, len(table_rows) + 1):
        tbl[(i, -1)].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')

    ax.set_title('Cross-Region Model Performance Comparison',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=12)
    plt.tight_layout()

    pdf_path = output_dir / 'comparison_table.pdf'
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved PDF: {pdf_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    def esc(s: str) -> str:
        return s.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')

    header = " & ".join(["Metric"] + col_labels) + " \\\\"
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'r' * len(col_labels)}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for csv_key, display in ROWS:
        vals = " & ".join(region_data[label].get(csv_key, "N/A") for _, _, label in REGIONS)
        latex_lines.append(f"{esc(display)} & {vals} \\\\")
    latex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Cross-region performance comparison (Model 1: South America, Model 2: USA, Model 3: South East Asia).}",
        "\\label{tab:comparison}",
        "\\end{table}",
    ]
    latex_path = output_dir / 'comparison_table.tex'
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines) + '\n')
    print(f"  Saved LaTeX: {latex_path}")


def create_pr_curve(df: pd.DataFrame, metrics_data: Dict[str, Any], output_dir: Path, model_type: str) -> None:
    """Create Precision-Recall curve with baseline prevalence."""
    print("\n" + "=" * 70)
    print("CREATING PRECISION-RECALL CURVE")
    print("=" * 70)
    
    y_true = df['y_true'].values
    y_pred_proba = df['y_pred_proba'].values
    
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Get baseline prevalence - try test_performance first, then metrics, then compute from df
    test_perf = metrics_data.get("test_performance", {})
    if not test_perf:
        test_perf = metrics_data.get("metrics", {})
    baseline_rate = test_perf.get("baseline_rate", np.mean(y_true))
    
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  Baseline prevalence: {baseline_rate:.6f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=PR_CURVE_FIGSIZE)
    
    # Plot PR curve
    ax.plot(recall, precision, linewidth=2, label=f'Model (AP = {pr_auc:.4f})')
    
    # Plot baseline (horizontal line at baseline rate)
    ax.axhline(y=baseline_rate, color='r', linestyle='--', linewidth=2, 
               label=f'Baseline (prevalence = {baseline_rate:.6f})')
    
    ax.set_xlabel('Recall', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Precision', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) Precision-Recall Curve', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()

    # Save PDF
    pdf_path = output_dir / f"pr_curve.pdf"
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")
    
    plt.close()


def create_roc_curve(df: pd.DataFrame, output_dir: Path, model_type: str) -> None:
    """Create ROC curve with random-guess diagonal and AUC annotation."""
    print("\n" + "=" * 70)
    print("CREATING ROC CURVE")
    print("=" * 70)

    y_true = df['y_true'].values
    y_pred_proba = df['y_pred_proba'].values

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"  ROC AUC: {auc:.4f}")

    # Downsample if very large (purely cosmetic)
    n_pts = len(fpr)
    if n_pts > CUMULATIVE_GAINS_MAX_PLOT_POINTS:
        idx = np.linspace(0, n_pts - 1, CUMULATIVE_GAINS_MAX_PLOT_POINTS, dtype=int)
        fpr, tpr = fpr[idx], tpr[idx]

    fig, ax = plt.subplots(figsize=PR_CURVE_FIGSIZE)
    ax.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random guess (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('True Positive Rate', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) ROC Curve',
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    pdf_path = output_dir / 'roc_curve.pdf'
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    print(f"  Saved PDF: {pdf_path}")
    plt.close()


def create_cumulative_gains_chart(df: pd.DataFrame, output_dir: Path, model_type: str) -> None:
    """Create cumulative gains chart (search efficiency): % area searched vs % PAs captured.
    
    Sorts by predicted probability descending; plots cumulative recall (fraction of all
    positive pixels captured) as the fraction of area searched increases. Includes a
    45-degree random-guess baseline.
    
    Args:
        df: DataFrame with y_true and y_pred_proba columns
        output_dir: Output directory for plots
        model_type: Model type (lgbm, rf)
    """
    print("\n" + "=" * 70)
    print("CREATING CUMULATIVE GAINS CHART (SEARCH EFFICIENCY)")
    print("=" * 70)
    
    y_true = df['y_true'].values
    y_pred_proba = df['y_pred_proba'].values
    n = len(y_true)
    total_positives = float(y_true.sum())
    
    if total_positives <= 0:
        print("  Skipping cumulative gains: no positive labels in data.")
        return
    
    # Sort by predicted probability descending (highest risk first)
    order = np.argsort(-y_pred_proba)
    y_true_sorted = y_true[order]
    cumulative_captured = np.cumsum(y_true_sorted)
    
    # x: % of area searched (0 to 100); y: % of PAs captured (recall)
    pct_area_searched = (np.arange(1, n + 1, dtype=float) / n) * 100
    pct_pas_captured = (cumulative_captured / total_positives) * 100

    # Plotting tens of millions of points to vector PDF can cause compression
    # timeouts/zlib failures on some filesystems; downsample uniformly for display.
    n_points = len(pct_area_searched)
    if n_points > CUMULATIVE_GAINS_MAX_PLOT_POINTS:
        idx = np.linspace(0, n_points - 1, CUMULATIVE_GAINS_MAX_PLOT_POINTS, dtype=int)
        pct_area_plot = pct_area_searched[idx]
        pct_pas_plot = pct_pas_captured[idx]
        print(f"  Downsampled cumulative gains curve: {n_points:,} -> {len(idx):,} points")
    else:
        pct_area_plot = pct_area_searched
        pct_pas_plot = pct_pas_captured
    
    fig, ax = plt.subplots(figsize=PR_CURVE_FIGSIZE)
    ax.plot(pct_area_plot, pct_pas_plot, linewidth=2, label='Model')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.5, label='Random guess')
    ax.set_xlabel('% of Area Searched', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('% of PAs Captured', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) Cumulative Gains (Search Efficiency)', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    pdf_path = output_dir / f"cumulative_gains.pdf"
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")
    plt.close()


def create_calibration_curve(df: pd.DataFrame, output_dir: Path, model_type: str) -> None:
    """Create reliability/calibration curve (uniform bins) for test predictions.

    Plots uncalibrated curve (dashed) if y_pred_proba_uncalibrated column is present,
    and the main calibrated curve from y_pred_proba_calibrated (or y_pred_proba).
    Saves as calibration_reliability.pdf only.

    Args:
        df: DataFrame with y_true and y_pred_proba columns
        output_dir: Output directory for plots
        model_type: Model type (lgbm, rf)
    """
    print("\n" + "=" * 70)
    print("CREATING CALIBRATION/RELIABILITY CURVE")
    print("=" * 70)

    y_true = df['y_true'].values

    has_uncalibrated = 'y_pred_proba_uncalibrated' in df.columns
    has_calibrated = 'y_pred_proba_calibrated' in df.columns

    # Main curve: prefer calibrated column, fall back to y_pred_proba
    if has_calibrated:
        y_main = df['y_pred_proba_calibrated'].values
        main_label = 'Calibrated'
    else:
        y_main = df['y_pred_proba'].values
        main_label = 'Model predictions'

    n_bins = 10

    fig, ax = plt.subplots(figsize=PR_CURVE_FIGSIZE)

    # Uncalibrated curve (dashed), if available
    if has_uncalibrated:
        frac_pos_uncal, mean_pred_uncal = calibration_curve(
            y_true, df['y_pred_proba_uncalibrated'].values, n_bins=n_bins, strategy='uniform'
        )
        ax.plot(mean_pred_uncal, frac_pos_uncal, 's--',
                label='Uncalibrated', linewidth=2, markersize=8, alpha=0.7)

    # Main (calibrated) curve
    frac_pos_main, mean_pred_main = calibration_curve(
        y_true, y_main, n_bins=n_bins, strategy='uniform'
    )
    ax.plot(mean_pred_main, frac_pos_main, 'o-',
            label=main_label, linewidth=2, markersize=8)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)

    ax.set_xlabel('Mean Predicted Probability', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Fraction of Positives', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) Calibration Reliability Diagram '
                 f'(Uniform Bins, n={n_bins})',
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    pdf_path = output_dir / 'calibration_reliability.pdf'
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")

    plt.close(fig)
    print(f"\nCalibration analysis complete")










def _add_latlon_ticks(ax, xlim_m: tuple, ylim_m: tuple) -> None:
    """Replace raw EPSG:3857 metre tick labels with readable lat/lon degree labels.

    Uses the standard Web Mercator forward/inverse formulae (no external dependency).
    Safe to call for any region — tick spacing is chosen automatically from the extent.
    """
    import math

    R = 6378137.0  # WGS-84 semi-major axis (metres)

    def m_to_lon(x_m):
        return x_m * 180.0 / (math.pi * R)

    def m_to_lat(y_m):
        return math.degrees(2.0 * math.atan(math.exp(y_m / R)) - math.pi / 2.0)

    def lon_to_m(lon):
        return lon * math.pi * R / 180.0

    def lat_to_m(lat):
        return R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))

    lon_min = m_to_lon(xlim_m[0])
    lon_max = m_to_lon(xlim_m[1])
    lat_min = m_to_lat(ylim_m[0])
    lat_max = m_to_lat(ylim_m[1])

    # Choose nice tick intervals based on the extent of the map
    lon_range = abs(lon_max - lon_min)
    lat_range = abs(lat_max - lat_min)
    lon_step = 20 if lon_range > 60 else 10 if lon_range > 30 else 5 if lon_range > 10 else 2
    lat_step = 10 if lat_range > 40 else 5 if lat_range > 15 else 2 if lat_range > 5 else 1

    lon_ticks = np.arange(
        math.ceil(lon_min / lon_step) * lon_step,
        math.floor(lon_max / lon_step) * lon_step + 0.01,
        lon_step,
    )
    lat_ticks = np.arange(
        math.ceil(lat_min / lat_step) * lat_step,
        math.floor(lat_max / lat_step) * lat_step + 0.01,
        lat_step,
    )

    ax.set_xticks([lon_to_m(lon) for lon in lon_ticks])
    ax.set_xticklabels(
        [f'{abs(lon):.0f}°W' if lon < -0.05 else f'{lon:.0f}°E' if lon > 0.05 else '0°'
         for lon in lon_ticks]
    )
    ax.set_yticks([lat_to_m(lat) for lat in lat_ticks])
    ax.set_yticklabels(
        [f'{abs(lat):.0f}°N' if lat > 0.05 else f'{abs(lat):.0f}°S' if lat < -0.05 else '0°'
         for lat in lat_ticks]
    )
    ax.set_xlabel('Longitude', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Latitude', fontsize=FONTSIZE_LABEL)


def _add_scale_bar_3857(
    ax,
    proj_bounds: tuple,
    ref_lat_deg: float = -20.0,
    bar_km: int = 1000,
    position: str = 'lower left',
    *,
    anchor_x_frac: Optional[float] = None,
    anchor_y_frac: Optional[float] = None,
) -> None:
    """Add a simple horizontal scale bar to an EPSG:3857 map axes.

    Web Mercator (EPSG:3857) preserves angles but distorts distances with latitude.
    We convert the desired ground distance to projected metres at *ref_lat_deg* so
    the bar length is correct at that reference latitude.

    Parameters
    ----------
    ax : matplotlib Axes
    proj_bounds : (minx, miny, maxx, maxy) in EPSG:3857 metres
    ref_lat_deg : reference latitude (degrees) for the distance conversion.
    bar_km : scale bar length in km (default 1000).
    position : one of 'lower left', 'lower right', 'upper left', 'upper right'.
    """
    import math

    R = 6_378_137.0  # WGS-84 semi-major axis (metres)
    cos_lat = math.cos(math.radians(ref_lat_deg))
    bar_m = bar_km * 1_000.0 / cos_lat  # projected x-width for bar_km km at ref_lat

    map_w = proj_bounds[2] - proj_bounds[0]
    map_h = proj_bounds[3] - proj_bounds[1]
    tick_h = 0.007 * map_h

    # Horizontal anchor (allow fine-grained placement via fractions of map extent)
    if 'right' in position:
        x1_frac = 0.94 if anchor_x_frac is None else anchor_x_frac
        x1 = proj_bounds[0] + x1_frac * map_w
        x0 = x1 - bar_m
    else:
        x0_frac = 0.06 if anchor_x_frac is None else anchor_x_frac
        x0 = proj_bounds[0] + x0_frac * map_w
        x1 = x0 + bar_m

    if 'upper' in position:
        y0_frac = 0.93 if anchor_y_frac is None else anchor_y_frac
        y0 = proj_bounds[1] + y0_frac * map_h
        label_va = 'bottom'
        label_y = y0 + 1.8 * tick_h
    else:
        y0_frac = 0.04 if anchor_y_frac is None else anchor_y_frac
        y0 = proj_bounds[1] + y0_frac * map_h
        label_va = 'bottom'
        label_y = y0 + 1.8 * tick_h

    # Horizontal bar
    ax.plot([x0, x1], [y0, y0], color='black', linewidth=2.0, zorder=12, solid_capstyle='butt')
    # End ticks
    for xk in (x0, x1):
        ax.plot([xk, xk], [y0 - tick_h / 2, y0 + tick_h / 2], color='black', linewidth=1.5, zorder=12)
    # Label centred above the bar
    ax.text(
        (x0 + x1) / 2, label_y,
        f'{bar_km:,} km',
        ha='center', va=label_va, fontsize=10, color='black', zorder=12,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5),
    )


def _add_north_arrow(
    ax,
    proj_bounds: tuple,
    position: str = 'upper right',
    *,
    anchor_x_frac: Optional[float] = None,
    anchor_y_frac: Optional[float] = None,
) -> None:
    """Add a simple N-arrow compass to an EPSG:3857 map axes.

    Parameters
    ----------
    ax : matplotlib Axes
    proj_bounds : (minx, miny, maxx, maxy) in EPSG:3857 metres
    position : one of 'lower left', 'lower right', 'upper left', 'upper right'
    """
    map_w = proj_bounds[2] - proj_bounds[0]
    map_h = proj_bounds[3] - proj_bounds[1]

    # Arrow length and width proportional to map
    arrow_len = 0.030 * map_h
    x_frac = (0.030 if 'left' in position else 0.968) if anchor_x_frac is None else anchor_x_frac
    y_frac = (0.040 if 'lower' in position else 0.940) if anchor_y_frac is None else anchor_y_frac
    offset_x = x_frac * map_w
    offset_y = y_frac * map_h

    x = proj_bounds[0] + offset_x
    y = proj_bounds[1] + offset_y

    ax.annotate(
        '',
        xy=(x, y + arrow_len),
        xytext=(x, y),
        xycoords='data', textcoords='data',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.8),
        zorder=13,
    )
    ax.text(
        x, y + arrow_len * 1.35,
        'N',
        ha='center', va='bottom', fontsize=10, fontweight='bold',
        color='black', zorder=13,
    )


def _plot_backbone_background(
    ax,
    backbone_path: Optional[Path],
    zorder: int = 0,
    hole_color: Optional[str] = None,
) -> bool:
    """Plot the backbone raster as the gray background layer for PA holes.

    The backbone (uint8, 1=inside region, 0=outside) is already in EPSG:3857
    so it can be passed directly to imshow without reprojection.  Pixels with
    value 0 are masked transparent so the white figure background (ocean) shows
    through.  Pixels with value 1 are drawn in PA_HOLE_COLOR; the prediction
    layers drawn on top will overwrite all non-PA pixels, leaving only the
    genuine PA holes visible in gray.

    Returns True if the backbone was plotted successfully, False if the caller
    should fall back to the region hull polygon (no country outlines;
    ``get_region_boundary`` / GeoJSON), filled with *hole_color*.
    """
    if backbone_path is None:
        return False
    try:
        import rasterio
        with rasterio.open(backbone_path) as src:
            data = src.read(1)          # uint8 array: 1=region, 0=outside
            t = src.transform
            left   = t.c
            top    = t.f
            right  = left + src.width  * t.a   # t.a > 0 (pixel width)
            bottom = top  + src.height * t.e   # t.e < 0 (pixel height, negative)
            extent = (left, right, bottom, top)

        backbone_float = data.astype(np.float32)
        backbone_float[data == 0] = np.nan
        backbone_masked = np.ma.masked_invalid(backbone_float)

        from matplotlib.colors import ListedColormap
        fill_color = hole_color if hole_color is not None else PA_HOLE_COLOR
        ax.imshow(
            backbone_masked,
            extent=extent,
            cmap=ListedColormap([fill_color]),
            vmin=0.5, vmax=1.5,
            origin='upper',
            interpolation='nearest',
            aspect='equal',
            zorder=zorder,
        )
        print(f"  Backbone background plotted from: {backbone_path}")
        return True
    except Exception as exc:
        print(f"  Warning: could not plot backbone raster ({exc}); use region hull fallback (no backbone).")
        return False


def points_to_raster(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                     target_resolution: Optional[float] = None,
                     agg_func: str = 'mean',
                     extent_bounds: Optional[tuple[float, float, float, float]] = None,
                     ) -> tuple[np.ndarray, tuple]:
    """Convert point data to a raster grid for pixel-perfect visualization.
    
    Uses histogram-based binning for maximum precision and performance.
    Works with any coordinate system (degrees, meters, etc.).
    For 1 km resolution in degrees: ~0.009 degrees. For meters: 1000.0 meters.
    
    Args:
        x: X coordinates (longitude in degrees, or easting in meters, etc.)
        y: Y coordinates (latitude in degrees, or northing in meters, etc.)
        values: Values to rasterize
        target_resolution: Target resolution in same units as coordinates. If None, auto-detects from data.
        agg_func: Aggregation for points in same cell: 'mean' (default) or 'max'.
                  Use 'max' for categorical rasters so overlap wins (e.g. risk map).
        extent_bounds: Optional (xmin, ymin, xmax, ymax) in the same CRS as x/y
            (standard GIS / rasterio bounding-box order).
            When set, the raster grid covers this box exactly (no extra padding),
            so stacked map layers share the same geometry.
    
    Returns:
        Tuple of (raster_array, extent) where extent is (xmin, xmax, ymin, ymax)
        (Matplotlib imshow order). Raster array has shape (nrows, ncols) and may
        contain NaN for empty cells.
    """
    # Calculate bounds
    if extent_bounds is not None:
        x_min, y_min, x_max, y_max = extent_bounds  # GIS order: left, bottom, right, top
    elif len(x) > 0:
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
    else:
        raise ValueError("points_to_raster: empty coordinates and extent_bounds not provided")
    
    # Determine grid resolution
    if target_resolution is None:
        # For 1 km resolution, ~0.009 degrees
        # Estimate from data: use median of non-zero coordinate differences
        sample_size = min(50000, len(x))
        np.random.seed(42)  # Reproducible resolution estimate
        sample_idx = np.random.choice(len(x), sample_size, replace=False)
        x_sample = np.sort(x[sample_idx])
        y_sample = np.sort(y[sample_idx])
        
        # Find smallest non-zero differences (these represent grid resolution)
        x_diffs = np.diff(x_sample)
        x_diffs = x_diffs[x_diffs > 1e-10]
        y_diffs = np.diff(y_sample)
        y_diffs = y_diffs[y_diffs > 1e-10]
        
        if len(x_diffs) > 100 and len(y_diffs) > 100:
            # Use 10th percentile of differences as resolution estimate
            x_res = np.percentile(x_diffs[x_diffs < np.percentile(x_diffs, 25)], 10)
            y_res = np.percentile(y_diffs[y_diffs < np.percentile(y_diffs, 25)], 10)
        else:
            # Fallback: use 1 km equivalent (~0.009 degrees)
            x_res = 0.009
            y_res = 0.009
    else:
        x_res = target_resolution
        y_res = target_resolution
    
    # Ensure resolution is reasonable (not too fine, not too coarse)
    x_res = max(x_res, (x_max - x_min) / 10000)  # Minimum: 10000 cells
    x_res = min(x_res, (x_max - x_min) / 100)    # Maximum: 100 cells
    y_res = max(y_res, (y_max - y_min) / 10000)
    y_res = min(y_res, (y_max - y_min) / 100)
    
    # Create grid edges
    if extent_bounds is not None:
        padding = 0.0
    else:
        # Extend slightly beyond bounds to ensure all points are included
        padding = max(x_res, y_res) * 0.1
    x_edges = np.arange(x_min - padding, x_max + padding + x_res, x_res)
    y_edges = np.arange(y_min - padding, y_max + padding + y_res, y_res)
    
    # Calculate grid dimensions
    ncols = len(x_edges) - 1
    nrows = len(y_edges) - 1

    if len(x) == 0:
        raster = np.full((nrows, ncols), np.nan, dtype=np.float32)
        raster = np.flipud(raster)
        extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
        return raster, extent
    
    # Use digitize for binning (no pandas - pure NumPy for 47M+ row scalability)
    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1
    
    # Clip indices to valid range
    x_idx = np.clip(x_idx, 0, ncols - 1).astype(np.intp)
    y_idx = np.clip(y_idx, 0, nrows - 1).astype(np.intp)
    
    # Flat index for 1D aggregation arrays
    flat_idx = y_idx * ncols + x_idx
    n_cells = nrows * ncols
    
    # Aggregate using np.add.at (mean) or np.maximum.at (max) - no pandas groupby
    values = np.asarray(values, dtype=np.float64)
    
    if agg_func == 'mean':
        sum_arr = np.zeros(n_cells, dtype=np.float64)
        count_arr = np.zeros(n_cells, dtype=np.float64)
        np.add.at(sum_arr, flat_idx, values)
        np.add.at(count_arr, flat_idx, 1.0)
        # Avoid RuntimeWarning: np.where evaluates both branches before selecting,
        # so divide only where count > 0 and fill the rest with nan.
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_arr = np.where(count_arr > 0, sum_arr / count_arr, np.nan)
        raster_flat = mean_arr
    else:  # 'max'
        raster_flat = np.full(n_cells, -np.inf, dtype=np.float64)
        np.maximum.at(raster_flat, flat_idx, values)
        raster_flat = np.where(np.isfinite(raster_flat), raster_flat, np.nan)
    
    raster = raster_flat.reshape(nrows, ncols).astype(np.float32)
    
    # Flip raster vertically because imshow with origin='upper' expects row 0 at top (highest y)
    # Currently, y_idx=0 corresponds to y_min (lowest), but we want it at the bottom
    # So we need to flip the array so that row 0 corresponds to y_max (highest)
    raster = np.flipud(raster)
    
    # Define extent for imshow (left, right, bottom, top)
    # Use the actual edge coordinates
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    
    return raster, extent


def create_risk_map(
    df: pd.DataFrame,
    region_boundary_path: Optional[Path],
    output_dir: Path,
    model_type: str,
    metrics_data: Optional[Dict[str, Any]] = None,
    test_parquet_path: Optional[Path] = None,
    test_years: Optional[Sequence[int]] = None,
    threshold_pct: float = 1.0,
    future_parquet_path: Optional[Path] = None,
    future_years: Optional[Sequence[int]] = None,
    sa_gdf: Optional[gpd.GeoDataFrame] = None,
    backbone_path: Optional[Path] = None,
    pixel_size_m: float = 1000.0,
) -> Optional[Dict[str, Any]]:
    """Create SIMPLIFIED risk map showing model predictions vs actual PA establishments.

    SIMPLIFIED DESIGN (radically simplified from previous complex version):
    - Shows 3-4 categories with clear, distinct colors:
      * Predicted high-risk (top threshold_pct%) - BLUE
      * Actual PA establishments (test period) - ORANGE-RED
      * Overlap (correct predictions) - GREEN
      * Future PA establishments (if provided) - YELLOW
    - Minimal text and legend
    - No protection layer overlays
    - No temporal dynamics visualization
    - Focus: Answer one question clearly: "Where did the model predict vs where did PAs actually establish?"

    Args:
        df: Scored prediction DataFrame.
        region_boundary_path: Optional path to South America boundary.
        output_dir: Directory where figures will be written.
        model_type: Model name (rf, lgbm, brf).
        metrics_data: Optional metrics/metadata dictionary.
        test_parquet_path: Path to original test parquet (e.g. *test_win5.parquet*).
        test_years: Explicit list of test years. If not provided, they will be
            derived from `test_parquet_path` (and, if needed, from metadata)
            using `derive_test_years`.
        future_parquet_path: Optional path to future period parquet (e.g., validation set for 2020-2024).
        future_years: List of years to consider as "future" for temporal validation (e.g., [2020, 2021, 2022, 2023, 2024]).
        pixel_size_m: Side length of one pixel in metres (default 1000 = 1 km raster).

    Returns:
        Dict with threshold_pct, total_pa_pixels/km2, captured_pixels/km2, recall_pct,
        precision_pct, and optionally future_capture_rate / combined_recall.
        Always returned (not None) so callers can build capture_summary.csv.
    """
    print("\n" + "=" * 70)
    print("CREATING SIMPLIFIED RISK MAP")
    print("=" * 70)
    
    # Get South America boundary (use cached if provided)
    if sa_gdf is None:
        sa_gdf = get_region_boundary(region_boundary_path)
    
    # Ensure CRS is set (assume EPSG:4326 if not set)
    if sa_gdf.crs is None:
        sa_gdf.set_crs('EPSG:4326', inplace=True)
    
    print(f"  {REGION_LABEL} boundary CRS: {sa_gdf.crs}")
    print(f"  {REGION_LABEL} boundary bounds: {sa_gdf.total_bounds}")

    # Reproject to EPSG:3857 (Web Mercator) for accurate visualization
    sa_gdf_proj = sa_gdf.to_crs('EPSG:3857')
    print(f"  Reprojected to EPSG:3857 (Web Mercator)")
    print(f"  Projected bounds (meters): {sa_gdf_proj.total_bounds}")

    # Create GeoDataFrame from predictions
    geometry = gpd.points_from_xy(df['x'], df['y'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # Reproject predictions to EPSG:3857
    gdf = gdf.to_crs('EPSG:3857')

    print(f"  Total rows (pixel-years): {len(gdf):,}")
    
    # Extract time period for labels
    if test_years is None or len(test_years) == 0:
        test_years = derive_test_years(metrics_data, None, test_parquet_path)
    time_period = f"{min(test_years)}-{max(test_years)}" if test_years else "2018-2019"
    
    # Step 1: Load WDPA data to identify ACTUAL establishments during test period
    # CRITICAL: Prioritize actual test-period establishments over y_true (which is 5-year lookahead)
    wdpa_test_df = load_wdpa_from_test_parquet(test_parquet_path, test_years)
    
    has_actual_establishments = False
    if wdpa_test_df is not None and 'established_in_test_period' in wdpa_test_df.columns:
        print(f"  ✓ WDPA data loaded - using ACTUAL test-period establishments")
        
        # Determine join columns
        join_cols = ['row', 'col'] if all(col in df.columns for col in ['row', 'col']) and all(col in wdpa_test_df.columns for col in ['row', 'col']) else ['x', 'y']
        
        if join_cols == ['row', 'col']:
            # Row/col join (preferred): use max so pixel is 1 if established in ANY test year
            join_df = wdpa_test_df[['row', 'col', 'established_in_test_period']].groupby(['row', 'col']).agg({'established_in_test_period': 'max'}).reset_index()
            df = df.merge(join_df, on=['row', 'col'], how='left')
            df['established_in_test_period'] = df['established_in_test_period'].fillna(0).astype(int)
            has_actual_establishments = True
            print(f"    Joined via row/col: {(df['established_in_test_period'] == 1).sum():,} actual establishments")
        elif join_cols == ['x', 'y']:
            # X/y join with rounding (fallback)
            df['x_rounded'] = df['x'].round(6)
            df['y_rounded'] = df['y'].round(6)
            join_df = wdpa_test_df[['x', 'y', 'established_in_test_period']].copy()
            join_df['x_rounded'] = join_df['x'].round(6)
            join_df['y_rounded'] = join_df['y'].round(6)
            join_df = join_df[['x_rounded', 'y_rounded', 'established_in_test_period']].groupby(['x_rounded', 'y_rounded']).agg({'established_in_test_period': 'max'}).reset_index()
            df = df.merge(join_df, on=['x_rounded', 'y_rounded'], how='left')
            df = df.drop(columns=['x_rounded', 'y_rounded'])
            df['established_in_test_period'] = df['established_in_test_period'].fillna(0).astype(int)
            has_actual_establishments = True
            print(f"    Joined via x/y: {(df['established_in_test_period'] == 1).sum():,} actual establishments")
    
    # Step 1b: Load FUTURE PA establishments (post-test period) if provided.
    # These are folded into is_observed so the map's recall matches the 5-year-horizon
    # metrics (PR-AUC, Recall@K) which use transition_01_win5 as ground truth.
    # The test-period subset is tracked separately for breakdown reporting only.
    has_future_establishments = False
    if future_parquet_path is not None and future_years is not None and len(future_years) > 0:
        future_df = load_future_pa_establishments(future_parquet_path, future_years)
        
        if future_df is not None and 'established_in_future_period' in future_df.columns:
            print(f"  ✓ Future PA establishments loaded - folded into observed (5-yr prediction window)")
            
            # Determine join columns (same logic as test period)
            join_cols = ['row', 'col'] if all(col in df.columns for col in ['row', 'col']) and all(col in future_df.columns for col in ['row', 'col']) else ['x', 'y']
            
            if join_cols == ['row', 'col']:
                join_df = future_df[['row', 'col', 'established_in_future_period']].drop_duplicates(subset=['row', 'col'])
                df = df.merge(join_df, on=['row', 'col'], how='left')
                df['established_in_future_period'] = df['established_in_future_period'].fillna(0).astype(int)
                has_future_establishments = True
                print(f"    Joined via row/col: {(df['established_in_future_period'] == 1).sum():,} post-test establishments")
            elif join_cols == ['x', 'y']:
                df['x_rounded'] = df['x'].round(6)
                df['y_rounded'] = df['y'].round(6)
                join_df = future_df[['x', 'y', 'established_in_future_period']].copy()
                join_df['x_rounded'] = join_df['x'].round(6)
                join_df['y_rounded'] = join_df['y'].round(6)
                join_df = join_df[['x_rounded', 'y_rounded', 'established_in_future_period']].drop_duplicates(subset=['x_rounded', 'y_rounded'])
                df = df.merge(join_df, on=['x_rounded', 'y_rounded'], how='left')
                df = df.drop(columns=['x_rounded', 'y_rounded'])
                df['established_in_future_period'] = df['established_in_future_period'].fillna(0).astype(int)
                has_future_establishments = True
                print(f"    Joined via x/y: {(df['established_in_future_period'] == 1).sum():,} post-test establishments")
    
    if not has_future_establishments:
        df['established_in_future_period'] = 0
    
    # Define is_observed as the union of test-period AND future-window establishments so
    # the map's recall is identical to the Recall@K metrics (both use the 5-yr horizon).
    # is_observed_test_only captures the 2017-2019 subset for breakdown reporting.
    if has_actual_establishments:
        df['is_observed_test_only'] = (df['established_in_test_period'] == 1).astype(int)
        df['is_observed'] = (
            (df['established_in_test_period'] == 1) | (df['established_in_future_period'] == 1)
        ).astype(int)
        print(f"  ✓ is_observed = test-period ({(df['established_in_test_period']==1).sum():,})"
              f" OR post-test window ({(df['established_in_future_period']==1).sum():,}) establishments")
    else:
        df['is_observed'] = (df['y_true'] == 1).astype(int)
        df['is_observed_test_only'] = df['is_observed']  # No period breakdown in fallback
        print(f"  ⚠ Falling back to y_true (5-year lookahead) - actual establishment data not available")
    
    gdf['is_observed'] = df['is_observed'].values
    gdf['is_observed_test_only'] = df['is_observed_test_only'].values
    
    # Step 2: Filter coordinate outliers BEFORE any analysis (using projected bounds)
    print(f"\n  Filtering coordinate outliers...")
    initial_count = len(gdf)
    proj_bounds = sa_gdf_proj.total_bounds  # (minx, miny, maxx, maxy) in meters
    gdf = gdf[(gdf.geometry.x >= proj_bounds[0]) & (gdf.geometry.x <= proj_bounds[2]) & 
              (gdf.geometry.y >= proj_bounds[1]) & (gdf.geometry.y <= proj_bounds[3])].copy()
    filtered_count = len(gdf)
    if initial_count != filtered_count:
        print(f"    Removed {initial_count - filtered_count:,} outliers ({initial_count:,} → {filtered_count:,})")
    
    # Step 3: Aggregate to PIXEL level (not row/pixel-year level)
    # CRITICAL: Map shows risky LOCATIONS, so aggregate across test years before selecting top-1%
    print(f"\n  Aggregating to pixel level (test years: {test_years})...")
    
    # Extract projected coordinates from filtered gdf (already in EPSG:3857)
    gdf['x_proj'] = gdf.geometry.x
    gdf['y_proj'] = gdf.geometry.y
    
    # For predictions: take MAX probability across test years (most risky prediction for that location)
    # For observations: take MAX across test years (established in test period OR 5-yr window)
    # is_observed_test_only: subset established during test period years only (for breakdown)
    pixel_agg = gdf.groupby(['row', 'col']).agg({
        'y_pred_proba': 'max',          # Highest predicted probability across test years
        'is_observed': 'max',           # 1 if established within 5-yr prediction window
        'is_observed_test_only': 'max', # 1 if established during test period only (breakdown)
        'x_proj': 'first',
        'y_proj': 'first'
    }).reset_index()
    
    print(f"    {len(gdf):,} rows → {len(pixel_agg):,} unique pixels")
    
    # Observed totals — full 5-yr window and test-period-only breakdown
    total_actual_establishments = int(pixel_agg['is_observed'].sum())
    observed_in_test_only = int(pixel_agg['is_observed_test_only'].sum())
    observed_in_future_window = total_actual_establishments - observed_in_test_only
    
    # Step 4: Select exactly top-threshold_pct% pixels using argpartition.
    # Using np.percentile + >= would include all ties at the boundary and over-select
    # (e.g. many RF probability ties could push predicted_count well above threshold_pct%).
    # argpartition gives exactly n_top_pixels; we add tiny random noise to break ties
    # in a spatially uniform way — without noise, argpartition resolves ties in memory
    # order (row/col order), which produces structured horizontal-band artefacts when
    # many pixels share the same floor probability (e.g. after isotonic calibration).
    n_top_pixels = max(1, int(len(pixel_agg) * threshold_pct / 100))
    proba_vals = pixel_agg['y_pred_proba'].values
    rng = np.random.default_rng(42)
    proba_for_selection = proba_vals + rng.uniform(0, 1e-12, len(proba_vals))
    top_pixel_idx = np.argpartition(proba_for_selection, -n_top_pixels)[-n_top_pixels:]
    is_predicted_arr = np.zeros(len(pixel_agg), dtype=np.int8)
    is_predicted_arr[top_pixel_idx] = 1
    pixel_agg = pixel_agg.copy()
    pixel_agg['is_predicted'] = is_predicted_arr
    top_pct_threshold = float(proba_vals[top_pixel_idx].min())

    print(f"  ✓ Top-{threshold_pct}% = {n_top_pixels:,} pixels (pixel-level min score: {top_pct_threshold:.6f})")
    
    # Step 5: Compute statistics from pixel_agg (unique physical pixels by row,col)
    predicted_count = int(pixel_agg['is_predicted'].sum())
    observed_count = int(pixel_agg['is_observed'].sum())
    overlap_count = int(((pixel_agg['is_predicted'] == 1) & (pixel_agg['is_observed'] == 1)).sum())
    predicted_only_count = predicted_count - overlap_count
    observed_only_count = observed_count - overlap_count
    # Overlap breakdown: correctly predicted pixels from test-period vs post-test window
    overlap_test_only = int(((pixel_agg['is_predicted'] == 1) & (pixel_agg['is_observed_test_only'] == 1)).sum())
    overlap_future_window = overlap_count - overlap_test_only

    recall_pct = (overlap_count / total_actual_establishments * 100) if total_actual_establishments > 0 else 0.0

    print(f"\n  FINAL STATISTICS (pixel-level, after filtering):")
    print(f"    Total pixels: {len(pixel_agg):,}")
    print(f"    Predicted high-risk (top {threshold_pct}%): {predicted_count:,} "
          f"({predicted_count / len(pixel_agg) * 100:.2f}% of pixels)")
    print(f"    PA establishments (5-yr window): {observed_count:,} "
          f"({observed_count / len(pixel_agg) * 100:.3f}% of pixels)")
    print(f"      of which test period ({time_period}): {observed_in_test_only:,}")
    if has_future_establishments and observed_in_future_window > 0:
        future_year_str = f"{min(future_years)}-{max(future_years)}"
        print(f"      of which post-test window ({future_year_str}): {observed_in_future_window:,}")
    print(f"    Overlap (correctly predicted): {overlap_count:,}")
    print(f"      of which test period: {overlap_test_only:,} | post-test window: {overlap_future_window:,}")
    if observed_count > 0:
        print(f"    NOTE: With only {observed_count:,} establishments out of "
              f"{len(pixel_agg):,} pixels, top-{threshold_pct}% selects "
              f"{predicted_count:,} pixels — high recall is legitimate if "
              f"designations are geographically concentrated.")
    if predicted_count > 0:
        hit_rate = overlap_count / predicted_count * 100
        print(f"    Hit rate (precision): {overlap_count:,}/{predicted_count:,} = {hit_rate:.1f}%")
    else:
        hit_rate = 0.0
        print(f"    Hit rate: N/A (no predictions)")
    print(f"    Recall (5-yr window): {overlap_count:,}/{total_actual_establishments:,} = {recall_pct:.1f}%")

    # Step 6: Build category raster (pixel-perfect, same resolution as probability map)
    # Category: 0=background, 1=predicted_only, 2=observed_only, 3=overlap
    # Priority: overlap (3) > observed (2) > predicted (1) > background (0)
    # Future-window establishments are now part of is_observed, so no separate category needed.
    category_codes = np.where(
        (pixel_agg['is_predicted'] == 1) & (pixel_agg['is_observed'] == 1), 3,  # Correctly predicted (green)
        np.where(pixel_agg['is_observed'] == 1, 2,                               # Established, not predicted (orange)
                 np.where(pixel_agg['is_predicted'] == 1, 1, 0)))                 # Predicted only (blue) or background
    
    print(f"\n  Creating rasterized risk map...")
    category_raster, extent = points_to_raster(
        pixel_agg['x_proj'].values,
        pixel_agg['y_proj'].values,
        category_codes.astype(np.float32),
        target_resolution=1000.0,  # 1000 meters resolution
        agg_func='max'
    )
    print(f"  Category raster shape: {category_raster.shape}, extent: {extent}")

    # Dilate colored pixels for print visibility (top 1% is very sparse across a continent).
    # Dilation is purely cosmetic — caption notes pixel size is enlarged for legibility.
    # Use a slightly larger radius for the 1% map where dots are fewest.
    from scipy.ndimage import binary_dilation
    dilation_radius = 3 if threshold_pct <= 1.0 else 2
    struct = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
    for cat_val in [1, 2, 3]:
        mask = category_raster == cat_val
        if mask.any():
            dilated = binary_dilation(mask, structure=struct)
            # Only paint into cells that are NaN (background) or same category — don't overwrite higher-priority cats
            paintable = np.isnan(category_raster) | (category_raster == cat_val)
            category_raster = np.where(dilated & paintable, cat_val, category_raster)
    print(f"  Applied {dilation_radius}-pixel dilation for print visibility")

    # Create background mask from all data points (shows where data exists; protected areas = holes)
    print(f"  Creating background mask from data coverage...")
    background_mask, _ = points_to_raster(
        pixel_agg['x_proj'].values,
        pixel_agg['y_proj'].values,
        np.ones(len(pixel_agg), dtype=np.float32),  # All pixels = 1
        target_resolution=1000.0,
        agg_func='max'
    )
    print(f"  Background mask created - protected areas will appear as natural holes")

    # Calculate figure dimensions dynamically from projected bounds
    proj_bounds = sa_gdf_proj.total_bounds  # (minx, miny, maxx, maxy) in meters
    proj_width = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    proj_aspect = proj_height / proj_width
    
    fig_width = 14
    fig_height = fig_width * proj_aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Base layer: gray fill so existing PA holes (pixels absent from the scored parquet
    # = already protected before the test period) appear gray instead of white ocean.
    # Prefer the backbone raster (pixel-perfect coastal alignment); fall back to the
    # Natural Earth polygon if backbone is unavailable.
    used_backbone = _plot_backbone_background(ax, backbone_path, zorder=0)
    if not used_backbone:
        sa_gdf_proj.plot(ax=ax, color=PA_HOLE_COLOR, edgecolor='#808080', linewidth=0.4, zorder=0)

    # Plot background mask from actual data (shows data coverage; protected areas = natural holes)
    background_masked = np.ma.masked_where(np.isnan(background_mask), background_mask)
    ax.imshow(background_masked, extent=extent, cmap=ListedColormap(['#EBEBEB']),
              vmin=0, vmax=1, origin='upper', interpolation='nearest',
              aspect='equal', zorder=1)

    # Three-category colormap — all colors are region-configurable (see config.py).
    # SA palette: gold=predicted-only, blue=hit, red=miss (intuitive: blue=correct, gold=unconfirmed)
    COLOR_PREDICTED_ONLY = RISK_MAP_PREDICTED_COLOR  # SA: gold; others: blue
    COLOR_OBSERVED_ONLY  = RISK_MAP_MISS_COLOR        # SA: pomegranate red; others: orange
    COLOR_OVERLAP        = RISK_MAP_OVERLAP_COLOR     # SA: slate blue; others: green

    cmap_categories = ListedColormap([COLOR_PREDICTED_ONLY, COLOR_OBSERVED_ONLY, COLOR_OVERLAP])
    norm_categories = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap_categories.N)
    max_category = 3

    # Category raster - pixel-perfect resolution
    cat_no_bg = np.where((category_raster >= 1) & (category_raster <= max_category), category_raster, np.nan)
    cat_masked = np.ma.masked_where(np.isnan(cat_no_bg), cat_no_bg)
    ax.imshow(cat_masked, extent=extent, cmap=cmap_categories, norm=norm_categories,
              origin='upper', interpolation='nearest', aspect='equal', zorder=2)
    
    # Set axis limits from projected bounds
    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect('equal', adjustable='box')

    # Country borders — barely-visible overlay for geographic orientation (Natural Earth 110m)
    sa_gdf_proj.plot(ax=ax, facecolor='none', edgecolor='#777777', linewidth=0.25, alpha=0.4, zorder=4)

    # Styling — replace raw EPSG:3857 metre labels with lat/lon degree labels
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))

    # Legend — true positives first, then unconfirmed predictions, then misses, then existing PAs.
    # Slightly downscale relative to the global map legend config to match the visual
    # balance of the forward "overlap" maps (which are typically less dense visually).
    legend_fontsize = max(MAP_LEGEND_FONTSIZE_CFG - 2, 6)
    legend_markersize = max(MAP_LEGEND_MARKERSIZE_CFG - 2, 4)
    legend_elements = []
    if overlap_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_OVERLAP,
                                      markersize=legend_markersize, alpha=0.9,
                                      label=f'Correctly predicted — PA established (n={overlap_count:,})'))
    if predicted_only_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_PREDICTED_ONLY,
                                      markersize=legend_markersize, alpha=0.9,
                                      label=f'Predicted high-risk, no PA established (n={predicted_only_count:,})'))
    if observed_only_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_OBSERVED_ONLY,
                                      markersize=legend_markersize, alpha=0.9,
                                      label=f'PA established, not predicted (n={observed_only_count:,})'))
    legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=PA_HOLE_COLOR,
                                  markersize=legend_markersize, alpha=0.9,
                                  label=f'Existing PAs (pre-{min(test_years)})'))
    # Consistent evaluation-map layout across regions:
    # legend bottom-right, north arrow top-left, scale bar top-right.
    legend_loc = 'lower right'
    if legend_elements:
        ax.legend(handles=legend_elements, loc=legend_loc, fontsize=legend_fontsize,
                  framealpha=0.95, borderpad=0.8, labelspacing=0.6)

    # Scale bar + north arrow
    # For South America, use: north arrow (top-left) + scale bar (top-right).
    if MAP_INSET_SIDE == 'right':
        _add_scale_bar_3857(
            ax,
            proj_bounds,
            ref_lat_deg=(Y_LIMITS[0] + Y_LIMITS[1]) / 2.0,
            bar_km=1000,
            position='upper right',
            anchor_x_frac=0.94,
            anchor_y_frac=0.915,
        )
        _add_north_arrow(ax, proj_bounds, position='upper left')
    else:
        _add_scale_bar_3857(
            ax,
            proj_bounds,
            ref_lat_deg=(Y_LIMITS[0] + Y_LIMITS[1]) / 2.0,
            bar_km=1000,
            position='upper right',
            anchor_x_frac=0.94,
            anchor_y_frac=0.915,
        )
        _add_north_arrow(ax, proj_bounds, position='upper left')

    plt.tight_layout(pad=0.5)

    # Save PDF with low DPI to avoid timeout/zlib/backend errors on large rasters
    pdf_path = output_dir / f"risk_map_top{int(threshold_pct)}pct.pdf"
    for pdf_dpi in (150, 100):
        try:
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            plt.savefig(pdf_path, dpi=pdf_dpi, bbox_inches='tight')
            print(f"✓ Saved simplified risk map (PDF): {pdf_path} (dpi={pdf_dpi})")
            break
        except (TimeoutError, zlib.error, OSError, AttributeError, RuntimeError) as e:
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
            if pdf_dpi == 100:
                print(f"Warning: Could not save risk map PDF after retries: {e}. Skipping PDF.")
            continue
    plt.close()
    print(f"✓ Simplified risk map complete!")

    # Return capture stats for capture_summary.csv and metrics table.
    # total_pa_pixels / recall_pct use the full 5-yr window, matching Recall@K in the
    # benchmark JSON. The _test_only and _future_window breakdown fields are extra context
    # for paper text ("of which X were established during 2017-2019, Y in 2020-2024").
    px_km2 = (pixel_size_m / 1000.0) ** 2  # km² per pixel (1.0 for 1-km raster)
    stats: Dict[str, Any] = {
        "threshold_pct": threshold_pct,
        "total_pa_pixels": total_actual_establishments,
        "total_pa_km2": round(total_actual_establishments * px_km2, 0),
        "captured_pixels": overlap_count,
        "captured_km2": round(overlap_count * px_km2, 0),
        "recall_pct": round(recall_pct, 2),
        "precision_pct": round(hit_rate, 2) if predicted_count > 0 else 0.0,
        "observed_in_test_only": observed_in_test_only,
        "observed_in_future_window": observed_in_future_window,
    }
    return stats


def create_p1pct_diagnostic_map(
    df: pd.DataFrame,
    region_boundary_path: Optional[Path],
    output_dir: Path,
    model_type: str,
    metrics_data: Optional[Dict[str, Any]] = None,
    test_years: Optional[list] = None,
    sa_gdf: Optional[gpd.GeoDataFrame] = None,
    backbone_path: Optional[Path] = None,
) -> None:
    """Create diagnostic map showing row-level P@1% metric visualization.
    
    This map uses the EXACT same logic as the P@1% metric computation:
    - Selects global top 1% rows (pixel-years) by probability
    - Marks overlap only when y_true == 1 in the SAME row
    - No pixel-level deduplication, no cross-year aggregation, no WDPA OR-logic
    
    This diagnostic helps visualize why P@1% metrics may differ from pixel-level risk maps.
    """
    print("\n" + "=" * 70)
    print("CREATING ROW-LEVEL P@1% DIAGNOSTIC MAP")
    print("=" * 70)
    
    # Get region boundary (use cached if provided)
    if sa_gdf is None:
        sa_gdf = get_region_boundary(region_boundary_path)

    # Ensure CRS is set (assume EPSG:4326 if not set)
    if sa_gdf.crs is None:
        sa_gdf.set_crs('EPSG:4326', inplace=True)

    print(f"  {REGION_LABEL} boundary CRS: {sa_gdf.crs}")

    # Reproject to EPSG:3857 (Web Mercator) for accurate visualization
    sa_gdf_proj = sa_gdf.to_crs('EPSG:3857')
    print(f"  Reprojected to EPSG:3857 (Web Mercator)")
    print(f"  Projected bounds (meters): {sa_gdf_proj.total_bounds}")

    # Create GeoDataFrame from predictions (keep all rows, no deduplication)
    geometry = gpd.points_from_xy(df['x'], df['y'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # Reproject predictions to EPSG:3857
    gdf = gdf.to_crs('EPSG:3857')

    print(f"  Total rows (pixel-years): {len(gdf):,}")

    # Compute top 1% threshold using EXACT same logic as compute_precision_at_k
    y_proba = df['y_pred_proba'].values
    n_top_k = max(1, int(len(y_proba) * 1 / 100))
    print(f"  Top 1% count: {n_top_k:,} rows")
    
    # Use argpartition to select exactly n_top_k rows.
    # Do NOT back-convert to a threshold and re-apply >=: ties at the boundary
    # would expand the selection beyond n_top_k and make overlap_count inconsistent
    # with the actual P@1% metric value printed in the same block.
    top_k_idx = np.argpartition(y_proba, -n_top_k)[-n_top_k:]
    top_k_threshold = float(np.min(y_proba[top_k_idx]))
    print(f"  Top 1% min score: {top_k_threshold:.6f}")
    top_k_mask = np.zeros(len(y_proba), dtype=bool)
    top_k_mask[top_k_idx] = True

    # Mark rows as top-1% (row-level, no aggregation, exactly n_top_k rows)
    gdf['is_top1pct'] = top_k_mask
    
    # Mark rows as positive using ONLY y_true (no WDPA OR-logic, no cross-year)
    gdf['is_positive'] = (df['y_true'] == 1)
    
    # Create categories at row-level (overlap only when both true in SAME row)
    gdf['category'] = 'none'
    gdf.loc[gdf['is_top1pct'] & ~gdf['is_positive'], 'category'] = 'predicted_only'
    gdf.loc[~gdf['is_top1pct'] & gdf['is_positive'], 'category'] = 'observed_only'
    gdf.loc[gdf['is_top1pct'] & gdf['is_positive'], 'category'] = 'overlap'
    
    # Count rows in each category
    predicted_only_count = (gdf['category'] == 'predicted_only').sum()
    observed_only_count = (gdf['category'] == 'observed_only').sum()
    overlap_count = (gdf['category'] == 'overlap').sum()
    
    # Compute actual P@1% metric for verification
    y_true = df['y_true'].values
    actual_p1pct = y_true[top_k_idx].sum() / n_top_k
    total_pos = overlap_count + observed_only_count
    print(f"  Row-level P@1% metric: {actual_p1pct:.4f} ({overlap_count:,}/{n_top_k:,})")
    print(f"  Total positive rows (y_true=1): {total_pos:,}")
    print(f"    Captured by top 1% (overlap): {overlap_count:,}")
    print(f"    Missed by top 1% (y_true=1, outside top 1%): {observed_only_count:,}")
    print(f"  Predicted high-risk only (y_true=0, in top 1%): {predicted_only_count:,}")
    
    # Create categorical raster
    category_codes = {
        'none': 0,
        'predicted_only': 1,
        'observed_only': 2,
        'overlap': 3
    }
    gdf['category_code'] = gdf['category'].map(category_codes)
    
    # Extract time period information, preferring explicitly provided test_years
    time_period = None
    prediction_horizon = None
    test_years_list: Optional[list] = None

    if test_years is not None and len(test_years) > 0:
        test_years_list = sorted(int(y) for y in test_years)
        time_period = f"{min(test_years_list)}-{max(test_years_list)}"

    lookahead = 5
    if metrics_data:
        metadata = metrics_data.get("metadata", {})
        lookahead = metadata.get("lookahead_years", lookahead)
        test_year_range = metadata.get("test_year_range")
        if test_year_range and isinstance(test_year_range, list) and len(test_year_range) == 2:
            if time_period is None:
                time_period = f"{test_year_range[0]}-{test_year_range[1]}"
        elif "test_years" in metadata:
            meta_test_years = metadata["test_years"]
            if (
                time_period is None
                and isinstance(meta_test_years, (list, tuple))
                and len(meta_test_years) >= 2
            ):
                time_period = f"{meta_test_years[0]}-{meta_test_years[-1]}"

        if isinstance(lookahead, int) and test_year_range:
            pred_end = test_year_range[1] + lookahead
            prediction_horizon = f"{lookahead}-year lookahead ({test_year_range[0]}-{pred_end})"

    # Fallbacks if metadata not sufficient
    if time_period is None:
        if test_years_list is not None:
            time_period = f"{min(test_years_list)}-{max(test_years_list)}"
        else:
            time_period = "2018-2019"

    if prediction_horizon is None:
        if test_years_list is not None and isinstance(lookahead, int):
            pred_end = max(test_years_list) + lookahead
            prediction_horizon = f"{lookahead}-year lookahead ({min(test_years_list)}-{pred_end})"
        else:
            prediction_horizon = "5-year lookahead (2018-2024)"
    
    # Convert to raster for visualization (using projected coordinates)
    print(f"  Converting {len(gdf):,} points to raster grid...")
    category_raster, extent = points_to_raster(
        gdf.geometry.x.values,
        gdf.geometry.y.values,
        gdf['category_code'].values,
        target_resolution=1000.0,  # 1000 meters resolution
        agg_func='max'  # Categorical: overlap (3) wins when multiple categories in same cell
    )
    print(f"  Category raster shape: {category_raster.shape}, extent: {extent}")
    print(f"  Valid (non-NaN) pixels: {np.sum(~np.isnan(category_raster)):,}")
    
    # Create background mask from all data points (shows where data exists; protected areas = holes)
    print(f"  Creating background mask from data coverage...")
    background_mask, _ = points_to_raster(
        gdf.geometry.x.values,
        gdf.geometry.y.values,
        np.ones(len(gdf), dtype=np.float32),  # All points = 1
        target_resolution=1000.0,
        agg_func='max'
    )
    print(f"  Background mask created - protected areas will appear as natural holes")

    # Calculate figure dimensions dynamically from projected bounds
    proj_bounds = sa_gdf_proj.total_bounds  # (minx, miny, maxx, maxy) in meters
    proj_width = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    proj_aspect = proj_height / proj_width
    
    fig_width = 14
    fig_height = fig_width * proj_aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Base layer: gray fill for PA holes — prefer backbone raster, fall back to polygon.
    used_backbone = _plot_backbone_background(ax, backbone_path, zorder=0)
    if not used_backbone:
        sa_gdf_proj.plot(ax=ax, color=PA_HOLE_COLOR, edgecolor='#808080', linewidth=0.4, zorder=0)

    # Plot background mask from actual data (shows data coverage; protected areas = natural holes)
    background_masked = np.ma.masked_where(np.isnan(background_mask), background_mask)
    ax.imshow(background_masked, extent=extent, cmap=ListedColormap(['#F5F5F5']),
              vmin=0, vmax=1, origin='upper', interpolation='nearest',
              aspect='equal', zorder=1)
    
    # Define colors for each category
    category_colors_rgba = {
        0: RISK_MAP_COLORS['none'],
        1: RISK_MAP_COLORS['predicted_only'],
        2: RISK_MAP_COLORS['observed_only'],
        3: RISK_MAP_COLORS['overlap']
    }
    
    def rgba_to_hex(rgba):
        """Convert RGBA tuple to hex color string."""
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    category_colors_hex = {
        0: rgba_to_hex(category_colors_rgba[0]),
        1: rgba_to_hex(category_colors_rgba[1]),
        2: rgba_to_hex(category_colors_rgba[2]),
        3: rgba_to_hex(category_colors_rgba[3])
    }
    
    # Create colormap and normalization
    colors_list_no_bg = [category_colors_rgba[i] for i in range(1, 4)]
    cmap_cat = ListedColormap(colors_list_no_bg)
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm_cat = BoundaryNorm(bounds, cmap_cat.N)
    
    # Plot main category raster
    category_raster_no_bg = np.where(category_raster == 0, np.nan, category_raster)
    category_raster_masked = np.ma.masked_where(np.isnan(category_raster_no_bg), category_raster_no_bg)
    
    im = ax.imshow(category_raster_masked,
                   extent=extent,
                   cmap=cmap_cat,
                   norm=norm_cat,
                   origin='upper',
                   interpolation='nearest',
                   aspect='equal',
                   zorder=2)
    
    # Set axis limits from projected bounds
    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect('equal', adjustable='box')

    # Styling — replace raw EPSG:3857 metre labels with lat/lon degree labels
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) Row-Level P@1% Diagnostic Map',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=15)
    
    # Add description text box
    description_text = (f"Why P@1% differs from the risk map:\n"
                       f"Training metric uses row-level logic —\n"
                       f"each pixel-year is a separate row.\n"
                       f"A pixel protected in 2019 appears in\n"
                       f"test years 2017 & 2018 as y_true=1.\n"
                       f"Many 'misses' are temporal leads:\n"
                       f"the model correctly scores the pixel\n"
                       f"high, but the match falls in a\n"
                       f"different test year's row.\n"
                       f"P@1% (row) = {actual_p1pct:.4f} ({overlap_count:,}/{n_top_k:,})")
    ax.text(0.98, 0.98, description_text, transform=ax.transAxes, fontsize=FONTSIZE_DESCRIPTION,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', 
           alpha=0.9, edgecolor='black', linewidth=1, pad=0.5), zorder=10)
    
    # Create legend
    legend_elements = []
    if overlap_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor=category_colors_hex[3], markersize=10, 
                                    alpha=1.0, label=f'Overlap: predicted + observed (n={overlap_count:,} rows)'))
    if predicted_only_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor=category_colors_hex[1], markersize=10, 
                                    alpha=1.0, label=f'Predicted high-risk only (n={predicted_only_count:,} rows)'))
    if observed_only_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                    markerfacecolor=category_colors_hex[2], markersize=10,
                                    alpha=1.0, label=f'Missed: y_true=1, outside top 1% (n={observed_only_count:,} rows)'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=FONTSIZE_LEGEND, 
              framealpha=0.95, fancybox=True, shadow=True)
    
    # Add time period text box
    time_text = f"Observation period: {time_period}\nPrediction horizon: {prediction_horizon}\nRow-level (no pixel deduplication)"
    ax.text(0.02, 0.02, time_text, transform=ax.transAxes, fontsize=FONTSIZE_STATS,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', 
           alpha=0.8, edgecolor='black', linewidth=1), zorder=10)
    
    plt.tight_layout(pad=0.5)

    # Save PDF (use low DPI first - diagnostic map has very large raster, avoids timeout/backend errors)
    pdf_path = output_dir / f"p1pct_diagnostic.pdf"
    for pdf_dpi in (150, 100):
        try:
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            plt.savefig(pdf_path, dpi=pdf_dpi, bbox_inches='tight')
            print(f"Saved PDF: {pdf_path} (dpi={pdf_dpi})")
            break
        except (TimeoutError, zlib.error, OSError, AttributeError, RuntimeError) as e:
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
            if pdf_dpi == 100:
                print(f"Warning: Could not save PDF after retries: {e}. Skipping PDF.")
            continue
    plt.close()


def create_probability_map(
    df: pd.DataFrame,
    region_boundary_path: Optional[Path],
    output_dir: Path,
    model_type: str,
    parquet_path: Path,
    metrics_data: Optional[Dict[str, Any]] = None,
    test_years: Optional[Sequence[int]] = None,
    test_parquet_path: Optional[Path] = None,
    sa_gdf: Optional[gpd.GeoDataFrame] = None,
    backbone_path: Optional[Path] = None,
) -> None:
    """Create continuous probability map showing predicted probabilities across South America.
    
    Uses percentile-based color scaling to make low probabilities visible.
    Colors every pixel by y_pred_proba (uses calibrated probabilities if available).

    Args:
        df: DataFrame with predictions.
        region_boundary_path: Path to South America boundary.
        output_dir: Output directory for maps.
        model_type: Model type (rf, lgbm, brf).
        parquet_path: Path to scored parquet file (for calibration status, etc.).
        metrics_data: Optional metrics/metadata dictionary.
        test_years: Explicit list of test years. If not provided, years will be
            derived dynamically from parquet/metadata using `derive_test_years`.
        test_parquet_path: Optional path to original test parquet (e.g. *test_win5.parquet*),
            used when deriving `test_years` if they are not passed explicitly.
    """
    print("\n" + "=" * 70)
    print("CREATING PROBABILITY MAP")
    print("=" * 70)
    
    # Get region boundary (use cached if provided)
    if sa_gdf is None:
        sa_gdf = get_region_boundary(region_boundary_path)

    # Ensure CRS is set (assume EPSG:4326 if not set)
    if sa_gdf.crs is None:
        sa_gdf.set_crs('EPSG:4326', inplace=True)

    print(f"  {REGION_LABEL} boundary CRS: {sa_gdf.crs}")

    # Reproject to EPSG:3857 (Web Mercator) for accurate visualization
    sa_gdf_proj = sa_gdf.to_crs('EPSG:3857')
    print(f"  Reprojected to EPSG:3857 (Web Mercator)")
    print(f"  Projected bounds (meters): {sa_gdf_proj.total_bounds}")

    # Create GeoDataFrame from predictions
    geometry = gpd.points_from_xy(df['x'], df['y'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # Reproject predictions to EPSG:3857
    gdf = gdf.to_crs('EPSG:3857')

    print(f"  Prediction points: {len(gdf):,}")

    # Get probability values
    proba_values = df['y_pred_proba'].values
    
    # Calculate statistics
    proba_min = proba_values.min()
    proba_max = proba_values.max()
    proba_mean = proba_values.mean()
    proba_median = np.median(proba_values)
    proba_std = proba_values.std()
    
    print(f"  Probability range: [{proba_min:.6f}, {proba_max:.6f}]")
    print(f"  Probability mean: {proba_mean:.6f}, median: {proba_median:.6f}, std: {proba_std:.6f}")
    
    # Extract time period information dynamically
    # Prefer explicitly provided test years; if not available, derive them from parquet/metadata
    if test_years is not None and len(test_years) > 0:
        test_years_list = sorted(int(y) for y in test_years)
    else:
        # When not explicitly provided, allow derive_test_years to inspect
        # scored parquet and/or original test parquet (if given)
        test_years_list = derive_test_years(metrics_data, parquet_path, test_parquet_path)
    time_period = f"{min(test_years_list)}-{max(test_years_list)}"
    
    # Derive prediction horizon from metadata
    prediction_horizon = "5-year lookahead"
    if metrics_data:
        metadata = metrics_data.get("metadata", {})
        lookahead = metadata.get("lookahead_years", 5)
        test_year_range = metadata.get("test_year_range")
        if test_year_range and isinstance(test_year_range, list) and len(test_year_range) == 2:
            pred_end = test_year_range[1] + lookahead
            prediction_horizon = f"{lookahead}-year lookahead ({test_year_range[0]}-{pred_end})"
        elif isinstance(lookahead, int):
            pred_end = max(test_years_list) + lookahead
            prediction_horizon = f"{lookahead}-year lookahead ({min(test_years_list)}-{pred_end})"
    
    # ---------------------------------------------------------------------------
    # Normalise probabilities to [0, 1] for the colourmap.
    # The method is region-specific (set in config.py via PROBABILITY_MAP_TRANSFORMATION):
    #   'sqrt'  (SA)  — percentile clip + square-root stretch; works well when the
    #                    distribution is moderately skewed.
    #   'log'   (USA) — log10 transform applied before percentile clipping; needed
    #                    when isotonic calibration collapses >98 % of values to the
    #                    same floor, making any linear/sqrt approach produce a flat map.
    # ---------------------------------------------------------------------------
    def _normalise(vals: np.ndarray, log_pmin: float = 0.0, log_pmax: float = 1.0,
                   proba_pmin: float = 0.0, proba_span: float = 1.0) -> np.ndarray:
        """Apply the region-appropriate normalisation to a probability array."""
        if PROBABILITY_MAP_TRANSFORMATION == 'log':
            log_v = np.log10(np.maximum(vals, 1e-10))
            span = log_pmax - log_pmin
            if span > 1e-6:
                n = np.clip((log_v - log_pmin) / span, 0.0, 1.0)
            else:
                n = np.full_like(log_v, 0.5)
        else:
            clipped = np.clip(vals, proba_pmin, proba_pmin + proba_span)
            shifted = clipped - proba_pmin
            if proba_span > 1e-12:
                linear = shifted / proba_span
            else:
                n = np.full_like(shifted, 0.5)
                # Apply display gamma below
                if abs(PROBABILITY_MAP_DISPLAY_GAMMA - 1.0) > 1e-9:
                    n = np.clip(n ** PROBABILITY_MAP_DISPLAY_GAMMA, 0.0, 1.0)
                return n
            if PROBABILITY_MAP_TRANSFORMATION == 'sqrt':
                n = np.sqrt(linear)
            else:
                n = linear  # linear fallback

        # Optional display gamma (forward uses this to reserve bright colours for tails)
        if abs(PROBABILITY_MAP_DISPLAY_GAMMA - 1.0) > 1e-9:
            n = np.clip(n ** PROBABILITY_MAP_DISPLAY_GAMMA, 0.0, 1.0)
        return n

    # Compute normalisation parameters from the FULL dataset (before coordinate filtering)
    if PROBABILITY_MAP_TRANSFORMATION == 'log':
        log_proba_full = np.log10(np.maximum(proba_values, 1e-10))
        log_pmin = float(np.percentile(log_proba_full, PROBABILITY_MAP_PERCENTILE_MIN))
        log_pmax = float(np.percentile(log_proba_full, PROBABILITY_MAP_PERCENTILE_MAX))
        proba_pmin_param = 0.0
        proba_span_param = 1.0
        transform_name = 'log10'
        print(f"  Using log10 colour range ({PROBABILITY_MAP_PERCENTILE_MIN}th–{PROBABILITY_MAP_PERCENTILE_MAX}th percentile of log values)")
        print(f"  log10 range: [{log_pmin:.4f}, {log_pmax:.4f}]  "
              f"(raw prob: [{10**log_pmin:.6f}, {10**log_pmax:.6f}])")
    else:
        log_pmin = 0.0
        log_pmax = 1.0
        proba_pmin_param = float(np.percentile(proba_values, PROBABILITY_MAP_PERCENTILE_MIN))
        proba_pmax_param = float(np.percentile(proba_values, PROBABILITY_MAP_PERCENTILE_MAX))
        proba_span_param = proba_pmax_param - proba_pmin_param
        transform_name = 'square root' if PROBABILITY_MAP_TRANSFORMATION == 'sqrt' else 'linear'
        print(f"  Using percentile-based colour range ({PROBABILITY_MAP_PERCENTILE_MIN}th–{PROBABILITY_MAP_PERCENTILE_MAX}th percentile)")
        print(f"  Colour range: [{proba_pmin_param:.6f}, {proba_pmax_param:.6f}]  span: {proba_span_param:.6f}")
        if proba_span_param <= 1e-12:
            print("  Warning: probability range is effectively zero; colour map will be flat.")

    proba_normalized = _normalise(proba_values, log_pmin, log_pmax,
                                   proba_pmin_param, proba_span_param)
    print(f"  Applied {transform_name} transformation to stretch distribution")
    print(f"  Normalised value range: [{proba_normalized.min():.4f}, {proba_normalized.max():.4f}]")
    print(f"  Normalised value mean: {proba_normalized.mean():.4f}, median: {np.median(proba_normalized):.4f}")

    # Filter coordinate outliers before rasterization (using projected bounds)
    print(f"  Filtering coordinate outliers...")
    initial_count = len(gdf)
    proj_bounds = sa_gdf_proj.total_bounds  # (minx, miny, maxx, maxy) in meters
    gdf = gdf[(gdf.geometry.x >= proj_bounds[0]) & (gdf.geometry.x <= proj_bounds[2]) &
              (gdf.geometry.y >= proj_bounds[1]) & (gdf.geometry.y <= proj_bounds[3])].copy()
    filtered_count = len(gdf)
    if initial_count != filtered_count:
        print(f"  Filtered out {initial_count - filtered_count:,} coordinate outliers ({initial_count:,} -> {filtered_count:,})")

    # Update df to match filtered gdf (keep in sync)
    df = df[df.index.isin(gdf.index)].copy()

    # Recompute proba_normalized from filtered data using the SAME parameters derived above
    proba_values_filtered = df['y_pred_proba'].values
    proba_normalized = _normalise(proba_values_filtered, log_pmin, log_pmax,
                                   proba_pmin_param, proba_span_param)
    
    # Convert points to raster for pixel-perfect visualization (using projected coordinates)
    print(f"  Converting {len(gdf):,} points to raster grid...")
    raster, extent = points_to_raster(
        gdf.geometry.x.values, 
        gdf.geometry.y.values, 
        proba_normalized,  # Use normalized values to ensure full colormap usage
        target_resolution=1000.0  # 1000 meters resolution
    )
    print(f"  Raster shape: {raster.shape}, extent: {extent}")
    print(f"  Valid (non-NaN) pixels: {np.sum(~np.isnan(raster)):,}")

    # Calculate figure dimensions dynamically from projected bounds
    proj_bounds = sa_gdf_proj.total_bounds  # (minx, miny, maxx, maxy) in meters
    proj_width = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    proj_aspect = proj_height / proj_width
    
    fig_width = 14
    fig_height = fig_width * proj_aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Base layer: gray fill for PA holes — prefer backbone raster, fall back to polygon.
    used_backbone = _plot_backbone_background(ax, backbone_path, zorder=1)
    if not used_backbone:
        sa_gdf_proj.plot(ax=ax, color=PA_HOLE_COLOR, edgecolor='#808080', linewidth=0.4, zorder=1)
    
    # Plot probability raster with pixel-perfect rendering
    # Use standardized colormap for consistent visualization across all models
    # interpolation='nearest' ensures no smoothing - pixel-perfect rendering
    # Values are normalized to [0, 1] so full colormap is used
    # aspect='equal' ensures pixels align perfectly with geographic boundary
    im = ax.imshow(raster,
                   extent=extent,
                   cmap=PROBABILITY_MAP_COLORMAP,
                   vmin=0, vmax=1,  # Normalized values, so use [0, 1] range
                   origin='upper',  # Match geographic convention (north at top)
                   interpolation='nearest',  # No smoothing - pixel perfect
                   aspect='equal',
                   zorder=2)
    
    # Set axis limits from projected bounds
    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect('equal', adjustable='box')

    # Country borders — barely-visible overlay for geographic orientation (Natural Earth 110m)
    sa_gdf_proj.plot(ax=ax, facecolor='none', edgecolor='#777777', linewidth=0.25, alpha=0.4, zorder=4)

    # Colorbar with actual probability values back-computed from normalised positions
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('5-year designation probability',
                   fontsize=FONTSIZE_LABEL, rotation=270, labelpad=15)

    # Five ticks evenly spaced in normalised [0,1] space; labels show real probability as %
    tick_positions_norm = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_labels_prob = []
    for t in tick_positions_norm:
        if PROBABILITY_MAP_TRANSFORMATION == 'log':
            actual = 10 ** (log_pmin + t * (log_pmax - log_pmin))
        elif PROBABILITY_MAP_TRANSFORMATION == 'sqrt':
            actual = proba_pmin_param + (t ** 2) * proba_span_param
        else:
            actual = proba_pmin_param + t * proba_span_param
        pct = actual * 100
        if pct < 0.01:
            tick_labels_prob.append(f'{pct:.3f}%')
        elif pct < 0.1:
            tick_labels_prob.append(f'{pct:.2f}%')
        elif pct < 1.0:
            tick_labels_prob.append(f'{pct:.1f}%')
        else:
            tick_labels_prob.append(f'{pct:.0f}%')
    # Prefix the top tick with "≥" to indicate the colorbar is clipped at 98th percentile
    tick_labels_prob[-1] = f'≥{tick_labels_prob[-1]}'
    cbar.set_ticks(tick_positions_norm)
    cbar.set_ticklabels(tick_labels_prob)
    cbar.ax.tick_params(labelsize=FONTSIZE_LEGEND)
    
    # Styling — replace raw EPSG:3857 metre labels with lat/lon degree labels
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))

    # Legend — consistent evaluation-map layout across regions (bottom-right).
    # Keep legend slightly smaller for consistency with forward overlap maps.
    legend_fontsize = max(MAP_LEGEND_FONTSIZE_CFG - 2, 6)
    prob_legend_loc = 'lower right'
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PA_HOLE_COLOR, edgecolor='none',
                             label=f'Existing PAs (pre-{min(test_years_list)})')],
              loc=prob_legend_loc, fontsize=legend_fontsize,
              framealpha=0.95, borderpad=0.8)

    # Scale bar + north arrow
    # For South America, use: north arrow (top-left) + scale bar (top-right).
    if MAP_INSET_SIDE == 'right':
        _add_scale_bar_3857(
            ax,
            proj_bounds,
            ref_lat_deg=(Y_LIMITS[0] + Y_LIMITS[1]) / 2.0,
            bar_km=1000,
            position='upper right',
            anchor_x_frac=0.94,
            anchor_y_frac=0.915,
        )
        _add_north_arrow(ax, proj_bounds, position='upper left')
    else:
        _add_scale_bar_3857(
            ax,
            proj_bounds,
            ref_lat_deg=(Y_LIMITS[0] + Y_LIMITS[1]) / 2.0,
            bar_km=1000,
            position='upper right',
            anchor_x_frac=0.94,
            anchor_y_frac=0.915,
        )
        _add_north_arrow(ax, proj_bounds, position='upper left')

    plt.tight_layout(pad=0.5)

    # Save PDF (lower DPI + retry to avoid timeout/zlib/AttributeError on large raster)
    pdf_path = output_dir / f"probability_map.pdf"
    png_path = output_dir / f"probability_map.png"
    saved = False
    for pdf_dpi in (150, 100):
        try:
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            plt.savefig(pdf_path, dpi=pdf_dpi, bbox_inches='tight')
            print(f"✓ Saved probability map (PDF): {pdf_path} (dpi={pdf_dpi})")
            saved = True
            break
        except (TimeoutError, zlib.error, OSError, AttributeError, RuntimeError) as e:
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
            print(f"  PDF save failed at dpi={pdf_dpi}: {type(e).__name__}: {e}")
    if not saved:
        # Fall back to PNG so at least a raster copy is available
        try:
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved probability map (PNG fallback): {png_path}")
        except Exception as e2:
            print(f"Warning: Could not save probability map in any format: {e2}")
    plt.close()


def compute_shap_analysis(
    model_path: Path,
    feature_cols: list,
    data_sources: Dict[str, Path],
    output_dir: Path,
    model_type: str,
    timestamp: str,
    allow_test_shap: bool = False,
    shap_n_samples: int = 500,
) -> None:
    """Compute SHAP analysis and generate visualizations.
    
    Data source priority:
    1. train_win5.parquet (train set)
    2. earlystop_win5.parquet (validation set)
    3. test_win5.parquet (only if allow_test_shap=True)
    
    Args:
        model_path: Path to trained model pickle file
        feature_cols: List of feature column names
        data_sources: Dict with keys 'earlystop', 'train', 'test' pointing to parquet files
        output_dir: Output directory for SHAP results
        model_type: Model type (lgbm, rf)
        timestamp: Timestamp for output files
        allow_test_shap: If True, allow SHAP on test set (only if earlystop/train unavailable)
        shap_n_samples: Maximum number of rows to sample for SHAP (default: 500)
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError(
            "SHAP is not installed. To use SHAP analysis, please install it:\n"
            "  conda install -c conda-forge shap\n"
            "or:\n"
            "  pip install shap"
        )
    
    print("\n" + "=" * 70)
    print("COMPUTING SHAP ANALYSIS")
    print("=" * 70)
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    import pickle
    
    # Determine file format and load accordingly
    if model_path.suffix == '.joblib':
        try:
            import joblib
            model = joblib.load(model_path)
        except ImportError:
            raise RuntimeError(
                "joblib is not installed. To load .joblib model files, please install it:\n"
                "  conda install -c conda-forge joblib\n"
                "or:\n"
                "  pip install joblib"
            )
    else:
        # Default to pickle for .pkl files
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    # Select data source (priority: train → earlystop → test)
    data_source_priority = ['train', 'earlystop', 'test']
    selected_source = None
    selected_path = None
    
    for source in data_source_priority:
        if source in data_sources and data_sources[source] is not None and data_sources[source].exists():
            # Check if test set requires explicit flag
            if source == 'test' and not allow_test_shap:
                print(f"  Skipping test set (requires --allow_test_shap flag)")
                continue
            selected_source = source
            selected_path = data_sources[source]
            break
    
    if selected_source is None:
        print("  WARNING: No suitable data source found for SHAP analysis")
        if 'test' in data_sources and data_sources['test'] is not None and data_sources['test'].exists():
            print("  Test set available but --allow_test_shap flag not set")
        return
    
    print(f"  Using data source: {selected_source} ({selected_path.name})")
    
    # Load feature data (sample up to shap_n_samples rows with fixed seed).
    # Use row-group reservoir sampling to avoid loading the full parquet into memory (prevents OOM on large train sets).
    print(f"  Loading feature data from {selected_source}...")
    pf = pq.ParquetFile(selected_path)
    n_total = pf.metadata.num_rows
    print(f"  Total rows in {selected_source}: {n_total:,}")
    n_sample = min(shap_n_samples, n_total)
    print(f"  Sampling {n_sample:,} rows for SHAP analysis (seed=42, max={shap_n_samples})...")
    
    rng = np.random.default_rng(42)
    n_features = len(feature_cols)
    reservoir = np.zeros((n_sample, n_features), dtype=np.float32)
    reservoir_fill = 0
    rows_seen = 0
    
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i)
        # Load only feature columns for this row group to minimize memory
        sub = rg.select(feature_cols)
        chunk = sub.to_pandas().astype(np.float32)
        chunk_np = chunk.values
        chunk_len = len(chunk_np)
        del sub, chunk
        for k in range(chunk_len):
            rows_seen += 1
            if rows_seen <= n_sample:
                reservoir[reservoir_fill] = chunk_np[k]
                reservoir_fill += 1
            else:
                r = rng.integers(0, rows_seen)
                if r < n_sample:
                    reservoir[r] = chunk_np[k]
        del chunk_np
        gc.collect()
    
    X_shap = reservoir[:reservoir_fill].copy()
    del reservoir
    gc.collect()
    print(f"  Feature matrix shape: {X_shap.shape}")
    print(f"  Memory: {X_shap.nbytes / 1024**2:.1f} MB")
    
    # Compute SHAP values
    print(f"  Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    # LightGBM models do not support approximate=True; enable it only for RF-type models
    model_type_lower = str(model_type).lower()
    if model_type_lower in {"rf", "brf"}:
        shap_values = explainer.shap_values(X_shap, check_additivity=False, approximate=True)
    else:
        shap_values = explainer.shap_values(X_shap, check_additivity=False)
    print(f"  SHAP values computed: shape {shap_values.shape}")
    
    # For binary classification, shap_values can be:
    # - a list [neg_class, pos_class]
    # - a 2D array (n_samples, n_features) for the positive class
    # - a 3D array (n_samples, n_features, n_classes) for tree ensembles (e.g. RF)
    # In all cases we want the SHAP values for the positive class only.
    if isinstance(shap_values, list):
        # Typical LightGBM / tree-ensemble multi-output: list per class
        shap_values_pos = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        # Array output; handle possible extra class dimension
        if shap_values.ndim == 3:
            # Shape (n_samples, n_features, n_classes) -> select positive class (index 1 if available)
            class_idx = 1 if shap_values.shape[-1] > 1 else shap_values.shape[-1] - 1
            shap_values_pos = shap_values[:, :, class_idx]
        else:
            # Already (n_samples, n_features) for the positive class
            shap_values_pos = shap_values
    
    # Generate SHAP summary plot
    print(f"  Generating SHAP summary plot...")

    # Human-readable display names for SHAP plot axes (shorter for space)
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
    display_names = [SHAP_FEATURE_RENAME.get(f, f) for f in feature_cols]

    # shap.summary_plot manages its own figure size via `plot_size`; any prior
    # plt.figure(figsize=...) call is overridden.  Pass plot_size as a (w, h)
    # tuple — both values must be real numbers (None crashes matplotlib).
    # 11 × 8 gives noticeably more horizontal room for the beeswarm dots while
    # keeping a sensible aspect ratio in the manuscript.
    # color_bar_label supported in shap >= 0.40; falls back gracefully in older versions
    try:
        shap.summary_plot(
            shap_values_pos, X_shap,
            feature_names=display_names,
            show=False, plot_type="dot",
            plot_size=(11, 8),
            color_bar_label="Feature value (blue = low, red = high)",
        )
    except TypeError:
        shap.summary_plot(
            shap_values_pos, X_shap,
            feature_names=display_names,
            show=False, plot_type="dot",
            plot_size=(11, 8),
        )
    plt.tight_layout()

    summary_plot_path = output_dir / f"shap_summary.pdf"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved SHAP summary plot: {summary_plot_path.name}")
    
    # Compute feature importance metrics
    print(f"  Computing feature importance metrics...")
    
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    
    # Mean SHAP value per feature (signed)
    mean_shap = shap_values_pos.mean(axis=0)
    
    # Percentage of positive SHAP values per feature
    pct_shap_positive = (shap_values_pos > 0).mean(axis=0) * 100
    
    # Create DataFrame with top 20 features
    shap_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
        'pct_shap_positive': pct_shap_positive
    }).sort_values('mean_abs_shap', ascending=False).head(20)
    
    # Save to CSV
    shap_csv_path = output_dir / f"shap_top20.csv"
    shap_importance_df.to_csv(shap_csv_path, index=False)
    print(f"  Saved top 20 features: {shap_csv_path.name}")
    
    # SHAP dependence plots for top 2 features by mean |SHAP| (most impactful features).
    # Distance features (dist_*) are stored in metres; rescale x-axis to km for readability.
    METER_FEATURES = {'dist_wdpa', 'dist_road', 'dist_indigenous', 'dist_water', 'dist_coast'}
    top5_indices = np.argsort(mean_abs_shap)[-2:][::-1]
    for feat_idx in top5_indices:
        feat_name = feature_cols[feat_idx]
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(feat_name))
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat_idx, shap_values_pos, X_shap, feature_names=feature_cols, show=False)
        # Convert x-axis from metres to km for distance features
        if feat_name in METER_FEATURES:
            cur_ax = plt.gca()
            xticks_m = cur_ax.get_xticks()
            cur_ax.set_xticks(xticks_m)
            cur_ax.set_xticklabels([f'{v/1000:.0f}' for v in xticks_m])
            cur_ax.set_xlabel(f'{feat_name} (km)', fontsize=FONTSIZE_LABEL)
        plt.tight_layout()
        dep_path = output_dir / f"shap_dep_{safe_name}.pdf"
        plt.savefig(dep_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved SHAP dependence: {dep_path.name}")
    
    # Thematic feature importance: group features by prefix and aggregate mean_abs_shap
    # Ordered so that the first matching prefix wins.
    FEATURE_THEME_PREFIXES = (
        ('dist_',            'Proximity'),
        ('WorldClim_',       'Climate'),
        ('GSN_',             'Biodiversity'),
        ('GPW',              'Population'),        # matches GPW and GPW_smooth*
        ('elevation_',       'Terrain'),
        ('topo_',            'Terrain'),
        ('HNTL',             'Nighttime Lights'),  # matches HNTL and HNTL_smooth*
        ('landcover',        'Land Cover'),
        ('deforestation_',   'Deforestation'),
        ('wildfire_',        'Fire'),
        ('policy_',          'Policy'),
        ('economic_value',   'Economic Value'),
        ('socio_',           'Socio-economic'),
    )
    # Fixed color palette for consistent theme coloring across regions
    THEME_COLORS = {
        'Proximity':       '#E07B54',
        'Climate':         '#4A90D9',
        'Biodiversity':    '#27AE60',
        'Population':      '#9B59B6',
        'Terrain':         '#795548',
        'Nighttime Lights':'#F39C12',
        'Land Cover':      '#16A085',
        'Deforestation':   '#C0392B',
        'Fire':            '#E74C3C',
        'Policy':          '#2980B9',
        'Economic Value':  '#8E44AD',
        'Socio-economic':  '#7F8C8D',
        'Other':           '#BDC3C7',
    }
    theme_sums = {}
    for i, fname in enumerate(feature_cols):
        theme = "Other"
        for prefix, theme_label in FEATURE_THEME_PREFIXES:
            if fname.startswith(prefix):
                theme = theme_label
                break
        theme_sums[theme] = theme_sums.get(theme, 0.0) + mean_abs_shap[i]
    theme_order = sorted(theme_sums.keys(), key=lambda k: theme_sums[k], reverse=True)
    themes = [theme_order[i] for i in range(len(theme_order))]
    sums = [theme_sums[t] for t in themes]
    bar_colors = [THEME_COLORS.get(t, '#BDC3C7') for t in themes]
    fig, ax = plt.subplots(figsize=(8, max(4, len(themes) * 0.5)))
    ax.barh(themes, sums, color=bar_colors)
    ax.set_xlabel('Sum of mean |SHAP|', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) SHAP importance by theme', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    thematic_path = output_dir / f"shap_thematic.pdf"
    plt.savefig(thematic_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved thematic SHAP: {thematic_path.name}")
    
    print(f"\n  Top 10 features by mean absolute SHAP:")
    for idx, row in shap_importance_df.head(10).iterrows():
        print(f"    {row['feature']:40s}: {row['mean_abs_shap']:.4f} (mean: {row['mean_shap']:+.4f}, {row['pct_shap_positive']:.1f}% positive)")
    
    print("  SHAP analysis completed.")


def create_calibration_improvement_figure(
    df: pd.DataFrame,
    output_dir: Path,
    model_type: str,
) -> None:
    """Bar chart comparing ECE and Brier Score before vs after calibration.

    Requires y_pred_proba_uncalibrated and y_pred_proba_calibrated columns in df.
    """
    print("\n" + "=" * 70)
    print("CREATING CALIBRATION IMPROVEMENT FIGURE")
    print("=" * 70)

    has_uncal = 'y_pred_proba_uncalibrated' in df.columns
    has_cal = 'y_pred_proba_calibrated' in df.columns

    if not (has_uncal and has_cal):
        print("  Skipping: need both y_pred_proba_uncalibrated and y_pred_proba_calibrated columns.")
        return

    y_true = df['y_true'].values
    y_uncal = df['y_pred_proba_uncalibrated'].values
    y_cal = df['y_pred_proba_calibrated'].values

    def compute_ece(y_true, y_prob, n_bins=10):
        """Expected Calibration Error with uniform bins."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
        return ece

    ece_before = compute_ece(y_true, y_uncal)
    ece_after = compute_ece(y_true, y_cal)
    brier_before = brier_score_loss(y_true, y_uncal)
    brier_after = brier_score_loss(y_true, y_cal)

    print(f"  ECE:   before={ece_before:.4f}  after={ece_after:.4f}  improvement={ece_before - ece_after:.4f}")
    print(f"  Brier: before={brier_before:.4f}  after={brier_after:.4f}  improvement={brier_before - brier_after:.4f}")

    metrics = ['ECE', 'Brier Score']
    before_vals = [ece_before, brier_before]
    after_vals = [ece_after, brier_after]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, before_vals, width, label='Uncalibrated', color='#E07B54', alpha=0.9)
    bars2 = ax.bar(x + width / 2, after_vals, width, label='Calibrated', color='#4A90D9', alpha=0.9)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=FONTSIZE_STATS)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=FONTSIZE_STATS)

    # Delta annotations: show improvement (↓ good) or degradation (↑ bad) between pairs
    overall_max = max(before_vals + after_vals)
    for i, (before, after) in enumerate(zip(before_vals, after_vals)):
        delta = after - before  # negative = improved
        if abs(delta) > 1e-8:
            arrow = '↓' if delta < 0 else '↑'
            color = '#27AE60' if delta < 0 else '#C0392B'
            label = f'{arrow} {abs(delta):.4f}'
            ax.text(i, max(before, after) + overall_max * 0.05, label,
                    ha='center', va='bottom', fontsize=FONTSIZE_STATS, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Error (lower is better)', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) Calibration Improvement', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    pdf_path = output_dir / 'calibration_improvement.pdf'
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    print(f"  Saved: {pdf_path}")
    plt.close()


def create_country_breakdown(
    df: pd.DataFrame,
    output_dir: Path,
    model_type: str,
) -> None:
    """Per-country ROC-AUC, Precision@1%, observed PAs → CSV + LaTeX table.

    Uses spatial join with Natural Earth country boundaries.
    Aggregates to pixel level first (unique row/col) for efficiency, then computes
    row-level metrics using country assignments.
    """
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 70)
    print("CREATING COUNTRY-LEVEL BREAKDOWN")
    print("=" * 70)

    required_cols = ['y_true', 'y_pred_proba', 'x', 'y']
    if not all(c in df.columns for c in required_cols):
        print(f"  Skipping: missing required columns {required_cols}")
        return

    # Build pixel-level lookup for country assignment (unique row/col or x/y)
    use_rowcol = 'row' in df.columns and 'col' in df.columns
    if use_rowcol:
        pixel_coords = df[['row', 'col', 'x', 'y']].drop_duplicates(subset=['row', 'col'])
    else:
        df['_x_r'] = df['x'].round(4)
        df['_y_r'] = df['y'].round(4)
        pixel_coords = df[['_x_r', '_y_r', 'x', 'y']].drop_duplicates(subset=['_x_r', '_y_r'])

    print(f"  Unique pixels: {len(pixel_coords):,} from {len(df):,} rows")

    # Spatial join with Natural Earth country boundaries
    try:
        world = get_world_countries()
        world = world[world.geometry.notnull()].copy()
        print(f"  Loaded {len(world)} country geometries from Natural Earth")
    except Exception as e:
        print(f"  Could not load country boundaries: {e}. Skipping country breakdown.")
        return

    # Create pixel GeoDataFrame (assume EPSG:4326 after coordinate detection)
    try:
        pixel_gdf = gpd.GeoDataFrame(
            pixel_coords.copy(),
            geometry=gpd.points_from_xy(pixel_coords['x'], pixel_coords['y']),
            crs='EPSG:4326'
        )
        world_4326 = world.to_crs('EPSG:4326')
        # Detect country name column (varies by Natural Earth version/source)
        name_col = next(
            (c for c in world_4326.columns if c.lower() == 'name'),
            None,
        )
        if name_col is None:
            # Fallback: ADMIN is present in 110m files
            name_col = next(
                (c for c in world_4326.columns if c.lower() in ('admin', 'sovereignt', 'country')),
                world_4326.columns[0],
            )
        join_cols = [name_col, 'geometry']
        joined = gpd.sjoin(pixel_gdf, world_4326[join_cols], how='left', predicate='within')
        joined['country'] = joined[name_col].fillna('Unknown')
        print(f"  Spatial join complete. Countries assigned: {joined['country'].nunique()}")
    except Exception as e:
        print(f"  Spatial join failed: {e}. Skipping country breakdown.")
        return

    # Build country lookup
    if use_rowcol:
        country_lookup = joined.set_index(['row', 'col'])['country'].to_dict()
        df = df.copy()
        df['country'] = df.apply(lambda r: country_lookup.get((r['row'], r['col']), 'Unknown'), axis=1)
    else:
        country_lookup = joined.set_index(['_x_r', '_y_r'])['country'].to_dict()
        df = df.copy()
        df['country'] = df.apply(lambda r: country_lookup.get((round(r['x'], 4), round(r['y'], 4)), 'Unknown'), axis=1)
        df = df.drop(columns=['_x_r', '_y_r'], errors='ignore')

    # Compute per-country metrics
    records = []
    for country, grp in df.groupby('country'):
        n_rows = len(grp)
        n_pos = int(grp['y_true'].sum())
        n_neg = n_rows - n_pos

        if n_pos < 10 or n_neg < 10:
            continue  # Skip countries with too few samples

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(grp['y_true'], grp['y_pred_proba'])
        except Exception:
            roc_auc = float('nan')

        # Precision@1%
        y_proba = grp['y_pred_proba'].values
        y_true_arr = grp['y_true'].values
        n_top = max(1, int(len(y_proba) * 0.01))
        top_idx = np.argpartition(y_proba, -n_top)[-n_top:]
        p_at_1pct = float(y_true_arr[top_idx].mean())

        records.append({
            'Country': country,
            'n_pixels': n_rows,
            'n_observed_PAs': n_pos,
            'ROC_AUC': round(roc_auc, 4),
            'Precision_at_1pct': round(p_at_1pct, 4),
        })

    if not records:
        print("  No countries with sufficient data. Skipping.")
        return

    result_df = pd.DataFrame(records).sort_values('n_observed_PAs', ascending=False)

    # Save CSV
    csv_path = output_dir / 'country_breakdown.csv'
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Save LaTeX table (top 15 countries by n_observed_PAs).
    # Written manually to avoid pandas Styler / jinja2 dependency
    # (to_latex() routes through Styler in pandas >= 1.3 and requires jinja2).
    top_df = result_df.head(15).copy()
    col_headers = ['Country', 'N Pixels', 'N Observed PAs', 'ROC-AUC', r'P@1\%']

    def _fmt(val):
        if isinstance(val, float):
            return f'{val:.4f}'
        return str(val).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    latex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\begin{tabular}{lrrrrr}',
        r'\toprule',
        ' & '.join(col_headers) + r' \\',
        r'\midrule',
    ]
    for _, row in top_df.iterrows():
        cells = [_fmt(v) for v in row]
        latex_lines.append(' & '.join(cells) + r' \\')
    latex_lines += [
        r'\bottomrule',
        r'\end{tabular}',
        rf'\caption{{Per-country model performance ({MODEL_LABEL}, {model_type.upper()})}}',
        rf'\label{{tab:country_breakdown_{model_type}}}',
        r'\end{table}',
    ]
    latex_path = output_dir / 'country_breakdown.tex'
    latex_path.write_text('\n'.join(latex_lines) + '\n')
    print(f"  Saved LaTeX: {latex_path}")
    print(f"  Countries included: {len(result_df)} (min 10 positive samples)")


def create_biome_breakdown(
    df: pd.DataFrame,
    output_dir: Path,
    model_type: str,
    gsn_tif_path: Optional[Path] = None,
    gsn_shp_path: Optional[Path] = None,
) -> None:
    """Per-biome and per-ecoregion metrics using GSN terrestrial ecoregions data.

    The raster (gsn_terrestrial_ecoregions_mask_1km.tif) is a single-band int16
    categorical raster in EPSG:3857 at 1 km resolution aligned to the backbone
    grid. Pixel values = ecoregion_id × 4 (63 ecoregions; 0 = nodata/ocean).
    Pixel assignment via direct row/col indexing — no reprojection needed.

    If the shapefile (Terrestrial_ecoregions.shp) is also provided, a spatial
    sample is used to build a raster_value → ECO_NAME → BIOME_NAME lookup,
    enabling named ecoregion labels and biome-level aggregation.

    Outputs:
        ecoregion_breakdown.csv  — per-ecoregion metrics (ECO_NAME, BIOME_NAME, ROC-AUC, P@1%)
        biome_breakdown.pdf      — biome-level bar chart (grouped by BIOME_NAME)
        ecoregion_breakdown.pdf  — top-20 ecoregions bar chart (only if shapefile available)

    Args:
        df: Scored predictions DataFrame (must have row, col, y_true, y_pred_proba).
        output_dir: Directory for output files.
        model_type: Model identifier string.
        gsn_tif_path: Path to gsn_terrestrial_ecoregions_mask_1km.tif.
        gsn_shp_path: Path to Terrestrial_ecoregions.shp (optional but recommended).
    """
    import rasterio
    import rasterio.transform
    from sklearn.metrics import roc_auc_score
    from shapely.geometry import Point

    print("\n" + "=" * 70)
    print("CREATING BIOME / ECOREGION BREAKDOWN")
    print("=" * 70)

    if gsn_tif_path is None or not Path(gsn_tif_path).exists():
        print(f"  Skipping: GSN raster not found.")
        if gsn_tif_path is not None:
            print(f"  Expected: {gsn_tif_path}")
        return

    if 'row' not in df.columns or 'col' not in df.columns:
        print("  Skipping: scored parquet missing row/col columns.")
        return

    print(f"  TIF:      {gsn_tif_path}")
    print(f"  Shapefile: {gsn_shp_path if gsn_shp_path and gsn_shp_path.exists() else 'not provided — numeric IDs only'}")

    # ── 1. Read raster ────────────────────────────────────────────────────────
    with rasterio.open(gsn_tif_path) as src:
        raster_height, raster_width = src.height, src.width
        raster_transform = src.transform
        raster_crs = src.crs
        eco_raster = src.read(1)  # int16; raw_value = eco_id × 4; 0 = nodata

    unique_raw = sorted(set(eco_raster.flatten()) - {0})
    print(f"  Ecoregions in raster: {len(unique_raw)}")

    # ── 2. Build raster_value → (ECO_NAME, BIOME_NAME) lookup via spatial join ─
    val_to_meta: dict[int, dict] = {}
    has_names = False

    if gsn_shp_path is not None and Path(gsn_shp_path).exists():
        try:
            eco_gdf = gpd.read_file(gsn_shp_path).to_crs(raster_crs)
            # Sample one representative pixel center per raster value
            sample_rows, sample_cols, sample_vals = [], [], []
            for val in unique_raw:
                r_idx, c_idx = np.where(eco_raster == val)
                mid = len(r_idx) // 2
                sample_rows.append(r_idx[mid])
                sample_cols.append(c_idx[mid])
                sample_vals.append(int(val))

            xs, ys = rasterio.transform.xy(raster_transform, sample_rows, sample_cols)
            pts_gdf = gpd.GeoDataFrame(
                {'raster_val': sample_vals},
                geometry=[Point(x, y) for x, y in zip(xs, ys)],
                crs=raster_crs,
            )
            joined = gpd.sjoin(
                pts_gdf,
                eco_gdf[['ECO_NAME', 'BIOME_NAME', 'REALM', 'geometry']],
                how='left', predicate='within',
            )
            for _, row in joined.iterrows():
                val_to_meta[int(row['raster_val'])] = {
                    'ECO_NAME':  str(row.get('ECO_NAME',  f'Ecoregion {int(row["raster_val"])//4}')),
                    'BIOME_NAME': str(row.get('BIOME_NAME', 'Unknown')),
                    'REALM':     str(row.get('REALM',     'Unknown')),
                }
            has_names = True
            print(f"  Name lookup built: {len(val_to_meta)} ecoregions matched")
        except Exception as e:
            print(f"  Warning: shapefile join failed ({e}). Using numeric IDs.")

    # Fallback: numeric labels
    for val in unique_raw:
        if val not in val_to_meta:
            val_to_meta[int(val)] = {
                'ECO_NAME':   f'Ecoregion {int(val)//4}',
                'BIOME_NAME': 'Unknown',
                'REALM':      'Unknown',
            }

    # ── 3. Assign ecoregion to each row via row/col lookup ───────────────────
    rows_arr = df['row'].values.astype(int)
    cols_arr = df['col'].values.astype(int)
    valid = (rows_arr >= 0) & (rows_arr < raster_height) & (cols_arr >= 0) & (cols_arr < raster_width)
    if (~valid).sum() > 0:
        print(f"  Warning: {(~valid).sum():,} rows out of raster bounds (treated as nodata)")

    raw_vals = np.zeros(len(df), dtype=np.int32)
    raw_vals[valid] = eco_raster[rows_arr[valid], cols_arr[valid]]

    n_covered = (raw_vals > 0).sum()
    print(f"  Rows with ecoregion: {n_covered:,} / {len(df):,} ({n_covered/len(df)*100:.1f}%)")

    # ── 4. Per-ecoregion metrics ──────────────────────────────────────────────
    y_true_all  = df['y_true'].values
    y_proba_all = df['y_pred_proba'].values

    eco_records = []
    for val in unique_raw:
        mask = (raw_vals == val)
        grp_true  = y_true_all[mask]
        grp_proba = y_proba_all[mask]
        n_rows = int(mask.sum())
        n_pos  = int(grp_true.sum())
        if n_pos < 5 or (n_rows - n_pos) < 100:
            continue

        try:
            roc_auc = roc_auc_score(grp_true, grp_proba)
        except Exception:
            roc_auc = float('nan')

        n_top   = max(1, int(n_rows * 0.01))
        top_idx = np.argpartition(grp_proba, -n_top)[-n_top:]
        p_at_1pct = float(grp_true[top_idx].mean())

        meta = val_to_meta[val]
        eco_records.append({
            'ecoregion_id':   int(val) // 4,
            'ECO_NAME':       meta['ECO_NAME'],
            'BIOME_NAME':     meta['BIOME_NAME'],
            'REALM':          meta['REALM'],
            'n_pixel_years':  n_rows,
            'n_observed_PAs': n_pos,
            'prevalence_pct': round(n_pos / n_rows * 100, 4),
            'ROC_AUC':        round(roc_auc, 4),
            'Precision_at_1pct': round(p_at_1pct, 4),
        })

    if not eco_records:
        print("  No ecoregions with sufficient data. Skipping.")
        return

    eco_df = pd.DataFrame(eco_records).sort_values('n_observed_PAs', ascending=False)
    csv_path = output_dir / 'ecoregion_breakdown.csv'
    eco_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}  ({len(eco_df)} ecoregions)")

    # ── 5. Per-biome aggregation ──────────────────────────────────────────────
    # Skip biome aggregation when no shapefile was available: every ecoregion
    # would have BIOME_NAME = 'Unknown', collapsing the entire region into a
    # single uninformative 'Unknown' biome row.
    if not has_names:
        print("  Biome aggregation skipped: no GSN shapefile provided (all ecoregions "
              "labelled 'Unknown'). Supply --gsn_shp for named biome outputs.")
        return

    biome_records = []
    for biome_name, grp_df in eco_df.groupby('BIOME_NAME'):
        # Re-aggregate metrics over raw pixel rows (not average of ecoregion metrics)
        biome_mask = np.isin(raw_vals, [v for v in unique_raw if val_to_meta[v]['BIOME_NAME'] == biome_name])
        grp_true  = y_true_all[biome_mask]
        grp_proba = y_proba_all[biome_mask]
        n_rows = int(biome_mask.sum())
        n_pos  = int(grp_true.sum())
        if n_pos < 5 or (n_rows - n_pos) < 100:
            continue
        try:
            roc_auc = roc_auc_score(grp_true, grp_proba)
        except Exception:
            roc_auc = float('nan')
        n_top   = max(1, int(n_rows * 0.01))
        top_idx = np.argpartition(grp_proba, -n_top)[-n_top:]
        p_at_1pct = float(grp_true[top_idx].mean())
        prevalence = n_pos / n_rows
        biome_records.append({
            'BIOME_NAME':     biome_name,
            'n_ecoregions':   len(grp_df),
            'n_pixel_years':  n_rows,
            'n_observed_PAs': n_pos,
            'prevalence_pct': round(prevalence * 100, 4),
            'ROC_AUC':        round(roc_auc, 4),
            'Precision_at_1pct': round(p_at_1pct, 4),
            'Lift_at_1pct':   round(p_at_1pct / prevalence, 2) if prevalence > 0 else 0.0,
        })

    if not biome_records:
        print("  No biomes with sufficient data for aggregation. Skipping biome chart.")
        return

    biome_df = pd.DataFrame(biome_records).sort_values('ROC_AUC', ascending=True)

    biome_csv = output_dir / 'biome_breakdown.csv'
    biome_df.to_csv(biome_csv, index=False)
    print(f"  Saved: {biome_csv}  ({len(biome_df)} biomes)")

    result_df = biome_df  # used for plotting below
    print(f"  Ecoregions with sufficient data: {len(eco_df)}")

    # Bar chart: ROC-AUC per band (horizontal, sorted by ROC-AUC)
    # ── 6. Biome-level bar chart (paper figure) ───────────────────────────────
    # Short biome labels for clean axis display
    SHORT_BIOME = {
        'Tropical & Subtropical Moist Broadleaf Forests':         'Trop. Moist Broadleaf',
        'Tropical & Subtropical Dry Broadleaf Forests':           'Trop. Dry Broadleaf',
        'Tropical & Subtropical Grasslands, Savannas & Shrublands': 'Trop. Grasslands/Savannas',
        'Tropical & Subtropical Coniferous Forests':               'Trop. Coniferous',
        'Temperate Broadleaf & Mixed Forests':                     'Temp. Broadleaf',
        'Temperate Grasslands, Savannas & Shrublands':             'Temp. Grasslands/Savannas',
        'Flooded Grasslands & Savannas':                           'Flooded Grasslands',
        'Montane Grasslands & Shrublands':                         'Montane Grasslands',
        'Mediterranean Forests, Woodlands & Scrub':                'Mediterranean',
        'Deserts & Xeric Shrublands':                              'Deserts & Xeric',
        'Mangroves':                                               'Mangroves',
        'Tundra':                                                  'Tundra',
    }

    labels     = [SHORT_BIOME.get(b, b[:30]) for b in biome_df['BIOME_NAME']]
    roc_vals   = biome_df['ROC_AUC'].tolist()
    lift_vals  = biome_df['Lift_at_1pct'].tolist()
    n_vals     = biome_df['n_pixel_years'].tolist()
    y_pos = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(biome_df) * 0.55)))

    # ROC-AUC panel
    ax1 = axes[0]
    bars1 = ax1.barh(y_pos, roc_vals, color='#4A90D9', alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=FONTSIZE_STATS)
    ax1.set_xlabel('ROC-AUC', fontsize=FONTSIZE_LABEL)
    ax1.set_title('ROC-AUC by Biome', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random (0.5)')
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=FONTSIZE_LEGEND)
    ax1.grid(True, axis='x', alpha=0.3)
    for bar, val, n in zip(bars1, roc_vals, n_vals):
        ax1.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}  (n={n:,})', va='center', ha='left', fontsize=FONTSIZE_STATS)

    # Lift@1% panel (Precision@1% normalised by positive rate — comparable across biomes)
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, lift_vals, color='#E07B54', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=FONTSIZE_STATS)
    ax2.set_xlabel('Lift@1% (×)', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Lift@1% by Biome', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random (1×)')
    lift_max = max(lift_vals) if lift_vals else 1
    ax2.set_xlim(0, lift_max * 1.3)
    ax2.legend(fontsize=FONTSIZE_LEGEND)
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars2, lift_vals):
        ax2.text(min(val + lift_max * 0.01, lift_max * 1.25), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}×', va='center', ha='left', fontsize=FONTSIZE_STATS)

    plt.suptitle(f'{MODEL_LABEL} ({model_type.upper()}) Performance by Biome (GSN Terrestrial Ecoregions)',
                fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.01)
    plt.tight_layout()

    pdf_path = output_dir / 'biome_breakdown.pdf'
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    print(f"  Saved PDF: {pdf_path}")
    plt.close()


def create_biome_scatter_comparison(
    output_dir: Path,
    repo_root: Path,
    scratch_root: Optional[Path],
) -> None:
    """Cross-region biome-level precision@1% vs recall@1% scatter (all 3 regions).

    Reads the latest biome_metrics_*.csv for each region from the
    spatial_generalisation directory (written by spatial_CV_2).
    Gracefully skips any region whose CSV is not yet available.
    """
    print("\n" + "=" * 70)
    print("CREATING BIOME SCATTER (CROSS-REGION)")
    print("=" * 70)

    REGION_CONFIGS = [
        ("south_america", "South America"),
        ("usa",           "United States"),
        ("se_asia",       "SE Asia"),
    ]
    BIOME_SHORT = {
        "Deserts & Xeric Shrublands":                                    "Deserts",
        "Flooded Grasslands & Savannas":                                 "Flooded Grass.",
        "Mangroves":                                                     "Mangroves",
        "Mediterranean Forests, Woodlands & Scrub":                     "Mediter. Forests",
        "Montane Grasslands & Shrublands":                               "Montane Grass.",
        "Temperate Broadleaf & Mixed Forests":                           "Temp. Broadleaf",
        "Tropical & Subtropical Dry Broadleaf Forests":                  "Trop. Dry BL",
        "Tropical & Subtropical Grasslands, Savannas & Shrublands":     "Trop. Grass.",
        "Tropical & Subtropical Moist Broadleaf Forests":               "Trop. Moist BL",
        "Temperate Conifer Forests":                                     "Temp. Conifers",
        "Temperate Grasslands, Savannas & Shrublands":                   "Temp. Grass.",
        "Tropical & Subtropical Coniferous Forests":                     "Trop. Conifers",
        "Boreal Forests/Taiga":                                          "Boreal",
    }

    def _biome_annotate_label(raw: object) -> str:
        """Short label for scatter annotations; biome column may be NaN or non-string."""
        if raw is None:
            return "?"
        if isinstance(raw, (float, np.floating)) and pd.isna(raw):
            return "?"
        text = str(raw).strip()
        if not text or text.lower() == "nan":
            return "?"
        return BIOME_SHORT.get(text, text[:15])

    def _latest_biome_csv(slug: str) -> Optional[Path]:
        search_dirs = []
        if scratch_root:
            search_dirs.append(scratch_root / f"outputs/{slug}/results/spatial_generalisation")
        search_dirs.append(repo_root / f"outputs/{slug}/results/spatial_generalisation")
        for d in search_dirs:
            if d.exists():
                matches = sorted(d.glob("biome_metrics_*.csv"))
                if matches:
                    return matches[-1]
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Biome-level Prediction Quality: Precision@1\\% vs. Recall@1\\% (LightGBM test set)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    any_data = False
    for ax, (slug, label) in zip(axes, REGION_CONFIGS):
        path = _latest_biome_csv(slug)
        if path is None:
            ax.set_title(f"{label}\n(biome_metrics CSV not found)")
            print(f"  Skipping {label}: biome_metrics CSV not found in spatial_generalisation/")
            continue

        print(f"  {label}: {path}")
        df_b = pd.read_csv(path)
        df_b = df_b[df_b["model"] == "LGBM"].copy()
        df_b = df_b[df_b["n_positives"] >= 20].copy()
        if df_b.empty:
            ax.set_title(f"{label}\n(no biomes with ≥20 positives)")
            continue

        any_data = True
        precision = df_b["precision_at_1pct"].values
        recall    = df_b["recall_at_1pct"].values
        n_pos     = df_b["n_positives"].values
        biomes    = df_b["biome"].values
        pos_rate  = df_b["positive_rate"].values

        sizes = np.clip(np.log1p(n_pos) * 12, 40, 600)
        sc = ax.scatter(recall, precision, s=sizes, c=pos_rate, cmap="YlOrRd",
                        alpha=0.85, edgecolors="0.3", linewidths=0.5, zorder=3)

        for r, p, b in zip(recall, precision, biomes):
            short = _biome_annotate_label(b)
            ax.annotate(short, (r, p), fontsize=6.5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points", zorder=4)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.4, zorder=1)
        ax.set_xlabel("Recall @ top-1\\%", fontsize=9)
        ax.set_ylabel("Precision @ top-1\\%" if ax == axes[0] else "", fontsize=9)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, zorder=0)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Positive rate", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)

    if not any_data:
        plt.close()
        print("  Skipped: no biome_metrics CSVs available.")
        return

    legend_elements = [
        plt.scatter([], [], s=np.log1p(500) * 12, c="grey", alpha=0.6,
                    edgecolors="0.3", label="N=500"),
        plt.scatter([], [], s=np.log1p(5000) * 12, c="grey", alpha=0.6,
                    edgecolors="0.3", label="N=5,000"),
        plt.scatter([], [], s=np.log1p(50000) * 12, c="grey", alpha=0.6,
                    edgecolors="0.3", label="N=50,000"),
    ]
    axes[1].legend(handles=legend_elements, title="N designations\n(bubble size)",
                   fontsize=6.5, title_fontsize=7, loc="lower right", framealpha=0.8)

    plt.tight_layout()
    out_path = output_dir / "biome_scatter_pr.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def create_model_comparison_landscape(
    output_dir: Path,
    repo_root: Path,
    scratch_root: Optional[Path],
) -> None:
    """Two-panel cross-model performance landscape figure.

    Panel A: Precision@K% threshold decay profiles (LGBM solid, RF dashed).
    Panel B: ROC-AUC vs Lift@1% scatter for all 6 models (3 regions × 2 algos).

    Reads metrics_table.csv written by Stage 7 for each region/algorithm.
    Gracefully skips models where data is unavailable.
    """
    from scipy.interpolate import PchipInterpolator
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D as _Line2D

    print("\n" + "=" * 70)
    print("CREATING MODEL COMPARISON LANDSCAPE")
    print("=" * 70)

    REGION_CONFIGS = [
        ("south_america", "model1", "South America"),
        ("usa",           "model2", "United States"),
        ("se_asia",       "model3", "SE Asia"),
    ]
    ALGO_CONFIGS = [
        ("lgbm", "LightGBM", "-",  "o", 1.00),
        ("rf",   "RF",       "--", "s", 0.65),
    ]
    REGION_COLORS = {
        "South America": "#E69F00",
        "United States": "#0072B2",
        "SE Asia":       "#7B2CBF",
    }

    def _find_metrics_csv(slug: str, model_id: str, algo: str) -> Optional[Path]:
        candidates = []
        if scratch_root:
            candidates.append(scratch_root / f"outputs/{slug}/results/{model_id}_{algo}/metrics_table.csv")
        candidates.append(repo_root / f"outputs/{slug}/results/{model_id}_{algo}/metrics_table.csv")
        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_metrics(slug: str, model_id: str, algo: str) -> Optional[dict]:
        path = _find_metrics_csv(slug, model_id, algo)
        if path is None:
            return None
        df_m = pd.read_csv(path)
        m = dict(zip(df_m["Metric"].str.strip(), df_m["Value"].astype(str)))
        def _f(key: str) -> Optional[float]:
            try:
                return float(m[key]) if key in m else None
            except (ValueError, TypeError):
                return None
        return {
            "roc":   _f("ROC AUC"),
            "lift1": _f("Lift @ 1%"),
            "p1":    _f("Precision @ 1%"),
            "p5":    _f("Precision @ 5%"),
            "p10":   _f("Precision @ 10%"),
            "base":  _f("Baseline Rate"),
        }

    all_models: dict[str, dict] = {}
    for slug, model_id, region_label in REGION_CONFIGS:
        for algo, algo_label, linestyle, marker, alpha in ALGO_CONFIGS:
            data = _load_metrics(slug, model_id, algo)
            if data:
                data.update({"region": region_label, "algo": algo_label,
                             "linestyle": linestyle, "marker": marker, "alpha": alpha})
                all_models[f"{region_label} {algo_label}"] = data
                print(f"  Loaded: {region_label} {algo_label}")
            else:
                print(f"  Not found: outputs/{slug}/results/{model_id}_{algo}/metrics_table.csv")

    if not all_models:
        print("  Skipped: no metrics_table.csv files found.")
        return

    THRESHOLDS = [1, 5, 10]
    X_SMOOTH = np.linspace(1, 100, 400)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    fig.subplots_adjust(wspace=0.36)

    # ── Panel A: Precision-at-threshold profiles ───────────────────────────────
    ax = ax_left
    K_BREAK, TAIL_COMPRESS = 10.0, 0.18

    def _k_fwd(x):
        x = np.asarray(x)
        return np.where(x <= K_BREAK, x, K_BREAK + (x - K_BREAK) * TAIL_COMPRESS)

    def _k_inv(x):
        x = np.asarray(x)
        return np.where(x <= K_BREAK, x, K_BREAK + (x - K_BREAK) / TAIL_COMPRESS)

    ax.set_xscale("function", functions=(_k_fwd, _k_inv))

    for key, m in all_models.items():
        vals = [m.get("p1"), m.get("p5"), m.get("p10"), m.get("base")]
        if any(v is None for v in vals):
            continue
        smooth = np.clip(PchipInterpolator([1, 5, 10, 100], vals)(X_SMOOTH), 0, None)
        col = REGION_COLORS.get(m["region"], "grey")
        lw  = 0.75 if m["algo"] == "LightGBM" else 0.55
        ms  = 4.2  if m["algo"] == "LightGBM" else 3.4
        ax.plot(X_SMOOTH, smooth, m["linestyle"], color=col, linewidth=lw,
                alpha=m["alpha"], zorder=4 if m["algo"] == "LightGBM" else 3)
        ax.plot(THRESHOLDS, [m["p1"], m["p5"], m["p10"]], m["marker"],
                color=col, markersize=ms, alpha=m["alpha"],
                zorder=5 if m["algo"] == "LightGBM" else 4)

    ax.set_xlabel("Top-$K$\\% risk zone", fontsize=9.5)
    ax.set_ylabel("Precision @ top-$K$\\%", fontsize=9.5)
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(["1\\%", "5\\%", "10\\%", "25\\%", "50\\%", "100\\%"], fontsize=8)
    ax.set_ylim(-0.02, 0.72)
    ax.grid(True, alpha=0.25)
    ax.set_title("(a) Targeting Precision at Different Risk Thresholds",
                 fontsize=9.5, fontweight="bold", pad=8)

    present_regions = {m["region"] for m in all_models.values()}
    region_handles = [
        _Line2D([], [], color=col, linewidth=0.9, label=reg)
        for reg, col in REGION_COLORS.items() if reg in present_regions
    ]
    algo_handles = [
        _Line2D([], [], color="0.3", linestyle="-",  linewidth=0.9, marker="o",
                markersize=5, label="LightGBM"),
        _Line2D([], [], color="0.3", linestyle="--", linewidth=0.65, marker="s",
                markersize=4, alpha=0.7, label="Random Forest"),
    ]
    leg1 = ax.legend(handles=region_handles, loc="upper right",
                     fontsize=7.5, title="Region", title_fontsize=8,
                     framealpha=0.85, handlelength=1.5)
    ax.add_artist(leg1)
    ax.legend(handles=algo_handles, loc="center right",
              fontsize=7.5, title="Algorithm", title_fontsize=8,
              framealpha=0.85, handlelength=1.8, bbox_to_anchor=(1.0, 0.45))

    # ── Panel B: ROC-AUC vs Lift@1% scatter ───────────────────────────────────
    ax = ax_right
    for key, m in all_models.items():
        roc, lift1 = m.get("roc"), m.get("lift1")
        if roc is None or lift1 is None:
            continue
        col = REGION_COLORS.get(m["region"], "grey")
        ax.scatter(roc, lift1,
                   s=120 if m["algo"] == "LightGBM" else 80,
                   c=col, marker=m["marker"],
                   edgecolors="white", linewidths=1.0, alpha=0.92, zorder=4)
        short = key.replace(" LightGBM", " LGB")
        ax.annotate(short, (roc, lift1), textcoords="offset points",
                    xytext=(5, 4) if m["algo"] == "LightGBM" else (5, -9),
                    fontsize=7, color=col, fontweight="bold")

    ax.set_xlabel("ROC-AUC", fontsize=9.5)
    ax.set_ylabel("Lift @ top-1\\%", fontsize=9.5)
    ax.grid(True, alpha=0.25)
    ax.set_title("(b) Discrimination vs.\\ Targeting Effectiveness",
                 fontsize=9.5, fontweight="bold", pad=8)

    patch_handles = [
        Patch(color=col, label=reg)
        for reg, col in REGION_COLORS.items() if reg in present_regions
    ]
    algo_handles2 = [
        _Line2D([], [], marker="o", color="0.3", linestyle="None",
                markersize=7, label="LightGBM"),
        _Line2D([], [], marker="s", color="0.3", linestyle="None",
                markersize=6, label="RF"),
    ]
    leg3 = ax.legend(handles=patch_handles, fontsize=7.5, loc="lower right",
                     title="Region", title_fontsize=8, framealpha=0.85)
    ax.add_artist(leg3)
    ax.legend(handles=algo_handles2, fontsize=7.5, loc="lower left",
              title="Algorithm", title_fontsize=8, framealpha=0.85)

    out_path = output_dir / "model_comparison_landscape.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Generate comprehensive results report for {MODEL_ID}_LGBM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--metrics_json',
        type=str,
        default=None,
        help=f'Path to {MODEL_ID}_lgbm_metrics_*.json (LGBM) or {MODEL_ID}_rf_win5_metrics_*.json (RF) file (default: auto-discover latest in outputs/{REGION_SLUG}/results/ml_models/)'
    )
    parser.add_argument(
        '--scored_parquet',
        type=str,
        default=None,
        help=f'Path to {MODEL_ID}_lgbm_scored_*.parquet (LGBM) or {MODEL_ID}_rf_win5_scored_*.parquet (RF) file (default: auto-discover latest in outputs/{REGION_SLUG}/results/ml_models/)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['lgbm', 'rf'],
        default='lgbm',
        help='Model type: lgbm or rf (default: lgbm)'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help=f'Output directory (default: outputs/{REGION_SLUG}/results/{MODEL_ID}_{{model_type}}/)'
    )
    parser.add_argument(
        '--region_boundary',
        type=str,
        default=None,
        help=f'Optional path to {REGION_LABEL} boundary GeoJSON/shapefile. If not provided, downloads from Natural Earth.'
    )
    parser.add_argument(
        '--test_parquet',
        type=str,
        default=None,
        help='Path to original test_win5.parquet file (for loading WDPA_b1 data). If not provided, will try to auto-discover.'
    )
    parser.add_argument(
        '--future_parquet',
        type=str,
        default=None,
        help=f'Path to future period parquet for temporal validation. If not provided, auto-discovers val_win5.parquet or merged_panel_final.parquet (including $SCRATCH/data/). Use --no_future to disable.'
    )
    parser.add_argument(
        '--future_years',
        type=str,
        default=DEFAULT_FUTURE_YEARS_STR,
        help=f'Comma-separated future years for temporal validation (default: {DEFAULT_FUTURE_YEARS_STR}).'
    )
    parser.add_argument(
        '--no_future',
        action='store_true',
        help='Disable temporal validation (do not load or show future PA establishments on risk maps).'
    )
    parser.add_argument(
        '--split-version',
        type=str,
        choices=["main", "robustness"],
        default="main",
        help="Split version: 'main' or 'robustness' (default: main)"
    )
    parser.add_argument(
        '--allow-uncalibrated',
        action='store_true',
        help='Allow using uncalibrated scored parquet if no calibrated LGBM file is found (LGBM only)'
    )
    parser.add_argument(
        '--allow_test_shap',
        action='store_true',
        help='Allow SHAP analysis on test set if earlystop/train data unavailable (default: False)'
    )
    parser.add_argument(
        '--shap_n_samples',
        type=int,
        default=500,
        help='Maximum number of rows to sample for SHAP analysis (default: 500)'
    )
    parser.add_argument(
        '--skip_shap',
        action='store_true',
        help='Skip SHAP analysis (default: False)'
    )
    parser.add_argument(
        '--model_pkl',
        type=str,
        default=None,
        help=f'Path to the exact .pkl/.joblib that produced the scored parquet. If provided, SHAP uses this model (ensures explanations match results). If not set, auto-discovers {MODEL_ID}_* only (never Model C).'
    )
    parser.add_argument(
        '--gsn_tif',
        type=str,
        default=None,
        help='Path to gsn_terrestrial_ecoregions_mask_1km.tif. If not provided, auto-discovers from data/{region}/ready/GSN/ under $SCRATCH or repo root. Pass --skip_gsn to disable.'
    )
    parser.add_argument(
        '--gsn_shp',
        type=str,
        default=None,
        help='Path to Terrestrial_ecoregions.shp. If not provided, auto-discovers from data/shared/GlobalSafetyNet/terrestrial_ecoregions/ under repo root. Adds ECO_NAME and BIOME_NAME labels to biome breakdown.'
    )
    parser.add_argument(
        '--skip_gsn',
        action='store_true',
        help='Skip biome/GSN breakdown (default: False)'
    )
    args = parser.parse_args()
    
    # Optional W&B run (for streaming logs and metrics).
    # Default: disabled for local runs. Enable explicitly (e.g. in SLURM) via PA3030_WANDB=1.
    use_wandb = False
    wandb_enabled = os.environ.get("PA3030_WANDB", "0").strip().lower() not in {"0", "false", "no", "off"}
    if wandb_enabled and wandb is not None:
        try:
            run_name = f"results_{MODEL_ID}_{args.model_type}_{args.split_version}"
            wandb.init(
                project=f"ml-results-{MODEL_ID}",
                entity=os.environ.get("WANDB_ENTITY"),
                name=run_name,
                config={
                    "script": f"{MODEL_ID}_results",
                    "model_type": args.model_type,
                    "split_version": args.split_version,
                },
            )
            use_wandb = True
            print("W&B connected")
        except Exception as err:
            print(f"W&B failed: {err}")
    
    # Get repo root and scratch root for auto-discovery
    repo_root = get_repo_root()
    scratch_root = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    
    print("=" * 70)
    print("PATH RESOLUTION")
    print("=" * 70)
    print(f"Repository root: {repo_root}")
    if scratch_root:
        print(f"SCRATCH directory: {scratch_root}")
    else:
        print("SCRATCH directory: Not set (running locally)")
    print(f"Split version: {args.split_version}")
    print()
    
    # Check both $SCRATCH and repo root for ml_models directory
    # Build list of candidate directories to search (prioritize $SCRATCH, then repo root)
    ml_models_candidates = []
    if scratch_root is not None:
        ml_models_candidates.append(scratch_root / f"outputs/{REGION_SLUG}/results/ml_models")
    ml_models_candidates.append(repo_root / f"outputs/{REGION_SLUG}/results/ml_models")
    
    # Also check if repo_root can be determined from script location (like training scripts do)
    script_repo_root = get_repo_root()
    script_ml_models = script_repo_root / f"outputs/{REGION_SLUG}/results/ml_models"
    # Add if not already in the list (compare as strings to avoid Path comparison issues)
    if str(script_ml_models) not in [str(c) for c in ml_models_candidates]:
        ml_models_candidates.append(script_ml_models)
    
    print("Searching for ml_models directory in:")
    for cand in ml_models_candidates:
        exists = "✓" if cand.exists() else "✗"
        print(f"  {exists} {cand}")
    print()
    
    # Auto-discover files if not provided (search ALL candidate directories)
    if args.metrics_json:
        metrics_path = Path(args.metrics_json).resolve()
    else:
        print("Auto-discovering metrics JSON file...")
        # Search in all candidate directories (training scripts may save to repo root)
        # find_latest_file will use correct pattern based on model_type
        metrics_path = None
        for search_dir in ml_models_candidates:
            if not search_dir.exists():
                continue
            found = find_latest_file(f"{MODEL_ID}_{args.model_type}_win5_metrics_*.json", search_dir, args.split_version, args.model_type)
            if found is not None:
                metrics_path = found
                print(f"  Found: {metrics_path}")
                break
        
        if metrics_path is None:
            expected_pattern = f"{MODEL_ID}_lgbm_metrics_*.json" if args.model_type == 'lgbm' else f"{MODEL_ID}_rf_win5_metrics_*.json"
            print(f"ERROR: Could not find {expected_pattern} in any of the following directories:")
            for cand in ml_models_candidates:
                print(f"  - {cand}")
            print("Please provide --metrics_json path explicitly.")
            sys.exit(1)
    
    if args.scored_parquet:
        parquet_path = Path(args.scored_parquet).resolve()
    else:
        print("Auto-discovering scored parquet file...")
        parquet_path = None
        
        if args.model_type == 'lgbm':
            # For LGBM, prefer calibrated scores written by `calibrate_1`:
            #   outputs/south_america/results/ml_models/<split-version>/model1_lgbm_scored_calibrated_*.parquet
            # Fall back to raw scores only when --allow-uncalibrated is set.
            search_dirs_scored: list[Path] = []
            for base in ml_models_candidates:
                # Calibrated files live in split-specific subdirs; raw scores are in the root.
                search_dirs_scored.append(base / args.split_version)
                search_dirs_scored.append(base)
            
            calibrated_pattern = f"{MODEL_ID}_lgbm_scored_calibrated_*.parquet"
            raw_pattern = f"{MODEL_ID}_lgbm_scored_*.parquet"
            
            calibrated_path = find_latest_file_in_dirs(calibrated_pattern, search_dirs_scored)
            raw_path = find_latest_file_in_dirs(raw_pattern, search_dirs_scored, exclude_substr="calibrated")
            
            if calibrated_path is not None:
                parquet_path = calibrated_path
                print(f"  Found calibrated scored file (preferred): {parquet_path}")
            elif raw_path is not None:
                if args.allow_uncalibrated:
                    parquet_path = raw_path
                    print("  No calibrated scored file found.")
                    print("  Using uncalibrated scores because --allow-uncalibrated was provided:")
                    print(f"    {parquet_path}")
                else:
                    print("ERROR: No calibrated LGBM scored file found.")
                    print(f"  Expected pattern (calibrated, preferred): {MODEL_ID}_lgbm_scored_calibrated_*.parquet")
                    print("  A raw (uncalibrated) scored file was found:")
                    print(f"    {raw_path}")
                    print("  To proceed with uncalibrated scores, rerun with --allow-uncalibrated,")
                    print("  or pass --scored_parquet explicitly to override.")
                    sys.exit(1)
            else:
                print("ERROR: Could not find any LGBM scored parquet files.")
                print("  Looked for:")
                print(f"    - {MODEL_ID}_lgbm_scored_calibrated_*.parquet")
                print(f"    - {MODEL_ID}_lgbm_scored_*.parquet (excluding 'calibrated')")
                print("  in the following directories:")
                for cand in search_dirs_scored:
                    print(f"    - {cand}")
                print("Please provide --scored_parquet path explicitly.")
                sys.exit(1)
        else:
            # RF: prefer calibrated scored files (written by calibrate script into the
            # split-specific subdir); fall back to uncalibrated raw scores.
            search_dirs_scored_rf: list[Path] = []
            for base in ml_models_candidates:
                search_dirs_scored_rf.append(base / args.split_version)
                search_dirs_scored_rf.append(base)

            calibrated_pattern_rf = f"{MODEL_ID}_rf_win5_scored_calibrated_*.parquet"
            raw_pattern_rf = f"{MODEL_ID}_rf_win5_scored_*.parquet"

            calibrated_path_rf = find_latest_file_in_dirs(calibrated_pattern_rf, search_dirs_scored_rf)
            raw_path_rf = find_latest_file_in_dirs(raw_pattern_rf, search_dirs_scored_rf, exclude_substr="calibrated")

            if calibrated_path_rf is not None:
                parquet_path = calibrated_path_rf
                print(f"  Found calibrated scored file (preferred): {parquet_path}")
            elif raw_path_rf is not None:
                parquet_path = raw_path_rf
                print("  No calibrated RF scored file found; using uncalibrated scores:")
                print(f"    {parquet_path}")
            else:
                print(f"ERROR: Could not find {raw_pattern_rf} in any of the following directories:")
                for cand in search_dirs_scored_rf:
                    print(f"  - {cand}")
                print("Please provide --scored_parquet path explicitly.")
                sys.exit(1)
    
    # Determine ml_models_dir for output (use the directory where we found the files, or repo root as fallback)
    ml_models_dir = repo_root / f"outputs/{REGION_SLUG}/results/ml_models"
    if metrics_path:
        ml_models_dir = metrics_path.parent
    print(f"Using ml_models directory: {ml_models_dir}")
    print()
    
    # Auto-discover test parquet if not provided
    if args.test_parquet:
        test_parquet_path = Path(args.test_parquet).resolve()
        print(f"Using provided test parquet: {test_parquet_path}")
    else:
        print("Auto-discovering test_win5.parquet using $SCRATCH-first path resolution...")
        try:
            test_parquet_path = resolve_parquet_file("test_win5.parquet", args.split_version)
            print(f"  ✓ Resolved path: {test_parquet_path}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            print(f"  Risk map will use y_true only (may miss pixels established in test period)")
            print(f"  For accurate results, provide --test_parquet path to original test data")
            test_parquet_path = None
    
    # Future parquet and years for temporal validation (default: auto-discover val_win5.parquet, years 2020-2024)
    future_parquet_path = None
    future_years = None
    
    def _parse_future_years(s: str):
        try:
            return [int(y.strip()) for y in s.split(',')]
        except ValueError:
            return None
    
    if args.no_future:
        print("Temporal validation disabled (--no_future).")
    else:
        if args.future_parquet:
            future_parquet_path = Path(args.future_parquet).resolve()
            print(f"Using provided future parquet: {future_parquet_path}")
            if not future_parquet_path.exists():
                print(f"  WARNING: Future parquet not found: {future_parquet_path}")
                print(f"  Temporal validation will be disabled")
                future_parquet_path = None
        else:
            print("Auto-discovering future parquet for temporal validation (val_win5.parquet or merged_panel_final.parquet)...")
            try:
                future_parquet_path = resolve_future_parquet(args.split_version)
                print(f"  ✓ Resolved path: {future_parquet_path}")
            except FileNotFoundError as e:
                print(f"  ✗ {e}")
                print("  Risk maps will not include future PA layer. Provide --future_parquet or add val_win5.parquet / merged_panel_final.parquet to enable.")
                future_parquet_path = None
        
        if future_parquet_path is not None:
            future_years = _parse_future_years(args.future_years)
            if future_years is None or len(future_years) == 0:
                print(f"  WARNING: Invalid --future_years '{args.future_years}'; temporal validation disabled")
                future_parquet_path = None
                future_years = None
            else:
                print(f"  Future years for temporal validation: {future_years}")
    
    # Resolve paths
    metrics_path = metrics_path.resolve()
    parquet_path = parquet_path.resolve()
    if test_parquet_path:
        test_parquet_path = test_parquet_path.resolve()
    
    if args.out_dir:
        output_dir = Path(args.out_dir).resolve()
    else:
        # Prefer $SCRATCH for outputs on cluster, otherwise use repo root
        if scratch_root is not None:
            output_dir = scratch_root / f"outputs/{REGION_SLUG}/results/{MODEL_ID}_{args.model_type}"
        else:
            output_dir = repo_root / f"outputs/{REGION_SLUG}/results/{MODEL_ID}_{args.model_type}"
    
    region_boundary_path = None
    if args.region_boundary:
        region_boundary_path = Path(args.region_boundary).resolve()
        if not region_boundary_path.exists():
            print(f"WARNING: Region boundary file not found: {region_boundary_path}")
            print("  Will attempt to download from Natural Earth instead")
            region_boundary_path = None
    
    # Validate inputs
    if not metrics_path.exists():
        print(f"ERROR: Metrics file not found: {metrics_path}")
        sys.exit(1)
    
    if not parquet_path.exists():
        print(f"ERROR: Scored parquet file not found: {parquet_path}")
        sys.exit(1)
    
    # Create output directory (with error handling)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        print(f"ERROR: Could not create output directory {output_dir}: {e}")
        sys.exit(1)
    
    print("=" * 70)
    print(f"{MODEL_LABEL.upper()} ({args.model_type.upper()}) RESULTS REPORTING (Split: {args.split_version})")
    print("=" * 70)
    print(f"\nModel type: {args.model_type}")
    print(f"Split version: {args.split_version}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Scored parquet: {parquet_path}")
    print(f"Output directory: {output_dir}")
    if region_boundary_path:
        print(f"Region boundary: {region_boundary_path}")
    else:
        print("Region boundary: Will download from Natural Earth")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    metrics_data = load_metrics_json(metrics_path)
    if use_wandb:
        # Log top-level test performance metrics if available
        test_perf = metrics_data.get("test_performance") or metrics_data.get("metrics") or {}
        if isinstance(test_perf, dict):
            numeric_metrics = {
                f"results/{k}": v
                for k, v in test_perf.items()
                if isinstance(v, (int, float))
            }
            if numeric_metrics:
                wandb.log(numeric_metrics)
    # Derive test years early for predicate pushdown (reduces I/O for 47M+ row parquets)
    test_years = derive_test_years(metrics_data, None, test_parquet_path)
    print(f"\nDerived test years: {test_years}")
    df = load_scored_parquet(parquet_path, test_years=test_years)
    
    # If we're using a calibrated scored parquet (written by calibrate script),
    # enforce that key metadata columns are preserved. These are required for
    # correct spatial and temporal joins in downstream analyses.
    parquet_name_lower = parquet_path.name.lower()
    if ("calibrated" in parquet_name_lower) or ("y_pred_proba_calibrated" in df.columns):
        required_metadata_cols = ['year', 'row', 'col', 'x', 'y']
        missing_meta = [col for col in required_metadata_cols if col not in df.columns]
        if missing_meta:
            print("\nERROR: Calibrated scored parquet is missing required metadata columns.\n")
            print(f"  Expected `{CALIBRATE_SCRIPT}` to preserve the following columns when")
            print("  writing calibrated scores so that results scripts can perform")
            print("  spatial joins and temporal diagnostics correctly.")
            print(f"  Missing columns: {missing_meta}")
            print(f"  File: {parquet_path}")
            sys.exit(1)
    
    # Detect and reproject coordinates if needed
    df = detect_and_reproject_coordinates(df)
    validate_coordinates(df)
    
    # Load region boundary once and reuse for all map outputs (avoids 5x load/download)
    sa_gdf = get_region_boundary(region_boundary_path)

    # Resolve backbone raster once — used as pixel-perfect gray background in all maps.
    # Returns None gracefully if not available; each map then falls back to the Natural
    # Earth polygon so the script runs correctly on any machine.
    print("\nResolving backbone raster for map backgrounds...")
    backbone_path = find_backbone_path()

    # Risk map generation (1%, 5%, 10%) - run first to get capture metrics for tables
    thresholds = [1, 5, 10]
    future_metrics = None
    all_threshold_stats = []
    for threshold_pct in thresholds:
        result = create_risk_map(
            df,
            region_boundary_path,
            output_dir,
            args.model_type,
            metrics_data,
            test_parquet_path,
            test_years=test_years,
            threshold_pct=threshold_pct,
            future_parquet_path=future_parquet_path,
            future_years=future_years,
            sa_gdf=sa_gdf,
            backbone_path=backbone_path,
        )
        if result is not None:
            all_threshold_stats.append(result)
            if future_metrics is None and "future_capture_rate" in result:
                future_metrics = {
                    "future_capture_rate": result["future_capture_rate"],
                    "combined_recall": result["combined_recall"],
                }
    # Save capture summary CSV with absolute pixel counts and km²
    if all_threshold_stats:
        capture_cols = [
            "threshold_pct", "total_pa_pixels", "total_pa_km2",
            "captured_pixels", "captured_km2", "recall_pct", "precision_pct",
        ]
        capture_df = pd.DataFrame(all_threshold_stats)
        # Reorder to standard columns (add optional future cols at end if present)
        extra_cols = [c for c in capture_df.columns if c not in capture_cols]
        capture_df = capture_df[[c for c in capture_cols if c in capture_df.columns] + extra_cols]
        capture_csv = output_dir / "capture_summary.csv"
        capture_df.to_csv(capture_csv, index=False)
        print(f"\nSaved capture summary: {capture_csv}")
        print(capture_df.to_string(index=False))
    # Generate outputs (metrics table includes future capture when available)
    create_metrics_table(metrics_data, output_dir, args.model_type, df=df, extra_metrics=future_metrics)
    create_pr_curve(df, metrics_data, output_dir, args.model_type)
    create_roc_curve(df, output_dir, args.model_type)
    create_cumulative_gains_chart(df, output_dir, args.model_type)
    # Pass explicit test years to the row-level P@1% diagnostic map
    create_p1pct_diagnostic_map(
        df,
        region_boundary_path,
        output_dir,
        args.model_type,
        metrics_data,
        test_years=test_years,
        sa_gdf=sa_gdf,
        backbone_path=backbone_path,
    )
    # Also pass explicit test years to the probability map
    create_probability_map(
        df,
        region_boundary_path,
        output_dir,
        args.model_type,
        parquet_path,
        metrics_data,
        test_years=test_years,
        test_parquet_path=test_parquet_path,
        sa_gdf=sa_gdf,
        backbone_path=backbone_path,
    )

    # New outputs: calibration improvement, reliability diagram, country breakdown, biome breakdown
    create_calibration_improvement_figure(df, output_dir, args.model_type)
    create_calibration_curve(df, output_dir, args.model_type)
    create_country_breakdown(df, output_dir, args.model_type)

    # GSN biome breakdown: resolve raster path
    gsn_tif_path = None
    if not args.skip_gsn:
        if args.gsn_tif:
            gsn_tif_path = Path(args.gsn_tif).resolve()
        else:
            # Auto-discover: $SCRATCH first, then repo root
            gsn_candidates = []
            if scratch_root is not None:
                gsn_candidates.append(scratch_root / f"data/{REGION_SLUG}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif")
            gsn_candidates.append(repo_root / f"data/{REGION_SLUG}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif")
            for cand in gsn_candidates:
                if cand.exists():
                    gsn_tif_path = cand
                    print(f"  Auto-discovered GSN raster: {gsn_tif_path}")
                    break
            if gsn_tif_path is None:
                print("  GSN raster not found in standard locations. Pass --gsn_tif to enable biome breakdown.")
    # Resolve GSN shapefile path
    gsn_shp_path = None
    if not args.skip_gsn:
        if args.gsn_shp:
            gsn_shp_path = Path(args.gsn_shp).resolve()
        else:
            # Auto-discover from repo root (shared across all regions)
            shp_candidate = repo_root / "data/shared/GlobalSafetyNet/terrestrial_ecoregions/Terrestrial_ecoregions.shp"
            if shp_candidate.exists():
                gsn_shp_path = shp_candidate
                print(f"  Auto-discovered GSN shapefile: {gsn_shp_path}")
            else:
                print("  GSN shapefile not found. Pass --gsn_shp to enable biome names.")

    create_biome_breakdown(df, output_dir, args.model_type, gsn_tif_path=gsn_tif_path, gsn_shp_path=gsn_shp_path)

    # ── New spatial analyses ───────────────────────────────────────────────────
    create_false_positive_map(df, output_dir, args.model_type, region_gdf=sa_gdf)
    create_economic_exposure(df, test_parquet_path, output_dir, args.model_type)
    create_hotspot_maps(df, output_dir, args.model_type, region_gdf=sa_gdf)

    # Cross-region comparison table (reads the other region's metrics_table.csv if available)
    create_comparison_table(output_dir, args.model_type, repo_root, scratch_root)
    # Cross-region scatter: biome-level precision@1% vs recall@1% (needs spatial_CV_2 outputs)
    create_biome_scatter_comparison(output_dir, repo_root, scratch_root)
    # Cross-model landscape: precision-decay curves + ROC vs lift scatter (all 3 regions × 2 algos)
    create_model_comparison_landscape(output_dir, repo_root, scratch_root)

    # Resolve model file for SHAP: explicit --model_pkl, or auto-discover configured model only.
    model_path = None
    if not args.skip_shap:
        if args.model_pkl:
            model_path = Path(args.model_pkl).resolve()
            if not model_path.exists():
                print(f"  ERROR: --model_pkl file not found: {model_path}")
                model_path = None
            elif "modelC" in model_path.name:
                print(f"  ERROR: --model_pkl must be the {MODEL_LABEL} artifact that produced the scored parquet, not Model C: {model_path.name}")
                model_path = None
            else:
                print(f"Using provided model for SHAP: {model_path}")
        else:
            print(f"\nAuto-discovering trained model file for SHAP analysis ({MODEL_LABEL} only)...")
            # Only model-specific patterns: never use *model*.pkl so we never pick modelC_lgbm_*.pkl
            model_patterns = [
                f"{MODEL_ID}_{args.model_type}_win5_*.pkl",
                f"{MODEL_ID}_{args.model_type}_win5_*.joblib",
                f"{MODEL_ID}_{args.model_type}_*.pkl",
                f"{MODEL_ID}_{args.model_type}_*.joblib",
            ]

            # Check $SCRATCH first, then local repo (./data/<region>/ml/models/) so SHAP works without SCRATCH
            model_candidates = []
            if scratch_root is not None:
                model_candidates.append(scratch_root / f"data/{REGION_SLUG}/ml/models" / args.split_version)
                model_candidates.append(scratch_root / f"data/{REGION_SLUG}/ml/models")
            model_candidates.append(repo_root / f"data/{REGION_SLUG}/ml/models" / args.split_version)
            model_candidates.append(repo_root / f"data/{REGION_SLUG}/ml/models")
            # Also search where metrics/parquet live (ml_models) in case .pkl is saved there
            for base in ml_models_candidates:
                if base.exists():
                    model_candidates.append(base)
                    model_candidates.append(base / args.split_version)
            # Local paths when $SCRATCH is missing: cwd and repo-relative
            cwd_models = Path.cwd() / "data" / "ml" / "models"
            for extra in (cwd_models, cwd_models / args.split_version):
                if extra.exists() and extra not in model_candidates:
                    model_candidates.append(extra)

            # Prefer a model whose filename contains the same timestamp as the metrics/parquet run
            run_ts = metrics_data.get("metadata", {}).get("timestamp", "") or ""
            if not run_ts and parquet_path:
                # e.g. modelX_lgbm_scored_calibrated_20260119_234939.parquet -> 20260119_234939
                m = re.search(r"(\d{8}_\d{6})", parquet_path.name)
                if m:
                    run_ts = m.group(1)

            all_matches: list[Path] = []
            for search_dir in model_candidates:
                if not search_dir.exists():
                    continue
                for model_pattern in model_patterns:
                    try:
                        matches = list(search_dir.glob(model_pattern))
                    except (OSError, PermissionError):
                        continue
                    for p in matches:
                        if "modelC" in p.name:
                            continue
                        all_matches.append(p)

            # Deduplicate by resolved path (same file may appear from multiple search dirs)
            seen = set()
            unique_matches = []
            for p in all_matches:
                key = p.resolve()
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(p)
            all_matches = unique_matches

            if all_matches:
                # Prefer file whose name contains run_ts (same run as parquet/metrics)
                if run_ts:
                    ts_matches = [p for p in all_matches if run_ts in p.name]
                    if ts_matches:
                        ts_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        model_path = ts_matches[0]
                if model_path is None:
                    all_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    model_path = all_matches[0]
                print(f"  Found: {model_path}")
            else:
                pattern_str = " or ".join(model_patterns)
                print(f"  WARNING: Could not find trained model file matching {pattern_str} ({MODEL_LABEL} only; Model C is never used for this script)")
                print(f"  To run SHAP with the exact model that produced the parquet, pass --model_pkl /path/to/{MODEL_ID}_*.pkl")
                print(f"  SHAP analysis will be skipped")
    
    # Auto-discover data sources for SHAP (earlystop, train)
    shap_data_sources = {}
    if not args.skip_shap and model_path is not None:
        print("\nAuto-discovering data sources for SHAP analysis...")
        
        # Try to find earlystop_win5.parquet
        earlystop_candidates = []
        if scratch_root is not None:
            earlystop_candidates.extend([
                scratch_root / f"outputs/{REGION_SLUG}/results/{args.split_version}/earlystop_win5.parquet",
                scratch_root / f"data/{REGION_SLUG}/ml/{args.split_version}/earlystop_win5.parquet"
            ])
        earlystop_candidates.extend([
            repo_root / f"outputs/{REGION_SLUG}/results/{args.split_version}/earlystop_win5.parquet",
            repo_root / f"data/{REGION_SLUG}/ml/{args.split_version}/earlystop_win5.parquet"
        ])
        
        for cand in earlystop_candidates:
            if cand.exists():
                shap_data_sources['earlystop'] = cand
                print(f"  Found earlystop: {cand}")
                break
        
        # Try to find train_win5.parquet
        train_candidates = []
        if scratch_root is not None:
            train_candidates.extend([
                scratch_root / f"outputs/{REGION_SLUG}/results/{args.split_version}/train_win5.parquet",
                scratch_root / f"data/{REGION_SLUG}/ml/{args.split_version}/train_win5.parquet"
            ])
        train_candidates.extend([
            repo_root / f"outputs/{REGION_SLUG}/results/{args.split_version}/train_win5.parquet",
            repo_root / f"data/{REGION_SLUG}/ml/{args.split_version}/train_win5.parquet"
        ])
        
        for cand in train_candidates:
            if cand.exists():
                shap_data_sources['train'] = cand
                print(f"  Found train: {cand}")
                break
        
        # Add test as fallback (only used if --allow_test_shap)
        if test_parquet_path is not None:
            shap_data_sources['test'] = test_parquet_path
            if args.allow_test_shap:
                print(f"  Found test (allowed with --allow_test_shap): {test_parquet_path}")
            else:
                print(f"  Found test (disabled, use --allow_test_shap to enable): {test_parquet_path}")
    
    # SHAP analysis (if model and data sources available)
    if not args.skip_shap and model_path is not None and shap_data_sources:
        # Extract timestamp from metrics or use current time
        timestamp = metrics_data.get("metadata", {}).get("timestamp", "")
        if not timestamp:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract feature columns from metrics
        feature_cols = metrics_data.get("metadata", {}).get("features", [])
        if not feature_cols:
            print("\n" + "!" * 70)
            print("WARNING: Feature columns not found in metrics JSON")
            print("SHAP analysis will be skipped")
            print("!" * 70)
        else:
            try:
                compute_shap_analysis(
                    model_path=model_path,
                    feature_cols=feature_cols,
                    data_sources=shap_data_sources,
                    output_dir=output_dir,
                    model_type=args.model_type,
                    timestamp=timestamp,
                    allow_test_shap=args.allow_test_shap,
                    shap_n_samples=args.shap_n_samples,
                )
            except RuntimeError as e:
                # Handle SHAP not installed more gracefully
                if "SHAP is not installed" in str(e):
                    print("\n" + "!" * 70)
                    print("WARNING: SHAP analysis skipped - SHAP is not installed")
                    print("!" * 70)
                    print(f"  {e}")
                    print("\n  To enable SHAP analysis, install SHAP:")
                    print("    conda install -c conda-forge shap")
                    print("    or: pip install shap")
                    print("!" * 70)
                else:
                    # Re-raise other RuntimeErrors
                    raise
            except Exception as e:
                print("\n" + "!" * 70)
                print(f"WARNING: SHAP analysis failed: {e}")
                print("!" * 70)
                import traceback
                traceback.print_exc()
    elif args.skip_shap:
        print("\n" + "=" * 70)
        print("SHAP ANALYSIS SKIPPED (--skip_shap flag set)")
        print("=" * 70)
    elif model_path is None:
        print("\n" + "=" * 70)
        print("SHAP ANALYSIS SKIPPED (model file not found)")
        print("=" * 70)
    elif not shap_data_sources:
        print("\n" + "=" * 70)
        print("SHAP ANALYSIS SKIPPED (no data sources found)")
        print("=" * 70)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

