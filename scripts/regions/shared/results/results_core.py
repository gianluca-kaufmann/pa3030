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
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import interp1d, griddata

from .boundaries import (
    detect_and_reproject_coordinates,
    get_region_boundary,
    validate_coordinates,
)
from .config import (
    CALIBRATE_SCRIPT,
    DEFAULT_FUTURE_YEARS_STR,
    MODEL_ID,
    MODEL_LABEL,
    REGION_LABEL,
    REGION_SLUG,
    X_LIMITS,
    Y_LIMITS,
    get_repo_root,
)
from .io import (
    derive_test_years,
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

# Probability map color scheme
PROBABILITY_MAP_COLORMAP = 'plasma'  # Colormap: dark purple -> yellow

# Probability map color scaling (percentile-based)
PROBABILITY_MAP_PERCENTILE_MIN = 25  # Lower percentile for colorbar range
PROBABILITY_MAP_PERCENTILE_MAX = 98  # Upper percentile for colorbar range
PROBABILITY_MAP_TRANSFORMATION = 'sqrt'  # Transformation: 'sqrt' (square root) or 'linear'

# PR curve figure dimensions
PR_CURVE_FIGSIZE = (8, 6)
PR_CURVE_DPI = 300


















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
        ("Precision @ 1%", test_perf.get("precision_at_1pct", np.nan)),
        ("Precision @ 5%", test_perf.get("precision_at_5pct", np.nan)),
        ("Precision @ 10%", test_perf.get("precision_at_10pct", np.nan)),
        ("Baseline Rate", test_perf.get("baseline_rate", np.nan)),
        ("Lift @ 1%", test_perf.get("lift_at_1pct", np.nan)),
        ("Lift @ 5%", test_perf.get("lift_at_5pct", np.nan)),
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
    
    fig, ax = plt.subplots(figsize=PR_CURVE_FIGSIZE)
    ax.plot(pct_area_searched, pct_pas_captured, linewidth=2, label='Model')
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










def points_to_raster(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                     target_resolution: Optional[float] = None,
                     agg_func: str = 'mean') -> tuple[np.ndarray, tuple]:
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
    
    Returns:
        Tuple of (raster_array, extent) where extent is (xmin, xmax, ymin, ymax)
        Raster array has shape (nrows, ncols) and may contain NaN for empty cells
    """
    # Calculate bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
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
    # Extend slightly beyond bounds to ensure all points are included
    padding = max(x_res, y_res) * 0.1
    x_edges = np.arange(x_min - padding, x_max + padding + x_res, x_res)
    y_edges = np.arange(y_min - padding, y_max + padding + y_res, y_res)
    
    # Calculate grid dimensions
    ncols = len(x_edges) - 1
    nrows = len(y_edges) - 1
    
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
        raster_flat = np.where(count_arr > 0, sum_arr / count_arr, np.nan)
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
) -> Optional[Dict[str, float]]:
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
    
    print(f"  South America boundary CRS: {sa_gdf.crs}")
    print(f"  South America boundary bounds: {sa_gdf.total_bounds}")
    
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
    
    # Define observed establishments: PREFER actual test-period data, ONLY fall back to y_true if unavailable
    if has_actual_establishments:
        df['is_observed'] = (df['established_in_test_period'] == 1).astype(int)
        print(f"  ✓ Using actual test-period establishments (established_in_test_period)")
    else:
        df['is_observed'] = (df['y_true'] == 1).astype(int)
        print(f"  ⚠ Falling back to y_true (5-year lookahead) - actual establishment data not available")
    
    # Update gdf with observed flag
    gdf['is_observed'] = df['is_observed'].values
    
    # Step 1b: Load FUTURE PA establishments (2020-2024) if provided for temporal validation
    has_future_establishments = False
    if future_parquet_path is not None and future_years is not None and len(future_years) > 0:
        future_df = load_future_pa_establishments(future_parquet_path, future_years)
        
        if future_df is not None and 'established_in_future_period' in future_df.columns:
            print(f"  ✓ Future PA establishments loaded - temporal validation enabled")
            
            # Determine join columns (same logic as test period)
            join_cols = ['row', 'col'] if all(col in df.columns for col in ['row', 'col']) and all(col in future_df.columns for col in ['row', 'col']) else ['x', 'y']
            
            if join_cols == ['row', 'col']:
                # Row/col join (preferred)
                join_df = future_df[['row', 'col', 'established_in_future_period']].drop_duplicates(subset=['row', 'col'])
                df = df.merge(join_df, on=['row', 'col'], how='left')
                df['established_in_future_period'] = df['established_in_future_period'].fillna(0).astype(int)
                has_future_establishments = True
                print(f"    Joined via row/col: {(df['established_in_future_period'] == 1).sum():,} future establishments")
            elif join_cols == ['x', 'y']:
                # X/y join with rounding (fallback)
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
                print(f"    Joined via x/y: {(df['established_in_future_period'] == 1).sum():,} future establishments")
    
    # Initialize future establishments flag if not loaded
    if not has_future_establishments:
        df['established_in_future_period'] = 0
    
    # Update gdf with future establishments flag
    gdf['is_future'] = df['established_in_future_period'].values
    
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
    # For observations: take MAX (observed if established in ANY test year)
    # For future: take MAX (future establishment if established in ANY future year)
    pixel_agg = gdf.groupby(['row', 'col']).agg({
        'y_pred_proba': 'max',  # Highest predicted probability for this pixel across test years
        'is_observed': 'max',   # 1 if established in any test year, 0 otherwise
        'is_future': 'max',     # 1 if established in any future year, 0 otherwise
        'x_proj': 'first',      # Projected coordinates (meters)
        'y_proj': 'first'
    }).reset_index()
    
    print(f"    {len(gdf):,} rows → {len(pixel_agg):,} unique pixels")
    
    # Total actual establishments (computed once for Recall across all thresholds)
    total_actual_establishments = int(pixel_agg['is_observed'].sum())
    total_future_establishments = int(pixel_agg['is_future'].sum())
    
    # Step 4: Calculate top-X% threshold at PIXEL level
    top_pct_threshold = np.percentile(pixel_agg['y_pred_proba'], 100 - threshold_pct)
    pixel_agg['is_predicted'] = (pixel_agg['y_pred_proba'] >= top_pct_threshold).astype(int)
    
    print(f"  ✓ Top-{threshold_pct}% threshold (pixel-level): {top_pct_threshold:.6f}")
    
    # Step 5: Compute statistics from pixel_agg (unique physical pixels by row,col)
    predicted_count = int(pixel_agg['is_predicted'].sum())
    observed_count = int(pixel_agg['is_observed'].sum())
    overlap_count = int(((pixel_agg['is_predicted'] == 1) & (pixel_agg['is_observed'] == 1)).sum())
    predicted_only_count = predicted_count - overlap_count
    observed_only_count = observed_count - overlap_count
    future_count = int(pixel_agg['is_future'].sum())
    future_overlap_count = int(((pixel_agg['is_predicted'] == 1) & (pixel_agg['is_future'] == 1)).sum())

    recall_pct = (overlap_count / total_actual_establishments * 100) if total_actual_establishments > 0 else 0.0
    
    # Calculate Future Capture Rate: % of future (2020-2024) PAs that fell in predicted high-risk areas
    future_capture_rate = (future_overlap_count / total_future_establishments * 100) if total_future_establishments > 0 else 0.0
    
    # Calculate Combined Recall: % of ALL PAs (test + future) captured by predictions
    total_combined_establishments = total_actual_establishments + total_future_establishments
    combined_overlap_count = overlap_count + future_overlap_count
    combined_recall_pct = (combined_overlap_count / total_combined_establishments * 100) if total_combined_establishments > 0 else 0.0

    print(f"\n  FINAL STATISTICS (pixel-level, after filtering):")
    print(f"    Total pixels: {len(pixel_agg):,}")
    print(f"    Predicted high-risk (top {threshold_pct}%): {predicted_count:,}")
    print(f"    Actual PA establishments (test period): {observed_count:,}")
    print(f"    Overlap (correct predictions): {overlap_count:,}")
    if predicted_count > 0:
        hit_rate = overlap_count / predicted_count * 100
        print(f"    Hit rate: {overlap_count:,}/{predicted_count:,} = {hit_rate:.1f}%")
    else:
        hit_rate = 0.0
        print(f"    Hit rate: N/A (no predictions)")
    print(f"    Recall (test period): {overlap_count:,}/{total_actual_establishments:,} = {recall_pct:.1f}%")
    
    # Print future PA statistics if available
    if has_future_establishments and total_future_establishments > 0:
        future_year_str = f"{min(future_years)}-{max(future_years)}"
        print(f"\n  TEMPORAL VALIDATION (Future PA Establishments {future_year_str}):")
        print(f"    Future PA establishments: {future_count:,}")
        print(f"    Future establishments in predicted areas: {future_overlap_count:,}")
        print(f"    Future Capture Rate: {future_overlap_count:,}/{total_future_establishments:,} = {future_capture_rate:.1f}%")
        print(f"\n  COMBINED STATISTICS (Test + Future):")
        print(f"    Total PA establishments (test + future): {total_combined_establishments:,}")
        print(f"    Total captured by predictions: {combined_overlap_count:,}")
        print(f"    Combined Recall: {combined_overlap_count:,}/{total_combined_establishments:,} = {combined_recall_pct:.1f}%")

    # Step 6: Build category raster (pixel-perfect, same resolution as probability map)
    # Category: 0=background, 1=predicted_only, 2=observed_only, 3=overlap, 4=future_only
    # Priority: future (4) > overlap (3) > observed (2) > predicted (1) > background (0)
    # This ensures future PAs are visible even if they overlap with other categories
    category_codes = np.where(
        pixel_agg['is_future'] == 1, 4,  # Future PAs take highest priority (yellow)
        np.where(
            (pixel_agg['is_predicted'] == 1) & (pixel_agg['is_observed'] == 1), 3,  # Overlap (green)
            np.where(pixel_agg['is_observed'] == 1, 2,  # Observed only (orange)
                     np.where(pixel_agg['is_predicted'] == 1, 1, 0))))  # Predicted only (blue) or background
    
    print(f"\n  Creating rasterized risk map...")
    category_raster, extent = points_to_raster(
        pixel_agg['x_proj'].values,
        pixel_agg['y_proj'].values,
        category_codes.astype(np.float32),
        target_resolution=1000.0,  # 1000 meters resolution
        agg_func='max'
    )
    print(f"  Category raster shape: {category_raster.shape}, extent: {extent}")
    
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

    # Plot background mask from actual data (shows data coverage; protected areas = natural holes)
    background_masked = np.ma.masked_where(np.isnan(background_mask), background_mask)
    ax.imshow(background_masked, extent=extent, cmap=ListedColormap(['#F5F5F5']),
              vmin=0, vmax=1, origin='upper', interpolation='nearest', 
              aspect='equal', zorder=1)

    # ListedColormap for 3-4 categories (Predicted, Observed, Overlap, Future)
    COLOR_PREDICTED_ONLY = '#5B9BD5'  # Lighter, more visible blue
    COLOR_OBSERVED_ONLY = '#D95A3C'   # Warm orange-red
    COLOR_OVERLAP = '#2EBD8A'         # Slightly lighter green, more saturated
    COLOR_FUTURE = '#FFD700'          # Bright yellow for future PAs
    
    # Use 4 colors if future establishments exist, otherwise 3
    if has_future_establishments and future_count > 0:
        cmap_categories = ListedColormap([COLOR_PREDICTED_ONLY, COLOR_OBSERVED_ONLY, COLOR_OVERLAP, COLOR_FUTURE])
        norm_categories = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap_categories.N)
        max_category = 4
    else:
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

    # Styling
    ax.set_xlabel('Easting (m, EPSG:3857)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Northing (m, EPSG:3857)', fontsize=FONTSIZE_LABEL)
    establishment_source = "Actual Establishments" if has_actual_establishments else "5-yr Lookahead Targets"
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}): Predicted Risk vs {establishment_source} ({time_period})',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=15)

    # Legend with counts from pixel_agg (unique physical pixels)
    legend_elements = []
    if predicted_only_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_PREDICTED_ONLY,
                                      markersize=10, alpha=0.9, label=f'Predicted future PA candidates (n={predicted_only_count:,})'))
    if observed_only_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_OBSERVED_ONLY,
                                      markersize=10, alpha=0.9, label=f'Established (not predicted) (n={observed_only_count:,})'))
    if overlap_count > 0:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_OVERLAP,
                                      markersize=10, alpha=0.9, label=f'Correct predictions (n={overlap_count:,})'))
    if has_future_establishments and future_count > 0:
        future_year_str = f"{min(future_years)}-{max(future_years)}"
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_FUTURE,
                                      markersize=10, alpha=0.9, label=f'Future PAs {future_year_str} (n={future_count:,})'))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=FONTSIZE_LEGEND, framealpha=0.95)

    # Stats box with key metrics
    stats_lines = [
        f"Pixels: {len(pixel_agg):,}",
        f"Top {int(threshold_pct)}% threshold: {top_pct_threshold:.4f}",
        f"Hit rate: {overlap_count:,}/{predicted_count:,} = {hit_rate:.1f}%",
        f"Recall (test): {overlap_count:,}/{total_actual_establishments:,} = {recall_pct:.1f}%"
    ]
    
    # Add future metrics if available
    if has_future_establishments and total_future_establishments > 0:
        stats_lines.append(f"Future Capture Rate: {future_overlap_count:,}/{total_future_establishments:,} = {future_capture_rate:.1f}%")
        stats_lines.append(f"Combined Recall: {combined_overlap_count:,}/{total_combined_establishments:,} = {combined_recall_pct:.1f}%")
    
    stats_text = "\n".join(stats_lines)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=FONTSIZE_STATS,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat',
           alpha=0.9, edgecolor='black', linewidth=1), zorder=10)

    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=3)

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

    # Return future capture metrics for inclusion in metrics table (when available)
    if has_future_establishments and total_future_establishments > 0:
        return {
            "future_capture_rate": future_capture_rate,
            "combined_recall": combined_recall_pct,
        }
    return None


def create_p1pct_diagnostic_map(
    df: pd.DataFrame,
    region_boundary_path: Optional[Path],
    output_dir: Path,
    model_type: str,
    metrics_data: Optional[Dict[str, Any]] = None,
    test_years: Optional[list] = None,
    sa_gdf: Optional[gpd.GeoDataFrame] = None,
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
    
    # Get South America boundary (use cached if provided)
    if sa_gdf is None:
        sa_gdf = get_region_boundary(region_boundary_path)
    
    # Ensure CRS is set (assume EPSG:4326 if not set)
    if sa_gdf.crs is None:
        sa_gdf.set_crs('EPSG:4326', inplace=True)
    
    print(f"  South America boundary CRS: {sa_gdf.crs}")
    
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
    
    # Use argpartition to get top 1% indices (same as training script)
    top_k_idx = np.argpartition(y_proba, -n_top_k)[-n_top_k:]
    top_k_threshold = np.min(y_proba[top_k_idx])
    print(f"  Top 1% threshold: {top_k_threshold:.6f}")
    
    # Mark rows as top-1% (row-level, no aggregation)
    gdf['is_top1pct'] = df['y_pred_proba'] >= top_k_threshold
    
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
    print(f"  Row-level P@1% metric: {actual_p1pct:.4f} ({overlap_count:,}/{n_top_k:,})")
    print(f"  Predicted high-risk only: {predicted_only_count:,}")
    print(f"  Observed positives only: {observed_only_count:,}")
    print(f"  Overlap (same row): {overlap_count:,}")
    
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

    # Styling
    ax.set_xlabel('Easting (m, EPSG:3857)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Northing (m, EPSG:3857)', fontsize=FONTSIZE_LABEL)
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
                                    alpha=1.0, label=f'Observed positives only (n={observed_only_count:,} rows)'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=FONTSIZE_LEGEND, 
              framealpha=0.95, fancybox=True, shadow=True)
    
    # Add time period text box
    time_text = f"Observation period: {time_period}\nPrediction horizon: {prediction_horizon}\nRow-level (no pixel deduplication)"
    ax.text(0.02, 0.02, time_text, transform=ax.transAxes, fontsize=FONTSIZE_STATS,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', 
           alpha=0.8, edgecolor='black', linewidth=1), zorder=10)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=3)

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
    
    # Get South America boundary (use cached if provided)
    if sa_gdf is None:
        sa_gdf = get_region_boundary(region_boundary_path)
    
    # Ensure CRS is set (assume EPSG:4326 if not set)
    if sa_gdf.crs is None:
        sa_gdf.set_crs('EPSG:4326', inplace=True)
    
    print(f"  South America boundary CRS: {sa_gdf.crs}")
    
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
    
    # Use percentile-based range to balance color distribution
    # This clips extreme outliers and better utilizes the full colormap
    proba_pmin = np.percentile(proba_values, PROBABILITY_MAP_PERCENTILE_MIN)
    proba_pmax = np.percentile(proba_values, PROBABILITY_MAP_PERCENTILE_MAX)
    
    print(f"  Using percentile-based color range ({PROBABILITY_MAP_PERCENTILE_MIN}th-{PROBABILITY_MAP_PERCENTILE_MAX}th percentile)")
    print(f"  Probability range: [{proba_min:.6f}, {proba_max:.6f}]")
    print(f"  Colorbar range: [{proba_pmin:.6f}, {proba_pmax:.6f}]")
    print(f"  Range span: {proba_pmax - proba_pmin:.6f}")
    
    # Clip values to percentile range
    proba_clipped = np.clip(proba_values, proba_pmin, proba_pmax)
    
    # Apply transformation to stretch the distribution
    proba_shifted = proba_clipped - proba_pmin  # Shift to start at 0
    proba_span = proba_pmax - proba_pmin
    proba_normalized_linear = proba_shifted / proba_span  # Linear normalization to [0, 1]
    
    # Apply transformation based on constant setting
    if PROBABILITY_MAP_TRANSFORMATION == 'sqrt':
        proba_normalized = np.sqrt(proba_normalized_linear)  # Square root stretches lower values
        transform_name = 'square root'
    else:  # linear
        proba_normalized = proba_normalized_linear
        transform_name = 'linear'
    
    print(f"  Applied {transform_name} transformation to stretch distribution")
    print(f"  Normalized value range: [{proba_normalized.min():.4f}, {proba_normalized.max():.4f}]")
    print(f"  Normalized value mean: {proba_normalized.mean():.4f}, median: {np.median(proba_normalized):.4f}")
    
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
    
    # Recompute proba_normalized from filtered data (using same percentile thresholds from original data)
    proba_values_filtered = df['y_pred_proba'].values
    proba_clipped_filtered = np.clip(proba_values_filtered, proba_pmin, proba_pmax)
    proba_shifted_filtered = proba_clipped_filtered - proba_pmin
    proba_normalized_linear = proba_shifted_filtered / proba_span
    
    if PROBABILITY_MAP_TRANSFORMATION == 'sqrt':
        proba_normalized = np.sqrt(proba_normalized_linear)
    else:
        proba_normalized = proba_normalized_linear
    
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

    # Clean white background: plot South America boundary as solid base
    sa_gdf_proj.plot(ax=ax, color='white', edgecolor='none', linewidth=1.5, zorder=1)
    
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

    # Add colorbar with custom formatter to show actual probability values
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Predicted Probability', 
                   fontsize=FONTSIZE_LABEL, rotation=270, labelpad=20)
    
    # Set colorbar ticks with text labels instead of numeric values
    tick_positions = [0.0, 1.0]
    tick_labels = ['Low probability', 'High probability']
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=FONTSIZE_LEGEND)
    
    # Styling
    ax.set_xlabel('Easting (m, EPSG:3857)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Northing (m, EPSG:3857)', fontsize=FONTSIZE_LABEL)
    
    # Check if calibrated probabilities were used
    calibration_note = ""
    has_uncalibrated_col = 'y_pred_proba_uncalibrated' in df.columns
    has_calibrated_col = 'y_pred_proba_calibrated' in df.columns
    filename_has_calibrated = 'calibrated' in parquet_path.name.lower()
    
    if filename_has_calibrated or (has_uncalibrated_col and has_calibrated_col):
        calibration_note = " (Calibrated Probabilities)"
    
    ax.set_title(f'{MODEL_LABEL} ({model_type.upper()}) Probability Map{calibration_note}',
                 fontsize=FONTSIZE_TITLE, fontweight='bold', pad=15)
    
    # Add simple description (removed verbose text)
    # Colorbar already shows "Low probability" to "High probability"
    # Title already shows what the map represents
    # No need for additional text boxes
    
    # Add statistics text box
    stats_text = (f"Min: {proba_min:.6f} | Max: {proba_max:.6f}\n"
                  f"Mean: {proba_mean:.6f} | Median: {proba_median:.6f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=FONTSIZE_STATS,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
           alpha=0.9, edgecolor='black', linewidth=1), zorder=10)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=3)

    plt.tight_layout(pad=0.5)

    # Save PDF (lower DPI + retry to avoid timeout/zlib errors on large raster)
    pdf_path = output_dir / f"probability_map.pdf"
    for pdf_dpi in (300, 150):
        try:
            plt.savefig(pdf_path, dpi=pdf_dpi, bbox_inches='tight')
            print(f"Saved PDF: {pdf_path} (dpi={pdf_dpi})")
            break
        except (TimeoutError, zlib.error, OSError) as e:
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
            if pdf_dpi == 150:
                print(f"Warning: Could not save PDF after retry: {e}. Skipping PDF.")
            continue
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
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pos, X_shap, feature_names=feature_cols, show=False, plot_type="dot")
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
    
    # SHAP dependence plots for top 5 features (feature value vs SHAP value)
    top5_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    for feat_idx in top5_indices:
        feat_name = feature_cols[feat_idx]
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(feat_name))
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat_idx, shap_values_pos, X_shap, feature_names=feature_cols, show=False)
        plt.tight_layout()
        dep_path = output_dir / f"shap_dep_{safe_name}.pdf"
        plt.savefig(dep_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved SHAP dependence: {dep_path.name}")
    
    # Thematic feature importance: group features by prefix and aggregate mean_abs_shap
    FEATURE_THEME_PREFIXES = (
        ('dist_', 'Proximity'),
        ('topo_', 'Terrain'),
        ('socio_', 'Socio-economic'),
    )
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
    fig, ax = plt.subplots(figsize=(8, max(4, len(themes) * 0.4)))
    ax.barh(themes, sums, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(themes))))
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
        import geopandas as gpd
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
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
        joined = gpd.sjoin(pixel_gdf, world_4326[['name', 'continent', 'geometry']],
                          how='left', predicate='within')
        joined['country'] = joined['name'].fillna('Unknown')
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

    # Save LaTeX table (top 15 countries by n_observed_PAs)
    top_df = result_df.head(15).copy()
    top_df.columns = ['Country', 'N Pixels', 'N Observed PAs', 'ROC-AUC', 'P@1\\%']
    latex_str = top_df.to_latex(index=False, float_format='%.4f',
                                 caption=f'Per-country model performance ({MODEL_LABEL}, {model_type.upper()})',
                                 label=f'tab:country_breakdown_{model_type}',
                                 escape=False)
    latex_path = output_dir / 'country_breakdown.tex'
    latex_path.write_text(latex_str)
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
        biome_records.append({
            'BIOME_NAME':     biome_name,
            'n_ecoregions':   len(grp_df),
            'n_pixel_years':  n_rows,
            'n_observed_PAs': n_pos,
            'prevalence_pct': round(n_pos / n_rows * 100, 4),
            'ROC_AUC':        round(roc_auc, 4),
            'Precision_at_1pct': round(p_at_1pct, 4),
        })

    biome_df = pd.DataFrame(biome_records).sort_values('ROC_AUC', ascending=True)
    biome_csv = output_dir / 'biome_breakdown.csv'
    biome_df.to_csv(biome_csv, index=False)
    print(f"  Saved: {biome_csv}  ({len(biome_df)} biomes)")

    result_df = biome_df  # used for plotting below
    print(f"  Ecoregions with sufficient data: {len(eco_df)}")

    # Save CSV
    csv_path = output_dir / 'biome_breakdown.csv'
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

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

    labels = [SHORT_BIOME.get(b, b[:30]) for b in biome_df['BIOME_NAME']]
    roc_vals = biome_df['ROC_AUC'].tolist()
    p1_vals  = biome_df['Precision_at_1pct'].tolist()
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
    for bar, val in zip(bars1, roc_vals):
        ax1.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=FONTSIZE_STATS)

    # P@1% panel
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, p1_vals, color='#E07B54', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=FONTSIZE_STATS)
    ax2.set_xlabel('Precision@1%', fontsize=FONTSIZE_LABEL)
    ax2.set_title('Precision@1% by Biome', fontsize=FONTSIZE_TITLE, fontweight='bold')
    p1_max = max(p1_vals) if p1_vals else 1
    ax2.set_xlim(0, p1_max * 1.3)
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars2, p1_vals):
        ax2.text(val + p1_max * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=FONTSIZE_STATS)

    plt.suptitle(f'{MODEL_LABEL} ({model_type.upper()}) Performance by Biome (GSN Terrestrial Ecoregions)',
                fontsize=FONTSIZE_TITLE, fontweight='bold', y=1.01)
    plt.tight_layout()

    pdf_path = output_dir / 'biome_breakdown.pdf'
    plt.savefig(pdf_path, dpi=PR_CURVE_DPI, bbox_inches='tight')
    print(f"  Saved PDF: {pdf_path}")
    plt.close()




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
    
    # Optional W&B run (for streaming logs and metrics)
    use_wandb = False
    if wandb is not None:
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
            # RF currently has only uncalibrated scored files; keep existing behaviour.
            # Search in all candidate directories (training scripts may save to repo root)
            parquet_path = None
            for search_dir in ml_models_candidates:
                if not search_dir.exists():
                    continue
                found = find_latest_file(
                    f"{MODEL_ID}_{args.model_type}_win5_scored*.parquet",
                    search_dir,
                    args.split_version,
                    args.model_type,
                )
                if found is not None:
                    parquet_path = found
                    print(f"  Found: {parquet_path}")
                    break
            
            if parquet_path is None:
                expected_pattern = f"{MODEL_ID}_rf_win5_scored_*.parquet"
                print(f"ERROR: Could not find {expected_pattern} in any of the following directories:")
                for cand in ml_models_candidates:
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
    
    # Load South America boundary once and reuse for all map outputs (avoids 5x load/download)
    sa_gdf = get_region_boundary(region_boundary_path)
    # Risk map generation (1%, 5%, 10%) - run first to get future capture metrics for table
    thresholds = [1, 5, 10]
    future_metrics = None
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
        )
        if result is not None and future_metrics is None:
            future_metrics = result
    # Generate outputs (metrics table includes future capture when available)
    create_metrics_table(metrics_data, output_dir, args.model_type, df=df, extra_metrics=future_metrics)
    create_pr_curve(df, metrics_data, output_dir, args.model_type)
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
    )

    # New outputs: calibration improvement, country breakdown, biome breakdown
    create_calibration_improvement_figure(df, output_dir, args.model_type)
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

