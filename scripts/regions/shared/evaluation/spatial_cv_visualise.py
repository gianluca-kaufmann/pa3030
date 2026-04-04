#!/usr/bin/env python3
"""Spatial CV visualisation — publication-ready figures for spatial generalisation.

Produces four figures from LOBO (Layer 1) and biome-stratified (Layer 2) results:

  bar_lobo_performance_<model_type>.pdf
      Horizontal grouped bar chart with two panels:
        Left  — ROC-AUC: LOBO (out-of-distribution) vs. in-distribution
        Right — Recall@5%: same comparison
      Biomes sorted by LOBO ROC-AUC (best at top).

  scatter_lobo_performance_<model_type>.pdf
      Bubble scatter:  x = biome PA prevalence (log scale)
                       y = LOBO ROC-AUC
                     size = √(PA events in held-out biome test set)
                    colour = LOBO Recall@5%
      Reveals whether generalisation correlates with event frequency.

  map_lobo_performance_<model_type>.pdf
      Two-panel choropleth map of the region, coloured by biome:
        Left  — LOBO ROC-AUC (colour scale 0.5–1.0)
        Right — LOBO Recall@5%
      Grey = biome absent or too few test events.

  scatter_biome_calibration_<model_type>.pdf
      Biome calibration scatter:  x = mean predicted probability (test set)
                                  y = observed transition rate (test set)
                                size = √(n test pixels)
                               colour = biome ROC-AUC
      1:1 diagonal = perfect calibration. Points below = over-prediction.
      Requires spatial_CV_2 to be run after the 2026-04 update that adds
      mean_pred_prob to biome_metrics output.

Data sources (latest file found by mtime, auto-detected; $SCRATCH checked first when set):
  LOBO summary:   outputs/{region}/results/spatial_cv/{model_type}/lobo_summary_*.csv
  Layer 2:        outputs/{region}/results/spatial_generalisation/biome_metrics_*.csv
  Biome shapefile: data/shared/GlobalSafetyNet/terrestrial_ecoregions/Terrestrial_ecoregions.shp

Output directory (same relative path; under $SCRATCH when SCRATCH is set, else repo):
  outputs/{region}/results/spatial_cv_visualisation/

Run after spatial_CV_1_aggregate (LOBO) and spatial_CV_2 (Layer 2).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    GEOPANDAS_AVAILABLE = True
except ImportError:
    gpd = None
    GEOPANDAS_AVAILABLE = False

from scripts.regions.shared.training.utils import Tee, get_repo_root


# =============================================================================
# Styling constants
# =============================================================================

FIGURE_DPI  = 300
MAP_DPI     = 300

COLOR_LOBO    = "#2166AC"   # dark blue  — LOBO / out-of-distribution
COLOR_INDIST  = "#D6604D"   # muted red  — in-distribution (full model)
COLOR_MISSING = "#CCCCCC"   # light grey — no data / skipped biome
COLOR_REF     = "#888888"   # grey dashed line — random-model reference

# Approximate bounding boxes per region (WGS84: minx, miny, maxx, maxy)
REGION_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "south_america": (-85.0, -57.0, -32.0, 15.0),
    "usa":           (-130.0, 23.0, -64.0, 52.0),
    "se_asia":       (90.0, -11.0, 145.0, 28.0),
}

# Short axis labels for biome names
_BIOME_SHORT: dict[str, str] = {
    "Tropical & Subtropical Moist Broadleaf Forests":        "Trop. Moist Broadleaf",
    "Tropical & Subtropical Dry Broadleaf Forests":          "Trop. Dry Broadleaf",
    "Tropical & Subtropical Grasslands, Savannas & Shrublands": "Trop. Grasslands",
    "Temperate Broadleaf & Mixed Forests":                   "Temp. Broadleaf",
    "Temperate Grasslands, Savannas & Shrublands":           "Temp. Grasslands",
    "Mediterranean Forests, Woodlands & Scrub":              "Mediterranean",
    "Montane Grasslands & Shrublands":                       "Montane Grasslands",
    "Flooded Grasslands & Savannas":                         "Flooded Grasslands",
    "Deserts & Xeric Shrublands":                            "Deserts & Xeric",
}


def _short(name: str) -> str:
    """Return a shortened biome label for axis/annotation use."""
    return _BIOME_SHORT.get(name, name)


def _get_cmap(name: str):
    """Matplotlib-version-agnostic colormap retrieval."""
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        return plt.get_cmap(name)


# =============================================================================
# Data loading helpers
# =============================================================================

def _find_latest(directory: Path, pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching *pattern* in *directory*."""
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime,
                     reverse=True)
    return matches[0] if matches else None


def load_lobo_results(region: str, model_type: str = "lgbm") -> Optional[pd.DataFrame]:
    """Load the latest LOBO summary CSV, keeping only valid per-biome rows.

    Filters out MEAN/STD summary rows, folds with all-NaN metrics (skipped
    biomes), and converts numeric columns to float.
    """
    repo_root = get_repo_root()
    scratch   = os.environ.get("SCRATCH")

    search_dirs: list[Path] = []
    if scratch:
        search_dirs.append(
            Path(scratch) / f"outputs/{region}/results/spatial_cv/{model_type}")
    search_dirs.append(
        repo_root / f"outputs/{region}/results/spatial_cv/{model_type}")

    csv_path = None
    for d in search_dirs:
        p = _find_latest(d, "lobo_summary_*.csv")
        if p:
            csv_path = p
            break

    if csv_path is None:
        print(f"  LOBO summary CSV not found for {region}/{model_type}. "
              f"Run spatial_CV_1_aggregate first.")
        return None

    print(f"  Loading LOBO summary: {csv_path}")
    df = pd.read_csv(csv_path)

    # Keep only genuine fold rows (fold_idx is a plain integer)
    numeric_idx = pd.to_numeric(df["fold_idx"], errors="coerce").notna()
    df = df[numeric_idx].copy()
    df["fold_idx"] = df["fold_idx"].astype(int)

    # Drop folds where the model was skipped (all metrics NaN)
    has_roc = pd.to_numeric(df.get("roc_auc"), errors="coerce").notna()
    df = df[has_roc].reset_index(drop=True)

    print(f"  {len(df)} valid LOBO folds loaded")
    return df


def load_layer2_results(region: str, model_label: str = "LGBM") -> Optional[pd.DataFrame]:
    """Load the latest Layer 2 biome metrics CSV for a given model.

    Returns a DataFrame with a 'biome_name' column (renamed from 'biome'),
    excluding the global 'ALL' row.
    """
    repo_root = get_repo_root()
    scratch   = os.environ.get("SCRATCH")

    search_dirs: list[Path] = []
    if scratch:
        search_dirs.append(
            Path(scratch) / f"outputs/{region}/results/spatial_generalisation")
    search_dirs.append(
        repo_root / f"outputs/{region}/results/spatial_generalisation")

    csv_path = None
    for d in search_dirs:
        p = _find_latest(d, "biome_metrics_*.csv")
        if p:
            csv_path = p
            break

    if csv_path is None:
        print(f"  Layer 2 biome metrics CSV not found for {region}. "
              f"Run spatial_CV_2 first.")
        return None

    print(f"  Loading Layer 2 metrics: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter to the requested model, drop the global ALL row
    df = df[(df["model"].str.upper() == model_label.upper()) &
            (df["biome"] != "ALL")].copy()
    df = df.rename(columns={"biome": "biome_name"})

    print(f"  {len(df)} biomes in Layer 2 results ({model_label})")
    return df


def _load_biome_geodataframe(region: str) -> Optional["gpd.GeoDataFrame"]:
    """Load, clip, and dissolve the WWF ecoregion shapefile to biome polygons.

    Returns a GeoDataFrame with columns ['biome_name', 'geometry'] where each
    row is the union of all ecoregion polygons belonging to that biome within
    the region's bounding box.  CRS is EPSG:4326.
    """
    if not GEOPANDAS_AVAILABLE:
        print("  Skipping map (geopandas not available).")
        return None

    repo_root = get_repo_root()
    shp_path  = (repo_root / "data" / "shared" / "GlobalSafetyNet" /
                 "terrestrial_ecoregions" / "Terrestrial_ecoregions.shp")
    if not shp_path.exists():
        print(f"  Warning: biome shapefile not found: {shp_path}")
        return None

    bounds   = REGION_BOUNDS.get(region, (-180.0, -90.0, 180.0, 90.0))
    clip_box = shapely_box(*bounds)

    print(f"  Loading biome shapefile: {shp_path.name}")
    gdf = gpd.read_file(shp_path)
    gdf["geometry"] = gdf.geometry.make_valid()

    # Normalise CRS to WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Clip to region
    try:
        gdf = gpd.clip(gdf, clip_box)
    except Exception as e:
        print(f"  clip() failed ({e}), falling back to intersection filter")
        gdf = gdf[gdf.geometry.intersects(clip_box)].copy()

    if gdf.empty:
        print("  Warning: no biome geometries within region bounds")
        return None

    # Dissolve all ecoregion polygons into their parent biome
    print(f"  Dissolving {len(gdf)} ecoregions → biomes...")
    biome_gdf = gdf.dissolve(by="BIOME_NAME", aggfunc="first").reset_index()
    biome_gdf = biome_gdf[["BIOME_NAME", "geometry"]].copy()
    biome_gdf.columns = ["biome_name", "geometry"]
    print(f"  {len(biome_gdf)} biome polygons after dissolve")
    return biome_gdf


# =============================================================================
# Figure 1: Horizontal grouped bar chart — LOBO vs in-distribution
# =============================================================================

def plot_lobo_vs_indistribution(
    lobo_df: pd.DataFrame,
    layer2_df: Optional[pd.DataFrame],
    output_path: Path,
) -> None:
    """Produce a two-panel horizontal bar chart.

    Left panel:  ROC-AUC  — LOBO (blue) vs. in-distribution (red)
    Right panel: Recall@5% — same comparison
    Biomes are sorted by LOBO ROC-AUC (ascending) so the best biome sits at
    the top of the chart.  The number of test-set PA events is annotated at
    the end of each LOBO bar.

    Random-model references:
        ROC-AUC  = 0.5
        Recall@5% = 0.05  (a random model selects 5% of positives at random)
    """
    # ── Build merged table ───────────────────────────────────────────────────
    keep = ["biome_name", "roc_auc", "recall_at_5pct", "n_test_pos"]
    for c in keep:
        if c not in lobo_df.columns:
            lobo_df = lobo_df.copy()
            lobo_df[c] = np.nan
    df = lobo_df[keep].copy()
    df.columns = ["biome_name", "lobo_roc", "lobo_r5", "n_pos"]

    has_recall  = df["lobo_r5"].notna().any()
    has_layer2  = layer2_df is not None
    has_l2_r5   = has_layer2 and "recall_at_5pct" in layer2_df.columns

    if has_layer2:
        l2_keep = ["biome_name", "roc_auc"]
        if has_l2_r5:
            l2_keep.append("recall_at_5pct")
        l2 = layer2_df[l2_keep].copy()
        rename = {"roc_auc": "l2_roc"}
        if has_l2_r5:
            rename["recall_at_5pct"] = "l2_r5"
        l2 = l2.rename(columns=rename)
        df = df.merge(l2, on="biome_name", how="left")

    # Sort ascending by LOBO ROC-AUC so best biome is at the top
    df = df.sort_values("lobo_roc", ascending=True).reset_index(drop=True)

    short_names = [_short(n) for n in df["biome_name"]]
    n_b      = len(df)
    y_pos    = np.arange(n_b)
    bh       = 0.33   # bar height
    n_panels = 2 if has_recall else 1

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7.5 * n_panels, max(4.5, n_b * 0.72 + 1.8)),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    # ── Helper to draw one panel ─────────────────────────────────────────────
    def _panel(ax, lobo_vals, l2_vals, xlabel, ref_x, xmin, xmax):
        lobo_vals = np.array(lobo_vals, dtype=float)

        # LOBO bars
        bars = ax.barh(
            y_pos + bh / 2, lobo_vals, height=bh,
            color=COLOR_LOBO, label="LOBO (OOD)", zorder=3,
        )
        # In-distribution bars (if available)
        if l2_vals is not None and has_layer2:
            l2v = np.array(l2_vals, dtype=float)
            ax.barh(
                y_pos - bh / 2, l2v, height=bh,
                color=COLOR_INDIST, alpha=0.80,
                label="In-distribution (full model)", zorder=3,
            )
        # Reference line (random baseline)
        ax.axvline(ref_x, color=COLOR_REF, lw=1.0, ls="--", alpha=0.7, zorder=2)
        ax.text(ref_x + 0.003, n_b - 0.1, "random",
                fontsize=7, color=COLOR_REF, va="top")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=9.5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.grid(axis="x", alpha=0.25, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate LOBO bar ends with value
        for i, val in enumerate(lobo_vals):
            if np.isfinite(val):
                ax.text(
                    val + (xmax - xmin) * 0.012,
                    y_pos[i] + bh / 2,
                    f"{val:.3f}", va="center", ha="left",
                    fontsize=7.5, color=COLOR_LOBO, fontweight="bold",
                )

    # ── Panel 1: ROC-AUC ────────────────────────────────────────────────────
    l2_roc = df.get("l2_roc", pd.Series(dtype=float)).to_numpy()
    _panel(
        axes[0],
        df["lobo_roc"].to_numpy(), l2_roc if has_layer2 else None,
        xlabel="ROC-AUC",
        ref_x=0.5, xmin=0.40, xmax=1.04,
    )
    axes[0].set_title("ROC-AUC by Biome", fontweight="bold", fontsize=12, pad=8)

    # Annotate n_pos on the left margin (each LOBO bar)
    for i, (_, row) in enumerate(df.iterrows()):
        n = row.get("n_pos")
        if pd.notna(n) and int(n) > 0:
            axes[0].text(
                0.405, y_pos[i] + bh / 2,
                f"n={int(n):,}", va="center", ha="left",
                fontsize=6.5, color="#555555",
            )

    # ── Panel 2: Recall@5% ──────────────────────────────────────────────────
    if has_recall:
        l2_r5 = df.get("l2_r5", pd.Series(dtype=float)).to_numpy()
        r5_max = max(0.10, float(np.nanmax(df["lobo_r5"])) + 0.06)
        _panel(
            axes[1],
            df["lobo_r5"].to_numpy(), l2_r5 if (has_layer2 and has_l2_r5) else None,
            xlabel="Recall at top-5% of predicted pixels",
            ref_x=0.05, xmin=0.0, xmax=min(1.04, r5_max),
        )
        axes[1].set_title(
            "Recall@top-5% by Biome\n"
            "(fraction of PA events captured when screening top-5%)",
            fontweight="bold", fontsize=11, pad=8,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=COLOR_LOBO, label="LOBO (trained without this biome)")]
    if has_layer2:
        handles.append(
            mpatches.Patch(color=COLOR_INDIST, alpha=0.80,
                           label="In-distribution (full model, biome seen in training)")
        )
    axes[-1].legend(handles=handles, loc="lower right", fontsize=8.5,
                    framealpha=0.92, edgecolor="lightgray")

    fig.suptitle(
        "Spatial Generalisation: LOBO vs. In-Distribution Performance\n"
        r"Leave-One-Biome-Out CV — Test set 2017–2019",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# =============================================================================
# Figure 2: Bubble scatter — prevalence vs. LOBO performance
# =============================================================================

def plot_prevalence_vs_performance(
    lobo_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bubble scatter: biome prevalence (x) vs LOBO ROC-AUC (y).

    Bubble size encodes the square root of the number of PA designation events
    in the held-out biome's test set, so rare biomes appear smaller.
    Bubble colour encodes LOBO Recall@5% (RdYlGn: red=low, green=high).

    The log x-axis reveals whether generalisation difficulty correlates with
    how frequently PAs are established (prevalence), which is a key confound
    when interpreting AUC in imbalanced settings.
    """
    df = lobo_df.copy()
    for col in ("positive_rate", "roc_auc", "n_test_pos", "recall_at_5pct"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df = df.dropna(subset=["positive_rate", "roc_auc"]).reset_index(drop=True)
    if df.empty:
        print("  Not enough LOBO data for scatter — skipping.")
        return

    # Bubble sizes: sqrt(n_pos), normalised to [60, 800]
    n_pos  = np.sqrt(df["n_test_pos"].clip(lower=1).fillna(1))
    sizes  = 60 + (n_pos / n_pos.max()) * 740

    # Colour: Recall@5% if available, otherwise uniform
    cmap        = _get_cmap("RdYlGn")
    has_recall  = df["recall_at_5pct"].notna().any()
    if has_recall:
        r5   = df["recall_at_5pct"].fillna(0.0).to_numpy()
        norm = Normalize(vmin=0.0, vmax=max(float(r5.max()), 0.01))
        colors = [cmap(norm(v)) for v in r5]
    else:
        colors = COLOR_LOBO

    # constrained_layout avoids tight_layout failures with log axes + colorbar + labels
    fig, ax = plt.subplots(figsize=(10.0, 6.8), constrained_layout=True)

    x_pct = df["positive_rate"].to_numpy(dtype=float) * 100.0
    ax.scatter(
        x_pct,
        df["roc_auc"],
        s=sizes,
        c=colors,
        edgecolors="white",
        linewidths=0.9,
        alpha=0.88,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Biome PA prevalence in test set (%, log scale)", fontsize=11)
    ax.set_ylabel("LOBO ROC-AUC", fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, which="both", alpha=0.25, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "LOBO Generalisation vs. Biome PA Prevalence\n"
        "Bubble size ∝ √(PA events in held-out test set)",
        fontsize=11,
        fontweight="bold",
    )

    # Baseline reference (axes coords so it stays visible with log x)
    ax.axhline(0.5, color=COLOR_REF, lw=1.0, ls="--", alpha=0.7, zorder=2)
    ax.text(
        0.02,
        0.08,
        "AUC = 0.5 (random)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLOR_REF,
        va="bottom",
    )

    # Biome labels — offset slightly to avoid overlap with bubble centres
    for i, row in df.iterrows():
        ax.annotate(
            _short(row["biome_name"]),
            xy=(x_pct[i], row["roc_auc"]),
            xytext=(7, 2),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
            zorder=4,
        )

    # Colorbar for recall
    if has_recall:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.03, aspect=24)
        cbar.set_label("LOBO Recall@top-5%", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# =============================================================================
# Figure 3: Choropleth map — geographic performance distribution
# =============================================================================

def plot_biome_performance_map(
    lobo_df: pd.DataFrame,
    region: str,
    output_path: Path,
) -> None:
    """Two-panel choropleth map of biome-level LOBO performance.

    Left panel:  LOBO ROC-AUC (ranking quality)
    Right panel: LOBO Recall@5% (policy-relevant coverage)

    Both panels share the RdYlGn colormap (red = poor, green = strong).
    Grey fill denotes biomes absent from the LOBO results (either out of the
    region or skipped due to too few test-set PA events).
    ROC-AUC colour scale is fixed to [0.5, 1.0] so the full legend range is shown.
    """
    if not GEOPANDAS_AVAILABLE:
        print("  Skipping map (geopandas not available).")
        return

    print("  Loading biome polygons...")
    biome_gdf = _load_biome_geodataframe(region)
    if biome_gdf is None:
        return

    # ── Merge LOBO metrics onto biome geodataframe ───────────────────────────
    metrics = lobo_df[["biome_name", "roc_auc", "recall_at_5pct",
                        "positive_rate", "n_test_pos"]].copy()
    for c in ("roc_auc", "recall_at_5pct", "positive_rate", "n_test_pos"):
        metrics[c] = pd.to_numeric(metrics.get(c), errors="coerce")

    gdf = biome_gdf.merge(metrics, on="biome_name", how="left")

    has_recall = ("recall_at_5pct" in gdf.columns and
                  gdf["recall_at_5pct"].notna().any())
    n_panels   = 2 if has_recall else 1

    # ── Map extent ───────────────────────────────────────────────────────────
    bounds = gdf.total_bounds        # [minx, miny, maxx, maxy]
    pad    = 1.5
    xlim   = (bounds[0] - pad, bounds[2] + pad)
    ylim   = (bounds[1] - pad, bounds[3] + pad)

    # ── Colormaps and normalisations ─────────────────────────────────────────
    cmap = _get_cmap("RdYlGn")

    # Full ROC-AUC colour scale (random = 0.5, perfect = 1.0) for interpretability
    norm_auc = Normalize(vmin=0.5, vmax=1.0)

    if has_recall:
        r5_vals = gdf["recall_at_5pct"].dropna().to_numpy()
        norm_r5 = Normalize(
            vmin=0.0,
            vmax=min(1.0, float(r5_vals.max()) + 0.02) if len(r5_vals) else 1.0,
        )

    # ── Figure layout ────────────────────────────────────────────────────────
    # Derive figure height from geographic aspect ratio
    lon_span  = xlim[1] - xlim[0]
    lat_span  = ylim[1] - ylim[0]
    mid_lat   = (ylim[0] + ylim[1]) / 2.0
    aspect    = lat_span / (lon_span * np.cos(np.radians(mid_lat)))
    panel_w   = 7.5
    panel_h   = panel_w * aspect

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_w * n_panels + 0.5, panel_h + 1.5),
    )
    if n_panels == 1:
        axes = [axes]

    # ── Inner drawing function ───────────────────────────────────────────────
    def _draw_panel(ax, col, norm, title, cbar_label, ref_note=""):
        # Grey fill for all biomes (background / no-data)
        gdf.plot(ax=ax, color=COLOR_MISSING, edgecolor="#AAAAAA",
                 linewidth=0.4, zorder=1)

        # Choropleth for biomes that have valid metric values
        valid = gdf[gdf[col].notna()].copy()
        if not valid.empty:
            valid.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                       edgecolor="#777777", linewidth=0.4, zorder=2)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.52, pad=0.025, aspect=22)
        cbar.set_label(cbar_label, fontsize=9.5)
        cbar.ax.tick_params(labelsize=8)

        # Map cosmetics
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
        ax.set_xlabel("Longitude", fontsize=8.5)
        ax.set_ylabel("Latitude", fontsize=8.5)
        ax.tick_params(labelsize=8)

        # Grey = no data note
        grey_patch = mpatches.Patch(
            color=COLOR_MISSING, label="No LOBO data (biome absent or skipped)")
        ax.legend(handles=[grey_patch], loc="lower left",
                  fontsize=7.5, framealpha=0.85, edgecolor="lightgray")

        if ref_note:
            ax.text(0.99, 0.01, ref_note, transform=ax.transAxes,
                    fontsize=6.5, color="#555555", ha="right", va="bottom")

    # ── Left panel: ROC-AUC ──────────────────────────────────────────────────
    _draw_panel(
        axes[0], "roc_auc", norm_auc,
        title="LOBO ROC-AUC\n(model trained without this biome)",
        cbar_label="ROC-AUC",
    )

    # ── Right panel: Recall@5% ───────────────────────────────────────────────
    if has_recall:
        _draw_panel(
            axes[1], "recall_at_5pct", norm_r5,
            title="LOBO Recall@top-5%\n(fraction of PA events captured)",
            cbar_label="Recall at top-5% of predictions",
            ref_note="Random baseline ≈ 0.05",
        )

    # ── Overall title ────────────────────────────────────────────────────────
    region_label = region.replace("_", " ").title()
    fig.suptitle(
        f"Biome-Level LOBO Spatial Generalisation — {region_label}\n"
        r"Test period: 2017–2019  |  5-year PA designation lookahead window",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# =============================================================================
# Entry point
# =============================================================================

def plot_biome_predicted_vs_actual(
    layer2_df: pd.DataFrame,
    output_path: Path,
    region_label: str = "",
) -> None:
    """Scatter: biome-level mean predicted probability vs observed transition rate.

    Requires the Layer 2 biome_metrics CSV to have a ``mean_pred_prob`` column
    (produced by spatial_CV_2 after the 2026-04 update to spatial_cv_core.py).
    Falls back gracefully if the column is absent.

    x-axis: mean predicted probability (calibrated) across test-set pixels in biome
    y-axis: observed transition rate (positive_rate) in the test period
    Size:   sqrt(n_samples), scaled
    Colour: biome ROC-AUC (RdYlGn)
    Diagonal: 1:1 line — points below = over-prediction, above = under-prediction.
    """
    df = layer2_df.copy()

    for col in ("mean_pred_prob", "positive_rate", "roc_auc", "n_samples"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    if "mean_pred_prob" not in df.columns or df["mean_pred_prob"].isna().all():
        print("  Biome predicted-vs-actual scatter: mean_pred_prob column absent "
              "— re-run spatial_CV_2 to generate it, then re-run this script.")
        return

    df = df.dropna(subset=["mean_pred_prob", "positive_rate"]).reset_index(drop=True)
    if len(df) < 2:
        print("  Not enough biome rows for scatter — skipping.")
        return

    x = df["mean_pred_prob"].to_numpy()
    y = df["positive_rate"].to_numpy()

    # Bubble sizes proportional to sqrt(n_samples)
    n = df["n_samples"].clip(lower=1).fillna(1).to_numpy()
    sizes = 60 + (np.sqrt(n) / np.sqrt(n).max()) * 700

    # Colour by ROC-AUC
    cmap = _get_cmap("RdYlGn")
    roc  = df["roc_auc"].fillna(0.5).to_numpy()
    norm = Normalize(vmin=0.5, vmax=1.0)
    colors = [cmap(norm(v)) for v in roc]

    # Axis range: cover both x and y to make the 1:1 line meaningful
    lo = 0.0
    hi = max(float(x.max()), float(y.max())) * 1.15

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    ax.scatter(x * 100, y * 100, s=sizes, c=colors,
               edgecolors="white", linewidths=0.8, alpha=0.88, zorder=3)

    # 1:1 diagonal
    diag = np.linspace(lo * 100, hi * 100, 100)
    ax.plot(diag, diag, color="#555555", lw=1.2, ls="--", zorder=2, label="1:1 (perfect calibration)")

    # Biome labels
    for _, row in df.iterrows():
        ax.annotate(
            _short(row.get("biome_name", "")),
            xy=(row["mean_pred_prob"] * 100, row["positive_rate"] * 100),
            xytext=(5, 2), textcoords="offset points",
            fontsize=7.5, color="#333333",
        )

    ax.set_xlabel("Mean predicted probability — calibrated (%, test set)", fontsize=11)
    ax.set_ylabel("Observed transition rate — actual PAs (%, test set)", fontsize=11)
    ax.set_xlim(lo * 100, hi * 100)
    ax.set_ylim(lo * 100, hi * 100)
    ax.grid(True, alpha=0.22, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    title_suffix = f" — {region_label}" if region_label else ""
    ax.set_title(
        f"Biome Calibration: Predicted vs Actual Designation Rate{title_suffix}\n"
        r"Bubble size $\propto$ $\sqrt{n_{\mathrm{test\ pixels}}}$ · "
        "Below diagonal = over-prediction · Above = under-prediction",
        fontsize=10.5, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=22)
    cbar.set_label("Biome ROC-AUC (full model)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def _resolve_spatial_cv_visualisation_output_dir(region: str) -> Path:
    """Figure output dir: ``$SCRATCH/outputs/...`` when SCRATCH is set, else repo."""
    repo_root = get_repo_root()
    scratch   = os.environ.get("SCRATCH")
    subdir    = "spatial_cv_visualisation"
    if scratch:
        d = Path(scratch) / f"outputs/{region}/results/{subdir}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = repo_root / f"outputs/{region}/results/{subdir}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_spatial_cv_visualise(region: str, model_type: str = "lgbm") -> None:
    """Produce all spatial CV visualisation figures for *region*.

    Args:
        region:     Region slug (e.g. 'south_america', 'usa').
        model_type: Model type whose LOBO folds to read ('lgbm' or 'rf').
    """
    output_dir = _resolve_spatial_cv_visualisation_output_dir(region)
    mt_safe    = model_type.lower().strip()
    log_path   = output_dir / f"visualise_{mt_safe}.txt"

    tee = Tee(log_path)
    sys.stdout = tee
    try:
        t0           = time.time()
        region_label = region.replace("_", " ").upper()

        print("=" * 70)
        print(f"SPATIAL CV VISUALISATION — {region_label}")
        print("=" * 70)
        print(f"Model type:  {model_type.upper()}")
        print(f"Output dir:  {output_dir}")

        # ── Load data ────────────────────────────────────────────────────────
        print("\n--- Loading data ---")
        lobo_df   = load_lobo_results(region, model_type)
        layer2_df = load_layer2_results(region, model_label=model_type.upper())

        if lobo_df is None:
            print("\nERROR: LOBO summary not found — cannot produce figures.")
            print("Run spatial_CV_1 (all biome folds) then spatial_CV_1_aggregate first.")
            return

        print(f"\nLOBO folds available : {len(lobo_df)}")
        if layer2_df is not None:
            print(f"Layer 2 biomes       : {len(layer2_df)}")
        else:
            print("Layer 2 data         : not available "
                  "(in-distribution bars omitted from bar chart)")

        # ── Figure 1: Bar chart ───────────────────────────────────────────────
        print("\n--- Figure 1: Bar chart (LOBO vs in-distribution) ---")
        fig1_path = output_dir / f"bar_lobo_performance_{mt_safe}.pdf"
        plot_lobo_vs_indistribution(lobo_df, layer2_df, fig1_path)

        # ── Figure 2: Scatter ─────────────────────────────────────────────────
        print("\n--- Figure 2: Scatter (prevalence vs performance) ---")
        fig2_path = output_dir / f"scatter_lobo_performance_{mt_safe}.pdf"
        plot_prevalence_vs_performance(lobo_df, fig2_path)

        # ── Figure 3: Map ─────────────────────────────────────────────────────
        print("\n--- Figure 3: Biome performance map ---")
        fig3_path = output_dir / f"map_lobo_performance_{mt_safe}.pdf"
        plot_biome_performance_map(lobo_df, region, fig3_path)

        # ── Figure 4: Biome predicted vs actual scatter ───────────────────────
        print("\n--- Figure 4: Biome predicted vs actual scatter ---")
        if layer2_df is not None:
            fig4_path = output_dir / f"scatter_biome_calibration_{mt_safe}.pdf"
            region_label = region.replace("_", " ").title()
            plot_biome_predicted_vs_actual(layer2_df, fig4_path, region_label=region_label)
        else:
            print("  Skipping — Layer 2 data not available.")

        elapsed = time.time() - t0
        print(f"\n{'='*70}")
        print(f"All figures saved to: {output_dir}")
        print(f"Total time: {elapsed:.1f}s")
        print("Done.")

    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nLog saved to: {log_path}")
