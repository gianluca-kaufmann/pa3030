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
      No continental coastline stroke — only filled land-mask + coloured biomes.

  scatter_biome_calibration_<model_type>.pdf
      Biome calibration scatter:  x = mean predicted probability (test set)
                                  y = observed transition rate (test set)
                                size = √(n test pixels)
                               colour = biome ROC-AUC
      1:1 diagonal = perfect calibration. Points below = over-prediction.
      Requires spatial_CV_2 to be run after the 2026-04 update that adds
      mean_pred_prob to biome_metrics output.

Data sources (newest ``*_YYYYMMDD_HHMMSS.csv`` by embedded timestamp, not mtime;
  $SCRATCH checked first when set):
  LOBO summary:   outputs/{region}/results/spatial_cv/{model_type}/lobo_summary_*.csv
  Layer 2:        outputs/{region}/results/spatial_generalisation/biome_metrics_*.csv
  Biome shapefile: data/shared/GlobalSafetyNet/terrestrial_ecoregions/Terrestrial_ecoregions.shp

Output directory (same relative path; under $SCRATCH when SCRATCH is set, else repo):
  outputs/{region}/results/spatial_cv_visualisation/

Run after spatial_CV_1_aggregate (LOBO) and spatial_CV_2 (Layer 2).
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
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

# Power-norm gamma for choropleth colour scales on the biome performance map.
# gamma < 1 applies a sqrt-like stretch: biomes with moderate lift/recall (which
# are genuinely good, just not extreme) map to yellow-green rather than orange.
# vmin and vmax anchors are unchanged, so the colorbar remains honest; only the
# perceptual spacing between values is non-linear.  gamma=0.5 (square root) is a
# well-known convention for this kind of skewed-ratio data.
CHOROPLETH_GAMMA: float = 0.35

# Map colour-scale caps (requested for thesis figures).
# These are applied only for regions listed here; others keep auto-scaling.
# Keys:
#   pr_auc_lift  — upper cap on the PR-AUC lift colour scale (left panel)
#   r5_vmax      — upper cap on the absolute Recall@5% colour scale (right panel)
LIFT_VMAX_CAPS: dict[str, dict[str, float]] = {
    # South America: lift can reach extreme values in tiny-prevalence biomes;
    # cap for readability/print.  Recall@5% capped at 0.70 (70%) to spread
    # colour contrast across the informative range.
    "south_america": {"pr_auc_lift": 30.0, "r5_vmax": 0.70},
}

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


def _get_label_points(
    geom,
    *,
    elongation_threshold: float = 4.0,
    fragment_significance: float = 0.03,
    n_elongated: int = 3,
) -> list:
    """Compute label placement points for a (possibly complex) biome geometry.

    Returns a list of (x, y) tuples where a biome number should be placed:
      - Compact polygon       → single representative_point()
      - Elongated polygon     → n_elongated evenly-spaced points along major axis
      - Fragmented MultiPolygon → one point per significant fragment (area > threshold)
    """
    try:
        from shapely.geometry import box as _sbox
    except ImportError:
        try:
            pt = geom.representative_point()
            return [(pt.x, pt.y)]
        except Exception:
            return []

    # ── Fragmentation check ──────────────────────────────────────────────────
    if hasattr(geom, "geoms"):
        parts      = sorted(geom.geoms, key=lambda g: g.area, reverse=True)
        total_area = geom.area or 1.0
        significant = [p for p in parts if p.area / total_area > fragment_significance]
        if len(significant) >= 2:
            pts = []
            for part in significant[:4]:
                try:
                    pt = part.representative_point()
                    pts.append((pt.x, pt.y))
                except Exception:
                    pass
            if pts:
                return pts
        # Single dominant part — fall through using only that polygon
        geom = parts[0] if parts else geom

    # ── Elongation check ─────────────────────────────────────────────────────
    try:
        minx, miny, maxx, maxy = geom.bounds
        width  = maxx - minx
        height = maxy - miny
        if width > 0 and height > 0:
            aspect = max(width, height) / min(width, height)
            if aspect >= elongation_threshold:
                major_vertical = height >= width
                pts = []
                for i in range(n_elongated):
                    lo, hi = i / n_elongated, (i + 1) / n_elongated
                    if major_vertical:
                        strip = _sbox(minx, miny + lo * height, maxx, miny + hi * height)
                    else:
                        strip = _sbox(minx + lo * width, miny, minx + hi * width, maxy)
                    try:
                        clipped = geom.intersection(strip)
                        if not clipped.is_empty:
                            pt = clipped.representative_point()
                            pts.append((pt.x, pt.y))
                    except Exception:
                        pass
                if pts:
                    return pts
    except Exception:
        pass

    # ── Compact — single point ────────────────────────────────────────────────
    try:
        pt = geom.representative_point()
        return [(pt.x, pt.y)]
    except Exception:
        return []


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


def _find_latest_versioned_csv(directory: Path, stem: str) -> Optional[Path]:
    """Return newest ``{stem}_YYYYMMDD_HHMMSS.csv`` by suffix timestamp (not mtime).

    Prefer this over :func:`_find_latest` for pipeline outputs so a freshly touched
    older file does not override a newer run. Files matching ``{stem}_*.csv`` but
    not the timestamp pattern are only used if no versioned name exists (then
    mtime tie-break via :func:`_find_latest`).
    """
    if not directory.exists():
        return None
    pat = re.compile(rf"^{re.escape(stem)}_(\d{{8}})_(\d{{6}})\.csv$", re.IGNORECASE)
    versioned: list[tuple[tuple[str, str], Path]] = []
    for p in directory.glob(f"{stem}_*.csv"):
        m = pat.match(p.name)
        if m:
            versioned.append(((m.group(1), m.group(2)), p))
    if versioned:
        versioned.sort(key=lambda t: t[0], reverse=True)
        return versioned[0][1]
    return _find_latest(directory, f"{stem}_*.csv")


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
        p = _find_latest_versioned_csv(d, "lobo_summary")
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
        p = _find_latest_versioned_csv(d, "biome_metrics")
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


def _load_country_borders_3857(region: str) -> Optional["gpd.GeoDataFrame"]:
    """Load Natural Earth country boundaries, clip to region, reproject to EPSG:3857.

    Returns a GeoDataFrame or None if the file is unavailable.  Used to overlay
    thin country-border lines on choropleth maps for geographic orientation.
    """
    if not GEOPANDAS_AVAILABLE:
        return None
    try:
        repo_root = get_repo_root()
        gpkg = (repo_root / "data" / "shared" / "admin" /
                "ne_110m_admin_0_countries.gpkg")
        if not gpkg.exists():
            print(f"  NOTE: country borders file not found: {gpkg}; skipping.")
            return None
        world = gpd.read_file(str(gpkg))
        if world.crs is None:
            world = world.set_crs("EPSG:4326")
        elif world.crs.to_epsg() != 4326:
            world = world.to_crs("EPSG:4326")
        from shapely.geometry import box as shapely_box
        bounds   = REGION_BOUNDS.get(region, (-180.0, -90.0, 180.0, 90.0))
        clip_box = shapely_box(*bounds)
        try:
            world = world.clip(clip_box)
        except Exception:
            world = world[world.geometry.intersects(clip_box)].copy()
        if world.empty:
            return None
        print(f"  Country borders loaded: {len(world)} countries for {region}")
        return world.to_crs("EPSG:3857")
    except Exception as exc:
        print(f"  NOTE: could not load country borders ({exc}); skipping.")
        return None


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
    keep = ["biome_name", "pr_auc", "recall_at_5pct", "n_test_pos", "positive_rate"]
    for c in keep:
        if c not in lobo_df.columns:
            lobo_df = lobo_df.copy()
            lobo_df[c] = np.nan
    df = lobo_df[keep].copy()
    df.columns = ["biome_name", "lobo_pr", "lobo_r5", "n_pos", "pos_rate"]

    has_recall  = df["lobo_r5"].notna().any()
    has_layer2  = layer2_df is not None
    has_l2_r5   = has_layer2 and "recall_at_5pct" in layer2_df.columns

    if has_layer2:
        l2_keep = ["biome_name", "pr_auc"]
        if has_l2_r5:
            l2_keep.append("recall_at_5pct")
        l2 = layer2_df[l2_keep].copy()
        rename = {"pr_auc": "l2_pr"}
        if has_l2_r5:
            rename["recall_at_5pct"] = "l2_r5"
        l2 = l2.rename(columns=rename)
        df = df.merge(l2, on="biome_name", how="left")

    # Sort ascending by LOBO PR-AUC so best biome is at the top
    df = df.sort_values("lobo_pr", ascending=True).reset_index(drop=True)

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

    # ── Panel 1: PR-AUC ────────────────────────────────────────────────────
    l2_pr = df.get("l2_pr", pd.Series(dtype=float)).to_numpy()
    pr_max = max(0.50, float(np.nanmax(df["lobo_pr"])) + 0.06)
    _panel(
        axes[0],
        df["lobo_pr"].to_numpy(), l2_pr if has_layer2 else None,
        xlabel="PR-AUC",
        ref_x=float(df["pos_rate"].mean()),   # mean positive rate as rough baseline
        xmin=0.0, xmax=min(1.04, pr_max),
    )
    axes[0].set_title("PR-AUC by Biome", fontweight="bold", fontsize=12, pad=8)
    # Add note about varying baseline
    axes[0].text(
        float(df["pos_rate"].mean()) + 0.01, 0.5,
        "≈ random\nbaseline",
        fontsize=6.5, color=COLOR_REF, va="center",
    )

    # Annotate n_pos on the left margin (each LOBO bar)
    for i, (_, row) in enumerate(df.iterrows()):
        n = row.get("n_pos")
        if pd.notna(n) and int(n) > 0:
            axes[0].text(
                0.005, y_pos[i] + bh / 2,
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
    layer2_df: Optional[pd.DataFrame],
    region: str,
    output_path: Path,
) -> None:
    """Two-panel choropleth map of biome-level LOBO performance.

    Left panel:  LOBO PR-AUC (precision-recall, primary metric under class imbalance)
    Right panel: LOBO Recall@5% (policy-relevant coverage)

    Both panels share the RdYlGn colormap (red = poor, green = strong).

    Biomes not in the model's evaluation scope (e.g. ecoregions unrepresented
    in the GSN raster for this region) are deliberately excluded from the
    choropleth — the continent land-mask provides a neutral light-grey backdrop
    so no misleadingly large "no data" polygons appear.
    """
    if not GEOPANDAS_AVAILABLE:
        print("  Skipping map (geopandas not available).")
        return

    print("  Loading biome polygons...")
    biome_gdf = _load_biome_geodataframe(region)
    if biome_gdf is None:
        return

    # ── Clip biome polygons to true region boundary ───────────────────────────
    # Uses the local Natural Earth 110 m file (data/shared/admin/), with a
    # network download as fallback so the script also works on Euler.
    # For South America this cleanly excludes Central America / Panama.
    boundary_gdf: Optional["gpd.GeoDataFrame"] = None
    try:
        if region == "south_america":
            repo_root    = get_repo_root()
            local_ne     = repo_root / "data" / "shared" / "admin" / "ne_110m_admin_0_countries.gpkg"
            world: Optional["gpd.GeoDataFrame"] = None

            if local_ne.exists():
                world = gpd.read_file(local_ne)
            else:
                # Fallback: download from Natural Earth (requires network)
                import tempfile
                import urllib.request
                import zipfile

                ne_zip_url = (
                    "https://naciscdn.org/naturalearth/110m/cultural/"
                    "ne_110m_admin_0_countries.zip"
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = Path(tmpdir) / "ne_countries.zip"
                    urllib.request.urlretrieve(ne_zip_url, zip_path)  # nosec
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(tmpdir)
                    shp_files = list(Path(tmpdir).glob("*.shp"))
                    if not shp_files:
                        raise RuntimeError("No .shp in Natural Earth download")
                    world = gpd.read_file(shp_files[0])

            if world.crs is None:
                world = world.set_crs("EPSG:4326", allow_override=True)
            elif world.crs.to_epsg() != 4326:
                world = world.to_crs("EPSG:4326")

            cont_col = next(
                (c for c in ("CONTINENT", "continent") if c in world.columns), None
            )
            if cont_col is None:
                raise RuntimeError("No continent column found in Natural Earth file")

            boundary_gdf = world[world[cont_col] == "South America"].copy()
            boundary_gdf = boundary_gdf[~boundary_gdf.geometry.is_empty].reset_index(drop=True)

            boundary_union = boundary_gdf.unary_union
            biome_gdf = biome_gdf.copy()
            biome_gdf["geometry"] = biome_gdf.geometry.intersection(boundary_union)
            biome_gdf = biome_gdf[~biome_gdf.geometry.is_empty].reset_index(drop=True)
    except Exception as exc:
        print(f"  Warning: could not clip biome polygons to boundary ({exc}).")

    # ── Merge LOBO + Layer 2 metrics onto biome polygons ─────────────────────
    # Primary metric: PR-AUC (main model metric — robust under class imbalance).
    # Fall back to Layer 2 (in-distribution) when LOBO fold is missing.
    lobo_cols = ["biome_name", "pr_auc", "recall_at_5pct", "positive_rate", "n_test_pos"]
    for c in lobo_cols[1:]:
        if c not in lobo_df.columns:
            lobo_df = lobo_df.copy()
            lobo_df[c] = np.nan
    metrics = lobo_df[lobo_cols].copy()
    for c in lobo_cols[1:]:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    gdf = biome_gdf.merge(metrics, on="biome_name", how="left")

    if layer2_df is not None and not layer2_df.empty:
        l2 = layer2_df.copy()
        l2["biome_name"] = l2["biome_name"].astype(str).str.strip()
        gdf["biome_name"]  = gdf["biome_name"].astype(str).str.strip()
        l2_keep = {"biome_name"}
        for col in ("pr_auc", "recall_at_5pct"):
            if col in l2.columns:
                l2_keep.add(col)
        l2 = l2[list(l2_keep)].rename(
            columns={"pr_auc": "l2_pr_auc", "recall_at_5pct": "l2_recall_at_5pct"}
        )
        gdf = gdf.merge(l2, on="biome_name", how="left")
        if "l2_pr_auc" in gdf.columns:
            gdf["pr_auc"] = gdf["pr_auc"].fillna(gdf["l2_pr_auc"])
        if "l2_recall_at_5pct" in gdf.columns:
            gdf["recall_at_5pct"] = gdf["recall_at_5pct"].fillna(gdf["l2_recall_at_5pct"])

    has_recall = "recall_at_5pct" in gdf.columns and gdf["recall_at_5pct"].notna().any()
    n_panels   = 2 if has_recall else 1

    # ── Two GDFs: full land-mask + evaluated biomes only ─────────────────────
    # The land-mask is the dissolved outline of all clipped biome polygons.
    # It provides a uniform grey backdrop for the whole continent — crucially,
    # biomes with no evaluation data (e.g. Temperate Grasslands / Argentina
    # pampas) are absent from the GSN raster used to build the model, so they
    # never appear in LOBO or Layer 2 results.  Rather than drawing a large
    # grey polygon for them (which looks like a bug), we simply omit those
    # biome polygons from the choropleth and let the land-mask show through.
    print("  Building land mask from biome polygon union...")
    try:
        land_mask_geom = biome_gdf.geometry.union_all()
    except AttributeError:
        # union_all added in geopandas 0.14; fall back to unary_union
        land_mask_geom = biome_gdf.unary_union

    land_mask = gpd.GeoDataFrame(geometry=[land_mask_geom], crs=biome_gdf.crs)

    # Only keep biome rows that have at least one metric value
    gdf_valid = gdf[gdf["pr_auc"].notna() | gdf["recall_at_5pct"].notna()].copy()
    gdf_nodata = gdf[~(gdf["pr_auc"].notna() | gdf["recall_at_5pct"].notna())].copy()
    print(f"  Biomes with metrics: {len(gdf_valid)} | without: {len(gdf_nodata)}")
    if not gdf_nodata.empty:
        print("  Biomes without metrics (outside model evaluation scope):")
        for name in gdf_nodata["biome_name"].tolist():
            print(f"    · {name}")

    # Reproject to EPSG:3857 — equal-area for proper aspect ratio
    land_mask_3857  = land_mask.to_crs("EPSG:3857")
    gdf_valid_3857  = gdf_valid.to_crs("EPSG:3857")

    # ── Colour columns ────────────────────────────────────────────────────────
    # Left panel (PR-AUC): colouring by raw PR-AUC is misleading because the
    # metric's random baseline varies with biome prevalence.  We use lift:
    #   PR-AUC lift = PR-AUC / positive_rate   (random ≈ prevalence; lift = 1)
    #
    # Right panel (Recall@5%): the random baseline is always exactly 0.05 (5%),
    # so raw Recall@5% IS directly comparable across biomes.  We display the
    # absolute fraction and anchor vmin at 0.05 so that biomes at or below
    # chance appear at the red end of the colormap.  This lets readers read off
    # the policy-relevant number directly: "screening the top 5% of pixels
    # captures X% of all future PA designation events."
    gdf_valid_3857 = gdf_valid_3857.copy()
    if "positive_rate" in gdf_valid_3857.columns and "pr_auc" in gdf_valid_3857.columns:
        prev = pd.to_numeric(gdf_valid_3857["positive_rate"], errors="coerce").replace(0, np.nan)
        gdf_valid_3857["pr_auc_lift"] = pd.to_numeric(
            gdf_valid_3857["pr_auc"], errors="coerce") / prev
    # No lift transform for Recall@5% — absolute values are biome-comparable.

    bounds_m = land_mask_3857.total_bounds   # [minx, miny, maxx, maxy]
    pad_m    = 200_000.0
    xlim     = (bounds_m[0] - pad_m, bounds_m[2] + pad_m)
    ylim     = (bounds_m[1] - pad_m, bounds_m[3] + pad_m)

    # ── Load country borders for geographic orientation ───────────────────────
    print("  Loading country borders...")
    country_borders_3857 = _load_country_borders_3857(region)

    # ── Colormaps and normalisations ─────────────────────────────────────────
    # Both panels use TwoSlopeNorm centred at the random baseline (lift=1 or
    # recall=0.05).  This places neutral yellow at the random-classifier line
    # and reserves red for below-random performance, making the single
    # below-random biome visually distinct without distorting the above-random
    # range.  The colorbar shows true data values; the centre tick is labelled
    # "(random)" so readers never need to consult the figure caption to decode
    # the anchor.
    cmap = _get_cmap("RdYlGn")

    def _pow_norm(
        vals: np.ndarray,
        label: str,
        *,
        vmax_cap: Optional[float] = None,
        vmax_fixed: Optional[float] = None,
        gamma: float = CHOROPLETH_GAMMA,
    ) -> PowerNorm:
        """Power-norm colour normalisation anchored at 0.

        Uses PowerNorm(gamma) so that moderately good performance already
        maps to yellow-green, while the highest performers stay deep green.
        The sqrt-like compression (gamma=0.5, the default) is a well-known
        convention for skewed-ratio data.

        vmax is data-driven with an optional cap.  Pass vmax_fixed to force
        an exact upper bound regardless of the data range (e.g. vmax_fixed=1.0
        for Recall@5% — academic honesty: the scale always shows 0 → 1).
        """
        vals = np.asarray(vals, dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            return PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)
        max_val = float(np.nanmax(finite))
        if vmax_fixed is not None:
            vmax = float(vmax_fixed)
        else:
            vmax = max_val * 1.05
            if vmax_cap is not None:
                vmax = min(vmax, float(vmax_cap))
        vmax = max(vmax, 1e-6)
        cap_note = (f" (fixed={vmax_fixed:g})" if vmax_fixed is not None
                    else f" (cap={vmax_cap:g})" if vmax_cap is not None else "")
        print(f"  PowerNorm [{label}]: vmin=0.0, vmax={vmax:.3f}, "
              f"gamma={gamma:.2f}{cap_note}")
        return PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

    # Font sizing: match forward risk-map style, but scaled down for smaller panels.
    # (Forward maps are 14" wide; these panels are ~7.5" wide.)
    legend_fs = 10
    try:
        # Import lazily to avoid coupling script execution order.
        from scripts.regions.shared.results.config import PROFILE as _RESULTS_PROFILE  # type: ignore
        legend_fs = int(round(float(_RESULTS_PROFILE.get(region, {}).get("map_legend_fontsize", 14)) * 0.75))
        legend_fs = max(10, legend_fs)
    except Exception:
        pass

    vmax_caps = LIFT_VMAX_CAPS.get(region, {})

    use_pr_lift = "pr_auc_lift" in gdf_valid_3857.columns and \
                  gdf_valid_3857["pr_auc_lift"].notna().any()

    if use_pr_lift:
        col_pr  = "pr_auc_lift"
        cbar_pr = "PR-AUC lift (×random baseline)"
        pr_cap  = vmax_caps.get("pr_auc_lift")
        norm_pr = _pow_norm(
            gdf_valid_3857["pr_auc_lift"].to_numpy(),
            label="PR-AUC lift",
            vmax_cap=pr_cap,
        )
        pr_cap_applied = (
            pr_cap is not None and
            float(np.nanmax(gdf_valid_3857["pr_auc_lift"].dropna())) > pr_cap
        )
    else:
        col_pr  = "pr_auc"
        cbar_pr = "PR-AUC"
        pr_cap  = None
        pr_vals = gdf_valid["pr_auc"].dropna().to_numpy()
        norm_pr = _pow_norm(pr_vals, label="PR-AUC (raw)")
        pr_cap_applied = False

    if has_recall:
        col_r5   = "recall_at_5pct"
        cbar_r5  = "Recall at top-5% of predictions (fraction of PA events)"
        r5_vals  = gdf_valid_3857["recall_at_5pct"].dropna().to_numpy()
        # vmax fixed at 1.0 for academic honesty: scale always shows the full range
        norm_r5  = _pow_norm(
            r5_vals,
            label="Recall@5%",
            vmax_fixed=1.0,
        )
        r5_cap_applied = False   # vmax is fixed, not clipped — no "≥" annotation needed

    # ── Biome number assignment (consistent across both panels) ─────────────
    # Sorted alphabetically by short name so the legend is easy to scan.
    # Unicode circled digits ①–⑳ (U+2460+) keep map annotations compact.
    _sorted_biome_names = sorted(
        gdf_valid_3857["biome_name"].dropna().unique(),
        key=lambda n: _short(n),
    )
    _biome_to_num = {
        name: chr(0x2460 + i) for i, name in enumerate(_sorted_biome_names)
    }

    # ── Figure layout ────────────────────────────────────────────────────────
    proj_w      = xlim[1] - xlim[0]
    proj_h      = ylim[1] - ylim[0]
    proj_aspect = proj_h / proj_w
    panel_w     = 7.5
    panel_h     = panel_w * proj_aspect

    import matplotlib.gridspec as _gs
    fig  = plt.figure(figsize=(panel_w * n_panels + 0.5, panel_h + 0.3))
    gs   = _gs.GridSpec(1, n_panels, figure=fig, wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]

    # ── Inner drawing function ───────────────────────────────────────────────
    def _draw_panel(
        ax,
        col,
        norm,
        panel_label,          # e.g. "(a)" — placed top-left corner; pass "" to skip
        cbar_label,
        *,
        random_baseline: Optional[float] = None,
        cap_applied: bool = False,
        show_labels: bool = True,
    ):
        # Layer 0: light grey continent land-mask (ocean stays white)
        land_mask_3857.plot(ax=ax, color=COLOR_MISSING, edgecolor="none", zorder=1)

        # Layer 1: choropleth for biomes with valid data only
        valid = gdf_valid_3857[gdf_valid_3857[col].notna()].copy()
        if not valid.empty:
            valid.plot(
                ax=ax, column=col, cmap=cmap, norm=norm,
                edgecolor="none", linewidth=0, zorder=2,
            )

        # Layer 2: country borders — geographic orientation (Natural Earth 110 m)
        if country_borders_3857 is not None:
            try:
                country_borders_3857.plot(
                    ax=ax, facecolor="none",
                    edgecolor="#555555", linewidth=0.4, alpha=0.65, zorder=3,
                )
            except Exception:
                pass

        # Layer 3: numbered biome labels + legend box.
        # Numbers are placed on both panels so readers can orient in either panel.
        # The legend (number ↔ name mapping) appears only on the left panel.
        # Placement:  fragmented biomes → label each significant fragment;
        #             elongated biomes  → evenly-spaced labels along major axis;
        #             compact biomes    → single representative_point().
        if not valid.empty:
            # Global deduplication: track placed label positions to avoid
            # crowding. Large biomes claim their spot first; small/fragmented
            # biomes are guaranteed ≥1 label even if their best candidate is
            # near an already-placed one.
            min_dist = (xlim[1] - xlim[0]) * 0.055
            placed: list[tuple[float, float]] = []

            try:
                rows_sorted = sorted(
                    valid.iterrows(),
                    key=lambda kv: kv[1].geometry.area,
                    reverse=True,
                )
            except Exception:
                rows_sorted = list(valid.iterrows())

            for _, row in rows_sorted:
                try:
                    name = str(row.get("biome_name", ""))
                    num  = _biome_to_num.get(name, "?")
                    pts  = _get_label_points(row.geometry)
                    if not pts:
                        continue

                    placed_for_biome: list[tuple[float, float]] = []
                    for px, py in pts:
                        too_close = any(
                            (px - px2) ** 2 + (py - py2) ** 2 < min_dist ** 2
                            for px2, py2 in placed
                        )
                        if not too_close:
                            ax.annotate(
                                num,
                                xy=(px, py),
                                fontsize=9.5,
                                ha="center", va="center",
                                fontweight="bold",
                                color="#111111",
                                zorder=5,
                            )
                            placed.append((px, py))
                            placed_for_biome.append((px, py))

                    # Guarantee ≥1 label per biome: pick the candidate that is
                    # furthest from all already-placed labels.
                    if not placed_for_biome:
                        if placed:
                            best_pt = max(
                                pts,
                                key=lambda p: min(
                                    (p[0] - px2) ** 2 + (p[1] - py2) ** 2
                                    for px2, py2 in placed
                                ),
                            )
                        else:
                            best_pt = pts[0]
                        ax.annotate(
                            num,
                            xy=best_pt,
                            fontsize=9.5,
                            ha="center", va="center",
                            fontweight="bold",
                            color="#111111",
                            zorder=5,
                        )
                        placed.append(best_pt)
                except Exception:
                    pass

        # Legend box — left panel only
        if show_labels and not valid.empty:
            legend_lines = [
                f"{chr(0x2460 + i)}\u2002{_short(name)}"
                for i, name in enumerate(_sorted_biome_names)
            ]
            ax.text(
                0.98, 0.03,
                "\n".join(legend_lines),
                transform=ax.transAxes,
                fontsize=8.5,
                va="bottom", ha="right",
                color="#111111",
                zorder=10,
                linespacing=1.6,
                bbox=dict(
                    boxstyle="round,pad=0.55",
                    facecolor="white",
                    edgecolor="#aaaaaa",
                    alpha=0.92,
                    linewidth=0.6,
                ),
            )

        # Colorbar with random-baseline reference line.
        # We annotate the bar with a dashed line + text in axes coordinates.
        # The fractional position is computed from the linear norm so the line
        # sits correctly at whatever value the random baseline maps to.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.52, pad=0.025, aspect=22)
        cbar.set_label(cbar_label, fontsize=9.5)
        cbar.ax.tick_params(labelsize=8)

        if random_baseline is not None:
            try:
                # Random-baseline reference on the colorbar.
                # norm(x) returns the correctly transformed fractional position
                # for any norm type (linear, power, two-slope, etc.).
                frac = float(norm(random_baseline))
                frac = max(0.02, min(0.98, frac))
                # The colorbar IMAGE is drawn as a raster that covers the axes,
                # so axhline/plot drawn before savefig can appear behind it.
                # The reliable workaround is to draw on top of the axes
                # AFTER the colorbar image by appending a Line2D artist with a
                # high zorder, and using transAxes so the position matches frac.
                import matplotlib.lines as _mlines
                ref_line = _mlines.Line2D(
                    [0, 1], [frac, frac],
                    transform=cbar.ax.transAxes,
                    color="#111111", linewidth=1.8, linestyle="--",
                    alpha=0.90, clip_on=False, zorder=99,
                )
                cbar.ax.add_artist(ref_line)

                # Label: left of the bar, vertically centred on the line.
                # Placed on the LEFT side (x<0) so it never conflicts with the
                # colorbar ylabel or tick labels, which live on the right.
                if random_baseline >= 1.0:
                    rand_lbl = f"{random_baseline:.3g}× (random)"
                else:
                    rand_lbl = f"{random_baseline * 100:.0f}% (random)"
                cbar.ax.text(
                    -0.1, frac, rand_lbl,
                    transform=cbar.ax.transAxes,
                    fontsize=7.5, va="center", ha="right",
                    color="#111111", clip_on=False, zorder=99,
                )
                # Cap indicator: prefix "≥" to the top tick if values are clipped
                if cap_applied:
                    try:
                        fig.canvas.draw()   # force tick label generation
                        tl = cbar.ax.get_yticklabels()
                        if tl and tl[-1].get_text():
                            tl[-1].set_text(f"≥{tl[-1].get_text()}")
                            cbar.ax.set_yticklabels(tl)
                    except Exception:
                        pass
            except Exception:
                pass

        # Map cosmetics — no title; panel label in top-left corner instead
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        if panel_label:
            ax.text(
                0.01, 0.99, panel_label,
                transform=ax.transAxes,
                fontsize=12, fontweight="bold",
                va="top", ha="left",
                color="#111111",
            )

    # ── Left panel: PR-AUC (lift) ────────────────────────────────────────────
    _draw_panel(
        axes[0], col_pr, norm_pr,
        panel_label="(a)",
        cbar_label=cbar_pr,
        random_baseline=1.0 if use_pr_lift else None,
        cap_applied=pr_cap_applied,
        show_labels=True,
    )

    # ── Right panel: Recall@5% (absolute) ────────────────────────────────────
    if has_recall:
        _draw_panel(
            axes[1], col_r5, norm_r5,
            panel_label="(b)",
            cbar_label=cbar_r5,
            random_baseline=0.05,
            cap_applied=r5_cap_applied,
            show_labels=False,
        )

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
        plot_biome_performance_map(lobo_df, layer2_df, region, fig3_path)

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
