#!/usr/bin/env python3
"""Stage 3: Forward results — maps, scenario analysis, breakdowns, gap analysis.

Reads forward_scored_2024.parquet from forward/{model_type}/ and
forward_coverage_baseline.json, and produces all paper-ready outputs.

BAU Forecast
  - forward_probability_map.pdf/.png   — continuous P(protection by 2030)
  - forward_risk_map_bau.pdf           — binary map, top X% by BAU designation volume

30x30 Scenario Analysis
  - forward_scenario_moderate.pdf      — top pixels to reach 25% coverage
  - forward_scenario_30x30.pdf         — top pixels to reach 30% coverage (headline)

Country and biome breakdowns
  - forward_country_breakdown.csv/.tex
  - forward_biome_breakdown.csv

Gap analysis (Biodiversity Capture Rate)
  - forward_gap_analysis.pdf           — 4-panel map

Summary JSON
  - forward_scenario_summary.json      — all structured numbers for paper tables

All km² figures use pixel_area_km2() with pixel_size_m extracted from the
backbone raster transform.

Outputs are written under resolve_forward_dir (repo or $SCRATCH/outputs/.../forward
when SCRATCH is set) in the {model_type}/ subfolder.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import platform

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

from scripts.regions.shared.geo_utils import pixel_area_km2  # noqa: E402
from scripts.regions.shared.forward.config import (  # noqa: E402
    DATA_SUBDIR,
    FORWARD_PA_HOLE_COLOR,
    HOTSPOT_REGIONS,
    ISO_CODES,
    MODEL_PREFIX,
    OUTPUTS_SUBDIR,
    PROBABILITY_MAP_DISPLAY_GAMMA,
    PROBABILITY_MAP_PERCENTILE_MAX,
    PROBABILITY_MAP_PERCENTILE_MIN,
    PROBABILITY_MAP_TRANSFORMATION,
    REGION_LABEL,
    forward_dir_search_paths,
    get_repo_root,
)
# Training-style boundaries expect PA3030_RESULTS_REGION to match forward region
os.environ["PA3030_RESULTS_REGION"] = OUTPUTS_SUBDIR

from scripts.regions.shared.results.boundaries import get_region_boundary  # noqa: E402
from scripts.regions.shared.results.results_core import (  # noqa: E402
    PROBABILITY_MAP_COLORMAP,
    _add_latlon_ticks,
    _plot_backbone_background,
    points_to_raster,
)

warnings.filterwarnings("ignore", category=UserWarning)

from scripts.regions.shared.training.utils import WandbRunLogger  # noqa: E402

# ── Map style constants ───────────────────────────────────────────────────────
MAP_DPI      = 300
FORWARD_BACKGROUND_COLOR = "#F2F2F2"  # very light gray: visible vs white, but unobtrusive
SCENARIO_COLORS = {
    "bau":      "#d62728",
    "moderate": "#e87e2b",
    "30x30":    "#1a78c2",
}


def _safe_savefig(
    path: Path,
    *,
    dpi: int,
    bbox_inches: str = "tight",
    pad_inches: float = 0.20,
) -> bool:
    """Save a figure (PNG reliably; PDF best-effort).

    Some macOS + Matplotlib/Python builds can fail when writing large PDFs due to
    font embedding/backends; those failures can also trigger secondary exceptions
    during PDF cleanup. We treat PDF outputs as optional so the pipeline can finish.
    """
    try:
        if str(path).lower().endswith(".pdf"):
            # Default: no PDFs for local runs. Enable explicitly (e.g. in SLURM) via PA3030_SAVE_PDF=1.
            if os.environ.get("PA3030_SAVE_PDF", "0").strip().lower() in {"0", "false", "no", "off"}:
                return False
            # Prefer TrueType (Type 42) over Type 3 fonts.
            with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
                plt.savefig(
                    path,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches if bbox_inches == "tight" else 0.0,
                )
        else:
            plt.savefig(
                path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches if bbox_inches == "tight" else 0.0,
            )
        return True
    except Exception as e:
        msg = f"  WARNING: Failed to save figure: {path} ({type(e).__name__}: {e})"
        # Known issue: Python 3.13 + Matplotlib PDF backend can error on some macOS builds.
        if str(path).lower().endswith(".pdf") and platform.system() == "Darwin":
            msg += " — PDF export is optional; PNG was still written. If you need PDFs, try Python 3.12 and/or Matplotlib<3.10, or set PA3030_SAVE_PDF=0 to silence."
        print(msg)
        return False

# ── Column names ──────────────────────────────────────────────────────────────
PROBA_COL = "y_pred_proba_calibrated"
RAW_COL   = "y_pred_proba_raw"

# GSN: regional merge stacks `ready/GSN/gsn_*_mask_1km.tif` alphabetically → band order
# is climate stabilisation (b1), high biodiversity (b2), … — see gsn_preprocessing /
# 3_merging merge (sorted glob). technical_guide.md treats GSN_b2 as biodiversity.
GSN_BIODIVERSITY_COL = "GSN_b2"
GSN_CLIMATE_STABILISATION_COL = "GSN_b1"


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_scored_parquet(forward_dirs: List[Path], model_type: str) -> Path:
    for fd in forward_dirs:
        cand = fd / model_type / "forward_scored_2024.parquet"
        if cand.exists():
            return cand
    checked = "\n".join(f"  {fd / model_type}" for fd in forward_dirs)
    raise FileNotFoundError(
        "forward_scored_2024.parquet not found. Checked:\n"
        f"{checked}\n"
        "Run predict_core (3_forward_predict.py) first."
    )


def _try_resolve_scored_parquet(forward_dirs: List[Path], model_type: str) -> Optional[Path]:
    try:
        return resolve_scored_parquet(forward_dirs, model_type)
    except Exception:
        return None


def _resolve_deployment_artifact_flexible(
    repo_root: Path, data_subdir: str, outputs_subdir: str, model_prefix: str, model_type: str,
) -> Optional[Path]:
    """Find the most recent deployment artifact (*.pkl).

    Primary: config.ml_models_dir_search_paths(repo_root, data_subdir)
    Fallback: outputs/{region}/results/ml_models/** (legacy cache)
    """
    from scripts.regions.shared.forward.config import ml_models_dir_search_paths  # local import

    pattern = f"{model_prefix}_{model_type}_deployment_*.pkl"
    for model_dir in ml_models_dir_search_paths(repo_root, data_subdir):
        candidates = sorted(model_dir.glob(pattern), reverse=True)
        if candidates:
            return candidates[0]

    # Legacy / dev-cache location (used in some local runs)
    legacy_root = repo_root / f"outputs/{outputs_subdir}/results/ml_models"
    if legacy_root.exists():
        try:
            candidates = sorted(legacy_root.rglob(pattern), reverse=True)
            if candidates:
                return candidates[0]
        except Exception:
            pass

    return None


def _load_scored_min(scored_path: Path) -> pd.DataFrame:
    """Load minimal columns for spatial agreement plots (fast)."""
    cols = [c for c in ["row", "col", "x", "y", PROBA_COL, RAW_COL] if c]
    tbl = pq.read_table(scored_path, columns=[c for c in cols if c in pq.ParquetFile(scored_path).schema_arrow.names])
    df = tbl.to_pandas()
    if PROBA_COL not in df.columns and RAW_COL in df.columns:
        df[PROBA_COL] = df[RAW_COL]
    return df


def create_model_agreement_map_top1pct(
    forward_dirs: List[Path],
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
    region_label: str,
    *,
    top_pct: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """Map where LGBM and RF agree on the top `top_pct` highest-risk pixels.

    Categories (top_pct of each model, computed by percentile threshold):
      - both     : pixel in top_pct for both models
      - lgbm_only: only LGBM top_pct
      - rf_only  : only RF top_pct
    """
    print("\n" + "=" * 70)
    print(f"MODEL AGREEMENT MAP (top {top_pct:.0%}: LGBM vs RF)")
    print("=" * 70)

    lgbm_path = _try_resolve_scored_parquet(forward_dirs, "lgbm")
    rf_path   = _try_resolve_scored_parquet(forward_dirs, "rf")
    if lgbm_path is None or rf_path is None:
        print("  Skipping — both LGBM and RF forward_scored_2024.parquet are required.")
        return None

    print(f"  LGBM scored: {lgbm_path}")
    print(f"  RF scored:   {rf_path}")

    df_l = _load_scored_min(lgbm_path)
    df_r = _load_scored_min(rf_path)

    # Inner join on pixel IDs (row, col) to align arrays robustly
    merged = df_l[["row", "col", "x", "y", PROBA_COL]].merge(
        df_r[["row", "col", PROBA_COL]],
        on=["row", "col"],
        how="inner",
        suffixes=("_lgbm", "_rf"),
    )
    if merged.empty:
        print("  WARNING: no overlapping pixels between LGBM and RF scored outputs — skipping.")
        return None

    p_l = merged[f"{PROBA_COL}_lgbm"].values.astype(np.float64)
    p_r = merged[f"{PROBA_COL}_rf"].values.astype(np.float64)
    thr_l = float(np.percentile(p_l, 100.0 * (1.0 - top_pct)))
    thr_r = float(np.percentile(p_r, 100.0 * (1.0 - top_pct)))
    l_top = p_l >= thr_l
    r_top = p_r >= thr_r

    both = l_top & r_top
    l_only = l_top & ~r_top
    r_only = r_top & ~l_top

    n = len(merged)
    stats = {
        "n_pixels_overlap": int(n),
        "top_pct": float(top_pct),
        "lgbm_threshold": thr_l,
        "rf_threshold": thr_r,
        "n_both": int(both.sum()),
        "n_lgbm_only": int(l_only.sum()),
        "n_rf_only": int(r_only.sum()),
        "share_both_of_union": float(both.sum() / max((l_top | r_top).sum(), 1)),
    }
    print(
        "  Counts (overlap pixels): "
        f"both={stats['n_both']:,}, lgbm_only={stats['n_lgbm_only']:,}, rf_only={stats['n_rf_only']:,}",
    )
    print(f"  Agreement (Jaccard on top sets): {stats['share_both_of_union']:.3f}")

    # Rasterize categorical overlay: 3=both, 2=lgbm_only, 1=rf_only, NaN=background
    xm = merged["x"].values.astype(np.float64)
    ym = merged["y"].values.astype(np.float64)
    cat = np.full(n, np.nan, dtype=np.float32)
    cat[r_only] = 1.0
    cat[l_only] = 2.0
    cat[both]   = 3.0

    proj_bounds = _resolve_plot_bounds_3857(repo_root, DATA_SUBDIR, baseline)
    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

    grid, gext = points_to_raster(
        xm, ym, cat,
        target_resolution=1000.0,
        agg_func="max",
        extent_bounds=proj_bounds,
    )

    proj_width  = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    fig_w = 14.0
    fig_h = fig_w * (proj_height / max(proj_width, 1e-9))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    used_bb = _plot_backbone_background(
        ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
    )
    if not used_bb:
        region_proj.plot(
            ax=ax, color=FORWARD_PA_HOLE_COLOR, edgecolor="none",
            linewidth=0, zorder=0,
        )

    cmap = mcolors.ListedColormap([
        "#00000000",  # placeholder for 0 (unused)
        "#ff7f0e",    # 1 = RF only
        "#1f77b4",    # 2 = LGBM only
        "#2ca02c",    # 3 = Both
    ])
    # Mask non-categories
    disp = np.where(np.isnan(grid), np.nan, grid)
    ax.imshow(
        disp,
        extent=gext,
        cmap=cmap,
        vmin=0,
        vmax=3,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        alpha=0.88,
        zorder=2,
    )

    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Model Agreement Map — Top {top_pct:.0%} Risk Pixels (2025–2030) — {region_label}\n"
        "Green = both models | Blue = LGBM only | Orange = RF only",
        fontsize=12,
    )
    legend_elements = [
        mpatches.Patch(color="#2ca02c", label=f"Both (n={stats['n_both']:,})"),
        mpatches.Patch(color="#1f77b4", label=f"LGBM only (n={stats['n_lgbm_only']:,})"),
        mpatches.Patch(color="#ff7f0e", label=f"RF only (n={stats['n_rf_only']:,})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)
    ax.text(
        0.98, 0.02,
        f"Jaccard(top sets) = {stats['share_both_of_union']:.3f}\n"
        f"thr(LGBM)={thr_l:.6f} | thr(RF)={thr_r:.6f}",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)
    plt.tight_layout(pad=0.5)

    for ext in ["png", "pdf"]:
        p = output_dir / f"model_agreement_top{int(top_pct*100)}pct.{ext}"
        if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass

    # Write a small JSON sidecar for paper tables / sanity checks
    try:
        out_json = output_dir / f"model_agreement_top{int(top_pct*100)}pct.json"
        with open(out_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: {out_json}")
    except Exception as e:
        print(f"  WARNING: could not write agreement JSON: {e}")

    return stats


def create_feature_importance_plot(
    repo_root: Path,
    data_subdir: str,
    outputs_subdir: str,
    model_prefix: str,
    model_type: str,
    output_dir: Path,
) -> None:
    """Plot feature importances from the deployment artifact (if available)."""
    print("\n" + "=" * 70)
    print(f"FEATURE IMPORTANCE PLOT ({model_type.upper()})")
    print("=" * 70)

    artifact_path = _resolve_deployment_artifact_flexible(
        repo_root, data_subdir, outputs_subdir, model_prefix, model_type,
    )
    if artifact_path is None or not artifact_path.exists():
        print("  Skipping — deployment artifact not found (cannot extract feature importance).")
        return
    print(f"  Artifact: {artifact_path}")

    import pickle
    try:
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)
    except ModuleNotFoundError as e:
        # Deployment artifacts may pickle model objects that require optional deps
        # (e.g. LightGBM) not present in lightweight analysis environments.
        print(f"  Skipping — cannot unpickle deployment artifact ({e}).")
        return
    except Exception as e:
        print(f"  Skipping — failed to load deployment artifact ({type(e).__name__}: {e}).")
        return

    model = artifact.get("model")
    feature_cols = artifact.get("feature_cols") or []
    if model is None or not feature_cols:
        print("  WARNING: artifact missing 'model' or 'feature_cols' — skipping.")
        return

    imp_vals: Optional[np.ndarray] = None
    title_suffix = ""
    try:
        if model_type == "lgbm":
            # LightGBM Booster or sklearn wrapper
            booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None) or model
            if hasattr(booster, "feature_importance"):
                imp_vals = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
                title_suffix = " (LightGBM gain)"
            elif hasattr(model, "feature_importances_"):
                imp_vals = np.asarray(model.feature_importances_, dtype=np.float64)
                title_suffix = " (LightGBM feature_importances_)"
        else:
            if hasattr(model, "feature_importances_"):
                imp_vals = np.asarray(model.feature_importances_, dtype=np.float64)
                title_suffix = " (RF impurity)"
    except Exception as e:
        print(f"  WARNING: failed to extract importances: {e}")
        return

    if imp_vals is None or len(imp_vals) == 0:
        print("  WARNING: no importances found on model — skipping.")
        return

    if len(imp_vals) != len(feature_cols):
        # Best-effort alignment
        n = min(len(imp_vals), len(feature_cols))
        imp_vals = imp_vals[:n]
        feature_cols = feature_cols[:n]

    order = np.argsort(imp_vals)[::-1]
    top_n = min(25, len(order))
    top_idx = order[:top_n]
    top_feats = [feature_cols[i] for i in top_idx][::-1]
    top_imps = imp_vals[top_idx][::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.28)))
    ax.barh(top_feats, top_imps, color="#4c96d7", edgecolor="white", linewidth=0.5)
    ax.set_title(f"Feature Importance{title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        p = output_dir / f"feature_importance_{model_type}.{ext}"
        if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass


def resolve_coverage_baseline(forward_dirs: List[Path]) -> Dict:
    for fd in forward_dirs:
        for name in ["coverage_baseline.json", "forward_coverage_baseline.json"]:
            cand = fd / name
            if cand.exists():
                with open(cand) as f:
                    return json.load(f)
    checked = "\n".join(f"  {fd}" for fd in forward_dirs)
    raise FileNotFoundError(
        "coverage_baseline.json not found. Checked:\n"
        f"{checked}\n"
        "Run coverage_core (1_forward_coverage_baseline.py) first."
    )


def resolve_backbone_pixel_size(repo_root: Path, data_subdir: str) -> float:
    """Extract pixel size from backbone raster; fall back to 1000.0 m."""
    try:
        import rasterio
        scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
        candidates: list[Path] = []
        if scratch is not None:
            candidates += [
                scratch / f"data/{data_subdir}/ready/backbone.tif",
                scratch / f"data/{data_subdir}/ready/backbone/backbone.tif",
            ]
        candidates += [
            repo_root / f"data/{data_subdir}/ready/backbone.tif",
            repo_root / f"data/{data_subdir}/ready/backbone/backbone.tif",
        ]
        for cand in candidates:
            if cand.exists():
                with rasterio.open(cand) as src:
                    ps = abs(src.transform.a)
                    print(f"  Pixel size from backbone: {ps:.2f} m")
                    return float(ps)
    except Exception as e:
        print(f"  Could not read backbone raster: {e}")
    print("  Falling back to pixel_size_m=1000.0 m")
    return 1000.0


def resolve_gsn_raster(repo_root: Path, data_subdir: str) -> Optional[Path]:
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    candidates: list[Path] = []
    if scratch is not None:
        candidates.append(
            scratch / f"data/{data_subdir}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif"
        )
    candidates.append(
        repo_root / f"data/{data_subdir}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif"
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_country_shapefile() -> Optional[Any]:
    """Load world country boundaries, compatible with GeoPandas ≥ 1.0."""
    try:
        from scripts.regions.shared.forward.coverage_core import _load_naturalearth_lowres
        return _load_naturalearth_lowres()
    except Exception:
        return None


def resolve_country_iso3_raster(repo_root: Path, data_subdir: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve country ISO3 ID raster + mapping JSON (policy preprocessing output).

    Expected outputs:
      data/{region}/ready/policy/country_iso3.tif
      data/{region}/ready/policy/country_iso3_mapping.json
    """
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    candidates: list[Tuple[Path, Path]] = []
    if scratch is not None:
        candidates.append(
            (
                scratch / f"data/{data_subdir}/ready/policy/country_iso3.tif",
                scratch / f"data/{data_subdir}/ready/policy/country_iso3_mapping.json",
            )
        )
    candidates.append(
        (
            repo_root / f"data/{data_subdir}/ready/policy/country_iso3.tif",
            repo_root / f"data/{data_subdir}/ready/policy/country_iso3_mapping.json",
        )
    )
    for tif, js in candidates:
        if tif.exists() and js.exists():
            return tif, js
    return None, None


def _load_country_iso3_lookup(
    repo_root: Path, data_subdir: str,
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
    """Load country ID raster and return (id_grid, id->iso3 mapping)."""
    tif_path, map_path = resolve_country_iso3_raster(repo_root, data_subdir)
    if tif_path is None or map_path is None:
        return None, None
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            grid = src.read(1)
        with open(map_path) as f:
            raw = json.load(f)
        mapping: Dict[int, str] = {}
        for k, v in (raw or {}).items():
            try:
                ki = int(k)
            except Exception:
                continue
            if v is None:
                continue
            mapping[ki] = str(v).strip().upper()
        return grid, mapping
    except Exception:
        return None, None


def resolve_backbone_path_for_plot(
    repo_root: Path, data_subdir: str, baseline: Dict[str, Any],
) -> Optional[Path]:
    """Resolve backbone GeoTIFF for map underlay (same search order as pixel size)."""
    raw = baseline.get("backbone_path")
    if raw:
        p = Path(str(raw))
        if p.exists():
            return p
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    candidates: list[Path] = []
    if scratch is not None:
        candidates += [
            scratch / f"data/{data_subdir}/ready/backbone.tif",
            scratch / f"data/{data_subdir}/ready/backbone/backbone.tif",
        ]
    candidates += [
        repo_root / f"data/{data_subdir}/ready/backbone.tif",
        repo_root / f"data/{data_subdir}/ready/backbone/backbone.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_wdpa_2024_path_for_plot(
    repo_root: Path, data_subdir: str, baseline: Dict[str, Any],
) -> Optional[Path]:
    """Resolve WDPA 2024 raster used to overlay current PA extent."""
    raw = baseline.get("wdpa_2024_path")
    if raw:
        p = Path(str(raw))
        if p.exists():
            return p
    try:
        from scripts.regions.shared.forward.config import WDPA_2024_FILENAME  # local import
        from scripts.regions.shared.forward.coverage_core import resolve_raster  # local import
        return resolve_raster(WDPA_2024_FILENAME, data_subdir, subdirs=["WDPA", "wdpa"])
    except Exception:
        return None


def _resolve_plot_bounds_3857(
    repo_root: Path, data_subdir: str, baseline: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    """Map bounds in EPSG:3857 from our own rasters (no internet / boundaries needed)."""
    try:
        import rasterio
        backbone = resolve_backbone_path_for_plot(repo_root, data_subdir, baseline)
        if backbone and Path(backbone).exists():
            with rasterio.open(backbone) as src:
                b = src.bounds
                return (float(b.left), float(b.bottom), float(b.right), float(b.top))
    except Exception:
        pass
    # Fallback: region boundary polygon (may depend on Natural Earth availability).
    region_gdf = get_region_boundary(None)
    if region_gdf.crs is None:
        region_gdf = region_gdf.set_crs("EPSG:4326", allow_override=True)
    region_proj = region_gdf.to_crs("EPSG:3857")
    return tuple(region_proj.total_bounds.astype(float))


def _overlay_current_pa_extent(
    ax: Any,
    repo_root: Path,
    data_subdir: str,
    baseline: Dict[str, Any],
    proj_bounds: Tuple[float, float, float, float],
    *,
    color: str = FORWARD_PA_HOLE_COLOR,
    alpha: float = 0.65,
    zorder: int = 1,
) -> bool:
    """Overlay WDPA 2024 protected pixels (current PA extent) on an EPSG:3857 axis."""
    wdpa_path = resolve_wdpa_2024_path_for_plot(repo_root, data_subdir, baseline)
    if wdpa_path is None or not Path(wdpa_path).exists():
        return False
    try:
        import rasterio
        with rasterio.open(wdpa_path) as src:
            wdpa = src.read(1)
            b = src.bounds
            mask = (wdpa == 1)
            if mask.sum() == 0:
                return False
            disp = np.where(mask, 1.0, np.nan).astype(np.float32)
            extent = (float(b.left), float(b.right), float(b.bottom), float(b.top))
        ax.imshow(
            disp,
            extent=extent,
            cmap=mcolors.ListedColormap([color]),
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
            alpha=alpha,
            zorder=zorder,
        )
        ax.set_xlim(proj_bounds[0], proj_bounds[2])
        ax.set_ylim(proj_bounds[1], proj_bounds[3])
        return True
    except Exception:
        return False


def _forward_norm_params(proba: np.ndarray) -> Dict[str, Any]:
    """Percentile edges for probability-map normalisation (full unprotected sample)."""
    t = PROBABILITY_MAP_TRANSFORMATION
    pmi, pma = PROBABILITY_MAP_PERCENTILE_MIN, PROBABILITY_MAP_PERCENTILE_MAX
    if t == "log":
        lv = np.log10(np.maximum(proba.astype(np.float64), 1e-10))
        return {
            "t": "log",
            "lo": float(np.percentile(lv, pmi)),
            "hi": float(np.percentile(lv, pma)),
        }
    lo = float(np.percentile(proba, pmi))
    hi = float(np.percentile(proba, pma))
    return {"t": t, "lo": lo, "hi": hi}


def _forward_apply_norm(proba: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    """Map raw probabilities to [0, 1] for display using params from _forward_norm_params."""
    v = np.asarray(proba, dtype=np.float64)
    if p["t"] == "log":
        lv = np.log10(np.maximum(v, 1e-10))
        span = max(p["hi"] - p["lo"], 1e-12)
        n = np.clip((lv - p["lo"]) / span, 0.0, 1.0)
    else:
        span = max(p["hi"] - p["lo"], 1e-12)
        cl = np.clip(v, p["lo"], p["hi"])
        n = (cl - p["lo"]) / span
        if p["t"] == "sqrt":
            n = np.sqrt(n)
    g = PROBABILITY_MAP_DISPLAY_GAMMA
    if abs(g - 1.0) > 1e-9:
        n = np.clip(n ** g, 0.0, 1.0)
    return n.astype(np.float32)


def _forward_norm_cbar_label(p: Dict[str, Any]) -> str:
    t = p["t"]
    pmi, pma = PROBABILITY_MAP_PERCENTILE_MIN, PROBABILITY_MAP_PERCENTILE_MAX
    if t == "log":
        core = f"log10 stretch ({pmi}–{pma} percentile of log p)"
    elif t == "sqrt":
        core = f"sqrt stretch ({pmi}–{pma} percentile)"
    else:
        core = f"linear stretch ({pmi}–{pma} percentile)"
    if abs(PROBABILITY_MAP_DISPLAY_GAMMA - 1.0) > 1e-9:
        core += f", γ={PROBABILITY_MAP_DISPLAY_GAMMA:.2f}"
    return f"P(protection by 2030) — {core}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_scored(scored_path: Path) -> pd.DataFrame:
    print(f"\nLoading scored parquet: {scored_path.name}")
    df = pq.read_table(scored_path).to_pandas()
    print(f"  {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    # Ensure calibrated column exists; fall back to raw
    if PROBA_COL not in df.columns:
        print(f"  WARNING: '{PROBA_COL}' not found — using raw probabilities")
        df[PROBA_COL] = df[RAW_COL]
    return df


# ── Area calculation ──────────────────────────────────────────────────────────

def add_area_col(df: pd.DataFrame, pixel_size_m: float) -> pd.DataFrame:
    """Add 'area_km2' column using latitude-corrected pixel area."""
    df = df.copy()
    if "y" not in df.columns:
        print("  WARNING: 'y' coordinate not in scored parquet — using pixel_size_m² / 1e6")
        df["area_km2"] = (pixel_size_m / 1000.0) ** 2
    else:
        df["area_km2"] = pixel_area_km2(df["y"].values, pixel_size_m=pixel_size_m)
    return df


# ── Coordinate helpers ────────────────────────────────────────────────────────

def epsg3857_to_lonlat(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert EPSG:3857 metres → WGS84 lon/lat degrees."""
    R = 6378137.0
    lon = np.degrees(x / R)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / R)) - np.pi / 2.0)
    return lon, lat


def lonlat_to_epsg3857(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """WGS84 lon/lat (degrees) → EPSG:3857 metres (Web Mercator)."""
    R = 6378137.0
    lon_r = np.radians(np.asarray(lon, dtype=np.float64))
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    x = R * lon_r
    y = R * np.log(np.tan(np.pi / 4.0 + lat_r / 2.0))
    return x, y


def _lonlat_bounds_to_proj_bounds(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
) -> Tuple[float, float, float, float]:
    """Axis-aligned lon/lat box → tight EPSG:3857 bounds (corner transform)."""
    lons = np.array([lon_min, lon_max, lon_max, lon_min], dtype=np.float64)
    lats = np.array([lat_min, lat_min, lat_max, lat_max], dtype=np.float64)
    xm, ym = lonlat_to_epsg3857(lons, lats)
    return (float(xm.min()), float(ym.min()), float(xm.max()), float(ym.max()))


# ── Scenario thresholds ───────────────────────────────────────────────────────

def compute_scenario_cutoffs(
    df: pd.DataFrame,
    km2_needed_moderate: float,
    moderate_target_pct: float,
    km2_needed_30: float,
    bau_km2: Optional[float] = None,
) -> Tuple[float, float, float, str]:
    """Compute BAU / moderate / 30x30 probability cutoffs.

    Returns (bau_cutoff, moderate_cutoff, full_30x30_cutoff) as probability
    thresholds such that cumulative area of top-ranked pixels equals target.

    BAU target: historical 5-year designation volume (bau_km2 from coverage baseline,
      i.e. new km² protected 2019→2024). Falls back to top-0.5% proxy if not available.
    Moderate target: midpoint between current coverage and 30% (region-adaptive).
    """
    print("\nComputing scenario cutoffs…")
    df_sorted = df.sort_values(PROBA_COL, ascending=False).reset_index(drop=True)
    cum_area = df_sorted["area_km2"].cumsum().values

    def _cutoff(km2_target: float, label: str) -> float:
        if km2_target <= 0:
            print(f"  {label}: target already met (km2_needed={km2_target:.0f})")
            return 1.0  # return max threshold (nothing to add)
        idx = np.searchsorted(cum_area, km2_target, side="left")
        idx = min(idx, len(df_sorted) - 1)
        cutoff = float(df_sorted[PROBA_COL].iloc[idx])
        n_pixels = idx + 1
        actual_km2 = float(cum_area[idx])
        print(f"  {label}: target {km2_target:,.0f} km² → "
              f"top {n_pixels:,} pixels ({actual_km2:,.0f} km²), cutoff p={cutoff:.6f}")
        return cutoff

    # BAU: use historical 5-year designation volume (2019→2024) if available;
    # otherwise fall back to top 0.5% of unprotected pixels as a proxy.
    if bau_km2 is not None and bau_km2 > 0:
        bau_cutoff = _cutoff(bau_km2, "BAU (historical 2019→2024 volume)")
        bau_subtitle = f"Historical 2019→2024 designation volume ({bau_km2:,.0f} km²)"
    else:
        n_bau = max(1, int(len(df_sorted) * 0.005))
        bau_cutoff = float(df_sorted[PROBA_COL].iloc[n_bau - 1])
        bau_km2_actual = float(cum_area[n_bau - 1])
        print(f"  BAU (top 0.5% proxy — WDPA_2019 not available): "
              f"{n_bau:,} pixels ({bau_km2_actual:,.0f} km²), cutoff p={bau_cutoff:.6f}")
        bau_subtitle = f"Top 0.5% highest-risk pixels (proxy — WDPA 2019 unavailable)"

    moderate_cutoff = _cutoff(km2_needed_moderate, f"Moderate (→{moderate_target_pct:.1%})")
    full_cutoff     = _cutoff(km2_needed_30,       "30x30 Full    (→30%)")

    return bau_cutoff, moderate_cutoff, full_cutoff, bau_subtitle


# ── Map: probability ──────────────────────────────────────────────────────────

def create_probability_map(
    df: pd.DataFrame,
    pixel_size_m: float,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
    region_label: str,
) -> None:
    """Web Mercator raster + backbone underlay; aspect matches training results maps."""
    from matplotlib.patches import Patch

    print("\n" + "=" * 70)
    print("CREATING PROBABILITY MAP (BAU forecast)")
    print("=" * 70)

    proba = df[PROBA_COL].values.astype(np.float64)
    norm_p = _forward_norm_params(proba)
    proba_norm = _forward_apply_norm(proba, norm_p)
    print(
        f"  Probability colour scaling: {PROBABILITY_MAP_TRANSFORMATION}, "
        f"percentiles {PROBABILITY_MAP_PERCENTILE_MIN}–{PROBABILITY_MAP_PERCENTILE_MAX}, "
        f"γ={PROBABILITY_MAP_DISPLAY_GAMMA}",
    )

    proj_bounds = _resolve_plot_bounds_3857(repo_root, DATA_SUBDIR, baseline)

    xm = df["x"].values.astype(np.float64)
    ym = df["y"].values.astype(np.float64)
    inside = (
        (xm >= proj_bounds[0]) & (xm <= proj_bounds[2])
        & (ym >= proj_bounds[1]) & (ym <= proj_bounds[3])
    )
    if inside.sum() < len(df):
        print(f"  Clipping {len(df) - inside.sum():,} points outside region hull for map")
    xm, ym, proba_norm = xm[inside], ym[inside], proba_norm[inside]

    raster, extent = points_to_raster(
        xm, ym, proba_norm,
        target_resolution=1000.0,
        agg_func="mean",
        extent_bounds=proj_bounds,
    )
    print(f"  Raster shape: {raster.shape}, extent (3857): {extent}")

    proj_width = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    proj_aspect = proj_height / max(proj_width, 1e-9)
    fig_width = 14.0
    fig_height = fig_width * proj_aspect
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)
    used_bb = _plot_backbone_background(
        ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
    )
    _overlay_current_pa_extent(ax, repo_root, DATA_SUBDIR, baseline, proj_bounds, zorder=1)

    im = ax.imshow(
        raster,
        extent=extent,
        cmap=PROBABILITY_MAP_COLORMAP,
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )
    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(_forward_norm_cbar_label(norm_p), fontsize=10, rotation=270, labelpad=18)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Lower", "Higher"])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"BAU Forecast: P(Protected-Area Designation by 2030) — {region_label}\n"
        "Deployment model (2001–2019), unprotected pixels as of 2024",
        fontsize=12,
    )
    ax.legend(
        handles=[
            Patch(
                facecolor=FORWARD_PA_HOLE_COLOR,
                edgecolor="none",
                label="Existing protected areas (2024)",
            ),
        ],
        loc="lower left",
        fontsize=9,
        framealpha=0.95,
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)
    plt.tight_layout(pad=0.5)

    # Write PNG first (reliable), then PDF best-effort.
    for ext in ["png", "pdf"]:
        outp = output_dir / f"probability_map.{ext}"
        if _safe_savefig(outp, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {outp}")
    try:
        plt.close(fig)
    except Exception:
        pass


# ── Map: binary scenario ──────────────────────────────────────────────────────

def _create_scenario_map(
    x_m: np.ndarray,
    y_m: np.ndarray,
    selected: np.ndarray,
    title: str,
    subtitle: str,
    color: str,
    filename_stem: str,
    output_dir: Path,
    n_pixels: int,
    area_km2: float,
    coverage_pct: float,
    proj_bounds: Tuple[float, float, float, float],
    region_proj: Any,
    backbone_path: Optional[Path],
    repo_root: Path,
    baseline: Dict[str, Any],
    data_subdir: str,
) -> None:
    """Scenario binary map in EPSG:3857 with backbone underlay (no country outlines)."""
    bg_raster, bg_ext = points_to_raster(
        x_m, y_m, np.ones(len(x_m), dtype=np.float32),
        target_resolution=1000.0,
        agg_func="max",
        extent_bounds=proj_bounds,
    )
    bg_disp = np.where(np.isnan(bg_raster), np.nan, 0.12)

    proj_width = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    fig_width = 14.0
    fig_height = fig_width * (proj_height / max(proj_width, 1e-9))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sel_raster, _ = points_to_raster(
        x_m[selected], y_m[selected], np.ones(int(selected.sum()), dtype=np.float32),
        target_resolution=1000.0,
        agg_func="max",
        extent_bounds=proj_bounds,
    )
    sel_disp = np.where(np.isnan(sel_raster), np.nan, 1.0)

    # Halo/outline: draw only the border of selected pixels in black underneath.
    sel_bin = ~np.isnan(sel_raster)
    if sel_bin.any():
        p = np.pad(sel_bin.astype(np.uint8), 1, mode="constant", constant_values=0)
        nbh = [
            p[0:-2, 0:-2], p[0:-2, 1:-1], p[0:-2, 2:],
            p[1:-1, 0:-2], p[1:-1, 1:-1], p[1:-1, 2:],
            p[2:,   0:-2], p[2:,   1:-1], p[2:,   2:],
        ]
        all_neigh = np.all(np.stack(nbh, axis=0).astype(bool), axis=0)
        outline = sel_bin & (~all_neigh)
        outline_disp = np.where(outline, 1.0, np.nan).astype(np.float32)
    else:
        outline_disp = np.full_like(sel_disp, np.nan, dtype=np.float32)

    used_bb = _plot_backbone_background(
        ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
    )
    if (not used_bb) and (region_proj is not None):
        region_proj.plot(
            ax=ax, color=FORWARD_PA_HOLE_COLOR, edgecolor="none",
            linewidth=0, zorder=0,
        )
    _overlay_current_pa_extent(ax, repo_root, data_subdir, baseline, proj_bounds, zorder=1)

    ax.imshow(
        bg_disp,
        extent=bg_ext,
        cmap=mcolors.ListedColormap([FORWARD_BACKGROUND_COLOR]),
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        alpha=1.0,
        zorder=1,
    )
    # Black halo/outline under the selected pixels
    ax.imshow(
        outline_disp,
        extent=bg_ext,
        cmap=mcolors.ListedColormap(["#111111"]),
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        alpha=0.90,
        zorder=2,
    )
    cmap_sel = mcolors.ListedColormap([color])
    ax.imshow(
        sel_disp,
        extent=bg_ext,
        cmap=cmap_sel,
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        alpha=0.98,
        zorder=3,
    )

    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))

    patch = mpatches.Patch(
        color=color,
        label=f"Projected designations ({n_pixels:,} pixels, {area_km2:,.0f} km²)",
    )
    ax.legend(handles=[patch], loc="lower left", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)

    plt.tight_layout(pad=0.5)
    for ext in ["png", "pdf"]:
        p = output_dir / f"{filename_stem}.{ext}"
        if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass


def create_scenario_maps(
    df: pd.DataFrame,
    baseline: Dict,
    bau_cutoff: float,
    moderate_cutoff: float,
    full_cutoff: float,
    output_dir: Path,
    repo_root: Path,
    region_label: str,
    bau_subtitle: str = "BAU designation volume",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("CREATING SCENARIO MAPS")
    print("=" * 70)

    region_proj = None
    proj_bounds = _resolve_plot_bounds_3857(repo_root, DATA_SUBDIR, baseline)
    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

    x_m = df["x"].values.astype(np.float64)
    y_m = df["y"].values.astype(np.float64)
    proba = df[PROBA_COL].values
    area  = df["area_km2"].values
    total_km2 = baseline.get("total_land_km2") or baseline.get("total_sa_km2", 1.0)
    protected_2024_km2 = baseline["protected_2024_km2"]

    scenario_info: Dict[str, Any] = {}
    moderate_pct = baseline.get("moderate_target_pct", 0.25)
    moderate_pct_str = f"{moderate_pct:.0%}"

    scenarios = [
        (proba >= bau_cutoff, "bau",
         f"BAU Forecast — Projected Designations (2025–2030) — {region_label}",
         bau_subtitle,
         SCENARIO_COLORS["bau"], "risk_map_bau"),
        (proba >= moderate_cutoff, "moderate",
         f"Moderate Scenario (→{moderate_pct_str} {region_label} coverage)",
         f"Spatial projection if historical designation preferences continue to {moderate_pct_str} coverage",
         SCENARIO_COLORS["moderate"], "scenario_moderate"),
        (proba >= full_cutoff, "30x30",
         f"30x30 Full Scenario (→30% {region_label} coverage)",
         "Spatial projection if historical designation preferences continue to 30% coverage",
         SCENARIO_COLORS["30x30"], "scenario_30x30"),
    ]

    for mask, key, title, subtitle, color, stem in scenarios:
        n = int(mask.sum())
        km2 = float(area[mask].sum())
        coverage_new = (protected_2024_km2 + km2) / max(total_km2, 1.0)
        print(f"\n  {key}: {n:,} pixels, {km2:,.0f} km², → coverage {coverage_new:.2%}")
        scenario_info[key] = {
            "n_pixels": n,
            "area_km2": round(km2, 2),
            "projected_total_coverage_pct": round(coverage_new, 6),
        }
        if n > 0:
            _create_scenario_map(
                x_m, y_m, mask, title, subtitle, color, stem, output_dir,
                n, km2, coverage_new, proj_bounds, region_proj, backbone_path,
                repo_root, baseline, DATA_SUBDIR,
            )
        else:
            print(f"  WARNING: no pixels selected for {key} scenario — target already met?")

    return scenario_info


# ── Country breakdown ─────────────────────────────────────────────────────────

def create_country_breakdown(
    df: pd.DataFrame,
    baseline: Dict,
    bau_cutoff: float,
    full_cutoff: float,
    output_dir: Path,
    world_gdf: Optional[Any],
    iso_codes: List[str],
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CREATING COUNTRY BREAKDOWN")
    print("=" * 70)

    try:
        # Prefer raster-based country lookup (no internet / no geopandas spatial join).
        iso_grid, iso_map = _load_country_iso3_lookup(repo_root=get_repo_root(), data_subdir=DATA_SUBDIR)
        use_raster = (
            iso_grid is not None
            and iso_map is not None
            and ("row" in df.columns)
            and ("col" in df.columns)
        )
        if use_raster:
            print("  Using country_iso3.tif raster lookup (policy preprocessing output).")
            row_arr = df["row"].values.astype(np.int64)
            col_arr = df["col"].values.astype(np.int64)
            h, w = iso_grid.shape
            valid = (row_arr >= 0) & (row_arr < h) & (col_arr >= 0) & (col_arr < w)
            ids = np.full(len(df), -1, dtype=np.int32)
            ids[valid] = iso_grid[row_arr[valid], col_arr[valid]].astype(np.int32)
            iso3 = np.array([iso_map.get(int(i), None) for i in ids], dtype=object)

            area = df["area_km2"].values
            proba = df[PROBA_COL].values
            joined = pd.DataFrame(
                {
                    "iso_a3": iso3,
                    "name": iso3,  # will be overwritten by baseline lookup if available
                    "area_km2": area,
                    "proba": proba,
                    "bau_sel":  (proba >= bau_cutoff).astype(np.int8),
                    "full_sel": (proba >= full_cutoff).astype(np.int8),
                }
            )
            joined = joined[joined["iso_a3"].isin(iso_codes)].copy()
        else:
            if world_gdf is None:
                print("  Skipping country breakdown — no country_iso3.tif and no Natural Earth boundaries available")
                return pd.DataFrame()
            import geopandas as gpd

            region_world = world_gdf[world_gdf["iso_a3"].isin(iso_codes)].copy()

        # Build per-country area lookup from coverage baseline (raster-based, accurate)
        country_stats_lookup: dict = {
            cs["iso_a3"]: cs
            for cs in baseline.get("country_stats", [])
        }
        if not country_stats_lookup:
            print("  NOTE: country_stats not found in baseline — run 1_forward_coverage_baseline.py "
                  "to populate per-country area stats. Current/projected % will be 0.")

        if not use_raster:
            lon, lat = epsg3857_to_lonlat(df["x"].values, df["y"].values)
            area  = df["area_km2"].values
            proba = df[PROBA_COL].values

            print(f"  Joining {len(df):,} pixels to country boundaries…")
            pts_gdf = gpd.GeoDataFrame(
                {
                    "area_km2": area,
                    "proba": proba,
                    "bau_sel":  (proba >= bau_cutoff).astype(np.int8),
                    "full_sel": (proba >= full_cutoff).astype(np.int8),
                },
                geometry=gpd.points_from_xy(lon, lat),
                crs="EPSG:4326",
            )
            joined = gpd.sjoin(
                pts_gdf, region_world[["iso_a3", "name", "geometry"]],
                how="left", predicate="within",
            )

        rows = []
        for iso, grp in joined.groupby("iso_a3"):
            if use_raster and country_stats_lookup.get(iso, {}).get("country"):
                country_name = country_stats_lookup[iso]["country"]
            else:
                country_name = grp["name"].iloc[0] if len(grp) > 0 else iso
            unprotected_km2 = float(grp["area_km2"].sum())
            bau_new_km2     = float(grp.loc[grp["bau_sel"] == 1, "area_km2"].sum())
            full_new_km2    = float(grp.loc[grp["full_sel"] == 1, "area_km2"].sum())

            cs = country_stats_lookup.get(iso, {})
            total_km2      = cs.get("total_km2", 0.0)
            prot_km2       = cs.get("protected_2024_km2", 0.0)
            current_pct    = cs.get("current_pct_protected", 0.0)
            proj_pct_bau   = (prot_km2 + bau_new_km2)  / max(total_km2, 1.0)
            proj_pct_30x30 = (prot_km2 + full_new_km2) / max(total_km2, 1.0)
            gap_km2        = max(0.0, 0.30 * total_km2 - (prot_km2 + full_new_km2))

            rows.append({
                "iso_a3": iso,
                "country": country_name,
                "total_km2": round(total_km2, 0),
                "protected_2024_km2": round(prot_km2, 0),
                "current_pct_protected": round(current_pct, 4),
                "unprotected_2024_km2": round(unprotected_km2, 0),
                "bau_new_km2": round(bau_new_km2, 0),
                "30x30_new_km2": round(full_new_km2, 0),
                "projected_pct_2030_bau": round(proj_pct_bau, 4),
                "projected_pct_2030_30x30": round(proj_pct_30x30, 4),
                "gap_to_30pct_km2": round(gap_km2, 0),
            })

        country_df = pd.DataFrame(rows).sort_values("30x30_new_km2", ascending=False)
        csv_path = output_dir / "country_breakdown.csv"
        country_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # LaTeX table (top 13 countries)
        tex_path = output_dir / "country_breakdown.tex"
        tex_cols = [
            "country", "total_km2", "current_pct_protected",
            "30x30_new_km2", "projected_pct_2030_30x30", "gap_to_30pct_km2",
        ]
        try:
            country_df[tex_cols].head(13).to_latex(
                tex_path, index=False, float_format="%.4g",
                caption=(
                    "Country-level breakdown: total land area, current protection (end-2024), "
                    "projected new designations under the 30×30 scenario, "
                    "projected coverage by 2030, and remaining gap to the 30\\% target."
                ),
                label="tab:forward_country",
            )
            print(f"  Saved: {tex_path}")
        except Exception as e:
            # Pandas >=2.0 uses the Styler backend for to_latex, which requires optional deps (jinja2).
            print(f"  NOTE: could not write LaTeX table ({type(e).__name__}: {e})")
        return country_df

    except Exception as e:
        print(f"  WARNING: Country breakdown failed: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()


# ── Biome breakdown ───────────────────────────────────────────────────────────

def create_biome_breakdown(
    df: pd.DataFrame,
    bau_cutoff: float,
    full_cutoff: float,
    gsn_path: Optional[Path],
    output_dir: Path,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("CREATING BIOME BREAKDOWN")
    print("=" * 70)

    if gsn_path is None or not gsn_path.exists():
        print("  Skipping biome breakdown — GSN raster not available")
        return pd.DataFrame()

    try:
        import rasterio

        print(f"  GSN raster: {gsn_path}")
        with rasterio.open(gsn_path) as src:
            gsn_data = src.read(1)

        row_arr = df["row"].values.astype(int)
        col_arr = df["col"].values.astype(int)

        max_row, max_col = gsn_data.shape
        valid = (row_arr >= 0) & (row_arr < max_row) & (col_arr >= 0) & (col_arr < max_col)
        biome_arr = np.full(len(df), -1, dtype=np.int32)
        biome_arr[valid] = gsn_data[row_arr[valid], col_arr[valid]]

        df2 = df.copy()
        df2["biome"]    = biome_arr
        df2["bau_sel"]  = (df2[PROBA_COL] >= bau_cutoff).astype(int)
        df2["full_sel"] = (df2[PROBA_COL] >= full_cutoff).astype(int)

        biome_rows = []
        for biome_id, grp in df2.groupby("biome"):
            if biome_id < 0:
                continue
            biome_rows.append({
                "biome_id": int(biome_id),
                "n_pixels": len(grp),
                "area_km2": round(float(grp["area_km2"].sum()), 0),
                "bau_new_km2": round(float(grp.loc[grp["bau_sel"] == 1, "area_km2"].sum()), 0),
                "30x30_new_km2": round(float(grp.loc[grp["full_sel"] == 1, "area_km2"].sum()), 0),
            })

        biome_df = pd.DataFrame(biome_rows).sort_values("30x30_new_km2", ascending=False)
        csv_path = output_dir / "biome_breakdown.csv"
        biome_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        return biome_df

    except Exception as e:
        print(f"  WARNING: Biome breakdown failed: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()


# ── Gap analysis (Biodiversity Capture Rate) ──────────────────────────────────

def create_gap_analysis(
    df: pd.DataFrame,
    bau_cutoff: float,
    full_cutoff: float,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
) -> Dict[str, Any]:
    """Compute and visualise Biodiversity Capture Rate (BCR).

    BCR = area(top-K ∩ biodiversity mask) / area(all unprotected biodiversity pixels)
    """
    print("\n" + "=" * 70)
    print("GAP ANALYSIS (Biodiversity Capture Rate)")
    print("=" * 70)

    gap_metrics: Dict[str, Any] = {}

    has_gsn = GSN_BIODIVERSITY_COL in df.columns
    if not has_gsn:
        print(f"  NOTE: '{GSN_BIODIVERSITY_COL}' column not found — using placeholder.")

    proba = df[PROBA_COL].values
    area  = df["area_km2"].values
    xm = df["x"].values.astype(np.float64)
    ym = df["y"].values.astype(np.float64)

    bau_mask  = proba >= bau_cutoff
    full_mask = proba >= full_cutoff

    if has_gsn:
        gsn_mask         = df[GSN_BIODIVERSITY_COL].values.astype(bool)
        total_biodiv_km2 = float(area[gsn_mask].sum())

        for key, sel_mask in [("bau", bau_mask), ("30x30_full", full_mask)]:
            overlap_km2 = float(area[sel_mask & gsn_mask].sum())
            bcr = overlap_km2 / max(total_biodiv_km2, 1.0)
            gap_metrics[key] = {
                "overlap_km2": round(overlap_km2, 0),
                "total_biodiv_km2": round(total_biodiv_km2, 0),
                "bcr": round(bcr, 4),
                "random_bcr": round(float(area[sel_mask].sum() / max(area.sum(), 1.0)), 4),
            }
            print(f"  BCR {key}: {bcr:.4f} vs. random {gap_metrics[key]['random_bcr']:.4f}")

    region_proj = None
    proj_bounds = _resolve_plot_bounds_3857(repo_root, DATA_SUBDIR, baseline)
    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

    proj_width = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    pa = proj_height / max(proj_width, 1e-9)
    panel_w = 7.0
    panel_h = panel_w * pa
    fig, axes = plt.subplots(2, 2, figsize=(panel_w * 2, panel_h * 2 + 1.2))
    fig.suptitle("Gap Analysis: BAU Forecast vs. Biodiversity Priority Areas", fontsize=14)

    gsn_mask_for_fig = (
        df[GSN_BIODIVERSITY_COL].values.astype(bool) if has_gsn else np.zeros(len(df), dtype=bool)
    )

    panel_defs = [
        (bau_mask, "BAU Projected Designations\n(2025–2030)",
         SCENARIO_COLORS["bau"]),
        (gsn_mask_for_fig,
         f"Biodiversity Priority ({GSN_BIODIVERSITY_COL}==1)\nHigh-priority unprotected pixels",
         "#2ca02c"),
        (bau_mask & gsn_mask_for_fig,
         "Overlap: BAU ∩ Biodiversity Priority\n(correctly-targeted designations)",
         "#d62728"),
        (~bau_mask & gsn_mask_for_fig,
         "Gap: Biodiversity Priority not in BAU\n(high-value land left unprotected)",
         "#ff7f0e"),
    ]

    for ax, (mask, panel_title, color) in zip(axes.flat, panel_defs):
        used_bb = _plot_backbone_background(
            ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
        )
        _overlay_current_pa_extent(ax, repo_root, DATA_SUBDIR, baseline, proj_bounds, zorder=1)
        if mask.any():
            grid, gext = points_to_raster(
                xm[mask], ym[mask], np.ones(int(mask.sum()), dtype=np.float32),
                target_resolution=1000.0,
                agg_func="max",
                extent_bounds=proj_bounds,
            )
            ax.imshow(
                np.where(np.isnan(grid), np.nan, 1.0),
                extent=gext,
                cmap=mcolors.ListedColormap([color]),
                vmin=0,
                vmax=1,
                origin="upper",
                interpolation="nearest",
                aspect="equal",
                alpha=0.88,
                zorder=2,
            )
        ax.set_xlim(proj_bounds[0], proj_bounds[2])
        ax.set_ylim(proj_bounds[1], proj_bounds[3])
        ax.set_aspect("equal", adjustable="box")
        _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))
        ax.set_title(panel_title, fontsize=10)
        n_px = int(mask.sum())
        km2 = float(area[mask].sum()) if mask.any() else 0.0
        ax.text(
            0.02, 0.02, f"{n_px:,} px / {km2:,.0f} km²",
            transform=ax.transAxes, fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4, zorder=3)

    if has_gsn and gap_metrics:
        bcr_bau  = gap_metrics.get("bau", {}).get("bcr", "N/A")
        bcr_full = gap_metrics.get("30x30_full", {}).get("bcr", "N/A")
        rand_bcr = gap_metrics.get("bau", {}).get("random_bcr", "N/A")
        fig.text(
            0.5, 0.01,
            f"Biodiversity Capture Rate — BAU: {bcr_bau:.3f}  |  "
            f"30x30 Full: {bcr_full:.3f}  |  Random baseline: {rand_bcr:.3f}",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ["png", "pdf"]:
        p = output_dir / f"gap_analysis.{ext}"
        if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass

    return gap_metrics


# ── Economic exposure ──────────────────────────────────────────────────────────

_NTL_CANDIDATES = [
    "ntl", "NTL", "ntl_mean", "VIIRS_ntl", "nighttime_lights", "lights_mean",
    "BU_ntl", "viirs_ntl",
]
_POP_CANDIDATES = [
    "pop_density", "population_density", "pop", "GPW_pop", "population",
    "pop_dens", "GPW",
]


def _detect_col(candidates: List[str], available: set) -> Optional[str]:
    for c in candidates:
        if c in available:
            return c
    avail_lower = {a.lower(): a for a in available}
    for c in candidates:
        if c.lower() in avail_lower:
            return avail_lower[c.lower()]
    return None


def create_economic_exposure(
    df: pd.DataFrame,
    forward_dir: Path,
    bau_cutoff: float,
    full_cutoff: float,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
) -> pd.DataFrame:
    """Quantify economic proxy values in top-1% / top-5% / BAU risk zones.

    Joins forward_features_2024.parquet (model-agnostic) to the scored
    parquet on (row, col) to retrieve NTL and population density columns,
    then computes summary statistics across three zones:
      - All unprotected 2024 pixels (baseline)
      - Top 5% by predicted probability
      - Top 1% by predicted probability

    Outputs:
      forward_economic_exposure.csv
      forward_economic_exposure_map.pdf/.png  (if NTL column found)
    """
    print("\n" + "=" * 70)
    print("ECONOMIC EXPOSURE ANALYSIS")
    print("=" * 70)

    feat_path = forward_dir / "forward_features_2024.parquet"
    if not feat_path.exists():
        print(f"  Skipping — features parquet not found: {feat_path}")
        return pd.DataFrame()

    # ── Detect economic proxy columns ────────────────────────────────────────
    schema_names = set(pq.ParquetFile(feat_path).schema_arrow.names)
    ntl_col = _detect_col(_NTL_CANDIDATES, schema_names)
    pop_col = _detect_col(_POP_CANDIDATES, schema_names)
    print(f"  NTL column detected:        {ntl_col or 'none'}")
    print(f"  Population column detected: {pop_col or 'none'}")

    proxy_cols = [c for c in [ntl_col, pop_col] if c is not None]
    if not proxy_cols:
        print("  No economic proxy columns found — skipping.")
        return pd.DataFrame()

    # ── Load and join features ────────────────────────────────────────────────
    print(f"  Loading features: {proxy_cols} …")
    feat_df = pq.read_table(feat_path, columns=["row", "col"] + proxy_cols).to_pandas()
    coord_cols = [c for c in ["x", "y"] if c in df.columns]
    merged  = df[["row", "col", PROBA_COL, "area_km2"] + coord_cols].merge(
        feat_df, on=["row", "col"], how="left"
    )
    del feat_df

    # ── Define zones ─────────────────────────────────────────────────────────
    top5_mask = merged[PROBA_COL] >= float(np.percentile(merged[PROBA_COL], 95))
    top1_mask = merged[PROBA_COL] >= float(np.percentile(merged[PROBA_COL], 99))
    bau_mask  = merged[PROBA_COL] >= bau_cutoff

    zones = [
        ("All unprotected (baseline)", np.ones(len(merged), dtype=bool)),
        ("BAU risk zone",              bau_mask),
        ("Top 5% risk",               top5_mask),
        ("Top 1% risk",               top1_mask),
    ]

    rows = []
    for zone_name, mask in zones:
        sub = merged[mask]
        row: Dict[str, Any] = {
            "zone": zone_name,
            "n_pixels": int(mask.sum()),
            "area_km2": round(float(sub["area_km2"].sum()), 0),
        }
        for col in proxy_cols:
            vals = sub[col].dropna()
            row[f"{col}_mean"]   = round(float(vals.mean()), 3)   if len(vals) else None
            row[f"{col}_median"] = round(float(vals.median()), 3) if len(vals) else None
            row[f"{col}_p75"]    = round(float(np.percentile(vals, 75)), 3) if len(vals) else None
        rows.append(row)

    exposure_df = pd.DataFrame(rows)
    csv_path = output_dir / "economic_exposure.csv"
    exposure_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(exposure_df.to_string(index=False))

    # ── Map: BAU risk zone coloured by NTL (EPSG:3857, backbone underlay) ───
    if ntl_col is not None:
        bau_sub = merged[bau_mask].dropna(subset=[ntl_col])
        if len(bau_sub) > 0:
            xm = bau_sub["x"].values.astype(np.float64)
            ym = bau_sub["y"].values.astype(np.float64)
            ntl_vals = bau_sub[ntl_col].values.astype(np.float32)

            region_proj = None
            proj_bounds = _resolve_plot_bounds_3857(repo_root, DATA_SUBDIR, baseline)
            backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

            grid, gext = points_to_raster(
                xm, ym, ntl_vals,
                target_resolution=1000.0,
                agg_func="mean",
                extent_bounds=proj_bounds,
            )

            proj_width = proj_bounds[2] - proj_bounds[0]
            proj_height = proj_bounds[3] - proj_bounds[1]
            fig_w = 14.0
            fig_h = fig_w * (proj_height / max(proj_width, 1e-9))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            used_bb = _plot_backbone_background(
                ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
            )
            _overlay_current_pa_extent(ax, repo_root, DATA_SUBDIR, baseline, proj_bounds, zorder=1)

            im = ax.imshow(
                grid,
                extent=gext,
                cmap="plasma",
                origin="upper",
                interpolation="nearest",
                aspect="equal",
                zorder=2,
            )
            ax.set_xlim(proj_bounds[0], proj_bounds[2])
            ax.set_ylim(proj_bounds[1], proj_bounds[3])
            ax.set_aspect("equal", adjustable="box")
            _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))
            ax.set_title(
                "Economic Exposure in BAU Risk Zone\n"
                f"BAU-designated pixels coloured by nighttime light intensity ({ntl_col})",
                fontsize=11,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"{ntl_col} (mean per ~1 km cell)", fontsize=9)
            plt.tight_layout(pad=0.5)
            for ext in ["png", "pdf"]:
                p = output_dir / f"economic_exposure_map.{ext}"
                if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
                    print(f"  Saved: {p}")
            try:
                plt.close(fig)
            except Exception:
                pass

    return exposure_df


# ── Hotspot zoom-in maps ───────────────────────────────────────────────────────

def create_hotspot_maps(
    df: pd.DataFrame,
    hotspot_regions: List[Tuple],
    bau_cutoff: float,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
    region_label: str,
) -> None:
    """Zoomed probability + BAU panels per hotspot (Web Mercator, backbone underlay).

    hotspot_regions: list of (label, lon_min, lon_max, lat_min, lat_max) in WGS84.
    Output: hotspot_maps.pdf/.png
    """
    print("\n" + "=" * 70)
    print("HOTSPOT ZOOM-IN MAPS")
    print("=" * 70)

    if not hotspot_regions:
        print("  No hotspot regions defined — skipping.")
        return

    xm_all = df["x"].values.astype(np.float64)
    ym_all = df["y"].values.astype(np.float64)
    lon_all, lat_all = epsg3857_to_lonlat(xm_all, ym_all)
    proba = df[PROBA_COL].values.astype(np.float64)
    norm_p = _forward_norm_params(proba)
    bau_mask = proba >= bau_cutoff

    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

    n = len(hotspot_regions)
    # Figure width from first valid hotspot aspect (fallback 4:3)
    panel_w, panel_h = 5.0, 4.0
    for _lbl, x0, x1, y0, y1 in hotspot_regions:
        zb = _lonlat_bounds_to_proj_bounds(x0, x1, y0, y1)
        pw = zb[2] - zb[0]
        ph = zb[3] - zb[1]
        if pw > 1e-6:
            panel_w, panel_h = 5.0, 5.0 * (ph / pw)
            break

    fig, axes = plt.subplots(2, n, figsize=(panel_w * n, panel_h * 2 + 0.8))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Hotspot Zoom-ins — {region_label}\n"
        "Top row: P(protection by 2030) | Bottom row: BAU designation zone",
        fontsize=12,
    )

    for col_idx, (label, x0, x1, y0, y1) in enumerate(hotspot_regions):
        zoom_bounds = _lonlat_bounds_to_proj_bounds(x0, x1, y0, y1)
        in_box = (lon_all >= x0) & (lon_all <= x1) & (lat_all >= y0) & (lat_all <= y1)
        if not in_box.any():
            print(f"  WARNING: no pixels in hotspot '{label}' — check coordinates")
            for row_idx in range(2):
                axes[row_idx, col_idx].set_visible(False)
            continue

        print(f"  {label}: {in_box.sum():,} pixels")
        xm_h = xm_all[in_box]
        ym_h = ym_all[in_box]
        proba_norm = _forward_apply_norm(proba[in_box], norm_p)

        # Row 0: continuous probability (same normalisation as continental map)
        ax0 = axes[0, col_idx]
        used_bb0 = _plot_backbone_background(
            ax0, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
        )
        _overlay_current_pa_extent(ax0, repo_root, DATA_SUBDIR, baseline, zoom_bounds, zorder=1)
        grid_p, gext_p = points_to_raster(
            xm_h, ym_h, proba_norm,
            target_resolution=1000.0,
            agg_func="mean",
            extent_bounds=zoom_bounds,
        )
        im = ax0.imshow(
            grid_p,
            extent=gext_p,
            cmap=PROBABILITY_MAP_COLORMAP,
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
            zorder=2,
        )
        ax0.set_xlim(zoom_bounds[0], zoom_bounds[2])
        ax0.set_ylim(zoom_bounds[1], zoom_bounds[3])
        ax0.set_aspect("equal", adjustable="box")
        _add_latlon_ticks(
            ax0, (zoom_bounds[0], zoom_bounds[2]), (zoom_bounds[1], zoom_bounds[3]),
        )
        ax0.set_title(label, fontsize=10)
        if col_idx == 0:
            ax0.set_ylabel(
                "P(protection by 2030)\n(same stretch as continental map)",
                fontsize=8,
            )
        ax0.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)
        plt.colorbar(im, ax=ax0, fraction=0.04, pad=0.03)

        # Row 1: BAU designation zone
        ax1 = axes[1, col_idx]
        used_bb1 = _plot_backbone_background(
            ax1, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
        )
        _overlay_current_pa_extent(ax1, repo_root, DATA_SUBDIR, baseline, zoom_bounds, zorder=1)
        bau_in = bau_mask & in_box
        n_bau = int(bau_in.sum())
        if bau_in.any():
            grid_b, gext_b = points_to_raster(
                xm_all[bau_in], ym_all[bau_in],
                np.ones(int(bau_in.sum()), dtype=np.float32),
                target_resolution=1000.0,
                agg_func="max",
                extent_bounds=zoom_bounds,
            )
            ax1.imshow(
                np.where(np.isnan(grid_b), np.nan, 1.0),
                extent=gext_b,
                cmap=mcolors.ListedColormap([SCENARIO_COLORS["bau"]]),
                vmin=0,
                vmax=1,
                origin="upper",
                interpolation="nearest",
                aspect="equal",
                alpha=0.88,
                zorder=2,
            )
        ax1.set_xlim(zoom_bounds[0], zoom_bounds[2])
        ax1.set_ylim(zoom_bounds[1], zoom_bounds[3])
        ax1.set_aspect("equal", adjustable="box")
        _add_latlon_ticks(
            ax1, (zoom_bounds[0], zoom_bounds[2]), (zoom_bounds[1], zoom_bounds[3]),
        )
        ax1.text(
            0.02, 0.02, f"{n_bau:,} BAU pixels",
            transform=ax1.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )
        if col_idx == 0:
            ax1.set_ylabel("BAU designation zone", fontsize=8)
        ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = output_dir / f"hotspot_maps.{ext}"
        if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass


# ── Forecast confidence ───────────────────────────────────────────────────────

def create_forecast_confidence_summary(model_output_dir: Path) -> Dict[str, Any]:
    """Load backtest results and compute confidence bounds for the 2030 forecast.

    Uses historical backtesting windows (e.g. T2013, T2015, T2017) to estimate
    expected precision and recall for the actual 2024→2030 deployment window.
    T2019 is excluded because it has no evaluable pixels (WDPA only covers to 2024).
    """
    print("\n" + "=" * 70)
    print("FORECAST CONFIDENCE SUMMARY (from backtesting)")
    print("=" * 70)

    backtest_path = model_output_dir / "forward_backtest_results.json"
    if not backtest_path.exists():
        # Search one level up (in case model_output_dir is model-type subdir)
        backtest_path = model_output_dir.parent / "forward_backtest_results.json"
    if not backtest_path.exists():
        print("  WARNING: forward_backtest_results.json not found — skipping confidence summary")
        return {}

    with open(backtest_path) as f:
        backtest_results = json.load(f)

    # Only use windows that have evaluable pixels and non-null metrics
    valid = [r for r in backtest_results if r.get("metrics") and r["n_pos_evaluable"] > 0]
    if not valid:
        print("  WARNING: No valid backtest windows found — skipping confidence summary")
        return {}

    p1  = [r["metrics"]["precision_at_1pct"]  for r in valid]
    p5  = [r["metrics"]["precision_at_5pct"]  for r in valid]
    r1  = [r["metrics"]["recall_at_1pct"]     for r in valid]
    r5  = [r["metrics"]["recall_at_5pct"]     for r in valid]
    roc = [r["metrics"]["roc_auc"]            for r in valid]
    years = [r["origin_year"] for r in valid]

    # Use the window most analogous to the actual deployment (latest clean window)
    # T2013 uses a complete 5-year lookahead and is the most conservative estimate
    clean_windows = [r for r in valid if r.get("clean_5yr_window")]
    primary = clean_windows[-1] if clean_windows else valid[-1]

    confidence = {
        "backtest_windows": years,
        "n_valid_windows": len(valid),
        "primary_window": {
            "origin_year": primary["origin_year"],
            "note": "most conservative estimate (complete 5-yr lookahead)",
            "n_pos_evaluable": primary["n_pos_evaluable"],
            "precision_at_1pct": round(primary["metrics"]["precision_at_1pct"], 4),
            "precision_at_5pct": round(primary["metrics"]["precision_at_5pct"], 4),
            "recall_at_1pct":    round(primary["metrics"]["recall_at_1pct"],    4),
            "recall_at_5pct":    round(primary["metrics"]["recall_at_5pct"],    4),
            "roc_auc":           round(primary["metrics"]["roc_auc"],           4),
        },
        "range_across_windows": {
            "precision_at_1pct_min": round(min(p1), 4),
            "precision_at_1pct_max": round(max(p1), 4),
            "precision_at_5pct_min": round(min(p5), 4),
            "precision_at_5pct_max": round(max(p5), 4),
            "recall_at_1pct_min":    round(min(r1), 4),
            "recall_at_1pct_max":    round(max(r1), 4),
            "recall_at_5pct_min":    round(min(r5), 4),
            "recall_at_5pct_max":    round(max(r5), 4),
            "roc_auc_min":           round(min(roc), 4),
            "roc_auc_max":           round(max(roc), 4),
        },
        "interpretation": (
            f"In the most conservative backtest window (origin {primary['origin_year']}), "
            f"the model achieved Precision@1% = {primary['metrics']['precision_at_1pct']:.1%} "
            f"and Recall@1% = {primary['metrics']['recall_at_1pct']:.1%}. "
            f"Across all {len(valid)} valid windows, Precision@1% ranged "
            f"{min(p1):.1%}–{max(p1):.1%} and Recall@1% ranged {min(r1):.1%}–{max(r1):.1%}."
        ),
    }

    print(f"  Valid windows: {years}")
    print(f"  Primary (conservative) window: {primary['origin_year']}")
    print(f"  Precision@1% range: {min(p1):.1%}–{max(p1):.1%}")
    print(f"  Recall@1%    range: {min(r1):.1%}–{max(r1):.1%}")
    print(f"  Precision@5% range: {min(p5):.1%}–{max(p5):.1%}")
    print(f"  Recall@5%    range: {min(r5):.1%}–{max(r5):.1%}")
    return confidence


# ── Conservation alignment ────────────────────────────────────────────────────

def create_conservation_alignment_map(
    df: pd.DataFrame,
    bau_cutoff: float,
    full_cutoff: float,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
) -> None:
    """Create a single-map 'Conservation Alignment' visualisation (biodiversity)."""
    _create_alignment_map(
        df=df,
        bau_cutoff=bau_cutoff,
        full_cutoff=full_cutoff,
        output_dir=output_dir,
        baseline=baseline,
        repo_root=repo_root,
        priority_col=GSN_BIODIVERSITY_COL,
        title="Conservation Alignment: Where 30×30 Targets Biodiversity",
        annotation_subject="biodiversity priority",
        out_stem="conservation_alignment",
        console_label="CONSERVATION ALIGNMENT MAP (sweet spots vs gaps)",
    )


def create_climate_alignment_map(
    df: pd.DataFrame,
    bau_cutoff: float,
    full_cutoff: float,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
) -> None:
    """Create a single-map 'Climate Alignment' visualisation (climate stabilisation areas)."""
    _create_alignment_map(
        df=df,
        bau_cutoff=bau_cutoff,
        full_cutoff=full_cutoff,
        output_dir=output_dir,
        baseline=baseline,
        repo_root=repo_root,
        priority_col=GSN_CLIMATE_STABILISATION_COL,
        title="Climate Alignment: Where 30×30 Targets Climate Stabilisation",
        annotation_subject="climate stabilisation priority",
        out_stem="climate_stabilisation_alignment",
        console_label="CLIMATE STABILISATION ALIGNMENT MAP (sweet spots vs gaps)",
    )


def _create_alignment_map(
    df: pd.DataFrame,
    bau_cutoff: float,
    full_cutoff: float,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
    priority_col: str,
    title: str,
    annotation_subject: str,
    out_stem: str,
    console_label: str,
) -> None:
    """Create a single-map 'Alignment' visualisation for any binary priority mask."""
    print("\n" + "=" * 70)
    print(console_label)
    print("=" * 70)

    if priority_col not in df.columns:
        print(f"  NOTE: '{priority_col}' not available — skipping alignment map")
        return

    proba   = df[PROBA_COL].values
    area    = df["area_km2"].values
    xm      = df["x"].values.astype(np.float64)
    ym      = df["y"].values.astype(np.float64)
    priority = df[priority_col].values.astype(bool)
    bau     = proba >= bau_cutoff
    full    = proba >= full_cutoff

    # Alignment categories (30x30 scenario for maximum impact)
    sweet_spot   =  full & priority          # designated AND priority
    gap          = ~full & priority          # priority NOT designated
    misdirected  =  full & ~priority         # designated but NOT priority

    total_sweet_km2       = float(area[sweet_spot].sum())
    total_gap_km2         = float(area[gap].sum())
    total_misdirected_km2 = float(area[misdirected].sum())
    total_priority_km2    = float(area[priority].sum())
    total_designated_km2  = float(area[full].sum())
    # We want: "How focused are new 30×30 designations on the priority layer?"
    # i.e., % of the 30×30 establishment area that falls inside the priority mask.
    alignment_pct = total_sweet_km2 / max(total_designated_km2, 1.0) * 100

    print(f"  Sweet spots (both):    {total_sweet_km2:,.0f} km²")
    print(f"  Conservation gaps:     {total_gap_km2:,.0f} km²")
    print(f"  Misdirected:           {total_misdirected_km2:,.0f} km²")
    print(f"  Total designated (30×30): {total_designated_km2:,.0f} km²")
    print(f"  Total priority area:      {total_priority_km2:,.0f} km²")
    print(f"  Alignment score:          {alignment_pct:.1f}% of 30×30 establishments inside {annotation_subject}")

    region_proj = None
    proj_bounds = _resolve_plot_bounds_3857(repo_root, DATA_SUBDIR, baseline)
    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

    proj_width  = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    fig_w, fig_h = 9.0, 9.0 * (proj_height / max(proj_width, 1e-9))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    used_bb = _plot_backbone_background(
        ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR,
    )
    _overlay_current_pa_extent(ax, repo_root, DATA_SUBDIR, baseline, proj_bounds, zorder=1)

    def _overlay(mask: np.ndarray, color: str, alpha: float, zorder: int) -> None:
        if not mask.any():
            return
        grid, gext = points_to_raster(
            xm[mask], ym[mask], np.ones(int(mask.sum()), dtype=np.float32),
            target_resolution=1000.0, agg_func="max", extent_bounds=proj_bounds,
        )
        ax.imshow(
            np.where(np.isnan(grid), np.nan, 1.0),
            extent=gext, cmap=mcolors.ListedColormap([color]),
            vmin=0, vmax=1, origin="upper", interpolation="nearest",
            aspect="equal", alpha=alpha, zorder=zorder,
        )

    # Draw in order: gaps first, misdirected second, sweet spots on top
    _overlay(gap,         "#e87e2b", 0.80, 2)   # orange  — conservation gap
    _overlay(misdirected, "#4c96d7", 0.65, 3)   # blue    — misdirected designation
    _overlay(sweet_spot,  "#2ca02c", 0.90, 4)   # green   — sweet spot

    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    legend_elements = [
        mpatches.Patch(color="#2ca02c", label=f"Sweet spot — designated & priority  ({total_sweet_km2:,.0f} km²)"),
        mpatches.Patch(color="#e87e2b", label=f"Conservation gap — priority, not designated  ({total_gap_km2:,.0f} km²)"),
        mpatches.Patch(color="#4c96d7", label=f"Misdirected — designated, not priority  ({total_misdirected_km2:,.0f} km²)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              framealpha=0.85, edgecolor="gray")

    ax.text(
        0.98, 0.02,
        f"Alignment score: {alignment_pct:.1f}%\nof 30×30 establishments\ninside {annotation_subject}",
        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.85, ec="gray"),
    )
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4, zorder=5)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = output_dir / f"{out_stem}.{ext}"
        if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass


def create_country_conservation_alignment(
    df: pd.DataFrame,
    full_cutoff: float,
    output_dir: Path,
    world_gdf,
    iso_codes: List[str],
) -> pd.DataFrame:
    """Bar chart: % of 30x30 new designations that overlap biodiversity priority, by country.

    This answers: 'Which countries are targeting biodiversity when they designate?'
    Higher = better conservation efficiency.
    """
    print("\n" + "=" * 70)
    print("COUNTRY CONSERVATION ALIGNMENT SCORES")
    print("=" * 70)

    has_gsn = GSN_BIODIVERSITY_COL in df.columns
    if not has_gsn:
        print(f"  NOTE: '{GSN_BIODIVERSITY_COL}' not available — skipping country alignment")
        return pd.DataFrame()

    proba = df[PROBA_COL].values
    area  = df["area_km2"].values
    gsn   = df[GSN_BIODIVERSITY_COL].values.astype(bool)
    full  = proba >= full_cutoff

    # Prefer raster-based ISO3 assignment (policy preprocessing output).
    iso_grid, iso_map = _load_country_iso3_lookup(repo_root=get_repo_root(), data_subdir=DATA_SUBDIR)
    use_raster = (
        iso_grid is not None
        and iso_map is not None
        and ("row" in df.columns)
        and ("col" in df.columns)
    )
    if use_raster:
        row_arr = df["row"].values.astype(np.int64)
        col_arr = df["col"].values.astype(np.int64)
        h, w = iso_grid.shape
        valid = (row_arr >= 0) & (row_arr < h) & (col_arr >= 0) & (col_arr < w)
        ids = np.full(len(df), -1, dtype=np.int32)
        ids[valid] = iso_grid[row_arr[valid], col_arr[valid]].astype(np.int32)
        iso3 = np.array([iso_map.get(int(i), None) for i in ids], dtype=object)
        joined = pd.DataFrame({"iso": iso3, "area_km2": area, "is_full": full, "is_gsn": gsn})
        joined = joined[joined["iso"].isin(iso_codes)].copy()
        if joined.empty:
            print("  WARNING: no pixels matched target ISO codes (raster lookup) — skipping")
            return pd.DataFrame()
        name_lookup = {c.get("iso_a3"): c.get("country") for c in baseline.get("country_stats", [])}
    else:
        if world_gdf is None:
            print("  NOTE: No country_iso3.tif and no country shapefile — skipping country alignment")
            return pd.DataFrame()
        # Spatial join fallback: assign country via lon/lat + vector boundaries.
        try:
            lon, lat = epsg3857_to_lonlat(df["x"].values, df["y"].values)
        except Exception as e:
            print(f"  WARNING: coordinate conversion failed ({e}) — skipping")
            return pd.DataFrame()
        try:
            import geopandas as gpd

            pts = gpd.GeoDataFrame(
                {"area_km2": area, "is_full": full, "is_gsn": gsn},
                geometry=gpd.points_from_xy(lon, lat),
                crs="EPSG:4326",
            )

            world_4326 = world_gdf.to_crs("EPSG:4326") if world_gdf.crs is not None else world_gdf
            iso_col = next(
                (c for c in ["ISO_A3", "ADM0_A3", "iso_a3", "GID_0"] if c in world_4326.columns),
                world_4326.columns[0],
            )
            name_col = next(
                (c for c in ["NAME", "ADMIN", "name", "NAME_LONG"] if c in world_4326.columns),
                iso_col,
            )
            joined = gpd.sjoin(pts, world_4326[[iso_col, name_col, "geometry"]],
                               how="left", predicate="within")
        except Exception as e:
            print(f"  WARNING: spatial join failed ({e}) — skipping")
            return pd.DataFrame()
        joined = joined[joined[iso_col].isin(iso_codes)].copy()
        if joined.empty:
            print("  WARNING: no pixels matched target ISO codes — skipping")
            return pd.DataFrame()

    records = []
    if use_raster:
        for iso, grp in joined.groupby("iso"):
            name = name_lookup.get(iso) or iso
            new_km2  = float(grp.loc[grp["is_full"],  "area_km2"].sum())
            sweet_km2= float(grp.loc[grp["is_full"] & grp["is_gsn"], "area_km2"].sum())
            bio_km2  = float(grp.loc[grp["is_gsn"],   "area_km2"].sum())
            align    = sweet_km2 / max(new_km2, 1.0) * 100
            records.append({
                "iso": iso, "country": name,
                "new_designated_km2": round(new_km2, 0),
                "sweet_spot_km2": round(sweet_km2, 0),
                "biodiv_priority_km2": round(bio_km2, 0),
                "alignment_pct": round(align, 1),
            })
    else:
        for iso, grp in joined.groupby(iso_col):
            name     = grp[name_col].iloc[0] if name_col != iso_col else iso
            new_km2  = float(grp.loc[grp["is_full"],  "area_km2"].sum())
            sweet_km2= float(grp.loc[grp["is_full"] & grp["is_gsn"], "area_km2"].sum())
            bio_km2  = float(grp.loc[grp["is_gsn"],   "area_km2"].sum())
            align    = sweet_km2 / max(new_km2, 1.0) * 100  # % of new designations that are biodiversity priority
            records.append({
                "iso": iso, "country": name,
                "new_designated_km2": round(new_km2, 0),
                "sweet_spot_km2": round(sweet_km2, 0),
                "biodiv_priority_km2": round(bio_km2, 0),
                "alignment_pct": round(align, 1),
            })

    result_df = pd.DataFrame(records).sort_values("alignment_pct", ascending=False)
    if result_df.empty:
        return result_df

    # Save CSV
    csv_path = output_dir / "country_alignment.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(result_df[["country", "new_designated_km2", "sweet_spot_km2", "alignment_pct"]].to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, max(4, len(result_df) * 0.55)))
    bars = ax.barh(result_df["country"], result_df["alignment_pct"],
                   color=["#2ca02c" if v >= 40 else "#e87e2b" if v >= 20 else "#d62728"
                          for v in result_df["alignment_pct"]],
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("% of new 30×30 designations targeting biodiversity priority land", fontsize=10)
    ax.set_title("Conservation Alignment by Country (30×30 Scenario)", fontsize=11, fontweight="bold")
    ax.axvline(x=result_df["alignment_pct"].mean(), color="gray", linestyle="--", linewidth=1.2,
               label=f"Mean: {result_df['alignment_pct'].mean():.1f}%")
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(result_df["alignment_pct"].max() * 1.15, 10))
    for bar, val in zip(bars, result_df["alignment_pct"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", ha="left", fontsize=8)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = output_dir / f"country_alignment.{ext}"
        if _safe_savefig(p, dpi=MAP_DPI, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass

    return result_df


# ── Summary JSON ──────────────────────────────────────────────────────────────

def save_scenario_summary(
    baseline: Dict,
    scenario_info: Dict,
    gap_metrics: Dict,
    country_df: pd.DataFrame,
    biome_df: pd.DataFrame,
    output_dir: Path,
    forecast_confidence: Optional[Dict] = None,
    country_alignment_df: Optional[pd.DataFrame] = None,
) -> None:
    summary = {
        "coverage_baseline": baseline,
        "scenarios": scenario_info,
        "gap_analysis": gap_metrics,
        "country_breakdown_top10": (
            country_df.head(10).to_dict("records") if not country_df.empty else []
        ),
        "biome_breakdown_top10": (
            biome_df.head(10).to_dict("records") if not biome_df.empty else []
        ),
    }
    if forecast_confidence:
        summary["forecast_confidence"] = forecast_confidence
    if country_alignment_df is not None and not country_alignment_df.empty:
        summary["country_alignment"] = country_alignment_df.to_dict("records")
    out = output_dir / "scenario_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved scenario summary: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Re-import config after runner.py reload
    from scripts.regions.shared.forward.config import (  # noqa: F401
        DATA_SUBDIR, HOTSPOT_REGIONS, ISO_CODES, MODEL_PREFIX, OUTPUTS_SUBDIR,
        REGION_LABEL, forward_dir_search_paths,
    )

    model_type = os.environ.get("PA3030_FORWARD_MODEL_TYPE", "lgbm").strip().lower()
    from datetime import datetime as _dt

    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    # Keep forward Stage 3 (results) runs separate from deployment/inference runs in W&B.
    # Override per-run if desired, e.g. PA3030_WANDB_PROJECT_RESULTS="forward_results_usa".
    wandb_project_results = os.environ.get("PA3030_WANDB_PROJECT_RESULTS", "forward_results").strip()
    wb = WandbRunLogger(
        project=wandb_project_results,
        run_name=f"results_{OUTPUTS_SUBDIR}_{model_type}_{_ts}",
        config={
            "region": OUTPUTS_SUBDIR,
            "model_type": model_type,
            "forward_stage": "results",
        },
    )
    wb.start()
    wb.log({"results/stage": "start"})

    repo_root     = get_repo_root()
    forward_dirs  = forward_dir_search_paths(repo_root, OUTPUTS_SUBDIR)
    forward_dir   = forward_dirs[0]
    features_forward_dir = forward_dir
    for _fd in forward_dirs:
        if (_fd / "forward_features_2024.parquet").exists():
            features_forward_dir = _fd
            break
    model_output_dir = forward_dir / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FORWARD RESULTS — MAPS, SCENARIOS, BREAKDOWNS, GAP ANALYSIS ({model_type.upper()})")
    print("=" * 70)
    print(f"  Forward output root: {forward_dir}")

    # ── Load inputs (search scratch + legacy repo paths) ──────────────────────
    scored_path  = resolve_scored_parquet(forward_dirs, model_type)
    baseline     = resolve_coverage_baseline(forward_dirs)
    pixel_size_m = resolve_backbone_pixel_size(repo_root, DATA_SUBDIR)
    # Fall back to value stored in baseline (from coverage_baseline run)
    if pixel_size_m == 1000.0 and "pixel_size_m" in baseline:
        pixel_size_m = float(baseline["pixel_size_m"])
        print(f"  Using pixel_size_m from coverage baseline: {pixel_size_m:.2f} m")

    gsn_path  = resolve_gsn_raster(repo_root, DATA_SUBDIR)
    world_gdf = resolve_country_shapefile()

    df = load_scored(scored_path)
    df = add_area_col(df, pixel_size_m)

    # ── Cross-model agreement (LGBM vs RF) ────────────────────────────────────
    # This is useful even when running only one model_type; it will skip cleanly
    # unless both forward_scored_2024.parquet files are present.
    create_model_agreement_map_top1pct(
        forward_dirs=forward_dirs,
        output_dir=forward_dir,
        baseline=baseline,
        repo_root=repo_root,
        region_label=REGION_LABEL,
        top_pct=0.01,
    )

    # Join GSN mask bands from features parquet (not present in scored output)
    if "GSN_b1" not in df.columns or "GSN_b2" not in df.columns:
        feat_path = None
        for _fd in forward_dirs:
            _p = _fd / "forward_features_2024.parquet"
            if _p.exists():
                feat_path = _p
                break
        if feat_path is not None:
            try:
                wanted = ["row", "col", "GSN_b1", "GSN_b2"]
                try:
                    feat_gsn = pd.read_parquet(feat_path, columns=wanted)
                except Exception:
                    # Fall back gracefully for older feature files that may not include all bands
                    feat_gsn = pd.read_parquet(feat_path, columns=["row", "col", "GSN_b1"])
                df = df.merge(feat_gsn, on=["row", "col"], how="left")
                for c in ["GSN_b1", "GSN_b2"]:
                    if c in feat_gsn.columns:
                        nn = int(df[c].notna().sum())
                        print(f"  Joined {c} from features parquet ({feat_path.name}): {nn:,} non-null values")
            except Exception as _e:
                print(f"  WARNING: Could not join GSN bands from features parquet: {_e}")
        else:
            print(
                "  NOTE: forward_features_2024.parquet not found under forward output dirs — "
                "gap analysis will use placeholder",
            )

    total_km2 = baseline.get("total_land_km2") or baseline.get("total_sa_km2", 1.0)
    print(f"\n  Total unprotected 2024 pixels: {len(df):,}")
    print(f"  Total unprotected 2024 area:   {df['area_km2'].sum():,.0f} km²")
    print(f"  Baseline total land km²:        {total_km2:,.0f}")
    print(f"  Baseline protected 2024 km²:   {baseline['protected_2024_km2']:,.0f}")
    print(f"  Baseline coverage 2024:        {baseline['coverage_pct_2024']:.2%}")

    # ── Scenario cutoffs ──────────────────────────────────────────────────────
    # Use adaptive moderate target (midpoint current→30%) from baseline.
    # Fall back to the legacy 25% key if re-running against an old baseline JSON.
    km2_moderate = baseline.get("km2_needed_for_moderate") or baseline.get("km2_needed_for_25pct", 0.0)
    moderate_pct = baseline.get("moderate_target_pct", 0.25)
    bau_cutoff, moderate_cutoff, full_cutoff, bau_subtitle = compute_scenario_cutoffs(
        df,
        km2_moderate,
        moderate_pct,
        baseline["km2_needed_for_30pct"],
        bau_km2=baseline.get("bau_km2"),
    )

    # ── Maps ──────────────────────────────────────────────────────────────────
    create_probability_map(
        df, pixel_size_m, model_output_dir, baseline, repo_root, REGION_LABEL,
    )

    scenario_info = create_scenario_maps(
        df, baseline, bau_cutoff, moderate_cutoff, full_cutoff,
        model_output_dir, repo_root, REGION_LABEL,
        bau_subtitle=bau_subtitle,
    )
    wb.log({"results/stage": "scenario_maps_done"})

    # ── Feature importance (deployment artifact) ─────────────────────────────
    create_feature_importance_plot(
        repo_root=repo_root,
        data_subdir=DATA_SUBDIR,
        outputs_subdir=OUTPUTS_SUBDIR,
        model_prefix=MODEL_PREFIX,
        model_type=model_type,
        output_dir=model_output_dir,
    )

    # ── Breakdowns ────────────────────────────────────────────────────────────
    country_df = create_country_breakdown(
        df, baseline, bau_cutoff, full_cutoff,
        model_output_dir, world_gdf, ISO_CODES,
    )
    biome_df = create_biome_breakdown(
        df, bau_cutoff, full_cutoff, gsn_path, model_output_dir
    )
    wb.log({"results/stage": "breakdowns_done"})

    # ── Gap analysis ──────────────────────────────────────────────────────────
    gap_metrics = create_gap_analysis(
        df, bau_cutoff, full_cutoff, model_output_dir, baseline, repo_root,
    )
    wb.log({"results/stage": "gap_analysis_done"})

    # ── Economic exposure ─────────────────────────────────────────────────────
    create_economic_exposure(
        df, features_forward_dir, bau_cutoff, full_cutoff,
        model_output_dir, baseline, repo_root,
    )
    wb.log({"results/stage": "economic_exposure_done"})

    # ── Hotspot zoom-ins ──────────────────────────────────────────────────────
    create_hotspot_maps(
        df, HOTSPOT_REGIONS, bau_cutoff,
        model_output_dir, baseline, repo_root, REGION_LABEL,
    )
    wb.log({"results/stage": "hotspot_maps_done"})

    # ── Forecast confidence (from backtesting) ────────────────────────────────
    forecast_confidence = create_forecast_confidence_summary(model_output_dir)

    # ── Conservation / climate alignment maps & country scores ───────────────
    create_conservation_alignment_map(
        df, bau_cutoff, full_cutoff, model_output_dir, baseline, repo_root,
    )
    create_climate_alignment_map(
        df, bau_cutoff, full_cutoff, model_output_dir, baseline, repo_root,
    )
    country_alignment_df = create_country_conservation_alignment(
        df, full_cutoff, model_output_dir, world_gdf, ISO_CODES,
    )

    # ── Summary JSON ──────────────────────────────────────────────────────────
    save_scenario_summary(
        baseline, scenario_info, gap_metrics, country_df, biome_df, model_output_dir,
        forecast_confidence=forecast_confidence,
        country_alignment_df=country_alignment_df,
    )

    _log: Dict[str, Any] = {
        "results/n_unprotected_pixels":    len(df),
        "results/coverage_pct_2024":       baseline.get("coverage_pct_2024"),
        "results/protected_2024_km2":      baseline.get("protected_2024_km2"),
        "results/km2_needed_for_30pct":    baseline.get("km2_needed_for_30pct"),
        "results/bau_cutoff":              float(bau_cutoff),
        "results/moderate_cutoff":         float(moderate_cutoff),
        "results/full_cutoff":             float(full_cutoff),
    }
    if isinstance(scenario_info, dict):
        for _sc, _sv in scenario_info.items():
            if isinstance(_sv, dict):
                for _k, _v in _sv.items():
                    if isinstance(_v, (int, float)):
                        _log[f"results/scenarios/{_sc}/{_k}"] = _v
    if isinstance(gap_metrics, dict):
        for _k, _v in gap_metrics.items():
            if isinstance(_v, (int, float)):
                _log[f"results/gap/{_k}"] = _v
    _log["results/stage"] = "done"
    wb.log(_log)
    wb.finish()

    print("\n" + "=" * 70)
    print("FORWARD RESULTS COMPLETE")
    print("=" * 70)
    outputs = [
        "probability_map.png",
        "risk_map_bau.png",
        "scenario_moderate.png",
        "scenario_30x30.png",
        "gap_analysis.png",
        "country_breakdown.csv",
        "biome_breakdown.csv",
        "economic_exposure.csv",
        "hotspot_maps.png",
        "scenario_summary.json",
        "conservation_alignment.png",
        "climate_stabilisation_alignment.png",
        "country_alignment.png",
        "country_alignment.csv",
        f"feature_importance_{model_type}.png",
    ]
    if os.environ.get("PA3030_SAVE_PDF", "0").strip().lower() not in {"0", "false", "no", "off"}:
        outputs.extend([
            "probability_map.pdf",
            "risk_map_bau.pdf",
            "scenario_moderate.pdf",
            "scenario_30x30.pdf",
            "gap_analysis.pdf",
            "hotspot_maps.pdf",
            "conservation_alignment.pdf",
            "climate_stabilisation_alignment.pdf",
            "country_alignment.pdf",
            f"feature_importance_{model_type}.pdf",
        ])
    for fn in outputs:
        p = model_output_dir / fn
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}]  {fn}")


if __name__ == "__main__":
    main()
