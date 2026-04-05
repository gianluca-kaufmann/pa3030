#!/usr/bin/env python3
"""Stage: Spatial clustering of top-risk pixels (DBSCAN).

Identifies major spatial clusters in the top-1% forward risk zone, names them
by dominant biome and country, and produces:
  cluster_summary.csv / .tex       — per-cluster statistics table
  cluster_hotspot_boxes.json       — bounding boxes for results_core hotspot maps
  cluster_map.png / .pdf           — continental map with cluster labels

Run via the 7_forward_clustering.py thin wrapper for each region, or via
runner.py with stage="clustering".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

from scripts.regions.shared.forward.config import (  # noqa: E402
    DATA_SUBDIR,
    FORWARD_PA_HOLE_COLOR,
    MODEL_PREFIX,
    OUTPUTS_SUBDIR,
    REGION_LABEL,
    forward_dir_search_paths,
    get_repo_root,
)
from scripts.regions.shared.forward.results_core import (  # noqa: E402
    PROBA_COL,
    _safe_savefig,
    _plot_backbone_background,
    _resolve_plot_bounds_3857,
    _add_latlon_ticks,
    epsg3857_to_lonlat,
    load_scored,
    add_area_col,
    resolve_scored_parquet,
    resolve_backbone_pixel_size,
    resolve_coverage_baseline,
    resolve_backbone_path_for_plot,
    resolve_country_iso3_raster,
    points_to_raster,
    _IGBP_GROUPS,
)
from scripts.regions.shared.results.results_core import (  # noqa: E402
    _plot_backbone_background as _pb_bg,  # already imported above via forward
)

# ── Constants ─────────────────────────────────────────────────────────────────
# DBSCAN parameters (EPSG:3857 metres)
DBSCAN_EPS_M     = 100_000   # 100 km neighbourhood radius
DBSCAN_MIN_SAMP  = 300       # minimum pixels to form a core point
TOP_PCT_THRESHOLD = 0.99     # percentile cutoff for "top-risk" pixels (top 1%)
MAX_CLUSTERS_TABLE = 10      # maximum rows in summary table
MAX_CLUSTERS_MAP   = 8       # maximum labelled clusters on map
# bbox buffer in degrees around each cluster centroid
BBOX_BUFFER_DEG  = 3.0

# MODIS IGBP group codes (same as results_core._IGBP_GROUPS)
_IGBP_AGRI_CODES = {12, 14}

# ── Cluster name generation ───────────────────────────────────────────────────

def _dominant(series: pd.Series) -> Any:
    """Return the most frequent non-null value in a Series."""
    counts = series.dropna().value_counts()
    return counts.index[0] if len(counts) > 0 else None


def _igbp_group_name(codes_arr: np.ndarray) -> str:
    """Return the dominant IGBP group name for an array of integer codes."""
    code_to_group: Dict[int, str] = {}
    for gname, codes in _IGBP_GROUPS.items():
        for c in codes:
            code_to_group[c] = gname
    groups = [code_to_group.get(int(c), "Unknown") for c in codes_arr if not np.isnan(c)]
    if not groups:
        return "Unknown"
    from collections import Counter
    return Counter(groups).most_common(1)[0][0]


# ── Country ISO3 lookup ───────────────────────────────────────────────────────

def _build_country_lookup(
    repo_root: Path,
    data_subdir: str,
    row_arr: np.ndarray,
    col_arr: np.ndarray,
) -> np.ndarray:
    """Return array of ISO3 strings (or '' for unresolved) for each pixel."""
    iso_path, id2iso_path = resolve_country_iso3_raster(repo_root, data_subdir)
    if iso_path is None or not iso_path.exists():
        return np.full(len(row_arr), "", dtype=object)

    try:
        import rasterio
        with rasterio.open(iso_path) as src:
            iso_grid = src.read(1)
        h, w = iso_grid.shape
        import json as _json
        with open(id2iso_path) as _f:
            _raw = _json.load(_f) or {}
        id2iso = {int(k): v for k, v in _raw.items() if v is not None}
        valid = (row_arr >= 0) & (row_arr < h) & (col_arr >= 0) & (col_arr < w)
        ids = np.zeros(len(row_arr), dtype=np.int32)
        ids[valid] = iso_grid[row_arr[valid], col_arr[valid]].astype(np.int32)
        return np.array([id2iso.get(int(i), "") for i in ids], dtype=object)
    except Exception as e:
        print(f"  WARNING: country lookup failed: {e}")
        return np.full(len(row_arr), "", dtype=object)


# ── Biome lookup via GSN ecoregions raster ───────────────────────────────────

def _build_biome_lookup(
    repo_root: Path,
    data_subdir: str,
    row_arr: np.ndarray,
    col_arr: np.ndarray,
) -> np.ndarray:
    """Return array of ecoregion integer IDs (-1 for unresolved)."""
    gsn_path_candidates = [
        Path(os.environ.get("SCRATCH", "")) / f"data/{data_subdir}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif",
        repo_root / f"data/{data_subdir}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif",
    ]
    gsn_path = next((p for p in gsn_path_candidates if p.exists()), None)
    if gsn_path is None:
        return np.full(len(row_arr), -1, dtype=np.int32)

    try:
        import rasterio
        with rasterio.open(gsn_path) as src:
            eco_grid = src.read(1)
        h, w = eco_grid.shape
        valid = (row_arr >= 0) & (row_arr < h) & (col_arr >= 0) & (col_arr < w)
        ids = np.full(len(row_arr), -1, dtype=np.int32)
        ids[valid] = eco_grid[row_arr[valid], col_arr[valid]].astype(np.int32)
        return ids
    except Exception as e:
        print(f"  WARNING: ecoregion lookup failed: {e}")
        return np.full(len(row_arr), -1, dtype=np.int32)


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_clustering(
    region: str,
    model_type: str = "lgbm",
    dbscan_eps_m: float = DBSCAN_EPS_M,
    dbscan_min_samples: int = DBSCAN_MIN_SAMP,
    top_pct: float = TOP_PCT_THRESHOLD,
    max_clusters: int = MAX_CLUSTERS_TABLE,
) -> None:
    """Run full spatial clustering pipeline for one region."""
    from sklearn.cluster import DBSCAN

    repo_root    = get_repo_root()
    forward_dirs = forward_dir_search_paths(repo_root, OUTPUTS_SUBDIR)
    forward_dir  = forward_dirs[0]
    output_dir   = forward_dir / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SPATIAL CLUSTERING — {REGION_LABEL} ({model_type.upper()})")
    print("=" * 70)

    # ── Load scored parquet ───────────────────────────────────────────────────
    scored_path  = resolve_scored_parquet(forward_dirs, model_type)
    baseline     = resolve_coverage_baseline(forward_dirs)
    pixel_size_m = resolve_backbone_pixel_size(repo_root, DATA_SUBDIR)
    if pixel_size_m == 1000.0 and "pixel_size_m" in baseline:
        pixel_size_m = float(baseline["pixel_size_m"])

    df = load_scored(scored_path)
    df = add_area_col(df, pixel_size_m)

    # ── Filter to top-1% pixels ───────────────────────────────────────────────
    threshold = float(np.percentile(df[PROBA_COL].values, 100 * top_pct))
    top_df = df[df[PROBA_COL] >= threshold].copy()
    print(f"  Top-{100*(1-top_pct):.0f}% threshold: {threshold:.6f}")
    print(f"  Top-{100*(1-top_pct):.0f}% pixels:    {len(top_df):,} of {len(df):,}")
    del df

    # ── Load features for top-1% pixels ──────────────────────────────────────
    feat_path: Optional[Path] = None
    for fd in forward_dirs:
        cand = fd / "forward_features_2024.parquet"
        if cand.exists():
            feat_path = cand
            break

    feat_cols_wanted = ["landcover", "dist_wdpa"]
    if feat_path is not None:
        schema_names = set(pq.ParquetFile(feat_path).schema_arrow.names)
        available_feat = [c for c in feat_cols_wanted if c in schema_names]
        if available_feat:
            print(f"  Loading features {available_feat} for top pixels …")
            feat_sub = pq.read_table(feat_path, columns=["row", "col"] + available_feat).to_pandas()
            top_df = top_df.merge(feat_sub, on=["row", "col"], how="left")
            del feat_sub

    # ── DBSCAN clustering ─────────────────────────────────────────────────────
    X = top_df[["x", "y"]].values.astype(np.float64)
    print(f"\n  Running DBSCAN (eps={dbscan_eps_m/1000:.0f} km, min_samples={dbscan_min_samples}) …")
    db = DBSCAN(eps=dbscan_eps_m, min_samples=dbscan_min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)
    top_df["cluster_id"] = labels

    n_clusters = int((labels >= 0).any()) and int(labels.max()) + 1
    n_noise    = int((labels == -1).sum())
    print(f"  Clusters found: {n_clusters}  |  Noise pixels: {n_noise:,}")
    if n_clusters == 0:
        print("  No clusters found — try reducing DBSCAN_MIN_SAMP or increasing DBSCAN_EPS_M")
        return

    # ── Country + biome lookup for top pixels ─────────────────────────────────
    row_arr = top_df["row"].values.astype(int)
    col_arr = top_df["col"].values.astype(int)
    top_df["country_iso3"] = _build_country_lookup(repo_root, DATA_SUBDIR, row_arr, col_arr)
    # (ecoregion lookup is optional — skip if raster not available locally)
    top_df["eco_id"] = _build_biome_lookup(repo_root, DATA_SUBDIR, row_arr, col_arr)

    # ── Compute per-cluster statistics ────────────────────────────────────────
    lon_all, lat_all = epsg3857_to_lonlat(
        top_df["x"].values.astype(np.float64),
        top_df["y"].values.astype(np.float64),
    )
    top_df["lon"] = lon_all
    top_df["lat"] = lat_all

    cluster_rows: List[Dict[str, Any]] = []
    for cid in range(n_clusters):
        mask = top_df["cluster_id"] == cid
        grp  = top_df[mask]
        if len(grp) == 0:
            continue

        centroid_lon = float(grp["lon"].mean())
        centroid_lat = float(grp["lat"].mean())
        lon_min = float(grp["lon"].min()) - BBOX_BUFFER_DEG
        lon_max = float(grp["lon"].max()) + BBOX_BUFFER_DEG
        lat_min = float(grp["lat"].min()) - BBOX_BUFFER_DEG
        lat_max = float(grp["lat"].max()) + BBOX_BUFFER_DEG

        # Dominant country
        dom_country = _dominant(grp["country_iso3"].replace("", np.nan)) or "Unknown"

        # Dominant IGBP group
        if "landcover" in grp.columns:
            lc_arr = grp["landcover"].dropna().values
            dom_landcover = _igbp_group_name(lc_arr)
            agri_frac = float(
                grp["landcover"].isin(list(_IGBP_AGRI_CODES)).sum() / max(1, len(grp))
            )
        else:
            dom_landcover = "Unknown"
            agri_frac = float("nan")

        # Distance to existing PA
        mean_dist_wdpa_km = float("nan")
        if "dist_wdpa" in grp.columns:
            mean_dist_wdpa_km = round(float(grp["dist_wdpa"].dropna().mean() / 1000.0), 1)

        # Biodiversity / climate zone fractions (GSN bands from top_df if available)
        bio_frac = float("nan")
        clim_frac = float("nan")
        if "GSN_b2" in grp.columns:
            bio_frac = round(float(grp["GSN_b2"].fillna(0).mean()), 3)
        if "GSN_b1" in grp.columns:
            clim_frac = round(float(grp["GSN_b1"].fillna(0).mean()), 3)

        cluster_rows.append({
            "cluster_id":          cid,
            "n_pixels":            len(grp),
            "area_km2":            round(float(grp["area_km2"].sum()), 0),
            "mean_score":          round(float(grp[PROBA_COL].mean()), 5),
            "centroid_lon":        round(centroid_lon, 2),
            "centroid_lat":        round(centroid_lat, 2),
            "dominant_country":    dom_country,
            "dominant_landcover":  dom_landcover,
            "agri_frac":           round(agri_frac, 3) if not np.isnan(agri_frac) else None,
            "mean_dist_wdpa_km":   mean_dist_wdpa_km if not np.isnan(mean_dist_wdpa_km) else None,
            "biodiversity_frac":   bio_frac if not np.isnan(bio_frac) else None,
            "climate_stab_frac":   clim_frac if not np.isnan(clim_frac) else None,
            "lon_min":             round(lon_min, 2),
            "lon_max":             round(lon_max, 2),
            "lat_min":             round(lat_min, 2),
            "lat_max":             round(lat_max, 2),
        })

    cluster_df = pd.DataFrame(cluster_rows).sort_values("area_km2", ascending=False).reset_index(drop=True)
    cluster_df.insert(0, "rank", range(1, len(cluster_df) + 1))
    # Human-readable label: Rank. Country (Landcover)
    cluster_df["label"] = cluster_df.apply(
        lambda r: f"C{int(r['rank'])}. {r['dominant_country']} ({r['dominant_landcover']})",
        axis=1,
    )

    # ── Save cluster summary CSV ──────────────────────────────────────────────
    csv_path = output_dir / "cluster_summary.csv"
    cluster_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(cluster_df[["rank", "label", "n_pixels", "area_km2", "mean_score",
                       "dominant_country", "dominant_landcover",
                       "mean_dist_wdpa_km"]].head(max_clusters).to_string(index=False))

    # ── Save LaTeX table ──────────────────────────────────────────────────────
    top_for_table = cluster_df.head(max_clusters)
    tex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{Top spatial clusters in the forward-model top-1\% risk zone.",
        rf"   Clustering by DBSCAN ($\varepsilon={dbscan_eps_m/1000:.0f}$~km, min\_samples={dbscan_min_samples}).}}",
        r"  \label{tab:spatial_clusters}",
        r"  \begin{tabular}{clrrrccc}",
        r"    \toprule",
        r"    Rank & Label & n & Area (km$^2$) & Mean score & Country & Land cover & Dist PA (km) \\",
        r"    \midrule",
    ]
    for _, row in top_for_table.iterrows():
        dist_str = f"{row['mean_dist_wdpa_km']:.0f}" if row['mean_dist_wdpa_km'] is not None else "--"
        tex_lines.append(
            f"    {int(row['rank'])} & {row['label']} & "
            f"{int(row['n_pixels']):,} & {int(row['area_km2']):,} & "
            f"{row['mean_score']:.5f} & "
            f"{row['dominant_country']} & {row['dominant_landcover']} & {dist_str} \\\\"
        )
    tex_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    tex_path = output_dir / "cluster_summary.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"  Saved: {tex_path}")

    # ── Save hotspot bounding boxes JSON ─────────────────────────────────────
    top_for_hotspot = cluster_df.head(MAX_CLUSTERS_MAP)
    boxes = []
    for _, row in top_for_hotspot.iterrows():
        boxes.append({
            "label":   row["label"],
            "rank":    int(row["rank"]),
            "lon_min": float(row["lon_min"]),
            "lon_max": float(row["lon_max"]),
            "lat_min": float(row["lat_min"]),
            "lat_max": float(row["lat_max"]),
        })
    boxes_path = output_dir / "cluster_hotspot_boxes.json"
    boxes_path.write_text(json.dumps(boxes, indent=2))
    print(f"  Saved: {boxes_path}")

    # ── Cluster map ───────────────────────────────────────────────────────────
    _plot_cluster_map(
        top_df=top_df,
        cluster_df=cluster_df,
        n_clusters=n_clusters,
        output_dir=output_dir,
        baseline=baseline,
        repo_root=repo_root,
        region_label=REGION_LABEL,
        max_clusters_map=MAX_CLUSTERS_MAP,
    )

    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETE")
    print("=" * 70)
    for fn in [csv_path, tex_path, boxes_path,
               output_dir / "cluster_map.png", output_dir / "cluster_map.pdf"]:
        status = "OK" if fn.exists() else "MISSING"
        print(f"  [{status}]  {fn.name}")


def _plot_cluster_map(
    top_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    n_clusters: int,
    output_dir: Path,
    baseline: Dict[str, Any],
    repo_root: Path,
    region_label: str,
    max_clusters_map: int = MAX_CLUSTERS_MAP,
) -> None:
    """Continental map showing top-1% pixels coloured by cluster ID."""
    print("\n  Creating cluster map …")

    proj_bounds  = _resolve_plot_bounds_3857(repo_root, DATA_SUBDIR, baseline)
    backbone_path = resolve_backbone_path_for_plot(repo_root, DATA_SUBDIR, baseline)

    proj_width  = proj_bounds[2] - proj_bounds[0]
    proj_height = proj_bounds[3] - proj_bounds[1]
    fig_w = 14.0
    fig_h = fig_w * (proj_height / max(proj_width, 1e-9))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    _plot_backbone_background(ax, backbone_path, zorder=0, hole_color=FORWARD_PA_HOLE_COLOR)

    # Colour palette for clusters
    cmap_clusters = plt.cm.get_cmap("tab20", max(n_clusters, 1))
    n_show = min(n_clusters, max_clusters_map)

    # Background noise pixels (cluster_id == -1) in light grey
    noise_mask = top_df["cluster_id"] == -1
    if noise_mask.any():
        noise_sub = top_df[noise_mask]
        xm_n = noise_sub["x"].values.astype(np.float64)
        ym_n = noise_sub["y"].values.astype(np.float64)
        grid_n, gext_n = points_to_raster(
            xm_n, ym_n, np.ones(len(xm_n), dtype=np.float32),
            target_resolution=3000.0, agg_func="max", extent_bounds=proj_bounds,
        )
        ax.imshow(
            np.where(np.isnan(grid_n), np.nan, 1.0),
            extent=gext_n, origin="upper", interpolation="nearest",
            cmap=plt.cm.colors.ListedColormap(["#cccccc"]),
            vmin=0, vmax=1, alpha=0.5, zorder=1,
        )

    # Ranked clusters (by area): plot largest first (lowest z-order) → top on top
    ranked_ids = cluster_df["cluster_id"].values
    for i, cid in enumerate(ranked_ids[:n_show]):
        mask = top_df["cluster_id"] == cid
        grp  = top_df[mask]
        xm   = grp["x"].values.astype(np.float64)
        ym   = grp["y"].values.astype(np.float64)
        color = cmap_clusters(i % cmap_clusters.N)
        grid_c, gext_c = points_to_raster(
            xm, ym, np.ones(len(xm), dtype=np.float32),
            target_resolution=2000.0, agg_func="max", extent_bounds=proj_bounds,
        )
        import matplotlib.colors as mcolors
        ax.imshow(
            np.where(np.isnan(grid_c), np.nan, 1.0),
            extent=gext_c, origin="upper", interpolation="nearest",
            cmap=mcolors.ListedColormap([color]),
            vmin=0, vmax=1, alpha=0.85, zorder=2 + i,
        )

    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    _add_latlon_ticks(ax, (proj_bounds[0], proj_bounds[2]), (proj_bounds[1], proj_bounds[3]))

    ax.set_title(
        f"Top-1% Risk Zone Spatial Clusters — {region_label}\n"
        f"DBSCAN  ε = {DBSCAN_EPS_M/1000:.0f} km  |  "
        f"Top {n_show} of {n_clusters} clusters labelled",
        fontsize=11,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4, zorder=10)

    # Legend
    handles = []
    for i, (_, row) in enumerate(cluster_df.head(n_show).iterrows()):
        color = cmap_clusters(i % cmap_clusters.N)
        handles.append(mpatches.Patch(color=color, label=row["label"]))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="lower left",
                  framealpha=0.85, ncol=1)

    plt.tight_layout(pad=0.5)
    for ext in ["png", "pdf"]:
        p = output_dir / f"cluster_map.{ext}"
        if _safe_savefig(p, dpi=200, bbox_inches="tight"):
            print(f"  Saved: {p}")
    try:
        plt.close(fig)
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    region     = os.environ.get("PA3030_FORWARD_REGION", "south_america").strip().lower()
    model_type = os.environ.get("PA3030_FORWARD_MODEL_TYPE", "lgbm").strip().lower()
    run_clustering(region=region, model_type=model_type)


if __name__ == "__main__":
    main()
