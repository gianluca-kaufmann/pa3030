#!/usr/bin/env python3
"""Stage 0a: Region coverage baseline.

Computes total land area and current protection coverage from the backbone
raster and WDPA 2024 raster using latitude-corrected km² calculations.

Background: merged_panel_final.parquet contains only unprotected pixels
(risk-set filter WDPA_prev==0 applied during feature engineering), so it
cannot be used alone for coverage calculations.  The backbone + WDPA rasters
are the correct source for total land area and current protection coverage.

Output:
    outputs/{region}/results/forward/forward_coverage_baseline.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import rasterio

from scripts.regions.shared.forward.config import (  # noqa: E402
    DATA_SUBDIR,
    OUTPUTS_SUBDIR,
    REGION_LABEL,
    get_repo_root,
    resolve_forward_dir,
)
from scripts.regions.shared.geo_utils import pixel_area_km2  # noqa: E402
from scripts.regions.shared.training.utils import WandbRunLogger  # noqa: E402


def resolve_raster(filename: str, data_subdir: str, subdirs: list[str] | None = None) -> Path:
    """Locate a raster: check $SCRATCH then repo, optionally inside subdirs."""
    repo_root = get_repo_root()
    scratch_root = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    bases = []
    if scratch_root is not None:
        bases.append(scratch_root / f"data/{data_subdir}/ready")
    bases.append(repo_root / f"data/{data_subdir}/ready")

    search_subdirs = [""] + (subdirs or [])
    candidates: list[Path] = []
    for base in bases:
        for sd in search_subdirs:
            candidates.append(base / sd / filename if sd else base / filename)

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"'{filename}' not found. Checked:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def _load_naturalearth_lowres() -> "gpd.GeoDataFrame":
    """Load world countries GeoDataFrame, compatible with GeoPandas >= 1.0.

    Priority:
      1. ``geodatasets`` package (ships data locally; official geopandas 1.0 replacement).
      2. Natural Earth 110m cultural shapefile via NACIS CDN (requires network / eth_proxy).
    """
    import geopandas as gpd

    # 1. geodatasets (installed as a transitive dependency of geopandas >= 1.0)
    try:
        import geodatasets
        return gpd.read_file(geodatasets.get_path("naturalearth lowres"))
    except Exception:
        pass

    # 2. Natural Earth 110m countries — direct download (needs internet / eth_proxy)
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    gdf = gpd.read_file(url)
    # 110m cultural uses uppercase column names; normalise to match naturalearth_lowres schema
    rename = {}
    if "ISO_A3" in gdf.columns:
        rename["ISO_A3"] = "iso_a3"
    if "NAME" in gdf.columns:
        rename["NAME"] = "name"
    if rename:
        gdf = gdf.rename(columns=rename)
    return gdf


def compute_coverage_baseline(
    backbone_path: Path,
    wdpa_2024_path: Path,
    output_dir: Path,
    data_subdir: str,
    region_label: str,
    iso_codes: list[str] | None = None,
    wdpa_2019_basename: str | None = None,
) -> dict:
    """Compute total area, 2024 coverage, and scenario thresholds."""
    print("\n" + "=" * 70)
    print(f"COMPUTING {region_label.upper()} COVERAGE BASELINE")
    print("=" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read backbone raster ───────────────────────────────────────────────
    print(f"\nReading backbone raster: {backbone_path}")
    with rasterio.open(backbone_path) as src:
        PIXEL_SIZE_M = abs(src.transform.a)
        transform = src.transform
        print(f"  Pixel size:  {PIXEL_SIZE_M:.2f} m")
        print(f"  CRS:         {src.crs}")
        print(f"  Shape:       {src.height} × {src.width}")
        print(f"  Transform:   {transform}")

        backbone_data = src.read(1)
        nodata = src.nodata

    # Valid land pixels (backbone == 1, excluding nodata)
    if nodata is not None:
        valid_mask = (backbone_data == 1)
    else:
        valid_mask = (backbone_data > 0)

    rows_all, cols_all = np.where(valid_mask)
    total_pixels = int(len(rows_all))
    print(f"  Total {region_label} land pixels: {total_pixels:,}")

    # EPSG:3857 y-centre for each backbone pixel: y = f + (row + 0.5) * e
    # transform.e < 0  (northing decreases as row increases)
    y_all = transform.f + (rows_all + 0.5) * transform.e
    print(f"  y range: [{y_all.min():.0f}, {y_all.max():.0f}] m (EPSG:3857)")

    areas_all = pixel_area_km2(y_all, pixel_size_m=PIXEL_SIZE_M)
    total_km2 = float(areas_all.sum())
    print(f"\n  Total {region_label} area (latitude-corrected): {total_km2:,.0f} km²")

    # ── 2. Read WDPA 2024 raster ──────────────────────────────────────────────
    print(f"\nReading WDPA 2024 raster: {wdpa_2024_path}")
    with rasterio.open(wdpa_2024_path) as src_wdpa:
        print(f"  CRS:   {src_wdpa.crs}")
        print(f"  Shape: {src_wdpa.height} × {src_wdpa.width}")
        wdpa_data = src_wdpa.read(1)

    # Protected = WDPA 2024 == 1 AND backbone == 1 (land pixels only).
    # When raster shapes differ, clip both to the common (h_clip, w_clip) and keep
    # valid_mask_clip aligned so all subsequent masks share the same dimensions.
    if wdpa_data.shape == valid_mask.shape:
        h_clip, w_clip = valid_mask.shape
        valid_mask_clip = valid_mask
        protected_mask = valid_mask & (wdpa_data == 1)
    else:
        print(
            f"  WARNING: WDPA shape {wdpa_data.shape} != backbone shape {valid_mask.shape}. "
            "Clipping to minimum common shape."
        )
        h_clip = min(valid_mask.shape[0], wdpa_data.shape[0])
        w_clip = min(valid_mask.shape[1], wdpa_data.shape[1])
        valid_mask_clip = valid_mask[:h_clip, :w_clip]
        protected_mask = valid_mask_clip & (wdpa_data[:h_clip, :w_clip] == 1)

    prot_rows, prot_cols = np.where(protected_mask)
    protected_pixels = int(len(prot_rows))
    print(f"  Protected 2024 pixels: {protected_pixels:,}")

    y_prot = transform.f + (prot_rows + 0.5) * transform.e
    areas_prot = pixel_area_km2(y_prot, pixel_size_m=PIXEL_SIZE_M)
    protected_2024_km2 = float(areas_prot.sum())
    coverage_pct_2024 = protected_2024_km2 / total_km2

    print(f"  Protected area 2024: {protected_2024_km2:,.0f} km² ({coverage_pct_2024:.1%})")

    # ── 3. BAU target: historical 5-year designation volume (2019→2024) ───────
    bau_km2 = None  # float or None
    try:
        if not wdpa_2019_basename:
            raise FileNotFoundError("no WDPA 2019 basename configured")
        wdpa_2019_path = resolve_raster(wdpa_2019_basename, data_subdir, subdirs=["WDPA", "wdpa"])
        print(f"\nReading WDPA 2019 raster for BAU baseline: {wdpa_2019_path}")
        with rasterio.open(wdpa_2019_path) as src_2019:
            wdpa_2019_data = src_2019.read(1)

        if wdpa_2019_data.shape == valid_mask.shape:
            protected_2019_mask = valid_mask & (wdpa_2019_data == 1)
        else:
            h = min(valid_mask.shape[0], wdpa_2019_data.shape[0])
            w = min(valid_mask.shape[1], wdpa_2019_data.shape[1])
            protected_2019_mask = valid_mask[:h, :w] & (wdpa_2019_data[:h, :w] == 1)

        prot_2019_rows, prot_2019_cols = np.where(protected_2019_mask)
        y_2019 = transform.f + (prot_2019_rows + 0.5) * transform.e
        areas_2019 = pixel_area_km2(y_2019, pixel_size_m=PIXEL_SIZE_M)
        protected_2019_km2 = float(areas_2019.sum())
        bau_km2 = max(0.0, protected_2024_km2 - protected_2019_km2)
        print(f"  Protected 2019: {protected_2019_km2:,.0f} km²")
        print(f"  BAU volume (2019→2024): {bau_km2:,.0f} km² over 5 years")
    except FileNotFoundError:
        print("\n  WDPA 2019 raster not found — BAU km² will not be computed.")
        print("  forward_results will fall back to top-0.5% proxy for BAU cutoff.")

    # ── 4. Scenario thresholds ────────────────────────────────────────────────
    # Moderate target: midpoint between current coverage and 30x30 goal.
    # This adapts per region (e.g. SA ~20% → moderate=25%; USA ~12% → moderate~21%).
    moderate_target_pct = (coverage_pct_2024 + 0.30) / 2
    km2_needed_moderate = max(0.0, moderate_target_pct * total_km2 - protected_2024_km2)
    km2_needed_30 = max(0.0, 0.30 * total_km2 - protected_2024_km2)
    print(f"\n  Scenario thresholds:")
    print(f"    Moderate target ({moderate_target_pct:.1%}, midpoint current→30%): "
          f"{km2_needed_moderate:,.0f} km² additional protection needed")
    print(f"    30% target: {km2_needed_30:,.0f} km² additional protection needed")

    # ── 5. Per-country area stats ─────────────────────────────────────────────
    country_stats: list[dict] = []
    if iso_codes:
        try:
            import geopandas as gpd
            from rasterio.features import rasterize as rio_rasterize

            print("\n  Computing per-country area stats (raster-based)...")
            world_gdf = _load_naturalearth_lowres()
            region_world = world_gdf[world_gdf["iso_a3"].isin(iso_codes)].copy()
            region_world = region_world.to_crs("EPSG:3857")

            # Use the clipped shape so all masks are broadcastable with protected_mask.
            for _, row in region_world.iterrows():
                iso = row["iso_a3"]
                country_name = row["name"]
                c_mask = rio_rasterize(
                    [(row.geometry, 1)],
                    out_shape=(h_clip, w_clip),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                ).astype(bool)

                land_rows, _ = np.where(valid_mask_clip & c_mask)
                y_land = transform.f + (land_rows + 0.5) * transform.e
                total_c_km2 = float(pixel_area_km2(y_land, pixel_size_m=PIXEL_SIZE_M).sum()) if len(land_rows) else 0.0

                prot_rows_c, _ = np.where(protected_mask & c_mask)
                y_prot_c = transform.f + (prot_rows_c + 0.5) * transform.e
                prot_c_km2 = float(pixel_area_km2(y_prot_c, pixel_size_m=PIXEL_SIZE_M).sum()) if len(prot_rows_c) else 0.0

                current_pct = prot_c_km2 / max(total_c_km2, 1.0)
                print(f"    {iso} ({country_name}): total={total_c_km2:,.0f} km², "
                      f"protected={prot_c_km2:,.0f} km² ({current_pct:.1%})")
                country_stats.append({
                    "iso_a3": iso,
                    "country": country_name,
                    "total_km2": round(total_c_km2, 2),
                    "protected_2024_km2": round(prot_c_km2, 2),
                    "current_pct_protected": round(current_pct, 6),
                })
        except Exception as e:
            print(f"  WARNING: Per-country stats failed: {e}")
            import traceback; traceback.print_exc()
            country_stats = []

    # ── 6. Save JSON ──────────────────────────────────────────────────────────
    baseline = {
        "total_land_pixels": total_pixels,
        "total_land_km2": round(total_km2, 2),
        "protected_2024_pixels": protected_pixels,
        "protected_2024_km2": round(protected_2024_km2, 2),
        "coverage_pct_2024": round(coverage_pct_2024, 6),
        "moderate_target_pct": round(moderate_target_pct, 6),
        "km2_needed_for_moderate": round(km2_needed_moderate, 2),
        "km2_needed_for_30pct": round(km2_needed_30, 2),
        "bau_km2": round(bau_km2, 2) if bau_km2 is not None else None,
        "pixel_size_m": float(PIXEL_SIZE_M),
        "backbone_path": str(backbone_path),
        "wdpa_2024_path": str(wdpa_2024_path),
        "country_stats": country_stats,
        # Legacy keys (backward-compat with 4_forward_results.py references)
        "total_sa_pixels": total_pixels,
        "total_sa_km2": round(total_km2, 2),
    }

    # Keep filenames short (the folder already encodes "forward").
    out_path = output_dir / "coverage_baseline.json"
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nSaved: {out_path}")
    return baseline


def main() -> None:
    # Re-import config here so the module-level reload in runner.py takes effect
    from datetime import datetime as _dt

    from scripts.regions.shared.forward.config import (  # noqa: F401
        DATA_SUBDIR,
        ISO_CODES,
        OUTPUTS_SUBDIR,
        REGION_LABEL,
        WDPA_2019_FILENAME,
        WDPA_2024_FILENAME,
        resolve_forward_dir,
    )

    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    wb = WandbRunLogger(
        project="forward",
        run_name=f"coverage_{OUTPUTS_SUBDIR}_{_ts}",
        config={"region": OUTPUTS_SUBDIR, "forward_stage": "coverage_baseline"},
    )
    wb.start()
    wb.log({"coverage/stage": "start"})

    repo_root = get_repo_root()
    backbone_path = resolve_raster("backbone.tif", DATA_SUBDIR, subdirs=["backbone"])
    wdpa_2024_path = resolve_raster(WDPA_2024_FILENAME, DATA_SUBDIR, subdirs=["WDPA", "wdpa"])
    output_dir = resolve_forward_dir(repo_root, OUTPUTS_SUBDIR)
    baseline = compute_coverage_baseline(
        backbone_path, wdpa_2024_path, output_dir,
        data_subdir=DATA_SUBDIR,
        region_label=REGION_LABEL,
        iso_codes=ISO_CODES,
        wdpa_2019_basename=WDPA_2019_FILENAME,
    )
    wb.log(
        {
            "coverage/total_land_km2": baseline.get("total_land_km2"),
            "coverage/protected_2024_km2": baseline.get("protected_2024_km2"),
            "coverage/coverage_pct_2024": baseline.get("coverage_pct_2024"),
            "coverage/km2_needed_for_30pct": baseline.get("km2_needed_for_30pct"),
            "coverage/moderate_target_pct": baseline.get("moderate_target_pct"),
            "coverage/bau_km2": baseline.get("bau_km2"),
            "coverage/stage": "done",
        }
    )
    wb.finish()


if __name__ == "__main__":
    main()
