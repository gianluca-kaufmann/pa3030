#!/usr/bin/env python3
"""
Issue F — WDPA label quality audit.

Loads the WDPA May2026 shapefile and for each region/year:
  - Builds a Designated-only reference raster (cumulative, incremental)
  - Compares with the existing TIF (exported from GEE without STATUS filter)
  - Reports:
      contamination%  = TIF-protected pixels NOT in Designated raster  (STATUS bug)
      missing%        = Designated pixels NOT in TIF                   (reporting lag)

Decision gates:
  contamination > 5%  →  GEE re-export with STATUS='Designated' filter required
  missing pixels      →  patch TIFs or rely on corrected uploads

With --fix:
  Burns missing Designated pixels into TIFs via OR-merge (in-place, backup first).
  Use to fix USA 2023/2024 the same way SA/SEA were already fixed.

Outputs:
  outputs/<region>/results/wdpa_audit_<region>.json   (machine-readable)
  stdout                                               (human-readable table)

Usage:
    # Full audit (all regions, no fixes):
    python scripts/regions/shared/audit_wdpa_status.py

    # Fast mode (panel-level stats only, no shapefile needed):
    python scripts/regions/shared/audit_wdpa_status.py --mode fast --region sa

    # Audit + fix missing pixels in-place (generates _pre_patch backups):
    python scripts/regions/shared/audit_wdpa_status.py --region usa --fix

    # Audit specific years only (faster for spot-checks):
    python scripts/regions/shared/audit_wdpa_status.py --region sa --years 2020 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

if "SCRATCH" in os.environ:
    DATA_ROOT = Path(os.environ["SCRATCH"]) / "data"
else:
    DATA_ROOT = PROJECT_ROOT / "data"

SHAPEFILE_PARTS = [
    DATA_ROOT / "shared" / "WDPA" / f"WDPA_May2026_Public_shp_{i}"
    / "WDPA_May2026_Public_shp-polygons.shp"
    for i in range(3)
]

REGION_CONFIG = {
    "sa": {
        "name": "South America",
        "wdpa_dir": DATA_ROOT / "south_america" / "ready" / "WDPA",
        "wdpa_prefix": "WDPA_SA_1km",
        "panel": DATA_ROOT / "south_america" / "ml" / "merged_panel_final.parquet",
        "output_dir": PROJECT_ROOT / "outputs" / "south_america" / "results",
        "iso3": {
            "ARG", "BOL", "BRA", "CHL", "COL", "ECU",
            "GUY", "PER", "PRY", "SUR", "URY", "VEN",
            "PAN", "CRI", "GTM", "HND", "MEX", "NIC", "SLV",
        },
    },
    "sea": {
        "name": "SE Asia",
        "wdpa_dir": DATA_ROOT / "se_asia" / "ready" / "WDPA",
        "wdpa_prefix": "WDPA_SEA_1km",
        "panel": DATA_ROOT / "se_asia" / "ml" / "merged_panel_final.parquet",
        "output_dir": PROJECT_ROOT / "outputs" / "se_asia" / "results",
        "iso3": {
            "IDN", "MYS", "PHL", "THA", "VNM", "KHM", "LAO", "MMR",
            "BRN", "SGP", "TLS",
        },
    },
    "usa": {
        "name": "USA",
        "wdpa_dir": DATA_ROOT / "usa" / "ready" / "WDPA",
        "wdpa_prefix": "WDPA_USA_1km",
        "panel": DATA_ROOT / "usa" / "ml" / "merged_panel_final.parquet",
        "output_dir": PROJECT_ROOT / "outputs" / "usa" / "results",
        "iso3": {"USA"},
    },
}

AUDIT_YEARS = list(range(2000, 2025))


# ---------------------------------------------------------------------------
# Mode 1: fast panel-based stats (no shapefile)
# ---------------------------------------------------------------------------
def audit_fast(cfg: dict) -> None:
    panel = cfg["panel"]
    if not panel.exists():
        print(f"  [SKIP] Panel not found: {panel}")
        return

    print(f"\n=== Fast audit: {cfg['name']} ===")
    df = pd.read_parquet(panel, columns=["year", "transition_01", "WDPA_prev"])
    total = len(df)
    positives = int((df["transition_01"] == 1).sum())
    print(f"  Total rows:      {total:>12,}")
    print(f"  Positives (t=1): {positives:>12,}  ({100*positives/total:.4f}%)")

    by_year = (
        df.groupby("year")["transition_01"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "positives", "count": "total"})
    )
    by_year["rate_pct"] = 100 * by_year["positives"] / by_year["total"]
    print("\n  Per-year positive rate:")
    print(by_year.to_string())
    print("\n  NOTE: Cannot separate Designated vs non-Designated without shapefile.")
    print("  Run without --mode fast to use the shapefile.")


# ---------------------------------------------------------------------------
# Load shapefile — all Designated polygons for a set of ISO3 codes
# ---------------------------------------------------------------------------
def load_designated(iso3_set: set[str]) -> "gpd.GeoDataFrame":
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("ERROR: geopandas required. pip install geopandas")

    missing = [p for p in SHAPEFILE_PARTS if not p.exists()]
    if missing:
        print("  [ERROR] Missing shapefile parts:")
        for p in missing:
            print(f"    {p}")
        sys.exit(1)

    parts = []
    for shp in SHAPEFILE_PARTS:
        gdf = gpd.read_file(shp, where=f"STATUS = 'Designated'")
        mask = gdf["ISO3"].isin(iso3_set) & gdf["STATUS_YR"].notna()
        parts.append(gdf[mask][["ISO3", "STATUS_YR", "geometry"]].copy())

    import pandas as pd_local
    combined = pd_local.concat(parts, ignore_index=True)
    combined["STATUS_YR"] = combined["STATUS_YR"].astype(int)
    combined = combined.sort_values("STATUS_YR").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Full audit (+ optional fix) for one region
# ---------------------------------------------------------------------------
def audit_region(
    cfg: dict,
    years: list[int],
    fix: bool,
    dry_run: bool,
) -> list[dict]:
    import geopandas as gpd  # noqa: F401

    name = cfg["name"]
    wdpa_dir = cfg["wdpa_dir"]
    prefix = cfg["wdpa_prefix"]

    print(f"\n{'='*60}")
    print(f"  Full audit: {name}")
    print(f"{'='*60}")

    print("  Loading Designated polygons from shapefile...")
    gdf = load_designated(cfg["iso3"])
    if len(gdf) == 0:
        print("  [SKIP] No Designated polygons found.")
        return []
    print(f"  Loaded {len(gdf):,} Designated polygons for {name}")

    # Get grid spec from the first available TIF
    ref_tif = None
    for y in years:
        p = wdpa_dir / f"{prefix}_{y}.tif"
        if p.exists():
            ref_tif = p
            break
    if ref_tif is None:
        print(f"  [SKIP] No TIF files found in {wdpa_dir}")
        return []

    with rasterio.open(ref_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
        crs_str = str(src.crs)

    print(f"  Grid: {shape[1]} x {shape[0]} = {shape[0]*shape[1]:,} pixels  CRS={crs_str}")

    # Reproject polygons to TIF CRS once
    gdf_proj = gdf.to_crs(crs_str)

    # Incremental cumulative Designated raster — each polygon burned exactly once.
    # For year Y we only rasterize polygons with STATUS_YR in (prev_year, Y].
    desig_cumulative = np.zeros(shape, dtype="float32")
    prev_year_processed: int | None = None

    results = []

    for year in sorted(years):
        tif_path = wdpa_dir / f"{prefix}_{year}.tif"
        if not tif_path.exists():
            print(f"  [SKIP] {tif_path.name} not found")
            continue

        # Delta: only new polygons since last processed year
        if prev_year_processed is None:
            delta = gdf_proj[gdf_proj["STATUS_YR"] <= year]
        else:
            delta = gdf_proj[
                (gdf_proj["STATUS_YR"] > prev_year_processed)
                & (gdf_proj["STATUS_YR"] <= year)
            ]

        if len(delta) > 0:
            burned_raster = rasterize(
                [(geom, 1.0) for geom in delta.geometry],
                out_shape=shape,
                transform=transform,
                fill=0.0,
                dtype="float32",
                all_touched=False,
            )
            desig_cumulative = np.fmax(desig_cumulative, burned_raster)

        prev_year_processed = year

        # Load existing TIF
        with rasterio.open(tif_path) as src:
            existing = src.read(1)

        valid = ~np.isnan(existing)

        tif_px = int((existing[valid] == 1).sum())
        desig_px = int((desig_cumulative[valid] == 1).sum())

        # Contamination: in TIF but NOT in Designated raster
        contam_px = int(((existing == 1) & (desig_cumulative == 0) & valid).sum())
        contam_pct = 100 * contam_px / tif_px if tif_px > 0 else 0.0

        # Missing: in Designated raster but NOT in TIF
        missing_px = int(((desig_cumulative == 1) & (existing == 0) & valid).sum())
        missing_pct = 100 * missing_px / desig_px if desig_px > 0 else 0.0

        results.append({
            "year": year,
            "tif_px": tif_px,
            "desig_px": desig_px,
            "contam_px": contam_px,
            "contam_pct": round(contam_pct, 3),
            "missing_px": missing_px,
            "missing_pct": round(missing_pct, 3),
        })

        flag = ""
        if contam_pct > 5:
            flag += " *** CONTAMINATION >5% ***"
        if missing_pct > 10:
            flag += " !! MISSING >10% !!"
        print(
            f"  {year}: TIF={tif_px:>8,}  Desig={desig_px:>8,}  "
            f"contam={contam_pct:5.2f}%  missing={missing_pct:5.2f}%{flag}"
        )

        # Fix: OR-merge missing pixels into TIF
        if fix and missing_px > 0:
            _fix_tif(tif_path, existing, desig_cumulative, valid, profile, dry_run)

    # Summary
    if results:
        df_r = pd.DataFrame(results)
        total_tif = df_r["tif_px"].sum()
        total_contam = df_r["contam_px"].sum()
        overall_contam = 100 * total_contam / total_tif if total_tif > 0 else 0.0
        print(f"\n  --- Overall contamination: {total_contam:,} / {total_tif:,} px = {overall_contam:.3f}% ---")
        if overall_contam > 5:
            print("  *** CONTAMINATION >5% — GEE re-export with STATUS='Designated' filter required ***")
        else:
            print(f"  Contamination {overall_contam:.3f}% < 5% threshold — labels acceptable.")

    return results


# ---------------------------------------------------------------------------
# Patch TIF in-place
# ---------------------------------------------------------------------------
def _fix_tif(
    tif_path: Path,
    existing: np.ndarray,
    desig_raster: np.ndarray,
    valid: np.ndarray,
    profile: dict,
    dry_run: bool,
) -> None:
    patched = np.fmax(existing, desig_raster)
    patched = np.where(np.isnan(existing), np.nan, patched)
    new_px = int(((patched == 1) & (existing != 1) & valid).sum())
    print(f"    [FIX] {tif_path.name}: +{new_px:,} pixels to burn", end="")

    if dry_run:
        print(" (dry-run, not written)")
        return

    backup = tif_path.parent / (tif_path.stem + "_pre_patch.tif")
    if not backup.exists():
        shutil.copy2(tif_path, backup)
    profile.update(dtype="float32")
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(patched.astype("float32"), 1)
    print(" → written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Issue F: WDPA label quality audit")
    parser.add_argument("--mode", choices=["fast", "full"], default="full",
                        help="fast=panel stats only (no shapefile); full=shapefile comparison (default)")
    parser.add_argument("--region", choices=["sa", "sea", "usa", "all"], default="all")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Years to audit (default: 2000-2024)")
    parser.add_argument("--fix", action="store_true",
                        help="Burn missing Designated pixels into TIFs (in-place, backup first)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --fix: show what would be done without writing")
    args = parser.parse_args()

    years = args.years or AUDIT_YEARS
    regions = list(REGION_CONFIG.keys()) if args.region == "all" else [args.region]

    all_results: dict[str, list[dict]] = {}

    for region in regions:
        cfg = REGION_CONFIG[region]
        if args.mode == "fast":
            audit_fast(cfg)
        else:
            results = audit_region(cfg, years, fix=args.fix, dry_run=args.dry_run)
            all_results[region] = results

    # Save JSON output
    if args.mode == "full" and all_results:
        for region, results in all_results.items():
            cfg = REGION_CONFIG[region]
            out_dir = cfg["output_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"wdpa_audit_{region}.json"
            with open(out_path, "w") as f:
                json.dump({"region": region, "years": results}, f, indent=2)
            print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
