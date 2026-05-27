#!/usr/bin/env python3
"""
Patch WDPA_SEA_1km_2023.tif and WDPA_SEA_1km_2024.tif with missing IDN and THA
designations from WDPA May2026 shapefile.

Background: The GEE WDPA asset (WCMC/WDPA/current/polygons) is confirmed missing
IDN 2023-2024 designations (GEE: ee.Filter(ISO3=IDN, STATUS_YR>=2023).size()==0).
WDPA May2026 CSV confirms: IDN 2023=25,320 km², IDN 2024=25,038 km², THA 2023=750 km².
The existing WDPA_SEA_1km_2022.tif == 2023.tif == 2024.tif (all 679,234 px) — confirmed artefact.

Fix: rasterize the missing polygons at 1km EPSG:3857 onto the existing TIF grid,
then OR-merge (np.maximum) into the existing cumulative TIF for each year.

Patched TIFs are written in-place (originals backed up with _pre_patch suffix).

Usage (run locally or on Euler — paths auto-detected via $SCRATCH):
    python scripts/regions/se_asia/2_preprocessing/patch_wdpa_sea.py

    # Dry-run: print stats without writing
    python scripts/regions/se_asia/2_preprocessing/patch_wdpa_sea.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent

if "SCRATCH" in os.environ:
    DATA_ROOT = Path(os.environ["SCRATCH"]) / "data"
else:
    DATA_ROOT = PROJECT_ROOT / "data"

WDPA_DIR = DATA_ROOT / "se_asia" / "ready" / "WDPA"
SHAPEFILE_PARTS = [
    PROJECT_ROOT / "data" / "shared" / "WDPA" / f"WDPA_May2026_Public_shp_{i}"
    / f"WDPA_May2026_Public_shp-polygons.shp"
    for i in range(3)
]

# SEA countries whose recent designations may be missing from the GEE asset
# (IDN confirmed; THA 2023 also found in May2026 shapefile)
SEA_ISO3 = {
    "BRN", "KHM", "IDN", "LAO", "MYS", "MMR",
    "PHL", "SGP", "THA", "TLS", "VNM",
}

# Patch years: cumulative TIF for year Y should include all STATUS_YR <= Y
PATCH_YEARS = [2023, 2024]


# ---------------------------------------------------------------------------
# Load missing polygons
# ---------------------------------------------------------------------------
def load_missing_polygons() -> gpd.GeoDataFrame:
    """Load all SEA Designated polygons with STATUS_YR in PATCH_YEARS from shapefile."""
    parts = []
    for shp in SHAPEFILE_PARTS:
        if not shp.exists():
            print(f"  [WARN] Shapefile part not found: {shp}")
            continue
        gdf = gpd.read_file(shp)
        mask = (
            gdf["ISO3"].isin(SEA_ISO3)
            & gdf["STATUS_YR"].isin(PATCH_YEARS)
            & (gdf["STATUS"] == "Designated")
        )
        parts.append(gdf[mask].copy())

    if not parts:
        sys.exit("ERROR: No shapefile parts found. Check SHAPEFILE_PARTS paths.")

    combined = pd.concat(parts, ignore_index=True)
    print(f"\nLoaded {len(combined)} missing SEA Designated polygons")
    print(combined.groupby(["ISO3", "STATUS_YR"])["GIS_AREA"].agg(["count", "sum"]).round(0))
    return gpd.GeoDataFrame(combined, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Rasterize and patch
# ---------------------------------------------------------------------------
def patch_year(gdf_all: gpd.GeoDataFrame, year: int, dry_run: bool) -> None:
    tif_path = WDPA_DIR / f"WDPA_SEA_1km_{year}.tif"
    if not tif_path.exists():
        print(f"  [SKIP] {tif_path} not found")
        return

    # Cumulative: include all STATUS_YR <= year
    gdf = gdf_all[gdf_all["STATUS_YR"] <= year].copy()
    if len(gdf) == 0:
        print(f"  [SKIP] No new polygons for year {year}")
        return

    print(f"\n--- Patching {tif_path.name} ---")
    print(f"  Polygons to burn: {len(gdf)}")
    area = gdf["GIS_AREA"].sum()
    print(f"  GIS_AREA total:   {area:,.0f} km²")

    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
        existing = src.read(1)  # float32, values 0/1/NaN

    # Reproject polygons to TIF CRS (EPSG:3857)
    gdf_3857 = gdf.to_crs("EPSG:3857")

    # Rasterize: burn 1.0 for any polygon pixel
    new_pixels = rasterize(
        [(geom, 1.0) for geom in gdf_3857.geometry],
        out_shape=shape,
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=False,
    )

    burned = int((new_pixels == 1).sum())
    print(f"  New pixels burned (1km²): {burned:,}")

    # OR-merge: max preserves existing=1 and NaN backbone pixels
    # NaN stays NaN (outside backbone) because max(NaN, x) = NaN in numpy
    patched = np.fmax(existing, new_pixels)  # fmax ignores NaN on one side
    # Pixels outside backbone (NaN in existing) should stay NaN
    patched = np.where(np.isnan(existing), np.nan, patched)

    added = int((patched == 1).sum()) - int((existing == 1).sum())
    print(f"  Net new protected pixels: {added:,}  (~{added} km²)")

    if dry_run:
        print("  [DRY-RUN] No files written.")
        return

    # Backup original
    backup = tif_path.with_suffix(".tif").parent / (tif_path.stem + "_pre_patch.tif")
    if not backup.exists():
        shutil.copy2(tif_path, backup)
        print(f"  Backed up original → {backup.name}")
    else:
        print(f"  Backup already exists: {backup.name}")

    # Write patched TIF in-place (same profile)
    profile.update(dtype="float32")
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(patched.astype("float32"), 1)

    print(f"  Written: {tif_path}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(years: list[int] = [2021, 2022, 2023, 2024]) -> None:
    print("\n=== Verification: protected pixel counts ===")
    prev = None
    for yr in years:
        tif = WDPA_DIR / f"WDPA_SEA_1km_{yr}.tif"
        if not tif.exists():
            continue
        with rasterio.open(tif) as src:
            d = src.read(1)
        n = int((d == 1).sum())
        delta = f" (+{n - prev:,})" if prev is not None else ""
        print(f"  {yr}: {n:>8,} protected pixels{delta}")
        prev = n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Patch WDPA SEA TIFs with May2026 IDN/THA data")
    parser.add_argument("--dry-run", action="store_true", help="Print stats, do not write files")
    args = parser.parse_args()

    print("=" * 60)
    print("WDPA SEA patch — May2026 shapefile → 2023/2024 TIFs")
    print(f"WDPA_DIR: {WDPA_DIR}")
    print(f"Dry-run:  {args.dry_run}")
    print("=" * 60)

    gdf_all = load_missing_polygons()

    for year in PATCH_YEARS:
        patch_year(gdf_all, year, dry_run=args.dry_run)

    verify()

    if not args.dry_run:
        print("\nDone. Next step: run SEA feature_engineering on Euler (Issue H).")
    else:
        print("\nDry-run complete. Re-run without --dry-run to apply patches.")


if __name__ == "__main__":
    main()
