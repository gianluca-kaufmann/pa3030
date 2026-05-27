#!/usr/bin/env python3
"""
Patch WDPA_SA_1km_2023.tif and WDPA_SA_1km_2024.tif with missing SA designations
from WDPA May2026 shapefile.

Background: The GEE WDPA export captured ~78% of SA 2023 terrestrial designations
and ~68% of SA 2024 terrestrial designations. The May2026 shapefile fills the gap.

Diagnosed gaps (from May2026 shapefile vs existing TIF, STATUS==Designated):
  2023 TIF: missing ~2,347 terrestrial pixels (22% of 2023 SA designations)
  2024 TIF: missing ~2,450 terrestrial pixels cumulative (103 additional 2024 pixels above 2023)

Source countries with 2023-2024 missing data:
  BRA, CHL, COL, CRI, ECU, MEX, PER, PRY (from May2026 shapefile scan)

SA 2024 is excluded from primary metric (D²_7yr = 2017-2023).
SA 2023 IS in the primary metric — patching is paper-critical for Stage 1 target integrity.

Patch should be applied AFTER the current SA chain (689639/689640) completes to avoid
wasted computation. SA FE must then be re-run on Euler, then Stage 1 re-built locally,
then Stage 2 re-tuned/re-trained.

Usage (run locally or on Euler — paths auto-detected via $SCRATCH):
    python scripts/regions/south_america/2_preprocessing/patch_wdpa_sa.py

    # Dry-run: print stats without writing
    python scripts/regions/south_america/2_preprocessing/patch_wdpa_sa.py --dry-run
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent

if "SCRATCH" in os.environ:
    DATA_ROOT = Path(os.environ["SCRATCH"]) / "data"
else:
    DATA_ROOT = PROJECT_ROOT / "data"

WDPA_DIR = DATA_ROOT / "south_america" / "ready" / "WDPA"
SHAPEFILE_PARTS = [
    PROJECT_ROOT / "data" / "shared" / "WDPA" / f"WDPA_May2026_Public_shp_{i}"
    / f"WDPA_May2026_Public_shp-polygons.shp"
    for i in range(3)
]

# All SA countries in the panel
SA_ISO3 = {
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU",
    "GUY", "PER", "PRY", "SUR", "URY", "VEN",
    # Central America countries included in SA backbone:
    "PAN", "CRI", "GTM", "HND", "MEX", "NIC", "SLV",
}

PATCH_YEARS = [2023, 2024]


# ---------------------------------------------------------------------------
# Load missing polygons
# ---------------------------------------------------------------------------
def load_missing_polygons() -> gpd.GeoDataFrame:
    """Load all SA Designated polygons with STATUS_YR in PATCH_YEARS from shapefile."""
    parts = []
    for shp in SHAPEFILE_PARTS:
        if not shp.exists():
            print(f"  [WARN] Shapefile part not found: {shp}")
            continue
        gdf = gpd.read_file(shp)
        mask = (
            gdf["ISO3"].isin(SA_ISO3)
            & gdf["STATUS_YR"].isin(PATCH_YEARS)
            & (gdf["STATUS"] == "Designated")
        )
        parts.append(gdf[mask].copy())

    if not parts:
        sys.exit("ERROR: No shapefile parts found. Check SHAPEFILE_PARTS paths.")

    combined = pd.concat(parts, ignore_index=True)
    print(f"\nLoaded {len(combined)} missing SA Designated polygons")
    print(combined.groupby(["ISO3", "STATUS_YR"])["GIS_AREA"].agg(["count", "sum"]).round(0))
    return gpd.GeoDataFrame(combined, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Rasterize and patch
# ---------------------------------------------------------------------------
def patch_year(gdf_all: gpd.GeoDataFrame, year: int, dry_run: bool) -> None:
    tif_path = WDPA_DIR / f"WDPA_SA_1km_{year}.tif"
    if not tif_path.exists():
        print(f"  [SKIP] {tif_path} not found")
        return

    gdf = gdf_all[gdf_all["STATUS_YR"] <= year].copy()
    if len(gdf) == 0:
        print(f"  [SKIP] No new polygons for year {year}")
        return

    print(f"\n--- Patching {tif_path.name} ---")
    print(f"  Polygons to burn: {len(gdf)}")
    print(f"  GIS_AREA total:   {gdf['GIS_AREA'].sum():,.0f} km²")

    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
        existing = src.read(1)

    gdf_proj = gdf.to_crs("EPSG:3857")

    new_pixels = rasterize(
        [(geom, 1.0) for geom in gdf_proj.geometry],
        out_shape=shape,
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=False,
    )

    burned = int((new_pixels == 1).sum())
    nan_burned = int(np.isnan(existing[new_pixels == 1]).sum())
    already = int(((existing == 1) & (new_pixels == 1)).sum())
    genuinely_new = burned - nan_burned - already
    print(f"  Pixels burned total:       {burned:,}")
    print(f"    outside backbone (NaN):  {nan_burned:,}")
    print(f"    already protected:       {already:,}")
    print(f"    genuinely new land px:   {genuinely_new:,}")

    # OR-merge: preserve NaN backbone pixels
    patched = np.fmax(existing, new_pixels)
    patched = np.where(np.isnan(existing), np.nan, patched)

    if dry_run:
        print("  [DRY-RUN] No files written.")
        return

    backup = tif_path.parent / (tif_path.stem + "_pre_patch.tif")
    if not backup.exists():
        shutil.copy2(tif_path, backup)
        print(f"  Backed up original → {backup.name}")
    else:
        print(f"  Backup already exists: {backup.name}")

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
        tif = WDPA_DIR / f"WDPA_SA_1km_{yr}.tif"
        if not tif.exists():
            continue
        with rasterio.open(tif) as src:
            d = src.read(1)
        n = int((d == 1).sum())
        delta = f" (+{n - prev:,})" if prev is not None else ""
        print(f"  {yr}: {n:>10,} protected pixels{delta}")
        prev = n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Patch WDPA SA TIFs with May2026 data")
    parser.add_argument("--dry-run", action="store_true", help="Print stats, do not write files")
    args = parser.parse_args()

    print("=" * 60)
    print("WDPA SA patch — May2026 shapefile → 2023/2024 TIFs")
    print(f"WDPA_DIR: {WDPA_DIR}")
    print(f"Dry-run:  {args.dry_run}")
    print("=" * 60)

    gdf_all = load_missing_polygons()

    for year in PATCH_YEARS:
        patch_year(gdf_all, year, dry_run=args.dry_run)

    verify()

    if not args.dry_run:
        print("\nDone. Next steps:")
        print("  1. scp WDPA_SA_1km_2023.tif + _2024.tif to Euler $SCRATCH/data/south_america/ready/WDPA/")
        print("  2. Re-run SA feature_engineering on Euler (rebuilds pixel panel with correct 2023/2024 labels)")
        print("  3. Re-run stage1_data_builder.py on Euler → download stage1_panel.parquet")
        print("  4. Re-run model1_expansion.py locally (Stage 1 final numbers)")
        print("  5. Re-queue SA Stage 2 tune+retrain chain")
    else:
        print("\nDry-run complete. Re-run without --dry-run to apply patches.")


if __name__ == "__main__":
    main()
