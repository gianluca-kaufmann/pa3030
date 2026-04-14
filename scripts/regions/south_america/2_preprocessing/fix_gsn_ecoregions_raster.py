#!/usr/bin/env python3
"""
Fix SA GSN Terrestrial Ecoregions Raster
=========================================

The existing raster (gsn_terrestrial_ecoregions_mask_1km.tif) was generated with
a broken encoding: raster_value = (sorted_SA_index) * 4 (int16, multiples of 4).

The correct encoding expected by load_biome_raster_and_mapping() is:
    raster_value = sorted_SA_index + 1 (1-based, uint8)

This is also how the USA and SE Asia rasters were generated.

This script regenerates ONLY the ecoregions raster with the correct encoding.
The backbone CRS is LOCAL_CS (not proper EPSG:3857), so we use EPSG:3857
directly for reprojection (same coordinate values, proper EPSG code for pyproj).

Run from the repo root:
    conda activate pa3030
    python scripts/regions/south_america/2_preprocessing/fix_gsn_ecoregions_raster.py
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKBONE  = REPO_ROOT / "data/south_america/ready/backbone/backbone.tif"
SHP_PATH  = REPO_ROOT / "data/shared/GlobalSafetyNet/terrestrial_ecoregions/Terrestrial_ecoregions.shp"
OUT_PATH  = REPO_ROOT / "data/south_america/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif"


def main() -> None:
    print("=== Fixing SA GSN Terrestrial Ecoregions Raster ===\n")

    # 1. Read backbone grid parameters
    print(f"Reading backbone: {BACKBONE}")
    with rasterio.open(BACKBONE) as src:
        transform = src.transform
        width     = src.width
        height    = src.height
        bounds    = src.bounds
    print(f"  Grid: {height} x {width}, bounds: {bounds}\n")

    # 2. Load and clip shapefile to SA extent
    # Use EPSG:3857 (same coordinate space as LOCAL_CS backbone)
    print(f"Loading shapefile: {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:3857")
    gdf["geometry"] = gdf.geometry.make_valid()
    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    gdf = gpd.clip(gdf, bbox)
    print(f"  Ecoregions in SA extent: {gdf['ECO_NAME'].dropna().nunique()}\n")

    # 3. Build encoding: sorted ECO_NAME → 1-based integer (same as USA/SE Asia)
    unique_eco = sorted(gdf["ECO_NAME"].dropna().unique())
    eco_to_int = {name: i + 1 for i, name in enumerate(unique_eco)}
    print(f"  Encoding: {len(unique_eco)} ecoregions → values 1 to {len(unique_eco)}")
    print(f"  Fits in uint8: {len(unique_eco) <= 255}\n")

    # 4. Rasterize
    print("Rasterizing...")
    shapes = [
        (row.geometry, eco_to_int[row["ECO_NAME"]])
        for _, row in gdf.iterrows()
        if row.geometry is not None
        and not row.geometry.is_empty
        and row["ECO_NAME"] in eco_to_int
    ]
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    covered = int((mask > 0).sum())
    unique_vals = np.unique(mask[mask > 0])
    print(f"  Covered pixels:  {covered:,}")
    print(f"  Unique codes:    {len(unique_vals)}")
    print(f"  Value range:     {unique_vals.min()} – {unique_vals.max()}\n")

    # 5. Save with EPSG:3857 CRS
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width":  width,
        "count":  1,
        "dtype":  "uint8",
        "crs":    "EPSG:3857",
        "transform": transform,
        "compress":  "deflate",
        "predictor": 2,
        "zlevel":    6,
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }
    with rasterio.open(OUT_PATH, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"Saved: {OUT_PATH}")
    print("\nDone. Next steps:")
    print("  1. Re-run spatial_CV_2 for South America:")
    print("     python scripts/regions/south_america/6_evaluation/spatial_CV_2")
    print("  2. Regenerate Figure 7:")
    print("     python scripts/regions/south_america/7_results/model1_results")


if __name__ == "__main__":
    main()
