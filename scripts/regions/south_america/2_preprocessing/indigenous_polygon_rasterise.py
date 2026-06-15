"""Indigenous Territory Polygon Rasterisation — South America.

Creates in_indigenous_poly (0/1) and dist_indigenous_poly_km aligned to the SA
backbone raster.  Uses POLYGON boundaries (RAISG AMAZONAS) rather than the
existing point-based dist_indigenous feature, giving broader and more accurate
coverage of indigenous territories.

Many Colombia PAs are designated over resguardos (indigenous reserves) that
were subsequently converted to national parks.  This feature encodes that
mechanism as a direct spatial signal.

Download:
  RAISG Amazon Socioenvironmental Map (Amazonía Socioambiental 2022)
  → Territories of Indigenous Peoples in Legal Amazon
  https://www.amazoniasocioambiental.org/es/mapas/#!/type=download
  - File: "Territorios Indígenas" (Indigenous Territories) GeoPackage or shapefile
  - Place as: data/shared/RAISG/indigenous_territories.shp   (or .gpkg)

  Alternative (covers all SA, not just Amazon basin):
  LandMark Global Platform of Indigenous and Community Lands
  https://www.landmarkmap.org/data/
  - File: Formal Recognition — "Formally Recognized" layer for South America
  - Place as: data/shared/RAISG/indigenous_territories.shp

Output:
  data/south_america/ready/indigenous_territories/indigenous_sa.tif  (2 bands, float32)
    Band 1: in_indigenous_poly  — 1.0 inside a territory, 0.0 outside
    Band 2: dist_indigenous_poly_km — km to nearest territory boundary; 0.0 if inside

Usage (after placing shapefile):
  python scripts/regions/south_america/2_preprocessing/indigenous_polygon_rasterise.py

Then inject (standard TIF workflow):
  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\
      --tif data/south_america/ready/indigenous_territories/indigenous_sa.tif \\
      --cols in_indigenous_poly dist_indigenous_poly_km
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKBONE_PATH = PROJECT_ROOT / "data/south_america/ready/backbone/backbone.tif"
RAISG_DIR = PROJECT_ROOT / "data/shared/RAISG"
OUTPUT_PATH = PROJECT_ROOT / "data/south_america/ready/indigenous_territories/indigenous_sa.tif"

PIXEL_KM = 1.0  # 1 km resolution → EDT units == km

# South America ISO3 codes
SA_ISO3 = {
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU",
    "FRA", "GUY", "PRY", "PER", "SUR", "URY", "VEN",
}


def _find_shapefile() -> Path:
    for pattern in (
        "indigenous_territories.*",
        "Territorios_Indigenas*",
        "territorios_indigenas*",
        "TI_*.shp",
        "ti_*.shp",
        "*.gpkg",
        "*.shp",
    ):
        matches = sorted(RAISG_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No RAISG shapefile found in {RAISG_DIR}.\n"
        "Download 'Territorios Indígenas' from:\n"
        "  https://www.amazoniasocioambiental.org/es/mapas/#!/type=download\n"
        "and place as:  data/shared/RAISG/indigenous_territories.shp"
    )


def main() -> None:
    if not BACKBONE_PATH.exists():
        sys.exit(f"Backbone raster not found: {BACKBONE_PATH}")

    shp_path = _find_shapefile()
    print(f"RAISG shapefile: {shp_path}")

    with rasterio.open(BACKBONE_PATH) as src:
        profile = src.profile.copy()
        backbone = src.read(1)
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        shape = (src.height, src.width)

    bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    print(f"SA grid: {shape[0]} × {shape[1]} pixels  CRS: {crs}")

    try:
        gdf = gpd.read_file(shp_path, bbox=bbox_geom)
    except Exception:
        gdf = gpd.read_file(shp_path)

    if gdf.empty:
        sys.exit("Shapefile returned no features within SA bounding box.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    try:
        gdf = gpd.clip(gdf, gpd.GeoDataFrame(geometry=[bbox_geom], crs=crs))
    except Exception:
        pass

    if gdf.empty:
        sys.exit("No features intersect the South America backbone extent.")

    print(f"Indigenous territory polygons clipped to SA: {len(gdf):,}")

    # Band 1: binary rasterisation
    in_poly = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(np.uint8)
    in_poly[backbone == 0] = 0

    # Band 2: distance to nearest territory boundary in km
    dist_raw = distance_transform_edt(1 - in_poly)
    dist_km = (dist_raw * PIXEL_KM).astype(np.float32)
    dist_km[backbone == 0] = np.nan

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=2,
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rasterio.open(OUTPUT_PATH, "w", **out_profile) as dst:
        dst.write(in_poly.astype(np.float32), 1)
        dst.write(dist_km, 2)
        dst.update_tags(1, name="in_indigenous_poly",
                        description="1=inside RAISG indigenous territory, 0=outside")
        dst.update_tags(2, name="dist_indigenous_poly_km",
                        description="km to nearest RAISG indigenous territory boundary")

    n_land = int((backbone > 0).sum())
    n_inside = int(in_poly.sum())
    max_dist = float(dist_km[backbone > 0].max())
    print(f"in_indigenous_poly:    {n_inside:,} / {n_land:,} pixels ({100*n_inside/max(n_land,1):.2f}%)")
    print(f"dist_indigenous_poly_km: max = {max_dist:.1f} km")
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Next:")
    print("  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\")
    print("      --tif data/south_america/ready/indigenous_territories/indigenous_sa.tif \\")
    print("      --cols in_indigenous_poly dist_indigenous_poly_km")


if __name__ == "__main__":
    main()
