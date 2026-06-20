"""KBA (Key Biodiversity Areas) Rasterisation — South America.

Creates is_kba (0/1) and dist_kba_km (distance to nearest KBA) aligned to the
SA backbone raster at 1 km resolution.

Input:
  data/shared/KBA/KBA_poly.shp  ← download from BirdLife International:
    https://www.keybiodiversityareas.org/kba-data/request
  (Look for the "KBA Digital Boundaries" / global KBA polygon shapefile)

Output:
  data/south_america/ready/KBA/kba_sa.tif  (2 bands, float32)
    Band 1: is_kba      — 1.0 inside a KBA, 0.0 outside
    Band 2: dist_kba_km — km to nearest KBA boundary; 0.0 if inside KBA

Usage:
  python scripts/regions/south_america/2_preprocessing/kba_rasterise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKBONE_PATH = PROJECT_ROOT / "data/south_america/ready/backbone/backbone.tif"
KBA_DIR = PROJECT_ROOT / "data/shared/KBA"
OUTPUT_PATH = PROJECT_ROOT / "data/south_america/ready/KBA/kba_sa.tif"

# Backbone pixel side is 1 000 m = 1 km, so EDT pixel units == km
PIXEL_KM = 1.0

# South America countries covered by this project (FRA for French Guiana)
SA_ISO3 = {
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU",
    "FRA", "GUY", "PRY", "PER", "SUR", "URY", "VEN",
}


def _find_kba_shapefile() -> Path:
    for pattern in ("KBA_poly*.shp", "kba_poly*.shp", "KBA*.shp", "kba*.shp"):
        matches = sorted(KBA_DIR.rglob(pattern))
        if matches:
            # prefer polygon files over point files
            pol = [m for m in matches if "POL" in m.name or "poly" in m.name.lower()]
            return pol[0] if pol else matches[0]
    raise FileNotFoundError(
        f"No KBA shapefile found in {KBA_DIR}.\n"
        "Download 'KBA Digital Boundaries' from:\n"
        "  https://www.keybiodiversityareas.org/kba-data/request\n"
        "and place as:  data/shared/KBA/KBA_poly.shp"
    )


def main() -> None:
    if not BACKBONE_PATH.exists():
        sys.exit(f"Backbone raster not found: {BACKBONE_PATH}")

    shp_path = _find_kba_shapefile()
    print(f"KBA shapefile: {shp_path}")

    with rasterio.open(BACKBONE_PATH) as src:
        profile = src.profile.copy()
        backbone = src.read(1)          # uint8: 1=land, 0=nodata/ocean
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        shape = (src.height, src.width)

    bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    print(f"SA grid: {shape[0]} × {shape[1]} pixels  CRS: {crs}")

    # Backbone CRS is LOCAL_CS (3857 metres); KBA file is WGS84 degrees.
    # Convert bbox to WGS84 for the spatial pre-filter on read.
    _t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = _t.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = _t.transform(bounds.right, bounds.top)
    bbox_wgs84 = box(lon_min, lat_min, lon_max, lat_max)

    # Spatial filter to SA bounding box (fast path on global file)
    try:
        gdf = gpd.read_file(shp_path, bbox=bbox_wgs84)
    except Exception:
        gdf = gpd.read_file(shp_path)

    if gdf.empty:
        sys.exit("KBA shapefile returned no features within SA bounding box.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gpd.clip(gdf, gpd.GeoDataFrame(geometry=[bbox_geom], crs=crs))

    if gdf.empty:
        sys.exit("No KBA features intersect the South America backbone extent.")

    print(f"KBA polygons clipped to SA: {len(gdf):,}")

    # Band 1: binary mask (all_touched=False → centre-of-pixel rule)
    is_kba = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(np.uint8)
    is_kba[backbone == 0] = 0          # zero out ocean / nodata pixels

    # Band 2: distance to nearest KBA boundary in km
    # distance_transform_edt(input): each 0-pixel gets distance to nearest non-0.
    # Using (1 - is_kba): inside KBA → 0 (zero-pixel, EDT=0); outside → 1 (get distance).
    dist_raw = distance_transform_edt(1 - is_kba)   # units = pixels = km for 1 km grid
    dist_kba_km = (dist_raw * PIXEL_KM).astype(np.float32)
    dist_kba_km[backbone == 0] = np.nan             # ocean → NaN

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
        dst.write(is_kba.astype(np.float32), 1)
        dst.write(dist_kba_km, 2)
        dst.update_tags(1, name="is_kba", description="1=inside KBA, 0=outside")
        dst.update_tags(2, name="dist_kba_km", description="km to nearest KBA boundary")

    n_land = int((backbone > 0).sum())
    n_kba = int(is_kba.sum())
    max_dist = float(dist_kba_km[backbone > 0].max())
    print(f"is_kba:      {n_kba:,} / {n_land:,} land pixels ({100*n_kba/max(n_land,1):.2f}%)")
    print(f"dist_kba_km: max = {max_dist:.1f} km")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
