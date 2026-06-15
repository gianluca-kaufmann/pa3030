"""REDD+ Project Rasterisation — South America.

Creates dist_redd_km aligned to the SA backbone raster at 1 km resolution.
REDD+ projects create financial incentives for governments to formally protect
carbon-rich forest, so proximity to a REDD+ project is a designation predictor.

Data source: ID-RECCO V5.0 database (tabular, point centroids).
Download: https://www.reddprojectsdatabase.org/download-the-the-id-recco-database/
Place zip as: data/REDD/ID-RECCO V5.0_20231201.zip  (already downloaded)

Note: ID-RECCO provides centroid coordinates (lat/lon), not polygon boundaries.
This script produces a single distance-to-centroid raster.  If polygon shapefiles
become available (e.g. from Global Forest Watch), drop them in data/REDD/*.shp
and the script will prefer polygons over points automatically.

Output:
  data/south_america/ready/REDD/redd_sa.tif  (1 band, float32)
    Band 1: dist_redd_km — km to nearest REDD+ project centroid (0 at centroid)

Usage:
  python scripts/regions/south_america/2_preprocessing/redd_rasterise.py

Then inject:
  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\
      --tif data/south_america/ready/REDD/redd_sa.tif --cols dist_redd_km
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKBONE_PATH = PROJECT_ROOT / "data/south_america/ready/backbone/backbone.tif"
REDD_DIR = PROJECT_ROOT / "data/REDD"
OUTPUT_PATH = PROJECT_ROOT / "data/south_america/ready/REDD/redd_sa.tif"

PIXEL_KM = 1.0

SA_COUNTRIES = {
    "Brazil", "Colombia", "Peru", "Ecuador", "Bolivia", "Venezuela",
    "Chile", "Argentina", "Paraguay", "Uruguay", "Guyana", "Suriname",
    "French Guiana",
}


def _load_from_shapefile(redd_dir: Path, bbox_geom, crs) -> gpd.GeoDataFrame | None:
    """Try to load polygon shapefiles — preferred over points if available."""
    for pattern in ("*.shp", "*.gpkg", "*.geojson"):
        matches = sorted(redd_dir.glob(pattern))
        if matches:
            print(f"Found polygon file: {matches[0].name} — using polygon boundaries")
            try:
                gdf = gpd.read_file(matches[0], bbox=bbox_geom)
            except Exception:
                gdf = gpd.read_file(matches[0])
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(crs)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
            return gdf
    return None


def _load_from_xlsx(redd_dir: Path) -> gpd.GeoDataFrame | None:
    """Load project centroids from ID-RECCO xlsx."""
    import zipfile

    zip_candidates = sorted(redd_dir.glob("*.zip")) + sorted(redd_dir.glob("*.xlsx"))
    if not zip_candidates:
        return None

    try:
        import openpyxl
    except ImportError:
        print("WARNING: openpyxl not installed. Install with: pip install openpyxl")
        return None

    xlsx_path = None
    tmp_dir = Path("/tmp/redd_idrecco")
    tmp_dir.mkdir(exist_ok=True)

    for candidate in zip_candidates:
        if candidate.suffix == ".zip":
            with zipfile.ZipFile(candidate) as z:
                xlsx_names = [n for n in z.namelist() if "project" in n.lower() and n.endswith(".xlsx")]
                if xlsx_names:
                    z.extract(xlsx_names[0], tmp_dir)
                    xlsx_path = tmp_dir / xlsx_names[0]
                    break
        elif candidate.suffix == ".xlsx":
            xlsx_path = candidate
            break

    if xlsx_path is None:
        return None

    print(f"Loading ID-RECCO from: {xlsx_path.name}")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["01_Projects"]
    rows = list(ws.rows)
    headers = [c.value for c in rows[1]]

    lon_i = headers.index("Longitude")
    lat_i = headers.index("Latitude")
    country_i = headers.index("country name")
    name_i = headers.index("project_name")

    points, names = [], []
    for r in rows[3:]:
        vals = [c.value for c in r]
        if vals[country_i] not in SA_COUNTRIES:
            continue
        try:
            lon, lat = float(vals[lon_i]), float(vals[lat_i])
            points.append(Point(lon, lat))
            names.append(vals[name_i])
        except (TypeError, ValueError):
            continue

    if not points:
        return None

    gdf = gpd.GeoDataFrame({"name": names, "geometry": points}, crs="EPSG:4326")
    print(f"  {len(gdf)} SA REDD+ project centroids loaded")
    return gdf


def main() -> None:
    if not BACKBONE_PATH.exists():
        sys.exit(f"Backbone not found: {BACKBONE_PATH}")

    with rasterio.open(BACKBONE_PATH) as src:
        profile = src.profile.copy()
        backbone = src.read(1)
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        shape = (src.height, src.width)

    bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Try polygon shapefiles first; fall back to xlsx centroids
    gdf = _load_from_shapefile(REDD_DIR, bbox_geom, crs)
    use_polygons = gdf is not None

    if gdf is None:
        gdf = _load_from_xlsx(REDD_DIR)
        if gdf is None:
            sys.exit(
                f"No REDD+ data found in {REDD_DIR}.\n"
                "Expected: data/REDD/ID-RECCO V5.0_20231201.zip  (already downloaded)"
            )
        gdf = gdf.to_crs(epsg=3857)
        try:
            gdf = gpd.clip(gdf, gpd.GeoDataFrame(geometry=[bbox_geom], crs=crs))
        except Exception:
            pass

    print(f"REDD+ features in SA: {len(gdf)} ({'polygons' if use_polygons else 'centroids'})")

    # Rasterise: for polygons → burn 1; for points → burn 1 at nearest pixel
    burned = rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=not use_polygons,  # all_touched=True for point rasterisation
    ).astype(np.uint8)
    burned[backbone == 0] = 0

    # Distance to nearest REDD+ feature in km
    dist_raw = distance_transform_edt(1 - burned)
    dist_km = (dist_raw * PIXEL_KM).astype(np.float32)
    dist_km[backbone == 0] = np.nan

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32", count=1, nodata=np.nan,
        compress="lzw", tiled=True, blockxsize=512, blockysize=512,
    )
    with rasterio.open(OUTPUT_PATH, "w", **out_profile) as dst:
        dst.write(dist_km, 1)
        label = "polygon boundary" if use_polygons else "project centroid"
        dst.update_tags(1, name="dist_redd_km",
                        description=f"km to nearest REDD+ project {label}")

    valid = dist_km[np.isfinite(dist_km)]
    print(f"dist_redd_km: min={valid.min():.0f}, mean={valid.mean():.0f}, max={valid.max():.0f} km")
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Next:")
    print("  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\")
    print("      --tif data/south_america/ready/REDD/redd_sa.tif --cols dist_redd_km")


if __name__ == "__main__":
    main()
