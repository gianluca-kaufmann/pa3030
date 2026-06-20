"""HydroSHEDS Level-7 Basin Rasterisation — South America.

Reads hybas_sa_lev07_v1c.shp directly from the local zip archive,
reprojects from WGS84 to the SA backbone CRS (EPSG:3857), and burns
a 1-based compact integer catchment index to the backbone grid.

Input:
  data/south_america/ready/hydroshed/hybas_sa_lev01-12_v1c.zip

Outputs:
  data/south_america/ready/hydroshed/catchment_id_sa.tif
    int32 single-band raster; value = 1-based catchment index; 0 = no data
  data/south_america/ready/hydroshed/catchment_id_mapping.json
    {"1": 6070000010, "2": 6070000020, ...}  (compact index → HYBAS_ID)

Usage:
  python scripts/regions/south_america/2_preprocessing/hydroshed_rasterise.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parents[4]
HYDROSHED_DIR = PROJECT_ROOT / "data/south_america/ready/hydroshed"
ZIP_PATH = HYDROSHED_DIR / "hybas_sa_lev01-12_v1c.zip"
BACKBONE_PATH = PROJECT_ROOT / "data/south_america/ready/backbone/backbone.tif"
OUT_TIF = HYDROSHED_DIR / "catchment_id_sa.tif"
OUT_MAP = HYDROSHED_DIR / "catchment_id_mapping.json"

L07_FILES = [
    "hybas_sa_lev07_v1c.shp",
    "hybas_sa_lev07_v1c.dbf",
    "hybas_sa_lev07_v1c.prj",
    "hybas_sa_lev07_v1c.shx",
    "hybas_sa_lev07_v1c.sbn",
    "hybas_sa_lev07_v1c.sbx",
]


def main() -> None:
    if not ZIP_PATH.exists():
        sys.exit(f"HydroSHEDS zip not found: {ZIP_PATH}")
    if not BACKBONE_PATH.exists():
        sys.exit(f"Backbone raster not found: {BACKBONE_PATH}")

    with rasterio.open(BACKBONE_PATH) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)

    print(f"Backbone grid: {shape[0]} × {shape[1]} pixels")
    print("Reading L07 shapefile from zip...")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(ZIP_PATH) as z:
            available = set(z.namelist())
            for name in L07_FILES:
                if name in available:
                    z.extract(name, tmp)
        shp_path = Path(tmp) / "hybas_sa_lev07_v1c.shp"
        gdf = gpd.read_file(shp_path)

    print(f"  {len(gdf):,} catchments  CRS: {gdf.crs}")
    gdf = gdf.sort_values("HYBAS_ID").reset_index(drop=True)
    gdf = gdf.to_crs(epsg=3857)

    # Assign compact 1-based index
    gdf["catchment_idx"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    id_map = {str(int(row.catchment_idx)): int(row.HYBAS_ID) for row in gdf.itertuples()}

    print(f"  Rasterising {len(gdf):,} catchments...")
    catchment_raster = rasterize(
        ((geom, idx) for geom, idx in zip(gdf.geometry, gdf["catchment_idx"])),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )

    covered = int((catchment_raster > 0).sum())
    print(f"  Covered pixels: {covered:,} / {shape[0]*shape[1]:,} "
          f"({100*covered/(shape[0]*shape[1]):.1f}%)")

    out_profile = profile.copy()
    out_profile.update(
        dtype="int32",
        count=1,
        nodata=0,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    HYDROSHED_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_TIF, "w", **out_profile) as dst:
        dst.write(catchment_raster, 1)
        dst.update_tags(1, description="1-based HydroSHEDS L07 catchment index")

    with open(OUT_MAP, "w") as f:
        json.dump(id_map, f)

    print(f"TIF  → {OUT_TIF}")
    print(f"Map  → {OUT_MAP}")


if __name__ == "__main__":
    main()
