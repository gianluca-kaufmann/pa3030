#!/usr/bin/env python3
"""Rasterise HydroSHEDS L7 catchments for South America to the SA backbone grid.

Produces a single-band INT32 raster where each pixel's value is the HYBAS_ID
of the HydroSHEDS Level 7 catchment it belongs to. Pixels outside any catchment
(ocean, nodata) get value 0.

Input:
  data/shared/HydroSHEDS/hybas_sa_lev07_v1c/hybas_sa_lev07_v1c.shp
  (download from https://www.hydrosheds.org/products/hydrobasins
   → "South America" → "Level 7" → HydroBASINS standard)

Output:
  data/south_america/ready/HydroSHEDS/hydrosheds_l7_sa.tif  (int32, 1 band)

Usage:
  python scripts/regions/south_america/2_preprocessing/hydrosheds_rasterise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import box

_ROOT          = Path(__file__).resolve().parents[4]
BACKBONE_PATH  = _ROOT / "data/south_america/ready/backbone/backbone.tif"
HYDRO_DIR      = _ROOT / "data/shared/HydroSHEDS"
OUTPUT_PATH    = _ROOT / "data/south_america/ready/HydroSHEDS/hydrosheds_l7_sa.tif"
HYBAS_ID_COL   = "HYBAS_ID"


def _find_shapefile() -> Path:
    for pattern in (
        "**/hybas_sa_lev07*.shp",
        "**/hybas_sa_lev07*.gpkg",
        "**/hybas_sa_lev06*.shp",  # L6 fallback
    ):
        hits = sorted(HYDRO_DIR.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"HydroSHEDS L7 SA shapefile not found under {HYDRO_DIR}.\n\n"
        "Download HydroBASINS South America Level 7 from:\n"
        "  https://www.hydrosheds.org/products/hydrobasins\n"
        "  → 'South America' → 'Level 7' (Standard) → Download\n\n"
        "Place files at:\n"
        "  data/shared/HydroSHEDS/hybas_sa_lev07_v1c/hybas_sa_lev07_v1c.shp\n\n"
        "Registration required (free)."
    )


def main() -> None:
    if not BACKBONE_PATH.exists():
        sys.exit(f"Backbone not found: {BACKBONE_PATH}")

    shp_path = _find_shapefile()
    print(f"HydroSHEDS shapefile: {shp_path}")

    with rasterio.open(BACKBONE_PATH) as src:
        profile   = src.profile.copy()
        backbone  = src.read(1)
        crs       = src.crs
        transform = src.transform
        bounds    = src.bounds
        shape     = (src.height, src.width)

    print(f"SA backbone: {shape}  CRS: {crs}")

    # Convert bbox to WGS84 for pre-filter
    _t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = _t.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = _t.transform(bounds.right, bounds.top)
    bbox_wgs84 = box(lon_min, lat_min, lon_max, lat_max)

    print("Reading HydroSHEDS shapefile (spatial pre-filter)...")
    try:
        gdf = gpd.read_file(shp_path, bbox=bbox_wgs84)
    except Exception:
        gdf = gpd.read_file(shp_path)

    if gdf.empty:
        sys.exit("HydroSHEDS returned no features within SA bounding box.")

    print(f"  {len(gdf):,} catchments loaded")

    if HYBAS_ID_COL not in gdf.columns:
        # Try to find the ID column
        id_cols = [c for c in gdf.columns if "HYBAS" in c.upper() or "ID" in c.upper()]
        if not id_cols:
            sys.exit(f"Column '{HYBAS_ID_COL}' not found. Available: {list(gdf.columns)}")
        HYBAS_ID_COL_USE = id_cols[0]
        print(f"  Using ID column: {HYBAS_ID_COL_USE}")
    else:
        HYBAS_ID_COL_USE = HYBAS_ID_COL

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    # HYBAS_ID values can be large int64 — rasterize needs int32-safe values.
    # Map HYBAS_ID → sequential int32 ID (1-based), save mapping separately.
    hybas_ids = gdf[HYBAS_ID_COL_USE].to_numpy()
    unique_ids = np.unique(hybas_ids)
    id_to_int  = {int(hid): int(i + 1) for i, hid in enumerate(unique_ids)}
    gdf["_burn_val"] = gdf[HYBAS_ID_COL_USE].map(id_to_int).fillna(0).astype(np.int32)

    print(f"  Rasterising {len(gdf):,} catchments ({len(unique_ids):,} unique IDs)...")
    catch_raster = rasterize(
        ((geom, val) for geom, val in zip(gdf.geometry, gdf["_burn_val"])),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    catch_raster[backbone == 0] = 0  # ocean / nodata → 0

    n_land     = int((backbone > 0).sum())
    n_assigned = int((catch_raster > 0).sum())
    n_catchments = int((catch_raster > 0).max()) if n_assigned > 0 else 0
    print(f"  Pixels assigned to a catchment: {n_assigned:,} / {n_land:,} land pixels"
          f" ({100*n_assigned/max(n_land,1):.1f}%)")
    print(f"  Unique catchment IDs in raster:  {len(np.unique(catch_raster[catch_raster > 0])):,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    with rasterio.open(OUTPUT_PATH, "w", **out_profile) as dst:
        dst.write(catch_raster, 1)
        dst.update_tags(
            1,
            name="catchment_id",
            description=f"Sequential int32 ID (1-based) mapping to HYBAS_ID. "
                        f"0 = outside SA or no catchment. "
                        f"Source: HydroSHEDS L7 {shp_path.name}",
        )

    # Save HYBAS_ID ↔ int32 mapping as JSON so downstream scripts can reverse-lookup
    import json
    mapping_path = OUTPUT_PATH.with_suffix(".id_mapping.json")
    # Store int32_id → HYBAS_ID (for reporting)
    int_to_hybas = {str(v): int(k) for k, v in id_to_int.items()}
    mapping_path.write_text(json.dumps({
        "int32_id_to_hybas_id": int_to_hybas,
        "n_catchments": len(unique_ids),
        "source_file": str(shp_path),
        "hybas_id_col": HYBAS_ID_COL_USE,
    }, indent=2))

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"ID mapping: {mapping_path}")
    print(f"\nNext step:")
    print(f"  python scripts/regions/south_america/3_merging/build_watershed_sample.py")


if __name__ == "__main__":
    main()
