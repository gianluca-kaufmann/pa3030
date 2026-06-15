"""Above-Ground Biomass (AGB) Rasterisation — South America.

Creates agb_tonne_ha aligned to the SA backbone raster at 1 km resolution.
High-biomass pixels (tropical forests) are REDD+ targets and face increasing
designation pressure as carbon revenue motives grow.

Uses ESA CCI Biomass v4.0 (2010 vintage — earliest available, minimises leakage
for our 2001-2013 training window).  Forest biomass changes slowly so the 2010
snapshot captures the structural signal without substantial temporal leakage.

Download:
  ESA CCI Biomass v4.0 — Above Ground Biomass (AGB), 2010
  https://climate.esa.int/en/projects/biomass/data/
  → https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v4.0/

  For South America you need the tiles covering lat bands -60 to +15 and
  lon bands -90 to -30 (geographic extent of SA).

  Registration at CEDA is free.  Download the 300m AGB tiles (GeoTIFF, .tif).
  Place all downloaded tiles in: data/shared/ESA_CCI_Biomass/

  Tile naming convention: ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv4.0.tif
  or regional tiles: N00E000 format

  Alternative (if CEDA access is slow):
  GlobBiomass product at the same resolution via:
  https://www.globbiomass.org/

Output:
  data/south_america/ready/AGB/agb_sa.tif  (1 band, float32)
    Band 1: agb_tonne_ha — above-ground biomass in tonnes per hectare
                           (NaN for ocean/nodata; ~0 for bare land)

Usage (after placing TIF tiles):
  python scripts/regions/south_america/2_preprocessing/agb_rasterise.py

Then inject (standard TIF workflow):
  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\
      --tif data/south_america/ready/AGB/agb_sa.tif \\
      --cols agb_tonne_ha
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKBONE_PATH = PROJECT_ROOT / "data/south_america/ready/backbone/backbone.tif"
AGB_DIR = PROJECT_ROOT / "data/shared/ESA_CCI_Biomass"
OUTPUT_PATH = PROJECT_ROOT / "data/south_america/ready/AGB/agb_sa.tif"


def _find_agb_tiles() -> list[Path]:
    patterns = ("*.tif", "*.TIF", "*.tiff", "*.TIFF")
    tiles = []
    for p in patterns:
        tiles.extend(AGB_DIR.glob(p))
    if not tiles:
        raise FileNotFoundError(
            f"No AGB GeoTIF tiles found in {AGB_DIR}.\n"
            "Download ESA CCI Biomass v4.0 (2010) from:\n"
            "  https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v4.0/\n"
            "Place all .tif tiles in: data/shared/ESA_CCI_Biomass/"
        )
    return sorted(tiles)


def main() -> None:
    if not BACKBONE_PATH.exists():
        sys.exit(f"Backbone raster not found: {BACKBONE_PATH}")

    tiles = _find_agb_tiles()
    print(f"Found {len(tiles)} AGB tile(s) in {AGB_DIR}")
    for t in tiles[:5]:
        print(f"  {t.name}")
    if len(tiles) > 5:
        print(f"  ... ({len(tiles) - 5} more)")

    # Load backbone metadata
    dst_crs = CRS.from_epsg(3857)
    with rasterio.open(BACKBONE_PATH) as src:
        backbone = src.read(1)
        dst_transform = src.transform
        dst_shape = (src.height, src.width)

    print(f"Target: {dst_shape[0]} × {dst_shape[1]}  CRS: {dst_crs}")

    # Merge all tiles into a single in-memory mosaic (in source CRS)
    # For large global datasets this may use significant memory; use SCRATCH or
    # process one tile at a time if needed.
    print("Merging AGB tiles...")
    if len(tiles) == 1:
        mosaic_path = tiles[0]
    else:
        datasets = [rasterio.open(t) for t in tiles]
        try:
            mosaic_arr, mosaic_transform = merge(
                datasets,
                method="first",
                nodata=None,
                res=None,
            )
        finally:
            for ds in datasets:
                ds.close()

        # Write merged mosaic to a temp file for reproject step
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()
        with rasterio.open(tiles[0]) as ref:
            ref_profile = ref.profile.copy()
        ref_profile.update(
            width=mosaic_arr.shape[2],
            height=mosaic_arr.shape[1],
            transform=mosaic_transform,
            count=1,
        )
        with rasterio.open(tmp_path, "w", **ref_profile) as dst:
            dst.write(mosaic_arr[0], 1)
        mosaic_path = tmp_path
        print(f"  Merged mosaic: {mosaic_arr.shape[1]} × {mosaic_arr.shape[2]}")

    # Reproject and resample to backbone grid (EPSG:3857, 1 km)
    print(f"Reprojecting to {dst_crs}...")
    agb_1km = np.full(dst_shape, np.nan, dtype=np.float32)

    with rasterio.open(mosaic_path) as src:
        src_nodata = src.nodata if src.nodata is not None else 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reproject(
                source=rasterio.band(src, 1),
                destination=agb_1km,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,  # average biomass across sub-pixels
            )

    # Clean up temp file if created
    if len(tiles) > 1 and "tmp_path" in dir():
        tmp_path.unlink(missing_ok=True)

    # Mask ocean / nodata pixels
    agb_1km[backbone == 0] = np.nan
    # Clamp negative values (artefact of resampling) to 0
    agb_1km = np.where(agb_1km < 0, 0.0, agb_1km)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "width": dst_shape[1],
        "height": dst_shape[0],
        "nodata": np.nan,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(OUTPUT_PATH, "w", **out_profile) as dst:
        dst.write(agb_1km, 1)
        dst.update_tags(1, name="agb_tonne_ha",
                        description="Above-ground biomass t/ha (ESA CCI Biomass v4.0, 2010)")

    valid = agb_1km[np.isfinite(agb_1km)]
    print(f"agb_tonne_ha: valid={len(valid):,}, mean={valid.mean():.1f}, "
          f"max={valid.max():.1f} t/ha")
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Next:")
    print("  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\")
    print("      --tif data/south_america/ready/AGB/agb_sa.tif \\")
    print("      --cols agb_tonne_ha")


if __name__ == "__main__":
    main()
