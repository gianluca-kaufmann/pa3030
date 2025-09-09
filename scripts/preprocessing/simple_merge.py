#!/usr/bin/env python3
"""
Simple and robust raster merge script.
Merges REE occurrence data onto the South America template.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

TEMPLATE_TIF = "viirs_pa_ciesin_population_south_america_10km.tif"
SOURCE_TIF   = "REE_occurrence_count_025deg.tif"
OUT_TIF      = "south_america_with_ree.tif"

def main():
    print("=== RASTER MERGE ===")
    
    # Load template
    print("Loading template...")
    with rasterio.open(TEMPLATE_TIF) as template:
        template_profile = template.profile.copy()
        template_data = template.read()  # All bands
        print(f"Template: {template.width}x{template.height}, {template.count} bands")
        print(f"Template bounds: {template.bounds}")
        print(f"Template CRS: {template.crs}")
        
    # Load source
    print("Loading source...")
    with rasterio.open(SOURCE_TIF) as source:
        source_data = source.read(1)  # Single band
        source_profile = source.profile
        print(f"Source: {source.width}x{source.height}, bounds: {source.bounds}")
        print(f"Source CRS: {source.crs}")
        print(f"Source nodata: {source.nodata}")
        
    # Create destination array for reprojection
    dest = np.full((template.height, template.width), np.nan, dtype=np.float32)
    
    print("Reprojecting source to template grid...")
    reproject(
        source=source_data.astype(np.float32),
        destination=dest,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=template.transform,
        dst_crs=template.crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest
    )
    
    # Convert source nodata (0) to NaN to match template
    if source.nodata is not None:
        dest = np.where(dest == source.nodata, np.nan, dest)
    
    print(f"Reprojected data - Valid pixels: {np.sum(~np.isnan(dest))}")
    print(f"Reprojected data range: {np.nanmin(dest):.1f} to {np.nanmax(dest):.1f}")
    
    # Stack the data
    print("Stacking bands...")
    stacked = np.vstack([template_data, dest[np.newaxis, ...]])
    
    # Update profile for output
    out_profile = template_profile.copy()
    out_profile.update({
        "count": template.count + 1,
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
        "tiled": True
    })
    
    # Write output
    print(f"Writing {OUT_TIF}...")
    with rasterio.open(OUT_TIF, "w", **out_profile) as dst:
        dst.write(stacked)
        
        # Add band descriptions
        for i in range(1, template.count + 1):
            dst.set_band_description(i, f"Template_Band_{i}")
        dst.set_band_description(template.count + 1, "REE_occurrence_count")
    
    print("Done!")
    
    # Verify output
    print("\n=== VERIFICATION ===")
    with rasterio.open(OUT_TIF) as dst:
        print(f"Output: {dst.width}x{dst.height}, {dst.count} bands")
        print(f"Output bounds: {dst.bounds}")
        print(f"Output CRS: {dst.crs}")
        
        # Check each band
        for i in range(1, dst.count + 1):
            band_data = dst.read(i)
            valid_mask = ~np.isnan(band_data)
            if np.any(valid_mask):
                valid_data = band_data[valid_mask]
                print(f"Band {i}: {np.min(valid_data):.3f} to {np.max(valid_data):.3f} ({np.sum(valid_mask)} valid pixels)")
            else:
                print(f"Band {i}: No valid data")

if __name__ == "__main__":
    main()
