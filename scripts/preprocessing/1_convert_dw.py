#!/usr/bin/env python3
"""
DynamicWorld Probability to Discrete Class Conversion

This script converts DynamicWorld probability data to discrete land cover classes
by selecting the band with the highest probability for each pixel.

Input: GeoTIFFs with 9 bands per pixel (each band = class probability in [0,1])
Output: GeoTIFFs with 9 bands per pixel (one band = 1, others = 0)
"""

import numpy as np
import rasterio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
data_dir = Path("/Users/gianluca/Desktop/Master's Thesis/code/data")
output_dir = Path("/Users/gianluca/Desktop/Master's Thesis/code/data")
output_dir.mkdir(parents=True, exist_ok=True)

# DynamicWorld files
dw_files = {
    2012: data_dir / "DW_2012.tif",
    2017: data_dir / "DW_2017.tif", 
    2022: data_dir / "DW_2022.tif"
}

# Output file names
output_files = {
    2012: output_dir / "DW_2012_discrete.tif",
    2017: output_dir / "DW_2017_discrete.tif",
    2022: output_dir / "DW_2022_discrete.tif"
}

def examine_file_structure(file_path):
    """Examine the structure of a DynamicWorld file"""
    print(f"\n🔍 Examining {file_path.name}...")
    
    try:
        with rasterio.open(file_path) as src:
            print(f"   Shape: {src.shape}")
            print(f"   Bands: {src.count}")
            print(f"   Data type: {src.dtypes[0]}")
            print(f"   CRS: {src.crs}")
            print(f"   Transform: {src.transform}")
            print(f"   No data value: {src.nodata}")
            
            # Read a small sample to check data structure
            sample_data = src.read(1, window=((0, 100), (0, 100)))
            print(f"   Sample data shape: {sample_data.shape}")
            print(f"   Sample data type: {sample_data.dtype}")
            print(f"   Sample min/max: {sample_data.min():.6f} / {sample_data.max():.6f}")
            
            return {
                'shape': src.shape,
                'count': src.count,
                'dtype': src.dtypes[0],
                'crs': src.crs,
                'transform': src.transform,
                'nodata': src.nodata
            }
            
    except Exception as e:
        print(f"   ❌ Error examining {file_path.name}: {e}")
        return None

def convert_probabilities_to_discrete(input_file, output_file, year):
    """Convert probability data to discrete classes"""
    print(f"\n🔄 Converting {input_file.name} to discrete classes...")
    
    try:
        with rasterio.open(input_file) as src:
            # Read all bands
            print(f"   Reading {src.count} bands...")
            all_bands = src.read()  # Shape: (bands, height, width)
            
            print(f"   Data shape: {all_bands.shape}")
            print(f"   Data type: {all_bands.dtype}")
            
            # Handle NaN values - replace with 0
            nan_mask = np.isnan(all_bands)
            all_bands_clean = np.where(nan_mask, 0, all_bands)
            
            print(f"   NaN pixels replaced: {np.sum(nan_mask):,}")
            
            # Find the band with maximum probability for each pixel
            print("   Finding maximum probability bands...")
            max_band_indices = np.argmax(all_bands_clean, axis=0)  # Shape: (height, width)
            
            # Create discrete output array
            print("   Creating discrete output...")
            discrete_bands = np.zeros_like(all_bands_clean, dtype=np.uint8)
            
            # For each pixel, set the band with max probability to 1
            for band_idx in range(all_bands.shape[0]):
                mask = (max_band_indices == band_idx)
                discrete_bands[band_idx, mask] = 1
            
            # Verify the conversion
            print("   Verifying conversion...")
            for band_idx in range(all_bands.shape[0]):
                band_sum = np.sum(discrete_bands[band_idx])
                print(f"     Band {band_idx}: {band_sum:,} pixels set to 1")
            
            # Check that each pixel has exactly one band set to 1
            pixel_sums = np.sum(discrete_bands, axis=0)
            valid_pixels = np.sum(pixel_sums == 1)
            invalid_pixels = np.sum(pixel_sums != 1)
            
            print(f"   Valid pixels (exactly 1 band = 1): {valid_pixels:,}")
            print(f"   Invalid pixels: {invalid_pixels:,}")
            
            if invalid_pixels > 0:
                print(f"   ⚠️  Warning: {invalid_pixels} pixels don't have exactly one band set to 1")
            
            # Write output file
            print(f"   Writing output to {output_file.name}...")
            
            # Prepare metadata
            profile = src.profile.copy()
            profile.update({
                'dtype': 'uint8',
                'nodata': 0,  # Set nodata to 0 for discrete data
                'compress': 'lzw'  # Add compression to reduce file size
            })
            
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(discrete_bands)
                
            print(f"   ✅ Successfully saved: {output_file.name}")
            
            # Verify output file
            print("   Verifying output file...")
            with rasterio.open(output_file) as verify_src:
                verify_data = verify_src.read()
                print(f"     Output shape: {verify_data.shape}")
                print(f"     Output dtype: {verify_data.dtype}")
                print(f"     Output min/max: {verify_data.min()} / {verify_data.max()}")
                
                # Check that values are only 0 or 1
                unique_vals = np.unique(verify_data)
                if set(unique_vals).issubset({0, 1}):
                    print("     ✅ Output contains only 0 and 1 values")
                else:
                    print(f"     ❌ Output contains unexpected values: {unique_vals}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error converting {input_file.name}: {e}")
        return False

def main():
    """Main conversion function"""
    print("🌍 DynamicWorld Probability to Discrete Class Conversion")
    print("=" * 70)
    
    # First, examine the structure of one file to understand the data
    print("\n📊 Examining file structure...")
    sample_file = dw_files[2012]
    file_info = examine_file_structure(sample_file)
    
    if file_info is None:
        print("❌ Cannot proceed - unable to read input files")
        return
    
    print(f"\n📋 File structure summary:")
    print(f"   Bands: {file_info['count']}")
    print(f"   Shape: {file_info['shape']}")
    print(f"   Data type: {file_info['dtype']}")
    print(f"   CRS: {file_info['crs']}")
    
    if file_info['count'] != 9:
        print(f"⚠️  Warning: Expected 9 bands, but found {file_info['count']} bands")
        print("   Proceeding with available bands...")
    
    # Convert each file
    print(f"\n🔄 Converting files...")
    success_count = 0
    
    for year, input_file in dw_files.items():
        if input_file.exists():
            output_file = output_files[year]
            success = convert_probabilities_to_discrete(input_file, output_file, year)
            if success:
                success_count += 1
        else:
            print(f"❌ Input file not found: {input_file.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Files processed: {len(dw_files)}")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {len(dw_files) - success_count}")
    
    if success_count > 0:
        print(f"\n✅ Output files saved to: {output_dir}")
        print("   - DW_2012_discrete.tif")
        print("   - DW_2017_discrete.tif") 
        print("   - DW_2022_discrete.tif")
        print("\n💡 Each output file contains:")
        print("   - Same spatial resolution and CRS as input")
        print("   - 9 bands with integer values (0 or 1)")
        print("   - One band = 1 per pixel (the class with highest probability)")
        print("   - All other bands = 0 per pixel")
    else:
        print("\n❌ No files were successfully converted")

if __name__ == "__main__":
    main()
