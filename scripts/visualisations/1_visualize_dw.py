#!/usr/bin/env python3
"""
DynamicWorld Discrete Land Cover Visualization

This script loads the three converted discrete DynamicWorld GeoTIFFs
and visualizes all 9 bands for each file.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
data_dir = Path("/Users/gianluca/Desktop/Master's Thesis/code/data")
output_dir = Path("/Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Discrete files
discrete_files = {
    2012: data_dir / "DW_2012_discrete.tif",
    2017: data_dir / "DW_2017_discrete.tif",
    2022: data_dir / "DW_2022_discrete.tif"
}

# Land cover class names and colors
land_cover_classes = {
    0: {"name": "Water", "color": "#0066CC"},
    1: {"name": "Trees", "color": "#00CC00"}, 
    2: {"name": "Grass", "color": "#99FF99"},
    3: {"name": "Flooded vegetation", "color": "#00FFCC"},
    4: {"name": "Crops", "color": "#FFCC00"},
    5: {"name": "Shrub and scrub", "color": "#CC9900"},
    6: {"name": "Built", "color": "#CC0000"},
    7: {"name": "Bare ground", "color": "#CCCCCC"},
    8: {"name": "Snow and ice", "color": "#FFFFFF"}
}

def load_discrete_file(file_path, year):
    """Load a discrete DynamicWorld file"""
    print(f"📁 Loading {file_path.name}...")
    
    try:
        with rasterio.open(file_path) as src:
            # Read all bands
            all_bands = src.read()
            
            print(f"   Shape: {all_bands.shape}")
            print(f"   Data type: {all_bands.dtype}")
            print(f"   CRS: {src.crs}")
            
            return {
                'data': all_bands,
                'year': year,
                'src': src,
                'file_path': file_path
            }
            
    except Exception as e:
        print(f"   ❌ Error loading {file_path.name}: {e}")
        return None

def create_band_visualization(data_info, output_file):
    """Create visualization for all 9 bands of a single file"""
    if data_info is None:
        return
    
    year = data_info['year']
    all_bands = data_info['data']
    
    print(f"🎨 Creating visualization for {year}...")
    
    # Create figure with subplots for all 9 bands
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'DynamicWorld Discrete Land Cover - {year}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for band_idx in range(9):
        ax = axes_flat[band_idx]
        
        # Get band data
        band_data = all_bands[band_idx]
        
        # Create visualization
        im = ax.imshow(band_data, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Set title and labels
        class_info = land_cover_classes[band_idx]
        ax.set_title(f'Band {band_idx}: {class_info["name"]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Presence (0/1)', rotation=270, labelpad=15)
        
        # Count pixels
        pixel_count = np.sum(band_data)
        total_pixels = band_data.size
        percentage = (pixel_count / total_pixels) * 100
        
        # Add text with pixel count
        ax.text(0.02, 0.98, f'Pixels: {pixel_count:,}\n({percentage:.1f}%)', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplot if needed
    if len(axes_flat) > 9:
        axes_flat[9].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved: {output_file.name}")
    
    plt.show()

def create_combined_visualization(all_data_info):
    """Create a combined visualization showing all years and bands"""
    print("🎨 Creating combined visualization...")
    
    # Filter out None values
    valid_data = {k: v for k, v in all_data_info.items() if v is not None}
    
    if len(valid_data) == 0:
        print("   ❌ No valid data for combined visualization")
        return
    
    # Create figure with subplots for each year
    years = list(valid_data.keys())
    fig, axes = plt.subplots(len(years), 9, figsize=(27, 6*len(years)))
    
    if len(years) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('DynamicWorld Discrete Land Cover - All Years and Bands', 
                fontsize=16, fontweight='bold')
    
    for year_idx, (year, data_info) in enumerate(valid_data.items()):
        all_bands = data_info['data']
        
        for band_idx in range(9):
            ax = axes[year_idx, band_idx]
            
            # Get band data
            band_data = all_bands[band_idx]
            
            # Create visualization
            im = ax.imshow(band_data, cmap='RdYlBu_r', vmin=0, vmax=1)
            
            # Set title and labels
            class_info = land_cover_classes[band_idx]
            if year_idx == 0:  # Only show band names on top row
                ax.set_title(f'Band {band_idx}: {class_info["name"]}', 
                            fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'Band {band_idx}', fontsize=10)
            
            if band_idx == 0:  # Only show year labels on left column
                ax.set_ylabel(f'{year}', fontsize=12, fontweight='bold')
            
            # Count pixels
            pixel_count = np.sum(band_data)
            total_pixels = band_data.size
            percentage = (pixel_count / total_pixels) * 100
            
            # Add text with pixel count
            ax.text(0.02, 0.98, f'{pixel_count:,}\n({percentage:.1f}%)', 
                    transform=ax.transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "dynamicworld_discrete_all_years_bands.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved: {output_file.name}")
    
    plt.show()

def create_class_distribution_plot(all_data_info):
    """Create a plot showing class distribution across years"""
    print("📊 Creating class distribution plot...")
    
    # Filter out None values
    valid_data = {k: v for k, v in all_data_info.items() if v is not None}
    
    if len(valid_data) == 0:
        print("   ❌ No valid data for distribution plot")
        return
    
    # Prepare data for plotting
    years = list(valid_data.keys())
    classes = list(land_cover_classes.keys())
    class_names = [land_cover_classes[c]["name"] for c in classes]
    colors = [land_cover_classes[c]["color"] for c in classes]
    
    # Calculate percentages for each year and class
    data_matrix = np.zeros((len(years), len(classes)))
    
    for year_idx, (year, data_info) in enumerate(valid_data.items()):
        all_bands = data_info['data']
        total_pixels = all_bands[0].size  # All bands have same size
        
        for class_idx, band_idx in enumerate(classes):
            pixel_count = np.sum(all_bands[band_idx])
            percentage = (pixel_count / total_pixels) * 100
            data_matrix[year_idx, class_idx] = percentage
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bottom = np.zeros(len(years))
    
    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        ax.bar(years, data_matrix[:, class_idx], bottom=bottom, 
               label=class_name, color=color, alpha=0.8)
        bottom += data_matrix[:, class_idx]
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Total Pixels', fontsize=12, fontweight='bold')
    ax.set_title('Land Cover Class Distribution Across Years', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "dynamicworld_discrete_class_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Saved: {output_file.name}")
    
    plt.show()

def main():
    """Main visualization function"""
    print("🌍 DynamicWorld Discrete Land Cover Visualization")
    print("=" * 60)
    
    # Load all files
    print("\n📁 Loading discrete files...")
    all_data_info = {}
    
    for year, file_path in discrete_files.items():
        if file_path.exists():
            data_info = load_discrete_file(file_path, year)
            all_data_info[year] = data_info
        else:
            print(f"❌ File not found: {file_path.name}")
            all_data_info[year] = None
    
    # Create individual visualizations
    print("\n🎨 Creating individual visualizations...")
    for year, data_info in all_data_info.items():
        if data_info is not None:
            output_file = output_dir / f"dynamicworld_discrete_{year}_all_bands.png"
            create_band_visualization(data_info, output_file)
    
    # Create combined visualization
    print("\n🎨 Creating combined visualization...")
    create_combined_visualization(all_data_info)
    
    # Create class distribution plot
    print("\n📊 Creating class distribution plot...")
    create_class_distribution_plot(all_data_info)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VISUALIZATION SUMMARY")
    print("=" * 60)
    
    valid_files = sum(1 for v in all_data_info.values() if v is not None)
    print(f"Files processed: {len(all_data_info)}")
    print(f"Valid files: {valid_files}")
    print(f"Visualizations created: {valid_files + 2}")  # Individual + combined + distribution
    
    print(f"\n💾 Output files saved to: {output_dir}")
    print("   - dynamicworld_discrete_2012_all_bands.png")
    print("   - dynamicworld_discrete_2017_all_bands.png")
    print("   - dynamicworld_discrete_2022_all_bands.png")
    print("   - dynamicworld_discrete_all_years_bands.png")
    print("   - dynamicworld_discrete_class_distribution.png")
    
    print("\n✅ Visualization complete! 🎉")

if __name__ == "__main__":
    main()
