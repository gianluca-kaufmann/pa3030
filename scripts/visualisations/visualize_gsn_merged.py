#!/usr/bin/env python3
"""
Visualize the complete GSN merged raster dataset.
This script creates comprehensive visualizations of all 9 bands in the south_america_with_ree_plus_hba_plus_gsn.tif file.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm

def _print_band_analysis(data: np.ndarray, band_names):
    """Print detailed analysis of each band."""
    print("\n=== COMPREHENSIVE GSN DATASET ANALYSIS ===")
    for i in range(data.shape[0]):
        band = data[i]
        mask = np.isfinite(band)
        num_total = band.size
        num_valid = int(mask.sum())
        num_nan = num_total - num_valid
        
        if num_valid == 0:
            print(f"Band {i+1} ({band_names[i]}): no valid data")
            continue
            
        valid = band[mask]
        p2, p50, p98 = np.percentile(valid, [2, 50, 98])
        vmin, vmax = valid.min(), valid.max()
        mean, std = float(valid.mean()), float(valid.std())
        num_zero = int((valid == 0).sum())
        
        print(f"Band {i+1} ({band_names[i]}):")
        print(f"  Valid: {num_valid:,} / {num_total:,} | NaNs: {num_nan:,}")
        print(f"  Min/Max: {vmin:.4g} / {vmax:.4g} | Mean±Std: {mean:.4g} ± {std:.4g}")
        print(f"  P2/P50/P98: {p2:.4g} / {p50:.4g} / {p98:.4g}")
        print(f"  Zeros: {num_zero:,} ({num_zero/num_valid:.1%})")
        
        # Additional analysis for GSN bands
        if i == 8:  # Terrestrial ecoregions - categorical
            unique_ecoregions = len(np.unique(valid[valid > 0]))
            ecoregion_pixels = int((valid > 0).sum())
            print(f"  Ecoregion Pixels: {ecoregion_pixels:,} ({ecoregion_pixels/num_valid:.1%})")
            print(f"  Unique Ecoregions: {unique_ecoregions}")
        elif i >= 5:  # Other GSN bands (6-8) - binary
            gsn_pixels = int((valid == 1).sum())
            print(f"  GSN Feature Pixels: {gsn_pixels:,} ({gsn_pixels/num_valid:.1%})")

def create_gsn_overview_plot(data, band_names):
    """Create an overview plot with all 9 bands in a grid."""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Complete GSN Dataset: All 9 Bands Overview', fontsize=20, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i in range(9):  # 9 bands
        ax = axes_flat[i]
        band_data = data[i]
        valid_mask = ~np.isnan(band_data)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Band {i+1}: {band_names[i]}', fontsize=12)
        else:
            valid_data = band_data[valid_mask]
            masked_data = np.ma.masked_where(np.isnan(band_data), band_data)
            
            if i == 8:  # Terrestrial ecoregions - categorical visualization
                unique_vals = np.unique(valid_data)
                n_colors = min(len(unique_vals), 20)  # Limit colors for visualization
                # Use plasma colormap for ecoregions (vibrant and visually appealing)
                cmap = plt.cm.plasma
                colors = cmap(np.linspace(0, 1, n_colors))
                cmap = ListedColormap(colors)
                im = ax.imshow(masked_data, cmap=cmap, vmin=unique_vals.min(), vmax=unique_vals.max())
                ax.set_title(f'Band {i+1}: {band_names[i]}\n({len(unique_vals)} Ecoregions)', fontsize=12)
            elif i >= 5:  # Other GSN bands (6-8) - binary visualization
                colors = ['white', 'darkgreen']
                cmap = ListedColormap(colors)
                im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f'Band {i+1}: {band_names[i]}\n(Green = Present)', fontsize=12)
            elif i == 1:  # Protected Areas band - binary
                colors = ['white', 'darkgreen']
                cmap = ListedColormap(colors)
                im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f'Band {i+1}: {band_names[i]}\n(Green = Protected)', fontsize=12)
            elif i == 4:  # HBA band - binary
                colors = ['white', 'darkgreen']
                cmap = ListedColormap(colors)
                im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f'Band {i+1}: {band_names[i]}\n(Green = HBA)', fontsize=12)
            elif i == 3:  # REE band - discrete values
                unique_vals = np.unique(valid_data)
                n_colors = len(unique_vals)
                colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
                cmap = ListedColormap(colors)
                im = ax.imshow(masked_data, cmap=cmap, vmin=unique_vals.min(), vmax=unique_vals.max())
                ax.set_title(f'Band {i+1}: {band_names[i]}\n(Discrete Values)', fontsize=12)
            else:  # Continuous data
                p2, p98 = np.percentile(valid_data, [2, 98])
                use_log = (p98 > 0) and (p98 / max(p2, 1e-12) > 1e3)
                
                if use_log:
                    min_pos = valid_data[valid_data > 0].min() if (valid_data > 0).any() else 1e-6
                    im = ax.imshow(masked_data, cmap='viridis', norm=LogNorm(vmin=max(min_pos, p2 if p2>0 else min_pos), vmax=max(p98, min_pos*10)))
                    ax.set_title(f'Band {i+1}: {band_names[i]}\n(Log Scale)', fontsize=12)
                else:
                    im = ax.imshow(masked_data, cmap='viridis', vmin=p2, vmax=p98)
                    ax.set_title(f'Band {i+1}: {band_names[i]}\n(P2-P98)', fontsize=12)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    output_path = "/Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures/gsn_complete_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved complete overview: {output_path}")

def create_gsn_analysis_plots(data, band_names):
    """Create specific analysis plots for GSN features."""
    
    # Extract GSN bands (6-9)
    gsn_bands = data[5:9]  # Bands 6-9
    gsn_names = band_names[5:9]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GSN Features Analysis', fontsize=16, fontweight='bold')
    
    for i, (band_data, band_name) in enumerate(zip(gsn_bands, gsn_names)):
        ax = axes[i//2, i%2]
        
        if i == 3:  # Terrestrial ecoregions - use continuous values with plasma palette
            # Use plasma colormap for ecoregions
            cmap = plt.cm.plasma
            im = ax.imshow(band_data, cmap=cmap, vmin=0, vmax=band_data.max())
            ax.set_title(f'{band_name}\n(113 Ecoregions)')
            
            # Add statistics for ecoregions
            ecoregion_pixels = np.sum(band_data > 0)
            total_pixels = band_data.size
            coverage_pct = (ecoregion_pixels / total_pixels) * 100
            unique_ecoregions = len(np.unique(band_data[band_data > 0]))
            ax.text(0.02, 0.98, f'Coverage: {coverage_pct:.2f}%\nPixels: {ecoregion_pixels:,}\nEcoregions: {unique_ecoregions}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:  # Binary features
            # Spatial distribution
            colors = ['white', 'darkgreen']
            cmap = ListedColormap(colors)
            im = ax.imshow(band_data, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f'{band_name}\n(Green = Present)')
            
            # Add statistics for binary features
            gsn_pixels = np.sum(band_data == 1)
            total_pixels = band_data.size
            coverage_pct = (gsn_pixels / total_pixels) * 100
            ax.text(0.02, 0.98, f'Coverage: {coverage_pct:.2f}%\nPixels: {gsn_pixels:,}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    output_path = "/Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures/gsn_features_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved GSN features analysis: {output_path}")


def visualize_gsn_merged_raster():
    """Main function to create all GSN visualizations."""
    
    print("Loading complete GSN merged raster...")
    
    with rasterio.open("/Users/gianluca/Desktop/Master's Thesis/code/data/south_america_with_ree_plus_hba_plus_gsn.tif") as src:
        print(f"Raster info: {src.width}x{src.height}, {src.count} bands")
        print(f"Bounds: {src.bounds}")
        print(f"CRS: {src.crs}")
        
        # Read all bands
        data = src.read()
        
        # Band names
        band_names = [
            'Night Lights',            # Band 1
            'Protected Areas',         # Band 2  
            'Population Density',      # Band 3
            'REE Occurrences',         # Band 4
            'High Biodiversity Areas', # Band 5
            'Wildlife Corridors',      # Band 6
            'Climate Stabilization',   # Band 7
            'Intact Wilderness',       # Band 8
            'Terrestrial Ecoregions'   # Band 9
        ]

        # Print comprehensive analysis
        _print_band_analysis(data, band_names)

        print("\nCreating GSN visualizations...")
        
        # Create complete overview plot
        print("1. Creating complete overview plot...")
        create_gsn_overview_plot(data, band_names)
        
        # Create GSN-specific analysis
        print("2. Creating GSN features analysis...")
        create_gsn_analysis_plots(data, band_names)
        
        print("\n=== GSN VISUALIZATION SUMMARY ===")
        print("All visualizations saved to: /Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures/")
        print("- Complete overview: gsn_complete_overview.png")
        print("- GSN features analysis: gsn_features_analysis.png")
        
        # Print final statistics
        print("\n=== FINAL GSN STATISTICS ===")
        gsn_bands = data[5:9]
        gsn_names = band_names[5:9]
        total_pixels = data[0].size
        
        for i, (band_data, name) in enumerate(zip(gsn_bands, gsn_names)):
            if i == 3:  # Terrestrial ecoregions - count non-zero pixels
                gsn_pixels = np.sum(band_data > 0)
                unique_ecoregions = len(np.unique(band_data[band_data > 0]))
                coverage_pct = (gsn_pixels / total_pixels) * 100
                print(f"{name}: {gsn_pixels:,} pixels ({coverage_pct:.2f}% of South America), {unique_ecoregions} unique ecoregions")
            else:  # Binary features
                gsn_pixels = np.sum(band_data == 1)
                coverage_pct = (gsn_pixels / total_pixels) * 100
                print(f"{name}: {gsn_pixels:,} pixels ({coverage_pct:.2f}% of South America)")

if __name__ == "__main__":
    visualize_gsn_merged_raster()
