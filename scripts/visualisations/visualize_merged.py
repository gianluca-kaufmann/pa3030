#!/usr/bin/env python3
"""
Visualize the merged raster to verify the merge worked correctly.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_merged_raster():
    """Create visualizations of the merged raster."""
    
    with rasterio.open("south_america_with_ree.tif") as src:
        print(f"Raster info: {src.width}x{src.height}, {src.count} bands")
        print(f"Bounds: {src.bounds}")
        print(f"CRS: {src.crs}")
        
        # Read all bands
        data = src.read()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Merged Raster: South America with REE Occurrences', fontsize=16)
        
        # Plot each band
        band_names = ['Template Band 1', 'Template Band 2', 'Template Band 3', 'REE Occurrences']
        
        for i in range(4):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            band_data = data[i]
            
            # Handle NaN values
            if np.any(~np.isnan(band_data)):
                # Create a masked array for NaN values
                masked_data = np.ma.masked_where(np.isnan(band_data), band_data)
                
                if i == 3:  # REE band - use discrete colors
                    # Create discrete colormap for REE values
                    unique_vals = np.unique(band_data[~np.isnan(band_data)])
                    n_colors = len(unique_vals)
                    colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
                    cmap = ListedColormap(colors)
                    
                    im = ax.imshow(masked_data, cmap=cmap, vmin=unique_vals.min(), vmax=unique_vals.max())
                    ax.set_title(f'{band_names[i]} (Discrete: {unique_vals.min():.0f}-{unique_vals.max():.0f})')
                else:  # Template bands - use continuous colors
                    im = ax.imshow(masked_data, cmap='viridis')
                    ax.set_title(f'{band_names[i]} (Range: {np.nanmin(band_data):.1f}-{np.nanmax(band_data):.1f})')
                
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{band_names[i]} (No Data)')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('merged_raster_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Visualization saved as 'merged_raster_visualization.png'")
        
        # Print statistics
        print("\n=== BAND STATISTICS ===")
        for i in range(4):
            band_data = data[i]
            valid_mask = ~np.isnan(band_data)
            if np.any(valid_mask):
                valid_data = band_data[valid_mask]
                print(f"Band {i+1} ({band_names[i]}):")
                print(f"  Valid pixels: {np.sum(valid_mask):,}")
                print(f"  Range: {np.min(valid_data):.3f} to {np.max(valid_data):.3f}")
                print(f"  Mean: {np.mean(valid_data):.3f}")
                if i == 3:  # REE band
                    unique_vals = np.unique(valid_data)
                    print(f"  Unique values: {len(unique_vals)} ({unique_vals[:10]}...)")
            else:
                print(f"Band {i+1} ({band_names[i]}): No valid data")

if __name__ == "__main__":
    visualize_merged_raster()
