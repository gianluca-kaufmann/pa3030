#!/usr/bin/env python3
"""
Visualize the merged raster with high biodiversity areas to verify the merge worked correctly.
This script creates comprehensive visualizations of all bands in the south_america_with_ree_plus_hba.tif file.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm, BoundaryNorm

def _print_band_analysis(data: np.ndarray, band_names):
    """Print detailed analysis of each band."""
    print("\n=== COMPREHENSIVE BAND ANALYSIS ===")
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
        
        # Additional analysis for specific bands
        if i == 4:  # HBA band
            hba_pixels = int((valid == 1).sum())
            print(f"  High Biodiversity Pixels: {hba_pixels:,} ({hba_pixels/num_valid:.1%})")

def create_individual_band_plots(data, band_names, transform, bounds):
    """Create individual detailed plots for each band."""
    
    for i in range(data.shape[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Band {i+1}: {band_names[i]}', fontsize=16, fontweight='bold')
        
        band_data = data[i]
        valid_mask = ~np.isnan(band_data)
        
        if not np.any(valid_mask):
            ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        else:
            valid_data = band_data[valid_mask]
            masked_data = np.ma.masked_where(np.isnan(band_data), band_data)
            
            # Left plot: Spatial distribution
            if i == 4:  # HBA band - binary visualization
                colors = ['white', 'darkgreen']
                cmap = ListedColormap(colors)
                im1 = ax1.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
                ax1.set_title('High Biodiversity Areas (Green = HBA)')
            elif i == 3:  # REE band - discrete values
                unique_vals = np.unique(valid_data)
                n_colors = len(unique_vals)
                colors = plt.cm.Set3(np.linspace(0, 1, n_colors))
                cmap = ListedColormap(colors)
                im1 = ax1.imshow(masked_data, cmap=cmap, vmin=unique_vals.min(), vmax=unique_vals.max())
                ax1.set_title(f'REE Occurrences (Discrete: {unique_vals.min():.0f}-{unique_vals.max():.0f})')
            else:  # Continuous data
                p2, p98 = np.percentile(valid_data, [2, 98])
                use_log = (p98 > 0) and (p98 / max(p2, 1e-12) > 1e3)
                
                if use_log:
                    min_pos = valid_data[valid_data > 0].min() if (valid_data > 0).any() else 1e-6
                    im1 = ax1.imshow(masked_data, cmap='viridis', norm=LogNorm(vmin=max(min_pos, p2 if p2>0 else min_pos), vmax=max(p98, min_pos*10)))
                    ax1.set_title(f'{band_names[i]} (Log Scale)')
                else:
                    im1 = ax1.imshow(masked_data, cmap='viridis', vmin=p2, vmax=p98)
                    ax1.set_title(f'{band_names[i]} (P2-P98: {p2:.3g}-{p98:.3g})')
            
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_aspect('equal')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Right plot: Histogram
            if i == 4:  # HBA band - bar chart
                counts = np.bincount(valid_data.astype(int))
                ax2.bar(['Not HBA', 'HBA'], counts, color=['lightcoral', 'darkgreen'])
                ax2.set_title('Pixel Counts')
                ax2.set_ylabel('Number of Pixels')
                for j, count in enumerate(counts):
                    ax2.text(j, count + max(counts)*0.01, f'{count:,}', ha='center', va='bottom')
            else:
                ax2.hist(valid_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title('Value Distribution')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Frequency')
                
                # Add statistics text
                stats_text = f'Mean: {np.mean(valid_data):.3f}\nStd: {np.std(valid_data):.3f}\nMin: {np.min(valid_data):.3f}\nMax: {np.max(valid_data):.3f}'
                ax2.text(0.7, 0.7, stats_text, transform=ax2.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
        
        plt.tight_layout()
        output_path = f"/Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures/band_{i+1}_{band_names[i].lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

def create_overview_plot(data, band_names):
    """Create an overview plot with all bands in a grid."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('South America Dataset: All Bands Overview', fontsize=18, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i in range(5):  # 5 bands
        ax = axes_flat[i]
        band_data = data[i]
        valid_mask = ~np.isnan(band_data)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Band {i+1}: {band_names[i]}', fontsize=12)
        else:
            valid_data = band_data[valid_mask]
            masked_data = np.ma.masked_where(np.isnan(band_data), band_data)
            
            if i == 4:  # HBA band
                colors = ['white', 'darkgreen']
                cmap = ListedColormap(colors)
                im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f'Band {i+1}: {band_names[i]}\n(Green = High Biodiversity)', fontsize=12)
            elif i == 3:  # REE band
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
    
    # Remove the empty subplot
    axes_flat[5].remove()
    
    plt.tight_layout()
    output_path = "/Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures/hba_merged_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overview: {output_path}")

def create_hba_analysis_plots(data, band_names):
    """Create specific analysis plots for high biodiversity areas."""
    
    hba_band = data[4]  # HBA is band 5 (index 4)
    hba_mask = hba_band == 1
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('High Biodiversity Areas Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: HBA distribution
    ax1 = axes[0, 0]
    colors = ['white', 'darkgreen']
    cmap = ListedColormap(colors)
    im1 = ax1.imshow(hba_band, cmap=cmap, vmin=0, vmax=1)
    ax1.set_title('High Biodiversity Areas Distribution')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot 2: HBA pixel counts
    ax2 = axes[0, 1]
    hba_flat = hba_band.flatten()
    counts = np.bincount(hba_flat.astype(int))
    bars = ax2.bar(['Not HBA', 'HBA'], counts, color=['lightcoral', 'darkgreen'])
    ax2.set_title('HBA Pixel Counts')
    ax2.set_ylabel('Number of Pixels')
    for i, count in enumerate(counts):
        ax2.text(i, count + max(counts)*0.01, f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: REE vs HBA relationship
    ax3 = axes[1, 0]
    ree_band = data[3]  # REE is band 4 (index 3)
    ree_in_hba = ree_band[hba_mask & ~np.isnan(ree_band)]
    ree_not_hba = ree_band[~hba_mask & ~np.isnan(ree_band)]
    
    if len(ree_in_hba) > 0 and len(ree_not_hba) > 0:
        ax3.hist([ree_not_hba, ree_in_hba], bins=20, alpha=0.7, 
                label=['Not in HBA', 'In HBA'], color=['lightcoral', 'darkgreen'])
        ax3.set_title('REE Occurrences: HBA vs Non-HBA')
        ax3.set_xlabel('REE Occurrence Count')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Insufficient REE data for comparison', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('REE vs HBA Analysis')
    
    # Plot 4: Coverage statistics
    ax4 = axes[1, 1]
    total_pixels = hba_band.size
    hba_pixels = np.sum(hba_mask)
    coverage_pct = (hba_pixels / total_pixels) * 100
    
    # Create a pie chart
    sizes = [hba_pixels, total_pixels - hba_pixels]
    labels = [f'High Biodiversity\n{hba_pixels:,} pixels\n({coverage_pct:.2f}%)', 
              f'Other Areas\n{total_pixels - hba_pixels:,} pixels\n({100-coverage_pct:.2f}%)']
    colors = ['darkgreen', 'lightgray']
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax4.set_title('HBA Coverage of South America')
    
    plt.tight_layout()
    output_path = "/Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures/hba_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved HBA analysis: {output_path}")

def visualize_hba_merged_raster():
    """Main function to create all visualizations."""
    
    print("Loading merged raster with high biodiversity areas...")
    
    with rasterio.open("/Users/gianluca/Desktop/Master's Thesis/code/data/south_america_with_ree_plus_hba.tif") as src:
        print(f"Raster info: {src.width}x{src.height}, {src.count} bands")
        print(f"Bounds: {src.bounds}")
        print(f"CRS: {src.crs}")
        
        # Read all bands
        data = src.read()
        
        # Band names based on the merge_high_biodiversity script
        band_names = [
            'Population Density',  # Band 1
            'Protected Areas',     # Band 2  
            'Night Lights',        # Band 3
            'REE Occurrences',     # Band 4
            'High Biodiversity Areas'  # Band 5
        ]

        # Print comprehensive analysis
        _print_band_analysis(data, band_names)

        print("\nCreating visualizations...")
        
        # Create individual band plots
        print("1. Creating individual band plots...")
        create_individual_band_plots(data, band_names, src.transform, src.bounds)
        
        # Create overview plot
        print("2. Creating overview plot...")
        create_overview_plot(data, band_names)
        
        # Create HBA-specific analysis
        print("3. Creating HBA analysis plots...")
        create_hba_analysis_plots(data, band_names)
        
        print("\n=== VISUALIZATION SUMMARY ===")
        print("All visualizations saved to: /Users/gianluca/Desktop/Master's Thesis/code/outputs/Figures/")
        print("- Individual band plots: band_1_*.png to band_5_*.png")
        print("- Overview plot: hba_merged_overview.png")
        print("- HBA analysis: hba_analysis.png")
        
        # Print final statistics
        print("\n=== FINAL STATISTICS ===")
        hba_band = data[4]
        hba_pixels = np.sum(hba_band == 1)
        total_pixels = hba_band.size
        print(f"High Biodiversity Areas: {hba_pixels:,} pixels ({hba_pixels/total_pixels*100:.2f}% of South America)")
        print(f"Total dataset size: {total_pixels:,} pixels")
        print(f"Dataset covers: {src.bounds}")

if __name__ == "__main__":
    visualize_hba_merged_raster()
