#!/usr/bin/env python3
"""
Quick plotting script for mineral occurrences - plots only, no GeoTIFF creation.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Optional plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

def read_csv_smart(path: str) -> pd.DataFrame:
    """Read CSV with automatic delimiter detection."""
    import csv
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(65536)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        sep = dialect.delimiter
    except Exception:
        sep = ","
    return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8", on_bad_lines='skip')

def detect_lon_lat_cols(columns):
    """Detect longitude and latitude columns."""
    cols_lower = {c.lower(): c for c in columns}
    lon_cands = ["longitude","lon","long","lng","x","lon_dd","longitude_dd","east","easting"]
    lat_cands = ["latitude","lat","y","lat_dd","latitude_dd","north","northing"]
    lon = next((cols_lower[c] for c in lon_cands if c in cols_lower), None)
    lat = next((cols_lower[c] for c in lat_cands if c in cols_lower), None)
    if lon is None or lat is None:
        raise ValueError(f"Could not find lon/lat columns in {list(columns)}.")
    return lon, lat

def create_mineral_type(row):
    """Create a simplified mineral type based on available columns."""
    # Check deposit type first (most specific)
    if pd.notna(row.get('Dep_Type', '')) and str(row.get('Dep_Type', '')).strip():
        dep_type = str(row.get('Dep_Type', '')).lower()
        if 'carbonatite' in dep_type:
            return 'Carbonatite'
        elif 'pegmatite' in dep_type:
            return 'Pegmatite'
        elif 'placer' in dep_type:
            return 'Placer'
        elif 'vein' in dep_type:
            return 'Vein'
        elif 'phosphorite' in dep_type:
            return 'Phosphorite'
        elif 'alkaline' in dep_type:
            return 'Alkaline Igneous'
    
    # Check commodities for multi-element deposits
    if pd.notna(row.get('Commods', '')) and str(row.get('Commods', '')).strip():
        commods = str(row.get('Commods', '')).lower()
        
        # Count different commodity types
        has_ree = 'ree' in commods
        has_nb_ta = any(metal in commods for metal in ['niobium', 'nb', 'tantalum', 'ta'])
        has_u_th = any(metal in commods for metal in ['uranium', 'u', 'thorium', 'th'])
        has_zr_hf = any(metal in commods for metal in ['zirconium', 'zr', 'hafnium', 'hf'])
        has_p = any(metal in commods for metal in ['phosphorus', 'p', 'phosphate'])
        has_v = any(metal in commods for metal in ['vanadium', 'v'])
        has_li_be = any(metal in commods for metal in ['lithium', 'li', 'beryllium', 'be'])
        has_gem = 'gem' in commods
        
        # Create composite categories
        if has_nb_ta and has_ree:
            return 'REE + Nb-Ta'
        elif has_u_th and has_ree:
            return 'REE + U-Th'
        elif has_zr_hf and has_ree:
            return 'REE + Zr-Hf'
        elif has_p and has_ree:
            return 'REE + P'
        elif has_li_be and has_ree:
            return 'REE + Li-Be'
        elif has_gem and has_ree:
            return 'REE + Gems'
        elif has_nb_ta:
            return 'Nb-Ta'
        elif has_u_th:
            return 'U-Th'
        elif has_zr_hf:
            return 'Zr-Hf'
        elif has_p:
            return 'P'
        elif has_v:
            return 'V'
        elif has_li_be:
            return 'Li-Be'
        elif has_gem:
            return 'Gems'
        elif has_ree:
            return 'REE Only'
    
    # Check REE minerals for more specific classification
    if pd.notna(row.get('REE_Mins', '')) and str(row.get('REE_Mins', '')).strip():
        ree_mins = str(row.get('REE_Mins', '')).lower()
        if 'monazite' in ree_mins and 'bastnäsite' in ree_mins:
            return 'Monazite + Bastnäsite'
        elif 'monazite' in ree_mins:
            return 'Monazite'
        elif 'bastnäsite' in ree_mins:
            return 'Bastnäsite'
        elif 'xenotime' in ree_mins:
            return 'Xenotime'
        elif 'allanite' in ree_mins:
            return 'Allanite'
        else:
            return 'Other REE Minerals'
    
    return 'Other'

def plot_mineral_occurrences(df, lon_col, lat_col, output_dir="outputs"):
    """Plot mineral occurrences on a global map with both interactive and static versions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Clean data for plotting
    df_plot = df.copy()
    df_plot[lon_col] = pd.to_numeric(df_plot[lon_col], errors="coerce")
    df_plot[lat_col] = pd.to_numeric(df_plot[lat_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[lon_col, lat_col])
    df_plot = df_plot[(df_plot[lon_col] >= -180) & (df_plot[lon_col] <= 180) & 
                      (df_plot[lat_col] >= -90) & (df_plot[lat_col] <= 90)]
    
    print(f"Plotting {len(df_plot)} mineral occurrences...")
    
    # Apply the mineral type classification
    df_plot['Mineral_Type'] = df_plot.apply(create_mineral_type, axis=1)
    
    # Use the new mineral type column for coloring
    color_col = 'Mineral_Type'
    
    print(f"Mineral type distribution:")
    print(df_plot['Mineral_Type'].value_counts())
    
    # Interactive plot with Plotly
    if PLOTLY_AVAILABLE:
        try:
            # Select key columns for hover information
            hover_cols = ['Name', 'Country', 'Commods', 'REE_Mins', 'Dep_Type', 'Mineral_Type']
            available_hover_cols = [col for col in hover_cols if col in df_plot.columns]
            
            fig = px.scatter_geo(
                df_plot,
                lat=lat_col,
                lon=lon_col,
                color=color_col,
                hover_data=available_hover_cols,
                projection='natural earth',
                opacity=0.8,
                height=800,
                title='Global Mineral Occurrences by Type',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                title_x=0.5,
                title_font_size=20,
                margin=dict(l=0, r=0, t=60, b=0),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            fig.update_traces(marker=dict(size=6))
            
            html_path = output_dir / 'mineral_occurrences_interactive.html'
            fig.write_html(str(html_path))
            print(f"Interactive map saved to: {html_path}")
        except Exception as e:
            print(f"Interactive plot failed: {e}")
    
    # Static plot with Matplotlib
    try:
        plt.figure(figsize=(15, 8))
        
        if CARTOPY_AVAILABLE:
            try:
                # Use Cartopy for better map projection
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.8)
                ax.add_feature(cfeature.OCEAN, alpha=0.3)
                ax.add_feature(cfeature.LAND, alpha=0.1)
                ax.set_global()
                
                if color_col and df_plot[color_col].nunique() > 1:
                    # Color by category if available
                    unique_vals = df_plot[color_col].unique()
                    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
                    for i, val in enumerate(unique_vals):
                        mask = df_plot[color_col] == val
                        ax.scatter(df_plot.loc[mask, lon_col], df_plot.loc[mask, lat_col], 
                                  s=8, c=[colors[i]], alpha=0.7, transform=ccrs.PlateCarree(),
                                  label=str(val))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.scatter(df_plot[lon_col], df_plot[lat_col], s=8, c='red', alpha=0.7, 
                              transform=ccrs.PlateCarree())
            except Exception as e:
                print(f"Cartopy plotting failed: {e}")
                print("Falling back to simple scatter plot...")
                # Fallback to simple plot
                if color_col and df_plot[color_col].nunique() > 1:
                    unique_vals = df_plot[color_col].unique()
                    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
                    for i, val in enumerate(unique_vals):
                        mask = df_plot[color_col] == val
                        plt.scatter(df_plot.loc[mask, lon_col], df_plot.loc[mask, lat_col], 
                                   s=8, c=[colors[i]], alpha=0.7, label=str(val))
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    plt.scatter(df_plot[lon_col], df_plot[lat_col], s=8, c='red', alpha=0.7)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
        else:
            # Fallback without Cartopy
            if color_col and df_plot[color_col].nunique() > 1:
                unique_vals = df_plot[color_col].unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
                for i, val in enumerate(unique_vals):
                    mask = df_plot[color_col] == val
                    plt.scatter(df_plot.loc[mask, lon_col], df_plot.loc[mask, lat_col], 
                               s=8, c=[colors[i]], alpha=0.7, label=str(val))
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.scatter(df_plot[lon_col], df_plot[lat_col], s=8, c='red', alpha=0.7)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
        
        plt.title('Global Mineral Occurrences by Type', fontsize=16, pad=20)
        plt.tight_layout()
        
        png_path = output_dir / 'mineral_occurrences_static.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Static map saved to: {png_path}")
        
    except Exception as e:
        print(f"Static plotting failed: {e}")
        print("Creating simple fallback plot...")
        # Ultra-simple fallback
        plt.figure(figsize=(12, 6))
        plt.scatter(df_plot[lon_col], df_plot[lat_col], s=8, c='red', alpha=0.7)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Global Mineral Occurrences by Type (Simple)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        png_path = output_dir / 'mineral_occurrences_static.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Simple static map saved to: {png_path}")

def main():
    """Main function to run the plotting."""
    CSV_PATH = "Global_REE_occurrence_database.csv"
    
    df = read_csv_smart(CSV_PATH)
    lon_col, lat_col = detect_lon_lat_cols(df.columns)
    
    # Create plots
    plot_mineral_occurrences(df, lon_col, lat_col)

if __name__ == "__main__":
    main()
