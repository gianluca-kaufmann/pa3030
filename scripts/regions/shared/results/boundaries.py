from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from scripts.regions.shared.results.config import (
    PROFILE,
    REGION_LABEL,
    REGION_SLUG,
    X_LIMITS,
    Y_LIMITS,
    get_repo_root,
)

def detect_and_reproject_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect coordinate CRS and reproject to EPSG:4326 if needed.
    
    Detects if coordinates are in EPSG:3857 (Web Mercator) based on large values,
    and reprojects to EPSG:4326 (WGS84) for visualization.
    
    Returns DataFrame with reprojected x/y coordinates in EPSG:4326.
    """
    print("\nDetecting coordinate CRS...")
    
    # Sample to check coordinate ranges
    n_rows = len(df)
    if n_rows > 100000:
        sample_df = df.sample(n=100000, random_state=42)
    else:
        sample_df = df
    
    x_abs_max = sample_df['x'].abs().max()
    y_abs_max = sample_df['y'].abs().max()
    
    print(f"  x range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"  y range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    
    # If coordinates have large absolute values (>1000), likely Web Mercator (EPSG:3857)
    # WGS84 coordinates are small (lon: -180 to 180, lat: -90 to 90)
    if x_abs_max > 1000 or y_abs_max > 1000:
        print("  Detected EPSG:3857 (Web Mercator) coordinates")
        print("  Reprojecting to EPSG:4326 (WGS84)...")
        
        # Create GeoDataFrame with Web Mercator CRS
        geometry = gpd.points_from_xy(df['x'], df['y'])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:3857')
        
        # Reproject to WGS84
        gdf = gdf.to_crs('EPSG:4326')
        
        # Update x/y columns with reprojected coordinates
        df = df.copy()
        df['x'] = gdf.geometry.x.values
        df['y'] = gdf.geometry.y.values
        
        print(f"  Reprojected x range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
        print(f"  Reprojected y range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    else:
        print("  Coordinates appear to be in EPSG:4326 (WGS84) already")
    
    return df

def validate_coordinates(df: pd.DataFrame) -> None:
    """Validate that x/y coordinates look like expected region lon/lat."""
    print("\nValidating coordinates...")
    
    # Sample if too large (for performance)
    n_rows = len(df)
    if n_rows > 100000:
        print(f"  Sampling 100,000 rows from {n_rows:,} for validation...")
        sample_df = df.sample(n=100000, random_state=42)
    else:
        sample_df = df
    
    x_min, x_max = X_LIMITS
    y_min, y_max = Y_LIMITS
    x_valid = (sample_df['x'] >= x_min) & (sample_df['x'] <= x_max)
    x_valid_pct = x_valid.sum() / len(sample_df) * 100
    
    y_valid = (sample_df['y'] >= y_min) & (sample_df['y'] <= y_max)
    y_valid_pct = y_valid.sum() / len(sample_df) * 100
    
    print(f"  x (lon) in [{x_min}, {x_max}]: {x_valid_pct:.2f}%")
    print(f"  y (lat) in [{y_min}, {y_max}]: {y_valid_pct:.2f}%")
    
    # Both must be valid for most rows
    both_valid = x_valid & y_valid
    both_valid_pct = both_valid.sum() / len(sample_df) * 100
    
    # Warn if less than 95% are in expected ranges (but don't fail)
    if both_valid_pct < 95.0:
        x_range = (df['x'].min(), df['x'].max())
        y_range = (df['y'].min(), df['y'].max())
        warning_msg = (
            f"WARNING: Only {both_valid_pct:.2f}% of coordinates are in expected {REGION_LABEL} ranges.\n"
            f"Expected: x in [{x_min}, {x_max}], y in [{y_min}, {y_max}] (EPSG:4326)\n"
            f"Found: x in [{x_range[0]:.2f}, {x_range[1]:.2f}], "
            f"y in [{y_range[0]:.2f}, {y_range[1]:.2f}]\n"
            f"This may be due to coastal/island pixels or coordinate system issues.\n"
            f"Proceeding with caution..."
        )
        print(f"\n{'!'*70}")
        print(warning_msg)
        print(f"{'!'*70}\n")
    else:
        print(f"  Coordinate validation passed ({both_valid_pct:.2f}% in expected ranges)")

def get_region_boundary(region_boundary_path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Get region boundary from file or download from URL.
    
    Strategy:
    1. Use provided path if available
    2. Try common local file locations (GeoJSON/GPKG)
    3. Fallback to URL download
    4. Ensure CRS is normalized to EPSG:4326
    """
    region_iso_codes = PROFILE["iso_codes"]
    region_names = PROFILE["country_names"]
    boundary_basename = REGION_SLUG
    
    # Strategy 1: Use provided path
    if region_boundary_path and region_boundary_path.exists():
        print(f"Loading {REGION_LABEL} boundary from: {region_boundary_path}")
        gdf = gpd.read_file(region_boundary_path)
    else:
        # Strategy 2: Try common local file locations
        repo_root = get_repo_root()
        local_candidates = [
            repo_root / "data" / "boundaries" / f"{boundary_basename}.geojson",
            repo_root / "data" / "boundaries" / f"{boundary_basename}.gpkg",
            repo_root / "data" / "boundaries" / f"{boundary_basename}.shp",
            repo_root / "data" / REGION_SLUG / "ready" / "boundaries" / f"{boundary_basename}.geojson",
            repo_root / "data" / REGION_SLUG / "ready" / "boundaries" / f"{boundary_basename}.gpkg",
        ]
        
        gdf = None
        for candidate in local_candidates:
            if candidate.exists():
                print(f"Loading {REGION_LABEL} boundary from local file: {candidate}")
                try:
                    gdf = gpd.read_file(candidate)
                    break
                except Exception as e:
                    print(f"  Failed to load {candidate}: {e}")
                    continue
        
        # Strategy 3: Fallback to URL download
        if gdf is None:
            print(f"Loading {REGION_LABEL} boundary from URL...")
            import urllib.request
            import tempfile
            import zipfile
            import shutil
            
            urls = [
                "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
                "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson",
            ]
            
            for url in urls:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip' if url.endswith('.zip') else '.geojson') as tmp:
                        urllib.request.urlretrieve(url, tmp.name)
                        
                        if url.endswith('.zip'):
                            extract_dir = tempfile.mkdtemp()
                            try:
                                with zipfile.ZipFile(tmp.name, 'r') as zip_ref:
                                    zip_ref.extractall(extract_dir)
                                shp_files = list(Path(extract_dir).glob('*.shp'))
                                if shp_files:
                                    gdf = gpd.read_file(shp_files[0])
                                else:
                                    raise ValueError("Could not find shapefile in downloaded zip")
                            finally:
                                os.unlink(tmp.name)
                                shutil.rmtree(extract_dir)
                        else:
                            gdf = gpd.read_file(tmp.name)
                            os.unlink(tmp.name)
                    
                    if gdf is not None and not gdf.empty:
                        print(f"  Successfully loaded from {url}")
                        break
                except Exception as e:
                    print(f"  Failed to load from {url}: {e}")
                    continue
            
            if gdf is None or gdf.empty:
                raise RuntimeError(
                    f"Could not load {REGION_LABEL} boundary. Please provide --region_boundary path "
                    "or ensure internet connection for URL download."
                )
    
    # Filter to configured region.
    if 'iso_a3' in gdf.columns:
        south_america = gdf[gdf['iso_a3'].isin(region_iso_codes)].copy()
    elif 'ISO_A3' in gdf.columns:
        south_america = gdf[gdf['ISO_A3'].isin(region_iso_codes)].copy()
    elif 'name' in gdf.columns:
        south_america = gdf[gdf['name'].isin(region_names)].copy()
    else:
        # If no filtering column found, assume single region file
        south_america = gdf.copy()
    
    if south_america is None or len(south_america) == 0:
        raise ValueError(f"Could not find {REGION_LABEL} entries in boundary dataset")
    
    # Ensure CRS is normalized to EPSG:4326
    if south_america.crs is None:
        print("  Warning: No CRS defined, assuming EPSG:4326")
        south_america = south_america.set_crs('EPSG:4326', allow_override=True)
    elif south_america.crs.to_string() != 'EPSG:4326':
        print(f"  Reprojecting from {south_america.crs} to EPSG:4326...")
        south_america = south_america.to_crs('EPSG:4326')
    
    print(f"  Found {REGION_LABEL} boundary with {len(south_america)} feature(s) in EPSG:4326")
    return south_america
