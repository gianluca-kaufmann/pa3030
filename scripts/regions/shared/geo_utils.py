"""Shared geographic utility functions.

Usage pattern (call once at script start, pass pixel_size_m everywhere):

    import rasterio
    from scripts.regions.shared.geo_utils import pixel_area_km2

    with rasterio.open(backbone_path) as src:
        PIXEL_SIZE_M = abs(src.transform.a)   # metres per pixel (cell width)

    areas = pixel_area_km2(y_array, pixel_size_m=PIXEL_SIZE_M)
"""

from __future__ import annotations

import numpy as np


def pixel_area_km2(y_epsg3857: np.ndarray, pixel_size_m: float = 1000.0) -> np.ndarray:
    """Return ground area in km² for each EPSG:3857 pixel centre.

    In Web Mercator (EPSG:3857) a projected pixel of side pixel_size_m metres
    represents cos²(lat) × (pixel_size_m/1000)² km² of actual ground area.

    Args:
        y_epsg3857: Array of EPSG:3857 northing coordinates (metres).
        pixel_size_m: Actual projected pixel side length in metres, extracted
                      from the raster transform (NOT assumed to be 1000).
                      Retrieve once via:
                          import rasterio
                          with rasterio.open(backbone_path) as src:
                              pixel_size_m = abs(src.transform.a)   # cell width

    Returns:
        Array of ground area values in km² (same shape as y_epsg3857).

    Notes:
        Distortion examples for South America (EPSG:3857 at 1000 m grid):
            Equator  (0°):  1.000 km²/pixel
            10°S (central Brazil): 0.970 km²/pixel
            30°S (Uruguay): 0.750 km²/pixel
            50°S (Patagonia): 0.413 km²/pixel — naive counting overestimates by 2.4×
    """
    R = 6378137.0  # WGS84 semi-major axis (metres)
    lat_rad = 2.0 * np.arctan(np.exp(y_epsg3857 / R)) - np.pi / 2.0
    pixel_km = pixel_size_m / 1000.0
    return (pixel_km ** 2) * np.cos(lat_rad) ** 2  # actual ground area in km²
