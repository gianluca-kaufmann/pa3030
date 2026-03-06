#!/usr/bin/env python3
"""
Shared utilities for economic value preprocessing.

Used by both South America and USA regional scripts to avoid code duplication.
All region-specific constants (paths, ISO3 sets, country ID maps) live in the
regional scripts; this module provides the parameterised logic.

Spatial allocation formula
--------------------------
For each agricultural pixel i in country c at year y:

    value_usd_ha_i = FAO_GPV(c, y) * LC_WEIGHT[lc_class_i]
                     / Σ_j (LC_WEIGHT[lc_class_j] * 100 ha)

Guarantees:  Σ_i (value_usd_ha_i * 100 ha)  =  FAO_GPV(c, y)

Non-agricultural pixels within a country boundary get value = 0.
Pixels outside all country boundaries remain NaN (nodata).

Land-cover class weights
------------------------
MODIS IGBP classes used and their weights:
  12  Croplands                         1.0  (pure cropland)
  14  Cropland/Natural Vegetation Mix   1.0  (partially cultivated)
   8  Woody Savannas                    0.3  (grazed tree-savanna)
   9  Savannas                          0.3  (Cerrado, llanos — mostly cattle)
  10  Grasslands                        0.3  (managed pasture, rangeland)

The 1.0 / 0.3 ratio (~3×) reflects that cropland productivity per hectare
is roughly 3× that of rangeland — consistent with FAO structural data for
South America and the USA. This is a transparent, fixed constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

# ── Temporal range ────────────────────────────────────────────────────────────

YEAR_MIN: int = 2000
YEAR_MAX: int = 2024

# ── Land-cover weights ────────────────────────────────────────────────────────

CROPLAND_CLASSES: frozenset[int] = frozenset({12, 14})
PASTURE_CLASSES: frozenset[int] = frozenset({8, 9, 10})
AG_CLASSES: frozenset[int] = CROPLAND_CLASSES | PASTURE_CLASSES

LC_WEIGHTS: dict[int, float] = {
    12: 1.0,  # Croplands
    14: 1.0,  # Cropland/Natural Vegetation Mosaics
    8: 0.3,   # Woody Savannas
    9: 0.3,   # Savannas
    10: 0.3,  # Grasslands
}

# ── Natural Earth fallback URL ─────────────────────────────────────────────────

_NE_10M_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_0_map_units.zip"
)

# ── Backbone ──────────────────────────────────────────────────────────────────


def load_backbone(backbone_path: Path) -> tuple:
    """Return (profile, backbone_array, transform, shape, crs)."""
    if not backbone_path.exists():
        raise FileNotFoundError(f"Backbone raster not found: {backbone_path}")
    with rasterio.open(backbone_path) as src:
        profile = src.profile.copy()
        backbone = src.read(1)
        transform = src.transform
        shape = (src.height, src.width)
        crs = src.crs
    return profile, backbone, transform, shape, crs


# ── Country raster ────────────────────────────────────────────────────────────


def load_country_raster_from_policy(
    raster_path: Path, map_path: Path
) -> tuple[np.ndarray, dict[int, str]] | None:
    """Load pre-built country raster + ISO3 mapping.

    Returns None when either file is absent so the caller can fall back to the
    Natural Earth rasterisation path.
    """
    if not (raster_path.exists() and map_path.exists()):
        return None
    with rasterio.open(raster_path) as src:
        country_raster = src.read(1)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping_raw = json.load(f)
    id_to_iso3 = {int(k): v for k, v in mapping_raw.items()}
    return country_raster, id_to_iso3


def build_country_raster_from_natural_earth(
    backbone: np.ndarray,
    shape: tuple[int, int],
    transform: Any,
    crs: Any,
    territories: list[str],
    country_id: dict[int, str],
    iso3_col_preference: list[str],
) -> tuple[np.ndarray, dict[int, str]]:
    """Rasterise Natural Earth admin boundaries onto the backbone grid.

    Used as a fallback when the pre-built policy country raster is absent.
    ``territories`` is the list of ISO3 codes to include (used to filter NE).
    ``country_id`` maps integer pixel IDs to ISO3 codes.
    ``iso3_col_preference`` lists candidate column names in preferred order.
    """
    print("Building country raster from Natural Earth (fallback)...")
    try:
        gdf = gpd.read_file(_NE_10M_URL)
    except Exception:
        gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    iso3_col = next(
        (c for c in iso3_col_preference if c in gdf.columns), None
    )
    if iso3_col is None:
        raise ValueError(
            f"Could not find any of {iso3_col_preference} in Natural Earth data. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf_region = gdf[gdf[iso3_col].isin(territories)].copy()
    gdf_region = gdf_region[[iso3_col, "geometry"]].rename(
        columns={iso3_col: "iso3"}
    )
    if gdf_region.crs is None:
        gdf_region = gdf_region.set_crs("EPSG:4326")
    gdf_region = gdf_region.to_crs(crs)

    territory_to_id = {v: k for k, v in country_id.items()}
    shapes_to_rasterize = [
        (row.geometry, territory_to_id[row["iso3"]])
        for _, row in gdf_region.iterrows()
        if row["iso3"] in territory_to_id
        and row.geometry is not None
        and not row.geometry.is_empty
    ]

    country_raster = rasterize(
        shapes_to_rasterize,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    country_raster[backbone == 0] = 0
    return country_raster, country_id


# ── FAO data ──────────────────────────────────────────────────────────────────

# Direct mapping from FAO area names → ISO3.
# Covers all South American countries + USA. Extend if new regions are added.
_FAO_AREA_TO_ISO3: dict[str, str] = {
    "Argentina": "ARG",
    "Bolivia (Plurinational State of)": "BOL",
    "Brazil": "BRA",
    "Chile": "CHL",
    "Colombia": "COL",
    "Ecuador": "ECU",
    "French Guiana": "GUF",
    "Guyana": "GUY",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Suriname": "SUR",
    "Uruguay": "URY",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "United States of America": "USA",
}


def load_fao_value(
    value_path: Path,
    policy_iso3: set[str],
    extra_iso3_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load FAO gross production value (element 58) for the requested ISO3 set.

    Returns a DataFrame with columns: iso3, year, gross_value_usd.

    Name → ISO3 mapping uses the built-in ``_FAO_AREA_TO_ISO3`` table, which
    covers all South American territories including French Guiana (GUF) as a
    separate entry. ``extra_iso3_overrides`` may provide additional mappings
    (FAO area name → ISO3) not present in the built-in table.
    """
    df = pd.read_csv(value_path, dtype={"Area Code (M49)": str})
    df = df[df["Element Code"] == 58].copy()
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["gross_value_usd"] = pd.to_numeric(df["Value"], errors="coerce") * 1000.0

    lookup = {**_FAO_AREA_TO_ISO3, **(extra_iso3_overrides or {})}
    df["iso3"] = df["Area"].map(lookup)

    df = df[df["iso3"].isin(policy_iso3)].copy()
    df = df[["iso3", "year", "gross_value_usd"]].dropna(
        subset=["year", "gross_value_usd"]
    )
    df["year"] = df["year"].astype(int)
    return df


def build_fao_table(
    value_path: Path,
    policy_iso3: set[str],
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    extra_iso3_overrides: dict[str, str] | None = None,
    proxy_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Build a complete iso3 × year table of FAO gross production value (USD).

    Parameters
    ----------
    value_path:
        Path to ``value_agr.prod.csv`` (FAO).
    policy_iso3:
        Set of ISO3 codes for countries/territories to include.
    year_min, year_max:
        Inclusive temporal range.
    extra_iso3_overrides:
        Additional FAO area name → ISO3 mappings beyond the built-in table.
    proxy_map:
        Territories with no FAO data at all, keyed to a list of ISO3 codes
        whose annual mean is used as a substitute.
        Example: ``{"GUF": ["SUR", "GUY"]}``

    Returns
    -------
    DataFrame with columns: iso3, year, gross_value_usd (float64, no NaNs
    after interpolation/fill).
    """
    fao = load_fao_value(value_path, policy_iso3, extra_iso3_overrides)

    if proxy_map:
        for target, sources in proxy_map.items():
            if target not in policy_iso3:
                continue
            if fao[fao["iso3"] == target].empty:
                print(
                    f"  No FAO data for {target}. "
                    f"Using proxy: mean of {sources}."
                )
                proxy = (
                    fao[fao["iso3"].isin(sources)]
                    .groupby("year")["gross_value_usd"]
                    .mean()
                    .reset_index()
                )
                proxy["iso3"] = target
                fao = pd.concat([fao, proxy], ignore_index=True)

    full_idx = pd.MultiIndex.from_product(
        [sorted(policy_iso3), list(range(year_min, year_max + 1))],
        names=["iso3", "year"],
    )
    fao = fao.set_index(["iso3", "year"]).reindex(full_idx).reset_index()
    fao["gross_value_usd"] = (
        fao.groupby("iso3")["gross_value_usd"]
        .transform(
            lambda s: s.interpolate(limit_direction="both").ffill().bfill()
        )
        .astype(np.float64)
    )
    return fao


# ── Land-cover helpers ────────────────────────────────────────────────────────


def get_available_lc_years(lc_dir: Path) -> dict[int, Path]:
    """Scan *lc_dir* for GeoTIFF files and extract years from filenames.

    Supports any naming convention where the year is the last underscore-
    delimited token in the stem, e.g.:
      - ``landcover_2005.tif``
      - ``MODIS_landcover_SA_1km_2005.tif``

    Returns a dict mapping year → Path, sorted by year.
    """
    result: dict[int, Path] = {}
    for p in sorted(lc_dir.glob("*.tif")):
        try:
            year = int(p.stem.split("_")[-1])
            if 1990 <= year <= 2030:
                result[year] = p
        except ValueError:
            continue
    return result


def load_landcover_for_year(
    year: int,
    year_to_path: dict[int, Path],
    target_shape: tuple[int, int] | None = None,
    target_transform: Any = None,
    target_crs: Any = None,
) -> np.ndarray:
    """Load the MODIS land-cover array for *year*, reprojecting to the
    backbone grid when necessary.

    Falls back to the nearest available year if *year* is not present.
    When *target_shape* / *target_transform* / *target_crs* are provided and
    the source raster does not already match, the array is warped with
    nearest-neighbour resampling (correct for categorical IGBP data).
    """
    if not year_to_path:
        raise FileNotFoundError("No land-cover rasters found.")

    if year in year_to_path:
        path = year_to_path[year]
    else:
        nearest = min(year_to_path.keys(), key=lambda y: abs(y - year))
        print(
            f"  Warning: no land-cover raster for {year}, "
            f"using {nearest} as fallback."
        )
        path = year_to_path[nearest]

    with rasterio.open(path) as src:
        lc = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    needs_warp = (
        target_shape is not None
        and target_transform is not None
        and target_crs is not None
        and lc.shape != target_shape
    )
    if needs_warp:
        print(
            f"  Reprojecting land-cover {year}: "
            f"{lc.shape} → {target_shape}"
        )
        out = np.zeros(target_shape, dtype=lc.dtype)
        reproject(
            source=lc,
            destination=out,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )
        return out
    return lc


# ── Core raster writing ───────────────────────────────────────────────────────


def write_yearly_rasters(
    fao_df: pd.DataFrame,
    country_raster: np.ndarray,
    id_to_iso3: dict[int, str],
    lc_dir: Path,
    output_dir: Path,
    profile: dict[str, Any],
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> None:
    """Write one economic-value GeoTIFF per year and a validation CSV.

    For each agricultural pixel i in country c at year y the output value is:

        value_usd_ha_i = FAO_GPV(c, y) * LC_WEIGHT[lc_class_i]
                         / Σ_j (LC_WEIGHT[lc_class_j] * 100 ha)

    This guarantees that the sum of (value_usd_ha * 100 ha) across all
    agricultural pixels in a country equals the FAO national total exactly
    (subject only to float32 rounding in the stored raster).

    Pixel values:
      NaN   — outside all country boundaries (nodata)
      0.0   — inside a country boundary but not agricultural land
      > 0   — agricultural pixel with allocated USD/ha value
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
        count=1,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    year_to_lc_path = get_available_lc_years(lc_dir)
    if not year_to_lc_path:
        raise FileNotFoundError(f"No land-cover rasters found in {lc_dir}")

    backbone_shape = country_raster.shape
    backbone_transform = profile["transform"]
    backbone_crs = profile["crs"]
    valid_cids = {cid: iso3 for cid, iso3 in id_to_iso3.items() if cid != 0}
    validation_rows: list[dict] = []

    for year in range(year_min, year_max + 1):
        lc = load_landcover_for_year(
            year, year_to_lc_path,
            target_shape=backbone_shape,
            target_transform=backbone_transform,
            target_crs=backbone_crs,
        )

        # Weight array: 0 for non-agricultural classes, LC_WEIGHTS otherwise.
        weight = np.zeros(lc.shape, dtype=np.float32)
        for cls, w in LC_WEIGHTS.items():
            weight[lc == cls] = w

        # Initialise: NaN outside countries, 0 inside (non-ag pixels stay 0).
        raster = np.full(backbone_shape, np.nan, dtype=np.float32)
        raster[country_raster > 0] = 0.0

        fao_year = fao_df[fao_df["year"] == year].set_index("iso3")[
            "gross_value_usd"
        ]

        for cid, iso3 in valid_cids.items():
            country_mask = country_raster == cid
            w_country = weight[country_mask]

            # Weighted area in hectares (each 1 km² pixel = 100 ha).
            total_weight_ha = float(w_country.sum()) * 100.0

            gpv = fao_year.get(iso3, np.nan)

            n_crop = int(np.isin(lc[country_mask], list(CROPLAND_CLASSES)).sum())
            n_past = int(np.isin(lc[country_mask], list(PASTURE_CLASSES)).sum())

            if np.isnan(gpv) or total_weight_ha == 0.0:
                if total_weight_ha == 0.0 and not np.isnan(gpv):
                    print(
                        f"  Warning: {iso3} {year} — FAO value present but "
                        "0 agricultural pixels found; all pixels set to 0."
                    )
                validation_rows.append(
                    _validation_row(
                        iso3, year, gpv, 0.0, n_crop, n_past, 0.0
                    )
                )
                continue

            # Spatial allocation.
            values = (gpv * w_country) / total_weight_ha  # USD/ha per pixel
            raster[country_mask] = values.astype(np.float32)

            # Validation (check round-trip: sum(value * 100 ha) ≈ GPV).
            raster_total = float(values.sum()) * 100.0
            ag_vals = values[w_country > 0]
            validation_rows.append(
                _validation_row(
                    iso3,
                    year,
                    gpv,
                    raster_total,
                    n_crop,
                    n_past,
                    float(ag_vals.mean()) if len(ag_vals) else 0.0,
                )
            )

        out_path = output_dir / f"economic_value_{year}.tif"
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(raster, 1)
        print(f"  {year}: wrote {out_path.name}")

    val_df = pd.DataFrame(validation_rows)
    val_path = output_dir / "validation_totals.csv"
    val_df.to_csv(val_path, index=False)
    print(
        f"\nWrote {year_max - year_min + 1} rasters and validation CSV "
        f"to {output_dir}"
    )


def _validation_row(
    iso3: str,
    year: int,
    fao_value_usd: float,
    raster_total_usd: float,
    n_cropland: int,
    n_pasture: int,
    mean_value_usd_ha: float,
) -> dict:
    rel_err = (
        abs(raster_total_usd - fao_value_usd) / fao_value_usd * 100.0
        if fao_value_usd and fao_value_usd > 0
        else np.nan
    )
    return {
        "iso3": iso3,
        "year": year,
        "fao_value_usd": fao_value_usd,
        "raster_total_usd": raster_total_usd,
        "rel_error_pct": rel_err,
        "n_cropland_pixels": n_cropland,
        "n_pasture_pixels": n_pasture,
        "ag_area_ha": (n_cropland + n_pasture) * 100.0,
        "mean_value_usd_ha": mean_value_usd_ha,
    }
