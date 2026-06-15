"""Ecoregion Protection Gap — compute lookup table for South America.

For each (ecoregion_id, year) pair, computes the fraction of the ecoregion that
was protected in the *previous* year (lag-1, leakage-free).  The resulting lookup
is joined to the mini-sample / full splits by (GSN ecoregion ID, year).

Two derived columns are produced:
  eco_protection_ratio_lag1  — fraction of ecoregion protected at year t-1  (0-1)
  eco_protection_gap_lag1    — 30x30 gap: max(0, 0.30 - ratio)             (0-0.30)

The second column directly encodes policy pressure: ecoregions far from 30%
coverage face stronger designation incentives.

Inputs (all already present locally):
  data/south_america/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif
  data/south_america/ready/WDPA/WDPA_SA_1km_{2000..2022}.tif
  data/south_america/ready/backbone/backbone.tif  (for land mask)

Output:
  data/south_america/ready/eco_protection_gap/eco_protection_gap_lookup.parquet
    Columns: ecoregion_id (int), panel_year (int),
             eco_protection_ratio_lag1 (float32), eco_protection_gap_lag1 (float32)

Usage:
  python scripts/regions/south_america/2_preprocessing/ecoregion_protection_gap_rasterise.py

  Then inject into mini_sample:
  python scripts/regions/south_america/3_merging/add_ecoregion_protection_gap_to_mini_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ECO_TIF = PROJECT_ROOT / "data/south_america/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif"
BACKBONE_TIF = PROJECT_ROOT / "data/south_america/ready/backbone/backbone.tif"
WDPA_DIR = PROJECT_ROOT / "data/south_america/ready/WDPA"
OUT_DIR = PROJECT_ROOT / "data/south_america/ready/eco_protection_gap"
OUT_PATH = OUT_DIR / "eco_protection_gap_lookup.parquet"

# Panel years in the SA dataset (mini-sample covers 2001-2013)
# We need WDPA_{year-1} for each panel year, so WDPA 2000-2022.
WDPA_YEARS = list(range(2000, 2023))   # WDPA years (lag); panel_year = wdpa_year + 1
TARGET_30X30 = 0.30
N_ECO = 138  # ecoregion IDs 0 (nodata) through 137


def _load_eco_grid() -> np.ndarray:
    with rasterio.open(ECO_TIF) as src:
        grid = src.read(1)   # uint8, values 0-137
    return grid


def _load_backbone() -> np.ndarray:
    with rasterio.open(BACKBONE_TIF) as src:
        bb = src.read(1)     # uint8: 1=land, 0=nodata/ocean
    return bb


def _load_wdpa(year: int) -> np.ndarray | None:
    path = WDPA_DIR / f"WDPA_SA_1km_{year}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    return data


def _compute_year_ratios(
    eco_slice: np.ndarray,
    wdpa_slice: np.ndarray,
    eco_totals: np.ndarray,
) -> np.ndarray:
    """Return per-ecoregion protection ratio for the given WDPA year.

    eco_slice: (h, w) uint8 ecoregion IDs, cropped to WDPA overlap
    wdpa_slice: (h, w) float32 WDPA values; >0 = protected, nan = nodata
    eco_totals: (N_ECO,) total pixel count per ecoregion in overlap region
    """
    protected = np.where(np.isfinite(wdpa_slice), wdpa_slice > 0, False).astype(np.float32)
    eco_flat = eco_slice.ravel().astype(np.int32)

    eco_protected = np.bincount(eco_flat, weights=protected.ravel(), minlength=N_ECO)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(eco_totals > 0, eco_protected / eco_totals, np.nan)
    return ratio.astype(np.float32)


def main() -> None:
    if not ECO_TIF.exists():
        sys.exit(f"GSN ecoregion TIF not found: {ECO_TIF}")
    if not BACKBONE_TIF.exists():
        sys.exit(f"Backbone TIF not found: {BACKBONE_TIF}")

    print("Loading GSN ecoregion grid...")
    eco_grid = _load_eco_grid()     # (9172, 8974)
    print(f"  Shape: {eco_grid.shape}, unique IDs: {len(np.unique(eco_grid))}")

    print("Loading backbone land mask...")
    backbone = _load_backbone()

    # Compute eco_totals in the WDPA overlap region (rows 0:9047 of backbone).
    # WDPA TIFs are (9047, 8975); backbone is (9172, 8974).
    # Intersection: rows 0:9047, cols 0:8974.
    h_wdpa, w_wdpa = 9047, 8975
    h_int = min(eco_grid.shape[0], h_wdpa)
    w_int = min(eco_grid.shape[1], w_wdpa)

    eco_int = eco_grid[:h_int, :w_int]
    backbone_int = backbone[:h_int, :w_int]

    # Count total LAND pixels per ecoregion in the intersection region.
    # Use land mask to exclude ocean/nodata pixels from the denominator.
    eco_land = np.where(backbone_int == 1, eco_int, 0).astype(np.int32)
    eco_totals = np.bincount(eco_land.ravel(), minlength=N_ECO).astype(np.float64)
    eco_totals[0] = 0  # ecoregion 0 = nodata; exclude from ratios

    n_land = int((backbone_int == 1).sum())
    print(f"  Land pixels in overlap region: {n_land:,}")
    print(f"  Ecoregions with >0 land pixels: {(eco_totals > 0).sum() - 1}")

    rows: list[dict] = []
    for wdpa_year in WDPA_YEARS:
        wdpa_raw = _load_wdpa(wdpa_year)
        if wdpa_raw is None:
            print(f"  SKIP: WDPA_{wdpa_year} not found")
            continue

        # Crop to intersection
        wdpa_int = wdpa_raw[:h_int, :w_int]

        # Apply land mask: treat non-land pixels as unprotected
        wdpa_land = np.where(backbone_int == 1, wdpa_int, 0.0)

        ratio = _compute_year_ratios(eco_int, wdpa_land, eco_totals)

        panel_year = wdpa_year + 1
        n_protected = int((wdpa_land > 0).sum())
        mean_ratio = float(np.nanmean(ratio[1:]))  # skip eco 0

        for eco_id in range(1, N_ECO):
            r = float(ratio[eco_id])
            if not np.isnan(r):
                rows.append({
                    "ecoregion_id": eco_id,
                    "panel_year": panel_year,
                    "eco_protection_ratio_lag1": r,
                    "eco_protection_gap_lag1": max(0.0, TARGET_30X30 - r),
                })

        print(f"  WDPA {wdpa_year} → panel year {panel_year}: "
              f"{n_protected:,} protected, mean eco ratio={mean_ratio:.3f}")

    df = pd.DataFrame(rows)
    df["ecoregion_id"] = df["ecoregion_id"].astype(np.int16)
    df["panel_year"] = df["panel_year"].astype(np.int16)
    df["eco_protection_ratio_lag1"] = df["eco_protection_ratio_lag1"].astype(np.float32)
    df["eco_protection_gap_lag1"] = df["eco_protection_gap_lag1"].astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nLookup saved: {OUT_PATH}")
    print(f"  Rows: {len(df):,}  (ecoregions × panel years)")
    print(f"  eco_protection_ratio_lag1: [{df['eco_protection_ratio_lag1'].min():.3f}, "
          f"{df['eco_protection_ratio_lag1'].max():.3f}]")
    print(f"  eco_protection_gap_lag1: [{df['eco_protection_gap_lag1'].min():.3f}, "
          f"{df['eco_protection_gap_lag1'].max():.3f}]")
    print()
    print("Next:")
    print("  python scripts/regions/south_america/3_merging/"
          "add_ecoregion_protection_gap_to_mini_sample.py")


if __name__ == "__main__":
    main()
