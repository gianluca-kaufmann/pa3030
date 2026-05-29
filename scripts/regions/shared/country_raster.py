"""Country raster lookup utilities for pre-flight checks and Stage 2 grouping.

Mirrors the W3 logic in regional ``feature_engineering`` scripts so diagnostics
can attach ``country_id`` without rerunning feature engineering.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "README.md").is_file():
            return parent
    raise RuntimeError("Cannot find repo root (no README.md found upward from script).")


def scratch_root() -> Path | None:
    scratch = os.environ.get("SCRATCH")
    return Path(scratch) if scratch else None


def load_country_raster(region: str) -> np.ndarray:
    """Load country_iso3.tif for *region* (uint8: row,col -> country_id)."""
    import rasterio

    root = repo_root()
    scratch = scratch_root()
    candidates: list[Path] = []
    if scratch is not None:
        candidates.append(scratch / f"data/{region}/ready/policy/country_iso3.tif")
    candidates.append(root / f"data/{region}/ready/policy/country_iso3.tif")
    candidates.append(root / "data/shared/country_iso3.tif")
    for path in candidates:
        if path.exists():
            with rasterio.open(path) as ds:
                return ds.read(1)
    raise FileNotFoundError(
        "country_iso3.tif not found. Searched:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def country_ids_for_rows(df: pd.DataFrame, country_raster: np.ndarray) -> np.ndarray:
    """Look up country_id per pixel via raster[(row, col)] indexing."""
    rows = df["row"].to_numpy(dtype=np.int64)
    cols = df["col"].to_numpy(dtype=np.int64)
    h, w = country_raster.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.zeros(len(df), dtype=np.uint8)
    out[valid] = country_raster[rows[valid], cols[valid]].astype(np.uint8)
    return out


def load_ecoregion_raster(region: str) -> np.ndarray:
    """Load gsn_terrestrial_ecoregions_mask_1km.tif for *region* (int16: row,col -> eco_id).

    Returns 0 for nodata pixels. SA values are multiples of 4 (63 ecoregions);
    USA/SE Asia are 1-based uint8 stored as int16.
    """
    import rasterio

    scratch = scratch_root()
    root = repo_root()
    candidates: list[Path] = []
    if scratch is not None:
        candidates.append(scratch / f"data/{region}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif")
    candidates.append(root / f"data/{region}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif")
    for path in candidates:
        if path.exists():
            with rasterio.open(path) as ds:
                return ds.read(1).astype(np.int16)
    raise FileNotFoundError(
        "gsn_terrestrial_ecoregions_mask_1km.tif not found. Searched:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def ecoregion_ids_for_rows(df: pd.DataFrame, eco_raster: np.ndarray) -> np.ndarray:
    """Look up ecoregion_id per pixel via raster[(row, col)] indexing.

    Returns int16 array; 0 = nodata (out-of-bounds or raster zero).
    """
    rows = df["row"].to_numpy(dtype=np.int64)
    cols = df["col"].to_numpy(dtype=np.int64)
    h, w = eco_raster.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.zeros(len(df), dtype=np.int16)
    out[valid] = eco_raster[rows[valid], cols[valid]]
    return out


def resolve_panel_path(region: str) -> Path:
    """Locate merged_panel_final.parquet for *region* (SCRATCH first).

    On Euler panels live at $SCRATCH/data/{region}/ml/; locally they may
    also appear under outputs/{region}/results/ (legacy location).
    """
    root = repo_root()
    scratch = scratch_root()
    candidates: list[Path] = []
    if scratch is not None:
        candidates.append(scratch / f"data/{region}/ml/merged_panel_final.parquet")
        candidates.append(scratch / f"outputs/{region}/results/merged_panel_final.parquet")
    candidates.append(root / f"outputs/{region}/results/merged_panel_final.parquet")
    candidates.append(root / f"data/{region}/ml/merged_panel_final.parquet")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"merged_panel_final.parquet not found for region={region!r}. Searched:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )
