from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq

from scripts.regions.shared.results.config import (
    DEFAULT_FUTURE_PARQUET_FILENAME,
    FALLBACK_FUTURE_PARQUET_FILENAME,
    MODEL_ID,
    REGION_SLUG,
    get_repo_root,
)

# Module-level caches for parquet loaders.
_WDPA_TEST_CACHE: Dict[tuple, Any] = {}
_FUTURE_PA_CACHE: Dict[tuple, Any] = {}

def resolve_parquet_file(filename: str, split_version: str = 'main') -> Path:
    """Locate parquet file using $SCRATCH-first path resolution (same as training scripts)."""
    repo_root = get_repo_root()
    scratch_root = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    candidates = []
    if scratch_root is not None:
        candidates.append(scratch_root / f"data/{REGION_SLUG}/ml/{split_version}/{filename}")
        candidates.append(scratch_root / f"data/{REGION_SLUG}/ml/{filename}")
        candidates.append(scratch_root / f"outputs/{REGION_SLUG}/results/{split_version}/{filename}")
    candidates.append(repo_root / f"data/{REGION_SLUG}/ml/{split_version}/{filename}")
    candidates.append(repo_root / f"data/{REGION_SLUG}/ml/{filename}")
    candidates.append(repo_root / f"outputs/{REGION_SLUG}/results/{split_version}/{filename}")
    candidates.append(repo_root / f"outputs/{REGION_SLUG}/results/{filename}")

    for cand in candidates:
        if cand.exists():
            return cand

    error_msg = f"{filename} not found in expected locations.\n"
    error_msg += "\nChecked locations (in order):\n"
    for i, cand in enumerate(candidates, 1):
        exists = "✓" if cand.exists() else "✗"
        error_msg += f"  {i}. {exists} {cand}\n"
    if scratch_root:
        error_msg += f"\nSCRATCH root: {scratch_root}\n"
    error_msg += f"Repository root: {repo_root}\n"
    error_msg += f"Split version: {split_version}\n"
    raise FileNotFoundError(error_msg)

def resolve_future_parquet(split_version: str = 'main') -> Path:
    """Locate future-period parquet for temporal validation (2020-2024 PA establishments)."""
    repo_root = get_repo_root()
    scratch_root = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None
    tried = []

    try:
        p = resolve_parquet_file(DEFAULT_FUTURE_PARQUET_FILENAME, split_version)
        return p
    except FileNotFoundError:
        pass
    for base in ([scratch_root] if scratch_root else []):
        for sub in [f"data/{REGION_SLUG}/ml/{split_version}", f"data/{REGION_SLUG}/ml", f"outputs/{REGION_SLUG}/results/{split_version}"]:
            tried.append(base / sub / DEFAULT_FUTURE_PARQUET_FILENAME)
    for sub in [f"data/{REGION_SLUG}/ml/{split_version}", f"data/{REGION_SLUG}/ml", f"outputs/{REGION_SLUG}/results/{split_version}", f"outputs/{REGION_SLUG}/results"]:
        tried.append(repo_root / sub / DEFAULT_FUTURE_PARQUET_FILENAME)

    try:
        p = resolve_parquet_file(FALLBACK_FUTURE_PARQUET_FILENAME, split_version)
        return p
    except FileNotFoundError:
        pass
    for base in ([scratch_root] if scratch_root else []):
        for sub in [f"data/{REGION_SLUG}/ml/{split_version}", f"data/{REGION_SLUG}/ml", f"outputs/{REGION_SLUG}/results/{split_version}"]:
            tried.append(base / sub / FALLBACK_FUTURE_PARQUET_FILENAME)
    for sub in [f"data/{REGION_SLUG}/ml/{split_version}", f"data/{REGION_SLUG}/ml", f"outputs/{REGION_SLUG}/results/{split_version}", f"outputs/{REGION_SLUG}/results"]:
        tried.append(repo_root / sub / FALLBACK_FUTURE_PARQUET_FILENAME)

    if scratch_root is not None:
        cand = scratch_root / "data" / FALLBACK_FUTURE_PARQUET_FILENAME
        tried.append(cand)
        if cand.exists():
            return cand
    cand = repo_root / "data" / FALLBACK_FUTURE_PARQUET_FILENAME
    tried.append(cand)
    if cand.exists():
        return cand

    error_msg = (
        f"{DEFAULT_FUTURE_PARQUET_FILENAME} and {FALLBACK_FUTURE_PARQUET_FILENAME} not found in expected locations.\n"
        "Checked: val_win5.parquet and merged_panel_final.parquet in ML paths; "
        "merged_panel_final.parquet in $SCRATCH/data/ and repo/data/.\n"
    )
    error_msg += "\nTried (examples):\n"
    for c in tried[:20]:
        error_msg += f"  - {c}\n"
    if len(tried) > 20:
        error_msg += f"  ... and {len(tried) - 20} more.\n"
    if scratch_root:
        error_msg += f"\nSCRATCH root: {scratch_root}\n"
    error_msg += f"Repository root: {repo_root}\n"
    raise FileNotFoundError(error_msg)

def load_metrics_json(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def check_calibration_status(parquet_path: Path, available_columns: list) -> None:
    """Check if the parquet file contains calibrated predictions and print status.
    
    Args:
        parquet_path: Path to the parquet file
        available_columns: List of available column names in the parquet file
    """
    # Check for calibration indicators
    has_uncalibrated_col = 'y_pred_proba_uncalibrated' in available_columns
    has_calibrated_col = 'y_pred_proba_calibrated' in available_columns
    filename_has_calibrated = 'calibrated' in parquet_path.name.lower()
    
    # Determine calibration status
    if has_uncalibrated_col and has_calibrated_col:
        # This is definitely a calibrated file
        print(f"\n{'='*70}")
        print("CALIBRATION STATUS: CALIBRATED")
        print(f"{'='*70}")
        print(f"  File contains both 'y_pred_proba_uncalibrated' and 'y_pred_proba_calibrated' columns")
        print(f"  Using calibrated probabilities (y_pred_proba = calibrated values)")
        print(f"{'='*70}\n")
    elif filename_has_calibrated:
        # Filename suggests calibrated but columns don't confirm
        print(f"\n{'!'*70}")
        print("WARNING: Filename suggests calibrated predictions, but calibration columns not found")
        print(f"{'!'*70}")
        print(f"  Filename: {parquet_path.name}")
        print(f"  Expected columns: 'y_pred_proba_uncalibrated', 'y_pred_proba_calibrated'")
        print(f"  Available columns: {', '.join(available_columns)}")
        print(f"  Proceeding with available 'y_pred_proba' column")
        print(f"{'!'*70}\n")
    else:
        # No calibration indicators - uncalibrated
        print(f"\n{'!'*70}")
        print("WARNING: You are plotting UNCALIBRATED probabilities")
        print(f"{'!'*70}")
        print(f"  File: {parquet_path.name}")
        print(f"  No calibration metadata found (no 'y_pred_proba_uncalibrated'/'y_pred_proba_calibrated' columns)")
        print(f"  For calibrated results, use: {MODEL_ID}_lgbm_scored_calibrated_*.parquet (LGBM) or {MODEL_ID}_rf_win5_scored_calibrated_*.parquet (RF)")
        print(f"{'!'*70}\n")

def load_scored_parquet(parquet_path: Path, test_years: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Load scored parquet file with only needed columns using pyarrow.
    
    Reads: y_true, y_pred_proba, x, y (with column name resolution)
    Optionally reads: wdpa_b1 (protected area coverage as of end of 2017)
    Also reads: year, row, col if available (needed for joining with WDPA data)
    
    If test_years is provided and the file has a year column, uses predicate pushdown
    to load only rows matching those years (reduces I/O and RAM for large datasets).
    """
    print(f"Reading parquet file: {parquet_path}")
    pfile = pq.ParquetFile(parquet_path)  # Used only for schema / calibration check
    
    # Read schema to get available columns
    # ParquetSchema doesn't have field() method, use schema_arrow to get Arrow schema
    arrow_schema = pfile.schema_arrow
    available_columns = arrow_schema.names
    print(f"  Available columns: {available_columns}")
    
    # Check calibration status
    check_calibration_status(parquet_path, available_columns)
    
    # Map required logical fields with fallbacks
    field_mappings = {
        'y_true': ['y_true', 'transition_01_win5', 'transition_01'],
        'y_pred_proba': ['y_pred_proba', 'pred_proba', 'score', 'p_pred', 'proba'],
        'x': ['x', 'lon', 'longitude'],
        'y': ['y', 'lat', 'latitude']
    }
    
    # Optional fields: protected area coverage, and metadata for joining
    wdpa_col = None
    for candidate in ['wdpa_b1', 'WDPA_b1']:
        if candidate in available_columns:
            wdpa_col = candidate
            print(f"  Found protected area column: '{wdpa_col}'")
            break
    
    # Optional metadata columns for joining with original test data
    optional_metadata_cols = []
    for col in ['year', 'row', 'col']:
        if col in available_columns:
            optional_metadata_cols.append(col)
            print(f"  Found metadata column: '{col}'")
    
    # Resolve column names
    resolved_columns = {}
    missing_fields = []
    
    for logical_name, candidates in field_mappings.items():
        found = None
        for candidate in candidates:
            if candidate in available_columns:
                found = candidate
                break
        if found:
            resolved_columns[logical_name] = found
            print(f"  Resolved '{logical_name}' -> '{found}'")
        else:
            missing_fields.append((logical_name, candidates))
    
    # Check for missing fields
    if missing_fields:
        error_msg = "ERROR: Required columns not found in parquet file.\n\n"
        error_msg += "Missing fields:\n"
        for logical_name, candidates in missing_fields:
            error_msg += f"  - {logical_name} (tried: {', '.join(candidates)})\n"
        error_msg += f"\nAvailable columns: {', '.join(available_columns)}\n"
        error_msg += "\nPlease ensure the parquet file contains one of the expected column names."
        raise ValueError(error_msg)
    
    # Read with pd.read_parquet (multi-threaded, predicate pushdown when possible)
    columns_to_read = list(resolved_columns.values()) + optional_metadata_cols
    if wdpa_col:
        columns_to_read.append(wdpa_col)
    
    # Predicate pushdown: filter by year to reduce I/O and RAM (critical for 47M+ rows)
    read_kwargs: Dict[str, Any] = {"columns": columns_to_read}
    if test_years is not None and len(test_years) > 0 and "year" in optional_metadata_cols:
        year_col = "year"  # We use logical name; parquet has same physical name
        read_kwargs["filters"] = [(year_col, "in", list(test_years))]
        print(f"  Reading parquet (columns={len(columns_to_read)}, filter year in {test_years})...")
    else:
        print(f"  Reading parquet (columns={len(columns_to_read)})...")
    result = pd.read_parquet(parquet_path, **read_kwargs)
    
    # Rename columns to standard names
    rename_dict = {v: k for k, v in resolved_columns.items()}
    result = result.rename(columns=rename_dict)
    
    # Rename wdpa column to standard name if present
    if wdpa_col:
        result = result.rename(columns={wdpa_col: 'wdpa_b1'})
    
    print(f"  Loaded {len(result):,} rows")
    print(f"  Final columns: {list(result.columns)}")
    
    return result

def derive_test_years(metrics_data: Optional[Dict[str, Any]] = None, 
                      parquet_path: Optional[Path] = None,
                      test_parquet_path: Optional[Path] = None) -> list:
    """Derive test years, prioritizing the original test parquet.
    
    Priority:
    1. Read unique years from the `year` column of `test_parquet_path`
    2. Use `metadata.test_year_range` or `metadata.test_years` from metrics
    3. Fallback to [2018, 2019] **only** if the test parquet is missing or unusable
    
    Args:
        metrics_data: Parsed metrics JSON (may contain `metadata` field)
        parquet_path: Kept for backward compatibility (no longer used for year inference)
        test_parquet_path: Path to original test parquet (preferred source of test years)
    
    Returns:
        List of test years (e.g., [2018, 2019])
    """
    # ------------------------------------------------------------------
    # 1) Prefer deriving years directly from the original test parquet
    #    (read only the `year` column, across all row groups)
    # ------------------------------------------------------------------
    parquet_unusable = False
    if test_parquet_path is None or not test_parquet_path.exists():
        parquet_unusable = True
    else:
        try:
            # Read only year column (fast, multi-threaded)
            df = pd.read_parquet(test_parquet_path, columns=['year'])
            if 'year' in df.columns:
                years_set = set(df['year'].dropna().unique())
                if years_set:
                    years = sorted(int(y) for y in years_set if pd.notna(y))
                    if years:
                        print(f"  Derived test years from test parquet: {years}")
                        return years
        except Exception:
            pass
        parquet_unusable = True
    
    # ------------------------------------------------------------------
    # 2) Fall back to metrics metadata
    # ------------------------------------------------------------------
    if metrics_data:
        metadata = metrics_data.get("metadata", {})
        test_year_range = metadata.get("test_year_range")
        if test_year_range and isinstance(test_year_range, list) and len(test_year_range) == 2:
            test_years = list(range(int(test_year_range[0]), int(test_year_range[1]) + 1))
            print(f"  Derived test years from metadata.test_year_range: {test_years}")
            return test_years
        
        test_years_meta = metadata.get("test_years")
        if test_years_meta and isinstance(test_years_meta, (list, tuple)):
            test_years = [int(y) for y in test_years_meta]
            print(f"  Derived test years from metadata.test_years: {test_years}")
            return test_years
    
    # ------------------------------------------------------------------
    # 3) Final fallback – only when test parquet is missing/unusable
    #    No WARNING printed to keep logs thesis-clean.
    # ------------------------------------------------------------------
    if parquet_unusable:
        return [2018, 2019]
    
    # In the extremely unlikely case that the parquet exists and is readable
    # but provided no usable years and no metadata is available, still return
    # a sensible default without emitting warnings.
    return [2018, 2019]

def load_future_pa_establishments(future_parquet_path: Optional[Path], future_years: list) -> Optional[pd.DataFrame]:
    """Load future PA establishments (e.g., 2020-2024) from validation/future parquet for temporal validation.
    Results are cached to avoid re-reading when create_risk_map is called multiple times.
    
    This function identifies pixels that were established AFTER the test period to assess the model's
    forward-looking predictive power. For example, if the model predicted high-risk areas in 2017-2019,
    this shows whether those predictions captured PAs established in 2020-2024.
    
    Args:
        future_parquet_path: Path to parquet file containing future period data (e.g., val_win5.parquet)
        future_years: List of years to consider as "future" (e.g., [2020, 2021, 2022, 2023, 2024])
    
    Returns:
        DataFrame with columns: row, col, x, y, established_in_future_period (or None if not available)
    """
    if future_parquet_path is None or not future_parquet_path.exists():
        return None
    
    cache_key = (str(future_parquet_path), tuple(sorted(future_years)))
    if cache_key in _FUTURE_PA_CACHE:
        return _FUTURE_PA_CACHE[cache_key]
    
    print(f"\nLoading future PA establishments from: {future_parquet_path}")
    try:
        pfile = pq.ParquetFile(future_parquet_path)
        available_columns = pfile.schema_arrow.names
        
        # Check for WDPA column (naming varies across datasets)
        wdpa_col = None
        for candidate in ['WDPA_b1', 'wdpa_b1', 'WDPA', 'wdpa']:
            if candidate in available_columns:
                wdpa_col = candidate
                break
        
        if wdpa_col is None:
            print(f"  WARNING: WDPA column not found. Available columns: {available_columns}")
            return None
        
        # Check for required join columns
        has_row_col = 'row' in available_columns and 'col' in available_columns
        has_xy = 'x' in available_columns and 'y' in available_columns
        
        if not has_row_col and not has_xy:
            print(f"  WARNING: Missing required join columns. Need either (row, col) or (x, y)")
            return None
        
        if 'year' not in available_columns:
            print(f"  WARNING: Missing 'year' column")
            return None
        
        # Build columns to read
        required_cols = ['year', wdpa_col]
        if has_row_col:
            required_cols.extend(['row', 'col'])
        if has_xy:
            required_cols.extend(['x', 'y'])
        
        # Predicate pushdown: load year in future_years + year before (needed for shift())
        years_to_load = [min(future_years) - 1] + list(future_years)
        read_kwargs: Dict[str, Any] = {"columns": required_cols}
        read_kwargs["filters"] = [("year", "in", years_to_load)]
        print(f"  Reading parquet (columns={len(required_cols)}, filter year in {years_to_load})...")
        result = pd.read_parquet(future_parquet_path, **read_kwargs)
        
        # Standardize column names
        if wdpa_col != 'WDPA_b1':
            result = result.rename(columns={wdpa_col: 'WDPA_b1'})
        
        # Fill NaN with 0 and convert to int
        result['WDPA_b1'] = result['WDPA_b1'].fillna(0).astype(int)
        
        # Identify pixels established in future period
        future_year_str = f"{min(future_years)}-{max(future_years)}"
        print(f"  Identifying pixels established in future period ({future_year_str})...")
        
        # Get unique identifier for each pixel
        if has_row_col:
            pixel_id_cols = ['row', 'col']
        else:
            pixel_id_cols = ['x', 'y']
        
        # Sort by pixel and year
        result = result.sort_values(pixel_id_cols + ['year'])
        
        # VECTORIZED: Use shift() to compare WDPA_b1 with previous year
        result['WDPA_b1_prev_year'] = result.groupby(pixel_id_cols)['WDPA_b1'].shift(1).fillna(0).astype(int)
        
        # Identify newly protected pixels in future years
        result['newly_protected_in_future'] = (
            (result['year'].isin(future_years)) & 
            (result['WDPA_b1'] == 1) & 
            (result['WDPA_b1_prev_year'] == 0)
        ).astype(int)
        
        # Create pixel-level flag
        estab_df = result.groupby(pixel_id_cols)['newly_protected_in_future'].max().reset_index()
        estab_df = estab_df[estab_df['newly_protected_in_future'] == 1].copy()
        estab_df['established_in_future_period'] = 1
        estab_df = estab_df[pixel_id_cols + ['established_in_future_period']]
        
        if len(estab_df) > 0:
            print(f"  ✓ Found {len(estab_df):,} pixels established in future period ({future_year_str})")
        else:
            print(f"  WARNING: No pixels found established in future period ({future_year_str})")
            estab_df = pd.DataFrame(columns=pixel_id_cols + ['established_in_future_period'])
        
        print(f"  Join method: {'row/col' if has_row_col else 'x/y'}")
        
        _FUTURE_PA_CACHE[cache_key] = estab_df
        return estab_df
    
    except Exception as e:
        print(f"  WARNING: Failed to load future PA establishments: {e}")
        return None

def load_wdpa_from_test_parquet(test_parquet_path: Optional[Path], test_years: list) -> Optional[pd.DataFrame]:
    """Load WDPA_b1 and WDPA_prev data from original test parquet file for joining with scored data.
    
    This is needed because transition_01_win5 (y_true) is a 5-year lookahead target that doesn't
    mark the actual year of establishment. For example, a pixel established in 2018 will have
    y_true=1 only for years before 2018, not for 2018 itself. By loading WDPA_b1 from the
    original test data, we can correctly identify pixels that were actually established in each year.
    
    Also loads WDPA_prev to distinguish between:
    - Already protected areas (WDPA_prev=1): Protected in previous year, defines risk set boundary
    - Newly protected areas (WDPA_prev=0, WDPA_b1=1): Became protected during test period
    
    Returns DataFrame with columns: row, col, year, x, y, WDPA_b1, WDPA_prev (or None if file not found/not available)
    Includes x, y coordinates as fallback for joining when row/col are missing in scored data.
    """
    if test_parquet_path is None or not test_parquet_path.exists():
        return None
    
    cache_key = (str(test_parquet_path), tuple(sorted(test_years)))
    if cache_key in _WDPA_TEST_CACHE:
        return _WDPA_TEST_CACHE[cache_key]
    
    print(f"\nLoading WDPA data from original test parquet: {test_parquet_path}")
    try:
        pfile = pq.ParquetFile(test_parquet_path)
        available_columns = pfile.schema_arrow.names
        
        # Check for WDPA column (naming varies across datasets)
        wdpa_col = None
        for candidate in ['WDPA_b1', 'wdpa_b1', 'WDPA', 'wdpa']:
            if candidate in available_columns:
                wdpa_col = candidate
                break
        
        if wdpa_col is None:
            print(f"  WARNING: WDPA column not found in test parquet. Available columns: {available_columns}")
            return None
        
        # Check for WDPA_prev column (optional, for two-layer protection visualization)
        wdpa_prev_col = None
        for candidate in ['WDPA_prev', 'wdpa_prev']:
            if candidate in available_columns:
                wdpa_prev_col = candidate
                break
        
        if wdpa_prev_col is not None:
            print(f"  Found WDPA_prev column: '{wdpa_prev_col}' (enables temporal protection visualization)")
        else:
            print(f"  Note: WDPA_prev column not found (single-layer protection visualization only)")
        
        # Check for required join columns - try row/col first, then x/y as fallback
        has_row_col = 'row' in available_columns and 'col' in available_columns
        has_xy = 'x' in available_columns and 'y' in available_columns
        
        if not has_row_col and not has_xy:
            print(f"  WARNING: Missing required join columns. Need either (row, col) or (x, y)")
            print(f"  Available columns: {available_columns}")
            return None
        
        if 'year' not in available_columns:
            print(f"  WARNING: Missing 'year' column for joining")
            return None
        
        # Build list of columns to read
        required_cols = ['year', wdpa_col]
        if wdpa_prev_col is not None:
            required_cols.append(wdpa_prev_col)
        if has_row_col:
            required_cols.extend(['row', 'col'])
        if has_xy:
            required_cols.extend(['x', 'y'])
        
        # Predicate pushdown: load test_years + year before (needed for shift())
        years_to_load = [min(test_years) - 1] + list(test_years)
        read_kwargs: Dict[str, Any] = {"columns": required_cols}
        read_kwargs["filters"] = [("year", "in", years_to_load)]
        print(f"  Reading parquet (columns={len(required_cols)}, filter year in {years_to_load})...")
        result = pd.read_parquet(test_parquet_path, **read_kwargs)
        
        # Standardize column names
        if wdpa_col != 'WDPA_b1':
            result = result.rename(columns={wdpa_col: 'WDPA_b1'})
        if wdpa_prev_col is not None and wdpa_prev_col != 'WDPA_prev':
            result = result.rename(columns={wdpa_prev_col: 'WDPA_prev'})
        
        # Fill NaN with 0 and convert to int
        result['WDPA_b1'] = result['WDPA_b1'].fillna(0).astype(int)
        if 'WDPA_prev' in result.columns:
            result['WDPA_prev'] = result['WDPA_prev'].fillna(0).astype(int)
        
        # Identify pixels established in test period using VECTORIZED approach
        # WDPA_b1 is cumulative (protected by that year), so we identify establishments by:
        # - WDPA_b1=1 in test years AND WDPA_b1=0 in previous year
        # Create a flag for pixels established in test period that can be joined to all years
        test_year_str = f"{min(test_years)}-{max(test_years)}"
        print(f"  Identifying pixels established in test period ({test_year_str}) using vectorized approach...")
        
        # Get unique identifier for each pixel (row/col or x/y)
        if has_row_col:
            pixel_id_cols = ['row', 'col']
        else:
            pixel_id_cols = ['x', 'y']
        
        # Sort by pixel and year for vectorized shift operation
        result = result.sort_values(pixel_id_cols + ['year'])
        
        # VECTORIZED: Use shift() to compare WDPA_b1 with previous year for same pixel
        # Group by pixel and shift WDPA_b1 to get previous year's status
        result['WDPA_b1_prev_year'] = result.groupby(pixel_id_cols)['WDPA_b1'].shift(1).fillna(0).astype(int)
        
        # Identify newly protected pixels: WDPA_b1=1 AND WDPA_b1_prev_year=0
        # Filter to only test years for robustness
        result['newly_protected_in_test'] = (
            (result['year'].isin(test_years)) & 
            (result['WDPA_b1'] == 1) & 
            (result['WDPA_b1_prev_year'] == 0)
        ).astype(int)
        
        # Create pixel-level flag: a pixel is "established in test period" if it was newly protected in ANY test year
        estab_df = result.groupby(pixel_id_cols)['newly_protected_in_test'].max().reset_index()
        estab_df = estab_df[estab_df['newly_protected_in_test'] == 1].copy()
        estab_df['established_in_test_period'] = 1
        estab_df = estab_df[pixel_id_cols + ['established_in_test_period']]
        
        if len(estab_df) > 0:
            print(f"  Found {len(estab_df):,} pixels established in test period ({test_year_str})")
        else:
            print(f"  WARNING: No pixels found established in test period ({test_year_str})")
            # Create empty DataFrame with correct columns
            estab_df = pd.DataFrame(columns=pixel_id_cols + ['established_in_test_period'])
        
        # Create pixel-level flag for newly protected in test period (for visualization)
        newly_protected_pixel_flag = result.groupby(pixel_id_cols)['newly_protected_in_test'].max().reset_index()
        newly_protected_pixel_flag = newly_protected_pixel_flag[newly_protected_pixel_flag['newly_protected_in_test'] == 1].copy()
        newly_protected_pixel_flag['newly_protected_in_test_period'] = 1
        newly_protected_pixel_flag = newly_protected_pixel_flag[pixel_id_cols + ['newly_protected_in_test_period']]
        
        # Merge both flags back into result
        result = result.merge(estab_df, on=pixel_id_cols, how='left')
        result['established_in_test_period'] = result['established_in_test_period'].fillna(0).astype(int)
        
        result = result.merge(newly_protected_pixel_flag, on=pixel_id_cols, how='left')
        result['newly_protected_in_test_period'] = result['newly_protected_in_test_period'].fillna(0).astype(int)
        
        # Clean up temporary column
        result = result.drop(columns=['WDPA_b1_prev_year', 'newly_protected_in_test'])
        
        print(f"  Loaded {len(result):,} rows with WDPA data")
        print(f"  Years: {sorted(result['year'].unique())}")
        print(f"  Protected pixels (WDPA_b1=1): {(result['WDPA_b1'] == 1).sum():,}")
        print(f"  Pixels with establishment flag: {(result['established_in_test_period'] == 1).sum():,}")
        
        # Report on temporal protection dynamics if WDPA_prev available
        if 'WDPA_prev' in result.columns:
            already_protected = ((result['WDPA_prev'] == 1) & (result['WDPA_b1'] == 1)).sum()
            newly_protected = ((result['WDPA_prev'] == 0) & (result['WDPA_b1'] == 1)).sum()
            print(f"  Temporal protection analysis:")
            print(f"    - Already protected (WDPA_prev=1): {already_protected:,}")
            print(f"    - Newly protected (WDPA_prev=0, WDPA_b1=1): {newly_protected:,}")
            print(f"    → Two-layer protection visualization will be used")
        
        if has_row_col:
            print(f"  Join method: row/col/year (with establishment flag)")
        elif has_xy:
            print(f"  Join method: x/y/year (with establishment flag)")
        
        _WDPA_TEST_CACHE[cache_key] = result
        return result
    
    except Exception as e:
        print(f"  WARNING: Failed to load WDPA data from test parquet: {e}")
        return None

def find_latest_file(pattern: str, search_dir: Path, split_version: str = 'main', model_type: str = 'lgbm') -> Optional[Path]:
    """Find the most recent file matching a pattern in a directory.
    
    Training scripts save to the root directory, not split-specific subdirectories.
    For metrics files, uses the correct pattern based on model type:
    - LGBM: model1_lgbm_metrics_*.json
    - RF: model1_rf_win5_metrics_*.json
    
    Args:
        pattern: Glob pattern to match (may be overridden for metrics files)
        search_dir: Base directory to search in (root ml_models directory)
        split_version: Split version ('main' or 'robustness') - unused, kept for API compatibility
        model_type: Model type ('lgbm' or 'rf') - used to determine correct metrics filename pattern
    
    Returns:
        Path to the most recent matching file, or None if not found
    """
    # For metrics files, use the correct pattern based on model type
    if 'metrics' in pattern and model_type:
        if model_type == 'lgbm':
            pattern = f"{MODEL_ID}_lgbm_metrics_*.json"
        elif model_type == 'rf':
            pattern = f"{MODEL_ID}_rf_win5_metrics_*.json"
    # For scored parquet files, use the correct pattern based on model type
    elif 'scored' in pattern and model_type:
        if model_type == 'lgbm':
            pattern = f"{MODEL_ID}_lgbm_scored_*.parquet"
        elif model_type == 'rf':
            pattern = f"{MODEL_ID}_rf_win5_scored_*.parquet"
    
    # Training scripts save to root directory, not split-specific subdirectory
    if not search_dir.exists():
        return None
    try:
        matches = list(search_dir.glob(pattern))
        if not matches:
            return None
        # Sort by modification time, most recent first
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]
    except (OSError, PermissionError) as e:
        print(f"  WARNING: Could not search in {search_dir}: {e}")
        return None

def find_latest_file_in_dirs(
    pattern: str,
    search_dirs: list[Path],
    exclude_substr: Optional[str] = None
) -> Optional[Path]:
    """Find the most recent file matching a pattern across multiple directories.
    
    This is used for calibrated scored files, which are saved in split-specific
    subdirectories (e.g., outputs/south_america/results/ml_models/main/), in addition to the
    root ml_models directory where training scripts write uncalibrated scores.
    
    Args:
        pattern: Glob pattern to match files (no automatic rewriting).
        search_dirs: List of directories to search in.
        exclude_substr: Optional substring to exclude from matches (case-insensitive).
    
    Returns:
        Path to most recent matching file, or None if no matches.
    """
    # IMPORTANT: Respect directory precedence.
    # We want split-specific directories to take priority over root when both
    # contain matches, and we must never accidentally select a cross-split file
    # just because it is newer.
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        try:
            matches = list(search_dir.glob(pattern))
        except (OSError, PermissionError) as e:
            print(f"  WARNING: Could not search in {search_dir}: {e}")
            continue
        
        if exclude_substr:
            matches = [m for m in matches if exclude_substr.lower() not in m.name.lower()]
        
        if not matches:
            continue
        
        # Within this directory, pick the most recent file, but never fall back
        # to later directories if we already have at least one match here.
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]
    
    return None
