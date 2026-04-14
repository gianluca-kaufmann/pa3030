#!/usr/bin/env python3
"""Regenerate risk map and probability map for South America without running the full pipeline.

Usage:
    conda run -n pa3030 python scripts/regen_maps.py
"""
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["PA3030_RESULTS_REGION"] = "south_america"

import importlib
import scripts.regions.shared.results.config  # noqa: F401 — sets region globals
importlib.import_module("scripts.regions.shared.results.config")

from scripts.regions.shared.results.results_core import (
    load_metrics_json,
    load_scored_parquet,
    derive_test_years,
    detect_and_reproject_coordinates,
    validate_coordinates,
    get_region_boundary,
    find_backbone_path,
    create_risk_map,
    create_probability_map,
    resolve_parquet_file,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_JSON = REPO_ROOT / "outputs/south_america/results/ml_models/model1_lgbm_metrics_20260406_165138.json"
SCORED_PARQUET = REPO_ROOT / "outputs/south_america/results/ml_models/main/model1_lgbm_scored_calibrated_isotonic_20260408_045921.parquet"
OUTPUT_DIR = REPO_ROOT / "outputs/south_america/results/model1_lgbm"
MODEL_TYPE = "lgbm"

print("=" * 60)
print("REGENERATING RISK MAP + PROBABILITY MAP")
print("=" * 60)
print(f"Metrics:  {METRICS_JSON}")
print(f"Parquet:  {SCORED_PARQUET}")
print(f"Output:   {OUTPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading metrics...")
metrics_data = load_metrics_json(METRICS_JSON)

# Try to resolve test parquet
try:
    test_parquet_path = resolve_parquet_file("test_win5.parquet", "main")
    print(f"Test parquet: {test_parquet_path}")
except FileNotFoundError:
    test_parquet_path = None
    print("Test parquet: not found (maps will use y_true only)")

test_years = derive_test_years(metrics_data, None, test_parquet_path)
print(f"Test years: {test_years}")

print("\nLoading scored parquet...")
df = load_scored_parquet(SCORED_PARQUET, test_years=test_years)
df = detect_and_reproject_coordinates(df)
validate_coordinates(df)
print(f"Loaded {len(df):,} rows")

# ── Region boundary & backbone ────────────────────────────────────────────────
print("\nLoading region boundary...")
sa_gdf = get_region_boundary(None)

print("Resolving backbone raster...")
backbone_path = find_backbone_path()

# ── Risk map (top 1% only — the one used in the paper) ───────────────────────
print("\n--- Risk Map (top 1%) ---")
create_risk_map(
    df,
    None,           # region_boundary_path (uses Natural Earth)
    OUTPUT_DIR,
    MODEL_TYPE,
    metrics_data,
    test_parquet_path,
    test_years=test_years,
    threshold_pct=1,
    future_parquet_path=None,
    future_years=None,
    sa_gdf=sa_gdf,
    backbone_path=backbone_path,
)

# ── Probability map ───────────────────────────────────────────────────────────
print("\n--- Probability Map ---")
create_probability_map(
    df,
    None,           # region_boundary_path (uses Natural Earth)
    OUTPUT_DIR,
    MODEL_TYPE,
    SCORED_PARQUET,
    metrics_data,
    test_years=test_years,
    test_parquet_path=test_parquet_path,
    sa_gdf=sa_gdf,
    backbone_path=backbone_path,
)

print("\nDone.")
