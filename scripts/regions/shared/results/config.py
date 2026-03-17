from __future__ import annotations

import os
from pathlib import Path

RUN_REGION = os.environ.get("PA3030_RESULTS_REGION", "south_america").strip().lower()
if RUN_REGION not in {"south_america", "usa"}:
    raise ValueError(f"Unsupported PA3030_RESULTS_REGION='{RUN_REGION}'")

PROFILE = {
    "south_america": {
        "region_slug": "south_america",
        "region_label": "South America",
        "model_id": "model1",
        "model_label": "Model 1",
        "x_limits": (-85, -32),
        "y_limits": (-56, 13),
        "iso_codes": ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'GUF', 'GUY', 'PRY', 'PER', 'SUR', 'URY', 'VEN'],
        "country_names": ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'French Guiana', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'],
        # Probability map: percentile-clipped sqrt stretch works well for SA's skewed distribution
        "probability_map_percentile_min": 25,
        "probability_map_percentile_max": 98,
        "probability_map_transformation": "sqrt",
        # (label, lon_min, lon_max, lat_min, lat_max)
        "hotspot_regions": [
            ("Amazon Basin",              -72, -48, -12,  4),
            ("Cerrado / Atlantic Forest", -55, -36, -28, -8),
            ("Andes Foothills",           -81, -64, -22,  3),
        ],
    },
    "usa": {
        "region_slug": "usa",
        "region_label": "USA",
        "model_id": "model2",
        "model_label": "Model 2",
        "x_limits": (-125, -66),
        "y_limits": (24, 50),
        "iso_codes": ['USA'],
        "country_names": ['United States of America', 'United States'],
        # Probability map: log transform needed because calibration collapses >98% of values
        # to the same floor probability, making percentile-based linear/sqrt normalization fail.
        "probability_map_percentile_min": 0,
        "probability_map_percentile_max": 99.9,
        "probability_map_transformation": "log",
        # (label, lon_min, lon_max, lat_min, lat_max)
        "hotspot_regions": [
            ("Pacific Northwest", -125, -110, 43, 50),
            ("Rocky Mountains",   -115, -100, 36, 48),
            ("Southeast",          -92,  -78, 25, 35),
        ],
    },
}[RUN_REGION]

REGION_SLUG     = PROFILE["region_slug"]
REGION_LABEL    = PROFILE["region_label"]
MODEL_ID        = PROFILE["model_id"]
MODEL_LABEL     = PROFILE["model_label"]
X_LIMITS        = PROFILE["x_limits"]
Y_LIMITS        = PROFILE["y_limits"]
ISO_CODES       = PROFILE.get("iso_codes", [])
HOTSPOT_REGIONS = PROFILE["hotspot_regions"]
PROBABILITY_MAP_PERCENTILE_MIN = PROFILE["probability_map_percentile_min"]
PROBABILITY_MAP_PERCENTILE_MAX = PROFILE["probability_map_percentile_max"]
PROBABILITY_MAP_TRANSFORMATION = PROFILE["probability_map_transformation"]
CALIBRATE_SCRIPT = "calibrate_1" if MODEL_ID == "model1" else "calibrate_2"
DEFAULT_FUTURE_PARQUET_FILENAME = "val_win5.parquet"
FALLBACK_FUTURE_PARQUET_FILENAME = "merged_panel_final.parquet"
DEFAULT_FUTURE_YEARS_STR = "2020,2021,2022,2023,2024"


def get_repo_root() -> Path:
    """Get repository root directory."""
    env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    script_dir = Path(__file__).resolve().parent
    current = script_dir
    for _ in range(10):
        if (current / "README.md").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise RuntimeError(
        f"Repository root not found. Searched upward from {script_dir} for README.md.\n"
        "Set PROJECT_ROOT or REPO_ROOT environment variable, or ensure README.md exists in repo root."
    )
