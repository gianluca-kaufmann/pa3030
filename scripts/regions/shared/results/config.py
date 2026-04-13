from __future__ import annotations

import os
from pathlib import Path

RUN_REGION = os.environ.get("PA3030_RESULTS_REGION", "south_america").strip().lower()
if RUN_REGION not in {"south_america", "usa", "se_asia"}:
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
        # Map annotation layout: SA has most whitespace in the bottom-right (Atlantic Ocean)
        "map_inset_side": "right",
        # Font sizes for map figures — both boxes use the same size for visual consistency
        "map_legend_fontsize": 16,
        "map_legend_markersize": 18,
        "map_stats_fontsize": 16,
        # Risk map colors — SA: blue=hit, gold=predicted-only, red=miss
        "risk_map_predicted_color": "#FFC000", # predicted only, no PA established → gold
        "risk_map_overlap_color":   "#4472C4", # correctly predicted (hit)         → slate blue
        "risk_map_miss_color":      "#C0392B", # established, not predicted (miss)  → pomegranate red
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
        "map_inset_side": "left",
        "map_legend_fontsize": 14,
        "map_legend_markersize": 16,
        "map_stats_fontsize": 12,
        # Risk map colors — consistent with SA: gold=predicted-only, blue=hit, red=miss
        "risk_map_predicted_color": "#FFC000", # predicted only, no PA established → gold
        "risk_map_overlap_color":   "#4472C4", # correctly predicted (hit)         → slate blue
        "risk_map_miss_color":      "#C0392B", # established, not predicted (miss)  → pomegranate red
    },
    "se_asia": {
        "region_slug": "se_asia",
        "region_label": "South East Asia",
        "model_id": "model3",
        "model_label": "Model 3",
        "x_limits": (90, 145),
        "y_limits": (-11, 28),
        "iso_codes": ['BRN', 'KHM', 'IDN', 'LAO', 'MYS', 'MMR', 'PHL', 'SGP', 'THA', 'TLS', 'VNM'],
        "country_names": ['Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam'],
        "probability_map_percentile_min": 25,
        "probability_map_percentile_max": 98,
        "probability_map_transformation": "sqrt",
        "hotspot_regions": [
            ("Borneo / Kalimantan",  108, 118,  -4,  7),
            ("Mekong / Indochina",   100, 110,   9, 22),
            ("Sumatra / Malay Pen.",  95, 110,  -6,  6),
        ],
        "map_inset_side": "left",
        "map_legend_fontsize": 14,
        "map_legend_markersize": 16,
        "map_stats_fontsize": 12,
        # Risk map colors — consistent with SA: gold=predicted-only, blue=hit, red=miss
        "risk_map_predicted_color": "#FFC000", # predicted only, no PA established → gold
        "risk_map_overlap_color":   "#4472C4", # correctly predicted (hit)         → slate blue
        "risk_map_miss_color":      "#C0392B", # established, not predicted (miss)  → pomegranate red
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
MAP_INSET_SIDE         = PROFILE["map_inset_side"]        # 'right' or 'left'
MAP_LEGEND_FONTSIZE    = PROFILE["map_legend_fontsize"]
MAP_LEGEND_MARKERSIZE  = PROFILE["map_legend_markersize"]
MAP_STATS_FONTSIZE     = PROFILE["map_stats_fontsize"]
RISK_MAP_OVERLAP_COLOR = PROFILE["risk_map_overlap_color"]
RISK_MAP_PREDICTED_COLOR = PROFILE["risk_map_predicted_color"]
RISK_MAP_MISS_COLOR      = PROFILE["risk_map_miss_color"]
if MODEL_ID == "model1":
    CALIBRATE_SCRIPT = "calibrate_1"
elif MODEL_ID == "model2":
    CALIBRATE_SCRIPT = "calibrate_2"
else:
    CALIBRATE_SCRIPT = "calibrate_3"
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
