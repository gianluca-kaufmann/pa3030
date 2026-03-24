from __future__ import annotations

import os
from pathlib import Path

RUN_REGION = os.environ.get("PA3030_FORWARD_REGION", "south_america").strip().lower()
if RUN_REGION not in {"south_america", "usa", "se_asia"}:
    raise ValueError(f"Unsupported PA3030_FORWARD_REGION='{RUN_REGION}'")

PROFILE = {
    "south_america": {
        "region_slug": "south_america",
        "region_label": "South America",
        "model_prefix": "model1",
        "x_limits": (-85, -32),
        "y_limits": (-56, 13),
        "iso_codes": [
            'ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'GUF', 'GUY',
            'PRY', 'PER', 'SUR', 'URY', 'VEN',
        ],
        "data_subdir": "south_america",
        "outputs_subdir": "south_america",
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
        "model_prefix": "model2",
        "x_limits": (-125, -66),
        "y_limits": (24, 50),
        "iso_codes": ['USA'],
        "data_subdir": "usa",
        "outputs_subdir": "usa",
        # (label, lon_min, lon_max, lat_min, lat_max)
        "hotspot_regions": [
            ("Pacific Northwest", -125, -110, 43, 50),
            ("Rocky Mountains",   -115, -100, 36, 48),
            ("Southeast",          -92,  -78, 25, 35),
        ],
    },
    "se_asia": {
        "region_slug": "se_asia",
        "region_label": "South East Asia",
        "model_prefix": "model3",
        "x_limits": (90, 145),
        "y_limits": (-11, 28),
        "iso_codes": [
            'BRN', 'KHM', 'IDN', 'LAO', 'MYS',
            'MMR', 'PHL', 'SGP', 'THA', 'TLS', 'VNM',
        ],
        "data_subdir": "se_asia",
        "outputs_subdir": "se_asia",
        # (label, lon_min, lon_max, lat_min, lat_max)
        "hotspot_regions": [
            ("Borneo / Kalimantan",  108, 118,  -4,  7),
            ("Mekong / Indochina",   100, 110,   9, 22),
            ("Sumatra / Malay Pen.",  95, 110,  -6,  6),
        ],
    },
}[RUN_REGION]

REGION_SLUG     = PROFILE["region_slug"]
REGION_LABEL    = PROFILE["region_label"]
MODEL_PREFIX    = PROFILE["model_prefix"]
X_LIMITS        = PROFILE["x_limits"]
Y_LIMITS        = PROFILE["y_limits"]
ISO_CODES       = PROFILE["iso_codes"]
DATA_SUBDIR     = PROFILE["data_subdir"]
OUTPUTS_SUBDIR  = PROFILE["outputs_subdir"]
HOTSPOT_REGIONS = PROFILE["hotspot_regions"]


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
