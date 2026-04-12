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
        "hotspot_max_regions": 2,
        # Indices into cluster_hotspot_boxes.json to select for display.
        # C1 (index 0): COL Forest — highest-ranked, most BAU pixels.
        # C5 (index 4): CHL Bare/Other — different biome type, Andean foothills context.
        "hotspot_region_indices": [0, 4],
        "wdpa_2024_filename": "WDPA_SA_1km_2024.tif",
        "wdpa_2019_filename": "WDPA_SA_1km_2019.tif",
        # Forward-only display: calibrated scores are massed at low p; raise lower
        # percentile + mild γ so typical pixels stay dark and high-risk tail reads yellow.
        "probability_map_percentile_min": 32,
        "probability_map_percentile_max": 98,
        "probability_map_transformation": "sqrt",
        "probability_map_display_gamma": 1.2,
        # Existing protected areas (2024) tone used across all forward maps.
        # Kept light so it doesn't compete with scenario/prediction overlays.
        "forward_pa_hole_color": "#D0D0D0",
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
        "hotspot_max_regions": 2,
        "hotspot_region_indices": None,  # use first max_regions by rank
        "wdpa_2024_filename": "WDPA_USA_1km_2024.tif",
        "wdpa_2019_filename": "WDPA_USA_1km_2019.tif",
        # USA calibrated forward scores collapse most mass into a narrow log band;
        # a high lower percentile + strong γ avoids a flat red “midfield” and
        # reserves bright colors for the rare high-likelihood tail.
        "probability_map_percentile_min": 47,
        "probability_map_percentile_max": 99,
        "probability_map_transformation": "log",
        "probability_map_display_gamma": 2.0,
        "forward_pa_hole_color": "#D0D0D0",
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
        "hotspot_max_regions": 2,
        "hotspot_region_indices": None,  # use first max_regions by rank
        "wdpa_2024_filename": "WDPA_SEA_1km_2024.tif",
        "wdpa_2019_filename": "WDPA_SEA_1km_2019.tif",
        # Same intent as south_america but slightly milder (see comments there).
        "probability_map_percentile_min": 32,
        "probability_map_percentile_max": 98,
        "probability_map_transformation": "sqrt",
        "probability_map_display_gamma": 1.12,
        "forward_pa_hole_color": "#D0D0D0",
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
HOTSPOT_MAX_REGIONS = PROFILE["hotspot_max_regions"]
HOTSPOT_REGION_INDICES = PROFILE["hotspot_region_indices"]
WDPA_2024_FILENAME = PROFILE["wdpa_2024_filename"]
WDPA_2019_FILENAME = PROFILE["wdpa_2019_filename"]
PROBABILITY_MAP_PERCENTILE_MIN = PROFILE["probability_map_percentile_min"]
PROBABILITY_MAP_PERCENTILE_MAX = PROFILE["probability_map_percentile_max"]
PROBABILITY_MAP_TRANSFORMATION = PROFILE["probability_map_transformation"]
PROBABILITY_MAP_DISPLAY_GAMMA = PROFILE["probability_map_display_gamma"]
FORWARD_PA_HOLE_COLOR = PROFILE["forward_pa_hole_color"]


# ── Path layout (scratch-first on Euler) ─────────────────────────────────────
# Same idea as ModelEvalConfig (evaluation/config.py): when SCRATCH is set,
# large artifacts use $SCRATCH/data/... and $SCRATCH/outputs/... instead of home.
# Optional overrides: PA3030_FORWARD_OUTPUT_DIR, PA3030_ML_MODELS_DIR (absolute paths).


def get_repo_root() -> Path:
    """Repository root (PROJECT_ROOT / REPO_ROOT, or search upward for README.md)."""
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


def _repo_forward_dir(repo_root: Path, outputs_subdir: str) -> Path:
    return repo_root / f"outputs/{outputs_subdir}/results/forward"


def resolve_forward_dir(repo_root: Path, outputs_subdir: str) -> Path:
    """Forward outputs root: features parquet, baseline JSON, lgbm/ | rf/ subdirs."""
    override = os.environ.get("PA3030_FORWARD_OUTPUT_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    scratch = os.environ.get("SCRATCH", "").strip()
    if scratch:
        return (Path(scratch) / f"outputs/{outputs_subdir}/results/forward").resolve()
    return _repo_forward_dir(repo_root, outputs_subdir).resolve()


def forward_dir_search_paths(repo_root: Path, outputs_subdir: str) -> list[Path]:
    """Read order: primary (scratch policy), then legacy repo path."""
    primary = resolve_forward_dir(repo_root, outputs_subdir)
    legacy = _repo_forward_dir(repo_root, outputs_subdir).resolve()
    out: list[Path] = [primary]
    if legacy != primary:
        out.append(legacy)
    return out


def _repo_ml_models_dir(repo_root: Path, data_subdir: str) -> Path:
    return repo_root / f"data/{data_subdir}/ml/models"


def resolve_ml_models_dir(repo_root: Path, data_subdir: str) -> Path:
    """Deployment *.pkl directory (data/.../ml/models)."""
    override = os.environ.get("PA3030_ML_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    scratch = os.environ.get("SCRATCH", "").strip()
    if scratch:
        return (Path(scratch) / f"data/{data_subdir}/ml/models").resolve()
    return _repo_ml_models_dir(repo_root, data_subdir).resolve()


def ml_models_dir_search_paths(repo_root: Path, data_subdir: str) -> list[Path]:
    """Read order for *_deployment_*.pkl: scratch policy, then repo."""
    primary = resolve_ml_models_dir(repo_root, data_subdir)
    legacy = _repo_ml_models_dir(repo_root, data_subdir).resolve()
    out: list[Path] = [primary]
    if legacy != primary:
        out.append(legacy)
    return out
