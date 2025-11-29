import os
import socket

# ---- Roots ----

# Euler scratch root (where you rsynced gs://protected-areas)
EULER_DATA_ROOT = "/cluster/scratch/gikaufmann/thesis_data"

# Optional local root (for small samples, if you ever use them)
LOCAL_DATA_ROOT = "/Users/gianluca/Master's Thesis/code/data"

# GCS buckets
GCS_PROTECTED_AREAS = "gs://protected-areas"
GCS_PROTECTED_AREAS_DATA = f"{GCS_PROTECTED_AREAS}/data"
GCS_PROTECTED_AREAS_ML = f"{GCS_PROTECTED_AREAS_DATA}/ml"


def _running_on_euler() -> bool:
    hostname = socket.gethostname()
    return hostname.startswith("eu-") or "euler" in hostname


def get_data_root() -> str:
    """
    Base directory for local filesystem data on this machine.

    - On Euler: /cluster/scratch/gikaufmann/thesis_data
    - On Mac:   /Users/gianluca/Master's Thesis/code/data  (if used)

    You can override using RUN_ENV=local|euler.
    """
    run_env = os.environ.get("RUN_ENV", "").lower()
    if run_env == "euler":
        return EULER_DATA_ROOT
    if run_env == "local":
        return LOCAL_DATA_ROOT
    return EULER_DATA_ROOT if _running_on_euler() else LOCAL_DATA_ROOT


# ---- Convenience helpers for subfolders ----

def get_protected_areas_root() -> str:
    # /.../thesis_data/protected-areas
    return os.path.join(get_data_root(), "protected-areas")


def get_ready_root() -> str:
    # /.../thesis_data/protected-areas/data/ready
    return os.path.join(get_protected_areas_root(), "data", "ready")


def get_ml_dir() -> str:
    # /.../thesis_data/protected-areas/data/ml
    return os.path.join(get_protected_areas_root(), "data", "ml")


def get_ndvi_dir() -> str:
    return os.path.join(get_ready_root(), "NDVI")


def get_wdpa_dir() -> str:
    return os.path.join(get_ready_root(), "WDPA")


def get_landcover_dir() -> str:
    return os.path.join(get_ready_root(), "landcover")

# add more like get_viirs_dir(), get_wildfire_dir(), etc. as needed