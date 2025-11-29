import os
import socket
from enum import Enum
from pathlib import Path

# ---- Storage Backend Configuration ----

class StorageBackend(Enum):
    """Available storage backends for data access."""
    GCS = "gcs"          # Google Cloud Storage (gs://...)
    EULER = "euler"      # ETH Euler cluster filesystem
    LOCAL = "local"      # Local machine filesystem


# ---- Storage Roots ----

# Euler scratch root (where you rsynced gs://protected-areas)
EULER_DATA_ROOT = "/cluster/scratch/gikaufmann/thesis_data"

# Local root (for small samples or local development)
LOCAL_DATA_ROOT = "/Users/gianluca/Desktop/Master's Thesis/code/data"

# GCS buckets (primary data storage)
GCS_PROTECTED_AREAS = "gs://protected-areas"
GCS_PROTECTED_AREAS_DATA = f"{GCS_PROTECTED_AREAS}/data"
GCS_PROTECTED_AREAS_ML = f"{GCS_PROTECTED_AREAS_DATA}/ml"


# ---- Backend Detection & Selection ----

def _running_on_euler() -> bool:
    """Detect if running on Euler cluster."""
    hostname = socket.gethostname()
    return hostname.startswith("eu-") or "euler" in hostname.lower()


def get_storage_backend() -> StorageBackend:
    """
    Determine which storage backend to use.
    
    Priority:
    1. STORAGE_BACKEND environment variable (gcs, euler, or local)
    2. Legacy RUN_ENV environment variable (for backward compatibility)
    3. Auto-detect: Euler cluster → EULER, otherwise → GCS (since data migrated to cloud)
    
    Returns:
        StorageBackend enum value
    """
    # Check explicit STORAGE_BACKEND env var
    storage_env = os.environ.get("STORAGE_BACKEND", "").lower()
    if storage_env in ["gcs", "cloud", "google"]:
        return StorageBackend.GCS
    elif storage_env in ["euler", "cluster"]:
        return StorageBackend.EULER
    elif storage_env in ["local", "mac"]:
        return StorageBackend.LOCAL
    
    # Check legacy RUN_ENV for backward compatibility
    run_env = os.environ.get("RUN_ENV", "").lower()
    if run_env == "euler":
        return StorageBackend.EULER
    elif run_env == "local":
        return StorageBackend.LOCAL
    
    # Auto-detect: on Euler use local filesystem, otherwise use GCS
    if _running_on_euler():
        return StorageBackend.EULER
    else:
        return StorageBackend.GCS


def get_data_root() -> str:
    """
    Get the base data directory path based on storage backend.
    
    Returns:
        - GCS:   gs://protected-areas
        - Euler: /cluster/scratch/gikaufmann/thesis_data/protected-areas
        - Local: /Users/gianluca/Desktop/Master's Thesis/code/data/protected-areas
    
    You can override using STORAGE_BACKEND=gcs|euler|local environment variable.
    """
    backend = get_storage_backend()
    
    if backend == StorageBackend.GCS:
        return GCS_PROTECTED_AREAS
    elif backend == StorageBackend.EULER:
        return os.path.join(EULER_DATA_ROOT, "protected-areas")
    else:  # LOCAL
        return os.path.join(LOCAL_DATA_ROOT, "protected-areas")


# ---- Path Construction Helpers ----

def _join_path(base: str, *parts: str) -> str:
    """
    Join path components, handling both filesystem and GCS paths.
    
    Args:
        base: Base path (could be gs://... or filesystem path)
        parts: Path components to join
        
    Returns:
        Properly joined path string
    """
    if base.startswith("gs://"):
        # GCS path: use forward slashes
        result = base
        for part in parts:
            result = result.rstrip("/") + "/" + part.lstrip("/")
        return result
    else:
        # Filesystem path: use os.path.join
        return os.path.join(base, *parts)


# ---- Convenience Helpers for Subfolders ----

def get_protected_areas_root() -> str:
    """Get path to protected-areas root directory."""
    return get_data_root()


def get_ready_root() -> str:
    """Get path to data/ready directory (processed datasets ready for use)."""
    return _join_path(get_data_root(), "data", "ready")


def get_ml_dir() -> str:
    """Get path to data/ml directory (ML training data)."""
    return _join_path(get_data_root(), "data", "ml")


def get_ndvi_dir() -> str:
    """Get path to NDVI data directory."""
    return _join_path(get_ready_root(), "NDVI")


def get_wdpa_dir() -> str:
    """Get path to WDPA data directory."""
    return _join_path(get_ready_root(), "WDPA")


def get_landcover_dir() -> str:
    """Get path to landcover data directory."""
    return _join_path(get_ready_root(), "landcover")


def get_viirs_dir() -> str:
    """Get path to VIIRS nighttime lights data directory."""
    return _join_path(get_ready_root(), "VIIRS")


def get_wildfire_dir() -> str:
    """Get path to wildfire data directory."""
    return _join_path(get_ready_root(), "wildfire")


def get_worldclim_dir() -> str:
    """Get path to WorldClim data directory."""
    return _join_path(get_ready_root(), "WorldClim")


def get_deforestation_dir() -> str:
    """Get path to deforestation data directory."""
    return _join_path(get_ready_root(), "deforestation")


def get_elevation_dir() -> str:
    """Get path to elevation data directory."""
    return _join_path(get_ready_root(), "elevation")


def get_dw_dir() -> str:
    """Get path to DW (dynamic world) data directory."""
    return _join_path(get_ready_root(), "DW")


def get_gpw_dir() -> str:
    """Get path to GPW (population) data directory."""
    return _join_path(get_ready_root(), "GPW")


# ---- Utility Functions ----

def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path."""
    return path.startswith("gs://")


def ensure_gcs_filesystem():
    """
    Ensure gcsfs is available for GCS access.
    
    Raises:
        ImportError: If gcsfs is not installed
    """
    try:
        import gcsfs  # noqa: F401
    except ImportError:
        raise ImportError(
            "gcsfs is required for GCS access. Install with: pip install gcsfs"
        )


def get_filesystem_handler():
    """
    Get appropriate filesystem handler for current storage backend.
    
    Returns:
        Filesystem handler object (gcsfs.GCSFileSystem or None for local)
    """
    backend = get_storage_backend()
    
    if backend == StorageBackend.GCS:
        ensure_gcs_filesystem()
        import gcsfs
        return gcsfs.GCSFileSystem()
    else:
        return None  # Use standard filesystem operations


def setup_duckdb_gcs_auth(con=None):
    """
    Configure DuckDB to authenticate with Google Cloud Storage.
    
    This sets up DuckDB's httpfs extension with proper authentication.
    
    Args:
        con: DuckDB connection object. If None, creates a new connection.
        
    Returns:
        DuckDB connection with GCS authentication configured
        
    Usage:
        import duckdb
        con = duckdb.connect()
        con = config.setup_duckdb_gcs_auth(con)
        # Now can read from gs:// paths
    """
    import duckdb
    import os
    
    if con is None:
        con = duckdb.connect()
    
    try:
        # Install and load the httpfs extension (required for GCS)
        print("  Installing DuckDB httpfs extension...")
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        
        # Try method 1: Use GOOGLE_APPLICATION_CREDENTIALS if set
        creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_file and os.path.exists(creds_file):
            print(f"  Using credentials from: {creds_file}")
            try:
                con.execute("DROP SECRET IF EXISTS gcs_secret;")
                con.execute(f"""
                    CREATE SECRET gcs_secret (
                        TYPE GCS,
                        KEY_ID '{creds_file}'
                    );
                """)
                print("✓ DuckDB GCS authentication configured successfully")
                return con
            except Exception as e:
                print(f"  Could not use credentials file: {e}")
        
        # Try method 2: Use credential chain (works with gcloud auth application-default)
        try:
            print("  Attempting to use GCS credential chain...")
            con.execute("DROP SECRET IF EXISTS gcs_secret;")
            con.execute("""
                CREATE SECRET gcs_secret (
                    TYPE GCS,
                    PROVIDER CREDENTIAL_CHAIN
                );
            """)
            print("✓ DuckDB GCS authentication configured (credential chain)")
            return con
        except Exception as e:
            print(f"  Credential chain failed: {e}")
        
        # If all methods fail, provide instructions
        print("\n" + "="*70)
        print("⚠ Could not configure DuckDB GCS authentication automatically")
        print("="*70)
        print("\nPlease authenticate using ONE of these methods:")
        print("\n1. Application Default Credentials (Recommended):")
        print("   gcloud auth application-default login")
        print("\n2. Service Account Key:")
        print("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        print("\n3. Use local files instead:")
        print("   export STORAGE_BACKEND=local")
        print("\nAfter authenticating, run the script again.")
        print("="*70)
            
    except Exception as e:
        print(f"⚠ Error setting up DuckDB GCS: {e}")
        print("\nDuckDB GCS support requires:")
        print("  - Network access")
        print("  - Google Cloud authentication")
        print("  - httpfs extension")
    
    return con


# ---- Debug / Info Functions ----

def print_config_info():
    """Print current configuration for debugging."""
    backend = get_storage_backend()
    print("=" * 70)
    print("DATA CONFIGURATION")
    print("=" * 70)
    print(f"Storage Backend:     {backend.value}")
    print(f"Data Root:           {get_data_root()}")
    print(f"ML Directory:        {get_ml_dir()}")
    print(f"Ready Directory:     {get_ready_root()}")
    print(f"Running on Euler:    {_running_on_euler()}")
    print("=" * 70)
    
    if backend == StorageBackend.GCS:
        print("\nNote: Using GCS storage. Ensure you have:")
        print("  1. gcsfs installed (pip install gcsfs)")
        print("  2. Google Cloud credentials configured")
        print("=" * 70)


if __name__ == "__main__":
    # When run directly, print configuration info
    print_config_info()