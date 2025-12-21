#!/bin/bash
#SBATCH --job-name=colombia_crs
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/colombia_crs_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/colombia_crs_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/Tables"

#------------------------------------------------------------------------------
# Load modules (minimal set, no GIS modules)
#------------------------------------------------------------------------------
module purge
# Explicitly unload any GIS/GDAL/PROJ modules that might be loaded
module unload gdal 2>/dev/null || true
module unload proj 2>/dev/null || true
module unload gis 2>/dev/null || true

module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

#------------------------------------------------------------------------------
# Activate conda environment with conda-forge rasterio/gdal/proj
#------------------------------------------------------------------------------
# Set conda environment name (adjust if different)
CONDA_ENV_NAME="${CONDA_ENV_NAME:-master_thesis}"

# Initialize conda (if not already initialized)
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    # Try common conda initialization paths
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "ERROR: Could not find conda initialization script."
        echo "Please ensure conda is installed and initialized."
        exit 1
    fi
fi

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME" || {
    echo "ERROR: Failed to activate conda environment: $CONDA_ENV_NAME"
    echo "Available environments:"
    conda env list
    exit 1
}

# Set PROJ_LIB and GDAL_DATA to conda environment paths
# This ensures we use conda-forge PROJ/GDAL, not system modules
if [ -n "$CONDA_PREFIX" ]; then
    export PROJ_LIB="$CONDA_PREFIX/share/proj"
    export GDAL_DATA="$CONDA_PREFIX/share/gdal"
    
    echo "Environment variables set:"
    echo "  CONDA_PREFIX: $CONDA_PREFIX"
    echo "  PROJ_LIB: $PROJ_LIB"
    echo "  GDAL_DATA: $GDAL_DATA"
    
    # Verify paths exist
    if [ ! -d "$PROJ_LIB" ]; then
        echo "WARNING: PROJ_LIB directory not found: $PROJ_LIB"
        echo "  This may cause PROJ database errors."
    fi
    if [ ! -d "$GDAL_DATA" ]; then
        echo "WARNING: GDAL_DATA directory not found: $GDAL_DATA"
    fi
else
    echo "ERROR: CONDA_PREFIX not set. Conda environment may not be activated correctly."
    exit 1
fi

#------------------------------------------------------------------------------
# Run the colombia_crs script
#------------------------------------------------------------------------------
echo "Starting colombia_crs script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/colombia/preprocessing/colombia_crs" ] || {
  echo "ERROR: colombia_crs not found"
  echo "NOTE: colombia_crs script is in colombia/preprocessing/ directory"
  find scripts -maxdepth 4 -type f -name "colombia_crs"
  exit 1
}
# Run the script
python "scripts/colombia/preprocessing/colombia_crs"

echo "colombia_crs script finished."
