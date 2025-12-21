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
    # Method 1: Check if conda is already in PATH
    if command -v conda >/dev/null 2>&1; then
        echo "Found conda in PATH, initializing..."
        # Try to initialize using conda's own method
        eval "$(conda shell.bash hook 2>/dev/null)" || {
            echo "Warning: conda shell hook failed, trying alternative methods..."
        }
    fi
    
    # Method 2: Try loading conda as a module (common on HPC systems)
    if [ -z "$CONDA_DEFAULT_ENV" ] && ! command -v conda >/dev/null 2>&1; then
        echo "Trying to load conda module..."
        module load conda 2>/dev/null || module load anaconda3 2>/dev/null || true
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook 2>/dev/null)" || true
        fi
    fi
    
    # Method 3: Try common conda initialization paths
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        CONDA_INIT_PATHS=(
            "$HOME/miniconda3/etc/profile.d/conda.sh"
            "$HOME/anaconda3/etc/profile.d/conda.sh"
            "$HOME/.conda/etc/profile.d/conda.sh"
            "/opt/conda/etc/profile.d/conda.sh"
            "/usr/local/miniconda3/etc/profile.d/conda.sh"
            "/usr/local/anaconda3/etc/profile.d/conda.sh"
        )
        
        for conda_init in "${CONDA_INIT_PATHS[@]}"; do
            if [ -f "$conda_init" ]; then
                echo "Found conda at: $conda_init"
                source "$conda_init"
                break
            fi
        done
    fi
    
    # Method 4: Try to find conda via which/whereis and derive init script
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        CONDA_BIN=$(which conda 2>/dev/null || command -v conda 2>/dev/null)
        if [ -n "$CONDA_BIN" ]; then
            # Derive conda root from binary path
            CONDA_ROOT=$(dirname "$(dirname "$CONDA_BIN")")
            CONDA_INIT="$CONDA_ROOT/etc/profile.d/conda.sh"
            if [ -f "$CONDA_INIT" ]; then
                echo "Found conda via PATH, initializing from: $CONDA_INIT"
                source "$CONDA_INIT"
            fi
        fi
    fi
    
    # Final check
    if [ -z "$CONDA_DEFAULT_ENV" ] && ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: Could not find or initialize conda."
        echo ""
        echo "Diagnostic information:"
        echo "  PATH: $PATH"
        echo "  HOME: $HOME"
        echo "  Checking for conda in common locations..."
        ls -d "$HOME"/miniconda* "$HOME"/anaconda* "$HOME"/.conda 2>/dev/null | head -5 || echo "  No conda directories found in HOME"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check if conda is installed: which conda || find $HOME -name conda 2>/dev/null | head -3"
        echo "2. Try loading conda module: module avail 2>&1 | grep -i conda"
        echo "3. If conda is installed elsewhere, find it and source the init script:"
        echo "   source <conda_root>/etc/profile.d/conda.sh"
        echo "4. Install miniconda if needed:"
        echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3"
        echo "   source $HOME/miniconda3/etc/profile.d/conda.sh"
        echo "5. Create conda environment with:"
        echo "   conda create -n master_thesis -c conda-forge python=3.11 rasterio gdal proj"
        echo ""
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
