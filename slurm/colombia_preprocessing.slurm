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
# Note: module purge may remove conda from PATH, so we bootstrap conda AFTER
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
# Bootstrap conda/micromamba (AFTER module purge to ensure it's available)
#------------------------------------------------------------------------------
# Set conda environment name (adjust if different)
CONDA_ENV_NAME="${CONDA_ENV_NAME:-master_thesis}"

echo "Bootstrapping conda/micromamba..."
echo "  HOME: $HOME"
echo "  Searching for conda/micromamba installations..."

# Try to bootstrap conda or micromamba from common installation locations
# Priority: micromamba (faster), then miniconda3, then anaconda3
CONDA_INITIALIZED=false

# Method 1: Try micromamba (if installed)
MICROMAMBA_PATHS=(
    "$HOME/.local/bin/micromamba"
    "$HOME/micromamba/bin/micromamba"
    "$HOME/.micromamba/bin/micromamba"
)
for MICROMAMBA_BIN in "${MICROMAMBA_PATHS[@]}"; do
    if [ -f "$MICROMAMBA_BIN" ]; then
        echo "  Found micromamba at: $MICROMAMBA_BIN"
        eval "$($MICROMAMBA_BIN shell hook --shell bash)"
        CONDA_INITIALIZED=true
        break
    fi
done

# Method 2: Try miniconda3 (most common user installation)
if [ "$CONDA_INITIALIZED" = false ]; then
    MINICONDA_PATHS=(
        "$HOME/miniconda3"
        "$HOME/miniconda"
        "$HOME/.miniconda3"
    )
    for conda_root in "${MINICONDA_PATHS[@]}"; do
        conda_init="$conda_root/etc/profile.d/conda.sh"
        if [ -f "$conda_init" ]; then
            echo "  Found miniconda3 at: $conda_init"
            source "$conda_init"
            CONDA_INITIALIZED=true
            break
        fi
    done
fi

# Method 3: Try anaconda3
if [ "$CONDA_INITIALIZED" = false ]; then
    ANACONDA_PATHS=(
        "$HOME/anaconda3"
        "$HOME/anaconda"
        "$HOME/.anaconda3"
    )
    for conda_root in "${ANACONDA_PATHS[@]}"; do
        conda_init="$conda_root/etc/profile.d/conda.sh"
        if [ -f "$conda_init" ]; then
            echo "  Found anaconda3 at: $conda_init"
            source "$conda_init"
            CONDA_INITIALIZED=true
            break
        fi
    done
fi

# Method 4: Try other common locations
if [ "$CONDA_INITIALIZED" = false ]; then
    OTHER_PATHS=(
        "$HOME/.conda"
        "/opt/conda"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
        "/opt/miniconda3"
        "/opt/anaconda3"
    )
    for conda_root in "${OTHER_PATHS[@]}"; do
        conda_init="$conda_root/etc/profile.d/conda.sh"
        if [ -f "$conda_init" ]; then
            echo "  Found conda at: $conda_init"
            source "$conda_init"
            CONDA_INITIALIZED=true
            break
        fi
    done
fi

# Method 5: Try to find conda via find command (slower, but comprehensive)
if [ "$CONDA_INITIALIZED" = false ]; then
    echo "  Searching for conda.sh in $HOME (this may take a moment)..."
    FOUND_CONDA=$(find "$HOME" -maxdepth 4 -name "conda.sh" -path "*/etc/profile.d/conda.sh" 2>/dev/null | head -1)
    if [ -n "$FOUND_CONDA" ] && [ -f "$FOUND_CONDA" ]; then
        echo "  Found conda at: $FOUND_CONDA"
        source "$FOUND_CONDA"
        CONDA_INITIALIZED=true
    fi
fi

# Method 6: Check if conda is available as a module (after module purge, try loading)
if [ "$CONDA_INITIALIZED" = false ]; then
    echo "  Checking for conda/anaconda modules..."
    if module avail conda 2>&1 | grep -q conda; then
        echo "  Found conda module, loading..."
        module load conda 2>/dev/null && {
            if [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
                source "$CONDA_PREFIX/etc/profile.d/conda.sh"
                CONDA_INITIALIZED=true
            fi
        } || true
    fi
    if [ "$CONDA_INITIALIZED" = false ] && module avail anaconda3 2>&1 | grep -q anaconda3; then
        echo "  Found anaconda3 module, loading..."
        module load anaconda3 2>/dev/null && {
            if [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
                source "$CONDA_PREFIX/etc/profile.d/conda.sh"
                CONDA_INITIALIZED=true
            fi
        } || true
    fi
fi

# Verify conda/micromamba is available
USE_MICROMAMBA=false
if [ "$CONDA_INITIALIZED" = true ]; then
    # Check if we're using micromamba (micromamba shell hook may provide 'conda' alias)
    if command -v micromamba >/dev/null 2>&1 && ! command -v conda >/dev/null 2>&1; then
        USE_MICROMAMBA=true
        echo "Using micromamba for environment management"
    elif command -v conda >/dev/null 2>&1; then
        echo "Using conda for environment management"
    else
        echo "WARNING: Conda/micromamba initialized but commands not found in PATH"
    fi
else
    echo "ERROR: Could not bootstrap conda or micromamba."
    echo ""
    echo "Diagnostic information:"
    echo "  HOME: $HOME"
    echo "  PATH: $PATH"
    echo ""
    echo "Checking what exists in $HOME:"
    echo "  miniconda3: $([ -d "$HOME/miniconda3" ] && echo "EXISTS" || echo "not found")"
    echo "  anaconda3: $([ -d "$HOME/anaconda3" ] && echo "EXISTS" || echo "not found")"
    echo "  .local/bin: $([ -d "$HOME/.local/bin" ] && echo "EXISTS" || echo "not found")"
    echo "  micromamba: $([ -f "$HOME/.local/bin/micromamba" ] && echo "EXISTS" || echo "not found")"
    echo ""
    echo "Checking for any conda-related directories:"
    ls -d "$HOME"/miniconda* "$HOME"/anaconda* "$HOME"/.conda* "$HOME"/micromamba* 2>/dev/null | head -5 || echo "  None found"
    echo ""
    echo "To install miniconda (recommended):"
    echo "  cd ~"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3"
    echo "  source $HOME/miniconda3/etc/profile.d/conda.sh"
    echo "  conda create -n master_thesis -c conda-forge python=3.11 rasterio gdal proj"
    echo ""
    echo "To install micromamba (alternative, faster):"
    echo "  mkdir -p $HOME/.local/bin"
    echo "  cd /tmp"
    echo "  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba"
    echo "  mv bin/micromamba $HOME/.local/bin/"
    echo "  chmod +x $HOME/.local/bin/micromamba"
    echo "  eval \"\$($HOME/.local/bin/micromamba shell hook --shell bash)\""
    echo "  micromamba create -n master_thesis -c conda-forge python=3.11 rasterio gdal proj"
    echo ""
    exit 1
fi

echo "✓ Conda/micromamba initialized successfully"

# Activate conda environment (works for both conda and micromamba)
echo "Activating conda environment: $CONDA_ENV_NAME"
if [ "$USE_MICROMAMBA" = true ]; then
    micromamba activate "$CONDA_ENV_NAME" || {
        echo "ERROR: Failed to activate micromamba environment: $CONDA_ENV_NAME"
        echo "Available environments:"
        micromamba env list
        exit 1
    }
else
    conda activate "$CONDA_ENV_NAME" || {
        echo "ERROR: Failed to activate conda environment: $CONDA_ENV_NAME"
        echo "Available environments:"
        conda env list
        exit 1
    }
fi

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
