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
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

# Load GDAL for command-line tools (gdal_translate, gdalinfo)
# Check available versions with: module avail gdal
module load gdal 2>/dev/null || {
    echo "Warning: GDAL module not found. Trying alternative..."
    # Try alternative module names if available
    module load gdal/3.8.0 2>/dev/null || {
        echo "Warning: Could not load GDAL module."
        echo "GDAL command-line tools may not be available."
        echo "Check available modules with: module avail gdal"
    }
}

#------------------------------------------------------------------------------
# Activate the Python virtual environment (uav-genAI-opti)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

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
