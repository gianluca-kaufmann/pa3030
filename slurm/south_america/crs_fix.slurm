#!/bin/bash
#SBATCH --job-name=crs_fix
#SBATCH --partition=normal.24h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/crs_fix_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/crs_fix_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

#------------------------------------------------------------------------------
# Activate the Python virtual environment (master-thesis)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Run CRS fix script (dry-run + apply)
#------------------------------------------------------------------------------
echo "Starting CRS fix script..."
echo "PWD = $(pwd)"
echo "SCRATCH = $SCRATCH"

[ -f "scripts/regions/south_america/2_preprocessing/crs_fix_engineering_mercator" ] || {
  echo "ERROR: crs_fix_engineering_mercator not found"
  exit 1
}

echo ""
echo "---- Dry-run scan ----"
python "scripts/regions/south_america/2_preprocessing/crs_fix_engineering_mercator"

echo ""
echo "---- Apply fixes ----"
python "scripts/regions/south_america/2_preprocessing/crs_fix_engineering_mercator" --apply

echo ""
echo "CRS fix script finished."

