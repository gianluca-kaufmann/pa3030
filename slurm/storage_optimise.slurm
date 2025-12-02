#!/bin/bash
#SBATCH --job-name=storage-optimise
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/cluster/scratch/%u/logs/storage_optimise_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/storage_optimise_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data/ready"
mkdir -p "$SCRATCH/data/ml/ready"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load gdal/3.8.4

#------------------------------------------------------------------------------
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Run the script
#------------------------------------------------------------------------------
echo "Starting storage optimization script..."
echo "Input:  $SCRATCH/data/ready"
echo "Output: $SCRATCH/data/ml/ready"

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/preprocessing/storage_optimise" ] || {
  echo "ERROR: storage_optimise not found under scripts/preprocessing"
  find scripts -maxdepth 3 -type f -name "storage_optimise"
  exit 1
}

# Check GDAL availability
which gdal_translate || { echo "ERROR: gdal_translate not found"; exit 1; }
which gdal_calc.py || { echo "ERROR: gdal_calc.py not found"; exit 1; }

# List input data
echo ""
echo "Input data structure:"
ls -la "$SCRATCH/data/ready/"
echo ""

# Run the script (remove --dry-run for actual execution)
# Options:
#   --dry-run    : Preview changes without writing files
#   --dataset X  : Process only dataset X (e.g., WorldClim, GSN)
python "scripts/preprocessing/storage_optimise"

echo ""
echo "Storage optimization finished."
echo "Output data structure:"
ls -la "$SCRATCH/data/ml/ready/"

