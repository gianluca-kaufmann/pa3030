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
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml/ready"
mkdir -p "$SCRATCH/wandb/storage_optimisation"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load gdal/3.7.3
module load python_cuda/3.11.6
module load eth_proxy

#------------------------------------------------------------------------------
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Configure Weights & Biases
#------------------------------------------------------------------------------
export WANDB_DIR="$SCRATCH/wandb/storage_optimisation"
export WANDB_CACHE_DIR="$SCRATCH/wandb/storage_optimisation/.cache"

#------------------------------------------------------------------------------
# Run the script
#------------------------------------------------------------------------------
echo "Starting storage optimization script..."
echo "Input:  $SCRATCH/data"
echo "Output: $SCRATCH/data/ml/ready"
echo "WANDB:  $WANDB_DIR"
echo ""
echo "Note: This will optimize all datasets including:"
echo "  - Regular datasets (elevation, NDVI, landcover, etc.)"
echo "  - Embeddings (embeddings/embeddings_2018, embeddings_2019, etc.)"

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/preprocessing/storage_optimise" ] || {
  echo "ERROR: storage_optimise not found under scripts/preprocessing"
  find scripts -maxdepth 3 -type f -name "storage_optimise"
  exit 1
}

# Check dependencies
echo "Checking dependencies..."
which gdal_translate || { echo "ERROR: gdal_translate not found"; exit 1; }
python -c "import rasterio" || { echo "ERROR: rasterio not found. Install with: pip install rasterio"; exit 1; }
python -c "import numpy" || { echo "ERROR: numpy not found. Install with: pip install numpy"; exit 1; }
echo "All dependencies available"

# List input data
echo ""
echo "Input data structure:"
ls -la "$SCRATCH/data/" | head -20
echo ""

# Run the script (remove --dry-run for actual execution)
# Options:
#   --dry-run    : Preview changes without writing files
#   --dataset X  : Process only dataset X (e.g., WorldClim, GSN)
python "scripts/preprocessing/storage_optimise"

echo ""
echo "Storage optimization finished."
echo "Output data structure:"
ls -la "$SCRATCH/data/ml/ready/" | head -20

