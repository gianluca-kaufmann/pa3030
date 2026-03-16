#!/bin/bash
#SBATCH --job-name=lobo_USA_rf
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/lobo_USA_rf_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/lobo_USA_rf_%A_%a.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis
#SBATCH --array=0-11

# ------------------------------------------------------------------------------
# Leave-One-Biome-Out (LOBO) spatial CV — USA — Random Forest
#
# Mirrors spatial_CV_1.slurm but trains a Random Forest in each fold.
# Outputs go to outputs/usa/results/spatial_cv/rf/
#
# Usage (submit all folds at once via array):
#   sbatch slurm/usa/spatial_CV_1_rf.slurm
#
# Usage (single fold, e.g. biome index 3):
#   sbatch --export=BIOME_IDX=3 --array=3 slurm/usa/spatial_CV_1_rf.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/usa/results/spatial_cv/rf"

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
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

#------------------------------------------------------------------------------
# Determine biome index
#------------------------------------------------------------------------------
FOLD_IDX="${BIOME_IDX:-${SLURM_ARRAY_TASK_ID}}"
echo "LOBO RF fold index: ${FOLD_IDX}"

#------------------------------------------------------------------------------
# Preflight checks
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/usa/6_evaluation/spatial_CV_1"
[ -f "$SCRIPT" ] || {
  echo "ERROR: $SCRIPT not found"
  find scripts -maxdepth 6 -type f -name "spatial_CV_1" 2>/dev/null
  exit 1
}

GSN_RASTER="data/usa/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif"
[ -f "$GSN_RASTER" ] || {
  echo "ERROR: USA GSN raster not found at $GSN_RASTER"
  exit 1
}

#------------------------------------------------------------------------------
# Run the RF LOBO fold
#------------------------------------------------------------------------------
echo "Starting LOBO RF fold ${FOLD_IDX} (USA)..."
export MAX_NEG_TRAIN=50000000      # Cap negative samples at 50M — matches production training_rf.slurm
export BATCH_SIZE=100000           # Reduced batch size for OOM safety — matches production training_rf.slurm
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT" --biome-idx "${FOLD_IDX}" --model-type rf

echo "spatial_CV_1 (rf) fold ${FOLD_IDX} (USA) finished."
