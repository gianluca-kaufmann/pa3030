#!/bin/bash
#SBATCH --job-name=lobo_SEA
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/lobo_SEA_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/lobo_SEA_%A_%a.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis
#SBATCH --array=0-8

# ------------------------------------------------------------------------------
# Leave-One-Biome-Out (LOBO) spatial CV — South East Asia
#
# Submits one SLURM array job per biome fold.  Each task trains a LightGBM
# model on all biomes except one and evaluates on the held-out biome.
#
# Usage (submit all folds at once via array):
#   sbatch slurm/se_asia/spatial_CV_1.slurm
#
# Usage (single fold, e.g. biome index 3):
#   sbatch --export=BIOME_IDX=3 --array=3 slurm/se_asia/spatial_CV_1.slurm
#
# The SLURM array index (SLURM_ARRAY_TASK_ID) is used as the biome index
# unless BIOME_IDX is set explicitly.
#
# To find valid biome indices:
#   python scripts/regions/se_asia/6_evaluation/spatial_CV_1 --list-biomes
#
# To run a representative biome subset (no manual edits needed):
#   python scripts/regions/shared/evaluation/select_lobo_biomes.py \
#       --region se_asia --n-biomes 5
#   sbatch --array=$(cat slurm/se_asia/lobo_biome_array.txt) \
#       slurm/se_asia/spatial_CV_1.slurm
# ------------------------------------------------------------------------------

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/se_asia/results/spatial_cv"

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
#   - If BIOME_IDX is set externally, use it.
#   - Otherwise fall back to the SLURM array task ID.
#------------------------------------------------------------------------------
FOLD_IDX="${BIOME_IDX:-${SLURM_ARRAY_TASK_ID}}"
echo "LOBO fold index: ${FOLD_IDX}"

#------------------------------------------------------------------------------
# Preflight checks
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/se_asia/6_evaluation/spatial_CV_1"
[ -f "$SCRIPT" ] || {
  echo "ERROR: $SCRIPT not found"
  find scripts -maxdepth 6 -type f -name "spatial_CV_1" 2>/dev/null
  exit 1
}

#------------------------------------------------------------------------------
# Run the LOBO fold
#------------------------------------------------------------------------------
echo "Starting LOBO fold ${FOLD_IDX} (South East Asia)..."
python -u "$SCRIPT" --biome-idx "${FOLD_IDX}" --model-type lgbm

echo "spatial_CV_1 (lgbm) fold ${FOLD_IDX} (South East Asia) finished."
