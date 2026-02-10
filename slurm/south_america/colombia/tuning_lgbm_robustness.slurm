#!/bin/bash
#SBATCH --job-name=tuning_lgbm_robustness
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/modelC_tuning_lgbm_robustness_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/modelC_tuning_lgbm_robustness_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/south_america/colombia/tables"

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
# Run the modelC_tuning_lgbm script with robustness split
#------------------------------------------------------------------------------
echo "Starting modelC_tuning_lgbm script (robustness split)..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/colombia/4_ml/training/tuning/modelC_tuning_lgbm" ] || {
  echo "ERROR: modelC_tuning_lgbm not found"
  echo "NOTE: modelC_tuning_lgbm script is in tuning/ directory"
  find scripts -maxdepth 4 -type f -name "modelC_tuning_lgbm"
  exit 1
}
# Run the script with robustness split
python "scripts/regions/south_america/colombia/4_ml/training/tuning/modelC_tuning_lgbm" --split-version robustness

echo "modelC_tuning_lgbm script (robustness split) finished."
