#!/bin/bash
#SBATCH --job-name=calibrate_LGBM
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/calibrate_LGBM_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/calibrate_LGBM_%j.err
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
# Activate the Python virtual environment (uav-genAI-opti)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Run the calibrate_C script with LGBM model type (main split)
#------------------------------------------------------------------------------
echo "Starting calibrate_C script (LGBM mode, main split)..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/colombia/4_ml/evaluation/calibrate_C" ] || {
  echo "ERROR: calibrate_C not found"
  echo "NOTE: calibrate_C script is in colombia/ML/evaluation/ directory"
  find scripts -maxdepth 4 -type f -name "calibrate_C"
  exit 1
}
# Run the script (main split, LGBM, temporal_cv source; auto selects best of Platt/Isotonic by test Brier)
python "scripts/regions/south_america/colombia/4_ml/evaluation/calibrate_C" --model_type lgbm --split-version main --calib-source temporal_cv --method auto

echo "calibrate_C script (LGBM mode, main split) finished."
