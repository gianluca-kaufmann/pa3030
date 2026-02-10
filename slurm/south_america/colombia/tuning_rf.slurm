#!/bin/bash
#SBATCH --job-name=tuning_rf
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/modelC_tuning_rf_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/modelC_tuning_rf_%j.err
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
# Run the modelC_tuning_rf script
#------------------------------------------------------------------------------
echo "Starting modelC_tuning_rf script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/colombia/4_ml/training/tuning/modelC_tuning_rf" ] || {
  echo "ERROR: modelC_tuning_rf not found"
  echo "NOTE: modelC_tuning_rf script is in tuning/ directory"
  find scripts -maxdepth 4 -type f -name "modelC_tuning_rf"
  exit 1
}
# Run the script
python "scripts/regions/south_america/colombia/4_ml/training/tuning/modelC_tuning_rf"

echo "modelC_tuning_rf script finished."
