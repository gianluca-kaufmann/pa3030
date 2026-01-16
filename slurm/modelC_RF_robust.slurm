#!/bin/bash
#SBATCH --job-name=modelC_RF_robust
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/modelC_RF_robust_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/modelC_RF_robust_%j.err
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

#------------------------------------------------------------------------------
# Activate the Python virtual environment (uav-genAI-opti)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Run the modelC_RF script with robust split variant
#------------------------------------------------------------------------------
echo "Starting modelC_RF script (robust split)..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/colombia/ML/training/modelC_RF" ] || {
  echo "ERROR: modelC_RF not found"
  echo "NOTE: modelC_RF script is in colombia/ML/training/ directory"
  find scripts -maxdepth 4 -type f -name "modelC_RF"
  exit 1
}

# Enable fast train mode
export FAST_TRAIN=1

# Set verbose output for scikit-learn
export SK_VERBOSE=1

# Run the script with robust split variant
python "scripts/colombia/ML/training/modelC_RF" --split-variant robust

echo "modelC_RF script finished (robust split)."
