#!/bin/bash
#SBATCH --job-name=model1_RF
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/model1_RF_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_RF_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/south_america/tables"

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
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

#------------------------------------------------------------------------------
# Run the model1_RF script
#------------------------------------------------------------------------------
echo "Starting model1_RF script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/5_training/model1_RF" ] || {
  echo "ERROR: model1_RF not found"
  echo "NOTE: model1_RF script is in scripts/regions/south_america/5_training/ directory"
  find scripts -maxdepth 6 -type f -name "model1_RF"
  exit 1
}

# OOM-safe configuration for large datasets (500M rows)
export MAX_NEG_TRAIN=50000000      # Cap negative samples at 50M for training (keeps all positives)

# Run the script
python "scripts/regions/south_america/5_training/model1_RF"

echo "model1_RF script finished."
