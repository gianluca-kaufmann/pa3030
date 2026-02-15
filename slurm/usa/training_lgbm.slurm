#!/bin/bash
#SBATCH --job-name=model2_LGBM
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/model2_LGBM_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model2_LGBM_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/usa/tables"

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
# Run the model2_LGBM script
#------------------------------------------------------------------------------
echo "Starting model2_LGBM script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/usa/4_ml/2_training/model2_LGBM" ] || {
  echo "ERROR: model2_LGBM not found"
  echo "NOTE: model2_LGBM script is in scripts/regions/usa/4_ml/2_training/ directory"
  find scripts -maxdepth 6 -type f -name "model2_LGBM"
  exit 1
}
# Run the script
srun --cpu-bind=cores --distribution=block:block python -u "scripts/regions/usa/4_ml/2_training/model2_LGBM"

echo "model2_LGBM script finished."
