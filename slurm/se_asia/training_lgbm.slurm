#!/bin/bash
#SBATCH --job-name=model3_LGBM
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/model3_LGBM_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model3_LGBM_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/se_asia/tables"

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
# Run the model3_LGBM script
#------------------------------------------------------------------------------
echo "Starting model3_LGBM script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/se_asia/5_training/model3_LGBM" ] || {
  echo "ERROR: model3_LGBM not found"
  echo "NOTE: model3_LGBM script is in scripts/regions/se_asia/5_training/ directory"
  find scripts -maxdepth 6 -type f -name "model3_LGBM"
  exit 1
}
# Run the script
srun --cpu-bind=cores --distribution=block:block python -u "scripts/regions/se_asia/5_training/model3_LGBM" --eval-only

echo "model3_LGBM script finished."
