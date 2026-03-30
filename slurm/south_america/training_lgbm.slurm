#!/bin/bash
#SBATCH --job-name=model1_LGBM
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/model1_LGBM_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_LGBM_%j.err
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
# Run the model1_LGBM script
#------------------------------------------------------------------------------
echo "Starting model1_LGBM script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/5_training/model1_LGBM" ] || {
  echo "ERROR: model1_LGBM not found"
  echo "NOTE: model1_LGBM script is in scripts/regions/south_america/5_training/ directory"
  find scripts -maxdepth 6 -type f -name "model1_LGBM"
  exit 1
}
# Full train + test (omit --eval-only). Use --eval-only only after a prior run produced model1_lgbm_metrics_*.json.
srun --cpu-bind=cores --distribution=block:block python -u "scripts/regions/south_america/5_training/model1_LGBM"

echo "model1_LGBM script finished."
