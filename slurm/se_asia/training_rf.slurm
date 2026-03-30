#!/bin/bash
#SBATCH --job-name=model3_RF
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/model3_RF_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model3_RF_%j.err
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
# Run the model3_RF script
#------------------------------------------------------------------------------
echo "Starting model3_RF script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/se_asia/5_training/model3_RF" ] || {
  echo "ERROR: model3_RF not found"
  echo "NOTE: model3_RF script is in scripts/regions/se_asia/5_training/ directory"
  find scripts -maxdepth 6 -type f -name "model3_RF"
  exit 1
}

# OOM-safe configuration for large datasets (500M rows)
export MAX_NEG_TRAIN=50000000      # Cap negative samples at 50M for training (keeps all positives)

# Full train + test (omit --eval-only). Use --eval-only only after a prior run produced model3_rf_win5_metrics_*.json.
python "scripts/regions/se_asia/5_training/model3_RF"

echo "model3_RF script finished."
