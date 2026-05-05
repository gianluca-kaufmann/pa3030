#!/bin/bash
#SBATCH --job-name=cv_transfer
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/cv_transfer_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/cv_transfer_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Cross-continental transfer evaluation (SA <-> USA) — Layer 3 spatial CV
#
# Loads the production models trained in step 5 (no retraining) and scores them
# on the other continent's test set.  Runs 4 combinations sequentially:
#   SA-LGBM → USA,  USA-LGBM → SA,  SA-RF → USA,  USA-RF → SA
#
# Missing model files (e.g. RF not yet trained) are skipped gracefully.
#
# Usage:
#   sbatch slurm/south_america/spatial_CV_3.slurm
#
# Prerequisites:
#   1. SA step-5 training done:  data/south_america/ml/models/model1_lgbm_*.pkl
#                                data/south_america/ml/models/model1_rf_*.joblib
#   2. USA step-5 training done: data/usa/ml/models/model2_lgbm_*.pkl
#                                data/usa/ml/models/model2_rf_*.joblib
#   3. Test parquets:            data/{region}/ml/main/test.parquet
#
# Output:
#   outputs/south_america/results/spatial_cv/transfer/
#       transfer_results_<timestamp>.json
#       transfer_results_<timestamp>.txt  (log)
# ------------------------------------------------------------------------------

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/south_america/results/spatial_cv/transfer"

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
# Preflight checks
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/south_america/6_evaluation/spatial_CV_3"
[ -f "$SCRIPT" ] || {
  echo "ERROR: $SCRIPT not found"
  find scripts -maxdepth 6 -type f -name "spatial_CV_3" 2>/dev/null
  exit 1
}

#------------------------------------------------------------------------------
# Run both transfer directions (SA→USA and USA→SA)
#------------------------------------------------------------------------------
echo "Starting cross-continental transfer evaluation..."
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT"

echo "Cross-continental transfer evaluation finished."
