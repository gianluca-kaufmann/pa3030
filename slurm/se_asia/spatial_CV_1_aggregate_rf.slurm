#!/bin/bash
#SBATCH --job-name=cv1_agg_SEA_rf
#SBATCH --partition=normal.4h
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/cv1_agg_SEA_rf_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/cv1_agg_SEA_rf_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Layer 1 aggregate — South East Asia — Random Forest
#
# Reads all fold_*.json files produced by spatial_CV_1_rf.slurm (SA) and
# assembles a summary CSV with per-biome metrics plus mean ± SD across folds.
# Run this AFTER all spatial_CV_1_rf array tasks have completed.
#
# Usage:
#   sbatch slurm/se_asia/spatial_CV_1_aggregate_rf.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/se_asia/results/spatial_cv/rf"

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

SCRIPT="scripts/regions/se_asia/6_evaluation/spatial_CV_1_aggregate"
[ -f "$SCRIPT" ] || {
  echo "ERROR: $SCRIPT not found"
  find scripts -maxdepth 6 -type f -name "spatial_CV_1_aggregate" 2>/dev/null
  exit 1
}

#------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------
echo "Starting spatial_CV_1_aggregate (South East Asia, rf)..."
python -u "$SCRIPT" --model-type rf

echo "spatial_CV_1_aggregate (South East Asia, rf) finished."
