#!/bin/bash
#SBATCH --job-name=group_ab_SA
#SBATCH --partition=normal.4h
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/cluster/scratch/%u/logs/group_ab_SA_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/group_ab_SA_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Group A/B leakage diagnostic — South America
#
# Splits test positives by designation-year cohort to quantify soft label
# leakage.  LEGACY: reads pre-W1 scored parquets + test_win5.parquet; no GPU required.
# This diagnostic is intentionally bound to the legacy 5-year window — it analyses
# label overlap that the W1 hazard refactor eliminates by construction.
# Peak memory: ~1.2 GB.  Runs for LGBM and RF sequentially (~5 min each).
#
# Usage:
#   sbatch slurm/south_america/group_ab_diagnostic.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/south_america/results/ml_models/main"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/south_america/6_evaluation/group_ab_diagnostic"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

echo "Starting Group A/B diagnostic (South America, both models)..."
srun --cpu-bind=cores python -u "$SCRIPT" --model both

echo "Group A/B diagnostic (South America) finished."
