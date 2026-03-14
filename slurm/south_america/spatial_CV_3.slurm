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
# Trains on South America, evaluates on USA (SA→USA), then trains on USA and
# evaluates on South America (USA→SA).  Both directions run sequentially in a
# single job.  No biome index is needed — the script handles everything internally.
#
# Usage:
#   sbatch slurm/south_america/spatial_cv_transfer.slurm
#
# Prerequisites:
#   1. South America model trained: outputs/south_america/results/ populated
#   2. USA model trained:           outputs/usa/results/ populated
#   3. Both regions' parquet data accessible under data/south_america/ and data/usa/
#
# Output:
#   outputs/south_america/results/spatial_cv/transfer_results_<timestamp>.json
#   outputs/south_america/results/spatial_cv/transfer_<timestamp>.txt  (log)
# ------------------------------------------------------------------------------

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/south_america/results/spatial_cv"
mkdir -p "$SCRATCH/outputs/usa/results/spatial_cv"

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
