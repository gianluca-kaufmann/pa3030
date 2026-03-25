#!/bin/bash
#SBATCH --job-name=forward_cov_sa
#SBATCH --partition=normal.4h
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/forward_cov_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_cov_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Stage 0a: Coverage Baseline — South America
#
# Computes total land area and current protection coverage from backbone +
# WDPA rasters using latitude-corrected km² calculations.  Also computes
# BAU designation volume (2019→2024) and scenario thresholds for moderate
# and 30×30 targets.
#
# Output:
#   outputs/south_america/results/forward/forward_coverage_baseline.json
#
# Notes:
#   - Can run independently of other forward pipeline stages.
#   - Must complete before forward_results.slurm (which reads the JSON).
#   - Reads backbone.tif and WDPA_2024.tif from $SCRATCH/data/south_america/ready/
#
# Usage:
#   sbatch slurm/south_america/forward_coverage.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/south_america/results/forward"

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
# Activate virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"

#------------------------------------------------------------------------------
# Preflight
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/south_america/8_forward/1_forward_coverage_baseline.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

#------------------------------------------------------------------------------
# Stage 0a: Coverage baseline
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 0a: Computing South America coverage baseline..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT"

EXIT_CODE=$?
echo "forward_coverage (SA) finished (exit code: ${EXIT_CODE})"
exit $EXIT_CODE
