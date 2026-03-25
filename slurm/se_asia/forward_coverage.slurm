#!/bin/bash
#SBATCH --job-name=forward_cov_sea
#SBATCH --partition=normal.4h
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/forward_cov_sea_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_cov_sea_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Stage 0a: Coverage Baseline — South East Asia
#
# Computes total land area and current protection coverage from backbone +
# WDPA rasters using latitude-corrected km² calculations.  Also computes
# BAU designation volume (2019→2024) and scenario thresholds for moderate
# and 30×30 targets.
#
# Output:
#   outputs/se_asia/results/forward/forward_coverage_baseline.json
#
# Notes:
#   - Can run independently of other forward pipeline stages.
#   - Must complete before forward_results.slurm (which reads the JSON).
#   - Reads backbone.tif and WDPA_2024.tif from $SCRATCH/data/se_asia/ready/
#
# Usage:
#   sbatch slurm/se_asia/forward_coverage.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/se_asia/results/forward"

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
export WANDB_MODE="${WANDB_MODE:-online}"

#------------------------------------------------------------------------------
# Preflight
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/se_asia/8_forward/1_forward_coverage_baseline.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

#------------------------------------------------------------------------------
# Stage 0a: Coverage baseline
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 0a: Computing South East Asia coverage baseline..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT"

EXIT_CODE=$?
echo "forward_coverage (SE Asia) finished (exit code: ${EXIT_CODE})"
exit $EXIT_CODE
