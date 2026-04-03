#!/bin/bash
#SBATCH --job-name=forward_bt_usa
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/forward_bt_usa_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_bt_usa_%A_%a.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis
#SBATCH --array=0-3

# ------------------------------------------------------------------------------
# Stage 0c (LGBM): Pseudo-forecast Backtesting — USA
#
# Validates the LGBM forward prediction methodology at four historical origins:
#   SLURM_ARRAY_TASK_ID=0 → T=2013 (train 2001–2008, eval 2014–2018)
#   SLURM_ARRAY_TASK_ID=1 → T=2015 (train 2001–2010, eval 2016–2020)
#   SLURM_ARRAY_TASK_ID=2 → T=2017 (train 2001–2012, eval 2018–2022)
#   SLURM_ARRAY_TASK_ID=3 → T=2019 (train 2001–2014, eval 2020–2024)
#
# Training cutoff = T − 5 (mirrors real deployment: last training year
# always has a complete 5-year lookahead window, no label leakage).
#
# Each task:
#   1. Trains a historical LGBM deployment model on 2001–(T-5)
#   2. Scores year-T pixels (WDPA_prev==0, WDPA==0)
#   3. Reconstructs 5-year window actuals from WDPA column
#   4. Saves forward/lgbm/forward_backtest_T{T}.json
#
# After all tasks complete, aggregate:
#   sbatch slurm/usa/forward_backtest_aggregate.slurm
#
# Usage:
#   sbatch slurm/usa/forward_backtest.slurm          # all 4 origins
#   sbatch --array=3 slurm/usa/forward_backtest.slurm  # T=2019 only
#
# Can be run in parallel with forward_deployment.slurm (usa) — no dependency.
# ------------------------------------------------------------------------------

ORIGIN_YEARS=(2013 2015 2017 2019)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ORIGIN_YEAR="${ORIGIN_YEARS[$TASK_ID]}"

echo "USA LGBM backtest task ${TASK_ID}: origin year T=${ORIGIN_YEAR}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/usa/results/forward/lgbm"

# Tree count kept at production value (2555) for methodological consistency:
# the backtest simulates running the exact same pipeline at time T.

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
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
mkdir -p "$WANDB_DIR"

#------------------------------------------------------------------------------
# Preflight
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/usa/8_forward/5_forward_backtest.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

#------------------------------------------------------------------------------
# Run LGBM backtest for this origin year
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Running USA LGBM backtest for T=${ORIGIN_YEAR}..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT" --origin-year "${ORIGIN_YEAR}"

EXIT_CODE=$?
echo "USA LGBM backtest T=${ORIGIN_YEAR} finished (exit code: ${EXIT_CODE})"

echo ""
echo "Task ${TASK_ID} (T=${ORIGIN_YEAR}) done. After all 4 tasks finish, aggregate with:"
echo "  sbatch slurm/usa/forward_backtest_aggregate.slurm"

exit $EXIT_CODE
