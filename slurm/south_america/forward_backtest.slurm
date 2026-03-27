#!/bin/bash
#SBATCH --job-name=forward_bt
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/forward_bt_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_bt_%A_%a.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis
#SBATCH --array=0

# ------------------------------------------------------------------------------
# Stage 0c: Pseudo-forecast Backtesting — South America
#
# Validates the forward prediction methodology at 1 historical origin:
#   SLURM_ARRAY_TASK_ID=0 → T=2019 (clean 5-yr window: 2020–2024)
#
# T=2019 mirrors the actual deployment setup most closely: trains on 2001–2018,
# then evaluates against the observed 2020–2024 WDPA designations — the same
# period the real forward prediction targets.
#
# Each task:
#   1. Trains a historical deployment model on 2001–(T-1)
#   2. Scores year-T pixels (WDPA_prev==0, WDPA==0)
#   3. Reconstructs 5-year window actuals from WDPA column
#   4. Saves forward_backtest_T{T}.json
#
# After all tasks complete, aggregate:
#   python scripts/regions/south_america/8_forward/5_forward_backtest.py --aggregate
#
# Usage:
#   sbatch slurm/south_america/forward_backtest.slurm
#
# Can be run in parallel with forward_deployment.slurm — no dependency.
# ------------------------------------------------------------------------------

ORIGIN_YEARS=(2019)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ORIGIN_YEAR="${ORIGIN_YEARS[$TASK_ID]}"

echo "Backtest task ${TASK_ID}: origin year T=${ORIGIN_YEAR}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/south_america/results/forward"

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

#------------------------------------------------------------------------------
# Preflight
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/south_america/8_forward/5_forward_backtest.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

#------------------------------------------------------------------------------
# Run backtest for this origin year
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Running backtest for T=${ORIGIN_YEAR}..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT" --origin-year "${ORIGIN_YEAR}"

EXIT_CODE=$?
echo "Backtest T=${ORIGIN_YEAR} finished (exit code: ${EXIT_CODE})"

if [ "${TASK_ID}" -eq 0 ]; then
  echo ""
  echo "Task 0 (T=2019) done. To aggregate results and produce the summary plot, run:"
  echo "  python scripts/regions/south_america/8_forward/5_forward_backtest.py --aggregate"
fi

exit $EXIT_CODE
