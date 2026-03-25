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
#SBATCH --array=0-3

# ------------------------------------------------------------------------------
# Stage 0c: Pseudo-forecast Backtesting — South East Asia
#
# Validates the forward prediction methodology at 4 historical origins:
#   SLURM_ARRAY_TASK_ID=0 → T=2013 (clean 5-yr window)
#   SLURM_ARRAY_TASK_ID=1 → T=2015 (clean 5-yr window)
#   SLURM_ARRAY_TASK_ID=2 → T=2017 (partial: only 2 years observable)
#   SLURM_ARRAY_TASK_ID=3 → T=2019 (consistency check only)
#
# Each task:
#   1. Trains a historical deployment model on 2001–(T-1)
#   2. Scores year-T pixels (WDPA_prev==0, WDPA==0)
#   3. Reconstructs 5-year window actuals from WDPA column
#   4. Saves forward_backtest_T{T}.json
#
# After all tasks complete, aggregate:
#   python scripts/regions/se_asia/8_forward/5_forward_backtest.py --aggregate
#
# Usage (all 4 origins):
#   sbatch slurm/se_asia/forward_backtest.slurm
#
# Usage (single origin, e.g. T=2013):
#   sbatch --array=0 slurm/se_asia/forward_backtest.slurm
#
# Can be run in parallel with forward_deployment.slurm — no dependency.
# ------------------------------------------------------------------------------

ORIGIN_YEARS=(2013 2015 2017 2019)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ORIGIN_YEAR="${ORIGIN_YEARS[$TASK_ID]}"

echo "Backtest task ${TASK_ID}: origin year T=${ORIGIN_YEAR}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/se_asia/results/forward"

# Cap negatives to avoid OOM when training on full panels (mirrors production training_lgbm.slurm)
export MAX_NEG_TRAIN=100000000

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

SCRIPT="scripts/regions/se_asia/8_forward/5_forward_backtest.py"
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

# Aggregate only from the last task (T=2019) once all others should be done
# (Not reliable in a pure array — run the aggregate step manually after all tasks complete)
if [ "${TASK_ID}" -eq 3 ]; then
  echo ""
  echo "Task 3 (T=2019) done. To aggregate all results and produce the summary plot, run:"
  echo "  python scripts/regions/se_asia/8_forward/5_forward_backtest.py --aggregate"
fi

exit $EXIT_CODE
