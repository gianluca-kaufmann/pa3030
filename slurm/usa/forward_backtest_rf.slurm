#!/bin/bash
#SBATCH --job-name=forward_bt_usa_rf
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/forward_bt_usa_rf_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_bt_usa_rf_%A_%a.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis
#SBATCH --array=0-2

# ------------------------------------------------------------------------------
# Stage 0c (RF): Pseudo-forecast Backtesting — USA (RF)
#
# Validates the RF forward prediction methodology at 3 historical origins:
#   SLURM_ARRAY_TASK_ID=0 → T=2013 (clean 5-yr window)
#   SLURM_ARRAY_TASK_ID=1 → T=2015 (4/5 years observable)
#   SLURM_ARRAY_TASK_ID=2 → T=2017 (2/5 years observable, temporal stability check)
#   T=2019 omitted: T+5=2024 > LAST_LABEL_YEAR=2019, zero evaluable labels
#
# Each task:
#   1. Trains a historical RF deployment model on 2001–(T-1)
#      with class_weight="balanced_subsample", n_estimators=500
#   2. Scores year-T pixels (WDPA_prev==0, WDPA==0)
#   3. Reconstructs 5-year window actuals from WDPA column
#   4. Saves forward/rf/forward_backtest_T{T}.json
#
# After all tasks complete, aggregate:
#   python scripts/regions/usa/8_forward/5_forward_backtest.py \
#     --model-type rf --aggregate
#
# Usage (all 4 origins):
#   sbatch slurm/usa/forward_backtest_rf.slurm
#
# Usage (single origin, e.g. T=2013):
#   sbatch --array=0 slurm/usa/forward_backtest_rf.slurm
#
# Can be run in parallel with forward_deployment_rf.slurm (usa) — no dependency.
# ------------------------------------------------------------------------------

ORIGIN_YEARS=(2013 2015 2017 2019)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ORIGIN_YEAR="${ORIGIN_YEARS[$TASK_ID]}"

echo "USA RF backtest task ${TASK_ID}: origin year T=${ORIGIN_YEAR}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/usa/results/forward/rf"

# Tree count kept at production value (500) for methodological consistency:
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
# RF training cap: reservoir-sample negatives to keep X matrix within RAM budget.
# USA has ~270 M total rows; 30 M neg keeps X at ~10 GB (73 features × float32).
export MAX_NEG_TRAIN="${MAX_NEG_TRAIN:-30000000}"

#------------------------------------------------------------------------------
# Preflight
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/usa/8_forward/5_forward_backtest.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

#------------------------------------------------------------------------------
# Run RF backtest for this origin year
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Running USA RF backtest for T=${ORIGIN_YEAR}..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT" --model-type rf --origin-year "${ORIGIN_YEAR}"

EXIT_CODE=$?
echo "USA RF backtest T=${ORIGIN_YEAR} finished (exit code: ${EXIT_CODE})"

if [ "${TASK_ID}" -eq 2 ]; then
  echo ""
  echo "Task 2 (T=2017) done. To aggregate all RF results and produce the summary plot, run:"
  echo "  python scripts/regions/usa/8_forward/5_forward_backtest.py --model-type rf --aggregate"
fi

exit $EXIT_CODE
