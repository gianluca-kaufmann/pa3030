#!/bin/bash
#SBATCH --job-name=forward_bt
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/forward_bt_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_bt_%A_%a.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis
#SBATCH --array=0-3

# ------------------------------------------------------------------------------
# Stage 0c: Pseudo-forecast Backtesting — South East Asia (LGBM)
#
# Validates the forward prediction methodology at four historical origins:
#   SLURM_ARRAY_TASK_ID=0 → T=2013 (train 2001–2008, eval 2014–2018)
#   SLURM_ARRAY_TASK_ID=1 → T=2015 (train 2001–2010, eval 2016–2020)
#   SLURM_ARRAY_TASK_ID=2 → T=2017 (train 2001–2012, eval 2018–2022)
#   SLURM_ARRAY_TASK_ID=3 → T=2019 (train 2001–2014, eval 2020–2024)
#
# Training cutoff = T − 5 (mirrors real deployment: last training year
# always has a complete 5-year lookahead window, no label leakage).
#
# Each task:
#   1. Trains a historical deployment model on 2001–(T-5)
#   2. Scores year-T pixels (WDPA_prev==0, WDPA==0)
#   3. Reconstructs 5-year window actuals from WDPA column
#   4. Saves forward_backtest_T{T}.json
#
# After all tasks complete, aggregate:
#   sbatch slurm/se_asia/forward_backtest_aggregate.slurm
#
# Usage:
#   sbatch slurm/se_asia/forward_backtest.slurm          # all 4 origins
#   sbatch --array=3 slurm/se_asia/forward_backtest.slurm  # T=2019 only
#
# Can be run in parallel with forward_deployment.slurm — no dependency.
# ------------------------------------------------------------------------------

ORIGIN_YEARS=(2013 2015 2017 2019)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ORIGIN_YEAR="${ORIGIN_YEARS[$TASK_ID]}"

echo "Backtest task ${TASK_ID}: origin year T=${ORIGIN_YEAR}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/se_asia/results/forward"

# Backtest-specific caps: much lower than production — ranking metrics (ROC-AUC,
# Precision@K, Lift@K) converge well below 100M negatives; scale_pos_weight handles
# class imbalance. Reduces training time ~5x vs production cap.
export MAX_NEG_TRAIN=20000000
# Tree count kept at production value (2555) for methodological consistency:
# the backtest simulates running the exact same pipeline at time T.
# The 5x speedup from MAX_NEG_TRAIN alone is sufficient to meet the time limit.

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
# W&B: WandbRunLogger only calls wandb.init if PA3030_WANDB=1 (utils.py).
export PA3030_WANDB="${PA3030_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
mkdir -p "$WANDB_DIR"

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

echo ""
echo "Task ${TASK_ID} (T=${ORIGIN_YEAR}) done. After all 4 tasks finish, aggregate with:"
echo "  sbatch slurm/se_asia/forward_backtest_aggregate.slurm"

exit $EXIT_CODE
