#!/bin/bash
#SBATCH --job-name=forward_bt_aggr_sea_rf
#SBATCH --partition=normal.4h
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/forward_bt_aggr_sea_rf_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_bt_aggr_sea_rf_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Stage 0c aggregate (RF): Forward backtest aggregation — South East Asia
#
# Aggregates the per-origin RF backtest outputs written by:
#   slurm/se_asia/forward_backtest_rf.slurm
#
# Usage:
#   sbatch slurm/se_asia/forward_backtest_aggregate_rf.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/se_asia/results/forward/rf"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-online}"

SCRIPT="scripts/regions/se_asia/8_forward/5_forward_backtest.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

echo "Aggregating South East Asia RF backtest outputs..."
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT" --model-type rf --aggregate

EXIT_CODE=$?
echo "forward_backtest aggregate (SE Asia RF) finished (exit code: ${EXIT_CODE})"
exit $EXIT_CODE
