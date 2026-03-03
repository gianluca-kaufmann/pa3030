#!/bin/bash
#SBATCH --job-name=tuning_rf_sa
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/model1_tuning_rf_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_tuning_rf_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/south_america/tables"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

# Paper-mode defaults (overridable by pre-set env vars)
export TUNING_MODE="${TUNING_MODE:-paper}"
export TUNING_OPTIMIZER="${TUNING_OPTIMIZER:-optuna}"
export CV_STRATEGY="${CV_STRATEGY:-rolling}"
export TUNING_SPLIT_VERSION="${TUNING_SPLIT_VERSION:-main}"
export ROLLING_FOLDS="${ROLLING_FOLDS:-5}"
export MAX_NEG_PER_YEAR_PAPER="${MAX_NEG_PER_YEAR_PAPER:-150000}"
export MAX_NEG_PER_YEAR_FAST="${MAX_NEG_PER_YEAR_FAST:-50000}"
# SA-specific positive cap (see tuning_lgbm.slurm for rationale).
# Critical for RF: RF uses a smaller row budget (500k) so the majority-positive
# distortion is even more severe without a cap.
export MAX_POS_PER_YEAR_PAPER="${MAX_POS_PER_YEAR_PAPER:-5000}"
export MAX_POS_PER_YEAR_FAST="${MAX_POS_PER_YEAR_FAST:-2000}"
# 15 trials fits in 24h (~82 min/trial); override to 60 for full tuning if you have more time
export N_ITER_RF_PAPER="${N_ITER_RF_PAPER:-15}"
export N_ITER_RF_FAST="${N_ITER_RF_FAST:-25}"
export TUNING_RANDOM_STATE="${TUNING_RANDOM_STATE:-42}"

echo "Starting model1_tuning_rf script..."
echo "Mode=$TUNING_MODE Optimizer=$TUNING_OPTIMIZER CV=$CV_STRATEGY Split=$TUNING_SPLIT_VERSION Folds=$ROLLING_FOLDS"

ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/4_ml/2_tuning/model1_tuning_rf" ] || {
  echo "ERROR: model1_tuning_rf not found"
  exit 1
}

python "scripts/regions/south_america/4_ml/2_tuning/model1_tuning_rf"

echo "model1_tuning_rf script finished."
