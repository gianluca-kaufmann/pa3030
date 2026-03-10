#!/bin/bash
#SBATCH --job-name=tuning_lgbm_sa
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/model1_tuning_lgbm_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_tuning_lgbm_%j.err
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
# SA-specific positive cap: the 5-year lookahead makes positives majority-class
# without a cap (~75% pos on 2.5M-row budget).  5000/year targets SPW ~30x,
# matching the production rare-event regime and making PR-AUC meaningful.
export MAX_POS_PER_YEAR_PAPER="${MAX_POS_PER_YEAR_PAPER:-5000}"
export MAX_POS_PER_YEAR_FAST="${MAX_POS_PER_YEAR_FAST:-2000}"
export N_ITER_LGBM_PAPER="${N_ITER_LGBM_PAPER:-70}"
export N_ITER_LGBM_FAST="${N_ITER_LGBM_FAST:-50}"
export TUNING_RANDOM_STATE="${TUNING_RANDOM_STATE:-42}"

echo "Starting model1_tuning_lgbm script..."
echo "Mode=$TUNING_MODE Optimizer=$TUNING_OPTIMIZER CV=$CV_STRATEGY Split=$TUNING_SPLIT_VERSION Folds=$ROLLING_FOLDS"

ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/4_tuning/model1_tuning_lgbm" ] || {
  echo "ERROR: model1_tuning_lgbm not found"
  exit 1
}

python "scripts/regions/south_america/4_tuning/model1_tuning_lgbm"

echo "model1_tuning_lgbm script finished."
