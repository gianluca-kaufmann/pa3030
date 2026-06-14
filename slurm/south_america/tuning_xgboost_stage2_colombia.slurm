#!/bin/bash
#SBATCH --job-name=col_tune_xgb
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/col_tune_xgb_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/col_tune_xgb_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
# Install xgboost if not already present (Issue BB dependency)
python -c "import xgboost" 2>/dev/null || pip install --quiet xgboost
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# Colombia dev panel: redirect data + output (Issue BB)
export STAGE2_DATA_ROOT="$HOME/master_thesis/data/dev/south_america/ml"
export STAGE2_OUTPUT_DIR="$HOME/master_thesis/data/dev/south_america/ml/params/xgboost"
export STAGE2_NEG_RATIO=100
export STAGE2_TUNING_TRIALS=30

mkdir -p "$STAGE2_OUTPUT_DIR"

echo "Stage 2 XGBoost rank:ndcg tuning — Colombia dev panel (30 trials, Issue BB)..."
srun python -u scripts/regions/south_america/4_tuning/model1_tuning_xgboost_stage2
echo "Colombia XGBoost tuning done."
