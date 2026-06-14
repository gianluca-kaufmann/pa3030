#!/bin/bash
#SBATCH --job-name=col_train_xgb
#SBATCH --partition=normal.24h
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/col_train_xgb_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/col_train_xgb_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# Colombia dev panel: redirect data + best_params input/output (Issue BB)
export STAGE2_DATA_ROOT="$HOME/master_thesis/data/dev/south_america/ml"
export STAGE2_OUTPUT_DIR="$HOME/master_thesis/data/dev/south_america/ml/params/xgboost"
export STAGE2_NEG_RATIO=100

echo "Stage 2 XGBoost rank:ndcg training — Colombia dev panel (Issue BB)..."
srun python -u scripts/regions/south_america/5_training/model1_XGB_stage2
echo "Colombia XGBoost training done."
