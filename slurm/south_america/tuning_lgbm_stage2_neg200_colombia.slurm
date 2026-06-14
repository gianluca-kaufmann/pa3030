#!/bin/bash
#SBATCH --job-name=col_tune_neg200
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/col_tune_neg200_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/col_tune_neg200_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# Colombia dev panel: neg_ratio=200 variant (Issue X)
# neg_ratio=100 baseline: params/lambdarank (existing)
export STAGE2_DATA_ROOT="$HOME/master_thesis/data/dev/south_america/ml"
export STAGE2_OUTPUT_DIR="$HOME/master_thesis/data/dev/south_america/ml/params/neg_ratio/neg200"
export STAGE2_NEG_RATIO=200
export STAGE2_TUNING_TRIALS=30
export STAGE2_ECO_GROUPS=1

mkdir -p "$STAGE2_OUTPUT_DIR"

echo "Stage 2 LambdaRank tuning — Colombia dev panel, neg_ratio=200 (Issue X)..."
srun python -u scripts/regions/south_america/4_tuning/model1_tuning_lgbm_stage2
echo "Colombia neg_ratio=200 tuning done."
