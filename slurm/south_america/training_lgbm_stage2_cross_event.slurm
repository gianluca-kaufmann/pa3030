#!/bin/bash
#SBATCH --job-name=ce_sa
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/ce_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ce_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Cross-event Stage 2 — full South America (CE.2).
# Prerequisite: Colombia pilot Recall@5% >= 0.60 (CE.1).
# Decision gate: SA macro Recall@5% >= 0.65 → proceed to CE.3 (add AGB + REDD).
#
# Settings: H6 (Recall@5% early stop) + H1b (inv_sqrt_npos) + H5 (no rank-norm).
# Data: $SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet (42 GB total).

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GRADED_RELEVANCE=1

echo "Cross-event Stage 2 — full South America (CE.2)"
echo "  normalize=$STAGE2_NORMALIZE_WITHIN_GROUPS  weights=$STAGE2_GROUP_WEIGHT_MODE  es=$STAGE2_EARLY_STOP_METRIC"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_cross_event.py

echo "Done. Check decision gate: Recall@5% >= 0.65 → proceed to CE.3 (AGB + REDD features)."
echo "Record result in ROADMAP.md experiment table."
