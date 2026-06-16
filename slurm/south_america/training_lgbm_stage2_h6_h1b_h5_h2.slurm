#!/bin/bash
#SBATCH --job-name=sa_h5_h2
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h5_h2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h5_h2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Full SA training: H6+H1b+H5+H2 (no rank norm + binary labels + Recall@5% stop + inv_sqrt_npos).
#
# Mini-sample result (job 3647627, 2026-06-17):
#   H6+H1b+H5+H2 proxy: [FILL FROM JOB OUTPUT]
#
# Submit only if proxy Recall@5% > best mini result. Record result in ROADMAP.md.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GRADED_RELEVANCE=0

echo "Full SA H6+H1b+H5+H2  neg_ratio=$STAGE2_NEG_RATIO  eco_groups=$STAGE2_ECO_GROUPS  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE  normalize=$STAGE2_NORMALIZE_WITHIN_GROUPS  graded=$STAGE2_GRADED_RELEVANCE"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2

echo "Training complete. Record Lift@1% and Recall@5% in:"
echo "  outputs/south_america/results/feature_ablation_sa.json"
echo "  ROADMAP.md"
