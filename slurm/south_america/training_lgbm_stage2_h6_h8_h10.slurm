#!/bin/bash
#SBATCH --job-name=sa_h8_h10
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h8_h10_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h8_h10_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Full SA training: H6+H8+H10
#   H8: inv_sqrt_npos_temporal group weights — combined size + temporal decay
#       weight = (1/sqrt(n_pos)) × exp(-0.2 × (max_year - year))
#       Addresses both gradient concentration sources: event size AND event age.
#   H10: ndcg_x_recall early stopping — product of ndcg_at_1pct × recall_at_5pct
#       Forces the model to jointly optimise ranking quality AND retrieval.
#
# Mini-sample result (job 3655220, run 3/4, 2026-06-17):
#   H6+H8+H10 proxy:  Lift=20.00×  Recall=24.8%
#   vs H6+H1b baseline: +7.1× Lift, -3pp Recall
#   Best Lift result on proxy to date; only modest Recall trade-off.
#
# Record result in feature_ablation_sa.json and ROADMAP.md.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1
export STAGE2_EARLY_STOP_METRIC=ndcg_x_recall
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos_temporal
export STAGE2_NORMALIZE_WITHIN_GROUPS=1
export STAGE2_GRADED_RELEVANCE=1

echo "Full SA H6+H8+H10  neg_ratio=$STAGE2_NEG_RATIO  eco_groups=$STAGE2_ECO_GROUPS  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE  normalize=$STAGE2_NORMALIZE_WITHIN_GROUPS  graded=$STAGE2_GRADED_RELEVANCE"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2

echo "Training complete. Record Lift@1% and Recall@5% in:"
echo "  outputs/south_america/results/feature_ablation_sa.json"
echo "  ROADMAP.md"
