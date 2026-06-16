#!/bin/bash
#SBATCH --job-name=sa_h6_h1b
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h6_h1b_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h6_h1b_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Full SA training: H6 (Recall@5% early stopping) + H1b (inv_sqrt_npos weights).
#
# Mini-sample result (2026-06-16, job 3622838):
#   baseline (NDCG@1% stop, year_weights): Recall@5%=13.3%, Lift@1%=20.32×
#   H6_H1b   (Recall@5% stop, inv_sqrt_npos): Recall@5%=27.9%, Lift@1%=12.86×
#
# Decision rule: if full SA Recall@5% > 0.20 → this is the new best. Proceed to
# retuning (Phase 2). Record result in feature_ablation_sa.json and ROADMAP.md.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos

echo "Full SA H6+H1b  neg_ratio=$STAGE2_NEG_RATIO  eco_groups=$STAGE2_ECO_GROUPS  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2

echo "Training complete. Record Lift@1% and Recall@5% in:"
echo "  outputs/south_america/results/feature_ablation_sa.json"
echo "  ROADMAP.md"
