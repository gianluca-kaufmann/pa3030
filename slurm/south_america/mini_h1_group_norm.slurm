#!/bin/bash
#SBATCH --job-name=mini_h1_group_norm
#SBATCH --partition=normal.4h
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/mini_h1_group_norm_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/mini_h1_group_norm_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# H1 mini-sample experiment: group-normalised sample weights.
#
# Runs three back-to-back training runs on the mini-sample
# (data/south_america/mini_splits/main/) using baseline hyperparameters:
#
#   1. baseline_mini    — default year_weights (current behaviour)
#   2. H1_inv_npos      — weight = 1/n_pos per group  (strong equalisation)
#   3. H1_inv_sqrt_npos — weight = 1/sqrt(n_pos)      (gentler equalisation)
#
# Results are appended to outputs/south_america/results/feature_ablation_sa.json.
# Compare proxy Lift@1% and Recall@5% across the three runs to decide whether
# to promote H1 to a full SA Euler job.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1

PARAMS="outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json"
TRAIN_CMD="scripts/regions/south_america/5_training/model1_phase1_local_train --params $PARAMS"

echo "======================================================"
echo "Run 1/3 — baseline (year_weights)"
echo "======================================================"
export STAGE2_GROUP_WEIGHT_MODE=year_weights
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag baseline_mini \
    --notes "baseline year_weights on mini-sample"

echo ""
echo "======================================================"
echo "Run 2/3 — H1: inv_npos (1/n_pos per group)"
echo "======================================================"
export STAGE2_GROUP_WEIGHT_MODE=inv_npos
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H1_inv_npos \
    --notes "group-norm weights 1/n_pos — equalises gradient across all expansion events"

echo ""
echo "======================================================"
echo "Run 3/3 — H1b: inv_sqrt_npos (1/sqrt(n_pos) per group)"
echo "======================================================"
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H1_inv_sqrt_npos \
    --notes "group-norm weights 1/sqrt(n_pos) — gentler equalisation"

echo ""
echo "All 3 runs complete. Check ablation log:"
echo "  outputs/south_america/results/feature_ablation_sa.json"
