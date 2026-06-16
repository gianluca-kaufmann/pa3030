#!/bin/bash
#SBATCH --job-name=mini_h6_combined
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/mini_h6_combined_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/mini_h6_combined_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# H6 + H1b + H3 mini-sample experiment batch.
#
# Three experiments, each building on the previous, evaluated on
# data/south_america/mini_splits/main/ using baseline hyperparameters.
#
# Root cause being addressed: LambdaRank gradient is 96% concentrated in ~10
# large Brazilian events from 2001-2009. The model never learns from small,
# targeted post-2010 events that dominate the test set (2017-2024).
#
#   Run 1: H6_only     — Recall@5% early stopping, year_weights, W9a on
#           Isolates H6 effect. Is Recall@5% a more stable stopping criterion?
#
#   Run 2: H6_H1b      — Recall@5% early stopping, inv_sqrt_npos weights, W9a on
#           Core hypothesis: H1b destabilised with NDCG@1% (best_iter=8);
#           Recall@5% stopping gives it room to train properly.
#
#   Run 3: H6_H1b_H3   — Recall@5% early stopping, inv_sqrt_npos, W9a off
#           Does removing eco sub-groups help when gradients are equalised?
#
# Decision rule: if any run shows Recall@5% > 0.20 on proxy → promote to
# full SA Euler job (training_lgbm_stage2_phase1.slurm with matching env vars).

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
PARAMS="outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json"
TRAIN_CMD="scripts/regions/south_america/5_training/model1_phase1_local_train --params $PARAMS"

echo "======================================================"
echo "Run 1/3 — H6_only (Recall@5% earlystop, year_weights, W9a on)"
echo "======================================================"
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=year_weights
export STAGE2_ECO_GROUPS=1
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_only \
    --notes "H6: Recall@5% early stopping; year_weights unchanged; W9a on"

echo ""
echo "======================================================"
echo "Run 2/3 — H6_H1b (Recall@5% earlystop, inv_sqrt_npos, W9a on)"
echo "======================================================"
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_ECO_GROUPS=1
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H1b \
    --notes "H6+H1b: Recall@5% earlystop + inv_sqrt_npos group weights; W9a on"

echo ""
echo "======================================================"
echo "Run 3/3 — H6_H1b_H3 (Recall@5% earlystop, inv_sqrt_npos, W9a off)"
echo "======================================================"
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_ECO_GROUPS=0
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H1b_H3 \
    --notes "H6+H1b+H3: Recall@5% earlystop + inv_sqrt_npos + W9a off"

echo ""
echo "All 3 runs complete."
echo "Check proxy Recall@5% in: outputs/south_america/results/feature_ablation_sa.json"
echo "If any run > 0.20 Recall@5% → promote to full SA Euler job."
