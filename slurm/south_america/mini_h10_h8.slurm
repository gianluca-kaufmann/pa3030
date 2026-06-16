#!/bin/bash
#SBATCH --job-name=mini_h10_h8
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/mini_h10_h8_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/mini_h10_h8_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# H10 and H8 mini-sample experiment batch — built on H6+H1b+H5 (new candidate baseline).
#
# Motivation: every experiment so far trades Lift for Recall. The publication bar
# requires BOTH ≥15× Lift AND ≥90% Recall simultaneously. We need a training regime
# that achieves both.
#
# Observation:
#   NDCG@1% stopping:     Lift=20.32×, Recall=13.3%  (high lift, low recall)
#   Recall@5% stopping:   Lift=12.86×, Recall=27.9%  (lower lift, higher recall)
#   H5 + Recall@5% stop: Lift=11.52×, Recall=34.8%  (lowest lift, highest recall)
#
# H10: STAGE2_EARLY_STOP_METRIC=ndcg_x_recall
#   Stop on ndcg_at_1pct × recall_at_5pct. This combined product is maximised only
#   when BOTH metrics are high. Hypothesis: training under this objective avoids the
#   lift/recall trade-off by jointly rewarding good ranking AND good retrieval.
#
#   Run 1: H6+H1b+H10           — H5 off; just combined stopping
#   Run 2: H6+H1b+H5+H10        — no rank norm + combined stopping
#
# H8: STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos_temporal
#   Combined size + temporal gradient weights. Addresses BOTH gradient concentration
#   sources: event size (H1b) AND event age. Decay=0.2/year from max train year.
#
#   Run 3: H6+H8+H10             — temporal+size weights + combined stopping
#   Run 4: H6+H8+H5+H10         — all four improvements together
#
# Proxy baseline (H6+H1b):  Lift=12.86×, Recall=27.9%
# Proxy (H5):               Lift=11.52×, Recall=34.8%
# Target: BOTH > baseline on at least one metric without hurting the other.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1
export STAGE2_EARLY_STOP_METRIC=ndcg_x_recall
PARAMS="outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json"
TRAIN_CMD="scripts/regions/south_america/5_training/model1_phase1_local_train --params $PARAMS"

echo "======================================================"
echo "Run 1/4 — H6+H1b+H10 (ndcg×recall stopping, rank-norm on)"
echo "======================================================"
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_NORMALIZE_WITHIN_GROUPS=1
export STAGE2_GRADED_RELEVANCE=1
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H1b_H10 \
    --notes "H10: combined ndcg_at_1pct x recall_at_5pct stopping; inv_sqrt_npos; rank-norm on"

echo ""
echo "======================================================"
echo "Run 2/4 — H6+H1b+H5+H10 (ndcg×recall stopping, no rank norm)"
echo "======================================================"
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GRADED_RELEVANCE=1
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H1b_H5_H10 \
    --notes "H5+H10: no rank norm + combined ndcg×recall stopping; inv_sqrt_npos"

echo ""
echo "======================================================"
echo "Run 3/4 — H6+H8+H10 (temporal+size weights, ndcg×recall stopping)"
echo "======================================================"
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos_temporal
export STAGE2_NORMALIZE_WITHIN_GROUPS=1
export STAGE2_GRADED_RELEVANCE=1
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H8_H10 \
    --notes "H8+H10: inv_sqrt_npos_temporal weights + combined ndcg×recall stopping"

echo ""
echo "======================================================"
echo "Run 4/4 — H6+H8+H5+H10 (all four: temporal+size, no norm, combined stop)"
echo "======================================================"
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos_temporal
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GRADED_RELEVANCE=1
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H8_H5_H10 \
    --notes "H5+H8+H10: no rank norm + temporal+size weights + combined ndcg×recall stopping"

echo ""
echo "All 4 runs complete."
echo "Compare proxy Lift@1% AND Recall@5% vs baselines:"
echo "  H6+H1b:     Lift=12.86x  Recall=27.9%"
echo "  H6+H1b+H5:  Lift=11.52x  Recall=34.8%"
echo "Target: runs with BOTH higher than H6+H1b → promote to full SA."
