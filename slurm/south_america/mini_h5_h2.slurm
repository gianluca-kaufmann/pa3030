#!/bin/bash
#SBATCH --job-name=mini_h5_h2
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/mini_h5_h2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/mini_h5_h2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# H5 and H2 mini-sample experiment batch — built on H6+H1b locked baseline.
#
# Root cause addressed: rank normalisation destroys absolute feature signals
# (globally high biodiversity, globally low population) that might distinguish
# protected pixels from non-protected pixels across country-years. Graded
# relevance (1–4) amplifies gradient from large cluster events, counteracting
# the H1b group-normalisation.
#
# Locked baseline (H6+H1b): Recall@5%=27.9%, Lift@1%=12.86× (mini proxy)
#
#   Run 1: H6+H1b+H5  — no rank normalisation; absolute feature values preserved
#           Hypothesis: model learns "globally top-5% biodiversity → protect"
#           rather than "top-5% biodiversity within this country-year → protect"
#
#   Run 2: H6+H1b+H2  — binary labels (0/1) instead of graded 1–4
#           Hypothesis: removes secondary gradient bias from large cluster events
#           (all positive pixels equal regardless of cluster size)
#
#   Run 3: H6+H1b+H5+H2 — both: absolute features + binary labels
#           Tests whether the combination compounds the individual gains
#
# All runs share: Recall@5% early stopping (H6), inv_sqrt_npos weights (H1b),
# eco sub-groups on (W9a), neg_ratio=100, baseline hyperparameters.
#
# Decision rule: if any run shows proxy Recall@5% > 0.279 → promote to full SA.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
PARAMS="outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json"
TRAIN_CMD="scripts/regions/south_america/5_training/model1_phase1_local_train --params $PARAMS"

echo "======================================================"
echo "Run 1/3 — H6+H1b+H5 (no rank normalisation)"
echo "======================================================"
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GRADED_RELEVANCE=1
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H1b_H5 \
    --notes "H6+H1b+H5: Recall@5% earlystop + inv_sqrt_npos + no rank normalisation"

echo ""
echo "======================================================"
echo "Run 2/3 — H6+H1b+H2 (binary labels)"
echo "======================================================"
export STAGE2_NORMALIZE_WITHIN_GROUPS=1
export STAGE2_GRADED_RELEVANCE=0
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H1b_H2 \
    --notes "H6+H1b+H2: Recall@5% earlystop + inv_sqrt_npos + binary labels (0/1)"

echo ""
echo "======================================================"
echo "Run 3/3 — H6+H1b+H5+H2 (no normalisation + binary labels)"
echo "======================================================"
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GRADED_RELEVANCE=0
srun --cpu-bind=cores python -u $TRAIN_CMD \
    --feature-tag H6_H1b_H5_H2 \
    --notes "H6+H1b+H5+H2: Recall@5% earlystop + inv_sqrt_npos + no rank norm + binary labels"

echo ""
echo "All 3 runs complete."
echo "Check proxy Recall@5% in: outputs/south_america/results/feature_ablation_sa.json"
echo "Baseline: H6+H1b proxy Recall@5%=0.279"
echo "If any run > 0.279 Recall@5% → promote to full SA Euler job."
