#!/bin/bash
#SBATCH --job-name=sa_h7
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h7_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h7_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Full SA training: H6+H1b+H5+H7
#   H7: STAGE2_TRAIN_YEAR_MIN=2010 — drop all 2001-2009 training groups.
#       Leaves 25 training groups (2010-2013), all from the post-mega-event era.
#   H6 + H1b + H5: locked baseline (Recall@5% stop + inv_sqrt_npos weights + no rank norm).
#
# Rationale: H8 (soft temporal decay) failed on full SA — Lift=4.17× vs 6.46× for H5 alone.
# H7 is the hard version: drop all ancient groups entirely. Tests whether 98.6% gradient
# from 2001-2009 mega-events is the root cause of poor test-era generalisation.
#
# Risk: only 25 training groups — training may be less stable.
# Cannot be tested on mini-sample (only 1 post-2009 year in mini train).
#
# Decision rule:
#   Lift > 6.46× → old data actively harmful; H7 becomes new baseline.
#   Lift < 6.46× but Recall ↑ substantially → H7 as Recall-specific intervention.
#   Both regress → 25 groups too few; year restriction abandoned.
#
# Record result in feature_ablation_sa.json and ROADMAP.md.

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
export STAGE2_GRADED_RELEVANCE=1
export STAGE2_TRAIN_YEAR_MIN=2010

echo "Full SA H6+H1b+H5+H7  neg_ratio=$STAGE2_NEG_RATIO  eco_groups=$STAGE2_ECO_GROUPS  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE  normalize=$STAGE2_NORMALIZE_WITHIN_GROUPS  graded=$STAGE2_GRADED_RELEVANCE  train_year_min=$STAGE2_TRAIN_YEAR_MIN"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2

echo "Training complete. Record Lift@1% and Recall@5% in:"
echo "  outputs/south_america/results/feature_ablation_sa.json"
echo "  ROADMAP.md"
