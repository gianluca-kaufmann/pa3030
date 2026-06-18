#!/bin/bash
#SBATCH --job-name=sa_within_group
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_within_group_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_within_group_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# P1.1 — Within-group 80/20 pixel split experiment.
#
# Answers: "Is the spatial logic of designation learnable at all?"
# For each (country_id, year) expansion group, 80% of pixels go to train
# and 20% are held out. The model is trained on temporal data (2001-2013)
# PLUS the 80% within-group pixels, then evaluated on the 20% held-out.
#
# Expected: ~94.5% Recall@5% (confirmed on Colombia scale; unconfirmed on full SA).
# If confirmed: two-evaluation Stage 2 story is locked.
# If substantially lower: re-diagnose.
#
# Same H6+H1b+H5 settings as the locked temporal model (20260617_011621).
# Temporal model result for comparison: Lift@1%=6.46x, Recall@5%=15.7%
#
# Record result in ROADMAP.md experiment table and update architecture section.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# H5: no rank normalisation (absolute feature signals)
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
# H6: Recall@5% early stopping
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
# H1b: inv_sqrt_npos group weights
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
# Graded relevance labels (event cluster size)
export STAGE2_GRADED_RELEVANCE=1
# Negative subsampling for train portion (memory control)
export STAGE2_NEG_RATIO=100

echo "SA within-group 80/20 experiment"
echo "  normalize_within_groups=$STAGE2_NORMALIZE_WITHIN_GROUPS"
echo "  early_stop_metric=$STAGE2_EARLY_STOP_METRIC"
echo "  group_weight_mode=$STAGE2_GROUP_WEIGHT_MODE"
echo "  graded_relevance=$STAGE2_GRADED_RELEVANCE"
echo "  neg_ratio=$STAGE2_NEG_RATIO"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_within_group

echo "Done. Record Recall@5% and Lift@1% in:"
echo "  ROADMAP.md experiment table"
echo "  outputs/south_america/results/ml_models/model1_lgbm_stage2_within_group_metrics_*.json"
