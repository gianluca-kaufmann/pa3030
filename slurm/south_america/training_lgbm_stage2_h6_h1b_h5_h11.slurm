#!/bin/bash
#SBATCH --job-name=sa_h11
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h11_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h11_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Full SA training: H6+H1b+H5+H11 (patch-context features)
#
# H11: adds 4 connected-component patch-context features (precomputed by
#      add_patch_features.slurm) that the LambdaRank model picks up automatically
#      because they appear in the augmented splits and are not in STAGE2_EXCLUDE_COLS.
#
#   log_patch_size_km2:      larger patches → more likely to be a viable PA polygon
#   patch_pa_adjacency_frac: patches adjacent to existing PAs → high designation rate
#   patch_mean_gsn_b2:       high-biodiversity patches → political priority
#   patch_designation_lag1:  patches touched by last year's designation → momentum
#
# H6+H1b+H5 env vars are identical to the locked baseline booster (20260617_011621).
# No STAGE2_TRAIN_YEAR_MIN restriction (full 2001-2013 training range).
#
# Decision rule:
#   Recall@5% > 17.2% (current ceiling) → patch features add signal; keep H11.
#   Lift also improves → H11 is a net positive; retrain and lock new booster.
#   Recall stagnates → polygon-framing mismatch requires deeper intervention.
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

echo "Full SA H6+H1b+H5+H11  neg_ratio=$STAGE2_NEG_RATIO  eco_groups=$STAGE2_ECO_GROUPS  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE  normalize=$STAGE2_NORMALIZE_WITHIN_GROUPS  graded=$STAGE2_GRADED_RELEVANCE"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2

echo "Training complete. Record Lift@1% and Recall@5% in:"
echo "  outputs/south_america/results/feature_ablation_sa.json"
echo "  ROADMAP.md"
