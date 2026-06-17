#!/bin/bash
#SBATCH --job-name=sa_h12_patch
#SBATCH --partition=normal.24h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h12_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h12_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Tier 3 (H12) — Step 2: Patch-level LambdaRank.
#
# Trains LambdaRank on patch-level panel (one row per connected unprotected
# component per country-year). Ranking unit is the patch; label=1 if any
# pixel inside was designated. This eliminates the polygon-vs-pixel framing
# mismatch that capped pixel-level Recall@5% at ~19%.
#
# Inherits locked baseline settings:
#   H6:  Recall@5% early stopping
#   H1b: inv_sqrt_npos group-norm weights
#   H5:  no rank normalisation (always off at patch level)
#   W9a: eco sub-groups OFF (not applicable at patch level)
#
# Decision rule:
#   Recall@5% > 30% → patch-level framing works; record result and iterate.
#   Recall@5% > 70% → exceeds Tier 2 gate; proceed to hyperparameter tuning.
#   Recall@5% < 30% → deeper structural fix needed; open discussion.
#
# Record result in ROADMAP.md experiment table.
#
# Submit after build_patch_panel.slurm completes:
#   sbatch --dependency=afterok:<BUILD_JOB_ID> slurm/south_america/training_lgbm_stage2_patch.slurm

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos

echo "Patch-level LambdaRank (H12)"
echo "  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE"
echo "  Started: $(date)"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_patch

echo "Training complete: $(date)"
echo "Record Lift@1% and Recall@5% in ROADMAP.md"
