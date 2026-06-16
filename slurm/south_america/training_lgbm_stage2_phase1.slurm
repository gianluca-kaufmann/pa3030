#!/bin/bash
#SBATCH --job-name=phase1_train_sa
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/phase1_train_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/phase1_train_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Phase 1 Euler Gate — Step 3: Full SA training + official test evaluation.
#
# Run after tuning_lgbm_stage2_phase1.slurm has completed.  Trains the final
# model on full SA train+earlystop splits and evaluates Lift@1% + Recall@5%
# on the full SA test set (2017-2024, 107M rows, 61 groups).
#
# These are the official Phase 1 metrics for the feature ablation log.
# Update feature_ablation_sa.json manually with:
#   full_sa_lift_at_1pct, full_sa_recall_at_5pct, kept (true/false)
#
# Lift@1% and Recall@5% are printed in the job output.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1
# H6: Recall@5% early stopping (set to "recall_at_5pct" to enable; default = "ndcg_at_1pct")
export STAGE2_EARLY_STOP_METRIC=ndcg_at_1pct
# H1b: group-normalised weights (set to "inv_sqrt_npos" or "inv_npos"; default = "year_weights")
export STAGE2_GROUP_WEIGHT_MODE=year_weights

echo "Phase 1 full SA training  neg_ratio=$STAGE2_NEG_RATIO  eco_groups=$STAGE2_ECO_GROUPS  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2

echo "Training complete. Record Lift@1% and Recall@5% in:"
echo "  outputs/south_america/results/feature_ablation_sa.json"
