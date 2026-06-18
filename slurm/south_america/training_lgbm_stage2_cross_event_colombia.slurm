#!/bin/bash
#SBATCH --job-name=ce_col
#SBATCH --partition=normal.24h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/ce_col_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ce_col_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Cross-event Stage 2 pilot on Colombia dev panel (CE.1).
# Colombia has ~20 meaningful events (≥200 pos); split → 14 train / 2 val / 4 test.
# Decision gate: macro Recall@5% >= 0.60 → submit training_lgbm_stage2_cross_event.slurm
#
# Settings: H6 (Recall@5% early stop) + H1b (inv_sqrt_npos) + H5 (no rank-norm).
# Colombia data is local at $HOME/master_thesis/data/dev/south_america/ml.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_DATA_ROOT="$HOME/master_thesis/data/dev/south_america/ml"
export STAGE2_NEG_RATIO=100
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GRADED_RELEVANCE=1

echo "Cross-event Stage 2 — Colombia pilot"
echo "  normalize=$STAGE2_NORMALIZE_WITHIN_GROUPS  weights=$STAGE2_GROUP_WEIGHT_MODE  es=$STAGE2_EARLY_STOP_METRIC"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_cross_event.py

echo "Done. Check decision gate in log: Recall@5% >= 0.60 → submit cross_event full SA."
