#!/usr/bin/env bash
# Phase 1 Euler gate: submit the 3-step job chain for full SA evaluation.
#
# Usage (from repo root on Euler):
#   bash slurm/south_america/phase1_euler_gate.sh <feature-tag> <tif-path> <col1> [col2 ...]
#
# Example:
#   bash slurm/south_america/phase1_euler_gate.sh kba \
#       $SCRATCH/data/south_america/ready/KBA/kba_sa.tif \
#       is_kba dist_kba_km
#
# The script submits three SLURM jobs in dependency order:
#   1. phase1_add_feature_to_splits   — inject feature into full SA splits
#   2. tuning_lgbm_stage2_phase1      — retune on full SA (100 trials, 12h)
#   3. training_lgbm_stage2_phase1    — retrain + test eval (official Lift@1%)
#
# Euler gate criterion: Lift@1% on full SA test > current best (3.82×).
# After job 3 completes, update feature_ablation_sa.json manually.

set -euo pipefail

FEATURE_TAG="${1:?Usage: $0 <feature-tag> <tif-path> <col1> [col2 ...]}"
TIF_PATH="${2:?TIF path required}"
shift 2
COLS="$*"   # remaining args are column names

if [ -z "$COLS" ]; then
    echo "ERROR: at least one column name required" >&2
    exit 1
fi

echo "=== Phase 1 Euler gate: $FEATURE_TAG ==="
echo "TIF:     $TIF_PATH"
echo "Columns: $COLS"
echo

# Job 1: inject feature into full SA splits
JOB1=$(PHASE1_TIF="$TIF_PATH" PHASE1_COLS="$COLS" \
    sbatch --parsable slurm/south_america/phase1_add_feature_to_splits.slurm)
echo "Submitted job 1 (feature injection): $JOB1"

# Job 2: retune — depends on job 1
JOB2=$(sbatch --parsable --dependency=afterok:"$JOB1" \
    slurm/south_america/tuning_lgbm_stage2_phase1.slurm)
echo "Submitted job 2 (full SA tuning):    $JOB2"

# Job 3: retrain + test eval — depends on job 2
JOB3=$(sbatch --parsable --dependency=afterok:"$JOB2" \
    slurm/south_america/training_lgbm_stage2_phase1.slurm)
echo "Submitted job 3 (full SA training):  $JOB3"

echo
echo "Job chain: $JOB1 → $JOB2 → $JOB3"
echo "Monitor: squeue -u $USER"
echo "After job 3: check Lift@1% + Recall@5% and update feature_ablation_sa.json"
