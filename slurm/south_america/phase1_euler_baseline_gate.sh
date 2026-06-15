#!/usr/bin/env bash
# Phase 1 Euler BASELINE gate: train on full SA splits, evaluate Recall@5% + Lift@1%.
#
# No feature injection or retuning. Uses existing params in:
#   scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json
#
# Usage (from repo root on Euler):
#   bash slurm/south_america/phase1_euler_baseline_gate.sh
#
# After job completes, record full_sa_lift_at_1pct + full_sa_recall_at_5pct
# in outputs/south_america/results/feature_ablation_sa.json (baseline entry, kept=true).

set -euo pipefail

echo "=== Phase 1 Euler BASELINE gate ==="
echo "Params: scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json"
echo "Splits: \$SCRATCH/data/south_america/ml/main/"
echo

JOB=$(sbatch --parsable slurm/south_america/training_lgbm_stage2_phase1.slurm)
echo "Submitted training job: $JOB"
echo "Monitor:  squeue -j $JOB"
echo "Log:      \$SCRATCH/logs/phase1_train_sa_${JOB}.out"
echo
echo "After completion, record metrics in feature_ablation_sa.json (baseline entry)."
