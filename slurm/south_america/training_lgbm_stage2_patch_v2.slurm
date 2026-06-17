#!/bin/bash
#SBATCH --job-name=sa_h12_v2
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h12_v2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h12_v2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# P1.2 — Train patch LambdaRank on enriched panel (shape + heterogeneity features).
#
# New features vs H12 baseline:
#   Shape:       log_patch_perimeter_km, patch_compactness (4πA/P²)
#   Heterog.:    patch_std_gsn_b2, patch_max_gsn_b2, patch_max_ndvi,
#                patch_std_ndvi, patch_std_elevation, patch_max_deforestation
#
# Decision rule (compare to H12 baseline: Recall@5%=87.7%, Lift@1%=81.32×):
#   Improvement  → enriched features help; continue adding features (P1.3)
#   No change    → size-sort still dominates; investigate feature importance
#   Regression   → new features introduce noise; inspect individually
#
# Submit after build_patch_panel_v2.slurm completes:
#   sbatch --dependency=afterok:<BUILD_JOB_ID> slurm/south_america/training_lgbm_stage2_patch_v2.slurm

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_RUN_TAG=v2_enriched

echo "Patch-level LambdaRank — enriched features (P1.2)"
echo "  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE"
echo "  run_tag=$STAGE2_RUN_TAG"
echo "  Started: $(date)"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_patch

echo "Training complete: $(date)"
echo "Record Lift@1% and Recall@5% in ROADMAP.md (P1.2)"
