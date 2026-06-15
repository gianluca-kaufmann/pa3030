#!/bin/bash
#SBATCH --job-name=phase1_eco_gap_inject
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/phase1_eco_gap_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/phase1_eco_gap_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Phase 1 Euler Gate — Ecoregion Protection Gap (time-varying feature).
#
# The eco protection gap is time-varying (protection fraction of pixel's ecoregion
# at year t-1), so it cannot use the standard TIF-based phase1_add_feature_to_splits.slurm.
# This script calls add_ecoregion_protection_gap_to_splits.py which:
#   1. Recomputes the (eco_id, year) lookup from WDPA + GSN TIFs on SCRATCH
#   2. Injects eco_protection_ratio_lag1 and eco_protection_gap_lag1 into all splits
#
# Submit in place of phase1_add_feature_to_splits.slurm for this feature:
#   JOB1=$(sbatch --parsable slurm/south_america/phase1_add_eco_protection_gap.slurm)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/south_america/tuning_lgbm_stage2_phase1.slurm)
#   JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/south_america/training_lgbm_stage2_phase1.slurm)

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Injecting ecoregion protection gap into full SA splits..."

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/3_merging/add_ecoregion_protection_gap_to_splits.py \
    --inplace

echo "Done. Submit next:"
echo "  sbatch slurm/south_america/tuning_lgbm_stage2_phase1.slurm"
