#!/bin/bash
#SBATCH --job-name=sa_build_patch
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_build_patch_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_build_patch_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Tier 3 (H12) — Step 1: Build patch-level panel from pixel splits.
#
# Reads train/earlystop/test.parquet (pixel level), computes 8-connected
# components of WDPA_prev==0 pixels per year, and aggregates to one row per
# patch (connected unprotected block) within each (country_id, year).
#
# A patch is positive (transition_01=1) if any pixel inside it was designated.
# Distance features (dist_*) are aggregated as min; all others as mean.
#
# Output: patch_{train,earlystop,test}.parquet in
#   $SCRATCH/data/south_america/ml/main/
#
# After this job completes, submit training_lgbm_stage2_patch.slurm with:
#   sbatch --dependency=afterok:$SLURM_JOB_ID slurm/south_america/training_lgbm_stage2_patch.slurm

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Building patch panel (Tier 3 / H12)"
echo "  Splits: $SCRATCH/data/south_america/ml/main/"
echo "  Started: $(date)"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/3_merging/build_patch_panel.py

echo "Done: $(date)"
echo "Submit training job with:"
echo "  sbatch --dependency=afterok:\$SLURM_JOB_ID slurm/south_america/training_lgbm_stage2_patch.slurm"
