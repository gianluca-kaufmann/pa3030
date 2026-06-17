#!/bin/bash
#SBATCH --job-name=sa_patch_h11
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_patch_h11_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_patch_h11_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Compute connected-component patch-context features (H11) and augment splits.
#
# Reads merged_panel_final.parquet year-by-year and computes 4 features per pixel:
#   log_patch_size_km2:      log(n_pixels) of the 8-connected unprotected patch
#   patch_pa_adjacency_frac: fraction of patch pixels with dist_wdpa ≤ 1500 m
#   patch_mean_gsn_b2:       mean GSN biodiversity score across the patch
#   patch_designation_lag1:  1 if any neighbour of this patch was designated t-1
#
# Output: $SCRATCH/data/south_america/ml/patch_features_sa.parquet
# Then LEFT JOINs each split (train/earlystop/test) in-place via DuckDB.
# Augmented splits are written back to $SCRATCH/data/south_america/ml/main/.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "H11 patch feature computation — SA merged panel → splits augmentation"
srun --cpu-bind=cores python -u \
    scripts/regions/south_america/3_merging/add_patch_features_to_splits.py

echo "Done. Splits augmented with patch features."
echo "Next: training_lgbm_stage2_h6_h1b_h5_h11.slurm"
