#!/bin/bash
#SBATCH --job-name=sa_build_patch_v2
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_build_patch_v2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_build_patch_v2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# P1.2 — Rebuild patch panels with enriched features:
#   Shape:       log_patch_perimeter_km, patch_compactness (4πA/P²)
#   Heterog.:    patch_std_gsn_b2, patch_max_gsn_b2, patch_max_ndvi,
#                patch_std_ndvi, patch_std_elevation, patch_max_deforestation
#
# These features break the size-only degeneracy by giving the model information
# about within-patch ecological heterogeneity and patch shape.
#
# Overwrites existing patch_{train,earlystop,test}.parquet on scratch.
#
# After this job completes, submit training_lgbm_stage2_patch_v2.slurm.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Building enriched patch panel (P1.2)"
echo "  Splits: $SCRATCH/data/south_america/ml/main/"
echo "  Started: $(date)"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/3_merging/build_patch_panel.py

echo "Done: $(date)"
