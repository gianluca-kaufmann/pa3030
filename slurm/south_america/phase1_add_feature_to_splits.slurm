#!/bin/bash
#SBATCH --job-name=phase1_feature_inject
#SBATCH --partition=normal.24h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/phase1_feature_inject_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/phase1_feature_inject_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Phase 1 Euler Gate — Step 1: Inject new feature raster into full SA splits.
#
# Run this once for each new feature that passes the local proxy gate.
# After completion, submit tuning_lgbm_stage2_phase1.slurm.
#
# Required env vars (set before sbatch):
#   PHASE1_TIF   — path to raster on SCRATCH (e.g. $SCRATCH/data/south_america/ready/KBA/kba_sa.tif)
#   PHASE1_COLS  — space-separated column names, one per band (e.g. "is_kba dist_kba_km")
#
# Example:
#   PHASE1_TIF=$SCRATCH/data/south_america/ready/KBA/kba_sa.tif \
#   PHASE1_COLS="is_kba dist_kba_km" \
#   sbatch slurm/south_america/phase1_add_feature_to_splits.slurm

mkdir -p "$SCRATCH/logs"

if [ -z "$PHASE1_TIF" ] || [ -z "$PHASE1_COLS" ]; then
    echo "ERROR: PHASE1_TIF and PHASE1_COLS must be set." >&2
    exit 1
fi

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Injecting feature columns: $PHASE1_COLS"
echo "Source TIF: $PHASE1_TIF"

# Convert space-separated PHASE1_COLS to --cols args
read -ra COL_ARRAY <<< "$PHASE1_COLS"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/3_merging/add_feature_to_splits.py \
    --tif "$PHASE1_TIF" \
    --cols "${COL_ARRAY[@]}" \
    --inplace

echo "Done. Review output, then submit:"
echo "  sbatch slurm/south_america/tuning_lgbm_stage2_phase1.slurm"
