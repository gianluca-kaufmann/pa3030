#!/bin/bash
#SBATCH --job-name=agb_inject
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/agb_inject_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/agb_inject_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.4 step 3 — Inject agb_tonne_ha into train/earlystop/test parquets (42 GB).
# Input:  data/south_america/ready/AGB/agb_sa.tif
# Modifies in-place: $SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet
# Follow-up: redd_inject_splits.slurm

mkdir -p "$SCRATCH/logs"

AGB_TIF="$HOME/master_thesis/data/south_america/ready/AGB/agb_sa.tif"

if [ ! -f "$AGB_TIF" ]; then
    echo "ERROR: AGB raster not found at $AGB_TIF — run agb_rasterise.slurm first." >&2
    exit 1
fi

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "CE.4 step 3: Injecting agb_tonne_ha into splits"
echo "  TIF: $AGB_TIF"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/3_merging/add_feature_to_splits.py \
    --tif "$AGB_TIF" \
    --cols agb_tonne_ha \
    --inplace

echo "Done. agb_tonne_ha injected into all three splits."
