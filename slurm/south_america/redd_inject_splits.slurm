#!/bin/bash
#SBATCH --job-name=redd_inject
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/redd_inject_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/redd_inject_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.4 step 4 — Inject dist_redd_km into train/earlystop/test parquets (42 GB).
# Input:  data/south_america/ready/REDD/redd_sa.tif
# Modifies in-place: $SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet
# Follow-up: submit CE.4 retrain manually (training_lgbm_stage2_ce4.slurm)

mkdir -p "$SCRATCH/logs"

REDD_TIF="$HOME/master_thesis/data/south_america/ready/REDD/redd_sa.tif"

if [ ! -f "$REDD_TIF" ]; then
    echo "ERROR: REDD raster not found at $REDD_TIF — run redd_rasterise.slurm first." >&2
    exit 1
fi

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "CE.4 step 4: Injecting dist_redd_km into splits"
echo "  TIF: $REDD_TIF"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/3_merging/add_feature_to_splits.py \
    --tif "$REDD_TIF" \
    --cols dist_redd_km \
    --inplace

echo "Done. dist_redd_km injected into all three splits."
echo "Splits now have AGB + REDD features. Manually submit CE.4 retrain:"
echo "  sbatch slurm/south_america/training_lgbm_stage2_ce3b.slurm"
echo "  (same script, same settings — uses the enriched splits automatically)"
