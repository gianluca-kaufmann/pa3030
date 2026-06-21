#!/bin/bash
#SBATCH --job-name=ws_build_sa
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/ws_build_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ws_build_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# B5 prep: aggregate full SA pixel splits to HydroSHEDS L7 catchment level.
# Produces $SCRATCH/data/south_america/ml/watershed/watershed_full_sa.parquet
#
# Prerequisites before submitting:
#   1. Upload catchment_id_sa.tif to Euler:
#      rsync data/south_america/ready/hydroshed/catchment_id_sa.tif \
#            euler:$SCRATCH/data/south_america/ready/hydroshed/catchment_id_sa.tif
#   2. Confirm full SA pixel splits exist:
#      ls $SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet
#
# Submit:
#   JOB=$(sbatch --parsable slurm/south_america/watershed_build.slurm)
#   sbatch --dependency=afterok:$JOB slurm/south_america/watershed_ce_w.slurm

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data/south_america/ml/watershed"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "=== B5: Watershed full SA build ==="
echo "Start: $(date)"
echo "Catchment TIF: $SCRATCH/data/south_america/ready/hydroshed/catchment_id_sa.tif"
echo "Output:        $SCRATCH/data/south_america/ml/watershed/watershed_full_sa.parquet"

for F in \
    "$SCRATCH/data/south_america/ml/main/train.parquet" \
    "$SCRATCH/data/south_america/ml/main/earlystop.parquet" \
    "$SCRATCH/data/south_america/ml/main/test.parquet" \
    "$SCRATCH/data/south_america/ready/hydroshed/catchment_id_sa.tif"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: required file not found: $F" >&2
        exit 1
    fi
done

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/3_merging/build_watershed_full_splits.py
STATUS=$?

echo "Done: $(date)  exit=$STATUS"
exit $STATUS
