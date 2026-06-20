#!/bin/bash
#SBATCH --job-name=mini_v2_sa
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/mini_v2_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/mini_v2_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Build SA mini-sample v2 (event-stratified, N_CAP=50000 rows per event).
# Pass 1: count unprotected rows per (country_id, year) — fast, 3 cols only.
# Pass 2: read all 94 cols, apply per-event fraction, write output.
# Expected: ~8-12M rows, ~400-800 MB, ~80-100 meaningful events.
# After completion, rsync to local and run prepare_mini_splits_v2.py.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export N_CAP_PER_EVENT=50000
export MINI_V2_SEED=42

echo "Building SA mini-sample v2 (event-stratified, N_CAP=${N_CAP_PER_EVENT})"
srun --cpu-bind=cores python -u \
    scripts/regions/south_america/3_merging/build_mini_sample_v2.py
STATUS=$?

echo "Done. Exit status: $STATUS"
exit $STATUS
