#!/bin/bash
#SBATCH --job-name=sa_mini_sample
#SBATCH --partition=normal.24h
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/sa_mini_sample_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_mini_sample_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Issue MS: build SA mini-sample (3-5M rows, natural ratio, stratified by country/year)
# for local Phase 1 feature iteration. Reads SA train.parquet (~20+ GB) in batches.
# Output: $SCRATCH/data/south_america/ml/mini_sample.parquet (~200 MB)
# After completion: rsync to Desktop/thesis/code/data/south_america/mini_sample.parquet

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export MINI_SAMPLE_N=4000000

echo "Building SA mini-sample (target 4M rows, natural ratio)..."
srun python -u scripts/regions/south_america/3_merging/build_mini_sample.py
echo "Mini-sample build done."
