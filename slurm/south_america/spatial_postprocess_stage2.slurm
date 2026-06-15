#!/bin/bash
#SBATCH --job-name=s2_spatial_pp
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/spatial_postprocess_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/spatial_postprocess_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Stage 2 spatial post-processing evaluation.
# Loads existing model + full SA test.parquet, applies 8-neighbour score
# propagation for alpha in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0], reports
# Lift@1% + Recall@5% before/after.
# Results saved to outputs/south_america/results/spatial_postprocess_alpha_sweep.json

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Spatial post-processing evaluation — SA Stage 2"
srun --cpu-bind=cores python -u \
    scripts/regions/south_america/6_evaluation/spatial_postprocess_stage2.py

echo "Done. Results: outputs/south_america/results/spatial_postprocess_alpha_sweep.json"
