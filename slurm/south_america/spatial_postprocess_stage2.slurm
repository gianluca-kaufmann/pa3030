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

# Tier 1 spatial diffusion sweep — Stage 2 post-processing.
#
# Loads the locked H6+H1b+H5 booster (20260617_011621) and the full SA
# test.parquet, then sweeps multi-step 8-neighbour mean diffusion:
#
#   s_k[i] += alpha × 0.95^k × mean(s_{k-1}[neighbours of i])
#
# Sweeps n_steps ∈ {0,1,5,10,20} × alpha ∈ {0.1,0.3,0.5,1.0,2.0}.
# n_steps=0 is the unmodified baseline (sanity check — should give Lift=6.46×).
# Results saved to outputs/south_america/results/spatial_diffusion_sweep.json

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Tier 1 spatial diffusion sweep — SA Stage 2 (H6+H1b+H5 booster)"
srun --cpu-bind=cores python -u \
    scripts/regions/south_america/6_evaluation/spatial_postprocess_stage2.py

echo "Done. Results: outputs/south_america/results/spatial_diffusion_sweep.json"
