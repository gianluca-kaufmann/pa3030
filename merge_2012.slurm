#!/bin/bash

#SBATCH --job-name=merge-2012
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/merge_2012_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/merge_2012_%j.err

set -euo pipefail

# --- Paths (mirror the "worked before" behavior) ---
PROJECT_DIR="$HOME/master_thesis"
export DATA_ROOT="$SCRATCH/data"
export OUTPUT_ROOT="$SCRATCH/outputs"
mkdir -p "$SCRATCH/logs" "$SCRATCH/outputs" "$SCRATCH/outputs/Results/merged_tifs"

# Make HOME->SCRATCH symlinks so scripts that ignore env vars still see the data/outputs
rm -rf "$PROJECT_DIR/data" 2>/dev/null || true
ln -sfn "$DATA_ROOT" "$PROJECT_DIR/data"
rm -rf "$PROJECT_DIR/outputs" 2>/dev/null || true
ln -sfn "$OUTPUT_ROOT" "$PROJECT_DIR/outputs"

# --- Modules & venv (same stack that worked) ---
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy
source "$HOME/venv/master-thesis/bin/activate"

# --- Weights & Biases (online; no key in file) ---
export WANDB_ENTITY="${WANDB_ENTITY:-gikaufmann}"
export WANDB_PROJECT="${WANDB_PROJECT:-merge}"   # this matches your earlier working URLs
export WANDB_MODE=online
# If the venv is already logged in (stored in ~/.netrc), this is enough.

# --- Tame hidden memory spikes (minimal) ---
export GDAL_CACHEMAX=512
export RASTERIO_MAX_DATASET_OPEN=32
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- Run ---
cd "$PROJECT_DIR"
echo "GIT COMMIT: $(git rev-parse --short HEAD || true)"
echo "DATA link -> $(readlink -f "$PROJECT_DIR/data" || true)"
echo "OUTPUT link -> $(readlink -f "$PROJECT_DIR/outputs" || true)"
echo "Python: $(python --version)"
python scripts/merging/merge_2012_optimized 