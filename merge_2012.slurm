#!/bin/bash

#SBATCH --job-name=merge-2012
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
# keep ONLY this CPU/mem pair (128 GiB total)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/merge_2012_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/merge_2012_%j.err

# paths
export DATA_ROOT="$SCRATCH/data"
export OUTPUT_ROOT="$SCRATCH/outputs"
mkdir -p "$SCRATCH/logs" "$SCRATCH/datasets" "$SCRATCH/outputs/Results/merged_tifs"

# modules
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy

# venv
source "$HOME/venv/master-thesis/bin/activate"

# W&B (no key in file; you already did `wandb login` once in this venv)
export WANDB_ENTITY=gikaufmann
export WANDB_PROJECT=master_thesis

# run
cd "$HOME/master_thesis" || { echo "ERROR: cd failed"; exit 1; }
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Starting merge script at $(date)"

python scripts/merging/merge_2012_optimized

echo "Merge script finished at $(date)"