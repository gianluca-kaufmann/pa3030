#!/bin/bash

#SBATCH --job-name=merge-2012
#SBATCH --time=24:00:00                    # 24 hours for data merging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                # Use 16 CPUs for parallel processing
#SBATCH --mem-per-cpu=4G                  # 64GB total memory
#SBATCH --output=/cluster/scratch/%u/logs/merge_2012_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/merge_2012_%j.err

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/datasets"
mkdir -p "$SCRATCH/outputs/Results/merged_tifs"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy

#------------------------------------------------------------------------------
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venv/master-thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Set up Weights & Biases credentials
#------------------------------------------------------------------------------
# Set your W&B API key here (paste it in on the cluster)
export WANDB_API_KEY="PASTE_YOUR_API_KEY_HERE"
export WANDB_ENTITY="gikaufmann-eth-z-rich"

if [ "$WANDB_API_KEY" = "PASTE_YOUR_API_KEY_HERE" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY not configured yet!"
    echo "   Edit merge_2012.slurm and replace PASTE_YOUR_API_KEY_HERE with your actual API key"
fi

#------------------------------------------------------------------------------
# Navigate to project directory
#------------------------------------------------------------------------------
# Project is located in home directory (git root)
PROJECT_DIR="$HOME/master_thesis"
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd into project directory: $PROJECT_DIR"; exit 1; }

echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Starting merge script at $(date)"

#------------------------------------------------------------------------------
# Run the merge script
#------------------------------------------------------------------------------
python scripts/merging/merge_2012_optimized

echo "Merge script finished at $(date)"