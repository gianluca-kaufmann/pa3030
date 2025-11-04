#!/bin/bash
#SBATCH --job-name=merge-2012
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --output=/cluster/scratch/%u/logs/merge_2012_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/merge_2012_%j.err

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/Results/merged_tifs"
mkdir -p "$SCRATCH/wandb/merge"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

#------------------------------------------------------------------------------
# Activate the Python virtual environment (uav-genAI-opti)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venv/master-thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Navigate to training script directory
#------------------------------------------------------------------------------
cd "$HOME/master-thesis" || { echo "ERROR: Could not cd into directory"; exit 1; }

#------------------------------------------------------------------------------
# Run the training
#------------------------------------------------------------------------------
echo "Starting merge script..."

# Run the HPO script
python scripts/merging/merge_2012_optimized

echo "merge script finished."