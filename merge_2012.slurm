#!/bin/bash

#SBATCH --job-name=merge-2012
#SBATCH --time=24:00:00                    # 24 hours for data merging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Use 8 CPUs for parallel processing
#SBATCH --mem-per-cpu=16G                 # 128GB total memory
#SBATCH --output=/cluster/scratch/%u/logs/merge_2012_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/merge_2012_%j.err

# Set data and output paths for clarity
export DATA_ROOT="$SCRATCH/data"
export OUTPUT_ROOT="$SCRATCH/outputs"

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/datasets"
mkdir -p "$SCRATCH/outputs/Results/merged_tifs"

#------------------------------------------------------------------------------
# Load modules and activate virtual environment
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
# Weights & Biases (log online; no key in the slurm file)
#------------------------------------------------------------------------------
export WANDB_ENTITY=gikaufmann
export WANDB_PROJECT=master_thesis
export WANDB_MODE=online

# If a key is present in the environment (e.g., from ~/.bashrc), refresh login silently.
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

# Fail early if this venv isn't logged in yet
wandb login --status >/dev/null 2>&1 || {
  echo "W&B not logged in for this venv. Run once on Euler:"
  echo "  module purge; module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy"
  echo "  source ~/venv/master-thesis/bin/activate"
  echo "  wandb login <YOUR_API_KEY>"
  exit 4
}

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