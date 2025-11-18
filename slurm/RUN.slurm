#!/bin/bash
#SBATCH --job-name=panel-stats
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --output=/cluster/scratch/%u/logs/merge_run_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/merge_run_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

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

#------------------------------------------------------------------------------
# Run the training
#------------------------------------------------------------------------------
echo "Starting panel statistics script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f scripts/consistency\ checks/compute_panel_statistics ] || {
  echo "ERROR: compute_panel_statistics not found under scripts/consistency checks"
  find scripts -maxdepth 2 -type f -name "compute_panel_statistics"
  exit 1
}
# Run the script
python "scripts/consistency checks/compute_panel_statistics"

echo "Panel statistics script finished."