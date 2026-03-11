#!/bin/bash
#SBATCH --job-name=inspect_panel
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/inspect_panel_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/inspect_panel_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/south_america/tables"

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
# Activate the Python virtual environment (master-thesis)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

#------------------------------------------------------------------------------
# Run the inspect_panel script
#------------------------------------------------------------------------------
echo "Starting inspect_panel script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/3_merging/consistency_checks/inspect_panel" ] || {
  echo "ERROR: inspect_panel not found"
  echo "NOTE: inspect_panel script is in 3_merging/consistency checks/ directory"
  find scripts -maxdepth 4 -type f -name "inspect_panel"
  exit 1
}
# Run the script
python "scripts/regions/south_america/3_merging/consistency_checks/inspect_panel"

echo "inspect_panel script finished."
