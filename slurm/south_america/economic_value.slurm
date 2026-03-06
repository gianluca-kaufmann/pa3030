#!/bin/bash
#SBATCH --job-name=economic_value_preprocessing
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/economic_value_preprocessing_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/economic_value_preprocessing_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/south_america/tables"

set -euo pipefail

cd ~/master_thesis
source ~/.zshrc
conda activate pa3030

ln -sfn /cluster/scratch/gikaufmann/data data

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

#------------------------------------------------------------------------------
# Run the economic_value_preprocessing script
#------------------------------------------------------------------------------
echo "Starting economic_value_preprocessing script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/2_preprocessing/economic_value_preprocessing" ] || {
  echo "ERROR: economic_value_preprocessing not found"
  echo "NOTE: economic_value_preprocessing script is in 2_preprocessing/ directory"
  find scripts -maxdepth 4 -type f -name "economic_value_preprocessing"
  exit 1
}
# Run the script
python scripts/regions/south_america/2_preprocessing/economic_value_preprocessing

echo "economic_value_preprocessing script finished."
