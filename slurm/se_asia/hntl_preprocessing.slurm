#!/bin/bash
#SBATCH --job-name=hntl_preprocess_sa
#SBATCH --partition=normal.24h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/hntl_preprocess_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/hntl_preprocess_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data/se_asia/ready/HNTL"

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
# Run the HNTL preprocessing script for South East Asia
#------------------------------------------------------------------------------
echo "Starting HNTL preprocessing for South East Asia..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/se_asia/2_preprocessing/HNTL_preprocessing"
[ -f "$SCRIPT" ] || {
  echo "ERROR: HNTL_preprocessing not found at $SCRIPT"
  find scripts -maxdepth 5 -type f -name "HNTL_preprocessing"
  exit 1
}

# Verify source data exists
echo "Checking source data..."
ls "$SCRATCH/data/shared/HNTL/" | head -5 || {
  echo "ERROR: No HNTL source data found at \$SCRATCH/data/shared/HNTL/"
  exit 1
}

echo "Checking backbone..."
ls -l "$SCRATCH/data/se_asia/ready/backbone/backbone.tif" || {
  echo "ERROR: Backbone not found at \$SCRATCH/data/se_asia/ready/backbone/backbone.tif"
  exit 1
}

# Run the script
python "$SCRIPT"

echo "HNTL preprocessing for South East Asia finished."
