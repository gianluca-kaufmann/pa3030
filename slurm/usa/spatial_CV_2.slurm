#!/bin/bash
#SBATCH --job-name=spatial_CV_2_USA
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/spatial_CV_2_USA_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/spatial_CV_2_USA_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Layer 2 spatial CV — biome-stratified metrics — USA
#
# Loads model2 scored parquets, assigns WWF biome labels via GSN raster,
# and computes per-biome metrics on the test years (2017-2019).
# Requires the full training pipeline (step 5) to have run first.
#
# Usage:
#   sbatch slurm/usa/spatial_CV_2.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/usa/results/spatial_generalisation"

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
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

#------------------------------------------------------------------------------
# Preflight checks
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/usa/6_evaluation/spatial_CV_2"
[ -f "$SCRIPT" ] || {
  echo "ERROR: $SCRIPT not found"
  find scripts -maxdepth 6 -type f -name "spatial_CV_2" 2>/dev/null
  exit 1
}

#------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------
echo "Starting spatial_CV_2 (USA)..."
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT"

echo "spatial_CV_2 (USA) finished."
