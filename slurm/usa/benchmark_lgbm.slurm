#!/bin/bash
#SBATCH --job-name=benchmark_2
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/benchmark_2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/benchmark_2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/usa/tables"

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
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Run the benchmark_2 script (main split)
#------------------------------------------------------------------------------
echo "Starting benchmark_2 script (main split)..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/usa/6_evaluation/benchmark_2" ] || {
  echo "ERROR: benchmark_2 not found"
  echo "NOTE: benchmark_2 script is in ML/evaluation/ directory"
  find scripts -maxdepth 4 -type f -name "benchmark_2"
  exit 1
}
# Run the script (defaults to main split, LGBM model type)
python "scripts/regions/usa/6_evaluation/benchmark_2" --model_type lgbm --split-version main

echo "benchmark_2 script (main split) finished."
