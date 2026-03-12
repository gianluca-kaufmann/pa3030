#!/bin/bash
#SBATCH --job-name=benchmark_1
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/benchmark_1_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/benchmark_1_%j.err
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
# Activate the Python virtual environment (uav-genAI-opti)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

#------------------------------------------------------------------------------
# Run the benchmark_1 script (main split)
#------------------------------------------------------------------------------
echo "Starting benchmark_1 script (main split)..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/south_america/6_evaluation/benchmark_1" ] || {
  echo "ERROR: benchmark_1 not found"
  echo "NOTE: benchmark_1 script is in ML/evaluation/ directory"
  find scripts -maxdepth 4 -type f -name "benchmark_1"
  exit 1
}
# Run the script (defaults to main split, LGBM model type)
python "scripts/regions/south_america/6_evaluation/benchmark_1" --model_type lgbm --split-version main

echo "benchmark_1 script (main split) finished."
