#!/bin/bash
#SBATCH --job-name=pre_merge_overview
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/pre_merge_overview_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/pre_merge_overview_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting pre-merge overview..."
SCRIPT="scripts/regions/se_asia/2_preprocessing/pre-merge_overview"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }
python "$SCRIPT"
echo "Pre-merge overview finished."
