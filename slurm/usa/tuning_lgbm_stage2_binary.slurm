#!/bin/bash
#SBATCH --job-name=model2_tune_stage2_binary
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/model2_tune_stage2_binary_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model2_tune_stage2_binary_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# Binary variant: no neg_ratio (scale_pos_weight handles imbalance), no truncation level.

echo "Starting Stage 2 binary tuning (USA)..."
python -u scripts/regions/usa/5_training/model2_tune_stage2_binary
echo "Done."
