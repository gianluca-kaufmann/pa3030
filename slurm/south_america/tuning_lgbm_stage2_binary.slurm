#!/bin/bash
#SBATCH --job-name=model1_tune_stage2_binary
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/model1_tune_stage2_binary_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_tune_stage2_binary_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# Binary variant: neg_ratio=100 for memory safety; true SPW pre-scanned from full data
# (Issue O fix) so scale_pos_weight reflects real imbalance (~300-500 for SA), not ~100.
export STAGE2_NEG_RATIO=100

echo "Starting Stage 2 binary tuning (SA)..."
python -u scripts/regions/south_america/5_training/model1_tune_stage2_binary
echo "Done."
