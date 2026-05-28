#!/bin/bash
#SBATCH --job-name=model2_LGBM_stage2_binary
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/model2_LGBM_stage2_binary_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model2_LGBM_stage2_binary_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs" "$SCRATCH/data" "$SCRATCH/outputs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# Binary variant: neg_ratio=100 for memory safety (Issue O fix). True SPW is
# pre-scanned from the full data so scale_pos_weight reflects real class imbalance.
export STAGE2_NEG_RATIO=100

echo "Starting model2_LGBM_stage2_binary (USA)..."
srun --cpu-bind=cores --distribution=block:block python -u \
  scripts/regions/usa/5_training/model2_LGBM_stage2_binary
echo "Done."
