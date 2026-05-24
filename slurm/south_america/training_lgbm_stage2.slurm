#!/bin/bash
#SBATCH --job-name=model1_LGBM_stage2
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/model1_LGBM_stage2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_LGBM_stage2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs" "$SCRATCH/data" "$SCRATCH/outputs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# SA tuning found truncation=499 (ceiling of old [50,500] range), confirming higher
# is better. 3000 covers ~33% of each 9000-row split sub-group and gives gradient
# to high-relevance (rel=4) pixels buried beyond position 499 in each sub-group.
export STAGE2_TRUNCATION_LEVEL=3000

echo "Starting model1_LGBM_stage2..."
srun --cpu-bind=cores --distribution=block:block python -u \
  scripts/regions/south_america/5_training/model1_LGBM_stage2
echo "Done."

echo "Starting naive baseline (dist_wdpa only)..."
python -u scripts/regions/south_america/5_training/model1_LGBM_stage2_naive
echo "Naive baseline done."
