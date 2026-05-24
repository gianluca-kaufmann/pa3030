#!/bin/bash
#SBATCH --job-name=model3_LGBM_stage2
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/model3_LGBM_stage2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model3_LGBM_stage2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs" "$SCRATCH/outputs"
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy
source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# Truncation override: SA tuning hit the [50,500] ceiling (best=499); wider search
# confirms higher is better. 3000 covers ~33% of each 9000-row split sub-group and
# gives gradient signal to high-relevance (rel=4) pixels buried beyond position 500.
export STAGE2_TRUNCATION_LEVEL=3000

python -u scripts/regions/se_asia/5_training/model3_LGBM_stage2

echo "Starting naive baseline (dist_wdpa only)..."
python -u scripts/regions/se_asia/5_training/model3_LGBM_stage2_naive
echo "Naive baseline done."
