#!/bin/bash
#SBATCH --job-name=sa_naive_full
#SBATCH --partition=normal.4h
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_naive_full_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_naive_full_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# P1.4 — Naive baseline on full 2017-2024 test period.
# Trains dist_wdpa-only LambdaRank, evaluates on all 61 expansion groups (2017-2024).
# Needed for apples-to-apples comparison with H6+H1b+H5 full model.
# Previous naive run only covered 2017-2019 (26 groups).
#
# After this, run stage2_baselines_comparison.py --naive <new_path> for the table.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=0
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GRADED_RELEVANCE=1

echo "SA naive baseline (dist_wdpa only) — full 2017-2024 test period"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2 --naive

echo "Done. Run stage2_baselines_comparison.py with the new naive scored parquet."
