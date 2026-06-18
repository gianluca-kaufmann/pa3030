#!/bin/bash
#SBATCH --job-name=sa_grid10km_diag
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/sa_grid10km_diag_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_grid10km_diag_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# P4.1 diagnostic: count 10×10 km cells per (country_id, year) expansion group.
# Reads test.parquet from $SCRATCH, needs ~4 GB for the full streaming load.
# Output: stage2_SA_grid10km_diagnostic.json in outputs/south_america/results/ml_models/
#
# Verdict threshold: viable if all groups have ≥50 cells AND ≥10 positive cells.
# If viable → proceed to rebuild splits at 10 km resolution (P4.1 full experiment).

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "P4.1: 10×10 km grid-size diagnostic..."
srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/6_evaluation/stage2_grid10km_diagnostic.py

echo "Done. Check stage2_SA_grid10km_diagnostic.json for verdict."
