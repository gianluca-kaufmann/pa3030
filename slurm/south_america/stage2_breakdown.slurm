#!/bin/bash
#SBATCH --job-name=stage2_breakdown_sa
#SBATCH --partition=normal.24h
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/stage2_breakdown_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/stage2_breakdown_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Running Issue C: SA Stage 2 year/country breakdown..."
srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/6_evaluation/stage2_year_country_breakdown.py
echo "Done."
