#!/bin/bash
#SBATCH --job-name=sa_baselines_cmp
#SBATCH --partition=normal.4h
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/sa_baselines_cmp_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_baselines_cmp_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

source "$HOME/venvs/master_thesis/bin/activate"
mkdir -p /cluster/scratch/${USER}/logs

echo "P1.4: SA Stage 2 baselines comparison table..."
srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/7_results/stage2_baselines_comparison.py
echo "Done."
