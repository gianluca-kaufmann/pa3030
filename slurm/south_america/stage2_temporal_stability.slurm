#!/bin/bash
#SBATCH --job-name=sa_temporal_stab
#SBATCH --partition=normal.4h
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/sa_temporal_stab_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_temporal_stab_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

source "$HOME/venvs/master_thesis/bin/activate"
mkdir -p /cluster/scratch/${USER}/logs

echo "P5.2: SA Stage 2 temporal stability (per-year Recall@5%)..."
srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/6_evaluation/stage2_temporal_stability.py
echo "Done."
