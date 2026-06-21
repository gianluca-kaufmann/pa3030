#!/bin/bash
#SBATCH --job-name=ws_diag_sa
#SBATCH --partition=normal.4h
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/ws_diag_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ws_diag_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "=== CE.W Diagnostic ==="
echo "Start: $(date)"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/5_training/ws_diagnostic.py

echo "Done: $(date)  exit=$?"
