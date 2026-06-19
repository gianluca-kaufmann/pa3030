#!/bin/bash
#SBATCH --job-name=stage1_nb_sa
#SBATCH --partition=normal.4h
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/stage1_nb_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/stage1_nb_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# A3 (E6): Stage 1 overdispersion test + Negative Binomial GLM comparison.
# Requires stage1_panel.parquet — submit after stage1.slurm completes.
# Output: outputs/south_america/results/stage1_nb_overdispersion.json

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

PANEL="$SCRATCH/data/south_america/stage1_panel.parquet"
if [ ! -f "$PANEL" ]; then
    echo "ERROR: stage1_panel.parquet not found at $PANEL" >&2
    echo "Submit stage1.slurm first and use --dependency=afterok" >&2
    exit 1
fi

echo "=== A3 (E6): Stage 1 NB overdispersion test ==="
echo "Panel: $PANEL"
echo "Start: $(date)"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/5_training/stage1_nb_overdispersion.py
STATUS=$?

echo "Done: $(date)  exit=$STATUS"
exit $STATUS
