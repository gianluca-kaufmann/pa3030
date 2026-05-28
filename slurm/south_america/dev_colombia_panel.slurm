#!/bin/bash
#SBATCH --job-name=col_panel
#SBATCH --partition=normal.24h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/col_panel_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/col_panel_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis
set -euo pipefail

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Creating Colombia dev panel from SA Stage 2 splits..."
srun python scripts/regions/south_america/dev/create_colombia_panel.py --country-id 5
echo "Colombia panel creation done."
