#!/bin/bash
#SBATCH --job-name=agb_rasterise
#SBATCH --partition=normal.4h
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/agb_rasterise_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/agb_rasterise_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.4 step 1 — Rasterise ESA CCI Biomass v4.0 (2010) to 1 km SA grid.
# Input:  data/shared/ESA_CCI_Biomass/*.tif  (29 tiles, 1.7 GB — already on Euler)
# Output: data/south_america/ready/AGB/agb_sa.tif
# Follow-up: agb_inject_splits.slurm

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "CE.4 step 1: AGB rasterise"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/2_preprocessing/agb_rasterise.py
STATUS=$?

echo "Done: $(date)  exit=$STATUS"
exit $STATUS
