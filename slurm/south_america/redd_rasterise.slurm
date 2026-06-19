#!/bin/bash
#SBATCH --job-name=redd_rasterise
#SBATCH --partition=normal.4h
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/%u/logs/redd_rasterise_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/redd_rasterise_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.4 step 2 — Rasterise ID-RECCO V5.0 REDD+ projects to 1 km SA grid.
# Input:  data/REDD/ID-RECCO V5.0_20231201.zip  (1.4 MB — already on Euler)
# Output: data/south_america/ready/REDD/redd_sa.tif
# Follow-up: agb_inject_splits.slurm → redd_inject_splits.slurm

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "CE.4 step 2: REDD rasterise"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/2_preprocessing/redd_rasterise.py
STATUS=$?

echo "Done: $(date)  exit=$STATUS"
exit $STATUS
