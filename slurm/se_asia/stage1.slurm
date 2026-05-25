#!/bin/bash
#SBATCH --job-name=stage1_sea
#SBATCH --partition=normal.24h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/stage1_sea_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/stage1_sea_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Building Stage 1 country-year panel (SE Asia)..."
python -u scripts/regions/se_asia/5_training/stage1_data_builder.py
echo "Panel done. Running Stage 1 Poisson expansion model (SE Asia)..."
python -u scripts/regions/se_asia/5_training/model3_expansion.py
echo "Stage 1 SE Asia done."
