#!/bin/bash
#SBATCH --job-name=model1_tune_stage2
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/model1_tune_stage2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_tune_stage2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy
source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=20
python -u scripts/regions/south_america/4_tuning/model1_tuning_lgbm_stage2
