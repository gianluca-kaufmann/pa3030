#!/bin/bash
#SBATCH --job-name=model3_LGBM_stage2
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/model3_LGBM_stage2_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model3_LGBM_stage2_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs" "$SCRATCH/outputs"
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy
source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# To test higher truncation level after initial training (recommended experiment):
# uncomment the export below, change the job name, and resubmit.
# export STAGE2_TRUNCATION_LEVEL=1500

python -u scripts/regions/se_asia/5_training/model3_LGBM_stage2

echo "Starting naive baseline (dist_wdpa only)..."
python -u scripts/regions/se_asia/5_training/model3_LGBM_stage2_naive
echo "Naive baseline done."
