#!/bin/bash
#SBATCH --job-name=ce3a_coh
#SBATCH --partition=normal.4h
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/ce3a_coherence_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ce3a_coherence_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.3a — Spatial coherence diagnostic (lightweight, ~30 min).
# Reads only row/col/target from parquets; no training.
# Output: outputs/south_america/results/ml_models/stage2_event_coherence_diagnostic.json
# Follow-up: set STAGE2_COHERENCE_THRESHOLD from the recommended_threshold field,
#             then submit training_lgbm_stage2_ce3b.slurm.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "CE.3a: spatial coherence diagnostic for SA expansion events"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/6_evaluation/stage2_event_coherence.py

echo "Done. Check outputs/south_america/results/ml_models/stage2_event_coherence_diagnostic.json"
echo "Copy recommended_threshold into training_lgbm_stage2_ce3b.slurm before submitting CE.3b."
