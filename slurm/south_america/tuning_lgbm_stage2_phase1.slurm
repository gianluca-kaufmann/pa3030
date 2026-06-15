#!/bin/bash
#SBATCH --job-name=phase1_tune_sa
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/phase1_tune_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/phase1_tune_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Phase 1 Euler Gate — Step 2: Full SA hyperparameter retune after feature injection.
#
# Run after phase1_add_feature_to_splits.slurm has completed.  Reads the augmented
# full SA train+earlystop splits and retunes for 100 trials to find the best
# hyperparameters for the new feature set.
#
# Output: scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json
#
# Next: sbatch slurm/south_america/training_lgbm_stage2_phase1.slurm

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1
export STAGE2_TUNING_TRIALS=100

echo "Phase 1 full SA tuning  neg_ratio=$STAGE2_NEG_RATIO  eco_groups=$STAGE2_ECO_GROUPS  trials=$STAGE2_TUNING_TRIALS"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/4_tuning/model1_tuning_lgbm_stage2

echo "Tuning complete. Submit:"
echo "  sbatch slurm/south_america/training_lgbm_stage2_phase1.slurm"
