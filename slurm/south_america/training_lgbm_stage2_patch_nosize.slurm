#!/bin/bash
#SBATCH --job-name=sa_h12_nosize
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/sa_h12_nosize_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sa_h12_nosize_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# P1.1 — No-size ablation: train H12 without log_patch_size_km2.
#
# Diagnostic: does any ecological signal exist beyond patch size?
#
# If Recall@5% stays >40%  → genuine signal beyond size; add richer features
# If Recall@5% ≈ 10-20%   → size carries almost all information; need new features
# If Recall@5% ≈ 0-5%     → model purely sorts by size; no signal at all
#
# Uses the same patch panels built by build_patch_panel.slurm.
# Reuses locked settings: H6 early stop, H1b weights, H5 off.
#
# Record result in ROADMAP.md experiment table.

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos
export STAGE2_FEATURE_EXCLUDE=log_patch_size_km2
export STAGE2_RUN_TAG=nosize

echo "Patch-level LambdaRank — no-size ablation (P1.1)"
echo "  Excluding: $STAGE2_FEATURE_EXCLUDE"
echo "  es_metric=$STAGE2_EARLY_STOP_METRIC  group_weights=$STAGE2_GROUP_WEIGHT_MODE"
echo "  Started: $(date)"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_patch

echo "Training complete: $(date)"
echo "Record Lift@1% and Recall@5% in ROADMAP.md (P1.1)"
