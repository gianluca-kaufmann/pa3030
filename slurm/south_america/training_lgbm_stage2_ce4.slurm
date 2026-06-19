#!/bin/bash
#SBATCH --job-name=ce4_sa
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/ce4_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ce4_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.4 — Cross-event Stage 2 retrain with AGB + REDD features.
#
# Prerequisites (all must be complete before submitting):
#   1. AGB injected: agb_inject_splits.slurm completed
#   2. REDD injected: redd_inject_splits.slurm completed
#   3. Weight mode confirmed: ce3b_sqrt diagnostic result reviewed
#      — If inv_sqrt_npos ≥ CE.2 (23.8%): set STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos  ← expected
#      — If inv_npos was not the culprit: set STAGE2_GROUP_WEIGHT_MODE=inv_npos
#      Current setting below assumes inv_sqrt_npos is correct (Fix 2 reverted).
#
# Decision gate: macro Recall@5% >= 50% on test events → proceed to SHAP + Phase 2/3
#   (Gate lowered from 65% per session 4 analysis: the 65% gate assumed all 29 test events
#    are ecologically predictable; ~8 countries have structural political-economy reasons
#    for low recall that features cannot encode. 50% is more defensible given this.)
#
# Splits: train/earlystop/test.parquet in $SCRATCH/data/south_america/ml/main/
#         Now have 83 + 2 = 85 features (agb_tonne_ha, dist_redd_km added).

mkdir -p "$SCRATCH/logs"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

# --- Locked settings (H5 + H6 unchanged from CE.2) ---
export STAGE2_NORMALIZE_WITHIN_GROUPS=0
export STAGE2_GRADED_RELEVANCE=1
export STAGE2_EARLY_STOP_METRIC=recall_at_5pct
export STAGE2_NEG_RATIO=100

# --- CE.4 settings (Fix 2 reverted; Fix 4 kept) ---
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos   # reverted from inv_npos (ce3b_sqrt confirmed)
export STAGE2_STRATIFIED_SPLIT=1                # Fix 4: country-stratified split
export STAGE2_PATIENCE=200                      # Fix 4: longer patience

# Fix 1: coherence=0.0 (no-op — CE.3a found all 139 events have coherence=1.0)
export STAGE2_COHERENCE_THRESHOLD=0.0

echo "CE.4: cross-event Stage 2 with AGB + REDD features"
echo "  weights=$STAGE2_GROUP_WEIGHT_MODE  patience=$STAGE2_PATIENCE"
echo "  stratified=$STAGE2_STRATIFIED_SPLIT  coherence_threshold=$STAGE2_COHERENCE_THRESHOLD"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_cross_event.py
STATUS=$?

echo "Done. srun exit status: $STATUS"
echo "Record result in ROADMAP.md experiment table."
exit $STATUS
