#!/bin/bash
#SBATCH --job-name=ce3b_sqrt
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/ce3b_sqrt_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ce3b_sqrt_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.3b regression diagnostic — isolates Fix 2 (inv_npos weighting) as culprit.
#
# CE.3b used inv_npos and scored 18.0% macro R@5% — WORSE than CE.2 (23.8%).
# This run reverts Fix 2 to inv_sqrt_npos while keeping Fix 4 (stratified split
# + patience=200).  If this matches or beats CE.2, inv_npos is the problem.
#
# Variables vs CE.3b:
#   STAGE2_GROUP_WEIGHT_MODE = inv_sqrt_npos  (was inv_npos in CE.3b)
#
# Everything else is identical to CE.3b (stratified split, patience=200,
# coherence_threshold=0.0 since all events scored 1.0 in CE.3a).

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

# --- CE.3b Fix 4 (kept) ---
export STAGE2_STRATIFIED_SPLIT=1
export STAGE2_PATIENCE=200

# --- Fix 2 REVERTED: inv_sqrt_npos (was inv_npos in CE.3b) ---
export STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos

# Fix 1: coherence=0.0 (no-op — CE.3a found all 139 events have coherence=1.0)
export STAGE2_COHERENCE_THRESHOLD=0.0

echo "CE.3b sqrt diagnostic: inv_sqrt_npos + stratified + patience=200"
echo "  weights=$STAGE2_GROUP_WEIGHT_MODE  patience=$STAGE2_PATIENCE"
echo "  stratified=$STAGE2_STRATIFIED_SPLIT  coherence_threshold=$STAGE2_COHERENCE_THRESHOLD"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_cross_event.py
STATUS=$?

echo "Done. srun exit status: $STATUS"
echo "Record result in ROADMAP.md experiment table."
exit $STATUS
