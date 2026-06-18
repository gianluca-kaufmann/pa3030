#!/bin/bash
#SBATCH --job-name=ce3b_sa
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/ce3b_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ce3b_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.3b — Redesigned cross-event Stage 2 with three fixes vs CE.2:
#
#   Fix 2: STAGE2_GROUP_WEIGHT_MODE=inv_npos
#           Each event contributes equal total gradient (1.0), regardless of
#           pixel count. CE.2 used inv_sqrt_npos — Brazil's 155K-pixel events
#           still dominated 15× more than Colombia's 1K-pixel events.
#
#   Fix 4: STAGE2_STRATIFIED_SPLIT=1
#           Country-stratified 80/20 split. CE.2's random seed=42 placed
#           all Argentina crisis-year events in the test set.
#         STAGE2_PATIENCE=200
#           CE.2 stopped at iter 66 because val set was full of politically-
#           driven events. More patience lets training find deeper signal.
#
#   Fix 1: STAGE2_COHERENCE_THRESHOLD
#           Set this to the recommended_threshold from CE.3a diagnostic JSON
#           (outputs/south_america/results/ml_models/stage2_event_coherence_diagnostic.json)
#           before submitting. 0.0 = disabled (no filter).
#
# Decision gate: macro Recall@5% >= 50% on coherent-event test set → proceed to CE.4
# If < 40%: investigate WDPA designation_type filter as separate diagnostic.
#
# Prerequisite: CE.3a (stage2_event_coherence.slurm) must complete first to set threshold.

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

# --- CE.3b fixes ---
export STAGE2_GROUP_WEIGHT_MODE=inv_npos      # Fix 2: event-level gradient norm
export STAGE2_STRATIFIED_SPLIT=1              # Fix 4: country-stratified split
export STAGE2_PATIENCE=200                    # Fix 4: was 100

# Fix 1: dynamically read recommended_threshold from CE.3a output JSON.
# Falls back to 0.0 (filter disabled) if CE.3a has not run yet.
COHERENCE_JSON="$HOME/master_thesis/outputs/south_america/results/ml_models/stage2_event_coherence_diagnostic.json"
if [ -f "$COHERENCE_JSON" ]; then
    STAGE2_COHERENCE_THRESHOLD=$(python3 -c "import json; d=json.load(open('$COHERENCE_JSON')); print(d['metadata']['recommended_threshold'])")
    echo "  coherence threshold from CE.3a: $STAGE2_COHERENCE_THRESHOLD"
else
    STAGE2_COHERENCE_THRESHOLD=0.0
    echo "  CE.3a JSON not found — coherence filter disabled"
fi
export STAGE2_COHERENCE_THRESHOLD

echo "CE.3b: redesigned cross-event Stage 2 — full South America"
echo "  weights=$STAGE2_GROUP_WEIGHT_MODE  patience=$STAGE2_PATIENCE"
echo "  stratified=$STAGE2_STRATIFIED_SPLIT  coherence_threshold=$STAGE2_COHERENCE_THRESHOLD"

srun --cpu-bind=cores --distribution=block:block python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_cross_event.py

echo "Done. Check decision gate: Recall@5% >= 50% on coherent events → proceed to CE.4."
echo "Record result in ROADMAP.md experiment table."
