#!/bin/bash
#SBATCH --job-name=gap_analysis
#SBATCH --output=logs/gap_analysis_%j.out
#SBATCH --error=logs/gap_analysis_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute

# E1 (Track A): Representation gap analysis
# Scores the full SA test parquet with the locked temporal model and
# quantifies the biodiversity-cost gap (score vs. GSN_b2).
#
# Submit after CE.3b-sqrt and inject chain finish, or independently.
# Model: temporal H6+H1b+H5 (Lift@1%=6.46x, iter=136)

set -euo pipefail

REPO="$HOME/master_thesis"
DATA="$SCRATCH/data/south_america/ml/main"
MODEL="$REPO/data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl"

cd "$REPO"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate pa3030

# Verify model file exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL"
    echo "Check path in ROADMAP.md — may be stored in outputs/south_america/results/ml_models/"
    exit 1
fi

mkdir -p logs

echo "=== E1 Gap Analysis ==="
echo "Model:       $MODEL"
echo "Test split:  $DATA/test.parquet"
echo "Started:     $(date)"

python scripts/regions/south_america/6_evaluation/stage2_gap_analysis.py \
    --model  "$MODEL" \
    --test   "$DATA/test.parquet" \
    --extra  "$DATA/earlystop.parquet" "$DATA/train.parquet" \
    --out    "outputs/south_america/results/gap_analysis/"

echo "Finished: $(date)"
echo "Results: outputs/south_america/results/gap_analysis/gap_analysis_results.json"
