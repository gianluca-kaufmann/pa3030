#!/bin/bash
#SBATCH --job-name=e1_gap
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/e1_gap_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/e1_gap_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# E1 (Track A): Representation gap analysis.
# Scores the full SA test parquet (2017-2024) with the locked temporal model
# and quantifies the biodiversity-cost gap (GSN_b2 of designated vs. all pixels).
#
# Independent of CE.3b-sqrt / AGB-REDD chain — submit any time.
# Model: temporal H6+H1b+H5 (Lift@1%=6.46×, iter=136)
# Note: GSN_b2 is a model feature, so Part B (score vs GSN_b2) is partially
#       circular. Part A (direct within-event comparison) is the clean result.

mkdir -p "$SCRATCH/logs"

MODEL="$HOME/master_thesis/data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl"
DATA="$SCRATCH/data/south_america/ml/main"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    exit 1
fi

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "=== E1 Gap Analysis (Track A) ==="
echo "Model: $MODEL"
echo "Test:  $DATA/test.parquet"
echo "Start: $(date)"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/6_evaluation/stage2_gap_analysis.py \
    --model  "$MODEL" \
    --test   "$DATA/test.parquet" \
    --extra  "$DATA/earlystop.parquet" "$DATA/train.parquet" \
    --out    "outputs/south_america/results/gap_analysis/"
STATUS=$?

echo "Done: $(date)  exit=$STATUS"
exit $STATUS
