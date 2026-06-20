#!/bin/bash
#SBATCH --job-name=kba_gap_sa
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=/cluster/scratch/%u/logs/kba_gap_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/kba_gap_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# A2 (E7): KBA representation gap analysis — full SA test parquet.
# Scores all test pixels (2017-2024) with the locked temporal Stage 2 model,
# looks up is_kba at each pixel's (row, col), and computes:
#   - Top-5% KBA coverage rate vs baseline
#   - KBA Recall@5% and lift
#   - Per-event direct gap (designated pixels vs all candidates)
# Outputs: outputs/south_america/results/gap_analysis/kba_gap_analysis.json + plot

mkdir -p "$SCRATCH/logs"

MODEL="$HOME/master_thesis/data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl"
TEST_PARQUET="$SCRATCH/data/south_america/ml/main/test.parquet"
KBA_TIF="$SCRATCH/data/south_america/ready/KBA/kba_sa.tif"
OUT_DIR="outputs/south_america/results/gap_analysis"

for F in "$MODEL" "$TEST_PARQUET" "$KBA_TIF"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: required file not found: $F" >&2
        exit 1
    fi
done

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "=== A2 (E7): KBA Representation Gap Analysis ==="
echo "Model:   $MODEL"
echo "Data:    $TEST_PARQUET"
echo "KBA:     $KBA_TIF"
echo "Out:     $OUT_DIR"
echo "Start:   $(date)"

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/6_evaluation/kba_gap_analysis.py \
    --model "$MODEL" \
    --data  "$TEST_PARQUET" \
    --kba   "$KBA_TIF" \
    --out   "$OUT_DIR" \
    --test-years-only
STATUS=$?

echo "Done: $(date)  exit=$STATUS"
exit $STATUS
