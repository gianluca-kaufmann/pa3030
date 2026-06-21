#!/bin/bash
#SBATCH --job-name=ws_ce_w_sa
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/ws_ce_w_sa_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ws_ce_w_sa_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# B5 (CE.W): Train LambdaRank watershed model on full SA catchment splits.
# Cross-event 80/20 split stratified by country (primary evaluation).
# Custom early stopping on pixel-level Recall@5%.
#
# Prerequisite: watershed_build.slurm must have completed successfully.
#
# Submit as chain:
#   JOB=$(sbatch --parsable slurm/south_america/watershed_build.slurm)
#   sbatch --dependency=afterok:$JOB slurm/south_america/watershed_ce_w.slurm
#
# Or standalone (if watershed_full_sa.parquet already exists):
#   sbatch slurm/south_america/watershed_ce_w.slurm

mkdir -p "$SCRATCH/logs"

WATERSHED="$SCRATCH/data/south_america/ml/watershed/watershed_full_sa.parquet"
if [ ! -f "$WATERSHED" ]; then
    echo "ERROR: watershed parquet not found: $WATERSHED" >&2
    echo "Run watershed_build.slurm first." >&2
    exit 1
fi

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "=== B5 (CE.W): Watershed cross-event training ==="
echo "Start:    $(date)"
echo "Input:    $WATERSHED"
echo "CPUs:     $SLURM_CPUS_PER_TASK"

export WS_PARQUET="$WATERSHED"
export WS_SEED=42
export WS_TEST_RATIO=0.20
export WS_VAL_RATIO=0.10
export WS_MIN_POS_PIXELS=200
export WS_NEG_RATIO=20
export WS_NUM_BOOST_ROUND=1000

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_watershed.py
STATUS=$?

echo "Done: $(date)  exit=$STATUS"
exit $STATUS
