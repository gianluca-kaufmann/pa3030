#!/bin/bash
#SBATCH --job-name=ws_ce_w_fixed
#SBATCH --partition=normal.4h
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/ws_ce_w_fixed_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/ws_ce_w_fixed_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# CE.W rerun: fixed 500 rounds, no early stopping (patience=9999).
# Diagnoses whether best_iter=28 was artificially deflating performance.
# Uses same seed/split/features as the original CE.W run (4112735).
# watershed_full_sa.parquet must already exist at $SCRATCH/data/south_america/ml/watershed/.

mkdir -p "$SCRATCH/logs"

WATERSHED="$SCRATCH/data/south_america/ml/watershed/watershed_full_sa.parquet"
if [ ! -f "$WATERSHED" ]; then
    echo "ERROR: watershed parquet not found: $WATERSHED" >&2
    exit 1
fi

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "=== CE.W fixed-iteration rerun (500 rounds, no early stop) ==="
echo "Start: $(date)"
echo "Input: $WATERSHED"
echo "CPUs:  $SLURM_CPUS_PER_TASK"

export WS_PARQUET="$WATERSHED"
export WS_SEED=42
export WS_TEST_RATIO=0.20
export WS_VAL_RATIO=0.10
export WS_MIN_POS_PIXELS=200
export WS_NEG_RATIO=20
export WS_NUM_BOOST_ROUND=500
export WS_ES_PATIENCE=9999

srun --cpu-bind=cores python -u \
    scripts/regions/south_america/5_training/model1_LGBM_stage2_watershed.py

echo "Done: $(date)  exit=$?"
