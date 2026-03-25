#!/bin/bash
#SBATCH --job-name=forward_deploy
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/forward_deploy_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_deploy_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Forward Prediction Pipeline — Stages 0b + 1 + 2
#
# Stage 0b: Train deployment model on 2001-2019 (+ fit calibrators)
# Stage 1:  Extract 2024 inference features from merged_panel_final.parquet
# Stage 2:  Run calibrated batch inference → forward_scored_2024.parquet
#
# Memory budget: ~128 GB  (32 CPUs × 4 GB each; deployment trains on 362M rows)
#
# Run AFTER:
#   - model3_LGBM (evaluation model) has completed and produced split parquets
#   - merged_panel_final.parquet contains year=2024 rows
#
# Stage 0c (backtesting) runs separately — see forward_backtest.slurm
# Stage 0a (coverage baseline) runs on its own: forward_coverage_baseline.py
# Stage 3  (results) runs after this job — see results_lgbm.slurm or run locally
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/se_asia/results/forward/lgbm"
mkdir -p "$SCRATCH/data/se_asia/ml/models"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

#------------------------------------------------------------------------------
# Activate virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MAX_NEG_TRAIN=100000000

#------------------------------------------------------------------------------
# Preflight checks
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

for SCRIPT in \
    "scripts/regions/se_asia/5_training/model3_LGBM_deployment.py" \
    "scripts/regions/se_asia/8_forward/2_forward_features.py" \
    "scripts/regions/se_asia/8_forward/3_forward_predict.py"; do
  [ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }
done

#------------------------------------------------------------------------------
# Stage 0b: Train deployment model (2001-2019) + fit calibrators
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 0b: Training deployment model (2001-2019)..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u scripts/regions/se_asia/5_training/model3_LGBM_deployment.py
STAGE0B_EXIT=$?
if [ "$STAGE0B_EXIT" -ne 0 ]; then
  echo "ERROR: Stage 0b (deployment model training) failed (exit code $STAGE0B_EXIT). Aborting."
  exit "$STAGE0B_EXIT"
fi
echo "Stage 0b complete."

#------------------------------------------------------------------------------
# Stage 1: Extract 2024 inference features
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 1: Extracting 2024 inference features..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u scripts/regions/se_asia/8_forward/2_forward_features.py
echo "Stage 1 complete."

#------------------------------------------------------------------------------
# Stage 2: Calibrated forward inference → forward_scored_2024.parquet
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 2: Running calibrated forward inference..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u scripts/regions/se_asia/8_forward/3_forward_predict.py
echo "Stage 2 complete."

echo ""
echo "forward_deployment job finished. Next steps:"
echo "  1. Run 1_forward_coverage_baseline.py (if not already done)"
echo "  2. Run 4_forward_results.py for maps and scenario analysis"
echo "  3. Run forward_backtest.slurm for backtesting validation"
