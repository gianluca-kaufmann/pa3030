#!/bin/bash
#SBATCH --job-name=forward_deploy_rf
#SBATCH --partition=normal.120h
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/forward_deploy_rf_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_deploy_rf_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Forward Prediction Pipeline (RF) — Stages 0b + 1 + 2
#
# Stage 0b: Train RF deployment model on 2001-2019 (+ fit calibrators)
# Stage 1:  Extract 2024 inference features from merged_panel_final.parquet
#           (model-agnostic — reuses features from LGBM run if already present)
# Stage 2:  Run calibrated RF batch inference → forward/rf/forward_scored_2024.parquet
#
# Memory budget: ~128 GB  (32 CPUs × 4 GB each; RF is memory-intensive)
#
# Run AFTER:
#   - model1_LGBM (evaluation model) has completed and produced split parquets
#   - merged_panel_final.parquet contains year=2024 rows
#
# Stage 0c (RF backtesting) runs separately — see forward_backtest_rf.slurm
# Stage 0a (coverage baseline) runs on its own: 1_forward_coverage_baseline.py
# Stage 3  (RF results) runs after this job — run 4_forward_results.py --model-type rf
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/south_america/results/forward/rf"
mkdir -p "$SCRATCH/data/south_america/ml/models"

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
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
mkdir -p "$WANDB_DIR"
export PA3030_WANDB=1
# RF training cap: reservoir-sample negatives to keep X matrix within RAM budget.
# SA has ~353 M negatives; 40 M keeps X at ~14 GB (73 features × float32).
export MAX_NEG_TRAIN="${MAX_NEG_TRAIN:-40000000}"

#------------------------------------------------------------------------------
# Preflight checks
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

for SCRIPT in \
    "scripts/regions/south_america/5_training/model1_rf_deployment.py" \
    "scripts/regions/south_america/8_forward/2_forward_features.py" \
    "scripts/regions/south_america/8_forward/3_forward_predict.py"; do
  [ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }
done

#------------------------------------------------------------------------------
# Stage 0b: Train RF deployment model (2001-2019) + fit calibrators
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 0b: Training RF deployment model (2001-2019)..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u scripts/regions/south_america/5_training/model1_rf_deployment.py
STAGE0B_EXIT=$?
if [ "$STAGE0B_EXIT" -ne 0 ]; then
  echo "ERROR: Stage 0b failed (exit code $STAGE0B_EXIT). Aborting."
  exit "$STAGE0B_EXIT"
fi
echo "Stage 0b complete."

#------------------------------------------------------------------------------
# Stage 1: Extract 2024 inference features (model-agnostic)
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 1: Extracting 2024 inference features..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u scripts/regions/south_america/8_forward/2_forward_features.py
echo "Stage 1 complete."

#------------------------------------------------------------------------------
# Stage 2: Calibrated RF forward inference → forward/rf/forward_scored_2024.parquet
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 2: Running calibrated RF forward inference..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u scripts/regions/south_america/8_forward/3_forward_predict.py --model-type rf
echo "Stage 2 complete."

echo ""
echo "forward_deployment_rf job finished. Next steps:"
echo "  1. Run 1_forward_coverage_baseline.py (if not already done)"
echo "  2. Run 4_forward_results.py --model-type rf for maps and scenario analysis"
echo "  3. Run forward_backtest_rf.slurm for RF backtesting validation"
