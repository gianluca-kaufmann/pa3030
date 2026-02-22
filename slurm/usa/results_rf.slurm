#!/bin/bash
#SBATCH --job-name=results_model2_RF
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/results_model2_RF_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/results_model2_RF_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/usa/tables"

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
# Activate the Python virtual environment (uav-genAI-opti)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Set threading/CPU hints for faster NumPy/PyArrow (use all allocated CPUs)
#------------------------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

#------------------------------------------------------------------------------
# Run the model2_results script with RF model type (main split)
#------------------------------------------------------------------------------
echo "Starting model2_results script (RF mode, main split)..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/regions/usa/4_ml/4_results/model2_results" ] || {
  echo "ERROR: model2_results not found"
  echo "NOTE: model2_results script is in ML/results/ directory"
  find scripts -maxdepth 4 -type f -name "model2_results"
  exit 1
}
# Run the script (main split, RF model type, prefers calibrated predictions if available)
# --allow_test_shap: use test set for SHAP when earlystop/train unavailable
python "scripts/regions/usa/4_ml/4_results/model2_results" --model_type rf --split-version main --allow_test_shap

echo "model2_results script (RF mode, main split) finished."
