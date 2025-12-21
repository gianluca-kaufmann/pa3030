#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/model1_preprocessing_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/model1_preprocessing_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/Tables"

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
# Run the model1_preprocessing script
#------------------------------------------------------------------------------
echo "Starting preprocessing script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/ML/training_preprocessing/model1_preprocessing" ] || {
  echo "ERROR: model1_preprocessing not found"
  echo "NOTE: model1_preprocessing script is in training_preprocessing/ directory"
  find scripts -maxdepth 4 -type f -name "model1_preprocessing"
  exit 1
}
# Run the script
python "scripts/ML/training_preprocessing/model1_preprocessing"

echo "model1_preprocessing script finished."

