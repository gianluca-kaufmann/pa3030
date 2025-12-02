#!/bin/bash
#SBATCH --job-name=sample-preprocessing-imbalanced
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --output=/cluster/scratch/%u/logs/sample_preprocessing_imbalanced_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/sample_preprocessing_imbalanced_%j.err
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
# Run the sample preprocessing imbalanced script
#------------------------------------------------------------------------------
echo "Starting sample preprocessing imbalanced script..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

[ -f "scripts/ML/training_preprocessing/sample_preprocessing_imbalanced" ] || {
  echo "ERROR: sample_preprocessing_balanced not found under scripts/ML/training_preprocessing"
  find scripts -maxdepth 4 -type f -name "sample_preprocessing_imbalanced"
  exit 1
}

# Run the script
python "scripts/ML/training_preprocessing/sample_preprocessing_imbalanced"

echo "Sample preprocessing imbalanced script finished."

