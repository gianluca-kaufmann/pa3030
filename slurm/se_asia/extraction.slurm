#!/bin/bash
#SBATCH --job-name=se_asia_extraction
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/se_asia_extraction_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/se_asia_extraction_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/data/ml"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/se_asia/tables"

set -euo pipefail

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
# Activate the Python virtual environment (master-thesis)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

#------------------------------------------------------------------------------
# Run stage-1 extraction scripts
#------------------------------------------------------------------------------
echo "Starting SE Asia extraction scripts..."

# --- Preflight checks ---
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Path missing"; exit 1; }
echo "PWD = $(pwd)"

if [ -z "${EE_PROJECT_ID:-}" ]; then
  echo "ERROR: EE_PROJECT_ID is not set."
  echo "Set it before submission, e.g. export EE_PROJECT_ID=<your-project-id>"
  exit 1
fi

EXTRACTION_DIR="scripts/regions/se_asia/1_extraction"
SCRIPTS=(
  "backbone_export"
  "WDPA_export"
  "elevation_export"
  "landcover_export"
  "NDVI_export"
  "WorldClim_export"
  "deforestation_export"
  "wildfire_export"
  "powerplants_export"
  "oil_export"
  "GPW_export"
)

[ -d "$EXTRACTION_DIR" ] || {
  echo "ERROR: extraction directory not found: $EXTRACTION_DIR"
  exit 1
}

for script_name in "${SCRIPTS[@]}"; do
  script_path="$EXTRACTION_DIR/$script_name"
  [ -f "$script_path" ] || {
    echo "ERROR: missing extraction script: $script_path"
    exit 1
  }

  echo "------------------------------------------------------------"
  echo "Running: $script_path (EE_PROJECT_ID=${EE_PROJECT_ID})"
  python "$script_path"
done

echo "SE Asia extraction scripts finished."
