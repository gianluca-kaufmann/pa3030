#!/bin/bash

#SBATCH --job-name=merge-2012
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/merge_2012_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/merge_2012_%j.err

set -euo pipefail

#------------------------------------------------------------------------------
# Create necessary folders on $SCRATCH
#------------------------------------------------------------------------------
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data"
mkdir -p "$SCRATCH/outputs"
mkdir -p "$SCRATCH/outputs/Results/merged_tifs"
mkdir -p "$SCRATCH/wandb/merge"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy

#------------------------------------------------------------------------------
# Activate virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venv/master-thesis"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Setup project directory
#------------------------------------------------------------------------------
PROJECT_DIR="$HOME/master_thesis"
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd into $PROJECT_DIR"; exit 1; }

#------------------------------------------------------------------------------
# Weights & Biases Authentication
#------------------------------------------------------------------------------
export WANDB_ENTITY="gikaufmann"
export WANDB_PROJECT="merge"
export WANDB_DIR="$SCRATCH/wandb/merge"
export WANDB_MODE=online

# W&B authentication: Check for API key in ~/.netrc (created by 'wandb login')
# If ~/.netrc doesn't work in batch jobs, uncomment and set WANDB_API_KEY:
# export WANDB_API_KEY="your-api-key-here"  # Get from: https://wandb.ai/authorize

echo "🔍 W&B Authentication Check:"
if [ -f "$HOME/.netrc" ]; then
    echo "  ✓ Found ~/.netrc (credentials from 'wandb login')"
else
    echo "  ✗ No ~/.netrc found"
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "  ✗ WANDB_API_KEY not set either"
        echo "  ⚠️  To fix: Either run 'wandb login' or set WANDB_API_KEY in this script"
    fi
fi

#------------------------------------------------------------------------------
# Environment variables for performance tuning
#------------------------------------------------------------------------------
export GDAL_CACHEMAX=512
export RASTERIO_MAX_DATASET_OPEN=32
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

#------------------------------------------------------------------------------
# Critical: Ensure SCRATCH environment variable is set for Python script
#------------------------------------------------------------------------------
# The Python script checks for SCRATCH to determine data paths
export SCRATCH="${SCRATCH}"

echo ""
echo "🔍 Data Path Diagnostics:"
echo "  SCRATCH = $SCRATCH"
echo "  SCRATCH/data exists: $([ -d "$SCRATCH/data" ] && echo "✓ YES" || echo "✗ NO")"
echo "  SCRATCH/data/ready exists: $([ -d "$SCRATCH/data/ready" ] && echo "✓ YES" || echo "✗ NO")"
echo "  SCRATCH/data/ready/WDPA exists: $([ -d "$SCRATCH/data/ready/WDPA" ] && echo "✓ YES" || echo "✗ NO")"

# List WDPA files if directory exists
if [ -d "$SCRATCH/data/ready/WDPA" ]; then
    echo "  WDPA files found:"
    ls -lh "$SCRATCH/data/ready/WDPA/"*.tif 2>/dev/null | head -5 || echo "    (no .tif files)"
fi

#------------------------------------------------------------------------------
# Run the merge script
#------------------------------------------------------------------------------
echo ""
echo "🚀 Starting merge script..."
echo "  Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "  Python: $(python --version)"
echo "  Working directory: $(pwd)"
echo ""

python scripts/merging/merge_2012_optimized

echo ""
echo "Script completed!" 