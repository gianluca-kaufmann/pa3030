#!/bin/bash
#SBATCH --job-name=select_lobo_biomes
#SBATCH --partition=normal.4h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/select_lobo_biomes_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/select_lobo_biomes_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Biome subset selection for LOBO spatial CV — both regions
#
# Samples 10M training rows per region, computes per-biome mean feature vectors,
# and selects the N most ecologically diverse biomes via greedy farthest-point
# selection.  Writes results to:
#   slurm/south_america/lobo_biome_array.txt
#   slurm/usa/lobo_biome_array.txt
#
# Usage:
#   sbatch slurm/select_lobo_biomes.slurm
#
# After this job completes, submit LOBO folds with:
#   sbatch --array=$(cat slurm/south_america/lobo_biome_array.txt) \
#       slurm/south_america/spatial_CV_1.slurm
#   sbatch --array=$(cat slurm/south_america/lobo_biome_array.txt) \
#       slurm/south_america/spatial_CV_1_rf.slurm
#   sbatch --array=$(cat slurm/usa/lobo_biome_array.txt) \
#       slurm/usa/spatial_CV_1.slurm
#   sbatch --array=$(cat slurm/usa/lobo_biome_array.txt) \
#       slurm/usa/spatial_CV_1_rf.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"

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
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

#------------------------------------------------------------------------------
# Preflight checks
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/shared/evaluation/select_lobo_biomes.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

#------------------------------------------------------------------------------
# Run biome selection for both regions
#------------------------------------------------------------------------------
echo "Selecting LOBO biomes — South America (10M sample rows)..."
python "$SCRIPT" --region south_america --n-biomes 5 --sample-rows 10000000

echo ""
echo "Selecting LOBO biomes — USA (10M sample rows)..."
python "$SCRIPT" --region usa --n-biomes 6 --sample-rows 10000000

echo ""
echo "Biome selection complete."
echo "  SA array:  $(cat slurm/south_america/lobo_biome_array.txt)"
echo "  USA array: $(cat slurm/usa/lobo_biome_array.txt)"
