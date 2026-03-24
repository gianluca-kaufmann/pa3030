#!/bin/bash
#SBATCH --job-name=forward_results_usa_rf
#SBATCH --partition=normal.4h
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/forward_results_usa_rf_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/forward_results_usa_rf_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# ------------------------------------------------------------------------------
# Stage 3 (RF): Forward Results — USA
#
# Reads forward/rf/forward_scored_2024.parquet + forward_coverage_baseline.json
# and produces all paper-ready outputs for the RF model:
#   - forward_probability_map.pdf/png
#   - forward_risk_map_bau.pdf
#   - forward_scenario_moderate.pdf / forward_scenario_30x30.pdf
#   - forward_country_breakdown.csv/.tex
#   - forward_biome_breakdown.csv
#   - forward_gap_analysis.pdf
#   - forward_economic_exposure.csv
#   - forward_hotspot_maps.pdf
#   - forward_scenario_summary.json
#
# Run AFTER:
#   - forward_deployment_rf.slurm (USA, Stage 0b + 1 + 2) has completed
#   - 1_forward_coverage_baseline.py (USA) has been run
#
# Usage:
#   sbatch slurm/usa/forward_results_rf.slurm
# ------------------------------------------------------------------------------

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/usa/results/forward/rf"

export PA3030_FORWARD_MODEL_TYPE=rf

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

#------------------------------------------------------------------------------
# Preflight
#------------------------------------------------------------------------------
ls -ld /cluster/home/gikaufmann/master_thesis || { echo "Repo path missing"; exit 1; }
echo "PWD = $(pwd)"

SCRIPT="scripts/regions/usa/8_forward/4_forward_results.py"
[ -f "$SCRIPT" ] || { echo "ERROR: $SCRIPT not found"; exit 1; }

#------------------------------------------------------------------------------
# Stage 3: Generate results (RF)
#------------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "Stage 3: Generating USA forward results (RF)..."
echo "========================================================================="
srun --cpu-bind=cores --distribution=block:block \
  python -u "$SCRIPT"

EXIT_CODE=$?
echo "USA forward_results (RF) finished (exit code: ${EXIT_CODE})"
exit $EXIT_CODE
