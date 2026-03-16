#!/bin/bash
# submit_spatial_cv.sh
#
# Run this on the Euler login node from the repo root:
#   bash slurm/submit_spatial_cv.sh
#
# What it does:
#   1. Queries the biome count for SA and USA via --list-biomes
#   2. Submits all spatial CV jobs with the correct --array range
#
# Aggregate scripts are NOT submitted here — run them manually once
# all LOBO array tasks have finished.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # always run from repo root

# --- activate venv (same as SLURM scripts) ---
source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

# helper: count biomes for a region's spatial_CV_1 script
n_biomes() {
    local script="$1"
    local count
    count=$(python -u "$script" --list-biomes 2>/dev/null | grep -c '^\s*\[')
    if [[ "$count" -eq 0 ]]; then
        echo "ERROR: --list-biomes returned 0 biomes for $script" >&2
        echo "  Check that the GSN raster exists and the venv is active." >&2
        exit 1
    fi
    echo "$count"
}

echo "======================================================"
echo " Querying biome counts"
echo "======================================================"

SA_N=$(n_biomes  "scripts/regions/south_america/6_evaluation/spatial_CV_1")
USA_N=$(n_biomes "scripts/regions/usa/6_evaluation/spatial_CV_1")
SA_MAX=$(( SA_N  - 1 ))
USA_MAX=$(( USA_N - 1 ))

echo "  South America : $SA_N biomes  → --array=0-$SA_MAX"
echo "  USA           : $USA_N biomes  → --array=0-$USA_MAX"

echo ""
echo "======================================================"
echo " Layer 2 — biome-stratified metrics (fast, no retraining)"
echo "======================================================"
sbatch slurm/south_america/spatial_CV_2.slurm
sbatch slurm/usa/spatial_CV_2.slurm

echo ""
echo "======================================================"
echo " Layer 3 — cross-continental transfer"
echo "======================================================"
sbatch slurm/south_america/spatial_CV_3.slurm

echo ""
echo "======================================================"
echo " Layer 1 LOBO — South America (LGBM + RF)"
echo "======================================================"
sbatch --array=0-${SA_MAX} slurm/south_america/spatial_CV_1.slurm
sbatch --array=0-${SA_MAX} slurm/south_america/spatial_CV_1_rf.slurm

echo ""
echo "======================================================"
echo " Layer 1 LOBO — USA (LGBM + RF)"
echo "======================================================"
sbatch --array=0-${USA_MAX} slurm/usa/spatial_CV_1.slurm
sbatch --array=0-${USA_MAX} slurm/usa/spatial_CV_1_rf.slurm

echo ""
echo "======================================================"
echo " All jobs submitted."
echo " Once ALL LOBO folds complete, run the aggregate scripts:"
echo "   sbatch slurm/south_america/spatial_CV_1_aggregate.slurm"
echo "   sbatch slurm/south_america/spatial_CV_1_aggregate_rf.slurm"
echo "   sbatch slurm/usa/spatial_CV_1_aggregate.slurm"
echo "   sbatch slurm/usa/spatial_CV_1_aggregate_rf.slurm"
echo "======================================================"
