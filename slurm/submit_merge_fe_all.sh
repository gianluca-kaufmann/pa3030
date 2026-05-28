#!/bin/bash
# Submit merge + feature_engineering for SA, SEA, USA — chained sequentially.
# All 6 jobs queue in order; each starts only after the previous completes.
#
# Usage:
#   bash slurm/submit_merge_fe_all.sh
#
# Output panels:
#   $SCRATCH/data/south_america/ml/merged_panel_final.parquet
#   $SCRATCH/data/se_asia/ml/merged_panel_final.parquet
#   $SCRATCH/data/usa/ml/merged_panel_final.parquet

set -euo pipefail

echo "=== Submitting merge + FE pipeline (SA → SEA → USA, sequential) ==="

# 1) SA merge
JOB_SA_MERGE=$(sbatch --parsable slurm/south_america/merging.slurm)
echo "  SA merge:   job $JOB_SA_MERGE"

# 2) SA FE (after SA merge)
JOB_SA_FE=$(sbatch --parsable --dependency=afterok:${JOB_SA_MERGE} slurm/south_america/feature_engineering.slurm)
echo "  SA FE:      job $JOB_SA_FE (after $JOB_SA_MERGE)"

# 3) SEA merge (after SA FE)
JOB_SEA_MERGE=$(sbatch --parsable --dependency=afterok:${JOB_SA_FE} slurm/se_asia/merging.slurm)
echo "  SEA merge:  job $JOB_SEA_MERGE (after $JOB_SA_FE)"

# 4) SEA FE (after SEA merge)
JOB_SEA_FE=$(sbatch --parsable --dependency=afterok:${JOB_SEA_MERGE} slurm/se_asia/feature_engineering.slurm)
echo "  SEA FE:     job $JOB_SEA_FE (after $JOB_SEA_MERGE)"

# 5) USA merge (after SEA FE)
JOB_USA_MERGE=$(sbatch --parsable --dependency=afterok:${JOB_SEA_FE} slurm/usa/merging.slurm)
echo "  USA merge:  job $JOB_USA_MERGE (after $JOB_SEA_FE)"

# 6) USA FE (after USA merge)
JOB_USA_FE=$(sbatch --parsable --dependency=afterok:${JOB_USA_MERGE} slurm/usa/feature_engineering.slurm)
echo "  USA FE:     job $JOB_USA_FE (after $JOB_USA_MERGE)"

echo ""
echo "=== All jobs queued ==="
echo "  Chain: $JOB_SA_MERGE → $JOB_SA_FE → $JOB_SEA_MERGE → $JOB_SEA_FE → $JOB_USA_MERGE → $JOB_USA_FE"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      \$SCRATCH/logs/"
