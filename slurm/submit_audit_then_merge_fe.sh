#!/bin/bash
# Submit full pipeline:
#   1. WDPA Issue F audit + fix (all regions)
#   2. SA merge → SA FE → SEA merge → SEA FE → USA merge → USA FE
#
# All jobs are sequential (--dependency=afterok).
# The audit+fix patches USA 2022/2023/2024 TIFs from the May2026 shapefile
# before the merge reads them.
#
# Usage:
#   bash slurm/submit_audit_then_merge_fe.sh

set -euo pipefail

echo "=== Submitting WDPA audit + merge + FE pipeline ==="

# 1) WDPA audit + fix (patches USA TIFs in-place)
JOB_AUDIT=$(sbatch --parsable slurm/wdpa_audit_fix.slurm)
echo "  WDPA audit+fix: job $JOB_AUDIT"

# 2) SA merge (after audit)
JOB_SA_MERGE=$(sbatch --parsable --dependency=afterok:${JOB_AUDIT} slurm/south_america/merging.slurm)
echo "  SA merge:       job $JOB_SA_MERGE (after $JOB_AUDIT)"

# 3) SA FE
JOB_SA_FE=$(sbatch --parsable --dependency=afterok:${JOB_SA_MERGE} slurm/south_america/feature_engineering.slurm)
echo "  SA FE:          job $JOB_SA_FE (after $JOB_SA_MERGE)"

# 4) SEA merge
JOB_SEA_MERGE=$(sbatch --parsable --dependency=afterok:${JOB_SA_FE} slurm/se_asia/merging.slurm)
echo "  SEA merge:      job $JOB_SEA_MERGE (after $JOB_SA_FE)"

# 5) SEA FE
JOB_SEA_FE=$(sbatch --parsable --dependency=afterok:${JOB_SEA_MERGE} slurm/se_asia/feature_engineering.slurm)
echo "  SEA FE:         job $JOB_SEA_FE (after $JOB_SEA_MERGE)"

# 6) USA merge
JOB_USA_MERGE=$(sbatch --parsable --dependency=afterok:${JOB_SEA_FE} slurm/usa/merging.slurm)
echo "  USA merge:      job $JOB_USA_MERGE (after $JOB_SEA_FE)"

# 7) USA FE
JOB_USA_FE=$(sbatch --parsable --dependency=afterok:${JOB_USA_MERGE} slurm/usa/feature_engineering.slurm)
echo "  USA FE:         job $JOB_USA_FE (after $JOB_USA_MERGE)"

echo ""
echo "=== All jobs queued ==="
echo "  Chain: $JOB_AUDIT → $JOB_SA_MERGE → $JOB_SA_FE → $JOB_SEA_MERGE → $JOB_SEA_FE → $JOB_USA_MERGE → $JOB_USA_FE"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    \$SCRATCH/logs/"
echo ""
echo "After all complete, run:"
echo "  python scripts/regions/shared/audit_wdpa_status.py --mode fast --region all"
echo "  to verify 2023/2024 positive rates in the rebuilt panels."
