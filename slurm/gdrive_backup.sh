#!/bin/bash
#SBATCH --job-name=gdrive_backup
#SBATCH --partition=normal.24h
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/gdrive_backup_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/gdrive_backup_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Create log folder if needed
mkdir -p "$SCRATCH/logs"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load eth_proxy

#------------------------------------------------------------------------------
# Run rclone backup
#------------------------------------------------------------------------------
echo "Starting rclone backup to Google Drive..."

rclone copy \
  /cluster/scratch/gikaufmann/data/ \
  gdrive:"Meine Ablage/data/" \
  --transfers=8 \
  --checkers=8 \
  --drive-chunk-size=512M \
  --buffer-size=512M \
  --log-file="$SCRATCH/logs/rclone_backup_${SLURM_JOB_ID}.log" \
  --log-level=INFO

echo "rclone backup finished."
