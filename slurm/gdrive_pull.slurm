#!/bin/bash
#SBATCH --job-name=gdrive_pull
#SBATCH --partition=normal.24h
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/logs/gdrive_pull_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/gdrive_pull_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"

module purge
module load eth_proxy

echo "=== Pulling South America (skipping — already done) ==="

echo "=== Pulling USA ==="
rclone copy googledrive:thesis/data/usa/ready/ \
  /cluster/scratch/gikaufmann/data/usa/ready/ \
  --exclude "policy/**" \
  --transfers=8 --checkers=8 \
  --drive-chunk-size=256M --buffer-size=256M \
  --tpslimit=5 --tpslimit-burst=10 \
  --log-file="$SCRATCH/logs/rclone_pull_usa_${SLURM_JOB_ID}.log" \
  --log-level=INFO
echo "USA done."

echo "Waiting 120s before next region to avoid Drive API quota..."
sleep 120

echo "=== Pulling SE Asia ==="
rclone copy googledrive:thesis/data/se_asia/ready/ \
  /cluster/scratch/gikaufmann/data/se_asia/ready/ \
  --exclude "policy/**" \
  --transfers=8 --checkers=8 \
  --drive-chunk-size=256M --buffer-size=256M \
  --tpslimit=5 --tpslimit-burst=10 \
  --log-file="$SCRATCH/logs/rclone_pull_sea_${SLURM_JOB_ID}.log" \
  --log-level=INFO
echo "SE Asia done."

echo "=== All pulls complete ==="
