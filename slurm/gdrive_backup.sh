#!/bin/bash
#SBATCH --job-name=gdrive_backup
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/home/gikaufmann/rclone_backup_%j.log

rclone copy \
  /cluster/scratch/gikaufmann/data/ \
  gdrive:"Meine Ablage/data/" \
  --transfers=16 \
  --checkers=8 \
  --drive-chunk-size=128M \
  --log-file=/cluster/home/gikaufmann/rclone_backup.log \
  --log-level=INFO
