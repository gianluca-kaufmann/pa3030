# ETH EULER — QUICK START GUIDE (MASTER'S THESIS)

## Daily Workflow

### On Euler (SSH session)

```bash
ssh gikaufmann@login.euler.ethz.ch
cd ~/master_thesis

cd /cluster/scratch/gikaufmann
ls

# Update code from GitHub
git pull

# Look at the script directly
nano scripts/merging/merge_total_optimized

# Submit job
sbatch slurm/RUN.slurm
squeue
```

### Sync from Euler to Mac

```bash
# scratch outputs
scp gikaufmann@login.euler.ethz.ch:/cluster/scratch/gikaufmann/outputs/Results/merged_panel_2000_2024.parquet ~/Desktop/

# txt and other outputs
scp -r gikaufmann@euler.ethz.ch:~/master_thesis/outputs/Tables/merged_panel_validation.txt ~/Desktop/

# Download from GCS → Desktop
gsutil -m cp gs://protected-areas/data/ml/merged_panel_final.parquet 

# whole folder
gsutil -m cp -r gs://protected-areas/outputs/Results /Users/gianluca/Desktop/
```
### Sync Between Euler ↔ GCS

```bash
# GCS → Euler (pull full dataset)
gsutil -m rsync -r gs://protected-areas/data \
  /cluster/scratch/gikaufmann/data_v2

# Euler → GCS (push full dataset)
gsutil -m rsync -r /cluster/scratch/gikaufmann/data_v2 \
  gs://protected-areas/data

# Euler → GCS (push outputs)
gsutil -m rsync -r /cluster/scratch/gikaufmann/outputs \
  gs://protected-areas/data/outputs

# Euler → GCS (push txt/json summaries)
cd ~/master_thesis
gsutil -m rsync -r outputs \
  gs://protected-areas/outputs

#For single files
gsutil cp outputs/Figures/similarity_vis_2000.png gs://protected-areas/outputs/Figures/ml/
```

### Job troubleshooting

```bash
# Find the log files for successful and failed runs
ls -lt $SCRATCH/logs | head

# Inspect the .out file (main stdout)
less $SCRATCH/logs/<LOGFILE>.out
# Inspect the .err file
less $SCRATCH/logs/<LOGFILE>.err

### Check Files & Folders on Euler

```bash
# Find recently modified files (last 1 day)
find /cluster/scratch/gikaufmann -type f -mtime -1 -print

# Check txt/json in repo
cd ~/master_thesis
find outputs -type f

# folder contents delete
rm -rf ~/master_thesis/wandb/*

# folder total delete
rm -rf ~/master_thesis/wandb
```