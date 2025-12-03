# ETH EULER — QUICK START GUIDE (MASTER'S THESIS)

## Daily Workflow

### On Euler (SSH session)

```bash
ssh gikaufmann@login.euler.ethz.ch
cd ~/master_thesis

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
```

### Job troubleshooting

```bash
# See your most recent jobs (today)
sacct -u $USER --starttime today \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,AllocCPUS | tail -n 20

# Inspect a specific job (replace <JOBID>)
job=<JOBID>
sacct -j ${job} --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,AllocCPUS,MaxRSS,MaxVMSize
scontrol show job ${job} | sed -n '1,80p'

# Find the log files for successful and failed runs
ls -lt $SCRATCH/logs | head

# Inspect the .out file (main stdout)
less $SCRATCH/logs/<LOGFILE>.out
# Inspect the .err file
less $SCRATCH/logs/<LOGFILE>.err

# Look at the training logs (STDOUT + STDERR)
#    (adapt the pattern if your TRAIN.slurm uses a different naming scheme)
ls -lt $SCRATCH/logs/TRAIN_*.out | head
ls -lt $SCRATCH/logs/TRAIN_*.err | head
```

### Check Files & Folders on Euler

```bash
# Go to your scratch root
cd /cluster/scratch/gikaufmann
ls

# Check sizes of main folders
du -sh data data_v2 outputs logs

# Inspect dataset
ls data/ready
ls data/ml

# Inspect outputs
ls outputs
ls outputs/Results
ls outputs/Tables

# Find recently modified files (last 1 day)
find /cluster/scratch/gikaufmann -type f -mtime -1 -print

# Check txt/json in repo
cd ~/master_thesis
find outputs -type f

# folder delete
rm -rf ~/master_thesis/wandb/*
```