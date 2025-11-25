# ETH EULER — QUICK START GUIDE (MASTER'S THESIS)
# Author: Gian-Luca Kaufmann

## Daily Workflow

### On Euler (SSH session)

```bash
ssh gikaufmann@login.euler.ethz.ch
cd ~/master_thesis

# Update code from GitHub
git pull

# Look at the script directly
nano scripts/merging/merge_total_optimized

# Load modules and activate venv
# module purge
# module load stack/2024-06
# module load gcc/12.2.0
# module load python_cuda/3.11.6
# module load eth_proxy
# source ~/venv/master-thesis/bin/activate

# Update dependencies (if needed)
# pip install --upgrade pip wheel
# pip install -r slurm/requirements.txt

# Submit job
sbatch slurm/RUN.slurm
squeue -u $USER
```

### On Mac (download results)

```bash
scp gikaufmann@login.euler.ethz.ch:/cluster/scratch/gikaufmann/outputs/Results/merged_panel_2000_2024.parquet ~/Desktop/
```

## One-Time Setup

Run these once when setting up a new environment:

```bash
# Load modules
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy

# Create virtual environment
python -m venv ~/venv/master-thesis
source ~/venv/master-thesis/bin/activate

# Install dependencies
pip install --upgrade pip wheel
pip install -r slurm/requirements.txt

# Setup scratch directories and symlinks
mkdir -p /cluster/scratch/$USER/data
mkdir -p /cluster/scratch/$USER/outputs
mkdir -p /cluster/scratch/$USER/logs

rm -rf ~/master_thesis/data ~/master_thesis/outputs
ln -s /cluster/scratch/$USER/data ~/master_thesis/data
ln -s /cluster/scratch/$USER/outputs ~/master_thesis/outputs

# Verify symlinks
ls -l ~/master_thesis | grep -E 'data|outputs'

# Optional: Setup Weights & Biases
wandb login <YOUR_API_KEY>
wandb settings set entity gikaufmann
wandb settings set project master_thesis
```

## Sync Data to Euler

Run from your Mac (non-SSH terminal):

```bash
# Keep Mac awake during transfer
caffeinate

# Sync data
rsync -av --progress --exclude='.DS_Store' \
  "/Users/gianluca/Desktop/Master's Thesis/code/data/ready/" \
  gikaufmann@login.euler.ethz.ch:~/master_thesis/data/ready/
```

If transfer fails, rerun the same command.

Verify on Euler:
```bash
cd ~/master_thesis
ls data/ready
du -sh data/ready/*
```

To clear data on Euler (careful - deletes everything):
```bash
rm -rf ~/master_thesis/data/ready/*
rm -rf ~/master_thesis/data/ready/.* 2>/dev/null
```

## Monitor Jobs

```bash
# Show your jobs
squeue -u $USER

# Follow live log output
tail -f /cluster/scratch/$USER/logs/merge_panel_*.out

# Inspect finished jobs
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS

# List result files
ls -lh /cluster/scratch/$USER/outputs/Results/
```

# ============================================================
#  TRAINING JOB TROUBLESHOOTING (SLURM / TRAIN.slurm)
# ============================================================

# See your most recent jobs (today)
sacct -u $USER --starttime today \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,AllocCPUS | tail -n 20

# Inspect a specific job (replace <JOBID>)
job=<JOBID>
sacct -j ${job} --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,AllocCPUS,MaxRSS,MaxVMSize
scontrol show job ${job} | sed -n '1,80p'

# Look at the training logs (STDOUT + STDERR)
#    (adapt the pattern if your TRAIN.slurm uses a different naming scheme)
ls -lt $SCRATCH/logs/TRAIN_*.out | head
ls -lt $SCRATCH/logs/TRAIN_*.err | head

# newest job id from sacct:
job=<JOBID>   # e.g. job=49479981
tail -n 80  $SCRATCH/logs/TRAIN_${job}.out
tail -n 80  $SCRATCH/logs/TRAIN_${job}.err

# Quickly search for typical failure reasons
grep -iE "error|exception|traceback|no such file|not found|OOM|out of memory|Killed" \
  $SCRATCH/logs/TRAIN_${job}.err || echo "No obvious errors in .err"

# If the job is still running, watch live memory usage
squeue -u $USER
sstat -j ${job}.batch --format=AveRSS,MaxRSS,MaxVMSize

# Confirm the training output actually exists (adapt path as needed)
find $SCRATCH/outputs -maxdepth 3 -type f -mmin -60 -printf "%Tc  %p\n" | sort