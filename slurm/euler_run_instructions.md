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
nano scripts/regions/south_america/3_merging/merge

# Submit job
sbatch slurm/RUN.slurm
squeue
```

---

## File Transfers

> **Legend**
> - `scp` = secure copy over SSH (Desktop ↔ Euler)
> - `gsutil cp` = single file/folder copy (any ↔ GCS)
> - `gsutil rsync` = mirror a whole directory (any ↔ GCS)
> - Add `-m` for parallel (multi-threaded) transfers
> - Add `-r` for recursive (entire folder)

### 1. Euler → Desktop

```bash
# Single file (from scratch)
scp gikaufmann@login.euler.ethz.ch:/cluster/scratch/gikaufmann/<PATH> ~/Desktop/

# Single file (from home repo)
scp gikaufmann@euler.ethz.ch:~/master_thesis/<PATH> ~/Desktop/

# Whole folder
scp -r gikaufmann@login.euler.ethz.ch:/cluster/scratch/gikaufmann/outputs/<FOLDER> ~/Desktop/
```

### 2. Desktop → Euler

```bash
# Single file (to scratch)
scp ~/Desktop/<FILE> gikaufmann@login.euler.ethz.ch:/cluster/scratch/gikaufmann/<PATH>

# Single file (to home repo)
scp ~/Desktop/<FILE> gikaufmann@euler.ethz.ch:~/master_thesis/<PATH>

# Whole folder
scp -r ~/Desktop/<FOLDER> gikaufmann@login.euler.ethz.ch:/cluster/scratch/gikaufmann/<PATH>
```

### 3. GCS → Euler

```bash
# Single file
gsutil cp gs://protected-areas/<PATH> /cluster/scratch/gikaufmann/<PATH>

# Whole folder
gsutil -m cp -r gs://protected-areas/<FOLDER> /cluster/scratch/gikaufmann/<PATH>

# Mirror a directory (skip already-synced files)
gsutil -m rsync -r gs://protected-areas/data /cluster/scratch/gikaufmann/data_v2
```

### 4. Euler → GCS

```bash
# Single file
gsutil cp /cluster/scratch/gikaufmann/<PATH> gs://protected-areas/<PATH>

# Mirror scratch outputs
gsutil -m rsync -r /cluster/scratch/gikaufmann/outputs gs://protected-areas/data/outputs

# Mirror home-repo outputs (txt, json, figures)
cd ~/master_thesis
gsutil -m rsync -r outputs gs://protected-areas/outputs
```

### 5. GCS → Desktop

```bash
# Single file
gsutil cp gs://protected-areas/<PATH> ~/Desktop/

# Whole folder
gsutil -m cp -r gs://protected-areas/<FOLDER> ~/Desktop/
```

### 6. Desktop → GCS

```bash
# Single file
gsutil cp ~/Desktop/<FILE> gs://protected-areas/<PATH>

# Whole folder
gsutil -m cp -r ~/Desktop/<FOLDER> gs://protected-areas/<PATH>
```

---

### Job Troubleshooting

```bash
# Find the log files for successful and failed runs
ls -lt $SCRATCH/logs | head

# Inspect the .out file (main stdout)
less $SCRATCH/logs/<LOGFILE>.out
# Inspect the .err file
less $SCRATCH/logs/<LOGFILE>.err
```

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