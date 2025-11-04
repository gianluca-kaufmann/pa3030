# 🚀 Quick Start: Running merge_2012_optimized on Euler

## TL;DR - 3 Steps to Success

### 1. Copy Data to Scratch
```bash
# On Euler login node:
rsync -av --progress ~/master_thesis/data/ $SCRATCH/data/
```

### 2. Authenticate W&B (choose one)
```bash
# Option A: Use wandb login (try this first)
wandb login

# Option B: If that doesn't work in batch jobs, edit merge_2012.slurm line 54:
# Uncomment and add your API key from https://wandb.ai/authorize
```

### 3. Submit Job
```bash
# Verify setup first:
cd ~/master_thesis
bash verify_euler_setup.sh

# If all checks pass, submit:
sbatch merge_2012.slurm

# Monitor:
squeue -u $USER
tail -f $SCRATCH/logs/merge_2012_*.out
```

---

## What Got Fixed

### ✅ W&B Authentication
- Added explicit `WANDB_DIR` for scratch space
- Checks for `.netrc` and reports status
- Option to use `WANDB_API_KEY` if `.netrc` doesn't work in batch

### ✅ Data Paths
- Explicitly exports `SCRATCH` variable for Python script
- Added diagnostics showing where data is looked for
- Lists WDPA files found (or not found)

---

## Quick Diagnostics

### Check data location:
```bash
ls -lh $SCRATCH/data/ready/WDPA/*.tif
```

### Check W&B auth:
```bash
cat ~/.netrc | grep wandb
```

### Check job status:
```bash
squeue -u $USER
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS
```

### Check job output:
```bash
# Find your latest job
ls -lt $SCRATCH/logs/merge_2012_*.out | head -1

# View it
tail -100 $SCRATCH/logs/merge_2012_*.out
```

---

## 🔗 Full Documentation

See `EULER_SETUP_GUIDE.md` for detailed explanations and troubleshooting.

---

## 📊 Expected Output in Logs

When the job runs successfully, you should see:

```
🔍 W&B Authentication Check:
  ✓ Found ~/.netrc (credentials from 'wandb login')

🔍 Data Path Diagnostics:
  SCRATCH = /cluster/scratch/yourusername
  SCRATCH/data exists: ✓ YES
  SCRATCH/data/ready exists: ✓ YES
  SCRATCH/data/ready/WDPA exists: ✓ YES
  WDPA files found:
    -rw-r--r-- 1 user group 123M Nov  1 12:00 WDPA_SA_1km_2012.tif
    -rw-r--r-- 1 user group 123M Nov  1 12:00 WDPA_SA_1km_2013.tif
    ...

🚀 Starting merge script...
🔄 Initializing Weights & Biases connection...
✅ Weights & Biases connected successfully!
📊 Starting optimized merge process...
```

---

## ❌ Common Error Messages & Fixes

### "No WDPA files found"
**Fix:** Copy data to scratch:
```bash
rsync -av ~/master_thesis/data/ready/WDPA/ $SCRATCH/data/ready/WDPA/
```

### "WANDB_API_KEY not found"
**Fix:** Either run `wandb login` or set API key in SLURM script

### "Could not cd into ~/master_thesis"
**Fix:** Make sure your project is at `~/master_thesis` or update `PROJECT_DIR` in the SLURM script

---

## 💡 Pro Tips

1. **Test with small data first**: Modify script to process only 2012-2013 for quick validation
2. **Monitor memory**: `watch -n 10 'sacct -j JOBID --format=JobID,MaxRSS,Elapsed'`
3. **Save costs**: Use `--partition=normal.4h` if your job finishes faster
4. **Parallel jobs**: Once working, run multiple years in parallel with array jobs

---

**Need help?** Check `EULER_SETUP_GUIDE.md` or run `verify_euler_setup.sh`

