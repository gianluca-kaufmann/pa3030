# Euler Cluster Setup Guide for merge_2012_optimized

This guide explains how to fix the two main issues with running the merge job on ETH Zurich's Euler cluster:

1. ✅ **Weights & Biases authentication in batch jobs**
2. ✅ **Correct data paths using $SCRATCH**

---

## 🔧 Problem 1: W&B Authentication

### Issue
- `wandb login` works interactively
- SLURM batch jobs fail with "user is not logged in, WANDB_API_KEY not found"

### Root Cause
When you run `wandb login` interactively, credentials are stored in `~/.netrc`. However, batch jobs may not always have access to this file, or the SLURM environment may not preserve it properly.

### Solution (Choose ONE)

#### Option A: Use .netrc (Recommended if it works)
The updated SLURM script now:
- Sets `WANDB_DIR="$SCRATCH/wandb/merge"` to use scratch space for W&B logs
- Checks for `~/.netrc` and reports its status in job logs
- Exports `WANDB_MODE=online` explicitly

**Test it first:**
```bash
# On Euler, in an interactive session:
cat ~/.netrc | grep wandb
```

If this shows your W&B credentials, batch jobs should work.

#### Option B: Set WANDB_API_KEY explicitly (Guaranteed to work)
If `.netrc` doesn't work in batch jobs:

1. Get your API key from: https://wandb.ai/authorize

2. Edit `merge_2012.slurm` line 54 and uncomment:
```bash
export WANDB_API_KEY="your-actual-api-key-here"
```

⚠️ **Security Note**: Don't commit files with API keys to git! Add a line to `.gitignore`:
```bash
echo "*.slurm" >> .gitignore  # If you want to ignore all SLURM scripts
```

Or use a separate config file:
```bash
# Create a secure config file (not tracked by git)
echo 'export WANDB_API_KEY="your-key-here"' > ~/.wandb_key.sh
chmod 600 ~/.wandb_key.sh

# In merge_2012.slurm, add:
source ~/.wandb_key.sh
```

---

## 🗂️ Problem 2: WDPA Data Not Found

### Issue
- Script logs: "No WDPA files found"
- Looking in wrong path: `~/master_thesis/data/...` instead of `$SCRATCH/data/...`

### Root Cause
Your data needs to be on `$SCRATCH` (the fast parallel filesystem) for cluster jobs. The Python script correctly checks for the `SCRATCH` environment variable, but:
1. Data may not be copied to `$SCRATCH/data/` yet
2. The SCRATCH variable must be exported properly

### Solution

#### Step 1: Copy data to $SCRATCH
```bash
# On Euler (from login node or interactive session):
# Copy from your project directory to scratch
rsync -av --progress ~/master_thesis/data/ $SCRATCH/data/

# Or if data is on your local machine, upload first:
# (from your local machine)
rsync -av --progress ~/Desktop/Master\'s\ Thesis/code/data/ \
    username@euler.ethz.ch:~/scratch/data/
```

#### Step 2: Verify WDPA files exist
```bash
# Check WDPA files are present
ls -lh $SCRATCH/data/ready/WDPA/*.tif

# Should show files like:
# WDPA_SA_1km_2012.tif
# WDPA_SA_1km_2013.tif
# etc.
```

#### Step 3: Verify SCRATCH variable
The updated SLURM script now:
- Explicitly exports `SCRATCH` variable (line 80)
- Prints diagnostic information showing:
  - Where it's looking for data
  - Whether directories exist
  - List of WDPA files found

This diagnostic output will appear in your job's `.out` file.

---

## 🔍 Pre-Flight Verification

Before submitting your job, run the verification script:

```bash
# On Euler:
cd ~/master_thesis
bash verify_euler_setup.sh
```

This will check:
- ✅ SCRATCH variable is set
- ✅ Data exists in $SCRATCH/data/ready/
- ✅ WDPA files are present
- ✅ W&B authentication is configured
- ✅ Virtual environment exists and has required packages
- ✅ Python scripts are in the right location

---

## 🚀 Submitting the Job

Once verification passes:

```bash
cd ~/master_thesis
sbatch merge_2012.slurm
```

Monitor the job:
```bash
# Check job status
squeue -u $USER

# Watch output in real-time (once job starts)
tail -f $SCRATCH/logs/merge_2012_<JOBID>.out

# Check for errors
tail -f $SCRATCH/logs/merge_2012_<JOBID>.err
```

---

## 🔎 What the Updated Script Does

### Key Changes in `merge_2012.slurm`:

1. **W&B Setup** (lines 44-65):
   - Sets `WANDB_DIR` to use scratch space
   - Checks for `.netrc` and reports status
   - Provides clear instructions for fixing auth issues

2. **Data Path Diagnostics** (lines 82-93):
   - Prints where it's looking for data
   - Shows which directories exist
   - Lists WDPA files found
   - Helps debug path issues quickly

3. **SCRATCH Export** (line 80):
   - Explicitly exports SCRATCH variable
   - Ensures Python script can detect cluster environment

4. **Clear Output** (lines 98-108):
   - Better logging and progress indicators
   - Easier to see what's happening in job logs

---

## 📊 Python Script Behavior

The Python script (`merge_2012_optimized`) automatically detects the environment:

```python
# Lines 56-61 in merge_2012_optimized
if "SCRATCH" in os.environ:
    DATA_ROOT = Path(os.environ["SCRATCH"]) / "data"
    READY_ROOT = DATA_ROOT / "ready"
else:
    DATA_ROOT = PROJECT_ROOT / "data"
    READY_ROOT = DATA_ROOT / "ready"
```

**On Euler (with SCRATCH set):**
- Looks in: `$SCRATCH/data/ready/WDPA/`

**On local machine (no SCRATCH):**
- Looks in: `~/Desktop/Master's Thesis/code/data/ready/WDPA/`

This means the same script works in both environments! 🎉

---

## 🐛 Troubleshooting

### Still getting "No WDPA files found"?

Check the job output file for the diagnostic lines:
```bash
grep "Data Path Diagnostics" $SCRATCH/logs/merge_2012_*.out -A 10
```

If it shows directories don't exist:
```bash
# Manually create and verify
mkdir -p $SCRATCH/data/ready/WDPA
rsync -av ~/master_thesis/data/ready/WDPA/ $SCRATCH/data/ready/WDPA/
ls -lh $SCRATCH/data/ready/WDPA/
```

### Still getting W&B authentication errors?

1. Check the job output:
```bash
grep "W&B Authentication" $SCRATCH/logs/merge_2012_*.out -A 5
```

2. If `.netrc` not found, use Option B (WANDB_API_KEY)

3. Test W&B in an interactive job:
```bash
srun --time=00:10:00 --mem=4G --pty bash
source ~/venv/master-thesis/bin/activate
python -c "import wandb; wandb.init(project='test')"
```

### Permission issues with .netrc?

```bash
# .netrc must have restrictive permissions
chmod 600 ~/.netrc
```

---

## 📝 Summary Checklist

Before submitting your job:

- [ ] Data copied to `$SCRATCH/data/ready/`
- [ ] WDPA files exist: `ls $SCRATCH/data/ready/WDPA/*.tif`
- [ ] W&B authenticated: `wandb login` or API key set
- [ ] Virtual environment exists: `~/venv/master-thesis`
- [ ] Verification script passes: `bash verify_euler_setup.sh`
- [ ] Project directory: `~/master_thesis`

Then submit:
```bash
cd ~/master_thesis
sbatch merge_2012.slurm
```

---

## 💡 Tips

- **First time?** Run a short test job with `--time=00:30:00` and a small subset of data
- **Large jobs**: Monitor memory usage with `sacct -j <JOBID> --format=JobID,MaxRSS`
- **Save logs**: Jobs' `.out` and `.err` files are in `$SCRATCH/logs/`
- **W&B Dashboard**: Monitor progress at https://wandb.ai/gikaufmann/merge

Good luck! 🚀

