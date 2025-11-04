# Euler Cluster - Fixed Setup for merge_2012_optimized

## ✅ What's Been Fixed

Your SLURM job now correctly handles:
1. **Weights & Biases authentication** in batch jobs
2. **Data paths** using `$SCRATCH/data/`

## 🚀 Quick Start (3 Steps)

### 1️⃣ Copy Data to Scratch
```bash
# On Euler:
rsync -av --progress ~/master_thesis/data/ $SCRATCH/data/
```

### 2️⃣ Configure W&B Authentication

**Option A** (try this first):
```bash
wandb login
```

**Option B** (if Option A doesn't work in batch jobs):
Edit `merge_2012.slurm` line 54 and uncomment:
```bash
export WANDB_API_KEY="your-api-key-here"  # Get from https://wandb.ai/authorize
```

### 3️⃣ Verify & Submit
```bash
cd ~/master_thesis

# Verify setup (IMPORTANT!)
bash verify_euler_setup.sh

# If all checks pass, submit:
sbatch merge_2012.slurm

# Monitor:
squeue -u $USER
tail -f $SCRATCH/logs/merge_2012_*.out
```

## 📚 Documentation

- **Quick Reference**: [`QUICK_START_EULER.md`](QUICK_START_EULER.md) - TL;DR version
- **Detailed Guide**: [`EULER_SETUP_GUIDE.md`](EULER_SETUP_GUIDE.md) - Full explanations & troubleshooting  
- **What Changed**: [`CHANGES_SUMMARY.md`](CHANGES_SUMMARY.md) - Technical details of fixes
- **Verification Tool**: `verify_euler_setup.sh` - Pre-flight checks

## 🔧 Files Modified/Created

### Modified
- ✏️ `merge_2012.slurm` - Fixed W&B auth and data paths with diagnostics

### Created
- 📝 `verify_euler_setup.sh` - Executable verification script
- 📖 `EULER_SETUP_GUIDE.md` - Comprehensive guide
- ⚡ `QUICK_START_EULER.md` - Fast reference
- 📋 `CHANGES_SUMMARY.md` - Technical details
- 📄 `README_EULER.md` - This file

## 🎯 Key Improvements in merge_2012.slurm

1. **W&B Setup**
   - Sets `WANDB_DIR="$SCRATCH/wandb/merge"`
   - Checks for `~/.netrc` and reports status
   - Provides clear instructions for authentication

2. **Data Path Diagnostics**
   - Explicitly exports `SCRATCH` variable
   - Shows where it's looking for data
   - Lists WDPA files found (or not found)

3. **Better Organization**
   - Clear section headers
   - Improved logging
   - Follows Euler best practices

## 📊 What You'll See When It Works

```bash
🔍 W&B Authentication Check:
  ✓ Found ~/.netrc (credentials from 'wandb login')

🔍 Data Path Diagnostics:
  SCRATCH = /cluster/scratch/yourusername
  SCRATCH/data exists: ✓ YES
  SCRATCH/data/ready exists: ✓ YES
  SCRATCH/data/ready/WDPA exists: ✓ YES
  WDPA files found:
    -rw-r--r-- 1 user group 123M WDPA_SA_1km_2012.tif
    -rw-r--r-- 1 user group 123M WDPA_SA_1km_2013.tif
    ...

🚀 Starting merge script...
🔄 Initializing Weights & Biases connection...
✅ Weights & Biases connected successfully!
📊 Starting optimized merge process...
```

## ❓ Still Having Issues?

### "No WDPA files found"
```bash
# Check if data exists:
ls $SCRATCH/data/ready/WDPA/*.tif

# If not, copy it:
rsync -av ~/master_thesis/data/ready/WDPA/ $SCRATCH/data/ready/WDPA/
```

### "WANDB_API_KEY not found"
```bash
# Check .netrc:
cat ~/.netrc | grep wandb

# If not found, either:
# 1. Run: wandb login
# 2. Or set WANDB_API_KEY in merge_2012.slurm
```

### Verification fails
```bash
# Run verification with verbose output:
bash verify_euler_setup.sh

# Check what's failing and follow the suggestions
```

## 🎓 Understanding the Fix

### Problem 1: W&B Authentication
- **Before**: Credentials from `wandb login` not accessible in batch jobs
- **After**: Script checks for credentials and provides fallback options

### Problem 2: Data Paths  
- **Before**: Script looked in `~/master_thesis/data/` (wrong location)
- **After**: Script uses `$SCRATCH/data/` (fast parallel filesystem)

### How It Works
The Python script (`merge_2012_optimized`) automatically detects the environment:

```python
if "SCRATCH" in os.environ:
    DATA_ROOT = Path(os.environ["SCRATCH"]) / "data"  # On Euler
else:
    DATA_ROOT = PROJECT_ROOT / "data"  # On local machine
```

The SLURM script now:
1. Explicitly exports `SCRATCH` variable
2. Shows diagnostics about data location
3. Configures W&B to use scratch space for logs

## 📞 Support

For detailed troubleshooting, see [`EULER_SETUP_GUIDE.md`](EULER_SETUP_GUIDE.md)

## ✅ Pre-Submission Checklist

Before running `sbatch merge_2012.slurm`:

- [ ] Data copied to `$SCRATCH/data/ready/`
- [ ] WDPA files exist: `ls $SCRATCH/data/ready/WDPA/*.tif` shows files
- [ ] W&B authenticated: `wandb login` or API key set in script
- [ ] Virtual environment exists: `~/venv/master-thesis`
- [ ] Verification passes: `bash verify_euler_setup.sh` shows all green checks
- [ ] In correct directory: `cd ~/master_thesis`

Then submit:
```bash
sbatch merge_2012.slurm
```

## 🎉 Success Indicators

Your job is working correctly when:
- ✅ Job status is `RUNNING` (not failing immediately)
- ✅ Output log shows "✅ Weights & Biases connected successfully!"
- ✅ Log shows "Found X WDPA files" (not "No WDPA files found")
- ✅ You can see progress at https://wandb.ai/gikaufmann/merge
- ✅ GeoTIFF files appear in `$SCRATCH/outputs/Results/merged_tifs/`

---

**Ready to go!** Start with the verification script, then submit your job. Good luck! 🚀

