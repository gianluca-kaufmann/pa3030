# Summary of Changes for Euler Cluster Setup

## 🎯 Problems Solved

### 1. ✅ Weights & Biases Authentication
**Before:** Batch jobs failed with "user is not logged in, WANDB_API_KEY not found"  
**After:** Script checks for credentials and provides clear diagnostics

### 2. ✅ WDPA Data Not Found  
**Before:** Script looked in `~/master_thesis/data/` and logged "No WDPA files found"  
**After:** Script correctly uses `$SCRATCH/data/` and shows exactly where it's looking

---

## 📝 Files Created/Modified

### Modified: `merge_2012.slurm`
**Key improvements:**

1. **W&B Authentication (lines 44-65)**
   ```bash
   export WANDB_DIR="$SCRATCH/wandb/merge"
   export WANDB_MODE=online
   # Checks for ~/.netrc and reports status
   # Provides instructions for setting WANDB_API_KEY if needed
   ```

2. **Data Path Diagnostics (lines 76-93)**
   ```bash
   export SCRATCH="${SCRATCH}"  # Explicit export for Python
   # Shows whether directories exist
   # Lists WDPA files found
   ```

3. **Better Structure**
   - Organized into clear sections with headers
   - Follows the pattern from the working `alm_hpo.slurm` template
   - Improved logging and error messages

### Created: `verify_euler_setup.sh`
**Purpose:** Pre-flight checks before submitting jobs

**Checks:**
- ✅ SCRATCH variable is set
- ✅ Data directories exist
- ✅ WDPA files are present (with count and samples)
- ✅ W&B authentication (.netrc or WANDB_API_KEY)
- ✅ Virtual environment exists
- ✅ Required Python packages are installed
- ✅ Project directory structure is correct

**Usage:**
```bash
cd ~/master_thesis
bash verify_euler_setup.sh
```

### Created: `EULER_SETUP_GUIDE.md`
**Purpose:** Comprehensive documentation

**Contents:**
- Detailed explanation of both problems
- Multiple solutions for W&B auth (with security notes)
- Step-by-step data copying instructions
- Troubleshooting guide
- How the Python script detects environments
- Checklist before submitting

### Created: `QUICK_START_EULER.md`
**Purpose:** Fast reference for experienced users

**Contents:**
- TL;DR 3-step process
- Quick diagnostic commands
- Common error messages & fixes
- Expected log output
- Pro tips

---

## 🔄 How the Fix Works

### W&B Authentication Flow
```
┌─────────────────────────────────────────────────┐
│  SLURM Job Starts                               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Set WANDB_DIR=$SCRATCH/wandb/merge             │
│  Set WANDB_MODE=online                          │
│  Set WANDB_ENTITY=gikaufmann                    │
│  Set WANDB_PROJECT=merge                        │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Check for ~/.netrc                             │
│  ├─ Found? ✓ Report success                    │
│  └─ Not found? ⚠️ Suggest WANDB_API_KEY        │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Python script runs                             │
│  wandb.init() uses credentials from:            │
│  1. WANDB_API_KEY (if set), or                  │
│  2. ~/.netrc (if exists)                        │
└─────────────────────────────────────────────────┘
```

### Data Path Resolution Flow
```
┌─────────────────────────────────────────────────┐
│  SLURM Job: export SCRATCH="${SCRATCH}"         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Show diagnostics:                              │
│  - SCRATCH value                                │
│  - Directory existence checks                   │
│  - List WDPA files                              │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Python script (merge_2012_optimized):          │
│  if "SCRATCH" in os.environ:                    │
│      DATA_ROOT = Path(SCRATCH) / "data"         │
│  else:                                          │
│      DATA_ROOT = PROJECT_ROOT / "data"          │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Script looks in:                               │
│  $SCRATCH/data/ready/WDPA/                      │
│  (on Euler)                                     │
│                                                 │
│  ~/Desktop/Master's Thesis/code/data/ready/WDPA/│
│  (on local machine)                             │
└─────────────────────────────────────────────────┘
```

---

## 📋 What You Need to Do

### Before First Run:

1. **Copy data to scratch:**
   ```bash
   rsync -av ~/master_thesis/data/ $SCRATCH/data/
   ```

2. **Authenticate W&B (choose one):**
   ```bash
   # Option A (try first):
   wandb login
   
   # Option B (if A doesn't work in batch):
   # Edit merge_2012.slurm line 54, add your API key
   ```

3. **Verify setup:**
   ```bash
   cd ~/master_thesis
   bash verify_euler_setup.sh
   ```

4. **Submit job:**
   ```bash
   sbatch merge_2012.slurm
   ```

### Every Time You Submit:

1. **Monitor job:**
   ```bash
   squeue -u $USER
   ```

2. **Check progress:**
   ```bash
   tail -f $SCRATCH/logs/merge_2012_*.out
   ```

3. **Check for errors:**
   ```bash
   tail -f $SCRATCH/logs/merge_2012_*.err
   ```

---

## 🎨 Visual Comparison

### Before (Problems):
```
┌──────────────────────────────────────┐
│  SLURM Job                           │
│  ├─ W&B: ❌ Not logged in           │
│  └─ Data: ❌ Looking in wrong place │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Job fails with errors               │
│  ├─ WANDB_API_KEY not found          │
│  └─ No WDPA files found              │
└──────────────────────────────────────┘
```

### After (Fixed):
```
┌──────────────────────────────────────┐
│  SLURM Job                           │
│  ├─ W&B: ✅ Credentials configured  │
│  │   └─ Checks & reports status     │
│  └─ Data: ✅ Using $SCRATCH/data    │
│      └─ Shows diagnostics            │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Python Script                       │
│  ├─ W&B: ✅ Connected successfully  │
│  └─ Data: ✅ Found WDPA files       │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Job runs successfully! 🎉           │
└──────────────────────────────────────┘
```

---

## 🔍 Verification Output Example

When you run `verify_euler_setup.sh`, you should see:

```
========================================================================
🔍 Euler Cluster Setup Verification for merge_2012_optimized
========================================================================

1️⃣  Checking SCRATCH environment variable...
  ✓ SCRATCH = /cluster/scratch/yourusername

2️⃣  Checking data directories...
  ✓ /cluster/scratch/yourusername/data exists
  ✓ /cluster/scratch/yourusername/data/ready exists
    Found 15 dataset directories
  ✓ /cluster/scratch/yourusername/data/ready/WDPA exists
    Found 13 WDPA .tif files
    Sample files:
      /cluster/scratch/yourusername/data/ready/WDPA/WDPA_SA_1km_2012.tif
      /cluster/scratch/yourusername/data/ready/WDPA/WDPA_SA_1km_2013.tif
      /cluster/scratch/yourusername/data/ready/WDPA/WDPA_SA_1km_2014.tif

3️⃣  Checking Weights & Biases authentication...
  ✓ Found /cluster/home/yourusername/.netrc (credentials from 'wandb login')
  ✓ W&B credentials found in .netrc

4️⃣  Checking virtual environment...
  ✓ Virtual environment exists at /cluster/home/yourusername/venv/master-thesis
  ✓ Python executable found
  ✓ Can activate virtual environment
    Checking required packages...
      ✓ wandb (0.15.12)
      ✓ rasterio (1.3.8)
      ✓ numpy (1.24.3)
      ✓ pandas (2.0.3)
      ✓ pyarrow (13.0.0)

5️⃣  Checking project directory...
  ✓ Project directory exists at /cluster/home/yourusername/master_thesis
  ✓ merge_2012_optimized script found

========================================================================
📋 Summary
========================================================================
✅ All checks passed! You're ready to submit the SLURM job.

To submit:
  cd /cluster/home/yourusername/master_thesis
  sbatch merge_2012.slurm
```

---

## 📚 Documentation Structure

```
Your Project
├── merge_2012.slurm              # Updated SLURM script (MODIFIED)
├── verify_euler_setup.sh         # Pre-flight verification (NEW)
├── QUICK_START_EULER.md          # Fast reference (NEW)
├── EULER_SETUP_GUIDE.md          # Detailed guide (NEW)
├── CHANGES_SUMMARY.md            # This file (NEW)
└── scripts/
    └── merging/
        └── merge_2012_optimized  # Your Python script (unchanged)
```

**Start here:** `QUICK_START_EULER.md` (3-step process)  
**Need details?** `EULER_SETUP_GUIDE.md` (full explanations)  
**Before submitting:** Run `verify_euler_setup.sh`  
**Understanding changes:** `CHANGES_SUMMARY.md` (this file)

---

## 🚀 Next Steps

1. Read `QUICK_START_EULER.md` for the TL;DR version
2. Run `verify_euler_setup.sh` on Euler to check your setup
3. If verification passes, submit your job!
4. If issues found, consult `EULER_SETUP_GUIDE.md` for detailed troubleshooting

---

## 🎉 Expected Success

Once everything is set up correctly, your job will:

1. ✅ Connect to Weights & Biases successfully
2. ✅ Find WDPA files in `$SCRATCH/data/ready/WDPA/`
3. ✅ Process all years (2012-2024)
4. ✅ Output merged GeoTIFFs to `$SCRATCH/outputs/Results/merged_tifs/`
5. ✅ Create Parquet panel at `$SCRATCH/outputs/Results/merged_panel_2012_2024.parquet`
6. ✅ Log progress to W&B dashboard

Monitor at: https://wandb.ai/gikaufmann/merge

Good luck! 🍀

