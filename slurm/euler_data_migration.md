# Euler Data Migration Guide

**Purpose:** Restructure `$SCRATCH/data/` from a flat layout to a geography-first layout.

**When to run:** In a single SSH session on Euler. After running these commands,
pull the updated scripts (with new data path references) and test a job.

---

## Pre-flight: Check what you have

Run these first to understand what's actually on disk. The migration commands
below are based on what the scripts reference, but your actual files may differ.

```bash
cd $SCRATCH

echo "=== data/ready/ contents ==="
ls data/ready/

echo "=== data/ready/policy/ contents ==="
ls data/ready/policy/

echo "=== data/ml/ top-level files ==="
ls data/ml/*.parquet 2>/dev/null
ls data/ml/*.csv 2>/dev/null

echo "=== data/ml/ subdirectories ==="
ls -d data/ml/*/

echo "=== data/ml/main/ contents ==="
ls data/ml/main/

echo "=== data/ml/robustness/ contents ==="
ls data/ml/robustness/

echo "=== data/ml/models/ contents (recursive) ==="
find data/ml/models -type f 2>/dev/null

echo "=== data/ml/tuning/ contents ==="
ls data/ml/tuning/

echo "=== data/ml/embeddings_aligned/ ==="
ls data/ml/embeddings_aligned/ 2>/dev/null

echo "=== data/ml/wdpa_aligned/ ==="
ls data/ml/wdpa_aligned/ 2>/dev/null
```

Review the output carefully. If you see files not covered by the move commands
below, decide where they belong before proceeding.

---

## Phase 1: Create the new directory structure

```bash
cd $SCRATCH

# Shared (flat -- policy files go directly here, no nested policy/ folder)
mkdir -p data/shared

# South America: preprocessed features
mkdir -p data/south_america/ready

# South America: ML data
mkdir -p data/south_america/ml/main
mkdir -p data/south_america/ml/robustness
mkdir -p data/south_america/ml/models/main
mkdir -p data/south_america/ml/models/robustness
mkdir -p data/south_america/ml/tuning

# Colombia: ML data
mkdir -p data/south_america/colombia/ml/main
mkdir -p data/south_america/colombia/ml/robustness
mkdir -p data/south_america/colombia/ml/models/main
mkdir -p data/south_america/colombia/ml/models/robustness
mkdir -p data/south_america/colombia/ml/tuning

# Embeddings: preprocessed + ML data
mkdir -p data/south_america/embeddings/ready
mkdir -p data/south_america/embeddings/ml/models
mkdir -p data/south_america/embeddings/ml/tuning
```

---

## Phase 2: Move files

### 2a. Shared policy data

Policy files go FLAT into `data/shared/` (no nested `policy/` subfolder).

```bash
cd $SCRATCH

mv data/ready/policy/* data/shared/
rmdir data/ready/policy 2>/dev/null
```

**Verify:**
```bash
ls data/shared/
# Expected: country_iso3.tif, country_iso3_mapping.json, V-Dem-CY-Core-v15.csv,
#           wgidataset_with_sourcedata.xlsx, DPI2020.csv, etc.
```

### 2b. South America preprocessed features

Move all remaining feature directories from `data/ready/` into `data/south_america/ready/`.
The `embeddings/` directory is handled separately in step 2e.

```bash
cd $SCRATCH

for dir in data/ready/*/; do
  dirname=$(basename "$dir")
  # Skip embeddings -- handled separately in step 2e
  if [ "$dirname" = "embeddings" ]; then
    echo "Skipping: $dir (handled in step 2e)"
    continue
  fi
  echo "Moving: $dir -> data/south_america/ready/$dirname"
  mv "$dir" data/south_america/ready/
done

# Also move any loose files sitting directly in data/ready/
find data/ready -maxdepth 1 -type f -exec mv -t data/south_america/ready/ {} + 2>/dev/null
```

**Verify:**
```bash
ls data/south_america/ready/
# Expected: backbone/, WDPA/, critical_assets/, and other feature directories
```

### 2c. South America ML data (continental)

```bash
cd $SCRATCH

# ---- Merged panels ----
mv data/ml/merged_panel_2000_2024.parquet data/south_america/ml/
mv data/ml/merged_panel_final.parquet data/south_america/ml/

# ---- Main split ----
mv data/ml/main/train_win5.parquet data/south_america/ml/main/
mv data/ml/main/earlystop_win5.parquet data/south_america/ml/main/
mv data/ml/main/test_win5.parquet data/south_america/ml/main/
mv data/ml/main/merged_panel_final_win5.parquet data/south_america/ml/main/

# Also move any SA metadata JSON files in main/
mv data/ml/main/train_win5_metadata.json data/south_america/ml/main/ 2>/dev/null
mv data/ml/main/earlystop_win5_metadata.json data/south_america/ml/main/ 2>/dev/null
mv data/ml/main/test_win5_metadata.json data/south_america/ml/main/ 2>/dev/null

# ---- Robustness split ----
mv data/ml/robustness/train_win5.parquet data/south_america/ml/robustness/ 2>/dev/null
mv data/ml/robustness/earlystop_win5.parquet data/south_america/ml/robustness/ 2>/dev/null
mv data/ml/robustness/test_win5.parquet data/south_america/ml/robustness/ 2>/dev/null

mv data/ml/robustness/train_win5_metadata.json data/south_america/ml/robustness/ 2>/dev/null
mv data/ml/robustness/earlystop_win5_metadata.json data/south_america/ml/robustness/ 2>/dev/null
mv data/ml/robustness/test_win5_metadata.json data/south_america/ml/robustness/ 2>/dev/null

# ---- Models (all model1_* files, any extension) ----
mv data/ml/models/main/model1_* data/south_america/ml/models/main/ 2>/dev/null
mv data/ml/models/robustness/model1_* data/south_america/ml/models/robustness/ 2>/dev/null

# ---- Tuning ----
mv data/ml/tuning/model1_* data/south_america/ml/tuning/ 2>/dev/null
```

**Verify:**
```bash
ls data/south_america/ml/
ls data/south_america/ml/main/
ls data/south_america/ml/models/main/
```

### 2d. Colombia ML data

```bash
cd $SCRATCH

# ---- Merged panel ----
mv data/ml/merged_panel_colombia_final.parquet data/south_america/colombia/ml/

# ---- Main split (all files with "colombia" in the name) ----
mv data/ml/main/*colombia* data/south_america/colombia/ml/main/ 2>/dev/null

# ---- Robustness split ----
mv data/ml/robustness/*colombia* data/south_america/colombia/ml/robustness/ 2>/dev/null

# ---- Models (all modelC_* files, any extension) ----
mv data/ml/models/main/modelC_* data/south_america/colombia/ml/models/main/ 2>/dev/null
mv data/ml/models/robustness/modelC_* data/south_america/colombia/ml/models/robustness/ 2>/dev/null

# ---- Tuning (modelC_* and *colombia* patterns) ----
mv data/ml/tuning/modelC_* data/south_america/colombia/ml/tuning/ 2>/dev/null
mv data/ml/tuning/*colombia* data/south_america/colombia/ml/tuning/ 2>/dev/null
```

**Verify:**
```bash
ls data/south_america/colombia/ml/
ls data/south_america/colombia/ml/main/
ls data/south_america/colombia/ml/models/main/
```

### 2e. Embeddings data

```bash
cd $SCRATCH

# ---- Preprocessed rasters ----
mv data/ml/embeddings_aligned data/south_america/embeddings/ready/
mv data/ml/wdpa_aligned data/south_america/embeddings/ready/

# Raw embedding tiles (from data/ready/embeddings/)
mv data/ready/embeddings data/south_america/embeddings/ready/raw_tiles 2>/dev/null

# ---- Panels ----
mv data/ml/embeddings_transition_panel_2018-2024.parquet data/south_america/embeddings/ml/
mv data/ml/SatelliteEmbeddings_SA_1km_*.parquet data/south_america/embeddings/ml/ 2>/dev/null

# ---- Models (all modelE_* files, any extension; check both root and main/) ----
mv data/ml/models/modelE_* data/south_america/embeddings/ml/models/ 2>/dev/null
mv data/ml/models/main/modelE_* data/south_america/embeddings/ml/models/ 2>/dev/null

# ---- Tuning ----
mv data/ml/tuning/*embeddings* data/south_america/embeddings/ml/tuning/ 2>/dev/null
mv data/ml/tuning/*modelE* data/south_america/embeddings/ml/tuning/ 2>/dev/null
```

**Verify:**
```bash
ls data/south_america/embeddings/ready/
ls data/south_america/embeddings/ml/
```

---

## Phase 3: Safety check and cleanup

### 3a. Check for leftover files

```bash
cd $SCRATCH

echo "=== Files remaining in data/ml/ ==="
find data/ml -type f 2>/dev/null | head -50

echo ""
echo "=== Files remaining in data/ready/ ==="
find data/ready -type f 2>/dev/null | head -50
```

**If files remain:** Inspect them and move manually to the correct location.
Do NOT proceed to cleanup until the above commands show **zero files**.

### 3b. Remove empty directories

Only run this after confirming no files remain.

```bash
cd $SCRATCH

# Recursively remove empty directories (deepest first)
find data/ml -depth -type d -empty -delete 2>/dev/null
find data/ready -depth -type d -empty -delete 2>/dev/null

# Verify they are gone
ls data/ml 2>/dev/null && echo "WARNING: data/ml/ still exists -- check for leftover files" || echo "OK: data/ml/ removed"
ls data/ready 2>/dev/null && echo "WARNING: data/ready/ still exists -- check for leftover files" || echo "OK: data/ready/ removed"
```

---

## Phase 4: Final verification

```bash
cd $SCRATCH

echo "=== New data structure ==="
echo ""
echo "--- data/shared/ ---"
ls data/shared/

echo ""
echo "--- data/south_america/ready/ ---"
ls data/south_america/ready/

echo ""
echo "--- data/south_america/ml/ ---"
ls data/south_america/ml/
echo "  main/:"
ls data/south_america/ml/main/
echo "  robustness/:"
ls data/south_america/ml/robustness/
echo "  models/main/:"
ls data/south_america/ml/models/main/
echo "  models/robustness/:"
ls data/south_america/ml/models/robustness/

echo ""
echo "--- data/south_america/colombia/ml/ ---"
ls data/south_america/colombia/ml/
echo "  main/:"
ls data/south_america/colombia/ml/main/
echo "  models/main/:"
ls data/south_america/colombia/ml/models/main/

echo ""
echo "--- data/south_america/embeddings/ ---"
echo "  ready/:"
ls data/south_america/embeddings/ready/
echo "  ml/:"
ls data/south_america/embeddings/ml/

echo ""
echo "=== DONE ==="
```

---

## What to do next

After completing the Euler migration:

1. **Update script data paths** -- The Python scripts still reference `data/ml/` and
   `data/ready/`. These need to be updated to point to the new geography-based
   paths. This will be done via a code update (ask the assistant to run it).

2. **Git pull on Euler** -- Once the script path updates are committed, pull
   them on Euler: `cd ~/master_thesis && git pull`

3. **Test a job** -- Submit a simple job (e.g., `inspect.slurm`) to verify
   the pipeline can find its data files.

4. **GCS restructure** -- Separately, replicate this same structure on GCS
   at `gs://protected-areas/data/`. This is done manually via gsutil.
