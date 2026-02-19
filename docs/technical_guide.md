# PA3030 Technical Guide

A detailed walkthrough of the PA3030 pipeline — how it works, what the results mean,
and how to improve it. Written for supervisors and collaborators who want to
understand the project without reading 1,600-line scripts.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [How LightGBM Works](#2-how-lightgbm-works)
3. [How Features Are Obtained](#3-how-features-are-obtained)
4. [The Full Pipeline: Step by Step](#4-the-full-pipeline-step-by-step)
5. [Understanding the Results](#5-understanding-the-results)
6. [Calibration and Reliability Curves](#6-calibration-and-reliability-curves)
7. [Policy Context: 30x30 and Carbon](#7-policy-context-3030-and-carbon)
8. [Prediction vs. Prescription](#8-prediction-vs-prescription)
9. [Code Quality Assessment](#9-code-quality-assessment)
10. [Improvement Priorities](#10-improvement-priorities)
11. [The 5-Year Window and Forward Projection to 2030](#11-the-5-year-window-and-forward-projection-to-2030)
12. [Missing Variables](#12-missing-variables)
13. [Key File Locations](#13-key-file-locations)

---

## 1. What This Project Does

PA3030 predicts the probability that a given 1 km × 1 km land pixel will become a
newly designated protected area within a 5-year window.

The core question: **"Given historical patterns of protected-area establishment,
which locations are most likely to be designated as protected areas in the future?"**

It answers this by training a gradient-boosted decision tree model (LightGBM) on
~60 geospatial features extracted from satellite imagery, climate data, and
infrastructure maps. The model is trained on historical data (2000-2013), validated
temporally (2014-2016), and tested against what actually happened (2017-2019).

### Key Numbers

| Property | Value |
|----------|-------|
| Spatial resolution | ~1 km × 1 km |
| Temporal coverage | 2000-2024 (annual) |
| South America dataset | ~350 million pixel-years |
| Colombia dataset | ~7 million pixel-years |
| Number of features | ~60 |
| Positive rate (class imbalance) | ~0.3-0.5% |
| Best ROC-AUC | 0.947 (SA Random Forest) |
| Best Precision@1% | 62.8% (Colombia LightGBM robustness) |
| Best Lift@1% | 77x (SA Random Forest) |

---

## 2. How LightGBM Works

### Decision Trees (The Intuition)

A single decision tree is a flowchart of yes/no questions:

```
Is distance to nearest protected area < 5km?
├── YES: Is biodiversity score > 0.7?
│   ├── YES: → 82% chance of protection
│   └── NO:  → 31% chance of protection
└── NO:  Is deforestation rate > 0.3?
    ├── YES: → 12% chance of protection
    └── NO:  → 2% chance of protection
```

The algorithm learns which questions to ask and what thresholds to use by finding
the split that best separates "became protected" from "stayed unprotected."

### Gradient Boosting (Chaining Trees Together)

One tree is weak — it overfits or misses patterns. LightGBM chains hundreds of
trees together, where each new tree focuses on correcting the errors of the
previous ones:

- Tree #1 gets the broad strokes right
- Tree #2 fixes where Tree #1 was wrong
- Tree #300 fine-tunes edge cases

The final prediction is the sum of all trees' contributions. This works well for
tabular/geospatial data because it naturally handles mixed feature types (continuous
elevation + categorical land cover), non-linear relationships, and feature
interactions (e.g., "high biodiversity AND near existing PA" matters more than
either alone).

### Configuration in This Project

- **Objective:** `binary` — outputs a probability (0 to 1) per pixel
- **Boosting:** `gbdt` (gradient boosted decision trees)
- **Key hyperparameters:** learning_rate=0.2, num_leaves=63, max_depth=10
- **Early stopping:** Training halts if validation score doesn't improve for 500 rounds

### Why a Random Seed Is Needed

`RANDOM_STATE = 42` is set because LightGBM has randomness in:

1. **Feature subsampling** (colsample_bynode=0.7) — each split considers 70% of features
2. **Row subsampling** (subsample=0.7) — each tree trains on 70% of rows
3. **Train/validation splits** — data partitioning involves randomness

Without a fixed seed, running the same code twice produces slightly different models.
The seed makes results exactly reproducible — critical for a thesis and paper.

---

## 3. How Features Are Obtained

### Layer 1: Raw Satellite/Geospatial Data (Google Earth Engine)

| Source | What It Captures | Type |
|--------|-----------------|------|
| MODIS NDVI | Vegetation greenness | Annual |
| Hansen deforestation | Forest loss per year (3 bands) | Annual |
| SRTM elevation | Terrain height + slope | Static |
| WorldClim | 19 climate variables (temperature, rainfall) | Static |
| MODIS land cover | Land type (forest, urban, water, etc.) | Annual |
| GPW population | People per km² | 5-year snapshots |
| Nighttime lights (HNTL/VIIRS) | Economic activity proxy | Annual |
| WDPA | Existing protected area boundaries | Annual |
| EDF oil/gas infrastructure | Industrial infrastructure count | Static |
| WRI power plants | Power plant locations + fuel type | Static |
| Global Safety Net (GSN) | Biodiversity importance (5 bands) | Static |
| MODIS burned area | Fire/wildfire history | Annual |

### Layer 2: Distance Transforms (Feature Engineering)

Raw infrastructure/PA presence is converted to "how far is the nearest X?" using
Euclidean distance transforms:

- `dist_wdpa` — distance to nearest existing protected area (most important)
- `dist_road` — distance to nearest road
- `dist_oil_gas` — distance to nearest oil/gas infrastructure
- `dist_powerplant` — distance to nearest power plant

### Layer 3: Spatial Smoothing (Multi-Scale Context)

Key variables are averaged over 16km and 64km neighborhoods using uniform filters.
This captures not just "what's happening at this pixel" but "what's the regional
trend":

- `NDVI_b1_smooth64` — is the broader 64km region green or degraded?
- `deforestation_b1_smooth64` — is the region experiencing widespread deforestation?
- `GPW_b1_smooth16` — how populated is the 16km neighborhood?

Smoothing window sizes: 4, 16, and 64 pixels (≈4km, 16km, 64km at 1km resolution).

---

## 4. The Full Pipeline: Step by Step

### Step 1: Extract Raw Data from Google Earth Engine

**Files:** `scripts/regions/south_america/colombia/preprocessing/colombia_export`
and `scripts/regions/*/1_extraction/*_export`

Google Earth Engine scripts export satellite/geospatial data as GeoTIFF rasters at
1km resolution in EPSG:3857 projection. There's one export script per data source.
Each produces annual or static GeoTIFF files.

### Step 2: Preprocess

**Files:** `scripts/regions/*/2_preprocessing/*`

Format harmonization, reprojection to EPSG:3857, resampling to 1000m:
- Continuous data (elevation, population, climate): bilinear resampling
- Categorical data (land cover): nearest-neighbor resampling

### Step 3: Merge Everything into One Table

**File:** `scripts/regions/south_america/colombia/3_merging/colombia_merge`

This is where separate rasters become a single dataset. The main loop:

```
For each spatial tile (window):
    Load static features once (elevation, roads, WorldClim...)
    For each year (2000-2024):
        Load yearly features (NDVI, WDPA, deforestation...)
        Combine static + yearly → single band stack
        Write row to Parquet
```

**Output:** `merged_panel_2000_2024.parquet` — one row per pixel per year, ~60 columns.

Forward-fill logic: if a dataset lacks data for a target year, use the most recent
prior year's data (e.g., population from 2015 used for 2016-2019).

### Step 4: Engineer Features

**File:** `scripts/regions/south_america/colombia/preprocessing/colombia_features`

Processes each year sequentially. For each year:

**4a. Compute WDPA_prev (1-year lag)**

For each pixel, look up its WDPA status from the **previous** year. This prevents
data leakage — you can't use the current year's protection status to predict
whether the pixel becomes protected this year.

Example:
```
Year 2011: pixel is unprotected (WDPA_b1 = 0)
Year 2012: pixel becomes protected (WDPA_b1 = 1)

When predicting for year 2012:
  WDPA_prev = 0 (from 2011)  ← this is what the model sees
  WDPA_b1 = 1 (from 2012)    ← this is the answer, excluded from features
```

First year in the dataset: WDPA_prev = 0 for all pixels (no prior year exists).

**4b. Compute Distance Features**

Using `scipy.ndimage.distance_transform_edt`:

1. Take the WDPA_prev grid (binary: 0 = unprotected, 1 = protected)
2. Compute Euclidean distance from each unprotected pixel to the nearest protected one
3. Multiply by 1000 to convert from pixels to meters

Same logic for dist_road, dist_oil_gas, dist_powerplant (computed once on a
reference year since these are static).

**4c. Compute Spatial Smoothing**

Using `scipy.ndimage.uniform_filter` — a moving average over a square window:

- `uniform_filter(grid, size=16)` → 16×16 pixel neighborhood (~16km)
- `uniform_filter(grid, size=64)` → 64×64 pixel neighborhood (~64km)

Applied to: NDVI, population, nighttime lights, deforestation, wildfire, GSN
biodiversity (all 5 bands), elevation, slope.

**4d. Risk-Set Filter**

Keep only pixels where `WDPA_prev == 0` (not already protected in the previous
year). Pixels already protected can't "become protected" again — they would create
meaningless data.

**4e. Create Transition Target**

For risk-set pixels: `transition_01 = WDPA_b1` (1 if the pixel became protected
this year, 0 otherwise). NaN values treated as 0.

**Output:** `merged_panel_final.parquet` — risk-set only, with engineered features.

### Step 5: Create Train/Test Splits and 5-Year Lookahead Target

**File:** `scripts/regions/south_america/colombia/4_ml/splits/modelC_splits`

**The 5-year lookahead target (transition_01_win5)**

The raw `transition_01` tells you if a pixel became protected THIS year. But we
want to predict 5 years ahead. So we compute `transition_01_win5`:

Example for a pixel that becomes protected in 2012:

| Year | Protected? | Within 5 years of becoming protected? | transition_01_win5 |
|------|-----------|---------------------------------------|-------------------|
| 2006 | No | 2012 within [2006, 2011]? No | 0 |
| 2007 | No | 2012 within [2007, 2012]? Yes | **1** |
| 2008 | No | 2012 within [2008, 2013]? Yes | **1** |
| 2009 | No | 2012 within [2009, 2014]? Yes | **1** |
| 2010 | No | 2012 within [2010, 2015]? Yes | **1** |
| 2011 | No | 2012 within [2011, 2016]? Yes | **1** |
| 2012 | Yes | Already protected → excluded from risk set | N/A |

The SQL logic:
```sql
CASE WHEN first_transition_year IS NOT NULL
      AND year < first_transition_year
      AND first_transition_year <= year + 5
     THEN 1 ELSE 0
END AS transition_01_win5
```

**Right-censoring protection**

WDPA data goes through 2024. A pixel in year 2020 has a 5-year window through 2025,
but we don't have 2025 data. The label would be incomplete — a pixel that becomes
protected in 2025 would be wrongly labeled 0. So:

```
LAST_LABEL_YEAR = 2024 - 5 = 2019
```

No pixel-year after 2019 gets a target label.

**Temporal splits**

After computing transition_01_win5, the data is split purely by year:

```
2000              2013  2014       2016  2017       2019
 |__________________|    |__________|    |__________|
       TRAIN              EARLYSTOP          TEST
    (learn patterns)    (stop training    (evaluate:
                         before            did we predict
                         overfitting)      reality?)
```

- **Train (2000-2013):** Model learns historical patterns
- **Earlystop (2014-2016):** Prevents overfitting (not used for evaluation)
- **Test (2017-2019):** Completely held out — checks predictions against reality

### Step 6: Train LightGBM

**File:** `scripts/regions/south_america/colombia/4_ml/training/modelC_LGBM`

**6a. Feature selection**

All numeric columns except EXCLUDE_COLS become features (~60 columns):
```python
EXCLUDE_COLS = {'transition_01', 'transition_01_win5',  # targets
                'WDPA_b1', 'WDPA_prev',                 # protection status
                'x', 'y', 'row', 'col', 'year'}         # identifiers
```

**6b. Time weights**

Recent training years are weighted higher than older ones:

```
weight = 0.5 + 0.5 × (year - 2000) / (2013 - 2000)
```

| Year | Weight |
|------|--------|
| 2000 | 0.50 (half weight) |
| 2006 | 0.73 |
| 2013 | 1.00 (full weight) |

A mistake on a 2013 row costs the model twice as much as one on a 2000 row.

**6c. Class imbalance handling (scale_pos_weight)**

With only ~0.5% positive pixels, the model could get 99.5% accuracy by predicting
"not protected" for everything. To prevent this:

```python
scale_pos_weight = n_negative / n_positive
# e.g., 1,200,000 / 6,000 = 200
```

This tells LightGBM: getting a positive example wrong costs 200x more than a negative.

**6d. Temporal cross-validation (3 folds)**

The model trains 3 times with expanding time windows:

```
Fold 1: Train [2000-2008] → Validate [2009-2011] → best_iteration = 847
Fold 2: Train [2000-2011] → Validate [2012-2013] → best_iteration = 923
Fold 3: Train [2000-2013] → Validate [2014-2016] → best_iteration = 1,102
```

Average × 1.05 buffer = chosen_iteration ≈ 1,009

**6e. Final model training**

A final model trains on all train + earlystop data (2000-2016) for exactly
chosen_iteration rounds — no early stopping, no validation set. The iteration
count is fixed from CV.

The model is saved as a pickle file (.pkl).

### Step 7: Predict on Test Set

**Same file, Phase 2**

The saved model is loaded and run on every test pixel (2017-2019). Each pixel gets
a predicted probability between 0 and 1.

Output: `modelC_lgbm_win5_scored_{timestamp}.parquet`

### Step 8: Evaluate

Metrics computed against actual outcomes:

```python
roc_auc = roc_auc_score(y_true, y_proba)          # overall ranking
pr_auc = average_precision_score(y_true, y_proba)  # precision-recall
precision_at_1pct = compute_precision_at_k(...)     # top 1% accuracy
```

### Step 9: Calibrate

**File:** `scripts/regions/south_america/colombia/4_ml/evaluation/calibrate_C`

Raw probabilities are corrected using Platt scaling (logistic regression on
logit-transformed predictions) so that predicted probabilities match actual rates.

### Step 10: Generate Figures

**File:** `scripts/regions/south_america/colombia/4_ml/results/modelC_results`

Produces: PR curves, risk maps, probability maps, calibration curves, metrics
tables (CSV + LaTeX), SHAP feature importance plots.

### Following One Pixel End-to-End

Pixel (row=100, col=200) in Colombia:

| Stage | What Happens |
|-------|-------------|
| Extraction | GEE exports NDVI=0.72, elevation=1200m, WDPA=0 for 2017 |
| Merging | Combines with WorldClim, GSN, deforestation into one row |
| Feature engineering | Computes dist_wdpa=12,400m, NDVI_smooth64=0.68, WDPA_prev=0 |
| Risk-set filter | WDPA_prev=0 → kept (not already protected) |
| Splits | Year 2017 → goes into TEST set |
| Target | Pixel became protected in 2020 → transition_01_win5 = 1 |
| Prediction | Model outputs y_proba = 0.034 (3.4% predicted probability) |
| Calibration | Platt scaling adjusts to 0.021 (2.1% calibrated) |
| Evaluation | This pixel is in model's top 1% → counted toward Precision@1% |

---

## 5. Understanding the Results

### What the Metrics Mean

- **ROC-AUC (0.92-0.95):** "If I pick a random positive and random negative pixel,
  there's a 92-95% chance the model scores the positive one higher." Perfect = 1.0,
  random = 0.5.

- **PR-AUC (0.41-0.53):** Precision-recall tradeoff. Looks low, but with ~0.5%
  positive rate, random would get ~0.005. So 0.47 is ~100x better than random.

- **Precision@1% (46-63%):** "Of the top 1% most confident predictions, how many
  actually became protected?" Answer: roughly half. Given the base rate is <1%,
  this is extremely powerful.

- **Lift@1% (50-77x):** The model is 50-77 times better than random guessing at
  identifying future PAs in its top 1%.

- **Brier Score:** Mean squared error of probabilities. Lower = better calibrated.

### Why "Half Right" Is Extraordinary

54.7% precision sounds mediocre like an exam score. But context matters:

Colombia has ~1.9M test pixels. The base rate is 0.92% (~17,500 become protected).

| Strategy | Pixels examined (1%) | Actual PAs found | Hit rate |
|----------|---------------------|-------------------|----------|
| Random selection | 19,000 | ~175 | 0.9% |
| Model's top 1% | 19,000 | ~10,400 | 54.7% |

The model finds ~60x more actual PAs in the same number of pixels. A conservation
agency looking at just 1% of the landscape can catch ~60% of all future PAs.

### Feature Importance Rankings

Top features across all models (from SHAP analysis):

| Rank | Feature | Interpretation |
|------|---------|---------------|
| 1 | GSN_b2_smooth16 | Biodiversity importance (16km neighborhood) |
| 2 | WorldClim_b14 | Precipitation of driest month |
| 3 | GSN_b2_smooth64 | Biodiversity importance (64km neighborhood) |
| 4 | dist_road | Distance to nearest road (remoteness) |
| 5 | HNTL_b1_smooth64 | Nighttime lights (low economic activity) |
| 6 | dist_wdpa | Distance to nearest existing PA |
| 7 | WorldClim_b17 | Precipitation seasonality |
| 8 | deforestation_b1_smooth64 | Regional deforestation rate |
| 9 | dist_powerplant | Distance to power plants |
| 10 | GPW_b1 | Population density |

**The story:** Protection happens where biodiversity is high, human presence is low,
the area is remote, and there's already conservation momentum nearby. Governments
protect cheap, remote, biodiverse land near existing PAs — because that's
historically what they've done.

### Model Comparison

| Model | Region | ROC-AUC | PR-AUC | P@1% | Lift@1% |
|-------|--------|---------|--------|------|---------|
| SA LightGBM (main) | South America | 0.926 | 0.430 | 49.4% | 62x |
| SA RF (main) | South America | 0.947 | 0.414 | 46.1% | 77x |
| Colombia LightGBM (main) | Colombia | 0.946 | 0.470 | 54.7% | 59x |
| Colombia LightGBM (robust.) | Colombia | 0.925 | 0.535 | 62.8% | 57x |
| Colombia RF (main) | Colombia | 0.931 | 0.435 | 46.7% | 51x |
| Colombia BRF (main) | Colombia | 0.899 | 0.259 | 36.7% | 54x |

Random Forest has higher ROC-AUC; LightGBM has higher PR-AUC and Precision@1%.

---

## 6. Calibration and Reliability Curves

### Why LightGBM Outputs Aren't True Probabilities

LightGBM optimizes for ranking — it scores high-risk pixels higher than low-risk
ones. But it doesn't guarantee that "0.03" literally means "3 in 100." The model
is like a teacher who ranks students perfectly but whose grade percentages are
arbitrary.

### What Reliability Curves Show

A reliability curve answers: "When the model says X% probability, does it actually
happen X% of the time?"

- X-axis: What the model predicted (e.g., 60% chance)
- Y-axis: What actually happened (e.g., only 20% of those became protected)
- Dashed diagonal: Perfect calibration (predicted = actual)

If the blue curve bows below the diagonal, the model is overconfident.

### Two Calibration Methods

**Platt scaling:** Fits a logistic regression on the model's raw outputs. Learns
a global correction (2 parameters: slope + intercept). Simple and robust.

**Isotonic regression:** Non-parametric staircase-shaped correction. More flexible
but can overfit with small calibration sets.

In this project, per-fold calibrators are fitted during temporal CV, then averaged.

### Results

Platt scaling reduces Brier score by 31% — meaning the calibrated probabilities
are significantly more trustworthy. This is a strength worth highlighting in the
paper, as many ML papers skip calibration entirely.

### Why Calibration Matters for Policy

Without calibration: "pixel A is *more likely* than pixel B" (ranking only).
With calibration: "pixel A has a 4% chance of becoming protected" — a much
stronger claim for policymakers.

---

## 7. Policy Context: 30x30 and Carbon

### The 30x30 Target

The Kunming-Montreal Global Biodiversity Framework (COP15, December 2022) commits
nearly 200 countries to protect 30% of land and ocean by 2030. Currently ~17% of
land is protected. Countries must roughly double PA coverage in under 5 years.

This model addresses: **"Where will that doubling happen?"**

Three policy angles:

1. **Predicting the expansion path.** Governments face pressure to designate fast.
   The model forecasts which land will be designated — useful for planners,
   investors, and indigenous communities.

2. **Exposing the representation gap.** If expansion follows historical patterns
   (cheap, remote land), 30% coverage ≠ 30% of biodiversity covered. Some biomes
   may remain systematically underprotected even after hitting 30%.

3. **Transition risk.** Landowners with agricultural or mining concessions can
   assess the probability their land gets designated.

### Why Carbon Stocks Matter

Carbon stocks influence PA designation through money:

- **REDD+** pays countries to keep forests standing. Designating carbon-rich
  forests generates payments, making it profitable for governments.
- **Voluntary carbon markets** create incentives to protect specifically
  high-carbon forests (not just biodiverse ones).
- **Paris Agreement NDCs** — many countries included forest conservation in climate
  pledges. Protecting high-carbon areas counts toward both climate AND biodiversity
  targets.

A pixel with 200 tonnes/ha of biomass is more likely to become a PA than an equally
biodiverse grassland with 20 tonnes/ha — because the forest has carbon market value.
The current model can't distinguish these (no carbon data).

---

## 8. Prediction vs. Prescription

### The Key Distinction

This model predicts **where governments WILL designate** protected areas — NOT
**where they SHOULD designate** them.

| | This Model (Descriptive) | Conservation Optimization (Prescriptive) |
|---|---|---|
| Goal | Predict political behavior | Maximize biodiversity per dollar |
| Inputs | Historical patterns, accessibility | Species richness, threat, cost |
| Output | "This area will likely be designated" | "This area should be designated" |
| Use case | Risk assessment, policy analysis | Conservation planning |

The model learns that governments protect easy land. But an agency should target
the most *effective* land — which may be entirely different.

### Paper Opportunity

The gap between "where protection goes" (model predictions) and "where it's most
needed" (biodiversity priority maps like KBAs) could be a key finding:

*"Our model reveals that historical PA designation follows a path of least
resistance — remote, low-conflict, biodiverse land. Comparing predictions with
biodiversity priority maps reveals systematic gaps where high-value land is unlikely
to receive protection under current political dynamics."*

---

## 9. Code Quality Assessment

### Strengths

- **Pipeline architecture:** Clean 4-stage flow (extraction → preprocessing →
  merging → ML), easy to follow
- **Memory efficiency:** Streaming I/O and batch processing for 350M+ row datasets
- **Reproducibility:** Fixed seeds (42), portable paths with $SCRATCH override
- **Documentation:** Excellent module-level docstrings (91 lines at top of merge)
- **No security issues:** No hardcoded credentials, proper environment variables
- **Type hints and PEP 8:** Consistent throughout

### Concerns

1. **No software tests** — Zero test files. The single biggest gap.

2. **Monolithic files** — The RF training script is 1,600 lines in one file.

3. **Code duplication** — The calibration script exists in 3 identical copies
   (South America, Colombia, USA) with only path strings changed. The
   USA and South America preprocessing scripts are also copy-pasted.

4. **Inconsistent logging** — Some files use `print()`, others `logging.info()`.

5. **Magic numbers** — Train year ranges, test start years, and other constants
   scattered inline rather than in a config file.

6. **Incomplete data validation** — No checks for empty DataFrames after loads,
   some bare `except Exception` blocks that swallow errors silently.

7. **Reproducibility gaps** — `np.random.seed()` not set globally; package
   versions in environment.yml not fully pinned.

---

## 10. Improvement Priorities

### High Priority (Paper-Critical)

1. **Add software tests** — Add pytest suite with:
   - Unit tests for feature engineering (distance transforms, spatial smoothing)
   - Integration test on a small synthetic dataset
   - Regression test: same seed produces same metrics
   - A test suite has been created at `tests/test_pipeline.py` (34 tests, all passing)

2. **Ablation study** — Remove feature groups one at a time and measure impact.
   Shows what drives protection decisions, not just that the model works.
   - A script has been created at
     `scripts/regions/south_america/colombia/4_ml/training/ablation_study`

3. **Second continent** — Run the pipeline on USA or another region. Transforms
   a regional result into a generalizable finding. USA templates already exist.

4. **Literature comparison** — Benchmark against published PA prediction methods.

### Medium Priority (Quality & Credibility)

5. **Pin all package versions** — environment.yml uses `python=3.12` not `3.12.1`.
   LightGBM version unspecified.

6. **Extract shared utilities** — Refactor duplicated scripts into `scripts/common/`.

7. **Highlight calibration in paper** — The Platt/isotonic correction is a strength
   many papers skip.

8. **Explain right-censoring in methods section** — The LAST_LABEL_YEAR = 2019
   constraint is methodologically rigorous. Make it prominent.

### Lower Priority (Polish)

9. **Consistent logging** — Replace mixed print/logging with unified framework.
10. **Break up monolithic files** — Split 1,600-line scripts into modules.
11. **Forward prediction to 2030** — "Where will protection happen next?" maps.

---

## 11. The 5-Year Window and Forward Projection to 2030

### Why 5 Years (and Not 6 or 10)?

The 5-year lookahead window (`LOOKAHEAD_YEARS = 5`) is a modeling choice driven by
several factors:

**Policy alignment.** The 30x30 target gives countries until 2030. With data through
2024, a 5-year lookahead answers exactly: "what gets protected by ~2029?" — directly
policy-relevant.

**Right-censoring constraint.** WDPA data ends in 2024. The model can only label
years where the full window is observable:
- 5-year window → labels through 2019 → Train (2000-2013), Earlystop (2014-2016), Test (2017-2019)
- 10-year window → labels through 2014 → only 9 training years, test ends in 2014 (misses recent trends)
- 3-year window → labels through 2021 → more data, but fewer positive examples per pixel-year

**Signal-to-noise tradeoff.**
- Too short (1-2 years): too few transitions (~0.1% positive) for the model to learn
- Too long (10+ years): features at year t become weakly predictive of year t+10 (conditions change)
- 5 years: enough positives (~0.5%) with features still predictive of near-future outcomes

**Convention.** 5-year windows are standard in land-use change and deforestation
prediction literature.

A **sensitivity analysis** testing 3, 5, and 7-year windows would strengthen the
paper by showing results are robust to this choice.

### Can the Model Project to 2030?

**Yes.** The model is trained to answer: "Given features at year t, what's the
probability this pixel becomes protected within 5 years?" Feed it 2024 features →
it outputs a probability of protection by ~2029.

The infrastructure already exists. The embeddings pipeline
(`modelE_splits`, lines 422-442) includes a `FUTURE_SCORING` export that extracts
feature data for recent years with no labels, specifically for inference. The results
script already generates probability maps and risk maps for the test period.

**What's needed:**
1. Assemble 2024 features for all currently-unprotected pixels
2. Run `model.predict()` on those features
3. Map the output — each pixel gets a probability of becoming protected by ~2029

**Caveats for the paper:**
- The projection shows where protection goes *if historical patterns continue*
- It assumes **stationarity** — that the features-to-designation relationship doesn't change
- The 30x30 push may change both *how much* gets protected (faster) and *what kind* of land (political urgency could shift patterns toward land governments wouldn't have historically protected)
- The gap between this projection and biodiversity priority maps is itself a finding

### Accounting for Acceleration Under 30x30

The model was trained on historical designation rates (~17% of land protected). The
30x30 target may accelerate designation. Several approaches can address this, from
simple to sophisticated:

**Approach 1: Threshold Adjustment (Simplest)**

The model ranks pixels by probability. Acceleration doesn't change the *ranking* —
the same types of land remain attractive. Only the *cutoff* changes. Instead of
asking "will this pixel be protected?" you ask "what if governments protect twice as
much land?"

```
Historical: protect top 2% of ranked pixels → matches observed ~17% coverage
30x30 scenario: protect top 4% of ranked pixels → projects toward ~30% coverage
```

This is easy to implement and defensible: it says "if governments scale up by
protecting the *same kinds* of land, here's what gets covered." The model already
outputs a ranked list — you just move the threshold.

**Approach 2: Rate Multiplier on Probabilities**

Calculate the ratio between the required designation rate and the historical rate,
then scale probabilities:

```
Current protected: 17%
Target: 30%
Remaining years: ~4
Historical annual rate: ~0.5%/year
Required rate: ~3.25%/year → multiplier ≈ 6.5×
```

Apply: `p_adjusted = min(1.0, p_historical × multiplier)`

Simple but crude — it assumes uniform acceleration across all pixel types. In
reality, some types may accelerate more than others.

**Approach 3: Scenario-Based Analysis (Recommended for Paper)**

Present multiple scenarios without claiming one is correct:

| Scenario | Assumption | Method |
|----------|-----------|--------|
| Business-as-usual | Historical patterns continue | Raw model predictions |
| Moderate acceleration | 2× designation rate | Threshold at top 4% |
| 30x30 compliance | Full 30% target met | Threshold at top ~15% of unprotected land |

This is the most honest approach for a paper. It acknowledges uncertainty about the
political dynamics while still providing useful projections. The reader can see what
changes under different assumptions.

**Approach 4: Time Trend Feature (Requires Retraining)**

Add a feature capturing the *political environment*:
- A binary "post-COP15" indicator (2023+ = 1)
- A continuous "years remaining until 2030" feature
- The cumulative % of land already protected nationally

This requires retraining the model and is speculative (you're extrapolating a trend
that hasn't fully played out yet), but it would allow the model to learn that
designation *accelerates* as deadlines approach.

**Approach 5: Post-2019 Recalibration (Most Rigorous)**

The model is trained on 2000-2019 data, but WDPA data exists through 2024. Use
2020-2024 designation data (which the model has never seen) to:

1. Run model predictions for 2020-2024 features
2. Compare against actual 2020-2024 designations
3. If designation rates increased post-COP15, fit a correction factor

This is the most data-driven approach — it uses real post-2022 acceleration data
rather than assumptions. The student could check: "Did designation rates actually
increase after COP15? By how much? Does the model's ranking still hold?"

**Recommendation for the paper:** Use **Approach 3** (scenario analysis) as the
primary framing, supplemented by **Approach 5** (post-2019 recalibration) as
empirical evidence. This gives the paper both honest uncertainty framing and
data-driven grounding.

### Bayesian Network as Interpretability Layer (Future Work)

A key critique of ML models in environmental science is the "black box" problem:
LightGBM achieves ROC-AUC 0.94, but what did it actually learn? SHAP provides
feature importance rankings, but doesn't reveal the **dependency structure** between
features.

**The idea:** Keep LightGBM as the prediction engine. Build a Bayesian network (BN)
as a second, post-hoc model that explains *what LightGBM learned* in a transparent,
human-readable form.

```
Layer 1: LightGBM (prediction)
   60 features → probability per pixel → ROC-AUC 0.94
   Role: maximize accuracy

Layer 2: Bayesian Network (interpretation)
   Top 10-15 features (discretized) → DAG + conditional probability tables
   Role: reveal dependency structure, address black-box critique
```

**Implementation steps:**

1. **Select key features.** Take top 10-15 from SHAP: `dist_wdpa`, `dist_road`,
   `GSN_b2`, `WorldClim_b14`, `HNTL_b1`, `NDVI`, `GPW_b1`, `deforestation`,
   `elevation`, etc.

2. **Discretize into meaningful categories:**
   ```
   dist_wdpa:     "adjacent" (<5km), "near" (5-50km), "far" (>50km)
   elevation:     "lowland" (<500m), "mid" (500-2000m), "highland" (>2000m)
   GSN_b2:        "low biodiversity", "medium", "high"
   GPW_b1:        "uninhabited", "sparse", "populated"
   designation:   "protected" (1) / "not protected" (0)
   ```

3. **Learn BN structure** using `pgmpy`:
   ```python
   from pgmpy.estimators import HillClimbSearch, BicScore
   from pgmpy.models import BayesianNetwork
   from pgmpy.estimators import BayesianEstimator

   hc = HillClimbSearch(df_discretized)
   best_dag = hc.estimate(scoring_method=BicScore(df_discretized))

   model = BayesianNetwork(best_dag.edges())
   model.fit(df_discretized, estimator=BayesianEstimator)
   ```

4. **Read the outputs:**
   - A **DAG** showing which features influence designation and through what paths
   - **Conditional probability tables** directly readable by policymakers:

   | Biodiversity | Dist to PA | P(designation) |
   |-------------|-----------|----------------|
   | High | Adjacent (<5km) | 8.2% |
   | High | Far (>50km) | 0.3% |
   | Low | Adjacent (<5km) | 2.1% |
   | Low | Far (>50km) | 0.05% |

**Why this strengthens the paper:**
- Deflects the "black box" critique with a transparent companion model
- Shows dependency structure (e.g., "population density works *through* remoteness,
  not directly") — richer than SHAP rankings alone
- Produces paper-ready DAG figures readable by non-technical audiences
- Validates LightGBM findings through convergent evidence from a different method

**Important caveat:** The BN structure is correlational, not causal. The paper should
state: *"The learned structure represents conditional dependencies consistent with the
data, not confirmed causal relationships."*

**Computational note:** Manageable because the BN uses ~15 discretized variables (not
60 continuous). Can run on Colombia (7M rows) or a sample thereof.

---

## 12. Missing Variables

Features that could improve predictive power:

| Variable | Why It Matters | Data Source |
|----------|---------------|-------------|
| Land tenure / ownership | Public land far easier to designate | National cadastral data |
| Indigenous territories | Many PAs overlap indigenous lands | RAISG, LandMark |
| Carbon stocks | REDD+ and carbon markets drive designation | ESA CCI Biomass |
| Species richness | More targeted than GSN proxy | IUCN Red List, GBIF |
| PA network connectivity | Corridor gaps are priority targets | Derived from WDPA |
| Land economic value | High-value land resists protection | FAO, national data |
| International commitments | 30x30 creates political pressure | CBD pledges |
| NGO presence / funding | Active NGOs accelerate designation | IUCN, WCS, WWF |

Most impactful additions would likely be **land tenure**, **indigenous territories**,
and **carbon stocks**.

---

## 13. Key File Locations

### Pipeline Scripts (Colombia)

| Stage | File |
|-------|------|
| GEE export | `scripts/regions/south_america/colombia/preprocessing/colombia_export` |
| Merge | `scripts/regions/south_america/colombia/3_merging/colombia_merge` |
| Feature engineering | `scripts/regions/south_america/colombia/preprocessing/colombia_features` |
| Splits | `scripts/regions/south_america/colombia/4_ml/splits/modelC_splits` |
| LightGBM training | `scripts/regions/south_america/colombia/4_ml/training/modelC_LGBM` |
| RF training | `scripts/regions/south_america/colombia/4_ml/training/modelC_RF` |
| Hyperparameters | `scripts/regions/south_america/colombia/4_ml/training/lgbm_best_params_main.json` |
| Calibration | `scripts/regions/south_america/colombia/4_ml/evaluation/calibrate_C` |
| Benchmarking | `scripts/regions/south_america/colombia/4_ml/evaluation/benchmark_C` |
| Spatial CV | `scripts/regions/south_america/colombia/4_ml/evaluation/spatial_cv_C` |
| Results/figures | `scripts/regions/south_america/colombia/4_ml/results/modelC_results` |
| Ablation study | `scripts/regions/south_america/colombia/4_ml/training/ablation_study` |

### Pipeline Scripts (South America Continental)

| Stage | File |
|-------|------|
| Merge | `scripts/regions/south_america/3_merging/merge` |
| Feature engineering | `scripts/regions/south_america/3_merging/feature_engineering` |
| LightGBM training | `scripts/regions/south_america/4_ml/2_training/model1_LGBM` |
| RF training | `scripts/regions/south_america/4_ml/2_training/model1_RF` |
| Calibration | `scripts/regions/south_america/4_ml/3_evaluation/calibrate_1` |
| Spatial generalization | `scripts/regions/south_america/4_ml/3_evaluation/spatial_generalisation` |
| Results/figures | `scripts/regions/south_america/4_ml/4_results/model1_results` |

### Outputs

| Output | Location |
|--------|----------|
| SA metrics tables | `outputs/south_america/results/model1_lgbm/main/` |
| SA reliability curves | `outputs/south_america/results/model1_lgbm/main/reliability_curve_*.png` |
| SA benchmark JSONs | `outputs/south_america/results/ml_models/main/` |
| SA SHAP importance | `outputs/south_america/results/model1_lgbm/main/*_shap_top20_*.csv` |
| Colombia metrics | `outputs/south_america/colombia/results/modelC_lgbm/main/` |
| Colombia calibration | `outputs/south_america/colombia/results/modelC_rf/main/*calibration*.pdf` |

### Tests and Infrastructure

| Item | Location |
|------|----------|
| Unit tests | `tests/test_pipeline.py` (34 tests) |
| Environment | `environment.yml` |
| SLURM jobs | `slurm/` |
| Project docs | `CLAUDE.md`, `README.md`, `context.md` |

### Running the Pipeline

```bash
# Setup environment
conda env create -f environment.yml
conda activate pa3030

# Run pipeline (Colombia example)
python scripts/regions/south_america/colombia/3_merging/merge_all.py
python scripts/regions/south_america/colombia/4_ml/1_splits/create_splits.py
python scripts/regions/south_america/colombia/4_ml/2_training/train_lgbm.py
python scripts/regions/south_america/colombia/4_ml/3_evaluation/evaluate.py

# Run tests
pytest tests/test_pipeline.py -v

# Run ablation study
python scripts/regions/south_america/colombia/4_ml/training/ablation_study --split-version main
```
