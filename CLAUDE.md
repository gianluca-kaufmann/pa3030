# PA3030 - Protected Area Designation Prediction

## Overview

PA3030 is a large-scale, reproducible machine learning pipeline that predicts the probability of a given 1 km x 1 km pixel becoming newly designated as a protected area within a 5-year window. 

The core question: **"Given historical patterns of protected-area establishment, which locations are most likely to be designated as protected areas in the future?"**

## Policy Context: The 30x30 Target

The Kunming-Montreal Global Biodiversity Framework (COP15, December 2022) commits
nearly 200 countries to protect 30% of land and ocean by 2030. Currently ~17% of
land is protected, meaning countries must roughly double PA coverage in under 5 years.

This model directly addresses: **"Where will that doubling happen?"**

Three policy-relevant angles:

1. **Predicting the expansion path.** Governments face pressure to designate fast.
   This model forecasts which land will be designated based on historical patterns —
   useful for land-use planners, investors, and indigenous communities anticipating
   changes.
2. **Exposing the representation gap.** If expansion follows historical patterns
   (cheap, remote land), 30% coverage does not mean 30% of biodiversity is covered.
   Some biomes (tropical dry forests, grasslands, wetlands) may remain systematically
   underprotected. The gap between predicted designations and biodiversity priority
   maps is a finding in itself.
3. **Transition risk.** Landowners and investors holding agricultural or mining
   concessions can assess the probability their land gets designated. With 30x30
   creating political urgency, designation rates may be accelerating.

### Why Carbon Stocks Are Relevant

Carbon stocks influence PA designation through financial incentives:

- **REDD+** pays countries to keep forests standing. Designating carbon-rich forests
  as PAs generates international payments or carbon credits, making it profitable.
- **Voluntary carbon markets** let companies buy offsets tied to "avoided
  deforestation" in protected areas, creating incentives to protect high-carbon forests
  specifically.
- **Paris Agreement NDCs** — many South American countries included forest
  conservation in climate pledges. Protecting high-carbon-stock areas counts toward
  both climate and biodiversity targets simultaneously.

A pixel with 200 tonnes/ha of biomass is more likely to become a PA than an equally
biodiverse grassland with 20 tonnes/ha — because the forest has carbon market value.
The current model cannot distinguish these because it has no carbon data.

## Use Cases

- Forecasting PA expansion under the 30x30 biodiversity target
- Transition-risk analysis for investors and policymakers
- Identifying the gap between where protection goes vs. where it's most needed
- Academic research on land-use change and environmental policy

## Pipeline Architecture

```
Continental regions (South America, USA, SE Asia):
  1_extraction/    -> Export from Google Earth Engine + external APIs to GeoTIFF rasters
  2_preprocessing/ -> Format harmonization, reprojection, storage optimization
  3_merging/       -> Build modeling panels (Parquet) + feature engineering
  4_tuning/        -> Hyperparameter search (Optuna / randomized, 3-fold temporal CV)
  5_training/      -> Train/test splits + final model training (LightGBM, RF)
  6_evaluation/    -> Temporal CV, spatial CV, benchmarking, calibration
  7_results/       -> Metrics, visualizations, risk maps
  8_forward/       -> Forward predictions to 2030 (deployment model + calibrated inference + scenario analysis)

Sub-region Colombia (south_america/colombia/):
  preprocessing/ -> Colombia-specific feature extraction + backbone
  3_merging/     -> Colombia-specific merge + feature engineering
  4_ml/
    splits/      -> Train/validation/test data splits
    training/    -> Model training + hyperparameter tuning (LightGBM, RF, BRF)
    evaluation/  -> Calibration, benchmarking, spatial CV
    results/     -> Figures, metrics tables, probability maps
  (Legacy sub-region used for early development; not on the critical path)

Shared utilities (scripts/regions/shared/):
  tuning/          -> Optuna HPO runner, CV, search spaces, artifacts
  training/        -> Shared training utilities
  evaluation/      -> benchmark_core, calibration_core, spatial_cv_core, spatial_cv_visualise
  results/         -> results_core, boundaries, io, runner, config
  forward/         -> Shared forward-prediction core (coverage_core, features_core,
                      predict_core, results_core, backtest_core, config, runner)
  1_preprocessing/ -> economic_value (shared preprocessing utility)
```

## Tech Stack

- **Language:** Python 3.12
- **ML:** LightGBM (primary), Scikit-learn, Imbalanced-learn, SHAP
- **Geospatial:** Rasterio, GeoPandas, Shapely, PyProj, Google Earth Engine
- **Data:** Pandas, NumPy, PyArrow (Parquet)
- **Visualization:** Matplotlib, Seaborn
- **Compute:** ETH Zurich Euler HPC cluster (SLURM) or local macOS/Linux
- **Experiment Tracking:** Weights & Biases (optional)

## Regions & Models

| Model | Region | Scale | Status | Description |
|-------|--------|-------|--------|-------------|
| Model 1 | South America (continental) | ~350M pixel-years | ✅ Complete | Full-scale primary model |
| Model 2 | USA (continental) | ~200M pixel-years | ✅ Complete | Cross-continental validation |
| Model 3 | SE Asia (continental) | ~100M pixel-years | ✅ Complete | Third continental region |
| Model C | Colombia (sub-region) | ~7M pixel-years | ✅ Legacy | Early dev sub-region; non-standard pipeline structure |
| Model E | South America | ~20M pixel-years | ⚠️ Partial | Satellite embedding experiments; ML stages incomplete |

## Data Characteristics

- **Spatial resolution:** ~1 km x 1 km (EPSG:3857)
- **Temporal coverage:** 2000-2024 (annual)
- **Target variable:** `transition_01_win5` (binary, 5-year lookahead)
- **Class imbalance:** ~0.3-0.5% positive (extreme)
- **Features (~60):** elevation, slope, climate normals, biodiversity importance, population density, night-time lights, NDVI, deforestation, wildfire, land-cover, distance to infrastructure, spatial smoothing

## Key Design Decisions

- **Right-censoring protection:** `LAST_LABEL_YEAR = 2019` ensures complete 5-year lookahead for all labels
- **Risk-set filtering:** Only unprotected pixels (`WDPA_prev == 0`) are eligible for transition
- **Temporal weighting:** Recent years weighted higher than earlier years
- **Class imbalance:** Handled via `scale_pos_weight = n_neg / n_pos`
- **Batch processing:** Memory-efficient streaming I/O for 100M+ row datasets
- **Reproducibility:** Fixed random seeds (42), environment-agnostic paths via `get_repo_root()`

## Model Evaluation (What "Testing" Means)

This project has no software unit tests. "Testing" refers to model evaluation — checking
how well predictions match reality on held-out future data.

### Temporal Splits
- **Train (2000-2013):** Learn historical protection patterns
- **Early-stop (2014-2016):** Prevent overfitting (not used for evaluation)
- **Test (2017-2019):** Evaluate predictions against what actually happened

### Metrics Computed
- **ROC-AUC:** Overall ranking quality (0.92 on Colombia)
- **PR-AUC:** Precision-recall tradeoff, more meaningful under extreme class imbalance
- **Precision@K%:** Of the top K% predicted pixels, how many actually became PAs?
- **Lift@K%:** How many times better than random guessing (65x at top 1%)
- **Brier Score:** Mean squared error of predicted probabilities
- **ECE/MCE:** Expected/maximum calibration error

### Calibration
Raw LightGBM probabilities are post-hoc calibrated using:
- **Platt scaling:** Logistic regression on logit-transformed predictions
- **Isotonic regression:** Non-parametric monotonic correction
Per-fold calibrators are fitted during temporal CV, then averaged.

### Spatial Generalization
Three-layer framework using WWF biomes from the GSN terrestrial ecoregions dataset:

- **Layer 1 — Leave-One-Biome-Out (LOBO)** (`spatial_CV_1`): For each biome fold,
  train on all other biomes, evaluate on the held-out biome's test pixels.
  Run as a SLURM array job (one task per biome). Aggregate results with
  `spatial_CV_1_aggregate`. Tests whether the model generalizes to unseen geographies.
- **Layer 2 — Biome-stratified metrics** (`spatial_CV_2`): Loads existing scored
  parquets from the full trained model, assigns WWF biome labels via GSN raster
  row/col lookup, computes per-biome metrics. Fast diagnostic; run after step 5.
- **Layer 3 — Cross-continental transfer** (`spatial_CV_3`): Trains on SA, evaluates
  on USA and/or SE Asia; cross-transfers between regions. Uses features common to
  all regions. Strongest generalization claim — success means patterns are
  continent-independent. Scripts exist for SA and SE Asia.

All logic lives in `scripts/regions/shared/evaluation/spatial_cv_core.py`.
Region-specific scripts are thin wrappers (2–3 lines each).

### Outputs
- Benchmark JSON (structured metrics for comparison)
- PR curves, cumulative gains charts, reliability diagrams (PNG + PDF)
- Risk maps (top K% predicted pixels vs observed PAs)
- Probability maps (continuous predictions across the continent)
- LaTeX metrics tables (paper-ready)
- SHAP feature importance plots (optional)

## Important Scope Limitation: Prediction vs. Prescription

This model predicts **where governments WILL designate** protected areas based on
historical patterns — NOT **where they SHOULD designate** them for maximum
conservation impact. These are different questions:

- **This model (descriptive):** Learns that governments protect cheap, remote,
  biodiverse land near existing PAs — because that's what they've historically done.
- **Conservation optimization (prescriptive):** Would maximize biodiversity
  protected per dollar, accounting for threat, connectivity, and irreplaceability.

The model implicitly learns the biases of past decisions (e.g., protecting "easy"
land rather than the most ecologically critical land). This is a feature, not a bug —
for transition-risk analysis (investors, policymakers) you want to know what WILL
happen. But for conservation planning, the gap between "where protection goes" and
"where it's most needed" is itself a finding worth discussing in the paper.

## Potentially Missing Variables

Features that could improve predictive power but are not currently included:

- **Land tenure / ownership** — Public land is far easier to designate than private.
  Source: national cadastral data
- **Indigenous territories** — Many new PAs overlap indigenous lands (political path
  of least resistance). Source: RAISG, LandMark
- **Carbon stocks** — REDD+ and carbon markets increasingly drive designation.
  Source: ESA CCI Biomass, GlobBiomass
- **Species richness / endemism** — Direct conservation value, more targeted than
  the GSN biodiversity proxy. Source: IUCN Red List, GBIF
- **PA network connectivity** — Corridor gaps between existing PAs are high-priority
  targets. Source: derived from WDPA network analysis
- **Land economic value** — Agricultural rent, mining concessions; high-value land
  resists protection. Source: FAO, national data
- **International commitments** — 30x30 pledge creates political pressure to
  designate. Source: CBD national pledges
- **NGO presence / funding** — Areas with active conservation organizations get
  designated faster. Source: IUCN, WCS, WWF activity data

## Improvement Priorities

### High Priority (Paper-Critical)
1. **Add software tests** — Zero automated tests exist. Add pytest suite with:
   - Unit tests for feature engineering (distance transforms, spatial smoothing)
   - Integration test on a small synthetic dataset
   - Regression test: same seed produces same metrics
2. **Ablation study** — Remove feature groups one at a time and measure impact.
   Shows what actually drives protection decisions (not just that the model works).
3. ~~**Second continent**~~ — **DONE.** USA pipeline mirrors SA structure (Model 2).
   Spatial CV Layer 3 provides cross-continental transfer evaluation (SA ↔ USA).
4. **Literature comparison** — Benchmark against published PA prediction methods.

### Medium Priority (Quality & Credibility)
5. **Pin all package versions** — `environment.yml` uses `python=3.12` not `3.12.1`.
   LightGBM version unspecified. Risk of irreproducible results over time.
6. **Extract shared utilities** — Spatial CV is now fully shared (`spatial_cv_core.py`).
   Preprocessing scripts for USA and SA are still copy-pasted — refactor into
   `shared/` to reduce duplication and bug risk.
7. **Highlight calibration in paper** — Many ML papers skip calibration.
   The reliability diagrams and Platt/isotonic correction are a strength.
8. **Explain right-censoring design** — The `LAST_LABEL_YEAR = 2019` constraint
   is methodologically rigorous. Make this prominent in the methods section.

### Medium-High Priority (Paper Differentiation)
9. **Bayesian network interpretability layer** — Train a BN on the top 10-15 features
   (discretized) as a post-hoc explanation of LightGBM's learned relationships.
   Produces a DAG showing dependency structure (e.g., "biodiversity → designation is
   mediated by remoteness") and conditional probability tables readable by policymakers.
   Addresses the "black box" critique without sacrificing predictive accuracy.
   Use `pgmpy` library. See `docs/technical_guide.md` Section 11 for full details.
10. ~~**Forward prediction to 2030**~~ — **DONE.** Full `8_forward/` pipeline built:
    deployment model trained on 2001–2019, calibrated batch inference producing
    `forward_scored_2024.parquet`, BAU / moderate / 30×30 scenario maps, per-country
    coverage table, economic exposure in top risk zones, false-positive spatial analysis,
    hotspot zoom-in figures, and backtesting validation (2009→2014, 2011→2016 windows).
    Temporal framing: 2024 data = start of 2025; 5-year lookahead ends 2030, consistent
    with 30×30 policy target. Moderate scenario is region-adaptive (midpoint between
    current coverage and 30%). All logic shared via `scripts/regions/shared/forward/`.

### Lower Priority (Polish)
11. **Consistent logging** — Replace mixed `print()`/`logging.info()` with unified logging.
12. **Break up monolithic files** — The 1,600-line training script should be split
    into data loading, training, and evaluation modules.

## Development Commands

```bash
# Setup environment
conda env create -f environment.yml
conda activate pa3030

# Run pipeline stages (South America continental example)
python scripts/regions/south_america/4_tuning/model1_tuning_lgbm
python scripts/regions/south_america/5_training/model1_splits
python scripts/regions/south_america/5_training/model1_LGBM
python scripts/regions/south_america/6_evaluation/calibrate_1
python scripts/regions/south_america/6_evaluation/benchmark_1
python scripts/regions/south_america/7_results/model1_results

# Same structure for USA (model2_*) and SE Asia (model3_*)
python scripts/regions/se_asia/4_tuning/model3_tuning_lgbm
python scripts/regions/se_asia/5_training/model3_LGBM
python scripts/regions/se_asia/6_evaluation/calibrate_3
python scripts/regions/se_asia/6_evaluation/benchmark_3
python scripts/regions/se_asia/7_results/model3_results

# Spatial CV — Layer 2 (after step 5 training; fast, run locally)
python scripts/regions/south_america/6_evaluation/spatial_CV_2
python scripts/regions/usa/6_evaluation/spatial_CV_2
python scripts/regions/se_asia/6_evaluation/spatial_CV_2

# Spatial CV — Layer 1 LOBO (SLURM array jobs on Euler)
# outputs go to outputs/{region}/results/spatial_cv/{lgbm,rf}/
python scripts/regions/south_america/6_evaluation/spatial_CV_1 --list-biomes
sbatch slurm/south_america/spatial_CV_1.slurm              # SA LGBM, all folds
sbatch slurm/south_america/spatial_CV_1_rf.slurm           # SA RF, all folds
sbatch slurm/usa/spatial_CV_1.slurm                        # USA LGBM, all folds
sbatch slurm/usa/spatial_CV_1_rf.slurm                     # USA RF, all folds
sbatch slurm/se_asia/spatial_CV_1.slurm                    # SE Asia LGBM, all folds
sbatch slurm/se_asia/spatial_CV_1_rf.slurm                 # SE Asia RF, all folds

# After all folds complete, aggregate:
sbatch slurm/south_america/spatial_CV_1_aggregate.slurm
sbatch slurm/south_america/spatial_CV_1_aggregate_rf.slurm
sbatch slurm/usa/spatial_CV_1_aggregate.slurm
sbatch slurm/usa/spatial_CV_1_aggregate_rf.slurm
sbatch slurm/se_asia/spatial_CV_1_aggregate.slurm
sbatch slurm/se_asia/spatial_CV_1_aggregate_rf.slurm

# Spatial CV — Layer 3 cross-continental transfer
sbatch slurm/south_america/spatial_CV_3.slurm
sbatch slurm/se_asia/spatial_CV_3.slurm

# Forward prediction pipeline (8_forward/) — same for all three regions
# Stage 0a: Coverage baseline
python scripts/regions/south_america/8_forward/1_forward_coverage_baseline.py
# Stages 0b + 1 + 2: Deployment model + features + inference (Euler)
sbatch slurm/south_america/forward_deployment.slurm
# Stage 0c: Backtesting
sbatch slurm/south_america/forward_backtest.slurm
# Stage 3: Results
python scripts/regions/south_america/8_forward/4_forward_results.py

# Replace south_america with usa or se_asia for other regions

# Run tests
pytest tests/test_pipeline.py -v
```

## Project Structure

```
/
├── scripts/regions/
│   ├── south_america/            # Model 1 — full 8-stage pipeline
│   │   ├── 1_extraction/         # 11 GEE + API extraction scripts
│   │   ├── 2_preprocessing/      # 11 preprocessing scripts
│   │   ├── 3_merging/            # merge + feature_engineering
│   │   ├── 4_tuning/             # model1_tuning_lgbm, model1_tuning_rf
│   │   ├── 5_training/           # model1_LGBM, model1_RF + deployment + splits + best_params.json
│   │   ├── 6_evaluation/         # calibrate_1, benchmark_1, spatial_CV_1/2/3 + aggregate
│   │   ├── 7_results/            # model1_results
│   │   ├── 8_forward/            # Forward predictions to 2030
│   │   │   ├── 1_forward_coverage_baseline.py
│   │   │   ├── 2_forward_features.py
│   │   │   ├── 3_forward_predict.py
│   │   │   ├── 4_forward_results.py
│   │   │   └── 5_forward_backtest.py
│   │   ├── colombia/             # Legacy sub-region (non-standard structure)
│   │   │   ├── preprocessing/    # backbone, export, features
│   │   │   ├── 3_merging/
│   │   │   ├── 4_ml/             # splits, training, evaluation, results
│   │   │   └── visualisations/
│   │   └── embeddings/           # Satellite embedding experiments (partial)
│   ├── usa/                      # Model 2 — mirrors SA structure (incl. 8_forward/)
│   ├── se_asia/                  # Model 3 — mirrors SA structure (incl. 8_forward/)
│   ├── tropical_africa/          # Planned (empty)
│   └── shared/                   # All shared utility code
│       ├── tuning/               # Optuna HPO runner, CV, search spaces, artifacts
│       ├── training/             # Training utilities
│       ├── evaluation/           # benchmark_core, calibration_core, spatial_cv_core,
│       │                         # spatial_cv_visualise, select_lobo_biomes
│       ├── results/              # results_core, boundaries, io, runner, config
│       ├── forward/              # coverage_core, features_core, predict_core,
│       │                         # results_core, backtest_core, config, runner
│       └── 1_preprocessing/      # economic_value (shared preprocessing)
├── slurm/                        # SLURM job scripts (~116 files across 3 regions)
│   ├── south_america/            # SA jobs: tuning, training, spatial_CV, forward
│   ├── usa/                      # USA jobs: same structure as SA
│   ├── se_asia/                  # SE Asia jobs: same structure as SA
│   ├── create_env.sh
│   ├── euler_run_instructions.md
│   └── submit_spatial_cv.sh
├── outputs/                      # Figures, tables, metrics, predictions
│   ├── south_america/            # results/, figures/, tables/, colombia/, embeddings/
│   ├── usa/                      # results/, figures/, tables/
│   └── se_asia/                  # results/, figures/, tables/
├── tests/                        # pytest suite (test_pipeline.py)
├── environment.yml
└── README.md
```
