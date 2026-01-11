# Predicting Future Protected Area Designations with Machine Learning

## 1. Motivation and Research Goal

Global biodiversity policy is entering an implementation-critical phase. Under the 30×30 target, governments aim to protect at least 30% of land and sea by 2030. While this target is well-defined at the global level, where future protected areas (PAs) will be established remains highly uncertain.

This project develops a large-scale, reproducible machine-learning pipeline to predict the probability that a given 1 km × 1 km pixel becomes newly protected within a 5-year window, conditional on its environmental, socio-economic, and spatial context.

**Outputs:**
- Probability maps of future PA establishment
- Ranked risk surfaces for conservation prioritization
- Quantitative evaluation of temporal and spatial generalization

**Intended use:**
- Conservation planning and prioritization
- Transition-risk analysis for investors and policymakers
- Academic research on land-use change and environmental policy

## 2. Core Research Question

**Given historical patterns of protected-area establishment, which locations are most likely to be designated as protected areas in the future?**

This is framed as a binary transition prediction problem:
- **Unit of observation:** pixel × year
- **Target variable:** `transition_01_win5` (1 if unprotected pixel becomes protected within next 5 years, else 0)
- **Risk-set logic:** Only pixels with `WDPA_prev == 0` (unprotected at time t) are eligible to transition, avoiding mechanical prediction and target leakage

## 3. Spatial and Temporal Scope

**Geography:**
- Primary region: South America (~350M pixel-years)
- Reference region: Colombia (used for rapid development and validation)
- Resolution: ~1 km × 1 km grid (EPSG:3857)

**Time:**
- Panel: 2000–2024 (annual observations)
- Training window: 2001–2014 (year 2000 excluded as it contains no transition events)
- Validation window: 2015–2017
- Test window: 2018–2019 (right-censored at 2019 to ensure complete 5-year lookahead)

The pipeline is designed to scale to forward-looking predictions through 2030.

## 4. Data Sources

**Target Variable:**
- WDPA (World Database on Protected Areas) – binary indicator of protection status per pixel and year

**Features (~60–80 total):**

*Static (time-invariant):*
- Elevation, slope, climate normals (WorldClim)
- Global Safety Net (GSN) biodiversity importance
- Distance to infrastructure (roads, power plants, oil & gas fields)

*Dynamic (annual):*
- Population density (GPW)
- Night-time lights (VIIRS)
- Vegetation indices (NDVI)
- Deforestation events
- Wildfire occurrence
- Land-cover dynamics

*Spatial context:*
- Distance to nearest existing PA
- Multi-scale spatial smoothing (4×4, 16×16, 64×64 pixel kernels)

All rasters are preprocessed, aligned, and optimized before panel construction.

## 5. Pipeline Architecture

```
1. Data Extraction
   └─> Google Earth Engine exports + external datasets
   
2. Preprocessing
   └─> Format harmonization, reprojection, storage optimization
   
3. Merging
   └─> Panel construction (pixel-year observations, Parquet format)

4. Feature engineering
   └─> spatial features (Distance, Smoothed neighbouring averages)
   
5. ML Preprocessing
   └─> Train/validation/test splits
   
6. Model Training
   └─> Hyperparameter tuning → Final model training → Out-of-fold predictions
   
7. Evaluation
   └─> Temporal CV, spatial CV, benchmarking, probability calibration
   
8. Results Generation
   └─> Metrics tables, PR curves, risk maps, probability maps
```

## 6. Model Variants

The project includes three complementary modeling approaches:

**Model 1 (Full):** South America dataset with standard features (~350M pixel-years)

**Model C (Colombia):** Colombia-only subset for rapid validation

**Model E (Embeddings):** Uses satellite image embeddings from AlphaEarth Foundations

For each variant, the following algorithms are evaluated:
- **LightGBM** (primary model) – scales to hundreds of millions of rows, handles extreme imbalance
- **Random Forest** (imbalanced) – baseline tree ensemble
- **Balanced Random Forest** (BRF) – class-balanced variant

## 7. Machine Learning Setup

**Problem type:** Binary classification with extreme class imbalance (≈0.3–0.5% positives)

**Imbalance handling:**
- Class weighting: `scale_pos_weight = n_neg / n_pos` (computed per split)
- Time-based sample weighting: Later years receive higher weight (linear 0.5 → 1.0)
- No resampling (preserves spatial structure)

**Hyperparameter tuning:**
- Randomized search over relevant parameter space
- 3-fold temporal cross-validation
- Early stopping on validation set

**Final model training:**
- Average `best_iteration` from CV folds
- No early stopping on final model
- Out-of-fold predictions for calibration

## 8. Evaluation Protocol

**8.1 Temporal Generalization (Primary)**

Rolling temporal splits: train strictly on the past, test on future unseen years.

*This answers:* **Can the model predict future PA expansion based only on historical information?**

**8.2 Spatial Generalization (Secondary)**

Spatial folds via coordinate-based clustering to test regional transferability.

**8.3 Metrics**

Given extreme imbalance, standard accuracy is meaningless. Focus metrics:
- **ROC-AUC** – overall discrimination
- **PR-AUC** – precision-recall tradeoff under imbalance
- **Precision@K%** – especially top 1% (operational prioritization)
- **Brier Score** – calibration quality
- **Reliability diagrams** – calibration assessment

These metrics directly correspond to real-world prioritization tasks.

## 9. Probability Calibration

Raw model probabilities are post-hoc calibrated using:
- **Platt scaling** (logistic calibration)
- **Isotonic regression** (non-parametric calibration)
- Cross-validated out-of-fold predictions
- Probability clipping for numerical stability

Ensures predicted probabilities are interpretable as true probabilities (e.g., 0.2 ≈ 20% risk).

## 10. Outputs

The pipeline produces thesis-ready outputs:

**Metrics:**
- JSON files with full metrics (ROC-AUC, PR-AUC, Brier Score, Precision@K)
- CSV and LaTeX tables for reporting

**Visualizations:**
- Precision-Recall curves (PNG + PDF)
- Reliability diagrams (calibration plots)
- High-resolution risk maps showing top K% predictions vs. observed establishments
- Probability maps (continuous predictions across study region)
- Top-1% diagnostic plots

**Predictions:**
- Scored Parquet files with pixel-level predictions (calibrated and uncalibrated)
- Ranked pixel lists for prioritization

All outputs stored in `outputs/Results/` and reproducible from scripts in `scripts/`.

## 11. Reproducibility and Design Philosophy

This project is built with:
- Deterministic pipelines (fixed random seeds)
- Environment-agnostic paths (via `get_repo_root()`)
- Clear separation of data / code / outputs
- Comprehensive logging and validation checks

**Goal:** An external researcher can fully understand, audit, and extend the project without direct supervision.

## 12. Key Technical Features

**Right-censoring protection:**
- All data splits filtered to `LAST_LABEL_YEAR = 2019` (WDPA_LAST_YEAR 2024 - LOOKAHEAD_YEARS 5)
- Ensures complete 5-year lookahead information for all labels

**Risk-set filtering:**
- Only pixels unprotected at time t are eligible for prediction
- Implemented via `WDPA_prev == 0` filter

**Temporal weighting:**
- Recent training years receive higher weight (linear interpolation from 0.5 to 1.0)
- Accounts for non-stationarity in PA establishment patterns

**Batch processing:**
- Data loaded in batches for memory efficiency
- Supports very large datasets on memory-constrained systems

## 13. Intended Use and Limitations

**Intended Use:**
- Strategic conservation planning ("where is protection likely next?")
- Business and Investment strategy (Transition Risks)
- Scenario analysis and policy counterfactuals
- Research on environmental policy dynamics

**Limitations:**
- Predictive, not causal
- Depends on historical patterns of designation
- Political shocks and legal changes not fully captured
- Predictions assume historical trends continue

These limitations are explicitly discussed in the thesis.

## 14. Repository Structure

```
├── scripts/
│   ├── data extraction/      # GEE and API exporters
│   ├── preprocessing/         # Format harmonization
│   ├── merging/               # Panel construction
│   ├── ML/                    # Model 1 (South America)
│   │   ├── ml_preprocessing/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── results/
│   ├── colombia/              # Model C (Colombia subset)
│   │   └── ML/
│   ├── embeddings/            # Model E (satellite embeddings)
│   │   └── ML/
│   └── visualisations/        # Exploratory plots
├── outputs/
│   ├── Results/               # Model outputs, predictions, metrics
│   └── Tables/                # Summary statistics
└── README.md                  # Usage instructions
```

## 15. Roadmap

Planned:
- Forward prediction to 2030
- Political and institutional covariates
- Embedding-based satellite features at scale
- Neural network architectures
- Policy counterfactual scenarios

## 16. Contact

For questions, collaboration, or reuse:

**Gian-Luca Kaufmann**  
ETH Zurich  
Email: (see thesis manuscript)

---

*This file serves as the canonical high-level documentation of the project.*
