# PA3030 - Protected Area Designation Prediction

## Overview

PA3030 is a large-scale, reproducible machine learning pipeline that predicts the probability of a given 1 km x 1 km pixel becoming newly designated as a protected area within a 5-year window. 

The core question: **"Given historical patterns of protected-area establishment, which locations are most likely to be designated as protected areas in the future?"**

## Use Cases

- Conservation planning and prioritization
- Transition-risk analysis for investors and policymakers
- Academic research on land-use change and environmental policy

## Pipeline Architecture

```
1. Data Extraction (1_extraction/)
   -> Export from Google Earth Engine + external APIs to GeoTIFF rasters

2. Preprocessing (2_preprocessing/)
   -> Format harmonization, reprojection, storage optimization

3. Merging (3_merging/)
   -> Build modeling panels: pixel-year observations in Parquet format
   -> Feature engineering: spatial features, distance metrics, smoothed neighbors

4. ML Pipeline (4_ml/)
   -> 1_splits/   : Train/validation/test data splits
   -> 2_training/ : Model training with hyperparameter tuning
   -> 3_evaluation/: Temporal CV, spatial CV, benchmarking, calibration
   -> 4_results/  : Generate metrics, visualizations, risk maps
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

| Model | Region | Scale | Description |
|-------|--------|-------|-------------|
| Model 1 | South America (continental) | ~350M pixel-years | Full-scale production model |
| Model C | Colombia (sub-region) | ~7M pixel-years | Rapid validation & development |
| Model E | South America | ~20M pixel-years | Satellite embedding experiments |

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
Region-wise (country/biome) breakdown of metrics to check whether
performance holds across geographies or is driven by specific areas.

### Outputs
- Benchmark JSON (structured metrics for comparison)
- PR curves, cumulative gains charts, reliability diagrams (PNG + PDF)
- Risk maps (top K% predicted pixels vs observed PAs)
- Probability maps (continuous predictions across the continent)
- LaTeX metrics tables (paper-ready)
- SHAP feature importance plots (optional)

## Improvement Priorities

### High Priority (Paper-Critical)
1. **Add software tests** — Zero automated tests exist. Add pytest suite with:
   - Unit tests for feature engineering (distance transforms, spatial smoothing)
   - Integration test on a small synthetic dataset
   - Regression test: same seed produces same metrics
2. **Ablation study** — Remove feature groups one at a time and measure impact.
   Shows what actually drives protection decisions (not just that the model works).
3. **Second continent** — Run the pipeline on USA or another region.
   Transforms a regional result into a generalizable finding.
4. **Literature comparison** — Benchmark against published PA prediction methods.

### Medium Priority (Quality & Credibility)
5. **Pin all package versions** — `environment.yml` uses `python=3.12` not `3.12.1`.
   LightGBM version unspecified. Risk of irreproducible results over time.
6. **Extract shared utilities** — USA and South America preprocessing are copy-pasted.
   Refactor into a shared `utils/` module to reduce duplication and bug risk.
7. **Highlight calibration in paper** — Many ML papers skip calibration.
   The reliability diagrams and Platt/isotonic correction are a strength.
8. **Explain right-censoring design** — The `LAST_LABEL_YEAR = 2019` constraint
   is methodologically rigorous. Make this prominent in the methods section.

### Lower Priority (Polish)
9. **Consistent logging** — Replace mixed `print()`/`logging.info()` with unified logging.
10. **Break up monolithic files** — The 1,600-line training script should be split
    into data loading, training, and evaluation modules.
11. **Forward prediction to 2030** — Generate "where will protection happen next?"
    maps for a compelling paper figure with policy relevance.

## Development Commands

```bash
# Setup environment
conda env create -f environment.yml
conda activate pa3030

# Run pipeline stages (example for Colombia)
python scripts/regions/south_america/colombia/3_merging/merge_all.py
python scripts/regions/south_america/colombia/4_ml/1_splits/create_splits.py
python scripts/regions/south_america/colombia/4_ml/2_training/train_lgbm.py
python scripts/regions/south_america/colombia/4_ml/3_evaluation/evaluate.py
```

## Project Structure

```
pa3030/
├── scripts/regions/
│   ├── south_america/        # Primary region
│   │   ├── 1_extraction/
│   │   ├── 2_preprocessing/
│   │   ├── 3_merging/
│   │   ├── 4_ml/
│   │   ├── colombia/         # Sub-region pipeline
│   │   └── embeddings/       # Satellite embedding features
│   ├── usa/                  # Template for expansion
│   ├── se_asia/              # Planned
│   └── tropical_africa/      # Planned
├── outputs/                  # Figures, tables, metrics, predictions
├── slurm/                    # SLURM job scripts for Euler cluster
├── environment.yml
└── README.md
```
