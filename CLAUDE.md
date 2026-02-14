# PA3030 - Protected Area Designation Prediction

## Overview

PA3030 is a large-scale, reproducible machine learning pipeline that predicts the probability of a given 1 km x 1 km pixel becoming newly designated as a protected area within a 5-year window. It was developed as a Master's thesis project at ETH Zurich by Gian-Luca Kaufmann.

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
