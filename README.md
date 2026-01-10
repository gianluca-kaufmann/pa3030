# Protected Area Designation Prediction using Machine Learning

A large-scale machine learning pipeline for predicting future protected area establishment across South America using historical patterns, environmental features, and socio-economic indicators.

**For a high-level project overview, see [`context.md`](context.md).**

---

## Repository Structure

```
├── scripts/
│   ├── data extraction/       # Google Earth Engine and API exporters
│   ├── preprocessing/          # Format harmonization, storage optimization
│   ├── merging/                # Panel dataset construction
│   ├── visualisations/         # Exploratory plots and data validation
│   ├── ML/                     # Model 1: Full South America pipeline
│   │   ├── ml_preprocessing/   # Train/test splits, feature engineering
│   │   ├── training/           # LightGBM, Random Forest, tuning
│   │   ├── evaluation/         # Temporal CV, spatial CV, calibration
│   │   └── results/            # Metrics, maps, and visualizations
│   ├── colombia/               # Model C: Colombia-only (rapid validation)
│   │   ├── export/
│   │   ├── preprocessing/
│   │   ├── merge/
│   │   └── ML/
│   └── embeddings/             # Model E: Satellite embedding features
│       ├── export/
│       ├── preprocessing/
│       ├── merge/
│       └── ML/
├── outputs/
│   ├── Results/                # Model predictions, metrics, maps
│   │   ├── ml_models/          # JSON metrics, text logs
│   │   ├── results_model1_lgbm/
│   │   ├── results_modelC_lgbm/
│   │   ├── results_modelC_rf/
│   │   └── results_modelC_brf/
│   └── Tables/                 # Summary statistics, validation reports
├── environment.yml             # Conda environment specification
├── context.md                  # High-level project overview (start here!)
└── README.md                   # This file
```

---

## Pipeline Overview

The pipeline consists of the following stages:

1. **Data Extraction** – Export datasets from Google Earth Engine and external APIs to GeoTIFF rasters
2. **Preprocessing** – Harmonize external datasets (format conversion, reprojection, optimization)
3. **Visualizations** – Generate exploratory plots to validate spatial alignment and data quality
4. **Merging** – Build modeling panels by stacking rasters and constructing pixel-year observations
5. **ML Preprocessing** – Split data into train/validation/test sets, engineer spatial features, prepare lookahead targets
6. **Model Training** – Train models with hyperparameter tuning (LightGBM, Random Forest, Balanced RF)
7. **Evaluation** – Temporal CV, spatial CV, benchmarking, probability calibration
8. **Results Generation** – Generate metrics tables, PR curves, risk maps, probability maps

---

## Model Variants

| Model | Region | Features | Dataset Size | Use Case |
|-------|--------|----------|--------------|----------|
| **Model 1** | South America | Standard features | ~350M pixel-years | Full-scale production |
| **Model C** | Colombia | Standard features | ~7M pixel-years | Rapid validation, development |
| **Model E** | South America | Satellite embeddings | ~20M pixel-years | Embedding experiments |

## Algorithms

| Algorithm | Class Balance | Use Case |
|-----------|---------------|----------|
| **LightGBM** | Weighted (primary) | Primary production model |
| **Random Forest** | Imbalanced | Tree ensemble baseline |
| **Balanced RF** | Balanced sampling | Class balance experiment |

## Data Splits

- **Training:** 2000–2014 (pixels unprotected at time t, with 5-year lookahead labels)
- **Validation:** 2015–2017 (for early stopping and hyperparameter selection)
- **Test:** 2018–2019 (right-censored at 2019 to ensure complete 5-year lookahead)

**Right-censoring:** `LAST_LABEL_YEAR = 2024 - 5 = 2019` ensures all labels have complete 5-year observation windows.

---

## Data Characteristics

| Property | Value |
|----------|-------|
| **Spatial resolution** | ~1 km × 1 km |
| **CRS** | EPSG:3857 (Web Mercator) |
| **Temporal coverage** | 2000–2024 (annual observations) |
| **Region** | South America (continental extent) |
| **Target variable** | `transition_01_win5` (5-year lookahead) |
| **Features** | ~60–80 (static, dynamic, spatial context) |
| **Class balance** | ~0.3–0.5% positive (extreme imbalance) |
| **Format** | GeoTIFF (rasters), Parquet (panels) |

---

## Outputs

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

---

## Dependencies

Core dependencies:

- **Geospatial:** rasterio, geopandas, shapely, pyproj, fiona
- **ML:** scikit-learn, lightgbm, imbalanced-learn
- **Data:** pandas, numpy, pyarrow, scipy
- **Visualization:** matplotlib, seaborn
- **Remote sensing:** earthengine-api
- **Experiment tracking:** wandb (optional)

See `environment.yml` for complete environment specification.

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@mastersthesis{kaufmann2025protectedareas,
  author = {Kaufmann, Gian-Luca},
  title = {Predicting Future Protected Area Designations with Machine Learning},
  school = {ETH Zurich},
  year = {2025},
  type = {Master's Thesis}
}
```

---

## Contact

**Gian-Luca Kaufmann**  
ETH Zurich  
Email: (see thesis manuscript)

For questions, collaboration, or reuse inquiries.

---

## License

This project is part of academic research at ETH Zurich. For reuse permissions, please contact the author.

---

**Related Files:**
- [`context.md`](context.md) – High-level project overview (read this first!)
- [`environment.yml`](environment.yml) – Conda environment specification
