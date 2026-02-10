# Protected Area Designation Prediction using Machine Learning

A large-scale machine learning pipeline for predicting future protected area establishment across multiple geographic regions using historical patterns, environmental features, and socio-economic indicators. Currently implemented for South America (continental + Colombia sub-region), with planned extensions to the USA, South-East Asia, and Tropical Africa.

**For a high-level project overview, see [`context.md`](context.md).**

---

## Repository Structure

```
├── scripts/
│   ├── regions/                         # Geography-specific pipelines
│   │   ├── south_america/               # Primary region
│   │   │   ├── 1_extraction/            # GEE export scripts
│   │   │   ├── 2_preprocessing/         # Format harmonization
│   │   │   ├── 3_merging/               # Panel construction
│   │   │   ├── 4_ml/                    # Continental ML pipeline
│   │   │   │   ├── splits/
│   │   │   │   ├── training/ (+ tuning/)
│   │   │   │   ├── evaluation/
│   │   │   │   └── results/
│   │   │   ├── colombia/                # Country sub-pipeline
│   │   │   │   ├── preprocessing/
│   │   │   │   ├── 3_merging/
│   │   │   │   └── 4_ml/
│   │   │   └── embeddings/              # Satellite embedding pipeline
│   │   │       ├── 1_extraction/
│   │   │       ├── 2_preprocessing/
│   │   │       ├── 3_merging/
│   │   │       └── 4_ml/
│   │   ├── usa/                         # (template for new regions)
│   │   ├── se_asia/
│   │   └── tropical_africa/
│   └── visualisations/                  # Feature-level EDA (cross-region)
├── outputs/
│   ├── south_america/
│   │   ├── figures/
│   │   ├── tables/
│   │   ├── results/ (lgbm/, rf/, ml_models/)
│   │   ├── colombia/ (figures/, tables/, results/)
│   │   └── embeddings/ (figures/, tables/, results/)
│   ├── usa/
│   └── ...
├── slurm/
│   ├── south_america/ (+ colombia/, embeddings/)
│   └── ...
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

Data structure on the Euler cluster:

```
$SCRATCH/data/
├── shared/                              # Cross-region, global reference data
│   ├── country_iso3.tif
│   ├── country_iso3_mapping.json
│   ├── policy_table.parquet
│   ├── DPI/
│   ├── VDem/
│   └── WGI/
│
├── south_america/                       # ── PRIMARY REGION ──
│   ├── ready/                           # SA-wide preprocessed rasters
│   │   ├── backbone/
│   │   ├── WDPA/
│   │   ├── WorldClim/
│   │   ├── elevation/
│   │   ├── slope/
│   │   ├── NDVI/
│   │   ├── GPW/
│   │   ├── HNTL/
│   │   ├── GSN/
│   │   ├── landcover/
│   │   ├── deforestation/
│   │   ├── wildfire/
│   │   ├── oil_gas/
│   │   ├── powerplants/
│   │   └── road_infrastructure/
│   │
│   ├── ml/                              # SA continental ML data
│   │   ├── merged_panel_2000_2024.parquet
│   │   ├── merged_panel_final.parquet
│   │   ├── main/
│   │   │   ├── train_win5.parquet
│   │   │   ├── earlystop_win5.parquet
│   │   │   ├── test_win5.parquet
│   │   │   └── merged_panel_final_win5.parquet
│   │   ├── robustness/                  # (same split files)
│   │   ├── models/
│   │   │   ├── main/
│   │   │   └── robustness/
│   │   └── tuning/
│   │
│   ├── colombia/                        # Country sub-pipeline (uses SA ready/)
│   │   └── ml/
│   │       ├── merged_panel_colombia_final.parquet
│   │       ├── main/
│   │       ├── robustness/
│   │       ├── models/
│   │       │   └── main/
│   │       └── tuning/
│   │
│   └── embeddings/                      # Alternative feature pipeline
│       ├── ready/
│       │   ├── raw_tiles/               # Original embedding tiles
│       │   ├── embeddings_aligned/      # Aligned to backbone grid
│       │   └── wdpa_aligned/            # WDPA aligned for embeddings
│       └── ml/
│           ├── models/
│           └── tuning/
│
├── usa/                                 # ── NEW REGION (same structure as SA top-level) ──
│   ├── ready/                           # USA preprocessed rasters
│   │   ├── backbone/
│   │   ├── WDPA/
│   │   ├── WorldClim/
│   │   ├── elevation/
│   │   ├── NDVI/
│   │   ├── GPW/
│   │   ├── HNTL/
│   │   ├── landcover/
│   │   ├── deforestation/
│   │   ├── wildfire/
│   │   ├── oil_gas/
│   │   ├── powerplants/
│   │   └── road_infrastructure/
│   │
│   └── ml/                              # USA ML data
│       ├── merged_panel_final.parquet
│       ├── main/
│       │   ├── train_win5.parquet
│       │   ├── earlystop_win5.parquet
│       │   └── test_win5.parquet
│       ├── robustness/
│       ├── models/
│       │   ├── main/
│       │   └── robustness/
│       └── tuning/
│
├── se_asia/                             # ── NEW REGION (same structure) ──
│   ├── ready/                           # SE Asia preprocessed rasters
│   │   ├── backbone/
│   │   ├── WDPA/
│   │   ├── WorldClim/
│   │   ├── ...                          # (same feature set as above)
│   │   └── road_infrastructure/
│   │
│   └── ml/
│       ├── merged_panel_final.parquet
│       ├── main/
│       ├── robustness/
│       ├── models/
│       └── tuning/
│
├── tropical_africa/                     # ── NEW REGION (same structure) ──
│   ├── ready/
│   │   ├── backbone/
│   │   ├── WDPA/
│   │   ├── ...                          # (same feature set as above)
│   │   └── road_infrastructure/
│   │
│   └── ml/
│       ├── merged_panel_final.parquet
│       ├── main/
│       ├── robustness/
│       ├── models/
│       └── tuning/
│
└── logs/
```

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

All outputs stored in `outputs/{region}/` and reproducible from scripts in `scripts/regions/{region}/`.

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
