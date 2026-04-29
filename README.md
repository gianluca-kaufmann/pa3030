# PA3030: Predicting Protected Area Expansion and Transition Risk under the 30×30 Target

This repository contains the code for the Master's thesis *Predicting Protected Area
Expansion and Transition Risk under the 30×30 Target* (Gian-Luca Kaufmann, ETH Zurich,
April 2026). It builds large-scale geospatial machine-learning panels and predicts the
probability that any unprotected 1 km² land cell will receive formal WDPA designation
within a five-year window.

The approach is descriptive rather than prescriptive: it asks where protected areas are
likely to be designated if future expansion continues historical spatial patterns. This is
relevant for studying 30×30 implementation pathways, nature-related transition risk, and
the gap between likely designations and biodiversity conservation priorities.

## Key Results

All results are out-of-sample on the 2017–2019 test period. Random Forest is the primary
model; LightGBM confirms the directional findings. (The United States is a special case:
apparent performance is strongly affected by geographic concentration of designations and
should be read mainly as a spatial ranking signal rather than calibrated absolute risk.)
In the table, **Precision@1%** is event-level (unique designated pixels) precision in the
top-1% risk tier, and **Lift@1%** is pixel-year lift relative to the baseline rate.

| Region | Model | ROC-AUC | PR-AUC | Precision@1% | Lift@1% |
|--------|-------|---------|--------|--------------|---------|
| South America | RF | 0.910 | 0.548 | 97.0% | 67.6× |
| United States | RF | 0.999 | 0.538 | 6.0% | 99.6× |
| Southeast Asia | RF | 0.987 | 0.571 | 91.9% | 82.6× |

The top 1% of predicted pixels in South America captures 69.1% of all pixels that were
newly designated during the test period. In Southeast Asia, the top 1% captures 65.0% of
unique designated pixels at 91.9% precision. Forward projections under business-as-usual
designation rates suggest South America would reach 30% coverage only after approximately
68 additional years from 2024; the United States and Southeast Asia require more than
170 additional years at current rates.

## Repository Status

| Model | Region | Panel size | Role |
|-------|--------|------------|------|
| Model 1 | South America | ~413M pixel-years | Primary thesis model |
| Model 2 | United States | ~200M pixel-years | Cross-region replication |
| Model 3 | Southeast Asia | ~100M pixel-years | Third regional replication |
| Model C | Colombia | ~7M pixel-years | Legacy development sub-region |

Non-thesis / experimental work:

- **Model E (South America embeddings)**: partial satellite-embedding experiment (~20M pixel-years). It is not on the thesis critical path and is not a complete, end-to-end pipeline.

Source code, SLURM job scripts, tests, generated figures, tables, and selected result
artifacts are versioned. Large raw rasters, intermediate Parquet panels, trained model
binaries, and credentials are not versioned.

## Method

### Spatial and temporal scope

- Spatial resolution: approximately 1 km × 1 km (EPSG:3857).
- Temporal coverage: annual observations, 2000–2024.
- Prediction target: new WDPA designation within the next five years (binary).
- Risk set: only pixels that were unprotected in the previous year are eligible.
- Right-censoring: labels are only included through 2019 because a pixel observed in 2020
  or later does not yet have a complete five-year lookahead in the 2024 WDPA snapshot.

### Temporal splits

- Training: 2001–2016.
- Test: 2017–2019.
- Forward deployment model (for 2025–2030 projections): trained on 2001–2019, scoring
  2024 covariates as the start-of-2025 state.

### Models and calibration

Random Forest (primary) and LightGBM are trained on the same panel and features.
Class imbalance (~0.3–0.5% positive rate) is handled via balanced subsampling (RF) or
`scale_pos_weight` (LightGBM). Raw scores are post-hoc calibrated using Platt scaling
(logistic regression on logit-transformed predictions), fitted per fold and averaged.
Random Forest is reported as the primary model because it achieves the best
precision-recall performance across all three regions.

### Features

Features are organised into six groups:

- **Existing conservation context:** distance to nearest PA boundary (lagged one year).
- **Environmental and ecological conditions:** NDVI, deforestation and forest cover,
  wildfire occurrence, elevation and slope, 19 WorldClim bioclimatic variables, land
  cover (MODIS IGBP), five Global Safety Net conservation-priority layers (biodiversity,
  wilderness, climate stabilisation, wildlife corridors, WWF ecoregions).
- **Human pressure and infrastructure:** population density, night-time lights (HNTL),
  road network density, energy and extractive infrastructure (power plants, oil and gas).
- **Economic conditions:** agricultural land value (FAO gross production value allocated
  to cropland and pasture pixels).
- **Indigenous and community lands:** distance to formally recognised indigenous
  territories (LandMark).
- **Governance and institutional context:** WGI Government Effectiveness and Rule of Law,
  V-Dem Liberal Democracy Index, DPI executive ideology.

All time-varying features are lagged by one year to prevent temporal leakage.
Spatial smoothing at 4 km, 16 km, and 64 km neighbourhoods is applied to selected
features. Euclidean distance transforms are computed to PA boundaries, roads, power
plants, and oil and gas installations.

### Metrics

ROC-AUC, PR-AUC (primary), Precision@K%, Lift@K%, Brier score, Expected Calibration
Error (ECE), and Maximum Calibration Error (MCE). PR-AUC is the primary metric because it
is more informative than ROC-AUC under extreme class imbalance.

### Spatial generalisation

Three-layer framework:

1. **LOBO (Leave-One-Biome-Out):** train on all biomes except one, evaluate on the
   held-out biome. Tests whether the model generalises to unseen geographies.
2. **Biome-stratified metrics:** per-biome evaluation on the full trained model.
3. **Cross-continental transfer:** train on South America, evaluate on USA and/or
   Southeast Asia, and vice versa.

### Forward projections

Scenarios: business-as-usual (BAU), moderate, and 30×30 (designation rate required to
hit 30% coverage by 2030). Backtesting validation uses 2009→2014 and 2011→2016 windows.
Forward results include per-country coverage tables, agricultural land area at risk of
designation, hotspot maps, and ecological priority overlap.

## Pipeline Layout

All three continental regions share the same eight-stage structure:

```text
scripts/regions/{south_america,usa,se_asia}/
  1_extraction/     Google Earth Engine and external data exports
  2_preprocessing/  Raster harmonisation, reprojection, and optimisation
  3_merging/        Raster stacking, panel construction, feature engineering
  4_tuning/         Optuna/randomised hyperparameter search
  5_training/       Split creation and final RF/LightGBM training
  6_evaluation/     Calibration, benchmarking, spatial CV
  7_results/        Tables, curves, maps, and comparison figures
  8_forward/        Deployment model, 2025–2030 prediction, scenarios, backtests
```

Shared implementation lives in:

```text
scripts/regions/shared/
  tuning/           Search spaces, temporal CV, tuning runner
  training/         Shared training utilities
  evaluation/       Benchmarking, calibration, spatial CV
  results/          Result aggregation and figure generation
  forward/          Forward prediction, coverage, backtesting, scenarios
  1_preprocessing/  Shared preprocessing utilities
```

Other important directories:

```text
slurm/              Euler HPC job scripts for each region and major stage
outputs/            Generated figures, tables, metrics, maps, and scored outputs
tests/              Lightweight pytest suite using synthetic data
docs/               Technical notes and runbooks
manuscript/         Thesis source and compiled manuscript artifacts
environment.yml     Conda environment specification
```

The Colombia pipeline is retained under `scripts/regions/south_america/colombia/`. It was
used for early development and has a non-standard `4_ml/` structure.

## Reproducing the Environment

```bash
conda env create -f environment.yml
conda activate pa3030
```

The environment uses Python 3.12. Key packages: Rasterio, GeoPandas, Shapely, PyProj,
Google Earth Engine, pandas, NumPy, PyArrow, scikit-learn, LightGBM, imbalanced-learn,
Optuna, SHAP, Matplotlib, Seaborn, and pytest.

Some stages require external data access, region-specific raster inputs, or Euler scratch
storage configured outside the repository.

## How to Run the Pipeline

The continental pipelines are organised identically for `south_america` (Model 1), `usa`
(Model 2), and `se_asia` (Model 3). In each region folder, scripts are grouped into stages
`1_extraction/` through `8_forward/`. Most full-resolution stages are designed for Euler
HPC via the provided `slurm/{region}/` job scripts.

### Start-to-finish stage order (complete)

Rebuilding the full pipeline from scratch requires access to the underlying datasets
described in the thesis (including Google Earth Engine exports and large raster sources).
From start to finish, the run order is:

1. **`1_extraction/` — raw exports**: export/collect source layers (primarily via Google
   Earth Engine and external APIs). Output: region-year GeoTIFF rasters (or equivalent
   raw layers) stored outside Git.
2. **`2_preprocessing/` — harmonise rasters**: reproject/resample to the common 1 km
   backbone grid, align to calendar years, and standardise formats. Output: aligned
   rasters ready for stacking.
3. **`3_merging/` — build modelling panels**: stack rasters into pixel–year Parquet,
   construct lags and eligibility/risk-set filters, compute spatial smoothing and
   distance transforms. Output: modelling panel(s) used by ML stages.
4. **`4_tuning/` — hyperparameter search**: Optuna/randomised search on the pre-test
   period with temporal CV. Output: best-parameter artifacts per algorithm.
5. **`5_training/` — final model training**: create temporal splits and train the final
   Random Forest and LightGBM models. Output: trained model artifacts and scored test
   predictions.
6. **`6_evaluation/` — calibration + validation**: fit Platt calibrators, compute test
   metrics, and run spatial generalisation (LOBO, biome-stratified, cross-region where
   applicable). Output: benchmark metrics, calibration diagnostics, spatial CV outputs.
7. **`7_results/` — paper-ready outputs**: aggregate results into figures/tables (PR
   curves, reliability diagrams, risk/probability maps, comparison figures). Output:
   `outputs/{region}/figures/` and `outputs/{region}/tables/`.
8. **`8_forward/` — 2025–2030 projections**: train a deployment model on the full
   labelled period, score 2024 covariates as start-of-2025 state, generate BAU/moderate/30×30
   scenarios, and run backtests. Output: forward scored Parquet(s), scenario maps, and
   coverage/exposure summaries.

In practice, full-resolution stages are typically executed on Euler via `sbatch slurm/{region}/`,
while some aggregation/visualisation steps can run locally.

## Testing

The software test suite uses small synthetic data and does not require the full geospatial
datasets:

```bash
pytest tests/test_pipeline.py -v
```

Tests cover distance transforms, spatial smoothing, WDPA lag construction, risk-set
filtering, right-censoring, temporal split integrity, class-imbalance weights, calibration
helpers, and grid reconstruction.

Model evaluation (metrics on held-out data) is separate from software testing and is
produced by `scripts/regions/*/6_evaluation/`, `7_results/`, and `8_forward/`.

## Outputs

Main outputs are written to `outputs/{region}/`:

- `results/model*_lgbm/` and `results/model*_rf/`: benchmark metrics, calibrated
  predictions, PR curves, calibration diagrams, and model diagnostics.
- `results/spatial_cv/`: leave-one-biome-out fold outputs.
- `results/spatial_generalisation/`: biome-stratified and cross-region generalisation
  metrics.
- `results/forward/`: 2025–2030 scored pixels, scenario outputs, coverage tables,
  exposure summaries, hotspot maps, and backtests.
- `figures/` and `tables/`: paper-ready maps, plots, and LaTeX/CSV tables.

## Data and Reproducibility

The repository is reproducible at the code and workflow level but is not self-contained at
the raw-data level. Re-running the full pipeline requires WDPA snapshots and other source
datasets described in the thesis, Google Earth Engine access for extraction scripts,
region-specific raster directories and Parquet panels (typically stored outside Git due to
size), and Euler or equivalent high-memory compute for full continental training and
spatial CV.

### Data availability (public repo)

Raw inputs and full intermediate panels are not included in this repository. If you would
like access to the data used in the thesis or details on the expected data layout, contact
the author.

Random seeds are fixed at 42 throughout. Paths are resolved via repository-root discovery
to remain environment-agnostic.

## Citation

```bibtex
@mastersthesis{kaufmann2026protectedareas,
  author = {Kaufmann, Gian-Luca},
  title  = {Predicting Protected Area Expansion and Transition Risk under the 30×30 Target},
  school = {ETH Zurich},
  year   = {2026},
  type   = {Master's Thesis}
}
```

## Contact

Gian-Luca Kaufmann, ETH Zurich (gikaufmann@ethz.ch). The repository is part of academic
thesis research; please contact the author for reuse beyond inspection and academic
reference.
