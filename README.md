# Protected Area Designation Prediction using Machine Learning

## Overview

This repository accompanies a master's thesis that studies the drivers of protected area establishment across South America and develops machine learning models to predict future designations. The pipeline combines large-scale remote sensing products, socio-economic indicators, and conservation datasets to create spatial panels, derive features, and train predictive models.

## Repository Layout

```
├── README.md
├── data/
│   ├── ready/                       # GeoTIFF layers ready for analysis and merging
│   ├── ml/                          # Data ready for ML model runs
│   └── old data/                    # Archived inputs kept for reproducibility
├── scripts/
│   ├── data extraction/             # Google Earth Engine (GEE) and API exporters
│   ├── preprocessing/               # Cleaning, harmonisation of external, non-GEE datasets
│   ├── visualisations/              # Exploratory figures & sanity checks
│   └── merging/                     # Merging scripts for the embeddings dataset + the standard dataset
├── outputs/
│   ├── Figures/                     # Generated plots (per dataset)
│   ├── Results/                     # Model outputs, predictions, diagnostics
│   └── Tables/                      # Summaries from preprocessing and merges
├── slurm/
│   ├── RUN.slurm                    # Main SLURM submission script
│   ├── requirements.txt             # Dependency lockfile for cluster runs
│   ├── create_env.sh                # Euler helper to load modules & create venvs
│   └── euler_run_instructions.md    # Step-by-step guide for running on Euler
└── wandb/                           # Weights & Biases run artefacts and logs
```

## Environment Setup

- Python 3.11 or newer.
- Raster and vector dependencies (GDAL ≥3.6, PROJ, GEOS). On macOS use `brew install gdal` prior to installing Python packages.
- Google Earth Engine (GEE) Python API with an authenticated account.
- Access to ETH Zurich's Euler cluster (or another SLURM cluster) for large-scale merges and model training.

### Quick start

```bash
# from the repository root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r slurm/requirements.txt
```

For Euler cluster setup, use the helper inside `slurm/`:

```bash
ssh <ethz_username>@euler.ethz.ch
cd /cluster/home/<ethz_username>/master_thesis
bash slurm/create_env.sh
```

Authenticate with GEE once per machine:

```bash
earthengine authenticate
```

## Workflow at a Glance

1. **Data extraction** – export datasets from GEE into `data/ready`.
2. **Preprocessing** – clean, resample, and harmonise external, non-GEE datasets.
3. **Visualisations** – inspect the processed layers with exploratory figures and standard statistical computations.
4. **Merging** – merge aligned layers into a panel dataset stored in `data/ml`. merge all annual embeddings datasets into a single dataset stored in `data/ml`.
5. **Model training** – launch SLURM jobs (or local scripts) to fit predictive models and generate outputs.

The sections below detail each stage, expected inputs/outputs, and common command sequences.

## 1. Data Extraction

Scripts under `scripts/data extraction/` pull data from Google Earth Engine and assorted APIs. Each script targets a specific theme (deforestation, land cover, NDVI, VIIRS night lights, infrastructure layers, etc.) and writes aligned GeoTIFF rasters into `data/ready/<dataset>/`.

### Typical run sequence

1. Confirm the Earth Engine project, spatial bounds, and export resolution in the script constants (top of each exporter).
2. Launch exports dataset by dataset. Use `python` directly; each script handles its own directories and logging.

```bash
python scripts/data\ extraction/deforestation_export
python scripts/data\ extraction/DW_export
python scripts/data\ extraction/NDVI_export
python scripts/data\ extraction/VIIRS_export
python scripts/data\ extraction/WDPA_export
python scripts/data\ extraction/powerplants_export
# ... continue for other datasets you require
```

3. Monitor Earth Engine Tasks (`earthengine task list`) until all jobs finish. Downloaded assets are stored locally.

**Tips**

- If you extend the study area or temporal coverage, update the `START_YEAR`, `END_YEAR`, or bounding geometries at the top of each exporter.

## 2. Preprocessing

Preprocessing externally downloaded scripts to harmonise formats. Outputs are saved back into `data/ready/` (overwriting intermediate files). Summaries of final, pre-merge overview over all prepared datasets in `outputs/Tables/`.

### Recommended order

```bash
# Dataset-specific cleaners
python scripts/preprocessing/assetlevel_preprocessing
python scripts/preprocessing/gdp_preprocessing
python scripts/preprocessing/gsn_preprocessing
python scripts/preprocessing/roads_preprocessing
# ... execute additional preprocessing scripts as relevant

# Global overview (generates QA tables and HTML report)
python scripts/preprocessing/pre-merge_overview
```

- Each script reports missing tiles, nodata percentages, and reprojected CRS. Inspect console output and the generated `pre-merge_overview.*` files before moving on.
- Some preprocessors expect auxiliary CSV or shapefiles. Refer to inline comments at the top of the script to adjust paths.

## 3. Visualisations

Visualisation scripts provide quick sanity checks and first-look analytics. They read the harmonised rasters from `data/ready/` and write PNGs to `outputs/Figures/<dataset>_vis/`.

```bash
python scripts/visualisations/deforestation_visualisation
python scripts/visualisations/dw_visualisation
python scripts/visualisations/ndvi_visualisation
python scripts/visualisations/viirs_visualisation
python scripts/visualisations/worldclim_visualisation
# ... run for each dataset of interest
```

- Review the generated maps/statistics to confirm spatial alignment, expected ranges, and time trends.
- Visualisations are optional for production runs but strongly recommended after changes to the extraction or preprocessing stages.

## 4. Merging

Merging scripts build the modelling panels by stacking rasters, masking to valid coverage, and aggregating temporal layers. The main entry points are in `scripts/merging/`. There are two distinct datasets and therefore merging scripts, one for the standard dataset, and one for the embeddings dataset.

```bash
# Merge satellite embeddings produced per tile (optional, heavy)
python scripts/merging/merge_embeddings_total

# Assemble the final standard panel dataset
python scripts/merging/merge_total_optimized
```

- The merged files are stored in `data/ml/` and `outputs/Results/merged_tifs/`.
- For very large runs (e.g. `merge_total_optimized`), submit the merging script to SLURM using the provided batch files (e.g., `sbatch slurm/RUN.slurm`); adapt partition and memory requirements as needed.

## 5. Model Training

Model fitting is orchestrated via the batch scripts in the `slurm/` directory. The typical workflow:

1. Configure hyperparameters, data splits, and output locations inside `slurm/RUN.slurm` (or any additional SLURM scripts you add).
2. Submit the job on Euler (or another SLURM cluster):

```bash
sbatch slurm/RUN.slurm
```

3. Monitor progress with `squeue -u <ethz_username>` and check logs stored under `wandb/` and `$SCRATCH/logs/merge_run_<jobid>.{out,err}`.

Consult `slurm/euler_run_instructions.md` for a detailed, step-by-step Euler walkthrough.

For small experiments you can mirror the SLURM script locally by running the referenced Python module directly; ensure you activate the same environment and set any required environment variables (`WANDB_API_KEY`, etc.).

## Outputs

- `outputs/Figures/` – raster snapshots, histograms, and trend plots used for exploratory analysis and reporting.
- `outputs/Tables/` – CSV/HTML summaries from preprocessing (coverage stats, missing data checks).
- `outputs/Results/` – merged GeoTIFFs, modelling matrices, predictions, and evaluation metrics.
- `wandb/` – Weights & Biases run metadata, configs, and logs (created automatically during training).

## Data Characteristics

- **Spatial resolution**: 1 km × 1 km grid cells.
- **Coordinate reference system**: EPSG:3857 (Web Mercator).
- **Formats**: GeoTIFF for rasters, CSV/Parquet for panel datasets.
- **Temporal coverage**: 2000–2024 (dataset-specific availability).
- **Region**: South America (continental extent).

## Troubleshooting

- Missing GDAL/PROJ libraries usually surface as `rasterio.errors.RasterioIOError`. Ensure GDAL is installed at the system level before installing Python packages.
- GEE `QuotaExceededError` indicates too many concurrent tasks; stagger export scripts or reduce tiling.
- When extending the pipeline to new regions, update bounding boxes, masks, and valid country codes consistently across extraction and preprocessing scripts.