# Protected Area Designation Prediction using Machine Learning

## Overview

This repository contains the code for a master's thesis project that aims to predict future protected area designations using machine learning. The project analyzes various geospatial datasets across South America to understand the factors that influence protected area establishment and develop predictive models for future designations.

## Project Structure

```
├── data/                          # Geospatial datasets (GeoTIFF format)
│   ├── ready/                    # Processed and ready-to-use datasets
│   │   ├── assetlevel/           # Asset-level data (power plants, infrastructure)
│   │   ├── deforestation/        # Hansen forest change data (2000-2024)
│   │   ├── DynamicWorld/         # Dynamic World land cover fractions (2017-2024)
│   │   ├── elevation/            # Elevation and slope data
│   │   ├── gdp/                  # GDP per capita data (2012-2021)
│   │   ├── GlobalSafetyNet/      # Conservation priority areas
│   │   ├── GPW/                  # Population density data (2000-2020)
│   │   ├── landcover/            # Land cover classifications
│   │   ├── NDVI/                 # Vegetation index data (2000-2024)
│   │   ├── oil_gas/              # Oil and gas infrastructure
│   │   ├── powerplants/          # Power plant locations
│   │   ├── road_infrastructure/  # Road network data
│   │   ├── VIIRS/                # Nighttime lights data
│   │   ├── WDPA/                 # World Database on Protected Areas
│   │   └── wildfire/             # Wildfire occurrence data
│   ├── ml/                       # Machine learning datasets and models
│   └── old data/                 # Archived and legacy datasets
├── scripts/
│   ├── data extraction/          # Google Earth Engine export scripts
│   ├── preprocessing/            # Data processing and preparation
│   ├── slurm/                    # SLURM job scripts for HPC
│   └── visualisations/           # Data visualization scripts
└── outputs/                      # Results and generated figures
    ├── Figures/                  # Visualization outputs
    ├── Results/                  # Model results and analysis
    └── Tables/                   # Statistical tables
```

## Key Scripts

### Data Extraction (`scripts/data extraction/`)
- **`deforestation_export`**: Exports Hansen Global Forest Change data using Google Earth Engine
- **`DW_export`**: Exports Dynamic World land cover fractions (2017-2024)
- **`GPW_export`**: Exports Gridded Population of the World data
- **`elevation_export`**: Exports elevation and slope data
- **`*_export`**: Additional export scripts for various datasets (NDVI, VIIRS, wildfire, etc.)

### Preprocessing (`scripts/preprocessing/`)
- **`pre-merge_overview`**: Comprehensive GeoTIFF scanner that analyzes all raster data
- **`gdp_preprocessing`**: Processes GDP per capita data from shapefiles to raster format
- **`assetlevel_preprocessing`**: Processes asset-level data (power plants) to raster format
- **`gsn_preprocessing`**: Processes Global Safety Net conservation priority areas
- **`roads_preprocessing`**: Processes road infrastructure data

### Visualizations (`scripts/visualisations/`)
- **`gdp_visualisation`**: Creates comprehensive GDP per capita visualizations
- **`deforestation_visualisation`**: Visualizes forest change patterns
- **`*_visualisation`**: Visualization scripts for all data types

### SLURM Jobs (`scripts/slurm/`)
- **`joint_model_train_array.slurm`**: SLURM script for training machine learning models on HPC

## Running the Code

### Prerequisites
- Python 3.11+ with required packages (rasterio, geopandas, numpy, pandas, matplotlib)
- Google Earth Engine account and authentication
- SLURM cluster access (for model training)

### Data Extraction
1. Authenticate with Google Earth Engine:
   ```bash
   earthengine authenticate
   ```

2. Run extraction scripts:
   ```bash
   python scripts/data\ extraction/deforestation_export
   python scripts/data\ extraction/DW_export
   # ... run other export scripts as needed
   ```

### Data Preprocessing
1. Run the overview scanner to analyze all data:
   ```bash
   python scripts/preprocessing/pre-merge_overview
   ```

2. Process specific datasets:
   ```bash
   python scripts/preprocessing/gdp_preprocessing
   python scripts/preprocessing/assetlevel_preprocessing
   ```

### Model Training
1. Submit SLURM job for model training:
   ```bash
   sbatch scripts/slurm/joint_model_train_array.slurm
   ```

### Visualizations
```bash
python scripts/visualisations/gdp_visualisation
python scripts/visualisations/deforestation_visualisation
# ... run other visualization scripts
```

## Data Sources and Formats

### Primary Data Sources
- **Hansen Global Forest Change**: Forest cover and loss data (2000-2024)
- **Dynamic World**: Near real-time land cover classification (2017-2024)
- **Gridded Population of the World (GPW)**: Population density estimates
- **World Database on Protected Areas (WDPA)**: Global protected area boundaries
- **Global Safety Net**: Conservation priority areas and corridors
- **VIIRS**: Nighttime lights data for economic activity
- **WorldClim**: Climate and weather data

### Data Format
- **Spatial Resolution**: 1km × 1km pixels
- **Coordinate System**: EPSG:3857 (Web Mercator)
- **File Format**: GeoTIFF (.tif)
- **Temporal Coverage**: 2000-2024 (varies by dataset)
- **Geographic Coverage**: South America

### Data Processing Pipeline
1. **Extraction**: Raw data exported from Google Earth Engine
2. **Preprocessing**: Reprojected and resampled to 1km resolution
3. **Organization**: Data organized in `data/ready/` subfolders by dataset type
4. **Validation**: Quality checks and consistency verification
5. **Integration**: Combined into unified datasets for modeling

## Notes
- All scripts include comprehensive error handling and logging
- Data processing is optimized for parallel processing where possible
- SLURM scripts are configured for ETH Zurich's Euler cluster
- Visualization scripts generate publication-ready figures
- The preprocessing overview script provides comprehensive data quality reports
