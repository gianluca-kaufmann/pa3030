# Master's Thesis Code Repository

This repository contains the complete codebase for a Master's Thesis project focused on geospatial analysis of South America, integrating multiple environmental and socio-economic datasets including population density, protected areas, night lights, land cover, biodiversity, and rare earth element occurrences.

## 📁 Repository Structure

```
code/
├── data/                        # Raw and processed geospatial datasets
│   ├── climate_stabilisation_areas/     # Climate stabilization shapefiles
│   ├── high_biodiversity_areas/         # High biodiversity area shapefiles
│   ├── intact_wilderness_areas/        # Intact wilderness area shapefiles
│   ├── potential_wildlife_corridors/    # Wildlife corridor shapefiles
│   ├── terrestrial_ecoregions/           # Terrestrial ecoregion shapefiles
│   ├── output_masks/                   # Generated binary masks
│   ├── old data/                       # Legacy datasets and figures
│   └── [various GeoTIFF files]         # Raster datasets
├── outputs/                     # Generated results and visualizations
│   ├── Figures/                        # Visualization plots and maps
│   ├── Results/                        # Analysis results and statistics
│   └── Tables/                         # Data tables and interactive outputs
├── scripts/                     # Processing and analysis scripts
│   ├── analysis/                       # Data validation and analysis
│   ├── consistency checks/             # Data quality verification
│   ├── preprocessing/                   # Data preparation and processing
│   ├── visualisations/                 # Data visualization scripts
│   └── README.md                       # Detailed scripts documentation
└── README.md                   # This file
```

---

## 🌍 Project Overview

### Research Focus
This project analyzes the spatial relationships between environmental conservation priorities, human activities, and mineral resource potential across South America. The analysis integrates multiple geospatial datasets to understand patterns of:

- **Population Distribution**: Human settlement patterns and density
- **Protected Areas**: Conservation coverage and effectiveness
- **Night Light Activity**: Economic activity and urbanization
- **Land Cover Change**: Environmental dynamics over time
- **Biodiversity Priority Areas**: Conservation importance
- **Rare Earth Element Occurrences**: Mineral resource potential

### Temporal Scope
The analysis covers three key time periods:
- **2012**: Baseline conditions
- **2017**: Mid-period assessment
- **2022**: Recent conditions

### Spatial Scope
- **Geographic Focus**: South America
- **Resolution**: Primarily 1km resolution
- **Coordinate System**: Consistent CRS across all datasets

---

## 📊 Data Directory (`data/`)

### Core Datasets

#### Population Data
- **`gpw_2012.tif`**, **`gpw_2017.tif`**, **`gpw_2022.tif`**
  - Source: Gridded Population of the World (GPW) v4.11
  - Content: Population density (persons/km²)
  - Resolution: 1km
  - Years: 2010, 2015, 2020 (mapped to analysis years)

#### Protected Areas
- **`wdpa_2012.tif`**, **`wdpa_2017.tif`**, **`wdpa_2022.tif`**
  - Source: World Database on Protected Areas (WDPA)
  - Content: Binary protected area masks (0/1)
  - Resolution: 1km
  - Status: As-of-date for each year

#### Night Light Data
- **`viirs_2012.tif`**, **`viirs_2017.tif`**, **`viirs_2022.tif`**
  - Source: VIIRS Day/Night Band
  - Content: Raw night light intensity (nW/cm²/sr)
  - Resolution: 1km
- **`viirs_*_cleaned.tif`**
  - Content: Cleaned night light data (processed versions)
  - Processing: Removed noise, capped extremes, handled negatives

#### Land Cover Data
- **`DW_2012.tif`**, **`DW_2017.tif`**, **`DW_2022.tif`**
  - Source: DynamicWorld (Google Earth Engine)
  - Content: 9-band probability data for land cover classes
  - Classes: Water, Trees, Grass, Flooded vegetation, Crops, Shrub/scrub, Built, Bare ground, Snow/ice
- **`DW_*_discrete.tif`**
  - Content: Discrete land cover classification (one-hot encoded)
  - Processing: Converted from probability to discrete classes

#### Rare Earth Elements
- **`Global_REE_occurrence_database.csv`** / **`.xlsx`**
  - Source: USGS Global REE Occurrence Database
  - Content: Comprehensive database of REE occurrences worldwide
  - Fields: Location, geology, mineralogy, resources, production status
- **`REE_occurrence_count_025deg.tif`**
  - Content: Rasterized REE occurrence density
  - Resolution: 0.25° (~25km)

### Conservation Priority Areas

#### Shapefile Collections
Each conservation dataset is stored as a complete shapefile with associated files:

- **`climate_stabilisation_areas/`**
  - Content: Areas critical for climate stabilization
  - Source: Climate-focused conservation prioritization

- **`high_biodiversity_areas/`**
  - Content: High biodiversity priority areas
  - Source: Biodiversity conservation assessments

- **`intact_wilderness_areas/`**
  - Content: Intact wilderness areas
  - Source: Wilderness conservation mapping

- **`potential_wildlife_corridors/`**
  - Content: Potential wildlife movement corridors
  - Source: Connectivity conservation planning

- **`terrestrial_ecoregions/`**
  - Content: Terrestrial ecoregion boundaries
  - Source: WWF Terrestrial Ecoregions

### Processed Outputs

#### Individual Masks (`output_masks/`)
Binary raster masks derived from shapefiles:
- `climate_stabilization_area_mask_1km.tif`
- `high_biodiversity_mask_1km.tif`
- `intact_wilderness_areas_mask_1km.tif`
- `potential_wildlife_corridors_mask_1km.tif`
- `terrestrial_ecoregions_mask_1km.tif`

#### Merged Datasets
- **`south_america_2012_complete_stack.tif`**
  - Content: Multi-band raster combining all 2012 datasets
  - Bands: Population, Protected Areas, Night Lights, REE, Biodiversity, Wilderness, Corridors, Ecoregions
  - Resolution: 1km
  - CRS: Consistent across all bands

#### Legacy Data (`old data/`)
- Historical processing outputs
- Previous analysis figures
- Superseded datasets

---

## 📈 Outputs Directory (`outputs/`)

### Figures (`Figures/`)
Comprehensive visualization collection:

#### DynamicWorld Visualizations
- `dynamicworld_discrete_*_all_bands.png`: Individual year land cover maps
- `dynamicworld_discrete_all_years_bands.png`: Multi-year comparison
- `dynamicworld_discrete_class_distribution.png`: Class distribution analysis

#### Population Analysis
- `gpw_*_map_linear.png`: Linear scale population density maps
- `gpw_*_map_log.png`: Logarithmic scale population density maps

#### Night Light Analysis
- `viirs_*_cleaned_linear.png`: Cleaned night light intensity maps
- `nightlight_WDPA_*.png`: Night light vs protected area analysis

#### Protected Areas
- `wdpa_*_map.png`: Protected area coverage maps

#### Integrated Analysis
- `2012_complete_stack_visualization.png`: Multi-band dataset overview
- `band_*_*.png`: Individual band visualizations
- `merged_raster_visualization.png`: Merged dataset overview

#### Conservation Analysis
- `hba_*.png`: High biodiversity area analysis
- `gsn_*.png`: Global Safety Net features analysis

### Results (`Results/`)
- Analysis results and statistical outputs
- Model outputs and derived metrics

### Tables (`Tables/`)
- `mineral_occurrences_interactive.html`: Interactive REE occurrence table
- Statistical summaries and data tables

---

## 🔧 Scripts Directory (`scripts/`)

### Overview
The scripts directory contains the complete processing pipeline organized into four main categories:

#### Analysis (`analysis/`)
- **`1_validate_dw.py`**: DynamicWorld discrete file validation
- Data integrity checks and quality assessment

#### Preprocessing (`preprocessing/`)
- **`1_convert_dw.py`**: DynamicWorld probability to discrete conversion
- **`1_clean_viirs`**: VIIRS night light data cleaning
- **`1_2012_merge`**: Comprehensive dataset merging for 2012
- **`1_DynamicWorld`**: DynamicWorld processing utilities
- **`old/`**: Legacy preprocessing scripts

#### Visualizations (`visualisations/`)
- **`1_visualize_dw.py`**: DynamicWorld visualization suite
- **`1_gwp`**: Population density analysis and mapping
- **`1_viirs`**: Night light data visualization
- **`1_wdpa`**: Protected area mapping and analysis
- **`old/`**: Legacy visualization scripts

#### Consistency Checks (`consistency checks/`)
- **`alignment check`**: Spatial alignment verification
- Data quality and consistency validation

### Detailed Documentation
For comprehensive script documentation, see: [`scripts/README.md`](scripts/README.md)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required packages: `rasterio`, `numpy`, `matplotlib`, `geopandas`, `pathlib`

### Installation
```bash
# Clone or download the repository
cd /path/to/code

# Install required packages
pip install rasterio numpy matplotlib geopandas pathlib
```

### Basic Workflow

#### 1. Data Preparation
```bash
# Convert DynamicWorld probabilities to discrete classes
python3 scripts/preprocessing/1_convert_dw.py

# Clean VIIRS night light data
python3 scripts/preprocessing/1_clean_viirs

# Merge all datasets for 2012
python3 scripts/preprocessing/1_2012_merge
```

#### 2. Data Validation
```bash
# Validate DynamicWorld discrete files
python3 scripts/analysis/1_validate_dw.py

# Check spatial alignment
python3 "scripts/consistency checks/alignment check"
```

#### 3. Visualization
```bash
# Create DynamicWorld visualizations
python3 scripts/visualisations/1_visualize_dw.py

# Generate population density maps
python3 scripts/visualisations/1_gwp

# Create night light visualizations
python3 scripts/visualisations/1_viirs

# Map protected areas
python3 scripts/visualisations/1_wdpa
```

---

## 📋 Data Sources and Citations

### Primary Data Sources
1. **GPW v4.11**: Center for International Earth Science Information Network (CIESIN)
2. **WDPA**: World Database on Protected Areas (UNEP-WCMC)
3. **VIIRS**: Visible Infrared Imaging Radiometer Suite (NOAA)
4. **DynamicWorld**: Google Earth Engine
5. **USGS REE Database**: U.S. Geological Survey
6. **WWF Ecoregions**: World Wildlife Fund
7. **Conservation Priority Areas**: Various conservation organizations

### Data Processing
- All datasets reprojected to consistent coordinate system
- Spatial resolution standardized to 1km
- Temporal alignment to analysis years (2012, 2017, 2022)
- Quality control and validation procedures applied

---

## 🔍 Key Features

### Multi-Temporal Analysis
- Consistent temporal coverage across all datasets
- Change detection and trend analysis capabilities
- Temporal alignment of different data sources

### Multi-Scale Integration
- Integration of point data (REE occurrences) with raster data
- Combination of vector (shapefiles) and raster (GeoTIFFs) formats
- Consistent spatial resolution and coordinate systems

### Comprehensive Coverage
- Population dynamics and urbanization patterns
- Conservation effectiveness and coverage
- Economic activity indicators (night lights)
- Environmental change (land cover)
- Resource potential (REE occurrences)

### Quality Assurance
- Automated validation procedures
- Consistency checks across datasets
- Error handling and data quality reporting
- Reproducible processing pipeline

---

## 📊 Analysis Capabilities

### Spatial Analysis
- Overlay analysis between different datasets
- Spatial correlation and pattern analysis
- Hotspot identification and clustering
- Connectivity and corridor analysis

### Temporal Analysis
- Change detection over time
- Trend analysis and forecasting
- Temporal correlation between variables
- Before/after comparison studies

### Statistical Analysis
- Descriptive statistics for all datasets
- Correlation analysis between variables
- Spatial autocorrelation analysis
- Regression and modeling capabilities

---

## 🛠️ Technical Specifications

### Data Formats
- **Raster Data**: GeoTIFF format with LZW compression
- **Vector Data**: Shapefile format with complete attribute tables
- **Tabular Data**: CSV and Excel formats
- **Visualizations**: PNG format, 300 DPI resolution

### Coordinate Systems
- **Primary CRS**: Consistent projected coordinate system
- **Resolution**: 1km pixel size
- **Extent**: South America coverage
- **Alignment**: All datasets spatially aligned

### Processing Standards
- **Reproducibility**: All scripts include random seed setting
- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Robust error checking and reporting
- **Logging**: Detailed processing logs and progress reporting

---

## 📝 Usage Notes

### Data Access
- All datasets are stored locally in the `data/` directory
- Processed outputs are automatically organized in `outputs/`
- Scripts use absolute paths that may need adjustment for different environments

### Processing Order
- Scripts are numbered to indicate recommended processing sequence
- Some scripts depend on outputs from previous scripts
- Validation scripts should be run after data processing

### Customization
- Processing parameters can be adjusted in script configuration sections
- Visualization parameters (colors, scales, etc.) can be modified
- Additional datasets can be integrated following existing patterns

---

## 🔧 Troubleshooting

### Common Issues
1. **File Path Errors**: Update absolute paths in scripts for your environment
2. **Memory Issues**: Process large datasets in chunks or increase system memory
3. **CRS Mismatches**: Ensure all input data uses consistent coordinate systems
4. **Missing Dependencies**: Install required Python packages

### Data Quality Issues
1. **NoData Values**: Scripts handle NoData values appropriately
2. **Spatial Alignment**: Consistency checks verify proper alignment
3. **Data Type Issues**: Automatic data type conversion and validation

### Performance Optimization
1. **Chunked Processing**: Large datasets processed in manageable chunks
2. **Compression**: Output files use LZW compression to reduce size
3. **Parallel Processing**: Some operations can be parallelized for speed

---

## 📞 Support and Contact

For questions about this codebase:
1. Check the detailed script documentation in `scripts/README.md`
2. Review script docstrings and inline comments
3. Verify data paths and dependencies
4. Check error messages for specific guidance

---

## 📄 License and Citation

This codebase is part of a Master's Thesis project. Please cite appropriately if used in research or publications.

### Recommended Citation Format
```
[Author Name]. (Year). "Master's Thesis Title." 
Master's Thesis, [Institution]. 
Code Repository: [Repository URL]
```

---

*This documentation provides a comprehensive overview of the Master's Thesis codebase. For detailed information about specific scripts and processing steps, refer to the individual script documentation and inline comments.*
