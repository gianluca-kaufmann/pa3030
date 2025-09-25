# Scripts Folder Structure Documentation

This document provides a comprehensive overview of the `scripts` folder structure and explains the purpose and functionality of each script in this Master's Thesis project.

## 📁 Folder Structure Overview

```
scripts/
├── analysis/                    # Data validation and analysis scripts
├── consistency checks/           # Data quality and alignment verification
├── preprocessing/               # Data preparation and processing scripts
├── visualisations/              # Data visualization and plotting scripts
└── README.md                    # This documentation file
```

---

## 🔍 Analysis Folder (`analysis/`)

### Purpose
Contains scripts for data validation, quality assessment, and analytical verification of processed datasets.

### Scripts

#### `1_validate_dw.py`
**Purpose**: Validates DynamicWorld discrete files to ensure data integrity and proper conversion.

**Key Features**:
- Validates data type (uint8) and value range (0-1)
- Ensures each pixel has exactly one band set to 1
- Compares discrete files with original probability files
- Analyzes class distribution across all 9 land cover classes
- Provides comprehensive validation reports

**Input**: DynamicWorld discrete GeoTIFFs (`DW_*_discrete.tif`)
**Output**: Validation reports and statistics

**Usage**:
```bash
python3 analysis/1_validate_dw.py
```

---

## 🔧 Preprocessing Folder (`preprocessing/`)

### Purpose
Contains scripts for data preparation, cleaning, conversion, and merging operations.

### Scripts

#### `1_convert_dw.py`
**Purpose**: Converts DynamicWorld probability data to discrete land cover classes.

**Key Features**:
- Converts 9-band probability data to discrete classes
- Uses argmax to select highest probability class per pixel
- Handles NaN values and data type conversion
- Creates discrete files with one-hot encoding (one band = 1, others = 0)
- Validates conversion integrity

**Input**: DynamicWorld probability GeoTIFFs (`DW_*.tif`)
**Output**: Discrete DynamicWorld GeoTIFFs (`DW_*_discrete.tif`)

**Usage**:
```bash
python3 preprocessing/1_convert_dw.py
```

#### `1_clean_viirs`
**Purpose**: Cleans VIIRS night light data by addressing common data quality issues.

**Key Features**:
- Sets negative values to 0
- Removes faint background glow (values < 0.2 nW/cm²/sr)
- Caps extreme highs using P99.9 threshold
- Handles machine epsilon precision issues
- Provides detailed cleaning statistics

**Input**: Raw VIIRS GeoTIFFs (`viirs_*.tif`)
**Output**: Cleaned VIIRS GeoTIFFs (`viirs_*_cleaned.tif`)

**Usage**:
```bash
python3 preprocessing/1_clean_viirs
```

#### `1_2012_merge`
**Purpose**: Comprehensive script to merge multiple datasets into a single multi-band raster for 2012.

**Key Features**:
- Uses GPW population data as spatial template
- Merges raster files (protected areas, night lights)
- Rasterizes shapefiles (climate stabilization, biodiversity, wilderness, etc.)
- Creates individual masks and stacked output
- Generates comprehensive visualizations
- Handles CRS reprojection and spatial alignment

**Input**: Multiple datasets (GPW, WDPA, VIIRS, shapefiles)
**Output**: 
- Complete stack (`south_america_2012_complete_stack.tif`)
- Individual masks (`output_masks/`)
- Visualization plots

**Usage**:
```bash
python3 preprocessing/1_2012_merge
```

#### `1_DynamicWorld`
**Purpose**: Processes DynamicWorld data (placeholder for DynamicWorld-specific operations).

#### `old/` Subfolder
Contains legacy preprocessing scripts that have been superseded:
- `merge_high_biodiversity`
- `merge_other_GSN_bands`
- `mineral_viirs_pa_population_merge`
- `minerals_csv_conversion`

---

## 📊 Visualisations Folder (`visualisations/`)

### Purpose
Contains scripts for creating visualizations, plots, and analytical charts of the processed data.

### Scripts

#### `1_visualize_dw.py`
**Purpose**: Creates comprehensive visualizations of DynamicWorld discrete land cover data.

**Key Features**:
- Visualizes all 9 bands for each year (2012, 2017, 2022)
- Creates individual year plots and combined multi-year plots
- Generates class distribution charts
- Uses color-coded land cover classes
- Provides pixel count statistics and percentages

**Input**: DynamicWorld discrete GeoTIFFs (`DW_*_discrete.tif`)
**Output**: Multiple visualization PNG files

**Usage**:
```bash
python3 visualisations/1_visualize_dw.py
```

#### `1_gwp`
**Purpose**: Analyzes and visualizes GPW (Gridded Population of the World) population density data.

**Key Features**:
- Processes multiple years (2012, 2017, 2022)
- Creates both linear and logarithmic scale maps
- Calculates comprehensive statistics (mean, median, percentiles)
- Analyzes population density thresholds
- Generates comparison plots across years

**Input**: GPW GeoTIFFs (`gpw_*.tif`)
**Output**: Population density maps and statistics

**Usage**:
```bash
python3 visualisations/1_gwp
```

#### `1_viirs`
**Purpose**: Analyzes and visualizes cleaned VIIRS night light data.

**Key Features**:
- Processes cleaned VIIRS data for multiple years
- Creates standardized visualizations with consistent color scales
- Analyzes night light intensity statistics
- Provides data quality assessments
- Generates year-over-year comparisons

**Input**: Cleaned VIIRS GeoTIFFs (`viirs_*_cleaned.tif`)
**Output**: Night light intensity maps and analysis

**Usage**:
```bash
python3 visualisations/1_viirs
```

#### `1_wdpa`
**Purpose**: Analyzes and visualizes WDPA (World Database on Protected Areas) data.

**Key Features**:
- Processes binary protected area masks
- Calculates protected area statistics and coverage percentages
- Estimates protected area in km² (when CRS is projected)
- Creates protected area maps
- Provides multi-year comparisons

**Input**: WDPA GeoTIFFs (`wdpa_*.tif`)
**Output**: Protected area maps and statistics

**Usage**:
```bash
python3 visualisations/1_wdpa
```

#### `old/` Subfolder
Contains legacy visualization scripts that have been superseded:
- `firstlook_nightlight_WDPA`
- `firstlook_nightlight_WDPA_embed`
- `firstlook_nightlight_WDPA_Population`
- `minerals_plot_only.py`
- `visualize_gsn_merged.py`
- `visualize_hba_merged.py`
- `visualize_merged.py`

---

## 🔍 Consistency Checks Folder (`consistency checks/`)

### Purpose
Contains scripts for verifying data consistency, alignment, and quality across datasets.

### Scripts

#### `alignment check`
**Purpose**: Verifies spatial alignment and metadata consistency of merged raster files.

**Key Features**:
- Checks CRS, transform, and spatial dimensions
- Validates band count and data types
- Provides detailed metadata inspection
- Ensures proper file structure for analysis

**Input**: Merged raster files (e.g., `south_america_with_ree_plus_hba_plus_gsn.tif`)
**Output**: Alignment verification reports

**Usage**:
```bash
python3 "consistency checks/alignment check"
```

---

## 🚀 Workflow Overview

### Typical Processing Pipeline

1. **Data Preparation** (`preprocessing/`):
   ```bash
   # Convert DynamicWorld probabilities to discrete classes
   python3 preprocessing/1_convert_dw.py
   
   # Clean VIIRS night light data
   python3 preprocessing/1_clean_viirs
   
   # Merge all datasets for 2012
   python3 preprocessing/1_2012_merge
   ```

2. **Data Validation** (`analysis/`):
   ```bash
   # Validate DynamicWorld discrete files
   python3 analysis/1_validate_dw.py
   ```

3. **Consistency Checks** (`consistency checks/`):
   ```bash
   # Verify spatial alignment
   python3 "consistency checks/alignment check"
   ```

4. **Visualization** (`visualisations/`):
   ```bash
   # Create DynamicWorld visualizations
   python3 visualisations/1_visualize_dw.py
   
   # Create population density maps
   python3 visualisations/1_gwp
   
   # Create night light maps
   python3 visualisations/1_viirs
   
   # Create protected area maps
   python3 visualisations/1_wdpa
   ```

---

## 📋 Dependencies

### Required Python Packages
- `rasterio`: Geospatial raster I/O
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `geopandas`: Geospatial vector data processing
- `pathlib`: Path manipulation

### Data Requirements
- DynamicWorld probability GeoTIFFs
- GPW population density GeoTIFFs
- VIIRS night light GeoTIFFs
- WDPA protected area GeoTIFFs
- Various shapefiles (climate stabilization, biodiversity, etc.)

---

## 📁 Output Structure

### Generated Files
- **Processed Data**: Cleaned and converted GeoTIFFs in `data/`
- **Individual Masks**: Binary masks in `data/output_masks/`
- **Visualizations**: PNG plots in `outputs/Figures/`
- **Analysis Results**: Statistics and reports in `outputs/Results/`
- **Tables**: Data tables in `outputs/Tables/`

---

## 🔧 Configuration

### Path Configuration
Most scripts use hardcoded paths that should be updated for different environments:
```python
DATA_DIR = "/Users/gianluca/Desktop/Master's Thesis/code/data"
OUTPUT_DIR = "/Users/gianluca/Desktop/Master's Thesis/code/outputs"
```

### Processing Parameters
Key parameters that can be adjusted:
- VIIRS cleaning thresholds (baseline noise, extreme value caps)
- Visualization color scales and limits
- Spatial resolution and CRS settings
- Statistical percentiles for analysis

---

## 📝 Notes

- Scripts are numbered (`1_*`) to indicate processing order
- Legacy scripts are preserved in `old/` subfolders
- All scripts include comprehensive error handling and progress reporting
- Output files are automatically organized into appropriate directories
- Scripts are designed to be run independently or as part of a pipeline

---

## 🆘 Troubleshooting

### Common Issues
1. **File Not Found**: Check data paths in script configuration
2. **Memory Issues**: Process data in chunks for large files
3. **CRS Mismatches**: Ensure all input data uses consistent coordinate systems
4. **Permission Errors**: Verify write permissions for output directories

### Getting Help
- Check script docstrings for detailed parameter descriptions
- Review error messages for specific guidance
- Verify input data format and structure
- Ensure all dependencies are installed

---

*This documentation was generated to provide comprehensive guidance for using the scripts in this Master's Thesis project. For specific questions about individual scripts, refer to their docstrings and inline comments.*
