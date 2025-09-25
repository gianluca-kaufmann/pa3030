# Master's Thesis Code Repository  

This repository contains the code, data, and workflows for my Master’s thesis project at ETH Zurich at the Chair of Climate and Energy Economics of Professor Lint Barrage, in collaboration with Chiara Colesanti from the University of Zurich and Elena Almeida from the London School of Economics and Political Science (LSE).

The goal of the project is to predict where new protected areas (PAs) are most likely to be established under the global 30×30 biodiversity target.

Using geospatial data (WDPA, VIIRS night lights, population density, biodiversity and conservation layers, etc.) combined with machine learning models, the project builds a grid-based dataset to identify potential future PA designations.

The repository is organized into directories for raw and processed data, scripts for preprocessing, visualisations and analysis, and outputs such as maps, figures, and tables.


---

## Project at a Glance  

- **Focus**: Spatial relationships between environment, conservation, and economic activity  
- **Region**: South America (1 km resolution)  
- **Time Periods**: 2012 · 2017 · 2022  
- **Key Data Layers**:  
  - Population density (GPW)  
  - Protected areas (WDPA)  
  - Night lights (VIIRS)  
  - Land cover (DynamicWorld)  
  - Biodiversity and wilderness priority areas (GSN datasets)  
  - Rare earth element occurrences (USGS)
  - Mineral occurences
  - Mining locations
  - Weather data
  - Climate data
  - Powerplant locations (Hydro, coal, etc.)

---

## Repository Structure  

```
code/
├── data/          # Raw & processed datasets (GeoTIFFs, shapefiles, CSVs)
│   ├── core datasets (gpw, wdpa, viirs, dynamicworld, ree)
│   ├── conservation shapefiles (GSN: biodiversity, corridors, wilderness)
│   ├── output_masks/    # Binary masks from shapefiles
│   ├── old data/        # Legacy datasets & figures
├── outputs/       # Results and visualizations
│   ├── Figures/   # Maps & plots
│   ├── Results/   # Statistical outputs
│   └── Tables/    # Data tables, interactive files
├── scripts/       # Processing & analysis
│   ├── preprocessing/   # Cleaning & merging
│   ├── analysis/        # Validation & checks
│   ├── visualisations/  # Mapping & plots
│   └── consistency checks/
└── README.md
```

---

## Core Datasets  

| Dataset            | Files (examples) | Description | Resolution |
|--------------------|------------------|-------------|------------|
| **Population**     | `gpw_2012.tif`, `gpw_2017.tif`, `gpw_2022.tif` | GPW v4.11 population density | 1 km |
| **Protected Areas**| `wdpa_2012.tif`, `wdpa_2017.tif`, `wdpa_2022.tif` | WDPA protected area masks | 1 km |
| **Night Lights**   | `viirs_*.tif` (raw + cleaned) | VIIRS night light intensity | 1 km |
| **Land Cover**     | `DW_*.tif` (probabilities + discrete) | DynamicWorld land cover (9 classes) | 1 km |
| **Rare Earth Elements** | `Global_REE_occurrence_database.csv` · `REE_occurrence_count_025deg.tif` | USGS REE occurrences | 0.25° |
| **Conservation Areas** | Shapefiles: biodiversity, wilderness, corridors, climate, ecoregions | Global Safety Net (GSN) & WWF data | vector → rasterized 1 km |

---

## Outputs  

- **Figures**: land cover maps, population maps (linear/log), night light maps, PA coverage, integrated analyses  
- **Results**: statistics, model results  
- **Tables**: e.g. interactive REE occurrence database  

---

## Scripts  

Scripts are organized by workflow step:  

1. **Preprocessing**  
   - Convert DynamicWorld → discrete classes  
   - Clean VIIRS data  
   - Merge datasets (e.g., `1_2012_merge.py`)  

2. **Analysis**  
   - Validate dataset integrity (e.g., `validate_dw.py`)  
   - Run consistency/alignment checks  

3. **Visualization**  
   - Generate plots for GPW, VIIRS, WDPA, DynamicWorld  

See [`scripts/README.md`](scripts/README.md) for detailed documentation.  

---

## Workflow  

1. **Prepare Data**  
   ```bash
   python3 scripts/preprocessing/1_convert_dw.py
   python3 scripts/preprocessing/1_clean_viirs
   python3 scripts/preprocessing/1_2012_merge

2. **Validate**
   ```bash 
	python3 scripts/analysis/1_validate_dw.py
	python3 "scripts/consistency checks/alignment check"

3. **Visualize**
   ```bash 
	python3 scripts/visualisations/1_visualize_dw.py
	python3 scripts/visualisations/1_gwp.py
	python3 scripts/visualisations/1_viirs.py
	python3 scripts/visualisations/1_wdpa.py

Data Sources
```
	•	GPW v4.11 (CIESIN)
	•	WDPA (UNEP-WCMC)
	•	VIIRS Night Lights (NOAA)
	•	DynamicWorld (Google Earth Engine)
	•	USGS REE Occurrences
	•	WWF Terrestrial Ecoregions
	•	Global Safety Net conservation layers
	•	Alphaearth foundations (Satellite embeddings V1)
	•	S&P (Mining and mineral data)
	•	Global Power Plant Database (Google Earth Engine)
```

Citation
```
If you use this repository:
[insert citation later when we publish]
```
