"""Locations for the global WDPA designation-event work.

Large pulls (per-country WDPA attributes, zonal statistics, the event dataset) live under
DATA_DIR, which is untracked. PA3030_WORLD_DATA_DIR overrides it, for example to put the data
on MSI project storage. The three country-year tables are small, tracked inputs in
input_template/world/.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("PA3030_WORLD_DATA_DIR", REPO_ROOT / "data" / "world"))
RAW_DIR = DATA_DIR / "wdpa_raw"
DOWNLOAD_DIR = DATA_DIR / "downloads"
COUNTRY_YEAR_DIR = REPO_ROOT / "input_template" / "world"

CROSSWALK_PATH = DATA_DIR / "country_crosswalk.parquet"
EVENT_DATASET_PATH = DATA_DIR / "wdpa_polygon_dataset.parquet"
GOVERNANCE_PATH = COUNTRY_YEAR_DIR / "governance_country_year.csv"
FINANCE_PATH = COUNTRY_YEAR_DIR / "conservation_finance_country_year.csv"
REDD_PATH = COUNTRY_YEAR_DIR / "redd_entry_country_year.csv"
