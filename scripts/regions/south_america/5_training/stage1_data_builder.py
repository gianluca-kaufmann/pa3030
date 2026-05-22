#!/usr/bin/env python3
"""Build country-year panel for Stage 1 expansion model (South America).

Merges PA expansion counts with political covariates (V-Dem, WGI, WDI, CBD dummies).
CSV paths are optional; missing files log a warning and leave columns null.

Output:
    data/south_america/stage1_panel.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.country_raster import resolve_panel_path
from scripts.regions.shared.stage1_panel import build_country_year_panel

TRAIN_YEARS = (2001, 2013)

REGION = "south_america"
OUT_PATH = _ROOT / "data" / "south_america" / "stage1_panel.parquet"

# ISO3 -> country_id mapping from policy preprocessing (must match raster codes)
ISO3_TO_ID = {
    "ARG": 1, "BOL": 2, "BRA": 3, "CHL": 4, "COL": 5, "ECU": 6,
    "GUY": 7, "PRY": 8, "PER": 9, "SUR": 10, "URY": 11, "VEN": 12,
}
ID_TO_ISO3 = {v: k for k, v in ISO3_TO_ID.items()}

CBD_MEETING_YEARS = {1994, 2002, 2010, 2018, 2022}


def _load_csv(path: Path, required_cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  WARN: missing {path}")
        return None
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  WARN: {path} missing columns {missing}")
        return None
    return df


def attach_political_covariates(cy: pd.DataFrame) -> pd.DataFrame:
    cy = cy.copy()
    cy["iso3"] = cy["country_id"].map(ID_TO_ISO3)
    cy["target_30x30"] = (cy["year"] >= 2023).astype(int)
    cy["cbd_meeting_year"] = cy["year"].isin(CBD_MEETING_YEARS).astype(int)

    shared = _ROOT / "data" / "shared"
    vdem = _load_csv(shared / "vdem_v15.csv", ["iso3", "year", "v2x_polyarchy"])
    if vdem is not None:
        cy = cy.merge(vdem[["iso3", "year", "v2x_polyarchy"]], on=["iso3", "year"], how="left")

    wgi = _load_csv(shared / "wgi.csv", ["iso3", "year", "gov_wgi_ge_est"])
    if wgi is not None:
        cy = cy.merge(wgi, on=["iso3", "year"], how="left")

    wdi = _load_csv(shared / "wdi.csv", ["iso3", "year", "gdp_per_capita", "agricultural_land_pct"])
    if wdi is not None:
        cy = cy.merge(wdi, on=["iso3", "year"], how="left")

    return cy


def main() -> None:
    panel_path = resolve_panel_path(REGION)
    print(f"Building Stage 1 panel from {panel_path}")
    cy = build_country_year_panel(panel_path, REGION)
    cy = attach_political_covariates(cy)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cy.to_parquet(OUT_PATH, index=False)
    meta = {
        "region": REGION,
        "source_panel": str(panel_path),
        "train_years": list(TRAIN_YEARS),
        "n_rows": len(cy),
        "n_countries": int(cy["country_id"].nunique()),
        "columns": list(cy.columns),
        "output": str(OUT_PATH),
    }
    meta_path = _ROOT / "outputs" / "data_checks" / "stage1_panel_build.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_PATH} ({len(cy)} country-years)")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
