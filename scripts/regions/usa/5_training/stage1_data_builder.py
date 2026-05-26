#!/usr/bin/env python3
"""Build country-year panel for Stage 1 expansion model (USA).

Merges PA expansion counts with political covariates (V-Dem, WGI, WDI, election cycle, REDD+).
CSV paths are optional; missing files log a warning and leave columns null.

Output:
    data/usa/stage1_panel.parquet
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
from scripts.regions.shared.stage1_panel import (
    build_country_year_panel,
    compute_pre2001_expansion,
    compute_years_to_next_election,
    extend_panel_with_wdpa,
)

TRAIN_YEARS = (2001, 2013)
PANEL_YEARS = (2001, 2024)

_WDPA_DIR = _ROOT / "data" / "shared"
WDPA_CSV_PATHS = [
    _WDPA_DIR / "WDPA_May2026_Public_csv.csv",
    _WDPA_DIR / "wdpa" / "WDPA_WDOECM_Public_0.csv",
    _WDPA_DIR / "wdpa" / "WDPA_WDOECM_Public_1.csv",
    _WDPA_DIR / "wdpa" / "WDPA_WDOECM_Public_2.csv",
]
PRE2001_YEAR_RANGE = (1990, 2000)

REGION = "usa"
OUT_PATH = _ROOT / "data" / "usa" / "stage1_panel.parquet"

ISO3_TO_ID = {"USA": 1}
ID_TO_ISO3 = {v: k for k, v in ISO3_TO_ID.items()}

CBD_MEETING_YEARS = {1994, 2002, 2010, 2018, 2022}


def _load_csv(path: Path, required_cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  WARN: missing {path}")
        return None
    df = pd.read_csv(path, comment="#")
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
    _vdem_cols = ["iso3", "year", "v2x_polyarchy", "v2x_corr", "v2cseeorgs",
                  "v2xlg_legcon", "v2csprtcpt"]
    vdem = _load_csv(shared / "vdem_v15.csv", _vdem_cols)
    if vdem is not None:
        cy = cy.merge(
            vdem[[c for c in _vdem_cols if c in vdem.columns]],
            on=["iso3", "year"], how="left",
        )

    wgi = _load_csv(shared / "wgi.csv", ["iso3", "year", "gov_wgi_ge_est", "gov_wgi_rl_est"])
    if wgi is not None:
        cy = cy.merge(wgi, on=["iso3", "year"], how="left")

    wdi = _load_csv(shared / "wdi.csv", ["iso3", "year", "gdp_per_capita", "agricultural_land_pct"])
    if wdi is not None:
        wdi = wdi.sort_values(["iso3", "year"])
        wdi["gdp_growth_lag1"] = (
            wdi.groupby("iso3")["gdp_per_capita"].pct_change().shift(1)
        )
        cy = cy.merge(wdi, on=["iso3", "year"], how="left")

    try:
        elec = compute_years_to_next_election(list(ISO3_TO_ID.keys()), PANEL_YEARS)
        cy = cy.merge(elec, on=["iso3", "year"], how="left")
    except FileNotFoundError as e:
        print(f"  WARN: {e} — years_to_next_election will be null")

    redd = _load_csv(shared / "redd_plus.csv", ["iso3", "year", "redd_plus_enrolled"])
    if redd is not None:
        cy = cy.merge(
            redd[["iso3", "year", "redd_plus_enrolled"]],
            on=["iso3", "year"], how="left",
        )

    return cy


def main() -> None:
    panel_path = resolve_panel_path(REGION)
    print(f"Building Stage 1 panel from {panel_path}")
    cy = build_country_year_panel(panel_path, REGION, year_range=PANEL_YEARS)

    wdpa_present = [p for p in WDPA_CSV_PATHS if p.exists()]
    if wdpa_present:
        print(f"  WDPA files found ({len(wdpa_present)}): extending panel to {PRE2001_YEAR_RANGE}")
        wdpa_rows = compute_pre2001_expansion(wdpa_present, ISO3_TO_ID, PRE2001_YEAR_RANGE)
        if not wdpa_rows.empty:
            n_before = len(cy)
            cy = extend_panel_with_wdpa(cy, wdpa_rows)
            print(f"  Pre-2001 rows added: {len(cy) - n_before} ({n_before} → {len(cy)})")
    else:
        print(f"  WDPA CSV not found at {_WDPA_DIR} — skipping pre-2001 extension.")

    cy = attach_political_covariates(cy)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cy.to_parquet(OUT_PATH, index=False)
    meta = {
        "region": REGION,
        "source_panel": str(panel_path),
        "train_years": list(TRAIN_YEARS),
        "panel_years": list(PANEL_YEARS),
        "pre2001_extension": PRE2001_YEAR_RANGE if wdpa_present else None,
        "n_rows": len(cy),
        "n_countries": int(cy["country_id"].nunique()),
        "columns": list(cy.columns),
        "output": str(OUT_PATH),
    }
    meta_path = _ROOT / "outputs" / "data_checks" / "stage1_panel_usa_build.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_PATH} ({len(cy)} country-years)")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
