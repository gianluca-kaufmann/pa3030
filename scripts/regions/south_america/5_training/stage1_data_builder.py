#!/usr/bin/env python3
"""Build country-year panel for Stage 1 expansion model (South America).

Merges PA expansion counts with political covariates (V-Dem, WGI, WDI, CBD dummies).
CSV paths are optional; missing files log a warning and leave columns null.

Output:
    data/south_america/stage1_panel.parquet
"""

from __future__ import annotations

import json
import os
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
PANEL_YEARS = (2001, 2024)  # covers train + early-stop + test

# Optional: WDPA CSV files for pre-2001 expansion data.
# Download the WDPA global CSV from https://www.protectedplanet.net/en/thematic-areas/wdpa#download
# (accept terms → "WDPA CSV" → extract zip → rename to match pattern below).
# When these files are present, the panel is extended back to PRE2001_YEAR_RANGE.
# Political covariates (V-Dem, WDI) are available from 1990; WGI from 1996.
# Recommended start year: 1996 (all covariates available).
_WDPA_DIR = _ROOT / "data" / "shared"
# Single combined WDPA CSV (polygons + points) downloaded from protectedplanet.net.
# Rename to WDPA_Public.csv or keep the dated filename; the list is checked in order.
WDPA_CSV_PATHS = [
    _WDPA_DIR / "WDPA_May2026_Public_csv.csv",   # current download (May 2026)
    _WDPA_DIR / "wdpa" / "WDPA_WDOECM_Public_0.csv",  # fallback: three-file layout
    _WDPA_DIR / "wdpa" / "WDPA_WDOECM_Public_1.csv",
    _WDPA_DIR / "wdpa" / "WDPA_WDOECM_Public_2.csv",
]
PRE2001_YEAR_RANGE = (1990, 2000)  # change to (1996, 2000) to restrict to WGI coverage

REGION = "south_america"
# Write to scratch if available (home is quota-constrained); fall back to repo root.
_scratch = os.environ.get("SCRATCH", "")
OUT_PATH = (
    Path(_scratch) / "data/south_america/stage1_panel.parquet"
    if _scratch
    else _ROOT / "data" / "south_america" / "stage1_panel.parquet"
)

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


def _load_csv_partial(path: Path, wanted_cols: list[str]) -> pd.DataFrame | None:
    """Like _load_csv but loads whichever of wanted_cols are present (no fail on missing)."""
    if not path.exists():
        print(f"  WARN: missing {path}")
        return None
    df = pd.read_csv(path)
    available = [c for c in wanted_cols if c in df.columns]
    missing = [c for c in wanted_cols if c not in df.columns]
    if missing:
        print(f"  WARN: {path} missing columns {missing} — loading available: {available}")
    if not available:
        return None
    id_cols = ["iso3", "year"]
    keep = [c for c in id_cols if c in df.columns] + available
    return df[keep]


def attach_political_covariates(cy: pd.DataFrame) -> pd.DataFrame:
    cy = cy.copy()
    cy["iso3"] = cy["country_id"].map(ID_TO_ISO3)
    cy["target_30x30"] = (cy["year"] >= 2023).astype(int)
    cy["cbd_meeting_year"] = cy["year"].isin(CBD_MEETING_YEARS).astype(int)

    shared = _ROOT / "data" / "shared"
    _vdem_cols = ["v2x_polyarchy", "v2x_corr", "v2cseeorgs",
                  "v2xlg_legcon", "v2csprtcpt"]
    # Use partial load: vdem_v15.csv may not have sub-index columns (v2xlg_legcon,
    # v2csprtcpt); those are merged later from VDem/V-Dem-CY-Core-v15.csv in
    # model1_expansion.py. Loading what's available avoids silently dropping v2x_polyarchy.
    vdem = _load_csv_partial(shared / "vdem_v15.csv", _vdem_cols)
    if vdem is not None:
        cy = cy.merge(vdem, on=["iso3", "year"], how="left")

    wgi = _load_csv(shared / "wgi.csv", ["iso3", "year", "gov_wgi_ge_est", "gov_wgi_rl_est"])
    if wgi is not None:
        cy = cy.merge(wgi, on=["iso3", "year"], how="left")

    wdi = _load_csv(shared / "wdi.csv", ["iso3", "year", "gdp_per_capita", "agricultural_land_pct"])
    if wdi is not None:
        # gdp_growth_lag1: year-on-year % change in GDP/capita, lagged 1 year.
        # Uses WDI data back to 1990, so 2001+ values are all well-defined.
        # Lag-1 avoids using same-year GDP growth in forward prediction.
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

    # Optionally extend with pre-2001 WDPA expansion data.
    # Requires WDPA global CSV download from protectedplanet.net (see WDPA_CSV_PATHS above).
    wdpa_present = [p for p in WDPA_CSV_PATHS if p.exists()]
    if wdpa_present:
        print(f"  WDPA files found ({len(wdpa_present)}): extending panel to {PRE2001_YEAR_RANGE}")
        wdpa_rows = compute_pre2001_expansion(wdpa_present, ISO3_TO_ID, PRE2001_YEAR_RANGE)
        if not wdpa_rows.empty:
            n_before = len(cy)
            cy = extend_panel_with_wdpa(cy, wdpa_rows)
            print(f"  Pre-2001 rows added: {len(cy) - n_before} ({n_before} → {len(cy)})")
        else:
            print("  WARN: compute_pre2001_expansion returned no rows — check WDPA file")
    else:
        print(f"  WDPA CSV not found at {_WDPA_DIR} — skipping pre-2001 extension.")
        print(f"    To enable: download from https://www.protectedplanet.net/en/thematic-areas/wdpa#download")
        print(f"    Extract zip and copy CSVs as data/shared/wdpa/WDPA_WDOECM_Public_{{0,1,2}}.csv")

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
    meta_path = _ROOT / "outputs" / "data_checks" / "stage1_panel_build.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_PATH} ({len(cy)} country-years)")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
