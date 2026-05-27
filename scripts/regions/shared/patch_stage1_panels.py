#!/usr/bin/env python3
"""Patch stage1_panel.parquet files for SA and SEA with terrestrial-only pre-2001 data.

Problem: compute_pre2001_expansion previously used GIS_AREA (total, marine+terrestrial).
Major outlier: ECU 1998 = 138,722 km² (Galápagos Marine Reserve) — 86% of ECU
pre-2001 area is marine, contaminating momentum lag features for 2001-2003.

Fix: recompute pre-2001 rows using GIS_AREA - GIS_M_AREA (terrestrial only),
then call extend_panel_with_wdpa to rebuild the lag/cumsum chain.
Political covariates (V-Dem, WGI, etc.) for 2001+ are preserved via left-merge.

This script runs fully locally — no Euler access needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.stage1_panel import (
    compute_pre2001_expansion,
    extend_panel_with_wdpa,
)

WDPA_CSV_PATHS = [
    _ROOT / "data" / "shared" / "WDPA_May2026_Public_csv.csv",
    _ROOT / "data" / "shared" / "wdpa" / "WDPA_WDOECM_Public_0.csv",
    _ROOT / "data" / "shared" / "wdpa" / "WDPA_WDOECM_Public_1.csv",
    _ROOT / "data" / "shared" / "wdpa" / "WDPA_WDOECM_Public_2.csv",
]
PRE2001_YEAR_RANGE = (1990, 2000)

REGIONS = {
    "south_america": {
        "panel": _ROOT / "data" / "south_america" / "stage1_panel.parquet",
        "iso3_to_id": {
            "ARG": 1, "BOL": 2, "BRA": 3, "CHL": 4, "COL": 5, "ECU": 6,
            "GUY": 7, "PRY": 8, "PER": 9, "SUR": 10, "URY": 11, "VEN": 12,
        },
    },
    "se_asia": {
        "panel": _ROOT / "data" / "se_asia" / "stage1_panel.parquet",
        "iso3_to_id": {
            "BRN": 1, "KHM": 2, "IDN": 3, "LAO": 4,
            "MYS": 5, "MMR": 6, "PHL": 7, "SGP": 8,
            "THA": 9, "TLS": 10, "VNM": 11,
        },
    },
}


def patch_region(region: str, cfg: dict) -> None:
    panel_path = cfg["panel"]
    iso3_to_id = cfg["iso3_to_id"]

    if not panel_path.exists():
        print(f"  SKIP {region}: {panel_path} not found")
        return

    cy = pd.read_parquet(panel_path)
    print(f"\n=== {region} ===")
    print(f"  Loaded {len(cy)} rows, years {cy['year'].min()}–{cy['year'].max()}")

    # Snapshot old pre-2001 cumulative totals for comparison
    pre_old = cy[cy["year"] < 2001].copy()
    print(f"  Old pre-2001 rows: {len(pre_old)}, total pa_expansion_pixels: {pre_old['pa_expansion_pixels'].sum():,.0f} km²")

    # Separate 2001+ rows (keep all columns including political covariates)
    cy_2001plus = cy[cy["year"] >= 2001].copy()

    # Recompute pre-2001 with terrestrial-only fix
    wdpa_present = [p for p in WDPA_CSV_PATHS if p.exists()]
    if not wdpa_present:
        print("  ERROR: no WDPA CSV found — cannot patch pre-2001 rows")
        return

    wdpa_rows = compute_pre2001_expansion(wdpa_present, iso3_to_id, PRE2001_YEAR_RANGE)
    print(f"  New pre-2001 rows: {len(wdpa_rows)}, total pa_expansion_pixels: {wdpa_rows['pa_expansion_pixels'].sum():,.0f} km²")

    # Show biggest changes by country
    id_to_iso3 = {v: k for k, v in iso3_to_id.items()}
    old_by_cty = pre_old.groupby("country_id")["pa_expansion_pixels"].sum().rename("old")
    new_by_cty = wdpa_rows.groupby("country_id")["pa_expansion_pixels"].sum().rename("new")
    cmp = pd.concat([old_by_cty, new_by_cty], axis=1).fillna(0)
    cmp["diff"] = cmp["new"] - cmp["old"]
    cmp["iso3"] = cmp.index.map(id_to_iso3)
    print("  Per-country change (old → new):")
    for _, row in cmp[cmp["diff"].abs() > 10].sort_values("diff").iterrows():
        print(f"    {row['iso3']}: {row['old']:>10,.0f} → {row['new']:>10,.0f}  (Δ {row['diff']:+,.0f} km²)")

    # Rebuild panel: extend_panel_with_wdpa recomputes lags and cumsum
    # and merges political covariates back from cy_2001plus
    patched = extend_panel_with_wdpa(cy_2001plus, wdpa_rows)

    # Sanity: 2001+ pa_expansion_pixels must be unchanged
    orig_2001 = cy_2001plus.set_index(["country_id","year"])["pa_expansion_pixels"]
    patch_2001 = patched[patched["year"] >= 2001].set_index(["country_id","year"])["pa_expansion_pixels"]
    max_drift = (orig_2001 - patch_2001).abs().max()
    if max_drift > 1e-6:
        print(f"  WARN: 2001+ pa_expansion_pixels changed (max drift {max_drift:.2e}) — investigate")
    else:
        print(f"  2001+ pa_expansion_pixels unchanged ✓")

    # Save
    patched.to_parquet(panel_path, index=False)
    print(f"  Written {len(patched)} rows → {panel_path}")


def main() -> None:
    print("Patching stage1 panels: terrestrial-only pre-2001 fix")
    for region, cfg in REGIONS.items():
        patch_region(region, cfg)
    print("\nDone. Re-run model1_expansion.py and model3_expansion.py for updated Stage 1 numbers.")


if __name__ == "__main__":
    main()
