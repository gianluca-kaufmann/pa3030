#!/usr/bin/env python3
"""Parse World Governance Indicators from local Excel file and write wgi.csv.

Reads Government Effectiveness (ge sheet) and Rule of Law (rl sheet) from
data/shared/WGI/wgidataset_with_sourcedata-2025.xlsx.
Outputs data/shared/wgi.csv with columns:
    iso3, year, gov_wgi_ge_est, gov_wgi_rl_est

Usage:
    python scripts/regions/shared/data_prep/build_wgi.py

Source: World Bank Worldwide Governance Indicators (downloaded 2025)
        https://databank.worldbank.org/source/worldwide-governance-indicators
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
XLSX_PATH = _ROOT / "data" / "shared" / "WGI" / "wgidataset_with_sourcedata-2025.xlsx"
OUT_PATH = _ROOT / "data" / "shared" / "wgi.csv"

SHEET_COL = {
    "ge": "gov_wgi_ge_est",
    "rl": "gov_wgi_rl_est",
}


def _parse_sheet(sheet: str, col: str) -> pd.DataFrame:
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    estimate_col = "Governance estimate (approx. -2.5 to +2.5)"
    out = df[["Economy (code)", "Year", estimate_col]].copy()
    out.columns = ["iso3", "year", col]
    out = out.dropna(subset=["iso3", "year", col])
    out["year"] = out["year"].astype(int)
    return out.reset_index(drop=True)


def main() -> None:
    print(f"Reading WGI data from {XLSX_PATH.name}...")
    frames = [_parse_sheet(sheet, col) for sheet, col in SHEET_COL.items()]

    merged = frames[0].merge(frames[1], on=["iso3", "year"], how="outer")
    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(merged)} rows, {merged['iso3'].nunique()} countries)")
    print(f"Year range: {merged['year'].min()}–{merged['year'].max()}")
    print("SA coverage check:")
    sa = ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR", "URY", "VEN"]
    for iso in sa:
        n = len(merged[merged["iso3"] == iso])
        print(f"  {iso}: {n} years")


if __name__ == "__main__":
    main()
