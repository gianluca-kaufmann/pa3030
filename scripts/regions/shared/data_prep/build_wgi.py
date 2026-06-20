#!/usr/bin/env python3
"""Fetch World Governance Indicators from World Bank API and write wgi.csv.

Downloads Government Effectiveness (GE.EST) and Rule of Law (RL.EST) for all
countries, 1996–2024. Outputs data/shared/wgi.csv with columns:
    iso3, year, gov_wgi_ge_est, gov_wgi_rl_est

Usage:
    python scripts/regions/shared/data_prep/build_wgi.py

Requires: requests (stdlib-adjacent; always available)
Source: World Bank Worldwide Governance Indicators
        https://databank.worldbank.org/source/worldwide-governance-indicators
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
OUT_PATH = _ROOT / "data" / "shared" / "wgi.csv"

WB_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"
INDICATORS = {
    "GE.EST": "gov_wgi_ge_est",
    "RL.EST": "gov_wgi_rl_est",
}
YEAR_RANGE = "1996:2024"
PER_PAGE = 10_000


def _fetch_indicator(indicator: str) -> pd.DataFrame:
    col = INDICATORS[indicator]
    url = WB_API.format(indicator=indicator)
    params = {"format": "json", "per_page": PER_PAGE, "date": YEAR_RANGE}

    rows = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        meta, records = data[0], data[1]
        for r in records:
            if r["value"] is not None and r["countryiso3code"]:
                rows.append({
                    "iso3": r["countryiso3code"],
                    "year": int(r["date"]),
                    col: float(r["value"]),
                })
        total_pages = meta["pages"]
        print(f"  {indicator} page {page}/{total_pages} ({len(records)} records)")
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)

    return pd.DataFrame(rows)


def main() -> None:
    print(f"Fetching WGI data from World Bank API ({YEAR_RANGE})...")
    frames = []
    for indicator in INDICATORS:
        df = _fetch_indicator(indicator)
        frames.append(df)

    ge = frames[0]
    rl = frames[1]
    merged = ge.merge(rl, on=["iso3", "year"], how="outer")
    merged = merged.sort_values(["iso3", "year"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(merged)} rows, {merged['iso3'].nunique()} countries)")
    print(f"Year range: {merged['year'].min()}–{merged['year'].max()}")
    print(f"SA coverage check:")
    sa = ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR", "URY", "VEN"]
    for iso in sa:
        n = len(merged[merged["iso3"] == iso])
        print(f"  {iso}: {n} years")


if __name__ == "__main__":
    main()
