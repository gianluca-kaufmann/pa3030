#!/usr/bin/env python3
"""Verification/inspection script for the election-cycle feature.

The live computation happens in stage1_panel.compute_years_to_next_election(),
which reads V-Dem CY Core directly.  No intermediate CSV is written or needed.

Run this script to inspect what election years V-Dem identifies per country
or to spot-check the derived values.

Source
------
Coppedge, M. et al. (2024). V-Dem [Country-Year/Country-Date] Dataset v15.
  Varieties of Democracy (V-Dem) Project. https://doi.org/10.23696/vdemds24
Variable codebook IDs: v2eltype_6 (presidential/executive direct), v2eltype_0 (legislative).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ALL_COUNTRIES = [
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PAN", "PER", "PRY", "SUR", "URY", "VEN",
    "USA",
    "BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM",
]


def main() -> None:
    from scripts.regions.shared.stage1_panel import compute_years_to_next_election
    ec = compute_years_to_next_election(ALL_COUNTRIES)

    print("Election years identified (years_to_next_election == 0):")
    for iso3, grp in ec.groupby("iso3"):
        elec_yrs = sorted(grp.loc[grp["years_to_next_election"] == 0, "year"].tolist())
        if elec_yrs:
            print(f"  {iso3}: {elec_yrs}")
        else:
            print(f"  {iso3}: (none — no elections in V-Dem)")


if __name__ == "__main__":
    main()
