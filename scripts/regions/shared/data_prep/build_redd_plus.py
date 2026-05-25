#!/usr/bin/env python3
"""Build redd_plus.csv from documented official enrollment events.

Definition
----------
redd_plus_enrolled = 1 from the year a country formally committed to REDD+
through one of the three primary mechanisms:
  1. Bilateral agreement with a donor country (Norway, Germany, UK) that
     establishes REDD+ result-based payments;
  2. FCPF Readiness Preparation Proposal (R-PP) endorsed at a FCPF Participants
     Committee meeting; or
  3. UN-REDD national programme approved.
The variable stays 1 once activated (no back-tracking).

Sources (all public, document-level URLs noted per country)
-----------------------------------------------------------
FCPF R-PP decisions: https://www.forestcarbonpartnership.org/country-profiles
UN-REDD programme:   https://www.un-redd.org/partner-countries
Amazon Fund:         https://www.amazonfund.gov.br/en/home/

Each entry below carries a short provenance note so the claim can be verified.
Countries with no REDD+ program through 2024 are coded 0 throughout.

Countries marked # TO VERIFY need cross-checking against the cited URL before
the paper is submitted; the date used is a conservative estimate.

Output
------
data/shared/redd_plus.csv — columns: iso3, year, redd_plus_enrolled
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT_PATH = _ROOT / "data" / "shared" / "redd_plus.csv"

YEAR_MIN, YEAR_MAX = 2001, 2024

# Enrollment year = first year coded as 1.
# Source provenance is in the comment on each line.
# For countries not listed here the default is 0 (not enrolled).
ENROLLMENT_YEARS: dict[str, int | None] = {
    # ── South America ─────────────────────────────────────────────────────────
    # BRA: Amazon Fund first payment from Norway, 2009; formally operational 2010.
    # Source: Amazon Fund Annual Report 2010. https://www.amazonfund.gov.br
    "BRA": 2010,

    # GUY: Norway-Guyana GRIF Letter of Intent signed 09 Nov 2009.
    # Source: Norwegian Ministry of Foreign Affairs press release 2009-11-09.
    # https://www.regjeringen.no/en/aktuelt/climate-forest-initiative-guyana/id582555/
    "GUY": 2009,

    # COL: FCPF R-PP endorsed at PC9, May 2012.
    # Source: FCPF Colombia country page.
    # https://www.forestcarbonpartnership.org/country/colombia  # TO VERIFY exact PC date
    "COL": 2012,

    # PER: FCPF R-PP endorsed at PC8, March 2011.
    # Source: FCPF Peru country page.
    # https://www.forestcarbonpartnership.org/country/peru  # TO VERIFY exact PC date
    "PER": 2011,

    # ECU: FCPF R-PP endorsed at PC9, June 2011.
    # Source: FCPF Ecuador country page.
    # https://www.forestcarbonpartnership.org/country/ecuador  # TO VERIFY exact PC date
    "ECU": 2011,

    # Not enrolled (no FCPF/UN-REDD program through 2024):
    # ARG, BOL (explicitly opposed REDD+ under Morales), CHL, PRY, SUR, URY, VEN, PAN
    # ── SE Asia ───────────────────────────────────────────────────────────────
    # IDN: Norway-Indonesia LoI signed 26 May 2010; moratorium decree Jan 2011.
    # Source: Indonesian Ministry of Forestry + Norwegian MFA, 2010-05-26.
    # https://www.regjeringen.no/en/aktuelt/indonesia-and-norway-climate-forests/id605709/
    "IDN": 2010,

    # VNM: UN-REDD Viet Nam pilot programme Phase 1 launched 2009.
    # Source: UN-REDD Programme, Viet Nam country page.
    # https://www.un-redd.org/partner-countries/viet-nam  # TO VERIFY exact launch year
    "VNM": 2009,

    # KHM: UN-REDD national programme approved 2011.
    # Source: UN-REDD Programme, Cambodia country page.
    # https://www.un-redd.org/partner-countries/cambodia  # TO VERIFY
    "KHM": 2011,

    # PHL: FCPF REDD+ Readiness country, R-PP endorsed 2013.
    # Source: FCPF Philippines country page.
    # https://www.forestcarbonpartnership.org/country/philippines  # TO VERIFY
    "PHL": 2013,

    # MMR: UN-REDD partner since 2013. (Note: military coup 2021 may affect status.)
    # Source: UN-REDD Programme, Myanmar country page.
    # https://www.un-redd.org/partner-countries/myanmar  # TO VERIFY
    "MMR": 2013,

    # MYS: UN-REDD partner since 2014.
    # Source: UN-REDD Programme, Malaysia country page.
    # https://www.un-redd.org/partner-countries/malaysia  # TO VERIFY
    "MYS": 2014,

    # LAO: UN-REDD national programme, Phase 1 approved ~2011.
    # Source: UN-REDD Programme, Lao PDR country page.
    # https://www.un-redd.org/partner-countries/laos  # TO VERIFY
    "LAO": 2011,

    # THA: UN-REDD partner since 2014 (later than Myanmar/Malaysia).
    # Source: UN-REDD Programme, Thailand country page.
    # https://www.un-redd.org/partner-countries/thailand  # TO VERIFY
    "THA": 2014,

    # Not enrolled: BRN, SGP (not tropical developing), TLS (no formal program found)
}

ALL_COUNTRIES = [
    "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PAN", "PER", "PRY", "SUR", "URY", "VEN",
    "USA",
    "BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM",
]


def build_redd_plus(countries: list[str]) -> pd.DataFrame:
    rows = []
    for iso3 in countries:
        enroll_yr = ENROLLMENT_YEARS.get(iso3)
        for yr in range(YEAR_MIN, YEAR_MAX + 1):
            enrolled = 1 if (enroll_yr is not None and yr >= enroll_yr) else 0
            rows.append({"iso3": iso3, "year": yr, "redd_plus_enrolled": enrolled})
    return pd.DataFrame(rows).sort_values(["iso3", "year"]).reset_index(drop=True)


def main() -> None:
    df = build_redd_plus(ALL_COUNTRIES)

    print("REDD+ enrollment summary:")
    enrolled = df[df["redd_plus_enrolled"] == 1].groupby("iso3")["year"].min()
    for iso3, yr in enrolled.items():
        print(f"  {iso3}: enrolled from {yr}")
    not_enrolled = [c for c in ALL_COUNTRIES if c not in enrolled.index]
    print(f"  Not enrolled: {not_enrolled}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(df)} rows)")
    print("\nNOTE: Entries marked # TO VERIFY in the source dict must be cross-checked")
    print("against the cited FCPF and UN-REDD country pages before paper submission.")


if __name__ == "__main__":
    main()
