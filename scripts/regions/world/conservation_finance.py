"""
Download country-year conservation finance flows (session 50, STATUS.md "expand the
political-covariate data itself" — the top-priority candidate after three independent
architectures in the methodology bake-off all came up empty over the existing WGI/DPI/
libdem covariate class, see STATUS.md "Session 50").

Source:
  OECD DAC Creditor Reporting System (CRS), Rio marker for biodiversity — project-level
  aid activities by recipient country and year, disbursements, current prices. Public,
  no auth, via the OECD SDMX REST API (wrapped by the `oda_reader` package, PyPI). Spot-
  checked real and working (session 50): confirmed the SDMX endpoint responds, confirmed
  a genuine `biodiversity` marker column (0=not targeted, 1=significant objective,
  2=principal objective) populated at project level, confirmed real project titles for
  Brazil ("SUPPORTING COMMUNITY FOREST MANAGEMENT...", "BUILDING SUPPORT FOR LOW CARBON
  AGRICULTURE...") match the hypothesis this feature is meant to test.

Why this is conceptually distinct from what governance.py already has: WGI/DPI/libdem
measure governance QUALITY (slow-moving, close to static within a country's panel
window); CBD pledge year is a COMMITMENT signal. Conservation-finance aid flow is a
RESOURCE AVAILABILITY signal that varies year to year, generalizing the existing
REDD+/carbon-market "logic B" hypothesis (session 44, dist_redd_km) from one narrow
financing channel (voluntary carbon markets) to the broader class of "does external
conservation money actually flowing into a country in a given year predict designation
timing/location."

Coverage (confirmed real, not a bug): OECD DAC CRS is aid-RECIPIENT-side reporting —
donor countries in this project's pool (Canada/BC, Australia, France, Spain, Sweden)
have ZERO recipient-side rows, by design (verified: querying the SDMX API directly for
these 5 ISO3 codes alongside the other 22 pool countries returns real project-level
rows for all 22 developing/transition-economy recipients and 0 rows for the 5 donors).
This mirrors the already-documented DPI zero-coverage gap for a different subset of
countries (governance.py) — both are median-imputed downstream like any other missing
covariate, not treated as failures to fix.

Output:
  <root>/regions/shared/conservation_finance_country_year.csv
  Columns: country, year, biodiv_aid_usd_million, biodiv_aid_log1p,
           biodiv_aid_n_projects, biodiv_aid_share_of_oda

Join key: (country_name, year) — same convention as governance_country_year.csv.
"""

import os, sys

import numpy as np
import pandas as pd
import oda_reader as odr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from scripts.regions.world import paths as world_paths  # noqa: E402

OUT_DIR = str(world_paths.COUNTRY_YEAR_DIR)
OUT_PATH = os.path.join(OUT_DIR, "conservation_finance_country_year.csv")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(2001, 2024))

# All 27 real countries this project uses (BC/Brazil/Indonesia + 26 unique spot ISO3
# codes from spot_catalog.json — 4 of those repeat across 2 spots each, deduplicated
# here since CRS is queried once per country, not once per spot).
POOL_ISO3 = sorted({
    "CAN", "BRA", "IDN",  # BC, Brazil, Indonesia — not spots, added explicitly
    "AGO", "AUS", "BLR", "CHL", "COL", "DOM", "ESP", "FRA", "IND", "KHM", "LKA",
    "MEX", "MMR", "MYS", "NAM", "OMN", "PRK", "SAU", "SWE", "TKM", "TLS", "UKR",
    "VEN", "ZAF",
})

# CRS recipient_name vs. this project's country_name convention (spot_catalog.json /
# config.py) — same alias-reconciliation pattern as governance.py's WGI_NAME_ALIASES/
# DPI_NAME_ALIASES, verified by diffing CRS's actual returned recipient_name values
# against every country_name in spot_catalog.json (session 50).
CRS_NAME_ALIASES = {  # {crs_name: our_name}
    "Democratic People’s Republic of Korea": "North Korea",
    "Timor-Leste": "East Timor",
}

print("=== Conservation finance: OECD CRS Rio marker for biodiversity ===")
print(f"  Querying {len(POOL_ISO3)} pool countries, {min(YEARS)}-{max(YEARS)}, "
      f"disbursements, current prices (no auth, SDMX REST API)...")

raw = odr.download_crs(
    start_year=min(YEARS), end_year=max(YEARS),
    filters={"recipient": POOL_ISO3, "flow_type": "D", "price_base": "V"},
)
print(f"  {len(raw):,} project-activity rows returned")

raw["country"] = raw["recipient_name"].map(lambda n: CRS_NAME_ALIASES.get(n, n))
covered = sorted(raw["country"].unique())
print(f"  Countries with real recipient-side data ({len(covered)}/{len(POOL_ISO3)}): {covered}")

# biodiversity marker: 0 = not targeted, 1 = significant objective, 2 = principal
# objective, <NA> = not assessed / not a Rio-relevant sector. 1 and 2 both count as
# real biodiversity-relevant finance (matches standard OECD Rio-marker usage).
raw["is_biodiv"] = raw["biodiversity"].isin([1, 2])

by_country_year = raw.groupby(["country", "year"], as_index=False).agg(
    total_oda_usd_million=("value", "sum"),
)
biodiv = raw[raw["is_biodiv"]].groupby(["country", "year"], as_index=False).agg(
    biodiv_aid_usd_million=("value", "sum"),
    biodiv_aid_n_projects=("value", "size"),
)

merged = by_country_year.merge(biodiv, on=["country", "year"], how="left")
merged["biodiv_aid_usd_million"] = merged["biodiv_aid_usd_million"].fillna(0.0)
merged["biodiv_aid_n_projects"] = merged["biodiv_aid_n_projects"].fillna(0).astype(int)
merged["biodiv_aid_share_of_oda"] = np.where(
    merged["total_oda_usd_million"] > 0,
    merged["biodiv_aid_usd_million"] / merged["total_oda_usd_million"],
    np.nan,
)
merged["biodiv_aid_log1p"] = np.log1p(merged["biodiv_aid_usd_million"])

# Full (country, year) grid for every pool country — years/countries with no CRS rows
# at all get a real, honest zero (no aid moved that year), not a missing value; the
# share-of-oda column stays NaN there (0/0 is undefined, not "0% of nothing").
all_countries = sorted(set(raw["country"].unique()))
grid = pd.DataFrame([(c, y) for c in all_countries for y in YEARS],
                     columns=["country", "year"])
df = grid.merge(merged, on=["country", "year"], how="left")
for col in ("total_oda_usd_million", "biodiv_aid_usd_million", "biodiv_aid_n_projects",
            "biodiv_aid_log1p"):
    df[col] = df[col].fillna(0.0 if col != "biodiv_aid_n_projects" else 0)
df["biodiv_aid_n_projects"] = df["biodiv_aid_n_projects"].astype(int)

value_cols = ["biodiv_aid_usd_million", "biodiv_aid_log1p", "biodiv_aid_n_projects",
              "biodiv_aid_share_of_oda"]
out = df[["country", "year"] + value_cols].sort_values(["country", "year"]).reset_index(drop=True)

print(f"\nConservation finance table: {len(out)} rows, {out['country'].nunique()} countries "
      f"(of {len(POOL_ISO3)} pool countries queried — the gap is donor-side countries "
      f"with genuinely zero recipient-side CRS rows, e.g. Canada/Australia/France/Spain/"
      f"Sweden, see module docstring)")
print("Non-null coverage by column:")
print((out[value_cols].notna().mean() * 100).round(1).astype(str) + "%")
print(f"Total biodiversity-marked disbursements across the pool, {min(YEARS)}-{max(YEARS)}: "
      f"${out['biodiv_aid_usd_million'].sum():,.0f}M")

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
