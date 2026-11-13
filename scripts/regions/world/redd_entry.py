"""
Fetch country-level REDD+/carbon-market entry year — the treatment-timing variable
for Question 2's pillar 1 (causal backbone: "money rewires where protection happens",
STATUS.md "Active: Question 2", locked-in 2026-08-22). A country's `redd_entry_year`
is the year its first carbon-offset project (REDD+ avoided-deforestation, afforestation/
reforestation, or improved forest management) entered the market — the staggered
treatment date for the Callaway & Sant'Anna group-time ATT design.

Source: same GEE asset `redd.py` already uses for `dist_redd_km`,
`projects/sat-io/open-datasets/CARBON-OFFSET-PROJECTS-GLOBAL` (peer-reviewed,
Scientific Data 2025, 533 projects/50 countries) — no new data source, no account
gate. Verified live before building this (session 57): `Project St` (project start
date) is populated for all 533 rows, real MM/DD/YYYY strings, years 1995-2026, no
nulls; `Country` field is populated for all rows and already Natural-Earth-NAME-like
(47/50 match `regions/global/country_crosswalk.parquet`'s `country_name` directly —
only Central African Republic / Democratic Republic of the Congo / Republic of the
Congo need aliasing, confirmed via a real diff against the crosswalk, not guessed).

`redd_entry_year` = min(year(`Project St`)) per country, across all project types
(AD/ARR/IFM) — matches the existing `dist_redd_km` feature's broad "REDD+/carbon-
market" framing (redd.py's docstring), not narrowed to AD-only. `n_carbon_projects`
is included so a downstream model can distinguish "one small pilot project" from
"an established carbon-market presence" if that distinction turns out to matter.

Output:
  <root>/regions/shared/redd_entry_country_year.csv
  Columns: country, redd_entry_year, n_carbon_projects

Join key: country (Natural Earth `country_name` spelling) — join directly, no
NAME_CROSSWALK needed (unlike governance/conservation-finance, whose source data
uses a different naming convention; this asset's `Country` field already matches
Natural Earth's NAME for 47/50 countries, with the 3 exceptions aliased below).

This is a country-level (not country-year) table: every year for a country before
its redd_entry_year has no row (never-treated-yet), consistent with the staggered-
adoption design pillar 1 needs treatment timing for, not a treatment intensity that
varies year to year (that's what conservation_finance.py's continuous flow measures).
"""

import os, sys
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from scripts.regions.world import paths as world_paths  # noqa: E402

OUT_DIR = str(world_paths.COUNTRY_YEAR_DIR)
OUT_PATH = os.path.join(OUT_DIR, "redd_entry_country_year.csv")
os.makedirs(OUT_DIR, exist_ok=True)

GEE_ASSET = "projects/sat-io/open-datasets/CARBON-OFFSET-PROJECTS-GLOBAL"

# Asset `Country` -> Natural Earth `country_name` — the 3 real mismatches, found by
# diffing the asset's actual country list against country_crosswalk.parquet
# (session 57), not guessed.
ASSET_NAME_ALIASES = {
    "Democratic Republic of the Congo": "Dem. Rep. Congo",
    "Republic of the Congo": "Congo",
    "Central African Republic": "Central African Rep.",
}

import ee
ee.Initialize()
print("GEE initialized.")

print("=== REDD+/carbon-market entry year: CARBON-OFFSET-PROJECTS-GLOBAL ===")
fc = ee.FeatureCollection(GEE_ASSET)
n = fc.size().getInfo()
print(f"  {n} projects in the global collection")

rows = fc.reduceColumns(
    ee.Reducer.toList(3), ["Country", "Project St", "Project Ty"]
).get("list").getInfo()
print(f"  {len(rows)} (country, start_date, type) rows pulled")

df = pd.DataFrame(rows, columns=["country_raw", "project_start_raw", "project_type"])
df["project_start"] = pd.to_datetime(df["project_start_raw"], format="mixed", errors="coerce")
n_bad_date = df["project_start"].isna().sum()
if n_bad_date:
    print(f"  WARNING: {n_bad_date} row(s) with unparseable Project St — dropped")
df = df.dropna(subset=["project_start"])
df["year"] = df["project_start"].dt.year
df["country"] = df["country_raw"].map(lambda c: ASSET_NAME_ALIASES.get(c, c))

out = df.groupby("country", as_index=False).agg(
    redd_entry_year=("year", "min"),
    n_carbon_projects=("year", "size"),
)
out = out.sort_values("country").reset_index(drop=True)

print(f"\nREDD+ entry table: {len(out)} countries, entry years "
      f"{out['redd_entry_year'].min()}-{out['redd_entry_year'].max()}")
print(out.to_string(index=False))

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
