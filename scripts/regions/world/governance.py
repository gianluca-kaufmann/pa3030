"""
Download country-year governance indicators and build CBD 30×30 pledge table.

Sources:
  V-Dem Liberal Democracy Index via Our World in Data (public CSV, no auth required)
  World Bank Worldwide Governance Indicators (WGI), all 6 dimensions (public API, no auth)
  Database of Political Institutions (DPI) 2023, IADB — executive party ideology
  CBD pledge years — hardcoded from High Ambition Coalition and COP15 records

Output:
  <root>/regions/shared/governance_country_year.csv
  Columns: country, year, libdem, cbd_pledge_elapsed,
           wgi_va, wgi_pv, wgi_ge, wgi_rq, wgi_rl, wgi_cc, dpi_exec_ideology

Join key: (country_name, year) — all pixels in a country-year share the same values.

Data Coverage Expansion (decided 2026-08-07, STATUS.md): WGI adds real, distinct signal
beyond V-Dem's single libdem index — 6 separate governance dimensions, incl. Regulatory
Quality and Control of Corruption which V-Dem doesn't capture. DPI's executive ideology is
the same feature PA3030 used as one of its policy_b1-4 columns. Both are known to be highly
intercorrelated with libdem and with each other (WGI dims ρ=0.7-0.9 pairwise) — see
scripts/regions/world/vif_prune.py, run that against any built dataset before training on the
expanded feature set.
"""

import csv, io, os, sys
import requests
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from scripts.regions.world import paths as world_paths  # noqa: E402

OUT_DIR = str(world_paths.COUNTRY_YEAR_DIR)
DL_DIR  = str(world_paths.DOWNLOAD_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DL_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "governance_country_year.csv")

HEADERS = {"User-Agent": "pa-model-research/1.0 (research project)"}

# Widened 1990-2023 (session 53) for Question 2's global WDPA polygon dataset,
# which now spans STATUS_YR 1990-2023 (extended back from this pipeline's original
# 2001-2023 convention — see fetch_wdpa_global.py's docstring for why). V-Dem and
# DPI have real coverage back to 1990; WGI genuinely starts 1996 (biennial before
# 2002) — 1990-1995 WGI columns will be real NaN, not forward-filled backward, and
# that's correct, not a bug.
YEARS = list(range(1990, 2024))

# Session 40 (Axis C): the 3-country allowlist is gone — tiles can land in any country,
# so this now keeps the full global panel for every source. build_dataset.py already
# filters to whatever single country_name a given region/tile needs when it joins this CSV.
COUNTRIES = None  # None = no filter, keep every country each source reports

# Year each country made a formal 30x30 commitment. Everything not listed here falls
# back to 2022 (COP15 Montreal, Dec 2022 — the Kunming-Montreal Global Biodiversity
# Framework's actual global adoption date, itself a real reference point, not a filler
# guess). The entries below are countries confirmed to have pledged earlier via the High
# Ambition Coalition for Nature and People (launched Jan 2021); left deliberately short
# rather than guessing exact join dates for every possible tile country — this feature's
# measured LOO-CV impact has been marginal (+0.002, session 27) so precision here isn't
# worth the research time it would take to verify each country individually.
CBD_PLEDGE_YEAR = {
    "Canada":    2021,
    "Brazil":    2022,
    "Indonesia": 2022,
    "France":    2021,
    "United Kingdom": 2021,
    "Costa Rica": 2021,
}

# ══════════════════════════════════════════════════════════════════════════════
# Part A — V-Dem Liberal Democracy Index (Our World in Data)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Part A: V-Dem liberal democracy index (Our World in Data) ===")

OWID_URL = (
    "https://ourworldindata.org/grapher/liberal-democracy.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)

r = requests.get(OWID_URL, timeout=120, headers=HEADERS)
r.raise_for_status()
vdem_raw = pd.read_csv(io.StringIO(r.text))
print(f"  {len(vdem_raw):,} rows downloaded.")

libdem_col = "Liberal democracy index"
vdem_raw = vdem_raw.rename(columns={"Entity": "country", "Year": "year", libdem_col: "libdem"})
_mask = vdem_raw["country"].isin(COUNTRIES) if COUNTRIES is not None else True
vdem_raw = vdem_raw[_mask & vdem_raw["year"].between(min(YEARS), max(YEARS))]
vdem_raw = vdem_raw[["country", "year", "libdem"]].copy()
print(f"  Filtered to {len(vdem_raw)} rows for {vdem_raw['country'].nunique()} countries")

vdem_lookup = {}  # {(country, year): libdem}
for _, row in vdem_raw.iterrows():
    if pd.notna(row["libdem"]):
        vdem_lookup[(row["country"], int(row["year"]))] = float(row["libdem"])

# ══════════════════════════════════════════════════════════════════════════════
# Part B — World Bank Worldwide Governance Indicators (6 dimensions)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part B: World Bank WGI (6 dimensions) ===")

WGI_INDICATORS = {
    "wgi_va": "GOV_WGI_VA.EST",   # Voice and Accountability
    "wgi_pv": "GOV_WGI_PV.EST",   # Political Stability and Absence of Violence
    "wgi_ge": "GOV_WGI_GE.EST",   # Government Effectiveness
    "wgi_rq": "GOV_WGI_RQ.EST",   # Regulatory Quality
    "wgi_rl": "GOV_WGI_RL.EST",   # Rule of Law
    "wgi_cc": "GOV_WGI_CC.EST",   # Control of Corruption
}

# Our project's country names vs. World Bank's official names — verified 2026-08-07 by
# diffing the full WB country list against every country_name in use across
# _REGION_SETTINGS + regions/tiles/tile_catalog.json + regions/spots/spot_catalog.json.
WGI_NAME_ALIASES = {  # {our_name: world_bank_name}
    "East Timor":   "Timor-Leste",
    "North Korea":  "Korea, Dem. People's Rep.",
    "Russia":       "Russian Federation",
    "Turkey":       "Turkiye",
    "Venezuela":    "Venezuela, RB",
}
_WGI_REVERSE = {v: k for k, v in WGI_NAME_ALIASES.items()}

# Fetch year 2000 too (not in YEARS) so 2001 — a real WGI survey gap, biennial before
# 2002 — can be forward-filled from the nearest prior observation instead of left NaN.
wgi_fetch_years = f"{min(YEARS) - 1}:{max(YEARS)}"
wgi_table = {}  # {(country, year): {col: value}}
for col, code in WGI_INDICATORS.items():
    url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
           f"?format=json&date={wgi_fetch_years}&per_page=20000")
    resp = requests.get(url, timeout=120, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    if not data or len(data) < 2 or not data[1]:
        raise RuntimeError(f"WGI fetch returned no data for {code} — check the API response")
    n = 0
    for row in data[1]:
        country = row["country"]["value"]
        country = _WGI_REVERSE.get(country, country)
        val = row["value"]
        if val is None:
            continue
        yr = int(row["date"])
        wgi_table.setdefault((country, yr), {})[col] = float(val)
        n += 1
    print(f"  {col:8s} ({code}): {n:,} non-null country-year values")

# Forward-fill each country's own series across 2000-2023 (covers the real 1996-2000
# biennial-survey gap landing on 2001, plus any country-specific missing year).
wgi_countries = sorted(set(c for c, _ in wgi_table.keys()))
for country in wgi_countries:
    last = {}
    for yr in range(min(YEARS) - 1, max(YEARS) + 1):
        cell = wgi_table.setdefault((country, yr), {})
        for col in WGI_INDICATORS:
            if col in cell:
                last[col] = cell[col]
            elif col in last:
                cell[col] = last[col]
print(f"  WGI: {len(wgi_countries)} countries, forward-filled across {min(YEARS)}-{max(YEARS)}")

# ══════════════════════════════════════════════════════════════════════════════
# Part C — Database of Political Institutions (DPI 2023, IADB) — executive ideology
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Part C: DPI 2023 executive party ideology (IADB) ===")

DPI_URL = "https://data.iadb.org/file/download/30f45628-8608-4c0c-ad55-bc4c870dd62d"  # "DPI 2023 (CSV Version)"
DPI_CACHE = os.path.join(DL_DIR, "dpi2023_raw.csv")

# Same country-name reconciliation as WGI_NAME_ALIASES above, against DPI's own
# (differently) abbreviated names — verified 2026-08-07 against the same needed-country
# list. North Korea and Dominican Republic are genuinely absent from DPI's 183-country
# panel (not a naming mismatch) — left unmapped, dpi_exec_ideology will be NaN for them,
# median-filled downstream same as any other missing covariate.
DPI_NAME_ALIASES = {  # {our_name: dpi_name}
    "East Timor":         "Timor-Leste",
    "Papua New Guinea":   "P. N. Guinea",
    "South Africa":       "S. Africa",
    "United Kingdom":     "UK",
}
_DPI_REVERSE = {v: k for k, v in DPI_NAME_ALIASES.items()}
DPI_IDEOLOGY_MAP = {"Left": -1.0, "Center": 0.0, "Right": 1.0}  # "", "-999", "0" -> NaN (no info / no exec party)

if os.path.exists(DPI_CACHE):
    print(f"  Using cached: {DPI_CACHE}")
else:
    # data.iadb.org sits behind an AWS WAF bot challenge that intermittently returns
    # 202 + empty body instead of the file (observed 2026-08-07, not a permanent block —
    # retries with backoff clear it within a few attempts).
    import time
    for attempt in range(6):
        print(f"  Downloading DPI 2023 CSV from IADB (attempt {attempt+1}/6) …")
        resp = requests.get(DPI_URL, timeout=120, headers=HEADERS)
        resp.raise_for_status()
        if resp.status_code == 200 and len(resp.content) > 100_000:
            with open(DPI_CACHE, "wb") as f:
                f.write(resp.content)
            break
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(
            "DPI 2023 download kept returning empty (WAF challenge) after 6 attempts. "
            "Manually download 'DPI 2023 (CSV Version)' from "
            "https://data.iadb.org/dataset/the-database-of-political-institutions-dpi-2023 "
            f"and save it to {DPI_CACHE}"
        )

dpi_table = {}  # {(country, year): ideology}
with open(DPI_CACHE, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        try:
            yr = int(row["year"][:4])
        except (ValueError, TypeError, KeyError):
            continue
        if yr not in YEARS:
            continue
        country = row["countryname"]
        country = _DPI_REVERSE.get(country, country)
        code = row.get("execrlc", "")
        if code in DPI_IDEOLOGY_MAP:
            dpi_table[(country, yr)] = DPI_IDEOLOGY_MAP[code]

print(f"  DPI: {len(dpi_table)} country-year rows with a coded executive ideology "
      f"({sum(1 for c, y in dpi_table if y == YEARS[-1])} in {YEARS[-1]})")

# ══════════════════════════════════════════════════════════════════════════════
# Merge — union of countries across all three sources
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Merging V-Dem + WGI + DPI ===")

all_countries = sorted(set(c for c, _ in vdem_lookup.keys())
                        | set(c for c, _ in wgi_table.keys())
                        | set(c for c, _ in dpi_table.keys()))

rows = []
for country in all_countries:
    pledge_yr = CBD_PLEDGE_YEAR.get(country, 2022)
    for yr in YEARS:
        row = {
            "country":            country,
            "year":               yr,
            "libdem":             vdem_lookup.get((country, yr), np.nan),
            "cbd_pledge_elapsed": yr - pledge_yr,
        }
        wgi_cell = wgi_table.get((country, yr), {})
        for col in WGI_INDICATORS:
            row[col] = wgi_cell.get(col, np.nan)
        row["dpi_exec_ideology"] = dpi_table.get((country, yr), np.nan)
        rows.append(row)

df = pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)

value_cols = ["libdem", "cbd_pledge_elapsed"] + list(WGI_INDICATORS.keys()) + ["dpi_exec_ideology"]
print(f"\nGovernance table: {len(df)} rows, {df['country'].nunique()} countries")
print("Non-null coverage by column:")
print((df[value_cols].notna().mean() * 100).round(1).astype(str) + "%")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
