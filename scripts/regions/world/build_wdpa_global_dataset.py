"""
Merge WDPA attribute (fetch_wdpa_global.py) + zonal-stat (compute_wdpa_zonal_stats.py)
checkpoints into the single Question 2 foundation dataset: one row per WDPA
designation event, globally, with covariates and country-year governance/finance
joins. See STATUS.md "Active: Question 2" for why this dataset exists.

Steps:
  1. Concatenate all regions/global/wdpa_raw/attrs_<iso3>.parquet + matching
     zonal_<iso3>.parquet, joined on SITE_ID.
  2. Resolve country_name/continent via the Natural Earth crosswalk
     (regions/global/country_crosswalk.parquet, built by fetch_wdpa_global.py).
  3. Join regions/shared/governance_country_year.csv on (country_name, status_yr)
     — after dropping the 12 non-country aggregate rows OWID/V-Dem reports
     alongside real countries (exact literal list below; NOT a substring filter,
     which would wrongly catch real countries like Egypt/Saudi Arabia/UAE/South
     Africa that contain income-group-like substrings).
  4. Join regions/shared/conservation_finance_country_year.csv on
     (country_name, status_yr). Add is_conservation_finance_recipient (True if
     the country appears in that file for ANY year) instead of imputing donor
     countries' structural non-recipient status to 0 — conflating "never a
     recipient" with "recipient, missing this year" would corrupt the signal.
  4b. Join regions/shared/redd_entry_country_year.csv on country_name (country-
     level, not country-year — see redd_entry.py's docstring). Adds
     redd_entry_year (NaN if the country has never hosted a carbon-offset
     project) and n_carbon_projects — pillar 1's staggered-treatment-timing
     variable (STATUS.md "Active: Question 2", pillar 1).
  5. Discovery/confirmation split, single `split` column: countries with >=10
     qualifying polygons get a random 60/40 split; countries with <10 get a
     whole-country deterministic hash assignment (avoids a 2-3-polygon country
     leaking near-duplicate events across both sides).

Output: regions/global/wdpa_polygon_dataset.parquet

Run (after fetch_wdpa_global.py and compute_wdpa_zonal_stats.py):
    python3 scripts/regions/world/build_wdpa_global_dataset.py
"""

import hashlib
import os, sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from scripts.regions.world import paths as world_paths  # noqa: E402

RAW_DIR = str(world_paths.RAW_DIR)
CROSSWALK_PATH = str(world_paths.CROSSWALK_PATH)
GOVERNANCE_PATH = str(world_paths.GOVERNANCE_PATH)
FINANCE_PATH = str(world_paths.FINANCE_PATH)
REDD_PATH = str(world_paths.REDD_PATH)
OUTPUT_PATH = str(world_paths.EVENT_DATASET_PATH)

SEED = 42
SPLIT_MIN_COUNT = 10
DISCOVERY_FRAC = 0.6

# Exact non-country aggregate labels present in governance_country_year.csv's
# `country` column (V-Dem/OWID reports these alongside real countries) — an
# explicit literal list, not a substring filter, because substring rules like
# "contains 'income'" or "contains 'Arab'" would also drop real countries
# (Egypt, Saudi Arabia, UAE, South Africa, etc.).
NON_COUNTRY_AGGREGATES = {
    "Africa", "Africa (population-weighted)",
    "Asia", "Asia (population-weighted)",
    "Europe", "Europe (population-weighted)",
    "North America", "North America (population-weighted)",
    "South America", "South America (population-weighted)",
    "World", "World (population-weighted)",
}

# Natural-Earth NAME -> governance_country_year.csv `country` spelling, for the
# small number of real mismatches. Verified directly against both files' actual
# string content this session (session 53) — NOT guessed against a generic World
# Bank naming convention, which turned out to be wrong: governance_country_year.csv
# uses V-Dem/OWID's colloquial names (e.g. "Russia", "Vietnam", "Tanzania" already
# match Natural Earth's NAME with no mapping needed at all). Real mismatches only:
NAME_CROSSWALK = {
    "United States of America": "United States",
    "Dem. Rep. Congo": "Congo, Dem. Rep.",
    "Congo": "Congo, Rep.",   # NE "Congo" = Republic of Congo (COG); governance_country_year.csv
                               # also has bare "Congo" and "Congo (DRC)" from unreconciled
                               # upstream sources — this mapping is deliberately the
                               # unambiguous "Congo, Rep." entry, not a fix for that mess.
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Timor-Leste": "East Timor",
}


def load_wdpa_events():
    attrs_files = sorted(f for f in os.listdir(RAW_DIR) if f.startswith("attrs_") and f.endswith(".parquet"))
    frames = []
    for fname in attrs_files:
        iso3 = fname[len("attrs_"):-len(".parquet")]
        attrs_path = os.path.join(RAW_DIR, fname)
        zonal_path = os.path.join(RAW_DIR, f"zonal_{iso3}.parquet")
        attrs = pd.read_parquet(attrs_path)
        if attrs.empty:
            continue
        if os.path.exists(zonal_path):
            zonal = pd.read_parquet(zonal_path)
            if not zonal.empty:
                attrs = attrs.merge(zonal, on="SITE_ID", how="left")
            else:
                print(f"    WARNING: {zonal_path} exists but is empty — {iso3} has no zonal covariates")
        else:
            print(f"    WARNING: no zonal-stats file for {iso3} yet — covariates will be all-NaN")
        frames.append(attrs)
    if not frames:
        raise RuntimeError(f"No non-empty attrs_*.parquet found in {RAW_DIR} — run fetch_wdpa_global.py first")
    df = pd.concat(frames, ignore_index=True)

    # A country using bounds_fallback_exact legitimately re-pulls border-adjacent
    # parks tagged with a NEIGHBORING country's ISO3 (that's the intended behavior
    # — see fetch_wdpa_global.py's docstring) — but if that neighbor is ALSO in
    # the pilot set and fetched via its own correct ISO3 query, the same SITE_ID
    # lands in both country files. Confirmed live (session 53): Botswana's
    # bounds_fallback_exact pull duplicated 2 SITE_IDs already present in South
    # Africa's own attrs_ZAF.parquet. Keep the "iso3"-partition copy (the
    # administratively correct source) over any bounds_fallback duplicate.
    n_before = len(df)
    df["_method_rank"] = (df["partition_method"] != "iso3").astype(int)  # iso3=0 sorts first
    df = df.sort_values("_method_rank").drop_duplicates("SITE_ID", keep="first")
    df = df.drop(columns=["_method_rank"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"    Dropped {n_dropped} duplicate SITE_ID row(s) (kept the iso3-partition copy over bounds_fallback)")
    return df


def resolve_country(df):
    # Only country_name/continent — iso3 is dropped after the merge since df
    # already has the authoritative ISO3 from WDPA itself; keeping both would
    # leave two differently-cased columns meaning the same thing.
    crosswalk = pd.read_parquet(CROSSWALK_PATH)[["iso3", "country_name", "continent"]].drop_duplicates("iso3")
    df = df.merge(crosswalk, left_on="ISO3", right_on="iso3", how="left", suffixes=("", "_cw"))
    df = df.drop(columns=["iso3"], errors="ignore")
    n_missing = df["country_name"].isna().sum()
    if n_missing:
        print(f"    {n_missing} row(s) with ISO3 not found in Natural Earth crosswalk (country_name will be NaN)")
    return df


def join_governance(df):
    gov = pd.read_csv(GOVERNANCE_PATH)
    gov = gov[~gov["country"].isin(NON_COUNTRY_AGGREGATES)]
    gov = gov.rename(columns={"country": "gov_country_name", "year": "status_yr"})

    join_name = df["country_name"].map(lambda n: NAME_CROSSWALK.get(n, n))
    df = df.assign(_join_name=join_name)
    merged = df.merge(gov, left_on=["_join_name", "STATUS_YR"], right_on=["gov_country_name", "status_yr"],
                       how="left")
    # left_on/right_on with mismatched-case key names ("STATUS_YR" vs "status_yr")
    # both survive the merge as separate columns — drop the redundant right-side
    # key, keeping only the original STATUS_YR.
    merged = merged.drop(columns=["_join_name", "gov_country_name", "status_yr"], errors="ignore")
    return merged


def join_finance(df):
    fin = pd.read_csv(FINANCE_PATH)
    fin_countries = set(fin["country"].unique())
    fin = fin.rename(columns={"country": "fin_country_name", "year": "status_yr"})

    join_name = df["country_name"].map(lambda n: NAME_CROSSWALK.get(n, n))
    df = df.assign(_join_name=join_name)
    df["is_conservation_finance_recipient"] = df["_join_name"].isin(fin_countries)
    merged = df.merge(fin, left_on=["_join_name", "STATUS_YR"], right_on=["fin_country_name", "status_yr"],
                       how="left")
    merged = merged.drop(columns=["_join_name", "fin_country_name", "status_yr"], errors="ignore")
    return merged


def join_redd_entry(df):
    # Country-level (not country-year) join. redd_entry.py's `Country` field is
    # already Natural-Earth-NAME-like (aliased for its 3 real mismatches at
    # source) — join directly on country_name, no NAME_CROSSWALK needed here
    # (unlike governance/finance, whose upstream sources use a different naming
    # convention).
    redd = pd.read_csv(REDD_PATH)
    redd = redd.rename(columns={"country": "redd_country_name"})
    merged = df.merge(redd, left_on="country_name", right_on="redd_country_name", how="left")
    merged = merged.drop(columns=["redd_country_name"], errors="ignore")
    return merged


def _stable_seed(key):
    """Deterministic per-key seed — Python's built-in hash() is randomized per
    process (PYTHONHASHSEED) for strings, so it can't be used for a reproducible
    split across runs."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def assign_split(df):
    counts = df["ISO3"].value_counts()

    split = pd.Series(index=df.index, dtype=object)
    for iso3, grp in df.groupby("ISO3"):
        n = counts.get(iso3, 0)
        if n >= SPLIT_MIN_COUNT:
            idx = grp.index.to_numpy().copy()
            local_rng = np.random.default_rng(SEED + _stable_seed(iso3))
            local_rng.shuffle(idx)
            n_discovery = int(round(len(idx) * DISCOVERY_FRAC))
            split.loc[idx[:n_discovery]] = "discovery"
            split.loc[idx[n_discovery:]] = "confirmation"
        else:
            h = int(hashlib.sha256(iso3.encode()).hexdigest(), 16)
            side = "discovery" if (h % 100) < (DISCOVERY_FRAC * 100) else "confirmation"
            split.loc[grp.index] = side
    return split


def main():
    print("=== Step 1: load + merge attrs + zonal-stat checkpoints ===")
    df = load_wdpa_events()
    print(f"  {len(df):,} events loaded across {df['ISO3'].nunique()} countries\n")

    print("=== Step 2: resolve country_name/continent ===")
    df = resolve_country(df)

    print("\n=== Step 3: join governance ===")
    df = join_governance(df)

    print("\n=== Step 4: join conservation finance ===")
    df = join_finance(df)

    print("\n=== Step 4b: join REDD+/carbon-market entry year ===")
    df = join_redd_entry(df)
    n_treated = df["redd_entry_year"].notna().sum()
    print(f"  {n_treated:,}/{len(df):,} events ({n_treated / len(df):.1%}) in a country with "
          f"a real carbon-offset project history")

    print("\n=== Step 5: discovery/confirmation split ===")
    df["split"] = assign_split(df)
    print(df["split"].value_counts(normalize=True).round(3).to_dict())

    df["marine_frac"] = df["GIS_M_AREA"] / df["GIS_AREA"].replace(0, np.nan)

    df = df.rename(columns={
        "SITE_ID": "wdpaid", "SITE_PID": "site_pid", "NAME": "name", "NAME_ENG": "name_eng",
        "ISO3": "iso3", "PRNT_ISO3": "parent_iso3", "STATUS": "status", "STATUS_YR": "status_yr",
        "IUCN_CAT": "iucn_cat", "DESIG_ENG": "desig_eng", "DESIG_TYPE": "desig_type",
        "GOV_TYPE": "gov_type", "GIS_AREA": "gis_area_km2", "GIS_M_AREA": "gis_m_area_km2",
        "lon": "centroid_lon", "lat": "centroid_lat",
    })
    df = df.drop(columns=["iso3_cw"], errors="ignore")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")
    print(f"  Rows: {len(df):,}  Columns: {len(df.columns)}")
    print(f"  Countries: {df['country_name'].nunique()}  Continents: {df['continent'].nunique()}")
    print("\nNext:")
    print(f"  python3 scripts/regions/world/audit_dataset.py --path {OUTPUT_PATH} --group-col iso3 --key-cols wdpaid --strict")
    print(f"  python3 scripts/regions/world/describe_wdpa_global_dataset.py --path {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
