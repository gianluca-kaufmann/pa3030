"""
Simple, short, human-readable inspection of the Question 2 WDPA global polygon
dataset — a small add-on separate from audit_dataset.py (which is a QA
pass/fail gate). This is for understanding what got built and how, not for
gating a pipeline step: structure, build mechanism (which stage produced each
column), and completeness, all at a glance.

Run:
    python3 scripts/regions/world/describe_wdpa_global_dataset.py
    python3 scripts/regions/world/describe_wdpa_global_dataset.py --path <parquet>
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from scripts.regions.world import paths as world_paths  # noqa: E402

DEFAULT_PATH = str(world_paths.EVENT_DATASET_PATH)

# Column -> which build stage produced it, for the "build mechanism" section.
COLUMN_SOURCE = {
    "wdpaid": "fetch_wdpa_global.py (WDPA attrs)", "site_pid": "fetch_wdpa_global.py (WDPA attrs)",
    "name": "fetch_wdpa_global.py (WDPA attrs)", "name_eng": "fetch_wdpa_global.py (WDPA attrs)",
    "iso3": "fetch_wdpa_global.py (WDPA attrs)", "parent_iso3": "fetch_wdpa_global.py (WDPA attrs)",
    "status": "fetch_wdpa_global.py (WDPA attrs)", "status_yr": "fetch_wdpa_global.py (WDPA attrs)",
    "iucn_cat": "fetch_wdpa_global.py (WDPA attrs)", "desig_eng": "fetch_wdpa_global.py (WDPA attrs)",
    "desig_type": "fetch_wdpa_global.py (WDPA attrs)", "gov_type": "fetch_wdpa_global.py (WDPA attrs)",
    "gis_area_km2": "fetch_wdpa_global.py (WDPA attrs)", "gis_m_area_km2": "fetch_wdpa_global.py (WDPA attrs)",
    "marine_frac": "fetch_wdpa_global.py (WDPA attrs)", "n_polygon_parts": "fetch_wdpa_global.py (dissolve)",
    "centroid_lon": "fetch_wdpa_global.py (WDPA attrs)", "centroid_lat": "fetch_wdpa_global.py (WDPA attrs)",
    "country_query": "fetch_wdpa_global.py (WDPA attrs)", "partition_method": "fetch_wdpa_global.py (ISO3/bbox check)",
    "elev_mean": "compute_wdpa_zonal_stats.py", "slope_mean": "compute_wdpa_zonal_stats.py",
    "tree_frac_mean": "compute_wdpa_zonal_stats.py", "tree_frac_min": "compute_wdpa_zonal_stats.py",
    "tree_frac_max": "compute_wdpa_zonal_stats.py", "covariate_year_proxy": "compute_wdpa_zonal_stats.py",
    "human_mod_mean": "compute_wdpa_zonal_stats.py", "species_rich_mean": "compute_wdpa_zonal_stats.py",
    "access_min": "compute_wdpa_zonal_stats.py", "pa_density_10km": "compute_wdpa_zonal_stats.py",
    "pa_density_50km": "compute_wdpa_zonal_stats.py", "perimeter_km": "compute_wdpa_zonal_stats.py",
    "compactness": "compute_wdpa_zonal_stats.py",
    "country_name": "build_wdpa_global_dataset.py (Natural Earth crosswalk)",
    "continent": "build_wdpa_global_dataset.py (Natural Earth crosswalk)",
    "libdem": "governance.py join", "cbd_pledge_elapsed": "governance.py join",
    "wgi_va": "governance.py join", "wgi_pv": "governance.py join", "wgi_ge": "governance.py join",
    "wgi_rq": "governance.py join", "wgi_rl": "governance.py join", "wgi_cc": "governance.py join",
    "dpi_exec_ideology": "governance.py join",
    "biodiv_aid_usd_million": "conservation_finance.py join", "biodiv_aid_log1p": "conservation_finance.py join",
    "biodiv_aid_n_projects": "conservation_finance.py join", "biodiv_aid_share_of_oda": "conservation_finance.py join",
    "is_conservation_finance_recipient": "build_wdpa_global_dataset.py (derived flag)",
    "split": "build_wdpa_global_dataset.py (discovery/confirmation split)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--n-examples", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} does not exist. Run fetch_wdpa_global.py -> "
              f"compute_wdpa_zonal_stats.py -> build_wdpa_global_dataset.py first.")
        sys.exit(2)

    df = pd.read_parquet(args.path)
    print(f"=== {args.path} ===")
    print(f"Rows: {len(df):,}   Columns: {len(df.columns)}\n")

    print("--- Columns grouped by build stage ---")
    by_stage = {}
    for col in df.columns:
        stage = COLUMN_SOURCE.get(col, "(unlisted — check build_wdpa_global_dataset.py)")
        by_stage.setdefault(stage, []).append(col)
    for stage, cols in by_stage.items():
        print(f"  [{stage}]")
        print(f"    {', '.join(cols)}")
    print()

    print("--- Per-column non-null % ---")
    non_null_pct = (df.notna().mean() * 100).round(1).sort_values()
    for col, pct in non_null_pct.items():
        print(f"  {col:32s} {pct:5.1f}%")
    print()

    if "iso3" in df.columns:
        print("--- Country coverage ---")
        counts = df["iso3"].value_counts()
        print(f"  Distinct countries (iso3): {df['iso3'].nunique()}")
        print(f"  Top 10 by polygon count:")
        for iso3, n in counts.head(10).items():
            print(f"    {iso3:5s} {n:6,}")
        small = counts[counts <= 2]
        print(f"  Countries with <=2 polygons: {len(small)} ({list(small.index[:15])}{'…' if len(small) > 15 else ''})")
        print()

    if "status_yr" in df.columns:
        print("--- Year coverage (by decade) ---")
        decade = (df["status_yr"] // 10 * 10).astype("Int64")
        for d, n in decade.value_counts().sort_index().items():
            print(f"  {d}s: {n:,}")
        if "covariate_year_proxy" in df.columns:
            n_proxy = int(df["covariate_year_proxy"].fillna(False).sum())
            print(f"  covariate_year_proxy=True (pre-2000, using 2000 covariate baseline): {n_proxy:,}")
        print()

    if "split" in df.columns:
        print("--- Discovery/confirmation split ---")
        print(df["split"].value_counts())
        print(df["split"].value_counts(normalize=True).round(3))
        if "iso3" in df.columns:
            mixed = df.groupby("iso3")["split"].nunique()
            n_split_countries = int((mixed > 1).sum())
            print(f"  Countries with rows on both sides of the split: {n_split_countries} "
                  f"(expected: countries with >= 10 polygons)")
        print()

    print(f"--- {args.n_examples} example rows ---")
    sample_cols = [c for c in ["wdpaid", "name", "country_name", "status_yr", "iucn_cat",
                                "gis_area_km2", "tree_frac_mean", "pa_density_10km", "libdem", "split"]
                   if c in df.columns]
    if len(df) >= args.n_examples:
        sample = df[sample_cols].sample(n=args.n_examples, random_state=42)
    else:
        sample = df[sample_cols]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
