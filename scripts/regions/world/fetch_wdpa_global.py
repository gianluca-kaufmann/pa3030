"""
Fetch real WDPA polygon attributes globally, one row per designation event, for
Question 2 ("Two Logics") — see STATUS.md "Active: Question 2". This is the
foundation dataset: nothing before this script ever reads WDPA polygon attributes
beyond STATUS/STATUS_YR (build_target_wdpa.py rasterizes and discards them).

GEE asset schema note (confirmed live, session 53 — differs from the classic WDPA
CSV/shapefile export): WCMC/WDPA/current/polygons has no WDPAID/MARINE/PARENT_ISO3/
ORIG_NAME fields. The real fields are SITE_ID (int, near-unique key — ~0.7% repeat
across multi-part geometries, dissolved below), SITE_PID, NAME, NAME_ENG, ISO3,
PRNT_ISO3, GIS_AREA (km2, total), GIS_M_AREA (km2, marine component — no boolean
MARINE flag exists in this asset; marine_frac = GIS_M_AREA / GIS_AREA is used
instead), IUCN_CAT, DESIG_ENG, DESIG_TYPE, GOV_TYPE, STATUS, STATUS_YR.

Year range: 1990-2023 (not the existing pipeline's 2001-2023 convention) — a live
GEE query this session confirmed real density before 2001 (46,849 polygons in
1990-2000 alone, only 33,133 with truly unknown STATUS_YR globally), and pillar 1
(staggered REDD+/carbon-market entry) needs a real pre-entry baseline. Time-varying
covariates in compute_wdpa_zonal_stats.py can't be computed at true same-year
accuracy before 2000 (Hansen/MODIS start then) — those events get a 2000 baseline
+ covariate_year_proxy=True flag, applied in that script, not here.

ISO3 reliability (validated live this session, not assumed): for 7/8 spot-checked
countries, ISO3-filtered count vs. filterBounds-on-the-real-Natural-Earth-polygon
count agree within a normal methodological range (ratio 0.91-1.43 — filterBounds
legitimately catches border-adjacent transboundary PAs ISO3 doesn't, and vice
versa). Botswana is a real outlier (ISO3 count 4 vs bounds count 36, ratio 0.11) —
confirms session 41's finding was a genuine WDPA data-quality issue for that
country, not a bbox-vs-exact-polygon artifact. Strategy: use ISO3 as the primary
per-country partition (fast), but run a cheap bbox-based cross-check per country
and fall back to precise filterBounds on the real country polygon whenever the
two diverge by more than BBOX_FALLBACK_RATIO — catches Botswana-like outliers
without paying full filterBounds cost (slow on complex multi-island countries)
for every one of ~190 countries.

Output: regions/global/wdpa_raw/attrs_<iso3>.parquet, one checkpoint file per
country (resumable — skip a country if its file already exists). Multi-part
geometries sharing one SITE_ID are dissolved: GIS_AREA summed, centroid taken
as the area-weighted mean, n_polygon_parts recorded.

Run:
    python3 scripts/regions/world/fetch_wdpa_global.py
"""

import os, sys, zipfile, tempfile, time
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
from scripts.regions.world import paths as world_paths  # noqa: E402

RAW_DIR = str(world_paths.RAW_DIR)
os.makedirs(RAW_DIR, exist_ok=True)
CROSSWALK_PATH = str(world_paths.CROSSWALK_PATH)

NE_ZIP = str(world_paths.DOWNLOAD_DIR / "ne_countries.zip")

STATUS_YR_MIN = 1990
STATUS_YR_MAX = 2023
VALID_STATUS = ["Designated", "Established", "Inscribed", "Adopted"]
MARINE_FRAC_THRESHOLD = 0.1   # terrestrial-only: exclude GIS_M_AREA/GIS_AREA >= this
BBOX_FALLBACK_RATIO = 0.5     # if n_iso3 / n_bbox outside [ratio, 1/ratio], escalate to precise filterBounds

FIELDS = ["SITE_ID", "SITE_PID", "NAME", "NAME_ENG", "ISO3", "PRNT_ISO3",
          "STATUS", "STATUS_YR", "IUCN_CAT", "DESIG_ENG", "DESIG_TYPE", "GOV_TYPE",
          "GIS_AREA", "GIS_M_AREA"]

# Session 53's original 24-country pilot, diverse across continent, governance
# regime, income level, and includes Botswana deliberately to exercise the
# bbox-fallback path.
PILOT_COUNTRIES_S53 = [
    "Canada", "Brazil", "Indonesia", "Botswana", "Kenya", "Costa Rica", "Fiji",
    "Nepal", "Luxembourg", "France", "Germany", "Australia", "Mexico", "Peru",
    "South Africa", "India", "Vietnam", "Philippines", "Madagascar", "Tanzania",
    "Colombia", "Chile", "Norway", "Poland",
]

# Natural Earth country_name entries that are NOT a real independent conservation-
# policy jurisdiction — uninhabited islets/icefields, disputed buffer/military
# zones, or territory with no functioning unified government (Western Sahara,
# Somaliland, Northern Cyprus) — excluded from the session 57 scale-up. Judgment
# calls, documented rather than silent: a dependency with its own real WDPA
# reporting and de facto environmental governance (Greenland, Puerto Rico, Hong
# Kong, Taiwan, French Polynesia, New Caledonia, ...) is KEPT even though it isn't
# fully sovereign — it can still make real siting decisions, which is what
# pillars 1-3 actually need. Only entities with essentially no government to make
# a siting decision at all are dropped.
NON_JURISDICTION_EXCLUDE = {
    "Akrotiri", "Antarctica", "Ashmore and Cartier Is.", "Bajo Nuevo Bank",
    "Baikonur", "Bir Tawil", "Br. Indian Ocean Ter.", "Brazilian I.",
    "Coral Sea Is.", "Cyprus U.N. Buffer Zone", "Dhekelia",
    "Heard I. and McDonald Is.", "Indian Ocean Ter.", "Scarborough Reef",
    "Serranilla Bank", "Siachen Glacier", "S. Geo. and the Is.",
    "Southern Patagonian Ice Field", "Spratly Is.", "USNB Guantanamo Bay",
    "Vatican", "W. Sahara", "N. Cyprus", "Somaliland",
}

# Session 57 scale-up: every Natural Earth country/jurisdiction not in
# NON_JURISDICTION_EXCLUDE. Resumable (main() skips any iso3 whose attrs_<iso3>.
# parquet already exists), so this safely includes the 24 already-fetched pilot
# countries plus everything new — no need to diff them out by hand. The exact
# final count is printed at runtime (crosswalk has 258 raw entries; STATUS.md's
# "~190" estimate from session 56 was a rough prior, not a target to hit exactly).
def _build_scaleup_country_list():
    import pandas as pd
    if not os.path.exists(CROSSWALK_PATH):
        return PILOT_COUNTRIES_S53  # crosswalk not built yet on a first-ever run
    cw = pd.read_parquet(CROSSWALK_PATH)
    return sorted(n for n in cw["country_name"].unique() if n not in NON_JURISDICTION_EXCLUDE)


PILOT_COUNTRIES = _build_scaleup_country_list()


def load_natural_earth_countries():
    import geopandas as gpd
    with zipfile.ZipFile(NE_ZIP) as zf:
        shp_names = [n for n in zf.namelist() if n.endswith(".shp")]
        with tempfile.TemporaryDirectory() as tmp:
            members = [n for n in zf.namelist()
                       if os.path.splitext(n)[1] in (".shp", ".shx", ".dbf", ".prj", ".cpg")]
            zf.extractall(tmp, members=members)
            gdf = gpd.read_file(os.path.join(tmp, shp_names[0]))
    return gdf


def build_crosswalk(gdf):
    """One row per Natural Earth country: name, iso3, continent, bbox. Cached —
    reused by the ISO3 fallback check here and by build_wdpa_global_dataset.py's
    join later."""
    if os.path.exists(CROSSWALK_PATH):
        return pd.read_parquet(CROSSWALK_PATH)
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        w, s, e, n = geom.bounds
        rows.append({
            "country_name": row["NAME"], "sovereignt": row["SOVEREIGNT"],
            "iso3": row["ADM0_A3"], "continent": row["CONTINENT"],
            "region_un": row["REGION_UN"], "bbox_w": w, "bbox_s": s, "bbox_e": e, "bbox_n": n,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(CROSSWALK_PATH, index=False)
    return df


def base_filter(pas):
    import ee
    pas = (pas.filter(ee.Filter.inList("STATUS", VALID_STATUS))
              .filter(ee.Filter.gte("STATUS_YR", STATUS_YR_MIN))
              .filter(ee.Filter.lte("STATUS_YR", STATUS_YR_MAX))
              .filter(ee.Filter.gt("GIS_AREA", 0)))
    return pas


def add_marine_frac(pas):
    import ee
    return pas.map(lambda f: f.set(
        "marine_frac", ee.Number(f.get("GIS_M_AREA")).divide(f.get("GIS_AREA"))))


def fetch_country(iso3, bbox, name, geom=None, max_retries_depth=6, chunk_size=300):
    """Fetch all qualifying, terrestrial WDPA features for one country via ISO3
    filter, with a two-tier cross-check to catch ISO3 failures without paying
    the cost of precise filterBounds for every country:
      1. Cheap bbox-rectangle count vs. n_iso3 — a fast initial red flag.
      2. Only if that's suspicious: precise filterBounds on the REAL country
         polygon (not a rectangle) vs. n_iso3 — the rectangle over- flags
         elongated/irregular countries (France: bbox rectangle swept in 51,274
         Western-Europe candidates vs. a true ISO3 count of 5,232 — a bbox
         artifact, not an ISO3 data problem, found live session 53) so it can't
         be trusted as the fallback partition itself, only as a trigger.
    Falls back to the real-geometry filterBounds partition only if BOTH checks
    disagree with ISO3 — that's a genuine WDPA ISO3 tagging problem (Botswana:
    ISO3=4 vs real-polygon-bounds=36, confirmed session 53), not a bbox artifact.
    Then a batched getInfo() pull reusing the OOM-halving retry pattern from
    stratify_and_sample_spots.py::compute_pa_coverage."""
    import ee

    base = base_filter(ee.FeatureCollection("WCMC/WDPA/current/polygons"))
    base = add_marine_frac(base).filter(ee.Filter.lt("marine_frac", MARINE_FRAC_THRESHOLD))

    pas_iso3 = base.filter(ee.Filter.eq("ISO3", iso3))
    n_iso3 = pas_iso3.size().getInfo()

    gee_bbox = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], "EPSG:4326", False)
    n_bbox = base.filterBounds(gee_bbox).size().getInfo()

    bbox_ratio = n_iso3 / max(n_bbox, 1)
    bbox_suspicious = bbox_ratio < BBOX_FALLBACK_RATIO or bbox_ratio > (1.0 / BBOX_FALLBACK_RATIO)

    use_bounds = False
    pas, n_total, partition_method = pas_iso3, n_iso3, "iso3"

    if bbox_suspicious and geom is not None:
        gee_geom = ee.Geometry(geom.__geo_interface__, geodesic=False)
        n_exact = base.filterBounds(gee_geom).size().getInfo()
        exact_ratio = n_iso3 / max(n_exact, 1)
        exact_suspicious = exact_ratio < BBOX_FALLBACK_RATIO or exact_ratio > (1.0 / BBOX_FALLBACK_RATIO)
        print(f"    ISO3/bbox ratio {bbox_ratio:.2f} flagged {name} for a precise check: "
              f"n_iso3={n_iso3} n_bbox={n_bbox} n_exact_polygon={n_exact} exact_ratio={exact_ratio:.2f}")
        if exact_suspicious:
            print(f"    -> confirmed ISO3 problem (not a bbox-rectangle artifact) — "
                  f"falling back to precise filterBounds for {name}")
            pas, n_total, partition_method = base.filterBounds(gee_geom), n_exact, "bounds_fallback_exact"
            use_bounds = True
        else:
            print(f"    -> bbox rectangle was the artifact (elongated/irregular shape), "
                  f"ISO3 is fine — keeping iso3 partition for {name}")

    if use_bounds:
        ratio_display = exact_ratio
    else:
        ratio_display = bbox_ratio

    print(f"    {name} ({iso3}): {n_total} qualifying terrestrial polygons "
          f"[{partition_method}, n_iso3={n_iso3} n_bbox={n_bbox} ratio={ratio_display:.2f}]")

    if n_total == 0:
        return pd.DataFrame(columns=FIELDS + ["lon", "lat", "country_query", "partition_method"])

    feats_list = pas.toList(n_total)

    records = []

    def process(start, length, depth=0):
        if length == 0:
            return
        sub = feats_list.slice(start, start + length)
        sub_fc = ee.FeatureCollection(sub)
        with_centroid = sub_fc.map(
            lambda f: f.set({"lon": f.geometry().centroid(1).coordinates().get(0),
                              "lat": f.geometry().centroid(1).coordinates().get(1)}))
        try:
            info = with_centroid.getInfo()
        except Exception as e:
            if length <= 1 or depth > max_retries_depth:
                print(f"      WARNING: giving up on {name} slice [{start}:{start+length}]: {e}")
                return
            mid = length // 2
            process(start, mid, depth + 1)
            process(start + mid, length - mid, depth + 1)
            return
        for feat in info["features"]:
            props = feat["properties"]
            rec = {k: props.get(k) for k in FIELDS}
            rec["lon"] = props.get("lon")
            rec["lat"] = props.get("lat")
            records.append(rec)

    for start in range(0, n_total, chunk_size):
        process(start, min(chunk_size, n_total - start))

    df = pd.DataFrame(records)
    if len(df):
        df["country_query"] = name
        df["partition_method"] = partition_method
    return df


def dissolve_multipart(df):
    """SITE_ID can repeat across multi-part geometries. Sum GIS_AREA, take the
    area-weighted centroid, record n_polygon_parts. Non-numeric/attribute columns
    take the value from the largest-area part."""
    if df.empty:
        df["n_polygon_parts"] = pd.Series(dtype=int)
        return df

    df = df.copy()
    df["GIS_AREA"] = pd.to_numeric(df["GIS_AREA"], errors="coerce")
    n_parts = df.groupby("SITE_ID").size()
    dup_ids = n_parts[n_parts > 1].index
    if len(dup_ids) == 0:
        df["n_polygon_parts"] = 1
        return df

    print(f"    Dissolving {len(dup_ids)} SITE_ID(s) with multiple polygon parts …")
    keep_rows = []
    for site_id, grp in df.groupby("SITE_ID"):
        if len(grp) == 1:
            row = grp.iloc[0].to_dict()
            row["n_polygon_parts"] = 1
            keep_rows.append(row)
            continue
        total_area = grp["GIS_AREA"].sum()
        w = grp["GIS_AREA"].fillna(0)
        w = w if w.sum() > 0 else pd.Series(1.0, index=grp.index)
        lon = float((grp["lon"].astype(float) * w).sum() / w.sum())
        lat = float((grp["lat"].astype(float) * w).sum() / w.sum())
        biggest = grp.loc[grp["GIS_AREA"].idxmax()].to_dict()
        biggest["GIS_AREA"] = total_area
        biggest["lon"] = lon
        biggest["lat"] = lat
        biggest["n_polygon_parts"] = len(grp)
        keep_rows.append(biggest)
    return pd.DataFrame(keep_rows)


def main():
    print(f"=== Fetch WDPA global polygon attributes (STATUS_YR {STATUS_YR_MIN}-{STATUS_YR_MAX}, "
          f"terrestrial marine_frac < {MARINE_FRAC_THRESHOLD}) ===\n")

    import ee
    ee.Initialize()

    print("=== Step 0: total qualifying count (sanity print, before any per-country work) ===")
    base = base_filter(ee.FeatureCollection("WCMC/WDPA/current/polygons"))
    base = add_marine_frac(base).filter(ee.Filter.lt("marine_frac", MARINE_FRAC_THRESHOLD))
    n_global = base.size().getInfo()
    print(f"  Global terrestrial qualifying polygons: {n_global:,}\n")

    print("=== Step 1: Natural Earth country crosswalk ===")
    gdf = load_natural_earth_countries()
    crosswalk = build_crosswalk(gdf)
    print(f"  {len(crosswalk)} countries in crosswalk -> {CROSSWALK_PATH}\n")

    print(f"=== Step 2: fetch pilot countries ({len(PILOT_COUNTRIES)}) ===")
    for name in PILOT_COUNTRIES:
        row = crosswalk[crosswalk["country_name"] == name]
        if row.empty:
            row = crosswalk[crosswalk["sovereignt"] == name]
        if row.empty:
            print(f"  SKIP: '{name}' not found in Natural Earth crosswalk")
            continue
        row = row.iloc[0]
        iso3 = row["iso3"]
        out_path = os.path.join(RAW_DIR, f"attrs_{iso3}.parquet")
        if os.path.exists(out_path):
            print(f"  {name} ({iso3}): cached -> {out_path}")
            continue

        bbox = (row["bbox_w"], row["bbox_s"], row["bbox_e"], row["bbox_n"])
        geom_row = gdf[gdf["ADM0_A3"] == iso3]
        geom = geom_row.iloc[0].geometry if len(geom_row) else None
        df = fetch_country(iso3, bbox, name, geom=geom)
        df = dissolve_multipart(df)
        df.to_parquet(out_path, index=False)
        print(f"    -> {len(df)} unique polygons -> {out_path}")
        time.sleep(0.5)

    print("\nDone. Per-country checkpoints in", RAW_DIR)
    print("Next: scripts/regions/world/compute_wdpa_zonal_stats.py")


if __name__ == "__main__":
    main()
