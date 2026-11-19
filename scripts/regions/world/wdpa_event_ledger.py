"""Event ledger for the T1 test (docs/t1_protocol.md): protected-area records that are new after the
April 2020 cutoff, read from the archived Earth Engine WDPA releases with their own attributes and
geometries.

1. Site attributes of every release from FIRST_RELEASE to the observation release, polygons and
   points, checked per country against the release's own ISO3 counts.
2. Parcel lineage: a record that leaves the archive for good and a record of the same site first appearing
   in the next release, with the same kind and category and a matching area, are one parcel renamed
   (202512 renamed suffixes _A, _B to _1, _2). Successors take the earliest predecessor's WDPA_PID, and
   the ID in each release is kept for fetching geometry. Pairs go to parcel_lineage.csv.
   Site-ID screen: a parcel absent from a run of releases but present before and after is a
   transient omission of those releases. Counts per country and release go to completeness.csv.
3. Cutoff baseline: BASE_RELEASE, with each PATCH country taken from its patch release and the
   sites missing from BASE_RELEASE alone (present in the releases just before and after) taken
   from the next release. Longer gaps are left out, so no later record enters the baseline.
4. Ledger: every qualifying record present after the cutoff and absent from the baseline, plus
   baseline sites whose IUCN category or area changed, each in one of the protocol's five classes.
   Baseline sites missing from the observation release go to removals.csv.
5. Geometries (sites of 1 km2 or more simplified to SIMPLIFY_M): the baseline's qualifying polygons from their source
   release, and each ledger polygon at the end of the window, from the last release up to
   WINDOW_END_RELEASE holding it as a qualifying polygon (records first reported later: the first
   release holding them), with its category and area in that release. Each row also gets its
   expansion role (new site, dated extension, undated change) and its outcome category with how the
   category's timing is known. For each ledger polygon, the baseline polygon that covers most of it and the share of its
   area that polygon covers (site identity), and for IUCN I-VI polygons the share covered by the union
   of all baseline polygons (land already protected at the cutoff), in Mollweide.

Release attributes and geometries are cached per release. Outputs go to DATA_DIR/wdpa_event_ledger.

Usage:
    python -m scripts.regions.world.wdpa_event_ledger --project <ee project> --observation 202610
        [--countries CHE,COL]
"""
import argparse
import shutil

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

from scripts.regions.world import paths

FIRST_RELEASE = "201707"
BASE_RELEASE = "202004"
PATCH = {"RUS": "202003", "COL": "202005", "ECU": "202005", "MOZ": "202005"}
CUTOFF_YEAR = 2020
WINDOW = (2021, 2024)
WINDOW_END_RELEASE = "202412"
UNASSIGNED = ["Not Assigned", "Not Reported"]
LINEAGE_MIN_IOU = 0.5
QUALIFYING = ["Designated", "Inscribed", "Established"]
IUCN_CATEGORIES = ["Ia", "Ib", "II", "III", "IV", "V", "VI"]
AREA_TOLERANCE = 0.01
SIMPLIFY_M = 100
SIMPLIFY_MIN_M2 = 1_000_000
EQUAL_AREA_CRS = "ESRI:54009"
OVERLAP_BATCH = 20_000
REQUEST_DEADLINE_MS = 600_000

FIELDS = ["WDPAID", "WDPA_PID", "ISO3", "NAME", "STATUS", "STATUS_YR", "IUCN_CAT", "DESIG_TYPE", "PA_DEF", "MARINE", "AREA"]
NULL_FILL = {"STATUS_YR": 0, "GIS_AREA": -1, "REP_AREA": -1}
NEW_SCHEMA_FROM = "202512"
NEW_NAMES = {"WDPAID": "SITE_ID", "WDPA_PID": "SITE_PID", "PA_DEF": "SITE_TYPE", "MARINE": "REALM"}
NEW_CODES = {"PA_DEF": {"PA": "1", "OECM": "0"}, "MARINE": {"Terrestrial": "0", "Coastal": "1", "Marine": "2"}}

CACHE_DIR = paths.DATA_DIR / "wdpa_releases"
OUT_DIR = paths.DATA_DIR / "wdpa_event_ledger"


def source_fields(release, kind):
    names = [NEW_NAMES.get(f, f) if release >= NEW_SCHEMA_FROM else f for f in FIELDS]
    return [("GIS_AREA" if kind == "polygons" else "REP_AREA") if f == "AREA" else f for f in names]


def collection(release, kind, countries):
    fc = ee.FeatureCollection(f"WCMC/WDPA/{release}/{kind}")
    return fc.filter(ee.Filter.inList("ISO3", countries)) if countries else fc


def list_releases(observation):
    """Monthly releases from FIRST_RELEASE to `observation` that hold a polygon layer."""
    assets = ee.data.listAssets({"parent": "projects/earthengine-public/assets/WCMC/WDPA"})["assets"]
    releases = sorted(a["id"].split("/")[-1] for a in assets if a["id"].split("/")[-1].isdigit())
    releases = [r for r in releases if FIRST_RELEASE <= r <= observation]
    return [r for r in releases if ee.data.getInfo(f"WCMC/WDPA/{r}/polygons") is not None]


def release_attributes(release, countries):
    """Attributes of one release in the old-schema names, and per-country row counts against the
    release's ISO3 histogram. Cached."""
    cache = CACHE_DIR / ("_".join(countries) if countries else "all")
    attrs_path, counts_path = cache / f"attrs_{release}.parquet", cache / f"counts_{release}.csv"
    if attrs_path.exists():
        return pd.read_parquet(attrs_path), pd.read_csv(counts_path, keep_default_na=False, dtype={"release": str})
    names = {f for kind in ["polygons", "points"] for f in source_fields(release, kind)[2:]}
    fill = ee.Dictionary({f: NULL_FILL.get(f, "") for f in names})
    frames, counts = [], []
    for kind in ["polygons", "points"]:
        fc = collection(release, kind, countries)
        fields = source_fields(release, kind)
        # reduceColumns drops a row with any null field, so rows with nulls are read with NULL_FILL in their place.
        complete = ee.Filter.notNull(fields)
        filled = fc.filter(complete.Not()).map(lambda f: f.set(fill.combine(f.toDictionary(), True)))
        rows = [r for part in [fc.filter(complete), filled]
                for r in part.reduceColumns(ee.Reducer.toList(len(fields)), fields).get("list").getInfo()]
        df = pd.DataFrame(rows, columns=FIELDS).assign(kind=kind)
        if release >= NEW_SCHEMA_FROM:
            df = df.replace(NEW_CODES)
        frames.append(df)
        expected = pd.Series(fc.aggregate_histogram("ISO3").getInfo(), name="n_expected")
        counts.append(pd.concat([expected, df.ISO3.value_counts().rename("n_pulled")], axis=1)
                      .fillna(0).astype(int).rename_axis("ISO3").reset_index().assign(kind=kind))
    attrs = pd.concat(frames, ignore_index=True).astype({"WDPAID": "int64", "WDPA_PID": str, "STATUS_YR": "int64", "AREA": float})
    attrs[["PA_DEF", "MARINE"]] = attrs[["PA_DEF", "MARINE"]].astype(str)
    attrs["AREA"] = attrs.AREA.mask(attrs.AREA < 0)
    counts = pd.concat(counts, ignore_index=True).assign(release=release)
    cache.mkdir(parents=True, exist_ok=True)
    attrs.to_parquet(attrs_path, index=False)
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
    return attrs, counts


def transient_gaps(presence):
    """True where a site is absent but present in an earlier and a later release (columns ordered by release)."""
    before = np.maximum.accumulate(presence, axis=1)
    after = np.maximum.accumulate(presence[:, ::-1], axis=1)[:, ::-1]
    return ~presence & before & after


def classify(ledger):
    """Protocol class and reason for each ledger row (see docs/t1_protocol.md, Event ledger)."""
    yr = ledger.STATUS_YR
    conditions = [ledger.change != "new", ledger.prior_record, ledger.baseline_site, ledger.kind == "points",
                  yr.between(*WINDOW), yr == CUTOFF_YEAR, (yr > 0) & (yr < CUTOFF_YEAR), yr > WINDOW[1]]
    classes = [4, 4, 4, 5, 1, 3, 2, 5]
    reasons = [ledger.change, "prior_record", "new_parcel_of_baseline_site", "point_without_boundary",
               "designated_in_window", "status_year_2020", "designated_before_cutoff", "designated_after_window"]
    return ledger.assign(event_class=np.select(conditions, classes, default=5),
                         reason=np.select(conditions, reasons, default="no_status_year"))


def record_history(new, qualifying_before, baseline_wdpaids):
    """prior_record: the parcel ID was qualifying before the cutoff, or the site ID was but the site is not
    in the baseline (a re-entered record). baseline_site: the site ID is in the baseline. A new parcel ID of
    a baseline site is therefore not a prior record, and classify gives it new_parcel_of_baseline_site."""
    baseline_site = new.WDPAID.isin(baseline_wdpaids)
    prior = new.WDPA_PID.isin(qualifying_before["WDPA_PID"]) | (new.WDPAID.isin(qualifying_before["WDPAID"]) & ~baseline_site)
    return pd.DataFrame({"prior_record": prior, "baseline_site": baseline_site}, index=new.index)


def expansion_role(ledger):
    """Role of each classified ledger row in the expansion outcome (docs/t1_protocol.md, Cell labels). All
    roles are archive-derived. None is a legal verification.
    new_site: class 1.
    candidate_extension: a new parcel of a site protected before the cutoff, with its own STATUS_YR in the
    window. Its land outside the baseline is expansion on the same evidence as a new site.
    uncertain_extension: a re-entered record of a site protected before the cutoff whose STATUS_YR now falls
    in the window. The year may be a correction or a re-registration, so its new land has uncertain timing.
    undated_change: a category change or area increase of a baseline record. The archive shows when the
    change was reported, not when it took legal effect.
    none: every other row."""
    in_window = ledger.STATUS_YR.between(*WINDOW)
    conditions = [ledger.event_class == 1,
                  (ledger.event_class == 4) & (ledger.reason == "new_parcel_of_baseline_site") & in_window,
                  (ledger.event_class == 4) & (ledger.reason == "prior_record") & in_window,
                  ledger.event_class == 4]
    return pd.Series(np.select(conditions, ["new_site", "candidate_extension", "uncertain_extension", "undated_change"], default="none"),
                     index=ledger.index)


def category_timing(ledger):
    """Outcome category and what the archive shows about when the record held it.
    recorded_in_window: the category is recorded in the geometry release, which is no later than
    WINDOW_END_RELEASE. recorded_after_window: the record was first reported after the window, so its
    category is known only retrospectively, even with STATUS_YR in the window. assigned_later: the record was
    Not Assigned or Not Reported in its geometry release and received a category only later.
    never_assigned: no category in any release."""
    assigned_at_geometry = ~ledger.IUCN_CAT_geometry.isin(UNASSIGNED) & ledger.IUCN_CAT_geometry.notna()
    later = ~assigned_at_geometry & ledger.IUCN_CAT_assigned.notna()
    outcome = ledger.IUCN_CAT_geometry.where(assigned_at_geometry, ledger.IUCN_CAT_assigned.where(later, ledger.IUCN_CAT_geometry))
    in_window = ledger.geometry_release.fillna("999999") <= WINDOW_END_RELEASE
    timing = np.select([assigned_at_geometry & in_window, assigned_at_geometry, later],
                       ["recorded_in_window", "recorded_after_window", "assigned_later"], default="never_assigned")
    return pd.DataFrame({"IUCN_CAT_outcome": outcome, "category_timing": timing}, index=ledger.index)


def footprint_iou(a, b):
    """Intersection over union of paired lon/lat geometries, in EQUAL_AREA_CRS. NaN where either is missing."""
    a = gpd.GeoSeries(a, crs="EPSG:4326").reset_index(drop=True)
    b = gpd.GeoSeries(b, crs="EPSG:4326").reset_index(drop=True)
    ok = (a.notna() & b.notna()).to_numpy()
    out = np.full(len(a), np.nan)
    if ok.any():
        x = shapely.buffer(equal_area(gpd.GeoDataFrame(geometry=a[ok])), 0)
        y = shapely.buffer(equal_area(gpd.GeoDataFrame(geometry=b[ok])), 0)
        union = shapely.area(shapely.union(x, y))
        out[ok] = np.divide(shapely.area(shapely.intersection(x, y)), union, out=np.full(len(x), np.nan), where=union > 0)
    return out


def suffix_rank(pid):
    """Position of a parcel suffix: _A, _B ... and _1, _2 ... both give 1, 2 ... (None without a suffix)."""
    _, _, suffix = pid.rpartition("_")
    if not _ or not suffix:
        return None
    if suffix.isdigit():
        return int(suffix)
    if suffix.isalpha():
        return sum((ord(c) - 64) * 26 ** i for i, c in enumerate(reversed(suffix.upper())))
    return None


def parcel_lineage(vanished, appearing):
    """Match parcel records that leave the archive for good to records of the same site (WDPAID) that first
    appear in the same release, as when 202512 renamed parcel suffixes _A, _B to _1, _2. A pair needs the
    same kind and IUCN category and areas within AREA_TOLERANCE. Where a record has several candidates, the
    one with the same suffix position is kept, and records still ambiguous stay unmatched. Returns
    successor and predecessor parcel IDs, one to one."""
    pairs = vanished.merge(appearing, on=["WDPAID", "kind"], suffixes=("_old", "_new"))
    area_close = (pairs.AREA_old - pairs.AREA_new).abs() <= AREA_TOLERANCE * pairs.AREA_old.abs()
    pairs = pairs[(pairs.IUCN_CAT_old == pairs.IUCN_CAT_new) & (area_close | (pairs.AREA_old.isna() & pairs.AREA_new.isna()))]
    for side, other in [("old", "new"), ("new", "old")]:
        n = pairs.groupby(f"WDPA_PID_{side}")[f"WDPA_PID_{other}"].transform("size")
        same_rank = pairs[f"WDPA_PID_{side}"].map(suffix_rank) == pairs[f"WDPA_PID_{other}"].map(suffix_rank)
        pairs = pairs[(n == 1) | same_rank]
    unique = (pairs.groupby("WDPA_PID_old").WDPA_PID_new.transform("size") == 1) & \
             (pairs.groupby("WDPA_PID_new").WDPA_PID_old.transform("size") == 1)
    return pairs[unique][["WDPA_PID_new", "WDPA_PID_old"]].rename(columns={"WDPA_PID_new": "successor", "WDPA_PID_old": "predecessor"})


def read_geometries(release, pids, chunk, part_dir):
    """Geometries of the listed sites in one release. Sites of SIMPLIFY_MIN_M2 or more are simplified to
    SIMPLIFY_M, smaller ones are kept as they are, since simplifying would collapse them. Each request is
    saved to part_dir, so an interrupted release resumes from the last saved request."""
    pid_field = source_fields(release, "polygons")[1]
    fc = ee.FeatureCollection(f"WCMC/WDPA/{release}/polygons")

    def simplified(f):
        geom = ee.Geometry(ee.Algorithms.If(f.geometry().area(1).lt(SIMPLIFY_MIN_M2), f.geometry(), f.geometry().simplify(SIMPLIFY_M)))
        return ee.Feature(geom, {"WDPA_PID": f.get(pid_field)})
    part_dir.mkdir(exist_ok=True)
    for lo in range(0, len(pids), chunk):
        part = part_dir / f"{lo:07d}_{chunk}.parquet"
        if not part.exists():
            features = fc.filter(ee.Filter.inList(pid_field, pids[lo:lo + chunk])).map(simplified).getInfo()["features"]
            gpd.GeoDataFrame({"WDPA_PID": [f["properties"]["WDPA_PID"] for f in features], "geometry_release": release},
                             geometry=[shape(f["geometry"]) for f in features], crs="EPSG:4326").to_parquet(part)
    return pd.concat([gpd.read_parquet(part) for part in sorted(part_dir.glob(f"*_{chunk}.parquet"))], ignore_index=True)


def split_antimeridian(geoms):
    """Lon/lat geometries that cross 180 degrees, cut into pieces within -180 to 180. A crossing ring either
    jumps from about 180 to about -180 (the geometry is then wider than 180 degrees) or runs past 180
    (Papahanaumokuakea reaches 199). Projected as they are, both span the globe and cover land on other
    continents. Other geometries are left unchanged."""
    geoms = np.asarray(geoms, dtype=object).copy()
    bounds = shapely.bounds(geoms)
    wide = bounds[:, 2] - bounds[:, 0] > 180
    for k in np.flatnonzero(wide | (bounds[:, 0] < -180) | (bounds[:, 2] > 180)):
        g = geoms[k]
        if wide[k]:
            g = shapely.make_valid(shapely.transform(g, lambda c: np.where(c[:, :1] < 0, c + [360, 0], c)))
        geoms[k] = shapely.union_all([shapely.transform(shapely.intersection(g, shapely.box(-180 - o, -90, 180 - o, 90)),
                                                        lambda c, o=o: c + [o, 0]) for o in (-360, 0, 360)])
    return geoms


def polygon_parts(geoms):
    """Single polygons of each geometry, with the position of the geometry each came from."""
    parts, owner = shapely.get_parts(geoms, return_index=True)
    parts, inner = shapely.get_parts(parts, return_index=True)
    owner = owner[inner]
    keep = shapely.get_type_id(parts) == 3
    return parts[keep], owner[keep]


def equal_area(frame):
    """Geometries of a lon/lat GeoDataFrame in EQUAL_AREA_CRS, split at the antimeridian first."""
    split = gpd.GeoSeries(split_antimeridian(frame.to_crs("EPSG:4326").geometry.values), crs="EPSG:4326")
    return split.to_crs(EQUAL_AREA_CRS).values


def largest_baseline_overlap(sites, baseline):
    """For each site, the WDPA_PID of the baseline polygon covering most of it and the share of the site's
    area it covers, in EQUAL_AREA_CRS. Both sides are split into single polygons, since intersecting whole
    collections makes GEOS union all their parts, and a baseline part lying inside a site part is counted
    by its own area. Site parts are processed OVERLAP_BATCH at a time."""
    site_parts, site_of = polygon_parts(shapely.make_valid(equal_area(sites)))
    base_parts, base_of = polygon_parts(shapely.make_valid(equal_area(baseline)))
    base_pid = baseline.WDPA_PID.to_numpy()[base_of]
    tree = shapely.STRtree(base_parts)
    shapely.prepare(site_parts)
    frames = []
    for lo in range(0, len(site_parts), OVERLAP_BATCH):
        batch = site_parts[lo:lo + OVERLAP_BATCH]
        i, j = tree.query(batch, predicate="intersects")
        area = shapely.area(base_parts[j])
        crossing = ~shapely.contains_properly(batch[i], base_parts[j])
        area[crossing] = shapely.area(shapely.intersection(batch[i][crossing], base_parts[j][crossing]))
        frames.append(pd.DataFrame({"site": site_of[lo + i], "pid": base_pid[j], "area": area}))
        print(f"overlap {lo + len(batch)} of {len(site_parts)} site parts", flush=True)
    covered = pd.concat(frames).groupby(["site", "pid"]).area.sum().sort_values().groupby(level="site").tail(1).reset_index(level="pid")
    site_area = pd.Series(np.bincount(site_of, weights=shapely.area(site_parts), minlength=len(sites)))
    return pd.DataFrame({"largest_baseline_pid": covered.pid.reindex(site_area.index),
                         "largest_baseline_share": covered.area.reindex(site_area.index, fill_value=0.0) / site_area})


def window_footprints(first, end, differs):
    """Footprint established during the window and the part of it missing at the window end. Where a record's
    first and window-end geometries differ (differs), the established footprint is their union and the
    removed part is the first footprint minus the window-end one. Elsewhere the window-end geometry stands
    and nothing is removed."""
    union = np.array(end, dtype=object).copy()
    removed = np.full(len(union), None, dtype=object)
    if differs.any():
        f, e = shapely.make_valid(np.asarray(first, dtype=object)[differs]), shapely.make_valid(union[differs])
        union[differs] = shapely.union(f, e)
        removed[differs] = shapely.difference(f, e)
    return union, removed


def cached_geometries(out_dir, release, pids, chunk):
    """Geometries of the listed sites in one release, from out_dir/geometries_{release}.parquet. The cache
    holds the sites earlier runs needed from this release, so missing sites are fetched and added to it."""
    cache = out_dir / f"geometries_{release}.parquet"
    cached = gpd.read_parquet(cache) if cache.exists() else None
    missing = sorted(set(pids) - set(cached.WDPA_PID if cached is not None else []))
    if missing:
        part_dir = out_dir / f"geometries_{release}_parts"
        fetched = read_geometries(release, missing, chunk, part_dir)
        cached = gpd.GeoDataFrame(pd.concat([cached, fetched], ignore_index=True), crs="EPSG:4326")
        cached.to_parquet(cache)
        shutil.rmtree(part_dir)
    return cached[cached.WDPA_PID.isin(set(pids))]


def union_baseline_share(sites, baseline):
    """Share of each site's area covered by the union of all baseline polygons, in EQUAL_AREA_CRS. Unlike
    largest_baseline_overlap, overlapping baseline polygons count once and several polygons add up, so
    1 minus the share is the part of the site that was unprotected at the cutoff."""
    site_parts, site_of = polygon_parts(shapely.make_valid(equal_area(sites)))
    base_parts, _ = polygon_parts(shapely.make_valid(equal_area(baseline)))
    i, j = shapely.STRtree(base_parts).query(site_parts, predicate="intersects")
    covered = np.zeros(len(sites))
    for part, group in pd.DataFrame({"part": i, "base": j}).groupby("part"):
        pieces = shapely.intersection(site_parts[part], base_parts[group.base.to_numpy()])
        covered[site_of[part]] += shapely.area(shapely.union_all(pieces))
    site_area = np.bincount(site_of, weights=shapely.area(site_parts), minlength=len(sites))
    return pd.Series(np.divide(covered, site_area, out=np.full(len(sites), np.nan), where=site_area > 0), index=sites.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Earth Engine Cloud project id")
    parser.add_argument("--observation", required=True, help="observation release, YYYYMM")
    parser.add_argument("--countries", default=None, help="comma-separated ISO3 codes, for a test run")
    parser.add_argument("--chunk", type=int, default=1000, help="sites per geometry request")
    args = parser.parse_args()
    ee.Initialize(project=args.project)
    ee.data.setDeadline(REQUEST_DEADLINE_MS)
    countries = args.countries.split(",") if args.countries else None
    out_dir = OUT_DIR / ("_".join(countries) if countries else "all")
    out_dir.mkdir(parents=True, exist_ok=True)

    releases = list_releases(args.observation)
    print(f"{len(releases)} releases with polygons, {releases[0]} to {releases[-1]}", flush=True)
    pid_codes, pid_iso3, present, counts = {}, {}, [], []
    qualifying_before = {"WDPA_PID": set(), "WDPAID": set()}
    for release in releases:
        attrs, release_counts = release_attributes(release, countries)
        counts.append(release_counts)
        present.append(np.array([pid_codes.setdefault(p, len(pid_codes)) for p in attrs.WDPA_PID.unique()]))
        pid_iso3.update(zip(attrs.WDPA_PID, attrs.ISO3))
        if release <= BASE_RELEASE:
            q = attrs[attrs.STATUS.isin(QUALIFYING)]
            qualifying_before["WDPA_PID"].update(q.WDPA_PID)
            qualifying_before["WDPAID"].update(q.WDPAID)
        print(release, len(attrs), flush=True)

    raw_pids = np.array(list(pid_codes))
    raw = np.zeros((len(raw_pids), len(releases)), bool)
    for j, codes in enumerate(present):
        raw[codes, j] = True

    # Parcel lineage: a record leaving for good and a record of the same site first appearing in the next
    # release are one parcel under a new ID. Successors take their earliest predecessor's ID (lineage_pid).
    # Only renames after the baseline's source releases (202003 to 202005) are linked, so baseline records
    # keep their own IDs. Earlier renames are caught by WDPAID in prior_record.
    first_j, last_j = raw.argmax(axis=1), len(releases) - 1 - raw[:, ::-1].argmax(axis=1)
    lineage = []
    for j in range(releases.index(BASE_RELEASE) + 2, len(releases)):
        vanished, appearing = raw_pids[last_j == j - 1], raw_pids[first_j == j]
        if len(vanished) and len(appearing):
            prev = release_attributes(releases[j - 1], countries)[0].drop_duplicates("WDPA_PID")
            cur = release_attributes(releases[j], countries)[0].drop_duplicates("WDPA_PID")
            prev, cur = prev[prev.WDPA_PID.isin(vanished)], cur[cur.WDPA_PID.isin(appearing)]
            shared = list(set(prev.WDPAID) & set(cur.WDPAID))
            if shared:
                lineage.append(parcel_lineage(prev[prev.WDPAID.isin(shared)], cur[cur.WDPAID.isin(shared)]).assign(release=releases[j]))
    lineage = pd.concat(lineage, ignore_index=True) if lineage else pd.DataFrame(columns=["successor", "predecessor", "release"])
    # Spatial identity: a pair whose polygons share less than LINEAGE_MIN_IOU of their union is not linked.
    # Point records have no polygon, keep their attribute match and get no IoU.
    lineage["iou"] = np.nan
    for release, group in lineage.groupby("release"):
        previous = releases[releases.index(release) - 1]
        old = cached_geometries(out_dir, previous, group.predecessor, args.chunk).drop_duplicates("WDPA_PID").set_index("WDPA_PID").geometry
        new = cached_geometries(out_dir, release, group.successor, args.chunk).drop_duplicates("WDPA_PID").set_index("WDPA_PID").geometry
        lineage.loc[group.index, "iou"] = footprint_iou(old.reindex(group.predecessor).to_numpy(), new.reindex(group.successor).to_numpy())
    lineage["linked"] = lineage.iou.isna() | (lineage.iou >= LINEAGE_MIN_IOU)
    step = dict(zip(lineage.successor[lineage.linked], lineage.predecessor[lineage.linked]))
    root = {}
    for p in step:
        q = p
        while q in step:
            q = step[q]
        root[p] = q
    lineage["lineage_pid"] = lineage.successor.map(root)  # NaN for pairs not linked
    lineage.to_csv(out_dir / "parcel_lineage.csv", index=False, encoding="utf-8-sig")
    print(f"parcel lineage: {lineage.linked.sum()} renamed parcel records linked, {(~lineage.linked).sum()} rejected by IoU", flush=True)

    keep = ~np.isin(raw_pids, list(root))
    pids, presence = raw_pids[keep], raw[keep].copy()
    row_of = {p: k for k, p in enumerate(pids)}
    for successor, predecessor in root.items():
        presence[row_of[predecessor]] |= raw[pid_codes[successor]]
    pid_codes = row_of
    gaps = transient_gaps(presence)
    site_iso3 = np.array([pid_iso3[p] for p in pids])
    dropped = np.hstack([np.zeros((len(pids), 1), bool), ~presence[:, 1:] & presence[:, :-1]])
    screen = pd.concat([pd.DataFrame({"n_transient": pd.Series(site_iso3[gaps[:, j]]).value_counts(),
                                      "n_dropped": pd.Series(site_iso3[dropped[:, j]]).value_counts()}).assign(release=r)
                        for j, r in enumerate(releases)]).fillna(0).astype({"n_transient": int, "n_dropped": int})
    screen = screen.rename_axis("ISO3").reset_index()
    completeness = pd.concat(counts).pivot_table(index=["release", "ISO3"], columns="kind", values=["n_expected", "n_pulled"], fill_value=0)
    completeness.columns = [f"{a}_{b}" for a, b in completeness.columns]
    completeness = completeness.reset_index().merge(screen, on=["release", "ISO3"], how="outer").fillna(0)
    completeness.to_csv(out_dir / "completeness.csv", index=False, encoding="utf-8-sig")

    base_j = releases.index(BASE_RELEASE)
    base_attrs = release_attributes(BASE_RELEASE, countries)[0]
    only_base_missing = ~presence[:, base_j] & presence[:, base_j - 1] & presence[:, base_j + 1]
    fill_pids = {releases[base_j + 1]: list(pids[only_base_missing & ~np.isin(site_iso3, list(PATCH))])}
    parts = [base_attrs[~base_attrs.ISO3.isin(list(PATCH))].assign(source_release=BASE_RELEASE)]
    for iso3, release in PATCH.items():
        attrs = release_attributes(release, countries)[0]
        parts.append(attrs[attrs.ISO3 == iso3].assign(source_release=release))
    for release, fill in fill_pids.items():
        attrs = release_attributes(release, countries)[0]
        parts.append(attrs[attrs.WDPA_PID.isin(fill)].assign(source_release=release))
    baseline = pd.concat(parts, ignore_index=True)
    baseline.to_parquet(out_dir / "cutoff_baseline.parquet", index=False)
    print(f"baseline: {len(baseline)} records, {(baseline.kind == 'polygons').sum()} polygons", flush=True)

    base_qualifying = baseline[baseline.STATUS.isin(QUALIFYING)].drop_duplicates("WDPA_PID")
    after = list(range(base_j + 1, len(releases)))
    latest, first_seen, first_polygon, window_polygon, first_assigned, change_first = {}, {}, {}, {}, {}, {}
    base_cat = dict(zip(base_qualifying.WDPA_PID, base_qualifying.IUCN_CAT))
    base_area = dict(zip(base_qualifying.WDPA_PID, base_qualifying.AREA))
    for j in after[::-1]:
        attrs = release_attributes(releases[j], countries)[0]
        attrs = attrs.assign(release_pid=attrs.WDPA_PID, WDPA_PID=attrs.WDPA_PID.map(root).fillna(attrs.WDPA_PID))
        q = attrs[attrs.STATUS.isin(QUALIFYING)]
        latest.update({p: (releases[j], r) for p, r in zip(q.WDPA_PID, q.itertuples(index=False)) if p not in latest})
        first_seen.update(dict.fromkeys(attrs.WDPA_PID, releases[j]))
        qp = q[q.kind == "polygons"]
        first_polygon.update({p: (releases[j], r) for p, r in zip(qp.WDPA_PID, qp.itertuples(index=False))})
        if releases[j] <= WINDOW_END_RELEASE:
            window_polygon.update({p: (releases[j], r) for p, r in zip(qp.WDPA_PID, qp.itertuples(index=False)) if p not in window_polygon})
        qa = q[~q.IUCN_CAT.isin(UNASSIGNED)]
        first_assigned.update({p: (releases[j], c) for p, c in zip(qa.WDPA_PID, qa.IUCN_CAT)})
        qb = q[q.WDPA_PID.isin(base_cat.keys())]
        changed = (qb.IUCN_CAT.to_numpy() != qb.WDPA_PID.map(base_cat).to_numpy()) | \
                  (qb.AREA > qb.WDPA_PID.map(base_area) * (1 + AREA_TOLERANCE)).to_numpy()
        change_first.update(dict.fromkeys(qb.WDPA_PID[changed], releases[j]))
    last = pd.DataFrame([r._asdict() | {"last_release": rel} for rel, r in latest.values()])
    last["first_release"] = last.WDPA_PID.map(first_seen)
    last["present_at_observation"] = last.last_release == releases[-1]
    # Footprint at the end of the window: a record's geometry comes from the last release up to
    # WINDOW_END_RELEASE holding it as a qualifying polygon, or, for records first reported later, the first
    # release holding it. Its category and area in that same release are kept beside the latest ones.
    geometry_source = {**first_polygon, **window_polygon}
    at_geometry = pd.DataFrame([{"WDPA_PID": p, "geometry_release": rel, "geometry_pid": r.release_pid,
                                 "IUCN_CAT_geometry": r.IUCN_CAT, "AREA_geometry": r.AREA,
                                 "first_polygon_release": first_polygon[p][0], "first_polygon_pid": first_polygon[p][1].release_pid}
                                for p, (rel, r) in geometry_source.items()],
                               columns=["WDPA_PID", "geometry_release", "geometry_pid", "IUCN_CAT_geometry", "AREA_geometry",
                                        "first_polygon_release", "first_polygon_pid"])
    assigned = pd.DataFrame([{"WDPA_PID": p, "category_assigned_release": rel, "IUCN_CAT_assigned": c}
                             for p, (rel, c) in first_assigned.items()], columns=["WDPA_PID", "category_assigned_release", "IUCN_CAT_assigned"])

    new = last[~last.WDPA_PID.isin(base_qualifying.WDPA_PID)].assign(change="new")
    new[["prior_record", "baseline_site"]] = record_history(new, qualifying_before, base_qualifying.WDPAID)
    old = last.merge(base_qualifying[["WDPA_PID", "IUCN_CAT", "AREA"]], on="WDPA_PID", suffixes=("", "_baseline"))
    old["change"] = np.select([old.IUCN_CAT != old.IUCN_CAT_baseline, old.AREA > old.AREA_baseline * (1 + AREA_TOLERANCE)],
                              ["category_change", "area_increase"], default="")
    old = old[old.change != ""].assign(prior_record=True, baseline_site=True, change_first_release=lambda d: d.WDPA_PID.map(change_first))
    ledger = pd.concat([new, old], ignore_index=True).merge(at_geometry, on="WDPA_PID", how="left").merge(assigned, on="WDPA_PID", how="left")
    ledger = classify(ledger)
    ledger["expansion_role"] = expansion_role(ledger)
    ledger[["IUCN_CAT_outcome", "category_timing"]] = category_timing(ledger)
    ledger.to_parquet(out_dir / "ledger.parquet", index=False)
    print(ledger.groupby(["event_class", "reason"]).size().to_string(), flush=True)

    obs = release_attributes(releases[-1], countries)[0]
    removals = base_qualifying[~base_qualifying.WDPA_PID.isin(obs.WDPA_PID.map(root).fillna(obs.WDPA_PID))].assign(
        last_release=lambda d: d.WDPA_PID.map(lambda p: releases[int(np.flatnonzero(presence[pid_codes[p]])[-1])]))
    removals.to_csv(out_dir / "removals.csv", index=False, encoding="utf-8-sig")

    base_polygons = base_qualifying[base_qualifying.kind == "polygons"].assign(geometry_release=lambda d: d.source_release,
                                                                               geometry_pid=lambda d: d.WDPA_PID)
    ledger_polygons = ledger[ledger.kind == "polygons"]
    needed = pd.concat([base_polygons[["geometry_pid", "geometry_release"]], ledger_polygons[["geometry_pid", "geometry_release"]],
                        ledger_polygons[["first_polygon_pid", "first_polygon_release"]].set_axis(["geometry_pid", "geometry_release"], axis=1)]
                       ).dropna().drop_duplicates()
    frames = []
    for release, group in needed.groupby("geometry_release"):
        frames.append(cached_geometries(out_dir, release, group.geometry_pid, args.chunk).assign(geometry_release=release))
        print(release, len(frames[-1]), "of", len(group), "geometries", flush=True)
    geometries = pd.concat(frames, ignore_index=True)
    geometries = geometries.drop_duplicates(["WDPA_PID", "geometry_release"]).rename(columns={"WDPA_PID": "geometry_pid"})
    geometries = geometries.drop(columns=[c for c in geometries.columns if c not in ["geometry_pid", "geometry_release", "geometry"]])

    baseline_geometries = gpd.GeoDataFrame(base_polygons.merge(geometries, on=["geometry_pid", "geometry_release"]), crs="EPSG:4326")
    events = gpd.GeoDataFrame(ledger_polygons.merge(geometries, on=["geometry_pid", "geometry_release"]), crs="EPSG:4326")
    # Land protected during the window and removed or cut before its end still counts as established: the
    # event geometry is the union of the first footprint and the window-end footprint, and the part missing
    # at the window end is kept separately (removed_in_window).
    first = geometries.rename(columns={"geometry_pid": "first_polygon_pid", "geometry_release": "first_polygon_release",
                                       "geometry": "geometry_first"})
    events = events.merge(first, on=["first_polygon_pid", "first_polygon_release"], how="left")
    differs = (events.first_polygon_release != events.geometry_release).to_numpy() & events.geometry_first.notna().to_numpy()
    end = events.geometry.to_numpy().copy()
    union, removed = window_footprints(events.geometry_first.to_numpy(), end, differs)
    events = gpd.GeoDataFrame(pd.DataFrame(events.drop(columns=["geometry_first", "geometry"])), geometry=union, crs="EPSG:4326")
    events["geometry_window_end"] = gpd.GeoSeries(end, crs="EPSG:4326")
    events["removed_in_window"] = gpd.GeoSeries(removed, crs="EPSG:4326")
    print(f"{len(baseline_geometries)} of {len(base_polygons)} baseline polygons and {len(events)} of {len(ledger_polygons)} "
          "ledger polygons with geometry", flush=True)
    baseline_geometries.to_parquet(out_dir / "cutoff_baseline_polygons.parquet")
    overlap = largest_baseline_overlap(events, baseline_geometries)
    events[overlap.columns] = overlap.to_numpy()
    iucn = events.IUCN_CAT.isin(IUCN_CATEGORIES) | events.IUCN_CAT_outcome.isin(IUCN_CATEGORIES)
    events["union_baseline_share"] = np.nan
    events.loc[iucn, "union_baseline_share"] = union_baseline_share(events[iucn], baseline_geometries)
    events.to_parquet(out_dir / "events.parquet")

if __name__ == "__main__":
    main()
