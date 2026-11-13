"""Does an archived WDPA release reproduce the protected flags stored by Mouillot et al. (2024)?

Draws a stratified sample of the stored 10 km Mollweide land cells (background, IUCN Ia/Ib,
IUCN II-VI), rebuilds each cell as a 10 km square around its stored centre, and counts the
WDPA polygons of the chosen Earth Engine release that intersect it. A cell counts as protected
in the paper if any PA partly overlaps it.

Usage: python mouillot_wdpa_agreement.py <FRBCesab-pelagic-f4a2b42 dir> <ee project> <release, e.g. 202004> <output csv> [n_per_stratum]
"""
import sys

import ee
import numpy as np
import pandas as pd
import pyreadr
from pyproj import Transformer

CELL_M = 10_000
EDGE_POINTS = 6
BATCH = 250
SEED = 42
IUCN_ORDER = ["Ia", "Ib", "II", "III", "IV", "V", "VI"]

pkg, project, release, out_path = sys.argv[1:5]
n_per_stratum = int(sys.argv[5]) if len(sys.argv) > 5 else 1000

cells = pyreadr.read_r(f"{pkg}/data/IUCN.I.VI.ATP.RData").popitem()[1]
cells = cells.loc[:, ~cells.columns.duplicated()][["ISO3", "long", "lat", "IUCN_CAT", "Reserve"]]
cells["stratum"] = np.select([cells.IUCN_CAT == "NONE", cells.IUCN_CAT.isin(["Ia", "Ib"])], ["background", "Ia_Ib"], "II_VI")
sample = cells.groupby("stratum").sample(n=n_per_stratum, random_state=SEED).reset_index(drop=True)

to_moll = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
to_wgs = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True)
cx, cy = to_moll.transform(sample.long.values, sample.lat.values)
t = np.linspace(-0.5, 0.5, EDGE_POINTS)
ring_x = np.concatenate([t, np.full_like(t, 0.5), t[::-1], np.full_like(t, -0.5)]) * CELL_M
ring_y = np.concatenate([np.full_like(t, -0.5), t, np.full_like(t, 0.5), t[::-1]]) * CELL_M

ee.Initialize(project=project)
wdpa = ee.FeatureCollection(f"WCMC/WDPA/{release}/polygons")

def cell_feature(i):
    lon, lat = to_wgs.transform(cx[i] + ring_x, cy[i] + ring_y)
    ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
    return ee.Feature(ee.Geometry.Polygon([ring + [ring[0]]], None, False), {"i": i})

def overlaps(f):
    hits = wdpa.filterBounds(f.geometry())
    return f.set({"iucn": hits.aggregate_array("IUCN_CAT"), "status": hits.aggregate_array("STATUS"),
                  "status_yr": hits.aggregate_array("STATUS_YR")}).setGeometry(None)

rows = []
for start in range(0, len(sample), BATCH):
    fc = ee.FeatureCollection([cell_feature(i) for i in range(start, min(start + BATCH, len(sample)))])
    rows += [f["properties"] for f in fc.map(overlaps).getInfo()["features"]]
    print(f"{len(rows)}/{len(sample)}", flush=True)

hits = pd.DataFrame(rows).set_index("i").sort_index()
sample["n_pa"] = hits.iucn.str.len()
sample["n_pa_iucn"] = hits.iucn.apply(lambda v: sum(c in IUCN_ORDER for c in v))
sample["wdpa_strictest"] = hits.iucn.apply(lambda v: next((c for c in IUCN_ORDER if c in v), "NONE"))
sample["max_status_yr"] = hits.status_yr.apply(lambda v: max(v) if v else 0)
sample["statuses"] = hits.status.apply(lambda v: "|".join(sorted(set(v))))
sample.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"\nrelease {release}, {n_per_stratum} cells per stratum")
print(sample.groupby("stratum").agg(any_pa=("n_pa", lambda s: (s > 0).mean()), any_iucn_pa=("n_pa_iucn", lambda s: (s > 0).mean())).round(3))
print("\nstored category (rows) vs strictest IUCN category of overlapping release polygons (cols):")
print(pd.crosstab(sample.IUCN_CAT, sample.wdpa_strictest).reindex(columns=IUCN_ORDER + ["NONE"], fill_value=0))
