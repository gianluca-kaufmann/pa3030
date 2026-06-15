"""PA Connectivity Gap Rasterisation — South America.

Creates pa_connectivity_gap (0/1) — pixels that bridge two or more separate PA
clusters within a 5 km radius.  These "stepping stone" pixels face elevated
designation pressure because connecting fragmented PAs is a key conservation
goal (corridor policies, IUCN guidelines).

Note: GSN_b4 (potential wildlife corridors from the Global Safety Net) already
encodes a coarser version of this signal.  Run the full experiment with --force
to determine if this finer-grained WDPA-derived version adds signal beyond GSN_b4.

This feature is STATIC (computed from the 2000 WDPA baseline) because:
  - PA clusters change slowly; 2000 reflects the structural connectivity pattern
  - Time-varying connectivity gap would require labelling per year (expensive)
  - The static 2000 baseline captures which areas were structural gaps at study start

Input:
  data/south_america/ready/WDPA/WDPA_SA_1km_2000.tif  (already present locally)
  data/south_america/ready/backbone/backbone.tif

Output:
  data/south_america/ready/pa_connectivity_gap/pa_connectivity_gap_sa.tif (1 band, float32)
    Band 1: pa_connectivity_gap — 1.0 if pixel bridges ≥2 PA clusters within 5 km

Usage:
  python scripts/regions/south_america/2_preprocessing/pa_connectivity_gap_rasterise.py

Then inject (standard TIF workflow):
  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\
      --tif data/south_america/ready/pa_connectivity_gap/pa_connectivity_gap_sa.tif \\
      --cols pa_connectivity_gap

Experiment decision: Only run if eco_protection_gap + KBA + REDD+ are insufficient
to close the Recall@5% gap.  Likely redundant with GSN_b4 — confirm with SHAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label, maximum_filter, minimum_filter

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKBONE_PATH = PROJECT_ROOT / "data/south_america/ready/backbone/backbone.tif"
WDPA_2000_PATH = PROJECT_ROOT / "data/south_america/ready/WDPA/WDPA_SA_1km_2000.tif"
OUTPUT_PATH = PROJECT_ROOT / "data/south_america/ready/pa_connectivity_gap/pa_connectivity_gap_sa.tif"

CONNECTIVITY_RADIUS_KM = 5   # pixels (1 km resolution)


def main() -> None:
    if not BACKBONE_PATH.exists():
        sys.exit(f"Backbone not found: {BACKBONE_PATH}")
    if not WDPA_2000_PATH.exists():
        sys.exit(f"WDPA 2000 TIF not found: {WDPA_2000_PATH}")

    print("Loading backbone and WDPA 2000...")
    with rasterio.open(BACKBONE_PATH) as src:
        profile = src.profile.copy()
        backbone = src.read(1)
        shape = (src.height, src.width)

    with rasterio.open(WDPA_2000_PATH) as src:
        wdpa = src.read(1).astype(np.float32)

    # Align WDPA to backbone grid (WDPA is 9047×8975 vs backbone 9172×8974)
    h_int = min(shape[0], wdpa.shape[0])
    w_int = min(shape[1], wdpa.shape[1])

    pa_baseline = np.zeros(shape, dtype=np.uint8)
    wdpa_crop = wdpa[:h_int, :w_int]
    pa_baseline[:h_int, :w_int] = np.where(np.isfinite(wdpa_crop), wdpa_crop > 0, False)
    pa_baseline[backbone == 0] = 0

    n_pa = int(pa_baseline.sum())
    n_land = int((backbone == 1).sum())
    print(f"Protected pixels (2000): {n_pa:,} / {n_land:,} ({100*n_pa/n_land:.1f}%)")

    # Label connected PA clusters (4-connectivity)
    print("Labelling PA clusters (this may take ~30 s)...")
    labeled, n_clusters = label(pa_baseline, structure=np.ones((3, 3), dtype=np.uint8))
    print(f"  Found {n_clusters:,} distinct PA clusters")

    # Algorithm: O(n_pixels × kernel) using max/min filters — much faster than
    # per-cluster dilation (which is O(n_clusters × n_pixels)).
    #
    # A non-PA pixel is a connectivity gap iff it has ≥2 distinct PA cluster IDs
    # within a CONNECTIVITY_RADIUS_KM neighbourhood.
    #
    # min_cluster < max_cluster (both > 0) → at least 2 distinct IDs are nearby.

    print(f"Computing connectivity gap (radius={CONNECTIVITY_RADIUS_KM} km, "
          f"kernel={2*CONNECTIVITY_RADIUS_KM+1}×{2*CONNECTIVITY_RADIUS_KM+1})...")
    kernel_size = 2 * CONNECTIVITY_RADIUS_KM + 1

    non_pa = (pa_baseline == 0) & (backbone == 1)

    labeled_f = labeled.astype(np.float32)  # float for max/min filter speed

    print("  max_filter pass...")
    max_cl = maximum_filter(labeled_f, size=kernel_size)

    # For min: replace 0s with a sentinel > n_clusters so they don't count
    sentinel = float(n_clusters + 1)
    labeled_for_min = np.where(labeled_f > 0, labeled_f, sentinel)
    print("  min_filter pass...")
    min_cl = minimum_filter(labeled_for_min, size=kernel_size)

    # Connectivity gap: non-PA pixel where ≥2 distinct cluster IDs are within radius
    connectivity_gap = (
        non_pa
        & (min_cl < max_cl)     # at least 2 distinct IDs
        & (min_cl < sentinel)   # at least one of them is a real cluster
    ).astype(np.float32)
    connectivity_gap[backbone == 0] = np.nan

    n_gap = int((connectivity_gap == 1).sum())
    print(f"  Connectivity gap pixels: {n_gap:,} ({100*n_gap/max(int(non_pa.sum()),1):.2f}% of non-PA land)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rasterio.open(OUTPUT_PATH, "w", **out_profile) as dst:
        dst.write(connectivity_gap, 1)
        dst.update_tags(1, name="pa_connectivity_gap",
                        description="1=pixel bridges ≥2 PA clusters within 5 km (2000 WDPA baseline)")

    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Next:")
    print("  python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \\")
    print("      --tif data/south_america/ready/pa_connectivity_gap/pa_connectivity_gap_sa.tif \\")
    print("      --cols pa_connectivity_gap")
    print()
    print("NOTE: Verify SHAP rank before submitting Euler gate — may be redundant with GSN_b4.")


if __name__ == "__main__":
    main()
