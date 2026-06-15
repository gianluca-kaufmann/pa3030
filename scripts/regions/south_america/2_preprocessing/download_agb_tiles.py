"""Download ESA CCI Biomass v7.0 AGB tiles for South America.

Fetches the 2010 vintage (earliest available — minimises leakage for our 2001-2013
training window) from CEDA open-access storage.  No login required.

Tile naming: {TILE}_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif
Base URL: https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v7.0/geotiff/2010/

SA bounding box: lat -56 to +12, lon -82 to -34 → 10° tiles: N10, N00, S10–S50 × W040–W090
Only the AGB band is downloaded (not _SD uncertainty band) to save ~200 GB.

Output: data/shared/ESA_CCI_Biomass/<TILE>_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2010-fv7.0.tif
Then run: python scripts/regions/south_america/2_preprocessing/agb_rasterise.py

Usage:
  python scripts/regions/south_america/2_preprocessing/download_agb_tiles.py
  python scripts/regions/south_america/2_preprocessing/download_agb_tiles.py --year 2010 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = PROJECT_ROOT / "data/shared/ESA_CCI_Biomass"

BASE_URL = "https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v7.0/geotiff/{year}"
FILENAME_TPL = "{tile}_ESACCI-BIOMASS-L4-AGB-MERGED-100m-{year}-fv7.0.tif"

# All 10° tiles intersecting SA bounding box (lat -56 to +12, lon -82 to -34).
# Not all combinations exist (ocean tiles are absent); 404s are skipped silently.
SA_TILES = [
    # Northern SA
    "N10W090", "N10W080", "N10W070", "N10W060", "N10W050",
    "N00W090", "N00W080", "N00W070", "N00W060", "N00W050", "N00W040",
    # Tropical / subtropical
    "S10W080", "S10W070", "S10W060", "S10W050", "S10W040",
    "S20W080", "S20W070", "S20W060", "S20W050", "S20W040",
    # Temperate
    "S30W080", "S30W070", "S30W060", "S30W050",
    "S40W080", "S40W070", "S40W060",
    # Patagonia
    "S50W080", "S50W070", "S50W060", "S50W050", "S50W040",
]


def _download_tile(tile: str, year: int, out_dir: Path) -> tuple[str, str]:
    """Download one tile. Returns (tile, status) where status is 'ok'/'skip'/'error'."""
    filename = FILENAME_TPL.format(tile=tile, year=year)
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        return tile, "exists"

    url = f"{BASE_URL.format(year=year)}/{filename}"
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 404:
            return tile, "skip"
        r.raise_for_status()

        size = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                size += len(chunk)
        return tile, f"ok ({size/1e6:.0f} MB)"
    except Exception as e:
        return tile, f"error: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESA CCI Biomass v7.0 SA tiles")
    parser.add_argument("--year", type=int, default=2010,
                        help="Vintage year (default: 2010, earliest available)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel download threads (default: 3; be polite to CEDA)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ESA CCI Biomass v7.0 — {args.year} vintage")
    print(f"Output: {OUT_DIR}")
    print(f"Tiles:  {len(SA_TILES)} candidates (ocean tiles will be skipped as 404)")
    print(f"Threads: {args.workers}")
    print()

    results: dict[str, str] = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_tile, tile, args.year, OUT_DIR): tile
            for tile in SA_TILES
        }
        for fut in as_completed(futures):
            tile, status = fut.result()
            results[tile] = status
            icon = "✓" if status.startswith("ok") else ("·" if status in ("skip", "exists") else "✗")
            print(f"  {icon} {tile:10s}  {status}")

    elapsed = time.time() - t0
    n_ok = sum(1 for s in results.values() if s.startswith("ok") or s == "exists")
    n_new = sum(1 for s in results.values() if s.startswith("ok"))
    n_skip = sum(1 for s in results.values() if s == "skip")
    n_err = sum(1 for s in results.values() if s.startswith("error"))

    print()
    print(f"Done in {elapsed:.0f}s — {n_new} downloaded, {n_ok - n_new} already existed, "
          f"{n_skip} ocean/missing, {n_err} errors")

    if n_ok == 0:
        sys.exit("No tiles downloaded. Check connectivity to dap.ceda.ac.uk")

    print()
    print("Next step:")
    print("  python scripts/regions/south_america/2_preprocessing/agb_rasterise.py")


if __name__ == "__main__":
    main()
