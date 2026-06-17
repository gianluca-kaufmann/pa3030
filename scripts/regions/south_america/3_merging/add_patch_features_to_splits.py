#!/usr/bin/env python3
"""Compute connected-component patch-context features (H11) and augment Stage 2 splits.

For each year t, "patches" are defined as 8-connected components of pixels with
WDPA_prev == 0 (unprotected land). Four features are computed per pixel:

  log_patch_size_km2:      log(n_pixels_in_patch) — at ~1 km resolution this ≈ log(km²)
  patch_pa_adjacency_frac: fraction of patch pixels with dist_wdpa ≤ 1500 m
                           (i.e., within one diagonal pixel of an existing PA).
                           Uses the dist_wdpa column already in the panel, which is
                           computed from the full grid including protected pixels.
  patch_mean_gsn_b2:       mean GSN_b2 biodiversity score across the patch
  patch_designation_lag1:  1 if any pixel adjacent to this patch was newly protected
                           in year t-1 (captures "designation momentum" —
                           the same area is being actively expanded)

Pipeline:
  1. Read merged_panel_final.parquet year by year.
  2. For each year t, reconstruct the 2D unprotected grid and run
     scipy.ndimage.label (8-connectivity) to assign patch IDs.
  3. Compute per-patch statistics and map back to pixels.
  4. Write patch_features_sa.parquet  (row, col, year, 4 features).
  5. Use DuckDB to LEFT JOIN patch features into each split (train/earlystop/test)
     and write augmented splits in-place via temp rename.

Usage (Euler, from repo root):
    srun python scripts/regions/south_america/3_merging/add_patch_features_to_splits.py

Env vars:
  SCRATCH — Euler scratch path (mandatory on Euler)
  STAGE2_DATA_ROOT — override splits directory
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import binary_dilation, label as scipy_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Distance threshold for "PA-adjacent": √2 × 1000 m ≈ 1414 m → use 1500 m
# so both straight (1000 m) and diagonal (1414 m) neighbours qualify.
PA_ADJACENCY_DIST_M: float = 1500.0

# 8-connectivity kernel for scipy.ndimage
_STRUCT8 = np.ones((3, 3), dtype=np.int32)

PATCH_FEATURE_COLS = [
    "log_patch_size_km2",
    "patch_pa_adjacency_frac",
    "patch_mean_gsn_b2",
    "patch_designation_lag1",
]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_panel() -> Path:
    scratch = os.environ.get("SCRATCH")
    candidates: list[Path] = []
    if scratch:
        candidates += [
            Path(scratch) / "data/south_america/ml/merged_panel_final.parquet",
            Path(scratch) / "outputs/south_america/results/merged_panel_final.parquet",
        ]
    candidates += [
        _ROOT / "data/south_america/ml/merged_panel_final.parquet",
        _ROOT / "outputs/south_america/results/merged_panel_final.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"merged_panel_final.parquet not found. Searched:\n  " +
        "\n  ".join(str(c) for c in candidates)
    )


def _resolve_splits_dir() -> Path:
    data_root = os.environ.get("STAGE2_DATA_ROOT")
    if data_root:
        for p in [Path(data_root) / "main", Path(data_root)]:
            if p.exists():
                return p
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "data/south_america/ml/main"
    raise RuntimeError("Set SCRATCH or STAGE2_DATA_ROOT to locate the splits directory.")


# ---------------------------------------------------------------------------
# Patch feature computation for one year
# ---------------------------------------------------------------------------

def _compute_year_patches(
    df: pd.DataFrame,
    seeds_prev: np.ndarray | None,
) -> pd.DataFrame:
    """Compute patch features for one year's Risk-Set pixels.

    Args:
        df:          All unprotected pixels for year t with columns
                     [row, col, dist_wdpa, GSN_b2].
                     These are ALL WDPA_prev==0 pixels — the full Risk Set footprint.
        seeds_prev:  (n, 2) int64 array of [row, col] for pixels with
                     transition_01==1 in year t-1 (newly designated last year).
                     None for the first year.

    Returns:
        DataFrame with columns [row, col] + PATCH_FEATURE_COLS, same length as df.
    """
    n = len(df)
    rows = df["row"].to_numpy(np.int64)
    cols = df["col"].to_numpy(np.int64)

    # ── Build 2D boolean grid of unprotected pixels ───────────────────────
    min_row = int(rows.min())
    max_row = int(rows.max())
    min_col = int(cols.min())
    max_col = int(cols.max())
    h = max_row - min_row + 1
    w = max_col - min_col + 1

    r = (rows - min_row).astype(np.int64)
    c = (cols - min_col).astype(np.int64)

    unprotected = np.zeros((h, w), dtype=np.bool_)
    unprotected[r, c] = True

    # ── Connected-component labelling (8-connectivity) ───────────────────
    labeled, n_labels = scipy_label(unprotected, structure=_STRUCT8)
    pixel_labels = labeled[r, c]  # values 1..n_labels for unprotected pixels

    if n_labels == 0:
        result = df[["row", "col"]].copy()
        for col_name in PATCH_FEATURE_COLS:
            result[col_name] = np.float32(np.nan)
        return result

    # ── Feature 1: patch size (number of pixels ≈ km²) ──────────────────
    patch_sizes = np.bincount(pixel_labels.astype(np.intp), minlength=n_labels + 1)
    # patch_sizes[0] = background pixels (shouldn't exist here since all are labeled)

    # ── Feature 2: PA adjacency fraction ────────────────────────────────
    # Proxy: fraction of patch pixels with dist_wdpa ≤ PA_ADJACENCY_DIST_M.
    # dist_wdpa was computed from the FULL grid (protected + unprotected) during
    # feature_engineering, so it correctly captures distance to PA-occupied pixels.
    dist_wdpa = df["dist_wdpa"].to_numpy(np.float64)
    pa_adj_flag = (dist_wdpa <= PA_ADJACENCY_DIST_M).astype(np.float64)
    patch_pa_adj_sum = np.bincount(
        pixel_labels.astype(np.intp),
        weights=pa_adj_flag,
        minlength=n_labels + 1,
    )

    # ── Feature 3: mean GSN biodiversity (b2) ───────────────────────────
    gsn = df["GSN_b2"].to_numpy(np.float64)
    valid_gsn = ~np.isnan(gsn)
    patch_gsn_sum = np.bincount(
        pixel_labels[valid_gsn].astype(np.intp),
        weights=gsn[valid_gsn],
        minlength=n_labels + 1,
    )
    patch_gsn_cnt = np.bincount(
        pixel_labels[valid_gsn].astype(np.intp),
        minlength=n_labels + 1,
    )

    # ── Feature 4: designation lag1 ──────────────────────────────────────
    # A patch receives designation_lag1=1 if any pixel ADJACENT to it was
    # newly protected last year (seeds_prev).  "Adjacent" = within 1 pixel
    # (8-connectivity), captured by dilating the seed grid by 1 step.
    if seeds_prev is not None and len(seeds_prev) > 0:
        sr = seeds_prev[:, 0] - min_row
        sc = seeds_prev[:, 1] - min_col
        in_bounds = (sr >= 0) & (sr < h) & (sc >= 0) & (sc < w)
        sr, sc = sr[in_bounds], sc[in_bounds]

        seed_grid = np.zeros((h, w), dtype=np.bool_)
        if len(sr) > 0:
            seed_grid[sr, sc] = True
        # Dilate seeds and intersect with unprotected to find adjacent pixels
        seed_adj = binary_dilation(seed_grid, structure=_STRUCT8) & unprotected
        del seed_grid
        pixel_seed_adj = seed_adj[r, c].astype(np.float64)
        del seed_adj
        patch_seed_sum = np.bincount(
            pixel_labels.astype(np.intp),
            weights=pixel_seed_adj,
            minlength=n_labels + 1,
        )
        lag1_available = True
    else:
        lag1_available = False

    # ── Map patch-level stats back to pixels ─────────────────────────────
    safe_sizes = np.maximum(patch_sizes, 1).astype(np.float64)

    pixel_size      = patch_sizes[pixel_labels]
    pixel_pa_frac   = patch_pa_adj_sum[pixel_labels] / safe_sizes[pixel_labels]

    safe_gsn_cnt    = np.maximum(patch_gsn_cnt, 1).astype(np.float64)
    pixel_gsn_mean  = np.where(
        patch_gsn_cnt[pixel_labels] > 0,
        patch_gsn_sum[pixel_labels] / safe_gsn_cnt[pixel_labels],
        np.nan,
    )

    result = df[["row", "col"]].copy()
    result["log_patch_size_km2"]      = np.log(np.maximum(pixel_size, 1)).astype(np.float32)
    result["patch_pa_adjacency_frac"] = pixel_pa_frac.astype(np.float32)
    result["patch_mean_gsn_b2"]       = pixel_gsn_mean.astype(np.float32)

    if lag1_available:
        result["patch_designation_lag1"] = (patch_seed_sum[pixel_labels] > 0).astype(np.float32)
    else:
        result["patch_designation_lag1"] = np.float32(0.0)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    panel_path  = _resolve_panel()
    splits_dir  = _resolve_splits_dir()
    scratch     = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else _ROOT

    patch_out   = scratch / "data/south_america/ml/patch_features_sa.parquet"
    patch_out.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("PATCH FEATURE COMPUTATION  (H11)")
    log.info("=" * 70)
    log.info(f"Panel:       {panel_path}")
    log.info(f"Splits dir:  {splits_dir}")
    log.info(f"Patch output:{patch_out}")

    # ── Discover years ────────────────────────────────────────────────────
    ydf = pd.read_parquet(panel_path, columns=["year"])
    years = sorted(ydf["year"].unique())
    del ydf
    gc.collect()
    log.info(f"Years: {years[0]}–{years[-1]}  ({len(years)} years)")

    # ── Compute patch features year by year ───────────────────────────────
    READ_COLS = ["row", "col", "year", "dist_wdpa", "GSN_b2", "transition_01"]

    writer: pq.ParquetWriter | None = None
    seeds_prev: np.ndarray | None = None  # (row, col) of t-1 designations
    t_total = time.time()

    for i, year in enumerate(years):
        t0 = time.time()
        log.info(f"  Year {year} [{i+1}/{len(years)}]...")

        df = pq.read_table(
            panel_path,
            columns=READ_COLS,
            filters=[("year", "==", year)],
        ).to_pandas()

        if df.empty:
            log.warning(f"    Year {year}: no rows — skipping")
            seeds_prev = None
            continue

        # Compute patch features
        feat_df = _compute_year_patches(df, seeds_prev)
        feat_df["year"] = np.int16(year)

        # Save seeds for next year's designation_lag1
        designated = df[df["transition_01"] > 0][["row", "col"]]
        seeds_prev = designated.to_numpy(dtype=np.int64) if len(designated) > 0 else None

        del df
        gc.collect()

        # Write to patch features parquet
        table = pa.Table.from_pandas(feat_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(patch_out, table.schema, compression="snappy")
        writer.write_table(table)
        del feat_df, table
        gc.collect()

        n_patches_est = None  # logged from within the function if needed
        log.info(f"    Done in {time.time()-t0:.0f}s")

    if writer is not None:
        writer.close()
    log.info(f"Patch features written: {patch_out}  ({patch_out.stat().st_size/1e9:.2f} GB)")
    log.info(f"Total patch computation: {time.time()-t_total:.0f}s")

    # ── Augment splits via DuckDB LEFT JOIN ───────────────────────────────
    try:
        import duckdb
    except ImportError:
        log.error("duckdb not installed — augmenting splits with PyArrow merge instead")
        _augment_splits_pyarrow(splits_dir, patch_out)
        return

    log.info("\nAugmenting splits with DuckDB LEFT JOIN...")
    duckdb_tmp = scratch / "data/south_america/ml/.duckdb_tmp"
    duckdb_tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '100GB'")
    con.execute(f"SET temp_directory = '{duckdb_tmp}'")

    patch_path_sql = str(patch_out).replace("'", "''")

    for split_name in ("train.parquet", "earlystop.parquet", "test.parquet"):
        split_path = splits_dir / split_name
        if not split_path.exists():
            log.warning(f"  SKIP (not found): {split_path}")
            continue

        tmp_path = split_path.with_suffix(".parquet.new")
        split_sql = str(split_path).replace("'", "''")
        tmp_sql   = str(tmp_path).replace("'", "''")

        log.info(f"  {split_name}  ({split_path.stat().st_size/1e9:.1f} GB)...")
        t0 = time.time()

        # Drop old patch columns if present (idempotent re-run)
        drop_cols = ", ".join(
            f"s.{c}" for c in PATCH_FEATURE_COLS
        )
        select_others = f"""
            SELECT * EXCLUDE ({", ".join(PATCH_FEATURE_COLS)})
            FROM read_parquet('{split_sql}')
        """
        # Check if patch columns already exist in this split
        existing = con.execute(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name='dummy'"
        )
        # Simpler: always try to exclude; DuckDB ignores non-existent columns in EXCLUDE
        try:
            con.execute(f"""
                COPY (
                    SELECT s.* EXCLUDE ({", ".join(PATCH_FEATURE_COLS)}),
                        p.log_patch_size_km2,
                        p.patch_pa_adjacency_frac,
                        p.patch_mean_gsn_b2,
                        p.patch_designation_lag1
                    FROM read_parquet('{split_sql}') s
                    LEFT JOIN read_parquet('{patch_path_sql}') p
                        ON s.row = p.row AND s.col = p.col AND s.year = p.year
                ) TO '{tmp_sql}' (FORMAT PARQUET, CODEC 'SNAPPY')
            """)
        except Exception:
            # EXCLUDE not available in older DuckDB — fall back
            con.execute(f"""
                COPY (
                    SELECT s.*,
                        p.log_patch_size_km2,
                        p.patch_pa_adjacency_frac,
                        p.patch_mean_gsn_b2,
                        p.patch_designation_lag1
                    FROM read_parquet('{split_sql}') s
                    LEFT JOIN read_parquet('{patch_path_sql}') p
                        ON s.row = p.row AND s.col = p.col AND s.year = p.year
                ) TO '{tmp_sql}' (FORMAT PARQUET, CODEC 'SNAPPY')
            """)

        tmp_path.rename(split_path)
        log.info(f"    Augmented in {time.time()-t0:.0f}s → {split_path}")

    con.close()
    log.info("Done. All splits augmented with patch features.")


def _augment_splits_pyarrow(splits_dir: Path, patch_path: Path) -> None:
    """Fallback: augment splits with PyArrow batch merge (no DuckDB)."""
    log.info("Augmenting splits with PyArrow batch merge (year-by-year)...")

    patch_years_df = pq.read_table(patch_path, columns=["year"]).to_pandas()
    patch_years = sorted(patch_years_df["year"].unique())

    for split_name in ("train.parquet", "earlystop.parquet", "test.parquet"):
        split_path = splits_dir / split_name
        if not split_path.exists():
            log.warning(f"  SKIP: {split_path}")
            continue

        tmp_path = split_path.with_suffix(".parquet.new")
        log.info(f"  {split_name}...")
        t0 = time.time()

        pf = pq.ParquetFile(split_path)
        schema_cols = pf.schema_arrow.names
        # Drop existing patch columns if present
        drop = set(PATCH_FEATURE_COLS)

        writer: pq.ParquetWriter | None = None
        rows_written = 0

        for batch in pf.iter_batches(batch_size=2_000_000):
            df = batch.to_pandas()
            # Drop stale patch cols
            df = df[[c for c in df.columns if c not in drop]]

            years_in_batch = df["year"].unique()
            # Build lookup from patch features for the years in this batch
            patch_dfs = []
            for yr in years_in_batch:
                if yr not in patch_years:
                    continue
                p = pq.read_table(
                    patch_path,
                    columns=["row", "col", "year"] + PATCH_FEATURE_COLS,
                    filters=[("year", "==", int(yr))],
                ).to_pandas()
                patch_dfs.append(p)

            if patch_dfs:
                patch_batch = pd.concat(patch_dfs, ignore_index=True)
                df = df.merge(patch_batch, on=["row", "col", "year"], how="left")
            else:
                for col in PATCH_FEATURE_COLS:
                    df[col] = np.float32(np.nan)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
            writer.write_table(table)
            rows_written += len(table)

        if writer is not None:
            writer.close()
            tmp_path.rename(split_path)
            log.info(f"    {rows_written:,} rows written in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
