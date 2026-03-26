#!/usr/bin/env python3
"""Stage 1: Extract 2024 inference feature set.

Filters merged_panel_final.parquet to:
  - year == 2024
  - WDPA_prev == 0   (pixel not protected as of 2023 — consistent with risk-set)
  - WDPA == 0        (pixel not protected in 2024 — not newly designated in 2024)

Both filters are required.  Without the WDPA==0 filter, pixels newly protected
in 2024 would have dist_wdpa≈0 in their feature row and receive inflated model
scores, despite already being protected.

Selects the same numeric training features (minus EXCLUDE_COLS) plus the y
(EPSG:3857 northing) coordinate for downstream area correction.

Output:
    outputs/{region}/results/forward/forward_features_2024.parquet
    (columns: row, col, x, y, <feature cols>)
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.regions.shared.forward.config import DATA_SUBDIR, OUTPUTS_SUBDIR, get_repo_root  # noqa: E402
from scripts.regions.shared.training.utils import WandbRunLogger  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
INFERENCE_YEAR = 2024
EXCLUDE_COLS = {
    "transition_01", "transition_01_win5", "WDPA_b1", "WDPA_prev",
    "x", "y", "row", "col", "year",
}
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "200_000"))
# ─────────────────────────────────────────────────────────────────────────────


def resolve_panel(data_subdir: str) -> Path:
    repo_root = get_repo_root()
    scratch = Path(os.environ["SCRATCH"]) if os.environ.get("SCRATCH") else None

    fn = "merged_panel_final.parquet"
    candidates: list[Path] = []
    if scratch is not None:
        candidates += [
            scratch / f"data/{data_subdir}/ml/{fn}",
            scratch / f"outputs/{data_subdir}/results/{fn}",
            scratch / f"outputs/{data_subdir}/results/main/{fn}",
        ]
    candidates += [
        repo_root / f"data/{data_subdir}/ml/{fn}",
        repo_root / f"outputs/{data_subdir}/results/{fn}",
        repo_root / f"outputs/{data_subdir}/results/main/{fn}",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"'{fn}' not found.\nChecked:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def extract_2024_features(
    panel_path: Path, output_dir: Path, wb: WandbRunLogger | None = None
) -> tuple[Path, int, int, str]:
    print("\n" + "=" * 70)
    print(f"EXTRACTING 2024 INFERENCE FEATURES (year={INFERENCE_YEAR})")
    print("=" * 70)
    print(f"  Source: {panel_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "forward_features_2024.parquet"

    # ── Determine columns ─────────────────────────────────────────────────────
    pf = pq.ParquetFile(panel_path)
    schema = pf.schema_arrow
    del pf

    all_numeric = [
        name for name, fld in zip(schema.names, schema)
        if pa.types.is_integer(fld.type) or pa.types.is_floating(fld.type)
    ]
    feature_cols = [c for c in all_numeric if c not in EXCLUDE_COLS]
    print(f"\n  Feature columns:  {len(feature_cols)}")
    if wb is not None:
        wb.log({"features/n_feature_cols": len(feature_cols), "features/stage": "schema"})

    # Metadata columns to preserve
    meta_cols = ["row", "col"]
    coord_cols = [c for c in ["x", "y"] if c in schema.names]
    print(f"  Coordinate cols:  {coord_cols}")

    # Columns needed for filtering
    filter_cols = ["year", "WDPA_prev", "WDPA"]
    read_cols = list(dict.fromkeys(meta_cols + coord_cols + filter_cols + feature_cols))
    read_cols = [c for c in read_cols if c in schema.names]

    # ── Pass 1: count qualifying rows ─────────────────────────────────────────
    print(f"\n  Pass 1: counting year=={INFERENCE_YEAR}, WDPA_prev==0, WDPA==0 rows…")
    n_total = 0
    pf = pq.ParquetFile(panel_path)
    try:
        for batch in pf.iter_batches(
            batch_size=BATCH_SIZE, columns=["year", "WDPA_prev", "WDPA"], use_threads=True
        ):
            yr   = batch["year"].to_numpy(zero_copy_only=False)
            wp   = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
            wdpa = batch["WDPA"].to_numpy(zero_copy_only=False)
            mask = (yr == INFERENCE_YEAR) & (wp == 0) & (wdpa == 0)
            n_total += int(mask.sum())
            del batch, yr, wp, wdpa, mask
    finally:
        del pf
        gc.collect()

    print(f"  Found {n_total:,} qualifying pixels")
    if wb is not None:
        wb.log({"features/n_rows_target": n_total, "features/stage": "pass1_done"})
    if n_total == 0:
        raise ValueError(
            f"No pixels found for year={INFERENCE_YEAR} with WDPA_prev==0 AND WDPA==0. "
            "Check that merged_panel_final.parquet contains year=2024 data."
        )

    # ── Pass 2: stream and write qualifying rows ───────────────────────────────
    print(f"\n  Pass 2: streaming and writing to {out_path.name}…")
    writer = None
    n_written = 0
    _mile = -1
    out_col_names = None

    pf = pq.ParquetFile(panel_path)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols, use_threads=True):
            yr   = batch["year"].to_numpy(zero_copy_only=False)
            wp   = batch["WDPA_prev"].to_numpy(zero_copy_only=False)
            wdpa = batch["WDPA"].to_numpy(zero_copy_only=False)
            mask = (yr == INFERENCE_YEAR) & (wp == 0) & (wdpa == 0)

            if not mask.any():
                del batch, yr, wp, wdpa, mask
                continue

            # Build output table (meta + coords + features), drop filter cols
            if out_col_names is None:
                out_col_names = [
                    c for c in (meta_cols + coord_cols + feature_cols)
                    if c in batch.schema.names
                ]
            pa_mask = pa.array(mask)
            out_arrays = {c: batch.column(c).filter(pa_mask) for c in out_col_names}
            out_table = pa.table(out_arrays)

            if writer is None:
                writer = pq.ParquetWriter(out_path, out_table.schema)
            writer.write_table(out_table)
            n_written += len(out_table)

            pct = n_written * 100 // n_total
            ms = pct // 25
            if ms > _mile:
                _mile = ms
                print(f"  {pct}% — {n_written:,}/{n_total:,}")
                if wb is not None:
                    wb.log(
                        {
                            "features/progress_pct": int(pct),
                            "features/n_rows_written": int(n_written),
                        }
                    )

            del batch, yr, wp, wdpa, mask, out_arrays, out_table
    finally:
        if writer is not None:
            writer.close()
        del pf
        gc.collect()

    if n_written != n_total:
        print(f"  WARNING: wrote {n_written:,} rows but counted {n_total:,}")

    print(f"\n  Written: {n_written:,} rows → {out_path}")
    if out_col_names is not None:
        print(f"  Columns: {out_col_names[:5]} … ({len(out_col_names)} total)")
    n_feature_cols = len(feature_cols)
    return out_path, n_written, n_feature_cols, str(panel_path)


def main() -> None:
    from datetime import datetime as _dt

    # Re-import config after runner.py reload
    from scripts.regions.shared.forward.config import DATA_SUBDIR, OUTPUTS_SUBDIR  # noqa: F401

    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    wb = WandbRunLogger(
        project="forward",
        run_name=f"features_{OUTPUTS_SUBDIR}_{_ts}",
        config={"region": OUTPUTS_SUBDIR, "forward_stage": "features"},
    )
    wb.start()
    wb.log({"features/stage": "start"})

    repo_root = get_repo_root()
    panel_path = resolve_panel(DATA_SUBDIR)
    output_dir = repo_root / f"outputs/{OUTPUTS_SUBDIR}/results/forward"
    out_path, n_rows, n_feat, panel_s = extract_2024_features(panel_path, output_dir, wb=wb)
    print(f"\nDone. Output: {out_path}")
    wb.log(
        {
            "features/n_rows": n_rows,
            "features/n_feature_cols": n_feat,
            "features/output_path": str(out_path),
            "features/panel_path": panel_s,
            "features/stage": "done",
        }
    )
    wb.finish()


if __name__ == "__main__":
    main()
