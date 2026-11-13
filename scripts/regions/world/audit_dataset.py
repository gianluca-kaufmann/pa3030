"""
Data-quality audit for any built panel parquet (single region or pooled).

Why this exists: PA3030 (the now-frozen thesis project, see STATUS.md "PA3030 —
frozen reference") had a dedicated per-column missing-value + duplicate-row audit
script and this project didn't. That gap is what let `reserve_density_50km` (70%
NaN, silently median-imputed) slip all the way through a full Euler LOO-CV run
before anyone noticed — a whole session spent debugging a result that a five-second
audit would have flagged before training even started. This script is that check.

Checks (fast — uses parquet column statistics, no full-dataset load for the NaN
pass):
  1. Per-column missing-value fraction, flagged WARN / HIGH against thresholds.
  2. Exact duplicate rows on an identity key (year + pixel location) — should be
     impossible by construction (positives/negatives are sampled from disjoint,
     non-overlapping pixel sets per year), so any duplicate is a real bug, not
     noise.
  3. If --group-col is given (e.g. pool_region on a pooled dataset), also breaks
     down missingness per group — this is what would have caught session 42's
     "BC's AEF columns are ~all-NaN for the other 5 pooled regions" OOM-triggering
     bug before it silently doubled memory use.

Usage:
    python3 scripts/regions/world/audit_dataset.py
        (audits the current PA_REGION's built dataset)
    python3 scripts/regions/world/audit_dataset.py --path regions/pool/pooled_spots.parquet --group-col pool_region
    python3 scripts/regions/world/audit_dataset.py --path <parquet> --strict
        (exit code 1 if anything crosses the HIGH threshold or duplicates exist —
        for wiring into a pipeline step that should hard-stop on bad data)

Report is printed to stdout and written to <dataset>_qa_report.txt next to the
input file (or --out).
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow.parquet as pq


NAN_WARN_THRESHOLD = 0.20   # flag: worth a look
NAN_HIGH_THRESHOLD = 0.50   # flag: this column is mostly imputed, treat with suspicion

# Columns where missingness is expected/benign by design and shouldn't be flagged
# as a data-quality problem — imputed once, deliberately, at build_dataset.py
# (e.g. velocity features are zero-imputed for years without a t-3 pair, not NaN).
EXPECTED_SPARSE_COLS = frozenset({"embed_velocity_mag", "embed_velocity_cos"})


def per_column_null_stats(parquet_path):
    """Null fraction per column, computed from parquet row-group statistics —
    doesn't require loading the dataset into memory. Falls back to a full read
    only for columns missing statistics (rare, e.g. some pyarrow write paths)."""
    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    schema_names = pf.schema_arrow.names
    null_counts = {name: 0 for name in schema_names}
    missing_stats_cols = set()

    for rg in range(pf.metadata.num_row_groups):
        rg_meta = pf.metadata.row_group(rg)
        for col_i in range(rg_meta.num_columns):
            col_meta = rg_meta.column(col_i)
            name = col_meta.path_in_schema
            stats = col_meta.statistics
            if stats is not None and stats.has_null_count:
                null_counts[name] = null_counts.get(name, 0) + stats.null_count
            else:
                missing_stats_cols.add(name)

    if missing_stats_cols:
        print(f"  (no column stats for {len(missing_stats_cols)} col(s), "
              f"falling back to full read for those: {sorted(missing_stats_cols)[:5]}…)")
        df = pd.read_parquet(parquet_path, columns=sorted(missing_stats_cols))
        for name in missing_stats_cols:
            null_counts[name] = int(df[name].isna().sum())

    return n_rows, {name: null_counts.get(name, 0) for name in schema_names}


def find_duplicates(parquet_path, key_cols):
    df = pd.read_parquet(parquet_path, columns=key_cols)
    dup_mask = df.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_groups = int(df[dup_mask].drop_duplicates().shape[0]) if n_dup_rows else 0
    examples = df[dup_mask].drop_duplicates().head(5).to_dict("records") if n_dup_rows else []
    return n_dup_rows, n_dup_groups, examples


def infer_key_cols(columns):
    if {"year", "x_3857", "y_3857"}.issubset(columns):
        base = ["year", "x_3857", "y_3857"]
    elif {"year", "lon", "lat"}.issubset(columns):
        base = ["year", "lon", "lat"]
    else:
        return None
    if "pool_region" in columns:
        base = ["pool_region"] + base
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, help="Parquet to audit (default: current region's built dataset)")
    ap.add_argument("--group-col", default=None, help="Optional column to break missingness down by (e.g. pool_region)")
    ap.add_argument("--key-cols", nargs="+", default=None, help="Columns identifying a unique row (default: inferred)")
    ap.add_argument("--nan-warn", type=float, default=NAN_WARN_THRESHOLD)
    ap.add_argument("--nan-high", type=float, default=NAN_HIGH_THRESHOLD)
    ap.add_argument("--out", default=None, help="Report path (default: <dataset>_qa_report.txt)")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if anything crosses the HIGH threshold or duplicates exist")
    args = ap.parse_args()

    if args.path is None:
        ap.error("--path is required")

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} does not exist.")
        sys.exit(2)

    print(f"Auditing: {args.path}\n")

    n_rows, null_counts = per_column_null_stats(args.path)
    lines = [f"Data-quality audit: {args.path}", f"Rows: {n_rows:,}", ""]

    rows = []
    for name, n_null in null_counts.items():
        frac = n_null / n_rows if n_rows else 0.0
        rows.append((name, n_null, frac))
    rows.sort(key=lambda r: -r[2])

    high = [r for r in rows if r[2] >= args.nan_high and r[0] not in EXPECTED_SPARSE_COLS]
    warn = [r for r in rows if args.nan_warn <= r[2] < args.nan_high and r[0] not in EXPECTED_SPARSE_COLS]

    lines.append(f"## Missing-value audit (WARN >= {args.nan_warn:.0%}, HIGH >= {args.nan_high:.0%})")
    if high:
        lines.append(f"\nHIGH ({len(high)} column(s) — treat any feature here as suspect, likely a build/merge bug, not real signal):")
        for name, n_null, frac in high:
            lines.append(f"  [HIGH] {name}: {frac:.1%} missing ({n_null:,} / {n_rows:,})")
    if warn:
        lines.append(f"\nWARN ({len(warn)} column(s) — worth a look, not necessarily a problem):")
        for name, n_null, frac in warn:
            lines.append(f"  [warn] {name}: {frac:.1%} missing ({n_null:,} / {n_rows:,})")
    if not high and not warn:
        lines.append("\nNone — all columns below the WARN threshold.")

    lines.append("\n## Duplicate-row audit")
    key_cols = args.key_cols or infer_key_cols(set(pd.read_parquet(args.path, columns=[]).columns) | set(null_counts.keys()))
    if key_cols is None:
        lines.append("  SKIPPED — could not infer an identity key (need year+lon/lat or year+x_3857/y_3857 columns); pass --key-cols explicitly.")
        n_dup_rows = 0
    else:
        n_dup_rows, n_dup_groups, examples = find_duplicates(args.path, key_cols)
        lines.append(f"  Key columns: {key_cols}")
        if n_dup_rows:
            lines.append(f"  [HIGH] {n_dup_rows:,} duplicate row(s) across {n_dup_groups:,} distinct key(s) — "
                          f"should be impossible by construction (disjoint pos/neg sampling per year); "
                          f"this is a real bug, not noise.")
            for ex in examples:
                lines.append(f"    e.g. {ex}")
        else:
            lines.append(f"  None found ({n_rows:,} rows checked).")

    if args.group_col:
        lines.append(f"\n## Per-group missingness breakdown (--group-col {args.group_col})")
        df_g = pd.read_parquet(args.path, columns=[args.group_col] + [c for c, _, f in rows if f > 0])
        if df_g.empty or args.group_col not in df_g.columns:
            lines.append(f"  SKIPPED — column '{args.group_col}' not found.")
        else:
            group_keys = df_g[args.group_col]
            value_cols = [c for c in df_g.columns if c != args.group_col]
            group_null_frac = df_g[value_cols].groupby(group_keys).apply(lambda g: g.isna().mean())
            for col in group_null_frac.columns:
                worst = group_null_frac[col].sort_values(ascending=False)
                worst = worst[worst >= args.nan_high]
                if len(worst):
                    lines.append(f"  {col}: HIGH in group(s) {dict(worst.round(3))}")

    report = "\n".join(lines)
    print(report)

    out_path = args.out or (args.path.rsplit(".parquet", 1)[0] + "_qa_report.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written: {out_path}")

    if args.strict and (high or n_dup_rows):
        print("\nSTRICT mode: failing due to HIGH-severity findings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
