"""
Correlation / VIF (variance inflation factor) pruning for any built panel parquet.

Why this exists: Data Coverage Expansion (STATUS.md, decided 2026-08-07) adds WGI's 6
governance dimensions and DPI executive ideology alongside the existing V-Dem libdem —
these are known to be highly intercorrelated (typically pairwise rho=0.7-0.9), and
training on them uncorrected would multiply the multicollinearity risk already flagged
as a shared weak point of this project and PA3030 (see STATUS.md's frozen-reference
comparison). This script is the pre-training check STATUS.md commits to running before
the expanded feature set gets used anywhere.

VIF_i = 1 / (1 - R_i^2), where R_i^2 comes from regressing feature i on all other
retained features. Standard iterative pruning: repeatedly drop the single highest-VIF
feature above --threshold, recompute, repeat until every remaining feature clears the
threshold (dropping one at a time is required — VIF is a joint, not pairwise, quantity,
so it changes on every removal).

Runs on a random row sample (--sample-size), not the full dataset — VIF is a property
of the feature *correlation structure*, not something that needs every row, and a full
GBM-scale (~60M row) pooled dataset would make repeated OLS fits impractically slow.

Usage:
    python3 scripts/regions/world/vif_prune.py --path regions/british_columbia/data/pa_dataset.parquet
    python3 scripts/regions/world/vif_prune.py --path <parquet> --cols wgi_va,wgi_pv,wgi_ge,wgi_rq,wgi_rl,wgi_cc,dpi_exec_ideology,libdem
    python3 scripts/regions/world/vif_prune.py --path <parquet> --threshold 5.0 --out report.txt

Report is printed to stdout and written to <dataset>_vif_report.txt next to the input
file (or --out).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF_THRESHOLD_DEFAULT = 10.0   # conventional rule of thumb (5.0 is the stricter variant)
SAMPLE_SIZE_DEFAULT = 200_000
NON_FEATURE_DEFAULT = {"label"}


def compute_vif_table(X):
    """Return {col: vif} for the current column set. X must be finite (NaN-free)."""
    vifs = {}
    arr = X.values
    for i, col in enumerate(X.columns):
        try:
            vifs[col] = variance_inflation_factor(arr, i)
        except Exception:
            vifs[col] = np.inf  # perfectly collinear (e.g. a constant column)
    return vifs


def iterative_prune(X, threshold):
    """Repeatedly drop the single highest-VIF column above threshold. Returns
    (final_vif_table, dropped: [(col, vif_at_drop_time), ...])."""
    dropped = []
    cols = list(X.columns)
    while len(cols) > 1:
        vifs = compute_vif_table(X[cols])
        worst_col = max(vifs, key=vifs.get)
        worst_vif = vifs[worst_col]
        if worst_vif <= threshold:
            return vifs, dropped
        dropped.append((worst_col, worst_vif))
        cols.remove(worst_col)
    return (compute_vif_table(X[cols]) if cols else {}), dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help="Parquet to check")
    ap.add_argument("--cols", default=None,
                     help="Comma-separated feature columns to check (default: all numeric "
                          "columns except --exclude)")
    ap.add_argument("--exclude", default=",".join(sorted(NON_FEATURE_DEFAULT)),
                     help="Comma-separated columns to always exclude (label, ids, etc.)")
    ap.add_argument("--threshold", type=float, default=VIF_THRESHOLD_DEFAULT)
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE_DEFAULT)
    ap.add_argument("--corr-flag", type=float, default=0.7,
                     help="Also report raw pairwise |correlation| at or above this, "
                          "independent of the VIF pruning result (VIF can miss a strong "
                          "3-way relationship a pairwise check would still flag).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: {args.path} does not exist.")
        sys.exit(2)

    print(f"Loading: {args.path}")
    exclude = {c.strip() for c in args.exclude.split(",") if c.strip()}

    if args.cols:
        want_cols = [c.strip() for c in args.cols.split(",")]
        df = pd.read_parquet(args.path, columns=want_cols)
    else:
        df = pd.read_parquet(args.path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        want_cols = [c for c in numeric_cols if c not in exclude]
        df = df[want_cols]

    n_full = len(df)
    if n_full > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.seed)
        print(f"  Sampled {args.sample_size:,} / {n_full:,} rows for VIF computation")
    else:
        print(f"  Using all {n_full:,} rows (below --sample-size)")

    # Drop any column that's entirely NaN or constant (undefined/degenerate VIF),
    # median-impute the rest — VIF needs a finite, non-degenerate design matrix.
    degenerate = [c for c in df.columns if df[c].isna().all() or df[c].nunique(dropna=True) <= 1]
    if degenerate:
        print(f"  Dropping {len(degenerate)} degenerate column(s) (all-NaN or constant): {degenerate}")
        df = df.drop(columns=degenerate)
    df = df.fillna(df.median(numeric_only=True))

    lines = [f"VIF / correlation audit: {args.path}", f"Rows checked: {len(df):,} (of {n_full:,} total)",
              f"Columns checked: {len(df.columns)}", ""]

    # ── Pairwise correlation flag (independent of VIF, catches strong 2-way pairs) ──
    corr = df.corr()
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= args.corr_flag:
                pairs.append((a, b, r))
    pairs.sort(key=lambda p: -abs(p[2]))
    lines.append(f"## Pairwise |correlation| >= {args.corr_flag}")
    if pairs:
        for a, b, r in pairs:
            lines.append(f"  {a} <-> {b}: r={r:+.3f}")
    else:
        lines.append("  None.")

    # ── Iterative VIF pruning ────────────────────────────────────────────────────
    print("  Computing VIF (iterative pruning) …")
    final_vifs, dropped = iterative_prune(df, args.threshold)

    lines.append(f"\n## Iterative VIF pruning (threshold={args.threshold})")
    if dropped:
        lines.append(f"Dropped {len(dropped)} column(s), one at a time, highest-VIF-first:")
        for col, vif in dropped:
            vif_str = "inf" if np.isinf(vif) else f"{vif:.1f}"
            lines.append(f"  DROP {col}  (VIF={vif_str} at time of removal)")
    else:
        lines.append("None dropped — all columns already below threshold.")

    lines.append(f"\nFinal retained columns ({len(final_vifs)}), sorted by VIF:")
    for col, vif in sorted(final_vifs.items(), key=lambda kv: -kv[1]):
        vif_str = "inf" if np.isinf(vif) else f"{vif:.2f}"
        lines.append(f"  {col:<30} VIF={vif_str}")

    report = "\n".join(lines)
    print("\n" + report)

    out_path = args.out or (args.path.rsplit(".parquet", 1)[0] + "_vif_report.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
