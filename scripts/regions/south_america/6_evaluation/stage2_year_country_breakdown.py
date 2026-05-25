#!/usr/bin/env python3
"""Issue C diagnostic — SA Stage 2 performance by year and country.

Loads the most recent model1_lgbm_stage2_scored_*.parquet and computes
NDCG@1% and Lift@1% within (country_id, year) groups, broken down by:
  - year  (2017, 2018, 2019)
  - country_id (if column present in parquet)

Hypothesis: 2019 Brazil collapse → Bolsonaro structural break.
If uniform across years/countries → escalate to Issue D (LambdaRank 9K sub-window).

Usage:
    python stage2_year_country_breakdown.py [--parquet PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.evaluation.stage2_metrics import (
    ndcg_at_k_within_groups,
    precision_at_k_within_groups,
    lift_at_k_within_groups,
)

OUT_DIR = _ROOT / "outputs" / "south_america" / "results" / "ml_models"
SCORED_GLOB = "model1_lgbm_stage2_scored_*.parquet"

# Numerical country_id → ISO3 (matches ISO3_TO_ID in stage1_data_builder.py)
ID_TO_ISO3 = {
    1: "ARG", 2: "BOL", 3: "BRA", 4: "CHL", 5: "COL", 6: "ECU",
    7: "GUY", 8: "PRY", 9: "PER", 10: "SUR", 11: "URY", 12: "VEN",
}


def _metrics_for_subset(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-group NDCG@1% and Lift@1% for a slice of the scored df."""
    df = df.sort_values(["country_id", "year"]).reset_index(drop=True)
    group_sizes = df.groupby(["country_id", "year"], sort=True).size().to_numpy(dtype=np.int32)
    y = df["y_true"].to_numpy(dtype=np.float64)
    s = df["y_pred_score"].to_numpy(dtype=np.float64)
    prec = precision_at_k_within_groups(y, s, group_sizes, 1.0)
    baseline = float((y > 0).mean()) if len(y) else 0.0
    lift = prec / baseline if baseline > 0 else 0.0
    return {
        "ndcg_at_1pct": ndcg_at_k_within_groups(y, s, group_sizes, 1.0),
        "lift_at_1pct": lift,
        "precision_at_1pct": prec,
        "baseline_rate": baseline,
        "n_groups": int(len(group_sizes)),
        "n_rows": int(len(df)),
    }


def _find_scored_parquet(override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")
        return p
    # Search SCRATCH first, then local outputs
    candidates: list[Path] = []
    scratch = Path(__file__).resolve().parents[4]  # repo root fallback
    import os
    if os.environ.get("SCRATCH"):
        scratch_dir = Path(os.environ["SCRATCH"]) / "outputs" / "south_america" / "results" / "ml_models"
        candidates.extend(sorted(scratch_dir.glob(SCORED_GLOB), reverse=True))
    candidates.extend(sorted(OUT_DIR.glob(SCORED_GLOB), reverse=True))
    # Filter out naive-baseline files
    candidates = [p for p in candidates if "_naive_" not in p.name]
    if not candidates:
        raise FileNotFoundError(
            f"No scored parquet found in {OUT_DIR} or $SCRATCH. "
            f"Run model1_LGBM_stage2 first, or pass --parquet PATH."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="SA Stage 2 year/country breakdown")
    parser.add_argument("--parquet", default=None, help="Path to scored parquet (optional)")
    parser.add_argument("--out", default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    parquet_path = _find_scored_parquet(args.parquet)
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")

    has_country = "country_id" in df.columns
    if not has_country:
        print(
            "  NOTE: country_id column absent in this parquet (old run). "
            "Country breakdown unavailable. Year breakdown only."
        )
        # Reconstruct a dummy country_id=0 so _metrics_for_subset still works
        df["country_id"] = 0

    # --- overall ---
    print("\n=== Overall (test 2017–2024) ===")
    overall = _metrics_for_subset(df)
    print(f"  NDCG@1%: {overall['ndcg_at_1pct']:.4f}")
    print(f"  Lift@1%: {overall['lift_at_1pct']:.2f}×")
    print(f"  n_groups: {overall['n_groups']}")

    # --- by year ---
    print("\n=== By year ===")
    by_year: dict[str, dict] = {}
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        m = _metrics_for_subset(sub)
        iso = ID_TO_ISO3
        print(f"  {yr}: NDCG@1%={m['ndcg_at_1pct']:.4f}  Lift={m['lift_at_1pct']:.2f}×  "
              f"n_groups={m['n_groups']}  n_rows={m['n_rows']:,}")
        by_year[str(yr)] = m

    # --- by country (if available) ---
    by_country: dict[str, dict] = {}
    if has_country:
        print("\n=== By country (across 2017–2019) ===")
        for cid in sorted(df["country_id"].unique()):
            sub = df[df["country_id"] == cid]
            m = _metrics_for_subset(sub)
            iso3 = ID_TO_ISO3.get(int(cid), f"ID{cid}")
            print(f"  {iso3} (id={cid}): NDCG@1%={m['ndcg_at_1pct']:.4f}  "
                  f"Lift={m['lift_at_1pct']:.2f}×  n_rows={m['n_rows']:,}")
            by_country[iso3] = m

        # --- Brazil 2019 vs rest ---
        print("\n=== Bolsonaro hypothesis: Brazil 2019 vs rest ===")
        bra_2019 = df[(df["country_id"] == 3) & (df["year"] == 2019)]
        other = df[~((df["country_id"] == 3) & (df["year"] == 2019))]
        if len(bra_2019) > 0:
            m_bra = _metrics_for_subset(bra_2019)
            print(f"  BRA 2019: NDCG@1%={m_bra['ndcg_at_1pct']:.4f}  "
                  f"Lift={m_bra['lift_at_1pct']:.2f}×  n_rows={len(bra_2019):,}")
        if len(other) > 0:
            m_other = _metrics_for_subset(other)
            print(f"  Rest:     NDCG@1%={m_other['ndcg_at_1pct']:.4f}  "
                  f"Lift={m_other['lift_at_1pct']:.2f}×  n_rows={len(other):,}")

    result = {
        "parquet": str(parquet_path),
        "overall": overall,
        "by_year": by_year,
        "by_country": by_country,
        "has_country_id": has_country,
    }
    out_path = Path(args.out) if args.out else OUT_DIR / "stage2_SA_year_country_breakdown.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
