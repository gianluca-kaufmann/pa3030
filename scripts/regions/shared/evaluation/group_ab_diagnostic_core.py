#!/usr/bin/env python3
"""Group A / Group B leakage diagnostic — shared core.

Splits test-set positives by designation-year cohort to assess how much of
the model's reported test performance is explained by soft label leakage.

Background
----------
The training label `transition_01_win5=1` at year Y means "this pixel will
become a protected area in years Y+1 … Y+5".  When a pixel is designated in
year D (e.g. D=2019), it carries label=1 for training rows at years D-5 … D-1.
If D-5 ≤ max_train_year (2013), that pixel appeared as a positive *during
model training*, so it is not truly "unseen" at test time — the model has
already learned its feature profile as a positive.

Groups derived from the test parquet alone
------------------------------------------
  Group A (definite, D=2018 or D=2019)
    transition_01=1 at test year 2018 → designated in 2018 (appeared in
      training rows at year 2013 and earlystop rows 2014-2016)
    transition_01=1 at test year 2019 → designated in 2019 (appeared in
      earlystop rows 2014-2016)
    → Soft label leakage present: model saw these pixels as positives.

  Group B (definite, D=2023 or D=2024)
    First appear as win5-positive at test year 2018 → D=2023
    First appear as win5-positive at test year 2019 → D=2024
    → Never appeared as positive in any training or earlystop row.
    → Genuinely unseen positives: the cleanest OOS signal.

  Ambiguous (D=2020–2022)
    Positive at all three test years; cannot be resolved from test data alone.
    D=2020 and D=2021 are Group A (appeared in earlystop rows 2015-2016).
    D=2022 is Group B (earlystop max window 2016+5=2021 < 2022).
    Reported separately.  Resolve via --extended on Euler (reads train parquet).

Metric computation
------------------
For each group, y_true is relabelled on the FULL test set:
  y_true_g[i] = 1  if row i is a positive AND its pixel belongs to group g
              = 0  otherwise
AUC, Precision@K, Lift@K are computed over this relabelled set.  This tests
whether the model ranks group-g positives above the full negative background.

Memory profile (SA, 47 M rows)
--------------------------------
  Scored parquet (5 cols):  ~750 MB
  Test parquet (batched):   ~55 MB peak/batch → ~2 MB after filter
  Merge + group column:     ~800 MB total
  sklearn AUC pass:         + ~240 MB transient
  Peak:                     ~1.1 GB  (safe on 8 GB Mac)
"""

from __future__ import annotations

import gc
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score


# ── Group codes stored as int8 to minimise memory ─────────────────────────────
GROUP_A = np.int8(1)        # D=2018-2019 (definite leakage)
GROUP_B = np.int8(2)        # D=2023-2024 (genuinely unseen)
GROUP_AMB = np.int8(3)      # D=2020-2022 (ambiguous)
GROUP_NONE = np.int8(0)     # not a positive pixel

GROUP_LABELS: Dict[int, str] = {
    GROUP_A: "Group A (D=2018-2019, definite leakage)",
    GROUP_B: "Group B (D=2023-2024, genuinely unseen)",
    GROUP_AMB: "Ambiguous (D=2020-2022)",
}


# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_latest_scored_parquet(
    metrics_root: Path,
    model_id: str,
    model_type: Literal["lgbm", "rf"],
    split_version: str = "main",
) -> Path:
    """Return the most recently timestamped scored-and-calibrated parquet."""
    if model_type == "rf":
        pattern = f"{model_id}_rf_win5_scored_calibrated_platt_*.parquet"
        ts_re = re.compile(
            rf"{re.escape(model_id)}_rf_win5_scored_calibrated_platt_(\d{{8}}_\d{{6}})\.parquet"
        )
    else:
        pattern = f"{model_id}_lgbm_scored_calibrated_isotonic_*.parquet"
        ts_re = re.compile(
            rf"{re.escape(model_id)}_lgbm_scored_calibrated_isotonic_(\d{{8}}_\d{{6}})\.parquet"
        )

    search_dirs = [metrics_root / split_version, metrics_root]

    def _ts_key(path: Path) -> datetime:
        m = ts_re.match(path.name)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        return datetime.fromtimestamp(path.stat().st_mtime)

    for d in search_dirs:
        if not d.exists():
            continue
        files = list(d.glob(pattern))
        if files:
            return max(files, key=_ts_key)

    searched = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"No scored parquet found for {model_id} {model_type}. Searched: {searched}"
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_scored(scored_path: Path) -> pd.DataFrame:
    """Load scored parquet with compact dtypes.

    Loads only the 5 columns needed for the diagnostic and downcasts to
    save ~35% memory vs loading all 9 columns at native types.
    """
    print(f"  Loading scored parquet: {scored_path.name}")
    df = pd.read_parquet(
        scored_path,
        columns=["row", "col", "year", "y_true", "y_pred_proba"],
    )
    df["year"]       = df["year"].astype("int16")
    df["y_true"]     = df["y_true"].astype("int8")
    df["y_pred_proba"] = df["y_pred_proba"].astype("float32")
    n = len(df)
    n_pos = int(df["y_true"].sum())
    print(f"    {n:,} rows  |  {n_pos:,} positives  ({n_pos/n*100:.3f}%)")
    return df


def _build_designation_map(test_path: Path) -> Dict[Tuple[int, int], int]:
    """Read test parquet in 5 M-row batches; return {(row,col): designation_year}.

    transition_01=1 at test year Y means the pixel was designated in year Y.
    Only years 2017, 2018, 2019 can appear; we care about 2018 and 2019
    (Group A: D=2018-2019).
    """
    print(f"  Reading test parquet for transition_01 (batched): {test_path.name}")
    pf = pq.ParquetFile(test_path)
    chunks = []
    for batch in pf.iter_batches(
        batch_size=5_000_000, columns=["row", "col", "year", "transition_01"]
    ):
        tmp = batch.to_pandas()
        mask = tmp["transition_01"] == 1
        if mask.any():
            chunks.append(tmp[mask][["row", "col", "year"]].copy())
        del tmp, mask

    if not chunks:
        print("    No transition_01=1 rows found in test parquet.")
        return {}

    desig_df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    desig_map: Dict[Tuple[int, int], int] = dict(
        zip(zip(desig_df["row"].astype(int), desig_df["col"].astype(int)),
            desig_df["year"].astype(int))
    )
    by_year = desig_df.groupby("year").size().sort_index().to_dict()
    print(f"    {len(desig_map):,} designation events  |  by year: {by_year}")
    return desig_map


# ── Classification ─────────────────────────────────────────────────────────────

def _classify_pixels(
    scored: pd.DataFrame,
    desig_map: Dict[Tuple[int, int], int],
) -> pd.Series:
    """Return a Series (indexed like *scored*) with int8 group codes.

    Algorithm:
      1. Find minimum win5-positive test year for each positive pixel.
      2. Cross-reference with designation map (transition_01=1) to identify
         pixels with known D=2018 or D=2019 → Group A.
      3. Remaining positives classified by min_pos_year:
           2018 → D=2023 → Group B
           2019 → D=2024 → Group B
           2017 → D=2020-2022 → Ambiguous
    """
    print("  Classifying pixels into Group A / B / Ambiguous …")

    # --- Step 1: minimum positive year per pixel ----------------------------
    pos_mask = scored["y_true"] == 1
    pos = scored.loc[pos_mask, ["row", "col", "year"]].copy()
    min_pos_year = (
        pos.groupby(["row", "col"])["year"]
        .min()
        .rename("min_pos_year")
        .reset_index()
    )  # unique positive pixels only (~200K for SA)

    # --- Step 2: look up designation year -----------------------------------
    # Build a small DataFrame from the designation map for pandas merge
    if desig_map:
        desig_df = pd.DataFrame(
            [(r, c, y) for (r, c), y in desig_map.items()],
            columns=["row", "col", "desig_year"],
        )
        # Ensure matching dtypes for merge
        desig_df["row"] = desig_df["row"].astype(min_pos_year["row"].dtype)
        desig_df["col"] = desig_df["col"].astype(min_pos_year["col"].dtype)
        pixel_info = min_pos_year.merge(desig_df, on=["row", "col"], how="left")
    else:
        min_pos_year["desig_year"] = np.nan
        pixel_info = min_pos_year

    # --- Step 3: assign group codes -----------------------------------------
    group_codes = np.full(len(pixel_info), GROUP_NONE, dtype=np.int8)

    # Group A: designated in 2018 or 2019 (confirmed via transition_01)
    is_a = pixel_info["desig_year"].isin([2018, 2019])
    group_codes[is_a.values] = GROUP_A

    # Among pixels NOT identified via transition_01 (D ≥ 2020):
    no_desig = ~is_a & pixel_info["desig_year"].isna()
    first_in_2018 = no_desig & (pixel_info["min_pos_year"] == 2018)
    first_in_2019 = no_desig & (pixel_info["min_pos_year"] == 2019)
    first_in_2017 = no_desig & (pixel_info["min_pos_year"] == 2017)

    group_codes[first_in_2018.values] = GROUP_B     # D=2023
    group_codes[first_in_2019.values] = GROUP_B     # D=2024
    group_codes[first_in_2017.values] = GROUP_AMB   # D=2020-2022

    pixel_info["group"] = group_codes

    # Stats
    for code, label in GROUP_LABELS.items():
        n = int((group_codes == code).sum())
        print(f"    {label}: {n:,} unique pixels")
    # D=2017 edge case: transition_01=1 at 2017 (not a win5 positive, should be 0)
    n_d2017 = int(pixel_info["desig_year"].eq(2017).sum())
    if n_d2017 > 0:
        print(f"    Note: {n_d2017} pixels have desig_year=2017 (not win5 positives)")

    # --- Step 4: merge group codes back to ALL scored rows ------------------
    # Merge positive-pixel groups onto the full 47 M-row scored DataFrame.
    # Non-positive pixels get GROUP_NONE (already 0 by default).
    scored_with_group = scored.merge(
        pixel_info[["row", "col", "group"]],
        on=["row", "col"],
        how="left",
    )
    group_col = scored_with_group["group"].fillna(GROUP_NONE).astype("int8")

    return group_col


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_precision_at_k(
    y_true: np.ndarray, y_proba: np.ndarray, k: float
) -> float:
    """Precision in the top k% of predictions (by predicted probability)."""
    n_top = max(1, int(len(y_true) * k / 100))
    top_idx = np.argpartition(y_proba, -n_top)[-n_top:]
    return float(y_true[top_idx].sum() / n_top)


def _compute_recall_at_k(
    y_true: np.ndarray, y_proba: np.ndarray, k: float
) -> float:
    """Fraction of positives captured in top k% of predictions."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    n_top = max(1, int(len(y_true) * k / 100))
    top_idx = np.argpartition(y_proba, -n_top)[-n_top:]
    return float(y_true[top_idx].sum() / n_pos)


def _metrics_for_group(
    y_true_all: np.ndarray,
    y_proba: np.ndarray,
    group_mask: np.ndarray,
) -> Dict:
    """Compute metrics over the full test set for one group.

    y_true_group[i] = 1  iff  y_true_all[i]==1 AND group_mask[i]
                 = 0  otherwise
    """
    y_true_g = (y_true_all & group_mask).astype(np.int8)
    n_total = len(y_true_g)
    n_pos = int(y_true_g.sum())

    if n_pos == 0:
        return {"n_positives": 0, "n_total": n_total, "note": "no positives"}

    baseline = n_pos / n_total
    roc = float(roc_auc_score(y_true_g, y_proba))
    pr  = float(average_precision_score(y_true_g, y_proba))

    p1  = _compute_precision_at_k(y_true_g, y_proba, 1)
    p5  = _compute_precision_at_k(y_true_g, y_proba, 5)
    p10 = _compute_precision_at_k(y_true_g, y_proba, 10)
    r1  = _compute_recall_at_k(y_true_g, y_proba, 1)
    r5  = _compute_recall_at_k(y_true_g, y_proba, 5)
    r10 = _compute_recall_at_k(y_true_g, y_proba, 10)

    return {
        "n_positives":        n_pos,
        "n_total":            n_total,
        "positive_rate_pct":  round(baseline * 100, 4),
        "roc_auc":            round(roc, 6),
        "pr_auc":             round(pr, 6),
        "precision_at_1pct":  round(p1, 6),
        "precision_at_5pct":  round(p5, 6),
        "precision_at_10pct": round(p10, 6),
        "recall_at_1pct":     round(r1, 6),
        "recall_at_5pct":     round(r5, 6),
        "recall_at_10pct":    round(r10, 6),
        "lift_at_1pct":       round(p1 / baseline, 4) if baseline > 0 else 0,
        "lift_at_5pct":       round(p5 / baseline, 4) if baseline > 0 else 0,
        "lift_at_10pct":      round(p10 / baseline, 4) if baseline > 0 else 0,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_group_metrics(label: str, m: Dict) -> None:
    if "note" in m and m.get("n_positives", 0) == 0:
        print(f"\n  {label}: no positives — skipped.")
        return
    w = 74
    print()
    print(f"  {label}")
    print(f"  {'─' * (w - 2)}")
    print(f"    Positives: {m['n_positives']:>10,}  |  "
          f"Rate: {m['positive_rate_pct']:.4f}%  |  "
          f"Total rows: {m['n_total']:,}")
    print(f"    ROC-AUC:  {m['roc_auc']:.4f}   PR-AUC:  {m['pr_auc']:.4f}")
    print(f"    P@1%:  {m['precision_at_1pct']:.4f}   Lift@1%:  {m['lift_at_1pct']:.2f}×")
    print(f"    P@5%:  {m['precision_at_5pct']:.4f}   Lift@5%:  {m['lift_at_5pct']:.2f}×")
    print(f"    P@10%: {m['precision_at_10pct']:.4f}   Lift@10%: {m['lift_at_10pct']:.2f}×")
    print(f"    R@1%:  {m['recall_at_1pct']:.4f}   R@5%:  {m['recall_at_5pct']:.4f}   "
          f"R@10%: {m['recall_at_10pct']:.4f}")


# ── Main diagnostic routine ────────────────────────────────────────────────────

def run_group_ab_diagnostic(
    region: Literal["south_america", "usa", "se_asia"],
    model_id: str,
    model_type: Literal["lgbm", "rf"],
    metrics_root: Path,
    data_root: Path,
    output_dir: Path,
    split_version: str = "main",
) -> Dict:
    """Run the Group A/B diagnostic for one region/model combination.

    Parameters
    ----------
    region        : one of south_america / usa / se_asia
    model_id      : e.g. model1, model2, model3
    model_type    : lgbm or rf
    metrics_root  : directory containing scored parquets
    data_root     : directory containing test_win5.parquet (under split_version/)
    output_dir    : where to write results
    split_version : sub-directory name (default "main")

    Returns
    -------
    dict  Results JSON structure (also saved to output_dir).
    """
    w = 74
    tag = f"{model_id}_{model_type}"
    print("=" * w)
    print(f"GROUP A/B DIAGNOSTIC  —  {region.upper()}  {tag.upper()}")
    print("=" * w)

    # ── Resolve paths ────────────────────────────────────────────────────────
    scored_path = _find_latest_scored_parquet(
        metrics_root, model_id, model_type, split_version
    )
    test_path = data_root / split_version / "test_win5.parquet"
    if not test_path.exists():
        # Some setups omit the split_version subdirectory
        test_path = data_root / "test_win5.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"test_win5.parquet not found under {data_root}")

    print(f"\n  Scored parquet : {scored_path}")
    print(f"  Test parquet   : {test_path}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading scored predictions …")
    scored = _load_scored(scored_path)

    print("\n[2/4] Building pixel designation map from test parquet …")
    desig_map = _build_designation_map(test_path)

    # ── Classify pixels ──────────────────────────────────────────────────────
    print("\n[3/4] Classifying pixels …")
    group_col = _classify_pixels(scored, desig_map)

    # Attach group column, drop unneeded columns, free memory
    scored["group"] = group_col
    del group_col
    gc.collect()

    # Extract numpy arrays once (avoids repeated pandas overhead)
    y_true  = scored["y_true"].to_numpy(dtype=np.int8)
    y_proba = scored["y_pred_proba"].to_numpy(dtype=np.float32)
    groups  = scored["group"].to_numpy(dtype=np.int8)

    n_total = len(y_true)
    n_pos_total = int(y_true.sum())

    # ── Compute metrics ──────────────────────────────────────────────────────
    print("\n[4/4] Computing metrics …")

    # Overall (all positives, same as standard test evaluation)
    overall = _metrics_for_group(y_true, y_proba, y_true.astype(bool))

    # Per-group
    group_results = {}
    for code, label in GROUP_LABELS.items():
        print(f"  Computing: {label}")
        mask = groups == code
        group_results[label] = _metrics_for_group(y_true, y_proba, mask)

    # ── Print summary ────────────────────────────────────────────────────────
    print()
    print("=" * w)
    print("RESULTS")
    print("=" * w)

    _print_group_metrics("OVERALL (all positives)", overall)
    for label, m in group_results.items():
        _print_group_metrics(label, m)

    # Thesis verdict
    b_metrics = group_results.get(GROUP_LABELS[GROUP_B], {})
    if b_metrics.get("n_positives", 0) > 0:
        roc_b = b_metrics.get("roc_auc", 0)
        lift_b = b_metrics.get("lift_at_1pct", 0)
        print()
        print("  " + "─" * 70)
        print(f"  THESIS VERDICT  |  Group B  AUC={roc_b:.4f}  Lift@1%={lift_b:.1f}×")
        if roc_b >= 0.75 and lift_b >= 15:
            print("  ✓ Group B AUC ≥ 0.75 AND Lift@1% ≥ 15×")
            print("    → Test performance reflects genuine pattern learning.")
            print("    → Strongest single OOS evidence for the thesis.")
        elif roc_b >= 0.60:
            print("  ~ Group B AUC ≥ 0.60 — moderate OOS signal.")
        else:
            print("  ✗ Weak Group B signal — soft leakage may explain much of test AUC.")
        print("  " + "─" * 70)

    print()
    print("  Classification note:")
    print("  D=2020–2022 pixels (Ambiguous group) cannot be fully resolved")
    print("  from test data alone.  On Euler, re-run with --extended to scan")
    print("  train parquets and classify the ambiguous group completely.")
    print()

    # ── Save outputs ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{tag}_win5_group_ab_{ts}"

    result = {
        "schema_version": "1.0.0",
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "region": region,
            "model_id": model_id,
            "model_type": model_type,
            "split_version": split_version,
            "scored_parquet": str(scored_path),
            "test_parquet": str(test_path),
        },
        "dataset": {
            "n_total_rows": n_total,
            "n_positive_rows": n_pos_total,
            "positive_rate_pct": round(n_pos_total / n_total * 100, 4),
        },
        "classification_notes": {
            "group_A_definite": "D=2018-2019 (transition_01=1 in test parquet)",
            "group_B_definite": "D=2023-2024 (first win5-positive year ≥ 2018)",
            "ambiguous": "D=2020-2022 (positive at all 3 test years; unresolvable from test alone)",
            "method": "test-parquet-only (no training data required)",
        },
        "overall": overall,
        "groups": {label: m for label, m in group_results.items()},
    }

    json_path = output_dir / f"{base}.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON results saved: {json_path}")

    return result


# ── Entry point helper ─────────────────────────────────────────────────────────

_REGION_MODELS = {
    "south_america": ("model1", "lgbm", "rf"),
    "usa":           ("model2", "lgbm", "rf"),
    "se_asia":       ("model3", "lgbm", "rf"),
}


def run_group_ab_main(
    region: Literal["south_america", "usa", "se_asia"],
    model_id: Optional[str] = None,
    models: str = "both",          # "lgbm", "rf", or "both"
    split_version: str = "main",
    repo_root: Optional[Path] = None,
) -> None:
    """Entry point used by the thin region-specific wrappers."""
    import os

    if repo_root is None:
        # Walk upward from this file until README.md is found
        candidate = Path(__file__).resolve().parent
        for _ in range(10):
            if (candidate / "README.md").exists():
                repo_root = candidate
                break
            parent = candidate.parent
            if parent == candidate:
                break
            candidate = parent
        if repo_root is None:
            raise RuntimeError("Repository root not found (no README.md). Set REPO_ROOT env var.")

    scratch = os.environ.get("SCRATCH")
    if scratch:
        base = Path(scratch)
        metrics_root = base / f"outputs/{region}/results/ml_models"
        data_root    = base / f"data/{region}/ml"
    else:
        metrics_root = repo_root / f"outputs/{region}/results/ml_models"
        data_root    = repo_root / f"data/{region}/ml"

    output_dir = metrics_root / split_version

    default_model_id, _, _ = _REGION_MODELS[region]
    if model_id is None:
        model_id = default_model_id

    run_models: list[str] = []
    if models in ("lgbm", "both"):
        run_models.append("lgbm")
    if models in ("rf", "both"):
        run_models.append("rf")

    for mt in run_models:
        try:
            run_group_ab_diagnostic(
                region=region,
                model_id=model_id,
                model_type=mt,
                metrics_root=metrics_root,
                data_root=data_root,
                output_dir=output_dir,
                split_version=split_version,
            )
        except FileNotFoundError as exc:
            print(f"\n  SKIP {mt.upper()}: {exc}\n")
        finally:
            gc.collect()


# ── CLI (used when run directly, not via region wrapper) ──────────────────────

def _parse_args() -> "argparse.Namespace":
    import argparse
    p = argparse.ArgumentParser(
        description="Group A/B leakage diagnostic for PA3030 test predictions"
    )
    p.add_argument(
        "--region",
        choices=["south_america", "usa", "se_asia", "all"],
        default="south_america",
        help="Region to process (default: south_america)",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Model ID override (e.g. model1). Auto-detected from region if omitted.",
    )
    p.add_argument(
        "--model",
        choices=["lgbm", "rf", "both"],
        default="both",
        help="Which algorithm(s) to run (default: both)",
    )
    p.add_argument(
        "--split-version",
        default="main",
        help="Split sub-directory (default: main)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    regions = (
        list(_REGION_MODELS.keys()) if args.region == "all" else [args.region]
    )
    for reg in regions:
        run_group_ab_main(
            region=reg,
            model_id=args.model_id,
            models=args.model,
            split_version=args.split_version,
        )


if __name__ == "__main__":
    main()


__all__ = ["run_group_ab_main", "run_group_ab_diagnostic"]
