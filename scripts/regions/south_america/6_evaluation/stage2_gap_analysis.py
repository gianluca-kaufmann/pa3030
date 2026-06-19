#!/usr/bin/env python3
"""E1 (Track A): Representation gap analysis — model score vs. biodiversity priority.

Tests the core Nature Sustainability finding: PA designation follows cost-minimisation
logic that systematically under-protects high-biodiversity land.

Two complementary analyses:
  A. Direct:  within each expansion event, compare mean GSN_b2 of actually-designated
              pixels against mean of all unprotected candidate pixels.
  B. Model:   score all test pixels with the temporal Stage 2 model; bin by score
              decile; compute mean GSN_b2 per bin + Spearman correlation.

Run on Euler against the full SA test split (2017-2024).

Usage (Euler):
  python scripts/regions/south_america/6_evaluation/stage2_gap_analysis.py \
      --model  data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl \
      --test   $SCRATCH/data/south_america/ml/main/test.parquet \
      --out    outputs/south_america/results/gap_analysis/

  # Optional — include earlystop (2014-2016) and train (2001-2013) parquets to
  # maximise the number of expansion events for Part A:
      --extra  $SCRATCH/data/south_america/ml/main/earlystop.parquet \
               $SCRATCH/data/south_america/ml/main/train.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BIODIVERSITY_COL = "GSN_b2"
LABEL_COL        = "transition_01"
N_BINS           = 10          # deciles
BATCH_SIZE       = 500_000
MIN_EVENT_POS    = 50          # minimum positives for an event to enter Part A


def _load_model(model_path: Path) -> tuple:
    with open(model_path, "rb") as f:
        pkg = pickle.load(f)
    model       = pkg["model"]
    feature_cols = pkg["feature_cols"]
    print(f"Model loaded: {model_path.name}  "
          f"features={len(feature_cols)}  best_iter={getattr(model, 'best_iteration', 'N/A')}")
    return model, feature_cols


def _score_parquet(parquet_path: Path, model, feature_cols: list[str],
                   extra_cols: list[str]) -> pd.DataFrame:
    """Read parquet in batches, score each batch, return slim DataFrame."""
    need = list(dict.fromkeys(feature_cols + extra_cols))
    pf   = pq.ParquetFile(parquet_path)

    # Only load columns that exist in the parquet schema
    schema_names = set(pf.schema_arrow.names)
    missing_feat = [c for c in feature_cols if c not in schema_names]
    if missing_feat:
        raise ValueError(
            f"Model features missing from {parquet_path.name}: {missing_feat}\n"
            "Ensure the parquet was built with the same feature set as the model."
        )
    load_cols = [c for c in need if c in schema_names]

    chunks: list[pd.DataFrame] = []
    n_rows = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=load_cols):
        df = batch.to_pandas()
        X  = df[feature_cols].to_numpy(dtype=np.float32)
        df["score"] = model.predict(X, num_iteration=getattr(model, "best_iteration", None) or None)
        keep = [c for c in extra_cols + ["score"] if c in df.columns]
        chunks.append(df[keep].copy())
        n_rows += len(df)
        if n_rows % 5_000_000 < BATCH_SIZE:
            print(f"  scored {n_rows:,} rows…")

    result = pd.concat(chunks, ignore_index=True)
    print(f"  {parquet_path.name}: {len(result):,} rows scored")
    return result


def _part_a_direct(df: pd.DataFrame) -> dict:
    """Direct gap: per expansion event, compare GSN_b2 of positives vs all candidates."""
    if BIODIVERSITY_COL not in df.columns or LABEL_COL not in df.columns:
        return {"error": f"Missing columns: {BIODIVERSITY_COL} or {LABEL_COL}"}

    events = df.groupby(["country_id", "year"])
    rows: list[dict] = []
    for (cid, yr), g in events:
        n_pos = int(g[LABEL_COL].sum())
        if n_pos < MIN_EVENT_POS:
            continue
        pos_mask = g[LABEL_COL] == 1
        gsn_pos  = float(g.loc[pos_mask, BIODIVERSITY_COL].mean())
        gsn_all  = float(g[BIODIVERSITY_COL].mean())
        rows.append({
            "country_id": int(cid),
            "year":       int(yr),
            "n_pos":      n_pos,
            "n_all":      len(g),
            "gsn_pos_mean":  round(gsn_pos, 5),
            "gsn_all_mean":  round(gsn_all, 5),
            "delta":         round(gsn_pos - gsn_all, 5),
            "gap_confirmed": gsn_pos < gsn_all,
        })

    if not rows:
        return {"error": "No qualifying events (min_event_pos too high or no positives)"}

    deltas    = [r["delta"] for r in rows]
    n_neg     = sum(1 for d in deltas if d < 0)
    n_events  = len(rows)

    # Overall (pooled across all expansion pixels)
    pos_all = df[df[LABEL_COL] == 1][BIODIVERSITY_COL]
    neg_all = df[df[LABEL_COL] == 0][BIODIVERSITY_COL]

    return {
        "n_events":                  n_events,
        "n_events_gap_confirmed":    n_neg,
        "pct_events_gap_confirmed":  round(100 * n_neg / n_events, 1),
        "mean_delta_gsn":            round(float(np.mean(deltas)), 5),
        "median_delta_gsn":          round(float(np.median(deltas)), 5),
        "pooled_gsn_positives_mean": round(float(pos_all.mean()), 5),
        "pooled_gsn_negatives_mean": round(float(neg_all.mean()), 5),
        "pooled_gsn_ratio":          round(float(pos_all.mean() / neg_all.mean()), 4)
                                     if neg_all.mean() > 0 else None,
        "per_event": rows,
    }


def _part_b_model(df: pd.DataFrame) -> dict:
    """Model-score gap: bin by score decile, compute mean GSN_b2 per bin."""
    if BIODIVERSITY_COL not in df.columns or "score" not in df.columns:
        return {"error": "Missing score or GSN_b2 column"}

    df = df.dropna(subset=["score", BIODIVERSITY_COL])

    # Decile bins on score
    df["score_bin"] = pd.qcut(df["score"], N_BINS, labels=False, duplicates="drop")

    by_bin = (
        df.groupby("score_bin")
        .agg(
            n_pixels      =("score", "count"),
            score_mean    =("score", "mean"),
            score_p50     =("score", "median"),
            gsn_mean      =(BIODIVERSITY_COL, "mean"),
            gsn_median    =(BIODIVERSITY_COL, "median"),
            n_positives   =(LABEL_COL, "sum") if LABEL_COL in df.columns else ("score", lambda x: 0),
        )
        .reset_index()
    )

    spear_r, spear_p = spearmanr(df["score"], df[BIODIVERSITY_COL])

    # High-score pixels: does the model send us to low-biodiversity land?
    top10_pct = df[df["score_bin"] == df["score_bin"].max()][BIODIVERSITY_COL].mean()
    bot10_pct = df[df["score_bin"] == df["score_bin"].min()][BIODIVERSITY_COL].mean()

    return {
        "spearman_score_vs_gsn_r": round(float(spear_r), 5),
        "spearman_score_vs_gsn_p": float(spear_p),
        "gap_confirmed_by_model":  spear_r < 0,
        "top_decile_gsn_mean":     round(float(top10_pct), 5),
        "bottom_decile_gsn_mean":  round(float(bot10_pct), 5),
        "top_vs_bottom_ratio":     round(float(top10_pct / bot10_pct), 4)
                                   if bot10_pct > 0 else None,
        "note": (
            "GSN_b2 IS a model feature (SHAP rank #2). "
            "The model learned low-GSN_b2 predicts designation (cost-minimisation). "
            "Negative Spearman confirms model score anti-correlates with biodiversity."
        ),
        "per_bin": [
            {
                "bin":          int(r["score_bin"]),
                "n_pixels":     int(r["n_pixels"]),
                "score_mean":   round(float(r["score_mean"]), 5),
                "gsn_mean":     round(float(r["gsn_mean"]), 5),
                "gsn_median":   round(float(r["gsn_median"]), 5),
                "n_positives":  int(r["n_positives"]),
            }
            for _, r in by_bin.iterrows()
        ],
    }


def _cost_feature_summary(df: pd.DataFrame) -> dict:
    """Cost-minimisation signature: compare key features for designated vs. not."""
    cost_feats = [
        "dist_wdpa", "dist_road", "economic_value_lag1",
        "GPW", "GPW_smooth64", "NDVI_smooth64",
    ]
    out: dict = {}
    if LABEL_COL not in df.columns:
        return out
    for feat in cost_feats:
        if feat not in df.columns:
            continue
        pos_mean = float(df.loc[df[LABEL_COL] == 1, feat].mean())
        neg_mean = float(df.loc[df[LABEL_COL] == 0, feat].mean())
        out[feat] = {
            "designated_mean":     round(pos_mean, 4),
            "non_designated_mean": round(neg_mean, 4),
            "ratio":               round(pos_mean / neg_mean, 4) if neg_mean != 0 else None,
        }
    return out


def _save_plot(part_b: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = part_b.get("per_bin", [])
        if not bins:
            return
        x      = [b["bin"] for b in bins]
        y      = [b["gsn_mean"] for b in bins]
        labels = [f"{b['score_mean']:.3f}" for b in bins]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, y, color="#2c7bb6", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Model score decile (0=lowest, 9=highest designation probability)")
        ax.set_ylabel("Mean GSN_b2 (biodiversity priority score)")
        ax.set_title(
            f"Representation gap: model score vs. biodiversity priority\n"
            f"Spearman r = {part_b['spearman_score_vs_gsn_r']:.3f}  "
            f"({'negative → gap confirmed' if part_b['spearman_score_vs_gsn_r'] < 0 else 'positive — no gap'})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "score_vs_gsn_b2.png", dpi=150)
        plt.close(fig)
        print(f"Plot: {out_dir / 'score_vs_gsn_b2.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot")


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 gap analysis")
    parser.add_argument("--model",  required=True, help="Path to stage2 model .pkl")
    parser.add_argument("--test",   required=True, help="Test parquet (primary, e.g. 2017-2024)")
    parser.add_argument("--extra",  nargs="*",     help="Additional parquets for Part A direct analysis")
    parser.add_argument("--out",    default="outputs/south_america/results/gap_analysis/",
                        help="Output directory")
    args = parser.parse_args()

    model_path = Path(args.model)
    test_path  = Path(args.test)
    out_dir    = _ROOT / args.out

    model, feature_cols = _load_model(model_path)

    extra_cols = ["country_id", "year", LABEL_COL, BIODIVERSITY_COL,
                  "dist_wdpa", "dist_road", "economic_value_lag1",
                  "GPW", "GPW_smooth64", "NDVI_smooth64"]

    # ── Score primary test parquet ─────────────────────────────────────────────
    print(f"\n[Part B] Scoring test parquet: {test_path}")
    df_test = _score_parquet(test_path, model, feature_cols, extra_cols)

    # ── Optional extra parquets for Part A ───────────────────────────────────
    extra_dfs = [df_test]
    if args.extra:
        for ep in args.extra:
            ep = Path(ep)
            if ep.exists():
                print(f"[Part A] Loading extra parquet: {ep.name}")
                df_extra = _score_parquet(ep, model, feature_cols, extra_cols)
                extra_dfs.append(df_extra)
            else:
                print(f"[Part A] Extra parquet not found: {ep} — skipping")
    df_all = pd.concat(extra_dfs, ignore_index=True) if len(extra_dfs) > 1 else df_test

    # ── Run analyses ──────────────────────────────────────────────────────────
    print("\n[Part A] Direct gap analysis…")
    part_a = _part_a_direct(df_all)

    print("\n[Part B] Model-score gap analysis…")
    part_b = _part_b_model(df_test)

    print("\n[Cost] Cost-feature signature…")
    cost = _cost_feature_summary(df_all)

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "experiment":   "E1_representation_gap",
        "model":        model_path.name,
        "test_parquet": test_path.name,
        "n_test_rows":  len(df_test),
        "n_all_rows":   len(df_all),
        "part_a_direct_gap":    part_a,
        "part_b_model_gap":     part_b,
        "cost_feature_summary": cost,
        "headline": {
            "pct_events_gap_confirmed": part_a.get("pct_events_gap_confirmed"),
            "mean_delta_gsn":           part_a.get("mean_delta_gsn"),
            "spearman_r_score_gsn":     part_b.get("spearman_score_vs_gsn_r"),
            "top_decile_gsn_mean":      part_b.get("top_decile_gsn_mean"),
            "bottom_decile_gsn_mean":   part_b.get("bottom_decile_gsn_mean"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "gap_analysis_results.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))

    _save_plot(part_b, out_dir)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("E1: Representation Gap — Results")
    print("=" * 65)
    print(f"\n  Part A — Direct (within-event comparison):")
    if "error" not in part_a:
        print(f"    Events analysed:         {part_a['n_events']}")
        print(f"    Events gap confirmed:    {part_a['n_events_gap_confirmed']} / {part_a['n_events']}"
              f" ({part_a['pct_events_gap_confirmed']:.0f}%)")
        print(f"    Mean Δ GSN_b2:           {part_a['mean_delta_gsn']:+.4f}  (pos - all; negative = gap)")
        print(f"    Pooled GSN_b2 pos:       {part_a['pooled_gsn_positives_mean']:.4f}")
        print(f"    Pooled GSN_b2 all:       {part_a['pooled_gsn_negatives_mean']:.4f}")
        print(f"    Ratio (pos/all):         {part_a.get('pooled_gsn_ratio'):.3f}")
    else:
        print(f"    {part_a['error']}")

    print(f"\n  Part B — Model-score gap:")
    if "error" not in part_b:
        print(f"    Spearman(score, GSN_b2): r={part_b['spearman_score_vs_gsn_r']:.4f}"
              f"  p={part_b['spearman_score_vs_gsn_p']:.2e}")
        print(f"    Gap confirmed:           {'YES' if part_b['gap_confirmed_by_model'] else 'NO'}")
        print(f"    Top-decile GSN_b2 mean:  {part_b['top_decile_gsn_mean']:.4f}")
        print(f"    Bot-decile GSN_b2 mean:  {part_b['bottom_decile_gsn_mean']:.4f}")
        print(f"    Score-bin breakdown:")
        for b in part_b["per_bin"]:
            print(f"      bin {b['bin']}: score≈{b['score_mean']:.3f}  GSN_b2={b['gsn_mean']:.4f}"
                  f"  n={b['n_pixels']:,}  pos={b['n_positives']}")
    else:
        print(f"    {part_b['error']}")

    print(f"\n  Saved: {out_json}")


if __name__ == "__main__":
    main()
