#!/usr/bin/env python3
"""E7 (Track A, A2): KBA representation gap analysis.

The Nature headline: of the model's top-5% highest-scored pixels, what fraction
fall inside Key Biodiversity Areas (KBAs) vs the 15.1% baseline expected by chance?

KBA is NOT a model feature — it is an independent conservation benchmark.
If the top-5% has fewer KBAs than the baseline → governments protect non-KBA land
→ 30×30 will miss biodiversity priorities under business-as-usual.

Two analyses:
  A. Direct: within each expansion event, what fraction of actually-designated
     pixels fall inside KBAs vs all unprotected candidate pixels?
  B. Model: score all pixels with the temporal model; bin by score decile;
     compute mean is_kba per bin + Spearman correlation + KBA Recall@5%.

Usage:
  # Local (mini_sample_v2, all years):
  python scripts/regions/south_america/6_evaluation/kba_gap_analysis.py \\
      --model data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl \\
      --data  data/south_america/ml/mini_sample_v2.parquet \\
      --kba   data/south_america/ready/KBA/kba_sa.tif

  # Euler (full test parquet):
  python scripts/regions/south_america/6_evaluation/kba_gap_analysis.py \\
      --model data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl \\
      --data  $SCRATCH/data/south_america/ml/main/test.parquet \\
      --kba   $SCRATCH/data/south_america/ready/KBA/kba_sa.tif \\
      --test-years-only
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TEST_YEAR_MIN   = 2017
TEST_YEAR_MAX   = 2024
N_BINS          = 10
MIN_EVENT_POS   = 50
BATCH_SIZE      = 500_000
LABEL_COL       = "transition_01"


# ---------------------------------------------------------------------------
# Raster sampling
# ---------------------------------------------------------------------------

def _load_kba_band(kba_tif: Path) -> np.ndarray:
    """Load is_kba (band 1) into memory as float32 array."""
    with rasterio.open(kba_tif) as src:
        band = src.read(1).astype(np.float32)  # 1.0 inside KBA, 0.0 outside, NaN=ocean
    print(f"KBA raster: {band.shape}  sum(is_kba)={np.nansum(band):.0f}")
    return band


def _sample_kba(is_kba: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return is_kba value for each (row, col). Out-of-bounds → NaN."""
    h, w = is_kba.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.full(len(rows), np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = is_kba[rows[valid], cols[valid]]
    return out


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def _load_model(model_path: Path) -> tuple:
    with open(model_path, "rb") as f:
        pkg = pickle.load(f)
    model = pkg["model"]
    feature_cols = pkg["feature_cols"]
    print(f"Model: {model_path.name}  features={len(feature_cols)}  "
          f"best_iter={getattr(model, 'best_iteration', 'N/A')}")
    return model, feature_cols


def _score_and_sample(
    data_path: Path,
    model,
    feature_cols: list[str],
    is_kba: np.ndarray,
    test_years_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Score every (WDPA_prev==0) pixel.
    Returns: scores, labels, is_kba_vals, country_ids, years
    """
    pf = pq.ParquetFile(data_path)
    schema_names = set(pf.schema_arrow.names)

    missing = [c for c in feature_cols if c not in schema_names]
    if missing:
        sys.exit(f"Model features missing from {data_path.name}: {missing[:5]}...")

    need = list(dict.fromkeys(
        feature_cols + ["transition_01", "WDPA_prev", "row", "col", "country_id", "year"]
    ))
    load_cols = [c for c in need if c in schema_names]

    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    kba_chunks:   list[np.ndarray] = []
    cid_chunks:   list[np.ndarray] = []
    yr_chunks:    list[np.ndarray] = []

    n_total = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=load_cols):
        df = batch.to_pandas()
        df = df[df["WDPA_prev"] == 0]
        if df.empty:
            continue
        if test_years_only:
            df = df[(df["year"] >= TEST_YEAR_MIN) & (df["year"] <= TEST_YEAR_MAX)]
        if df.empty:
            continue

        X = df[feature_cols].to_numpy(dtype=np.float32)
        scores = model.predict(X, num_iteration=getattr(model, "best_iteration", None) or None)

        rows_arr = df["row"].to_numpy(dtype=np.int64)
        cols_arr = df["col"].to_numpy(dtype=np.int64)
        kba_vals = _sample_kba(is_kba, rows_arr, cols_arr)

        score_chunks.append(scores.astype(np.float32))
        label_chunks.append(df[LABEL_COL].to_numpy(dtype=np.float32))
        kba_chunks.append(kba_vals)
        cid_chunks.append(df["country_id"].to_numpy(dtype=np.int32))
        yr_chunks.append(df["year"].to_numpy(dtype=np.int32))
        n_total += len(df)

        if n_total % 2_000_000 < BATCH_SIZE:
            print(f"  scored {n_total:,} rows…")

    print(f"  Total scored: {n_total:,}")
    return (
        np.concatenate(score_chunks),
        np.concatenate(label_chunks),
        np.concatenate(kba_chunks),
        np.concatenate(cid_chunks),
        np.concatenate(yr_chunks),
    )


# ---------------------------------------------------------------------------
# Analysis: Part A — direct within-event comparison
# ---------------------------------------------------------------------------

def _part_a_direct(
    labels: np.ndarray,
    kba_vals: np.ndarray,
    country_ids: np.ndarray,
    years: np.ndarray,
) -> dict:
    """For each expansion event: KBA fraction among designated vs all candidates."""
    import pandas as pd
    df = pd.DataFrame({
        "label":      labels,
        "is_kba":     kba_vals,
        "country_id": country_ids,
        "year":       years,
    })
    df = df.dropna(subset=["is_kba"])

    rows: list[dict] = []
    for (cid, yr), g in df.groupby(["country_id", "year"]):
        n_pos = int((g["label"] > 0).sum())
        if n_pos < MIN_EVENT_POS:
            continue
        pos_mask = g["label"] > 0
        kba_pos  = float(g.loc[pos_mask, "is_kba"].mean())
        kba_all  = float(g["is_kba"].mean())
        rows.append({
            "country_id":    int(cid),
            "year":          int(yr),
            "n_pos":         n_pos,
            "n_all":         len(g),
            "kba_frac_designated": round(kba_pos, 5),
            "kba_frac_all":        round(kba_all, 5),
            "delta":               round(kba_pos - kba_all, 5),
            "gap_confirmed":       kba_pos < kba_all,
        })

    if not rows:
        return {"error": "No qualifying events"}

    n_gap = sum(1 for r in rows if r["gap_confirmed"])
    pooled_pos = df[df["label"] > 0]["is_kba"].mean()
    pooled_all = df["is_kba"].mean()

    return {
        "n_events": len(rows),
        "n_events_gap_confirmed": n_gap,
        "pct_events_gap_confirmed": round(100 * n_gap / len(rows), 1),
        "pooled_kba_frac_designated": round(float(pooled_pos), 5),
        "pooled_kba_frac_all": round(float(pooled_all), 5),
        "pooled_ratio": round(float(pooled_pos / pooled_all), 4) if pooled_all > 0 else None,
        "per_event": rows,
    }


# ---------------------------------------------------------------------------
# Analysis: Part B — model-score gap
# ---------------------------------------------------------------------------

def _part_b_model(
    scores: np.ndarray,
    labels: np.ndarray,
    kba_vals: np.ndarray,
) -> dict:
    """Bin pixels by score decile; compute mean is_kba per bin."""
    import pandas as pd
    df = pd.DataFrame({"score": scores, "label": labels, "is_kba": kba_vals})
    df = df.dropna(subset=["is_kba"])

    # Baseline KBA rate (all unprotected pixels)
    baseline_kba = float(df["is_kba"].mean())
    n_total      = len(df)

    # Top-5% by model score
    k5 = max(1, math.ceil(0.05 * n_total))
    top5_idx = np.argsort(df["score"].to_numpy())[::-1][:k5]
    top5_kba  = float(df["is_kba"].iloc[top5_idx].mean())
    top5_n_kba = int(df["is_kba"].iloc[top5_idx].sum())

    # KBA Recall@5% = how many KBA pixels are in the top-5%?
    n_kba_total = int(df["is_kba"].sum())
    kba_recall5 = top5_n_kba / max(n_kba_total, 1)
    kba_recall5_random = 0.05  # expected by chance

    # Score–KBA Spearman
    spear_r, spear_p = spearmanr(df["score"], df["is_kba"])

    # Decile bins
    df["score_bin"] = df["score"].rank(pct=True).mul(N_BINS).clip(upper=N_BINS).astype(int) - 1
    df["score_bin"] = df["score_bin"].clip(lower=0, upper=N_BINS - 1)
    by_bin = (
        df.groupby("score_bin")
        .agg(
            n_pixels   =("score",    "count"),
            score_mean =("score",    "mean"),
            kba_mean   =("is_kba",   "mean"),
            n_pos      =(LABEL_COL if LABEL_COL in df.columns else "label", "sum"),
        )
        .reset_index()
    )

    return {
        "baseline_kba_rate": round(baseline_kba, 5),
        "top5pct_kba_rate":  round(top5_kba, 5),
        "top5pct_vs_baseline_ratio": round(top5_kba / baseline_kba, 4) if baseline_kba > 0 else None,
        "kba_recall_at_5pct":        round(kba_recall5, 5),
        "kba_recall_at_5pct_random": kba_recall5_random,
        "kba_recall_lift":  round(kba_recall5 / kba_recall5_random, 3),
        "n_kba_pixels_total": n_kba_total,
        "n_kba_pixels_in_top5pct": top5_n_kba,
        "gap_confirmed": top5_kba < baseline_kba,
        "spearman_score_vs_kba_r": round(float(spear_r), 5),
        "spearman_score_vs_kba_p": float(spear_p),
        "interpretation": (
            "top5pct_vs_baseline_ratio < 1.0 → model steers toward non-KBA land → gap confirmed. "
            "kba_recall_lift < 1.0 → model predicts fewer KBA pixels than random → gap confirmed."
        ),
        "per_bin": [
            {
                "bin":        int(r["score_bin"]),
                "n_pixels":   int(r["n_pixels"]),
                "score_mean": round(float(r["score_mean"]), 5),
                "kba_mean":   round(float(r["kba_mean"]), 5),
                "n_pos":      int(r["n_pos"]),
            }
            for _, r in by_bin.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _save_plot(part_b: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins    = part_b.get("per_bin", [])
        baseline = part_b["baseline_kba_rate"]
        if not bins:
            return

        x       = [b["bin"] for b in bins]
        y       = [b["kba_mean"] for b in bins]
        colors  = ["#d73027" if yi < baseline else "#4575b4" for yi in y]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2,
                   label=f"Baseline (all pixels) = {baseline:.3f}")
        ax.set_xlabel("Model score decile (0=lowest, 9=highest designation probability)")
        ax.set_ylabel("Fraction of pixels inside a KBA")
        ax.set_title(
            "KBA representation gap: model score vs Key Biodiversity Areas\n"
            f"Top-5% KBA rate={part_b['top5pct_kba_rate']:.3f}  "
            f"Baseline={baseline:.3f}  "
            f"Ratio={part_b['top5pct_vs_baseline_ratio']:.3f}  "
            f"(red = below baseline)"
        )
        ax.legend()
        ax.set_xticks(x)
        fig.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "kba_gap_score_deciles.png", dpi=150)
        plt.close(fig)
        print(f"Plot: {out_dir / 'kba_gap_score_deciles.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E7 KBA gap analysis (A2)")
    parser.add_argument("--model",          required=True, help="Stage 2 model .pkl path")
    parser.add_argument("--data",           required=True, help="Parquet to score (mini_sample_v2 or test.parquet)")
    parser.add_argument("--kba",            required=True, help="kba_sa.tif path")
    parser.add_argument("--out", default="outputs/south_america/results/gap_analysis/",
                        help="Output directory")
    parser.add_argument("--test-years-only", action="store_true",
                        help="Restrict Part B scoring to 2017-2024 (temporal holdout)")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path  = Path(args.data)
    kba_path   = Path(args.kba)
    out_dir    = _ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not model_path.is_absolute():
        model_path = _ROOT / model_path
    if not data_path.is_absolute():
        data_path = _ROOT / data_path
    if not kba_path.is_absolute():
        kba_path = _ROOT / kba_path

    for p, name in [(model_path, "model"), (data_path, "data"), (kba_path, "kba tif")]:
        if not p.exists():
            sys.exit(f"{name} not found: {p}")

    model, feature_cols = _load_model(model_path)

    print(f"\nLoading KBA raster...")
    is_kba = _load_kba_band(kba_path)

    print(f"\nScoring pixels from: {data_path.name}  (test_years_only={args.test_years_only})")
    scores, labels, kba_vals, country_ids, years = _score_and_sample(
        data_path, model, feature_cols, is_kba, args.test_years_only
    )

    print(f"\n[Part A] Direct within-event KBA comparison...")
    part_a = _part_a_direct(labels, kba_vals, country_ids, years)

    print(f"\n[Part B] Model-score KBA gap (test pixels)...")
    test_mask = (years >= TEST_YEAR_MIN) & (years <= TEST_YEAR_MAX)
    part_b = _part_b_model(
        scores[test_mask], labels[test_mask], kba_vals[test_mask]
    )

    result = {
        "experiment":         "E7_kba_representation_gap",
        "model":              model_path.name,
        "data":               data_path.name,
        "test_years_only":    args.test_years_only,
        "n_pixels_scored":    int(len(scores)),
        "n_test_pixels":      int(test_mask.sum()),
        "part_a_direct_gap":  part_a,
        "part_b_model_gap":   part_b,
        "headline": {
            "pct_events_kba_gap_confirmed":  part_a.get("pct_events_kba_gap_confirmed"),
            "pooled_kba_ratio_designated_vs_all": part_a.get("pooled_ratio"),
            "top5pct_kba_rate":              part_b.get("top5pct_kba_rate"),
            "baseline_kba_rate":             part_b.get("baseline_kba_rate"),
            "top5pct_vs_baseline_ratio":     part_b.get("top5pct_vs_baseline_ratio"),
            "kba_recall_at_5pct":            part_b.get("kba_recall_at_5pct"),
            "kba_recall_lift":               part_b.get("kba_recall_lift"),
            "gap_confirmed_by_model":        part_b.get("gap_confirmed"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "kba_gap_analysis.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))

    _save_plot(part_b, out_dir)

    print(f"\n{'='*65}")
    print("E7: KBA Representation Gap — Results")
    print("=" * 65)
    print(f"\n  Part A — Direct (within-event):")
    if "error" not in part_a:
        print(f"    Events analysed:             {part_a['n_events']}")
        print(f"    Events KBA gap confirmed:    {part_a['n_events_gap_confirmed']}"
              f" / {part_a['n_events']} ({part_a['pct_events_gap_confirmed']:.0f}%)")
        print(f"    Pooled KBA frac designated:  {part_a['pooled_kba_frac_designated']:.4f}")
        print(f"    Pooled KBA frac all:         {part_a['pooled_kba_frac_all']:.4f}")
        print(f"    Ratio (pos/all):             {part_a.get('pooled_ratio'):.3f}"
              f"  ({'< 1 = gap confirmed' if part_a.get('pooled_ratio', 1) < 1 else '>= 1 = no gap'})")
    else:
        print(f"    {part_a['error']}")

    print(f"\n  Part B — Model-score KBA gap:")
    if "gap_confirmed" in part_b:
        print(f"    Baseline KBA rate (all):     {part_b['baseline_kba_rate']:.4f}")
        print(f"    Top-5% KBA rate:             {part_b['top5pct_kba_rate']:.4f}")
        print(f"    Top-5% / baseline ratio:     {part_b['top5pct_vs_baseline_ratio']:.4f}"
              f"  ({'< 1 = gap' if part_b['gap_confirmed'] else '>= 1 = no gap'})")
        print(f"    KBA Recall@5%:               {part_b['kba_recall_at_5pct']*100:.2f}%"
              f"  (random baseline = 5.00%)")
        print(f"    KBA Recall lift:             {part_b['kba_recall_lift']:.3f}×"
              f"  ({'< 1 = avoids KBAs' if part_b['kba_recall_lift'] < 1 else '>= 1 = finds KBAs'})")
        print(f"    Spearman(score, is_kba):     r={part_b['spearman_score_vs_kba_r']:.4f}"
              f"  p={part_b['spearman_score_vs_kba_p']:.2e}")
        print(f"    Gap confirmed:               {'YES' if part_b['gap_confirmed'] else 'NO'}")
        print()
        print(f"    Score decile breakdown (bin 9 = highest model score):")
        for b in part_b["per_bin"]:
            bar = "█" * int(b["kba_mean"] * 100)
            print(f"      bin {b['bin']}: KBA={b['kba_mean']:.4f}  {bar}")

    print(f"\n  Saved: {out_json}")


if __name__ == "__main__":
    main()
