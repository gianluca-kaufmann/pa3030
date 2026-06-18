#!/usr/bin/env python3
"""CE.3a — Spatial coherence diagnostic for Stage 2 expansion events.

For each of the 139 meaningful SA expansion events (≥200 positive pixels),
compute spatial coherence = fraction of positive pixels within 5 km of
another positive pixel. Cross-references CE.2 recall data to find the
coherence threshold that separates ecological campaigns from political
designations.

Runtime: ~20–40 min on Euler (reads 6 columns from 42 GB parquets).

Outputs
-------
outputs/south_america/results/ml_models/stage2_event_coherence_diagnostic.json
  Per-event coherence + CE.2 recall + recommended threshold.

Usage
-----
    sbatch slurm/south_america/stage2_event_coherence.slurm
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.country_raster import country_ids_for_rows, load_country_raster
from scripts.regions.shared.training.stage2_lgbm_core import (
    BATCH_SIZE,
    TARGET_COL,
    resolve_parquet_file,
)
from scripts.regions.shared.training.utils import get_repo_root

REGION = "south_america"
MODEL_PREFIX = "model1"
MIN_POSITIVE_PIXELS = 200
COHERENCE_RADIUS = 5  # pixels ≈ 5 km at 1 km/px resolution


# ---------------------------------------------------------------------------
# Coherence computation
# ---------------------------------------------------------------------------

def compute_coherence(rows: np.ndarray, cols: np.ndarray, radius: int = COHERENCE_RADIUS) -> float:
    """Fraction of positive pixels with at least one other positive pixel within `radius` pixels.

    O(n × radius²) — fast for events up to ~200K pixels.
    """
    if len(rows) == 0:
        return 0.0
    pos_set: set[tuple[int, int]] = set(zip(rows.astype(int).tolist(), cols.astype(int).tolist()))
    n_coherent = 0
    for r, c in pos_set:
        found = False
        for dr in range(-radius, radius + 1):
            if found:
                break
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in pos_set:
                    found = True
                    break
        if found:
            n_coherent += 1
    return n_coherent / len(pos_set)


# ---------------------------------------------------------------------------
# Collect positive pixel coordinates per event
# ---------------------------------------------------------------------------

def collect_positive_coords(
    parquet_sources: list[tuple[Path, tuple[int, int]]],
    region: str,
    meaningful: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Stream parquets and collect (rows, cols) arrays for each meaningful event.

    Only reads positive pixels (transition_01 > 0, WDPA_prev == 0).
    Returns dict: (country_id, year) → (rows_array, cols_array).
    """
    event_buf: dict[tuple[int, int], tuple[list, list]] = {k: ([], []) for k in meaningful}

    for panel_path, year_range in parquet_sources:
        schema_names = pq.ParquetFile(panel_path).schema_arrow.names
        has_country = "country_id" in schema_names
        raster = None if has_country else load_country_raster(region)

        cols_to_read = [TARGET_COL, "year", "WDPA_prev", "row", "col"]
        if has_country:
            cols_to_read.append("country_id")

        t0 = time.time()
        print(f"  Streaming {panel_path.name} ({year_range[0]}–{year_range[1]})...")
        pf = pq.ParquetFile(panel_path)
        for batch in pf.iter_batches(columns=cols_to_read, batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            if df.empty:
                continue
            df = df[
                df[TARGET_COL].notna()
                & (df["WDPA_prev"] == 0)
                & (df[TARGET_COL] > 0)
                & (df["year"] >= year_range[0])
                & (df["year"] <= year_range[1])
            ]
            if df.empty:
                continue
            if not has_country:
                df["country_id"] = country_ids_for_rows(df, raster)
            for (cid, yr), grp in df.groupby(["country_id", "year"]):
                key = (int(cid), int(yr))
                if key in event_buf:
                    event_buf[key][0].extend(grp["row"].tolist())
                    event_buf[key][1].extend(grp["col"].tolist())

        print(f"    done in {time.time()-t0:.1f}s")

    return {
        k: (np.array(v[0], dtype=np.int32), np.array(v[1], dtype=np.int32))
        for k, v in event_buf.items()
        if len(v[0]) > 0
    }


# ---------------------------------------------------------------------------
# Load CE.2 recall data
# ---------------------------------------------------------------------------

def load_ce2_recall(repo_root: Path) -> dict[tuple[int, int], dict]:
    """Load per-event recall from the most recent cross-event metrics JSON."""
    candidates = sorted(
        (repo_root / "outputs" / REGION / "results" / "ml_models").glob(
            f"{MODEL_PREFIX}_lgbm_stage2_cross_event_metrics_*.json"
        ),
        reverse=True,
    )
    if not candidates:
        print("  WARNING: No CE.2 metrics JSON found — recall data unavailable")
        return {}

    path = candidates[0]
    print(f"  Loading CE.2 recall from: {path.name}")
    with open(path) as f:
        data = json.load(f)

    recall_map: dict[tuple[int, int], dict] = {}
    for split_key, split_label in [("test_performance", "test"), ("val_performance", "val")]:
        for ev in data.get(split_key, {}).get("per_event", []):
            key = (int(ev["country_id"]), int(ev["year"]))
            recall_map[key] = {
                "recall_at_5pct": ev["recall_at_5pct"],
                "lift_at_1pct": ev["lift_at_1pct"],
                "n_pos": ev["n_pos"],
                "split": split_label,
            }
    return recall_map


def load_event_counts_from_ce2(repo_root: Path) -> dict[tuple[int, int], int]:
    """Parse event_counts from the CE.2 metrics JSON.

    Keys in the JSON look like '(1, 2001)' — we parse them back to int tuples.
    """
    candidates = sorted(
        (repo_root / "outputs" / REGION / "results" / "ml_models").glob(
            f"{MODEL_PREFIX}_lgbm_stage2_cross_event_metrics_*.json"
        ),
        reverse=True,
    )
    if not candidates:
        return {}
    with open(candidates[0]) as f:
        data = json.load(f)
    result: dict[tuple[int, int], int] = {}
    for raw_key, count in data.get("event_counts", {}).items():
        # raw_key looks like "(1, 2001)"
        parts = raw_key.strip("()").split(",")
        if len(parts) == 2:
            result[(int(parts[0].strip()), int(parts[1].strip()))] = int(count)
    return result


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

def find_coherence_threshold(
    per_event: list[dict],
    target_median_recall: float = 0.30,
) -> float:
    """Find lowest coherence threshold where events above it have median Recall@5% >= target.

    Returns the threshold (at 0.05 steps) that gives median Recall@5% >= target
    while retaining the most events. Returns 0.0 if no threshold works.
    """
    events_with_recall = [e for e in per_event if e.get("recall_at_5pct") is not None]
    if not events_with_recall:
        return 0.0

    best_threshold = 0.0
    best_n_events = len(events_with_recall)

    for threshold in np.arange(0.05, 1.0, 0.05):
        above = [e for e in events_with_recall if e["coherence"] >= threshold]
        if len(above) < 3:
            break
        median_recall = float(np.median([e["recall_at_5pct"] for e in above]))
        if median_recall >= target_median_recall:
            best_threshold = float(threshold)
            best_n_events = len(above)

    return round(best_threshold, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_coherence_diagnostic() -> None:
    repo_root = get_repo_root()
    out_dir = repo_root / f"outputs/{REGION}/results/ml_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_parquet_file(REGION, "train.parquet")
    earlystop_path = resolve_parquet_file(REGION, "earlystop.parquet")
    test_path = resolve_parquet_file(REGION, "test.parquet")

    parquet_sources: list[tuple[Path, tuple[int, int]]] = [
        (train_path,     (2001, 2013)),
        (earlystop_path, (2014, 2016)),
        (test_path,      (2017, 2024)),
    ]

    # Load event counts from CE.2 metrics JSON
    print("Loading event counts from CE.2 metrics JSON...")
    event_counts = load_event_counts_from_ce2(repo_root)
    if not event_counts:
        raise RuntimeError(
            "CE.2 metrics JSON not found. Run model1_LGBM_stage2_cross_event.py first."
        )
    meaningful: set[tuple[int, int]] = {k for k, v in event_counts.items() if v >= MIN_POSITIVE_PIXELS}
    print(f"Meaningful events (>={MIN_POSITIVE_PIXELS} pos): {len(meaningful)}")

    # Load CE.2 per-event recall
    print("\nLoading CE.2 recall data...")
    ce2_recall = load_ce2_recall(repo_root)
    print(f"  CE.2 recall data: {len(ce2_recall)} events (test + val splits)")

    # Collect positive pixel coordinates
    print("\nCollecting positive pixel coordinates from parquets...")
    t0 = time.time()
    coords = collect_positive_coords(parquet_sources, REGION, meaningful)
    print(f"Coordinates collected in {time.time()-t0:.1f}s  ({len(coords)} events)")

    # Compute coherence per event
    print(f"\nComputing spatial coherence (radius={COHERENCE_RADIUS} px ≈ 5 km)...")
    per_event: list[dict] = []
    t0 = time.time()
    for i, (key, (rows, cols)) in enumerate(sorted(coords.items())):
        cid, yr = key
        n_pos = len(rows)
        coherence = compute_coherence(rows, cols, COHERENCE_RADIUS)
        rec = ce2_recall.get(key, {})
        entry: dict = {
            "country_id": cid,
            "year": yr,
            "n_pos": n_pos,
            "coherence": round(float(coherence), 4),
            "recall_at_5pct": rec.get("recall_at_5pct"),
            "lift_at_1pct": rec.get("lift_at_1pct"),
            "in_ce2": key in ce2_recall,
            "ce2_split": rec.get("split", "train"),
        }
        per_event.append(entry)
        if (i + 1) % 10 == 0 or (i + 1) == len(coords):
            r5_str = f"{rec['recall_at_5pct']:.3f}" if rec.get("recall_at_5pct") is not None else "  n/a"
            print(f"  [{i+1:3d}/{len(coords)}]  cid={cid:2d} yr={yr}  "
                  f"n_pos={n_pos:>7,}  coh={coherence:.3f}  r@5%={r5_str}")

    print(f"Coherence computed in {time.time()-t0:.1f}s")

    # Sort by coherence descending for readability
    per_event.sort(key=lambda x: x["coherence"], reverse=True)

    # Find threshold
    threshold = find_coherence_threshold(per_event, target_median_recall=0.30)
    n_above = sum(1 for e in per_event if e["coherence"] >= threshold)
    n_with_recall = sum(1 for e in per_event if e["recall_at_5pct"] is not None)
    n_above_with_recall = sum(1 for e in per_event
                              if e["coherence"] >= threshold and e["recall_at_5pct"] is not None)

    print(f"\n{'='*60}")
    print(f"Coherence diagnostic results ({len(per_event)} events, {n_with_recall} with CE.2 recall):")
    print(f"  Recommended threshold: {threshold:.2f}")
    print(f"  Events above threshold: {n_above} / {len(per_event)}")
    print(f"  CE.2 events above threshold: {n_above_with_recall}")
    print()
    print("  Coherence distribution:")
    for t in np.arange(0.0, 1.05, 0.10):
        n = sum(1 for e in per_event if e["coherence"] >= t)
        with_recall = [e for e in per_event
                       if e["coherence"] >= t and e["recall_at_5pct"] is not None]
        med_r = float(np.median([e["recall_at_5pct"] for e in with_recall])) if with_recall else float("nan")
        print(f"    >= {t:.1f}: {n:3d} events | median Recall@5% = {med_r:.3f} (n={len(with_recall)})")
    print()
    print(f"  {'cid':>4} {'yr':>5} {'n_pos':>8} {'coh':>6} {'r@5%':>7} {'split':>5}")
    for e in per_event:
        r5 = f"{e['recall_at_5pct']:.3f}" if e["recall_at_5pct"] is not None else "  n/a"
        print(f"  {e['country_id']:>4} {e['year']:>5} {e['n_pos']:>8,} "
              f"{e['coherence']:>6.3f} {r5:>7} {e['ce2_split']:>5}")
    print(f"{'='*60}")

    result = {
        "metadata": {
            "region": REGION,
            "n_meaningful_events": len(meaningful),
            "n_events_with_coords": len(per_event),
            "coherence_radius_pixels": COHERENCE_RADIUS,
            "recommended_threshold": threshold,
            "n_events_above_threshold": n_above,
        },
        "per_event": per_event,
    }
    out_path = out_dir / "stage2_event_coherence_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nNext: set STAGE2_COHERENCE_THRESHOLD={threshold:.2f} in CE.3b SLURM script.")


if __name__ == "__main__":
    run_coherence_diagnostic()
