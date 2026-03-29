#!/usr/bin/env python3
"""Stage 2: Calibrated forward inference on 2024 features.

Loads the deployment model artifact ({model_prefix}_{model_type}_deployment_*.pkl)
plus its Platt / isotonic calibrators and runs batched inference on the 2024
feature set produced by features_core.py.

Output:
    outputs/{region}/results/forward/{model_type}/forward_scored_2024.parquet
    (repo or $SCRATCH/outputs/... when SCRATCH is set — see resolve_forward_dir)
    Columns: row, col, x, y, y_pred_proba_raw, y_pred_proba_calibrated

Model-type support:
  lgbm  — finds {model_prefix}_lgbm_deployment_*.pkl,
           calls model.predict(X, num_iteration=num_boost_round)
  rf    — finds {model_prefix}_rf_deployment_*.pkl,
           calls model.predict_proba(X)[:, 1]  (no num_boost_round)
"""

from __future__ import annotations

import gc
import os
import pickle
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

from scripts.regions.shared.forward.config import (  # noqa: E402
    DATA_SUBDIR,
    MODEL_PREFIX,
    OUTPUTS_SUBDIR,
    forward_dir_search_paths,
    get_repo_root,
    ml_models_dir_search_paths,
    resolve_forward_dir,
)
from scripts.regions.shared.training.utils import (  # noqa: E402
    WandbRunLogger,
    get_repo_root as _get_repo_root,
    report_memory_usage,
)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "500_000"))


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_deployment_artifact(
    repo_root: Path, data_subdir: str, model_prefix: str, model_type: str
) -> Path:
    """Find the most-recent {model_prefix}_{model_type}_deployment_*.pkl (scratch, then repo)."""
    pattern = f"{model_prefix}_{model_type}_deployment_*.pkl"
    for model_dir in ml_models_dir_search_paths(repo_root, data_subdir):
        candidates = sorted(model_dir.glob(pattern), reverse=True)
        if not candidates:
            continue
        chosen = candidates[0]
        if len(candidates) > 1:
            print(
                f"  Found {len(candidates)} deployment artifacts in {model_dir}; "
                f"using most recent: {chosen.name}",
            )
        return chosen
    checked = "\n".join(f"  {d}" for d in ml_models_dir_search_paths(repo_root, data_subdir))
    raise FileNotFoundError(
        f"No {pattern} found. Checked:\n{checked}\n"
        f"Run the {model_prefix}_{model_type}_deployment.py training script first.",
    )


def resolve_features_parquet(repo_root: Path, outputs_subdir: str) -> Path:
    """Locate forward_features_2024.parquet (scratch forward dir, then repo)."""
    for fd in forward_dir_search_paths(repo_root, outputs_subdir):
        cand = fd / "forward_features_2024.parquet"
        if cand.exists():
            return cand
    checked = "\n".join(str(d) for d in forward_dir_search_paths(repo_root, outputs_subdir))
    raise FileNotFoundError(
        f"forward_features_2024.parquet not found. Checked:\n{checked}\n"
        "Run features_core (2_forward_features.py) first.",
    )


# ── Calibration helpers ───────────────────────────────────────────────────────

def apply_platt(platt_model, raw_proba: np.ndarray) -> np.ndarray:
    eps = 1e-7
    logit = np.log(
        np.clip(raw_proba, eps, 1 - eps) / (1 - np.clip(raw_proba, eps, 1 - eps))
    ).reshape(-1, 1)
    return platt_model.predict_proba(logit)[:, 1].astype(np.float32)


def apply_isotonic(iso_model, raw_proba: np.ndarray) -> np.ndarray:
    return iso_model.predict(raw_proba).astype(np.float32)


def calibrate(artifact: dict, raw_proba: np.ndarray) -> np.ndarray:
    """Apply calibration.  Averages Platt and isotonic outputs (ensemble)."""
    platt = artifact.get("platt")
    iso   = artifact.get("isotonic")

    if platt is not None and iso is not None:
        cal_platt = apply_platt(platt, raw_proba)
        cal_iso   = apply_isotonic(iso, raw_proba)
        return ((cal_platt + cal_iso) / 2.0).astype(np.float32)
    elif platt is not None:
        return apply_platt(platt, raw_proba)
    elif iso is not None:
        return apply_isotonic(iso, raw_proba)
    else:
        print("  WARNING: No calibrators found — returning raw probabilities as calibrated.")
        return raw_proba.copy()


# ── Prediction helpers ────────────────────────────────────────────────────────

def _predict_batch(model, model_type: str, X_batch: np.ndarray, num_boost_round: int) -> np.ndarray:
    """Run model inference for one batch; handles LGBM vs RF differences."""
    if model_type == "lgbm":
        return model.predict(X_batch, num_iteration=num_boost_round).astype(np.float32)
    else:  # rf
        return model.predict_proba(X_batch)[:, 1].astype(np.float32)


# ── Main inference ────────────────────────────────────────────────────────────

def run_inference(
    features_path: Path,
    artifact_path: Path,
    output_dir: Path,
    model_type: str,
    wb: WandbRunLogger | None = None,
) -> Path:
    print("\n" + "=" * 70)
    print(f"FORWARD INFERENCE — 2024 ({model_type.upper()})")
    print("=" * 70)
    print(f"  Features:   {features_path}")
    print(f"  Artifact:   {artifact_path}")
    print(f"  Model type: {model_type}")
    if wb is not None:
        wb.log({"inference/stage": "start"})
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load deployment artifact ───────────────────────────────────────────────
    print("\nLoading deployment artifact…")
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    model         = artifact["model"]
    feature_cols  = artifact["feature_cols"]
    meta          = artifact.get("metadata", {})
    num_boost_round = int(meta.get("num_boost_round", 2555)) if model_type == "lgbm" else 0
    print(f"  Model trained on years: {meta.get('train_years')}")
    if model_type == "lgbm":
        print(f"  num_boost_round:        {num_boost_round}")
    print(f"  feature_cols:           {len(feature_cols)}")
    print(f"  calibrators:            platt={artifact.get('platt') is not None}, "
          f"isotonic={artifact.get('isotonic') is not None}")
    report_memory_usage("after loading artifact")

    # ── Validate features parquet has expected columns ─────────────────────────
    schema = pq.ParquetFile(features_path).schema_arrow
    schema_names = set(schema.names)
    missing = [c for c in feature_cols if c not in schema_names]
    if missing:
        raise ValueError(
            f"features parquet is missing {len(missing)} expected columns:\n  {missing[:10]}"
        )
    meta_cols  = [c for c in ["row", "col"] if c in schema_names]
    coord_cols = [c for c in ["x", "y"] if c in schema_names]
    read_cols  = meta_cols + coord_cols + feature_cols

    # ── Pass 1: count rows ─────────────────────────────────────────────────────
    print("\nPass 1: counting rows…")
    n_total = 0
    pf = pq.ParquetFile(features_path)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["row"], use_threads=True):
            n_total += len(batch)
            del batch
    finally:
        del pf
    print(f"  {n_total:,} inference pixels")
    if wb is not None:
        wb.log({"inference/n_pixels_target": int(n_total), "inference/stage": "pass1_done"})

    # ── Pass 2: batch inference + write ───────────────────────────────────────
    out_path = output_dir / "forward_scored_2024.parquet"
    print(f"\nPass 2: inference → {out_path.name}…")

    writer = None
    n_written = 0
    _mile = -1

    pf = pq.ParquetFile(features_path)
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols, use_threads=True):
            # Extract feature matrix
            X_batch = np.column_stack([
                batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                for c in feature_cols
            ])

            raw_proba = _predict_batch(model, model_type, X_batch, num_boost_round)
            cal_proba = calibrate(artifact, raw_proba)

            # Build output table
            out_dict: dict[str, pa.Array] = {}
            for c in meta_cols + coord_cols:
                arr = batch.column(c).to_numpy(zero_copy_only=False)
                out_dict[c] = pa.array(arr, type=pa.int32() if c in meta_cols else pa.float32())
            out_dict["y_pred_proba_raw"]        = pa.array(raw_proba, type=pa.float32())
            out_dict["y_pred_proba_calibrated"] = pa.array(cal_proba, type=pa.float32())

            out_table = pa.table(out_dict)
            if writer is None:
                writer = pq.ParquetWriter(out_path, out_table.schema)
            writer.write_table(out_table)
            n_written += len(out_table)

            pct = n_written * 100 // n_total if n_total > 0 else 0
            ms = pct // 10
            if ms > _mile:
                _mile = ms
                print(f"  {pct}% — {n_written:,}/{n_total:,}")
                report_memory_usage(f"  {pct}%")
                if wb is not None:
                    wb.log(
                        {
                            "inference/progress_pct": int(pct),
                            "inference/n_pixels_written": int(n_written),
                        }
                    )

            del batch, X_batch, raw_proba, cal_proba, out_dict, out_table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
        del pf
        gc.collect()

    print(f"\nWritten: {n_written:,} rows → {out_path}")

    # Quick stats on calibrated scores
    print("\nSampling calibrated probability statistics…")
    stats_sample = pq.read_table(out_path, columns=["y_pred_proba_calibrated"])
    proba_arr = stats_sample.column("y_pred_proba_calibrated").to_numpy(zero_copy_only=False)
    print(f"  min:    {proba_arr.min():.6f}")
    print(f"  p50:    {np.median(proba_arr):.6f}")
    print(f"  p95:    {np.percentile(proba_arr, 95):.6f}")
    print(f"  p99:    {np.percentile(proba_arr, 99):.6f}")
    print(f"  max:    {proba_arr.max():.6f}")
    del proba_arr, stats_sample

    return out_path


def main() -> None:
    # Re-import config after runner.py reload
    from scripts.regions.shared.forward.config import (  # noqa: F401
        DATA_SUBDIR,
        MODEL_PREFIX,
        OUTPUTS_SUBDIR,
        forward_dir_search_paths,
        ml_models_dir_search_paths,
        resolve_forward_dir,
    )

    model_type = os.environ.get("PA3030_FORWARD_MODEL_TYPE", "lgbm").strip().lower()
    from datetime import datetime as _dt

    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    wb = WandbRunLogger(
        project="forward",
        run_name=f"predict_{OUTPUTS_SUBDIR}_{model_type}_{_ts}",
        config={
            "region": OUTPUTS_SUBDIR,
            "model_type": model_type,
            "forward_stage": "inference",
        },
    )
    wb.start()

    repo_root   = get_repo_root()
    forward_dir = resolve_forward_dir(repo_root, OUTPUTS_SUBDIR)
    output_dir  = forward_dir / model_type

    artifact_path  = resolve_deployment_artifact(repo_root, DATA_SUBDIR, MODEL_PREFIX, model_type)
    features_path  = resolve_features_parquet(repo_root, OUTPUTS_SUBDIR)
    out = run_inference(features_path, artifact_path, output_dir, model_type, wb=wb)
    print(f"\nDone. Output: {out}")
    _stats = pq.read_table(out, columns=["y_pred_proba_calibrated"])
    _arr = _stats.column("y_pred_proba_calibrated").to_numpy(zero_copy_only=False)
    wb.log(
        {
            "inference/n_pixels": int(len(_arr)),
            "inference/prob_min": float(_arr.min()),
            "inference/prob_p50": float(np.median(_arr)),
            "inference/prob_p95": float(np.percentile(_arr, 95)),
            "inference/prob_p99": float(np.percentile(_arr, 99)),
            "inference/prob_max": float(_arr.max()),
            "inference/prob_mean": float(_arr.mean()),
            "inference/stage": "done",
        }
    )
    del _arr, _stats
    wb.finish()


if __name__ == "__main__":
    main()
