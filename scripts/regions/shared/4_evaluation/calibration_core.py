from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------


def load_cv_predictions(
    cv_path: Path,
    batch_size: int = 50_000,
    return_fold_info: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load CV predictions from parquet file using batch-based loading.

    Expected columns: y_true, y_pred_proba (and optional metadata columns like fold_idx).
    """
    print(f"Loading CV predictions from {cv_path.name}...")

    required_cols = ["y_true", "y_pred_proba"]
    parquet_file = pq.ParquetFile(cv_path)

    schema_cols = parquet_file.schema_arrow.names
    for col in required_cols:
        if col not in schema_cols:
            raise ValueError(
                f"Required column '{col}' not found in CV predictions file. "
                f"Found columns: {schema_cols}"
            )

    fold_col = None
    if "fold_idx" in schema_cols:
        fold_col = "fold_idx"
    elif "fold_id" in schema_cols:
        fold_col = "fold_id"
    elif "fold" in schema_cols:
        fold_col = "fold"

    if fold_col and return_fold_info:
        required_cols.append(fold_col)
        print(f"  Found fold column: {fold_col}")
    elif fold_col:
        print(f"  Note: Found fold column '{fold_col}' but not loading it (return_fold_info=False)")

    y_true_list = []
    y_pred_proba_list = []
    fold_idx_list = [] if (fold_col and return_fold_info) else None

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required_cols):
            batch_table = batch
            y_true_batch = batch_table["y_true"].to_numpy(zero_copy_only=False)
            y_pred_proba_batch = batch_table["y_pred_proba"].to_numpy(zero_copy_only=False)
            y_true_list.append(y_true_batch)
            y_pred_proba_list.append(y_pred_proba_batch)

            if fold_idx_list is not None:
                fold_idx_batch = batch_table[fold_col].to_numpy(zero_copy_only=False)
                fold_idx_list.append(fold_idx_batch)

            del batch_table, batch
    finally:
        del parquet_file
        gc.collect()

    y_true = np.concatenate(y_true_list, axis=0).astype(np.int32)
    y_pred_proba = np.concatenate(y_pred_proba_list, axis=0).astype(np.float32)

    fold_idx = None
    if fold_idx_list is not None:
        fold_idx = np.concatenate(fold_idx_list, axis=0)

    valid_mask = np.isfinite(y_pred_proba) & (0.0 <= y_pred_proba) & (y_pred_proba <= 1.0)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        error_msg = (
            f"\nERROR: Found {n_invalid} invalid predictions (NaN or out of [0,1] range) "
            f"in CV predictions.\n"
            f"This indicates a problem with the training script. Please investigate "
            f"the training script before trusting the calibration. Stopping execution."
        )
        print(error_msg)
        raise ValueError(error_msg)

    print(
        f"  Loaded {len(y_true):,} CV predictions "
        f"({y_true.sum():,} positive, {(y_true == 0).sum():,} negative)"
    )
    if fold_idx is not None:
        unique_folds = np.unique(fold_idx)
        print(f"  Found {len(unique_folds)} fold(s): {sorted(unique_folds.tolist())}")
        for fold_id in unique_folds:
            fold_mask = fold_idx == fold_id
            n_fold = int(fold_mask.sum())
            print(f"    Fold {fold_id}: {n_fold:,} samples ({100 * n_fold / len(y_true):.1f}%)")

    return y_true, y_pred_proba, fold_idx


def load_test_predictions(
    test_path: Path,
    batch_size: int = 50_000,
    load_minimal_cols: bool = True,
) -> Tuple[Optional[pd.DataFrame], np.ndarray, np.ndarray]:
    """
    Load test predictions from scored parquet file using batch-based loading.
    """
    print(f"Loading test predictions from {test_path.name}...")

    required_cols = ["y_true", "y_pred_proba"]
    parquet_file = pq.ParquetFile(test_path)

    schema_cols = parquet_file.schema_arrow.names
    for col in required_cols:
        if col not in schema_cols:
            raise ValueError(
                f"Required column '{col}' not found in test predictions file. "
                f"Found columns: {schema_cols}"
            )

    if load_minimal_cols:
        columns_to_read = required_cols
        df = None
    else:
        metadata_cols = [c for c in ["row", "col", "year", "x", "y", "id"] if c in schema_cols]
        columns_to_read = required_cols + metadata_cols
        metadata_dict: Dict[str, list[np.ndarray]] = {col: [] for col in metadata_cols}

    y_true_list = []
    y_pred_proba_list = []

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns_to_read):
            batch_table = batch
            y_true_batch = batch_table["y_true"].to_numpy(zero_copy_only=False)
            y_pred_proba_batch = batch_table["y_pred_proba"].to_numpy(zero_copy_only=False)

            y_true_list.append(y_true_batch)
            y_pred_proba_list.append(y_pred_proba_batch)

            if not load_minimal_cols:
                for col in metadata_dict:
                    col_array = batch_table[col].to_numpy(zero_copy_only=False)
                    metadata_dict[col].append(col_array)

            del batch_table, batch
    finally:
        del parquet_file
        gc.collect()

    y_true = np.concatenate(y_true_list, axis=0).astype(np.int32)
    y_pred_proba = np.concatenate(y_pred_proba_list, axis=0).astype(np.float32)

    valid_mask = np.isfinite(y_pred_proba) & (0.0 <= y_pred_proba) & (y_pred_proba <= 1.0)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        error_msg = (
            f"\nERROR: Found {n_invalid} invalid predictions (NaN or out of [0,1] range) "
            f"in test predictions.\n"
            f"This indicates a problem with the training script. Please investigate "
            f"the training script before trusting the calibration. Stopping execution."
        )
        print(error_msg)
        raise ValueError(error_msg)

    print(
        f"  Loaded {len(y_true):,} test predictions "
        f"({y_true.sum():,} positive, {(y_true == 0).sum():,} negative)"
    )

    if load_minimal_cols:
        df = None
    else:
        metadata_arrays = {col: np.concatenate(arrs, axis=0) for col, arrs in metadata_dict.items()}
        df = pd.DataFrame(metadata_arrays)

    return df, y_true, y_pred_proba


def verify_row_alignment_basic(
    y_true_1: np.ndarray,
    y_true_2: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> None:
    """
    Basic row-order verification used when reloading test predictions.
    """
    if not np.array_equal(y_true_1, y_true_2):
        n_mismatch = int((y_true_1 != y_true_2).sum())
        raise ValueError(
            f"y_true values do not match between loads: {n_mismatch} mismatches "
            f"out of {len(y_true_1)} rows. This indicates row order changed between loads."
        )

    if not np.allclose(y_pred_1, y_pred_2, rtol=1e-6):
        n_mismatch = int((~np.isclose(y_pred_1, y_pred_2, rtol=1e-6)).sum())
        raise ValueError(
            f"y_pred_proba values do not match between loads: {n_mismatch} mismatches "
            f"out of {len(y_pred_1)} rows. This indicates row order changed between loads."
        )


# -----------------------------------------------------------------------------
# Calibration models
# -----------------------------------------------------------------------------


def fit_isotonic_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray) -> IsotonicRegression:
    print("Fitting isotonic calibration...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred_proba, y_true)
    return calibrator


def fit_platt_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray) -> LogisticRegression:
    print("Fitting Platt scaling calibration...")
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-7, 1.0 - 1e-7)
    logits = np.log(y_pred_proba_clipped / (1.0 - y_pred_proba_clipped))
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(logits.reshape(-1, 1), y_true)
    return calibrator


def apply_isotonic_calibration(calibrator: IsotonicRegression, y_pred_proba: np.ndarray) -> np.ndarray:
    return calibrator.transform(y_pred_proba)


def apply_platt_calibration(calibrator: LogisticRegression, y_pred_proba: np.ndarray) -> np.ndarray:
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-7, 1.0 - 1e-7)
    logits = np.log(y_pred_proba_clipped / (1.0 - y_pred_proba_clipped))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def fit_calibration_per_fold(
    fold_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    calibration_method: Literal["isotonic", "platt"],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit calibration separately for each fold and return an averaged calibrator.
    """
    unique_folds = np.unique(fold_idx)
    n_folds = len(unique_folds)

    if n_folds < 2:
        error_msg = (
            f"\nERROR: Insufficient folds for per-fold calibration ({n_folds} fold(s) found).\n"
            "Per-fold calibration requires at least 2 folds."
        )
        print(error_msg)
        raise ValueError(error_msg)

    print(f"\nFitting {calibration_method} calibration per fold (n_folds={n_folds})...")

    if calibration_method == "isotonic":
        raise ValueError(
            "Isotonic calibration cannot be safely averaged across folds. "
            "Use pooled isotonic with guaranteed OOF predictions or Platt scaling for per-fold."
        )

    fold_data = []
    all_calibrated_proba = np.zeros_like(y_pred_proba)

    for fold_id in unique_folds:
        fold_mask = fold_idx == fold_id
        y_true_fold = y_true[fold_mask]
        y_pred_fold = y_pred_proba[fold_mask]

        if len(y_true_fold) < 10:
            print(f"  WARNING: Fold {fold_id} has only {len(y_true_fold)} samples, skipping")
            continue

        calibrator_fold = fit_platt_calibration(y_true_fold, y_pred_fold)
        calibrated_fold = apply_platt_calibration(calibrator_fold, y_pred_fold)
        all_calibrated_proba[fold_mask] = calibrated_fold

        coef = float(calibrator_fold.coef_[0][0])
        intercept = float(calibrator_fold.intercept_[0])
        fold_data.append((int(fold_id), calibrator_fold, len(y_true_fold), coef, intercept))

        print(f"  Fold {fold_id}: {len(y_true_fold):,} samples, coef={coef:.4f}, intercept={intercept:.4f}")

    if not fold_data:
        raise ValueError("No folds had sufficient samples for calibration")

    print("\n  Averaged calibrated probabilities across folds (weighted by fold size)")

    class AveragedPlattCalibrator:
        def __init__(self, fd):
            self.fold_data = fd
            self.fold_counts = [count for _, _, count, _, _ in fd]
            self.weights = np.array(self.fold_counts) / sum(self.fold_counts)
            self._is_averaged_platt = True

        def calibrate(self, y_pred_proba: np.ndarray) -> np.ndarray:
            calibrated_list = []
            for _, calibrator, _, _, _ in self.fold_data:
                calibrated = apply_platt_calibration(calibrator, y_pred_proba)
                calibrated_list.append(calibrated)
            calibrated_array = np.stack(calibrated_list, axis=0)
            return np.average(calibrated_array, axis=0, weights=self.weights)

    averaged_calibrator = AveragedPlattCalibrator(fold_data)

    fold_info = {
        "method": "per_fold_averaged_platt",
        "averaging_strategy": "calibrated_probabilities",
        "n_folds": int(n_folds),
        "n_folds_used": len(fold_data),
        "folds": sorted(int(fid) for fid, _, _, _, _ in fold_data),
        "fold_coefficients": {int(fid): float(coef) for fid, _, _, coef, _ in fold_data},
        "fold_intercepts": {int(fid): float(intercept) for fid, _, _, _, intercept in fold_data},
        "fold_counts": {int(fid): int(count) for fid, _, count, _, _ in fold_data},
        "weights": [float(w) for w in averaged_calibrator.weights],
    }

    return averaged_calibrator, fold_info


# -----------------------------------------------------------------------------
# Calibration metrics & reliability
# -----------------------------------------------------------------------------


def compute_reliability_plot_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    binning_method: Literal["uniform", "quantile"] = "uniform",
) -> Dict[str, Any]:
    y_pred_proba = np.clip(y_pred_proba.copy(), 0.0, 1.0 - 1e-9)

    if binning_method == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bin_edges[1:], right=False)
    elif binning_method == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_pred_proba, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_indices = np.digitize(y_pred_proba, bin_edges[1:], right=False)
    else:
        raise ValueError(f"Unknown binning_method: {binning_method}")

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_predicted = np.zeros(n_bins)
    fraction_positives = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted[i] = float(y_pred_proba[mask].mean())
            fraction_positives[i] = float(y_true[mask].mean())
            counts[i] = int(mask.sum())

    return {
        "binning_method": binning_method,
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "mean_predicted": mean_predicted.tolist(),
        "fraction_positives": fraction_positives.tolist(),
        "counts": counts.tolist(),
        "n_bins": n_bins,
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    binning_method: Literal["uniform", "quantile"] = "uniform",
) -> Dict[str, float]:
    brier_score = float(np.mean((y_pred_proba - y_true) ** 2))
    y_pred_clipped = np.clip(y_pred_proba.copy(), 0.0, 1.0 - 1e-9)
    n_bins = 10

    if binning_method == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_pred_clipped, bin_edges[1:], right=False)
    elif binning_method == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_pred_clipped, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_indices = np.digitize(y_pred_clipped, bin_edges[1:], right=False)
    else:
        raise ValueError(f"Unknown binning_method: {binning_method}")

    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_conf = float(y_pred_clipped[mask].mean())
            bin_acc = float(y_true[mask].mean())
            bin_weight = float(mask.sum()) / float(total_samples)
            bin_err = abs(bin_conf - bin_acc)
            ece += bin_weight * bin_err
            mce = max(mce, bin_err)

    return {
        "brier_score": float(brier_score),
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
    }


def plot_reliability_curve(
    plot_data: Dict[str, Any],
    title: str,
    output_path: Path,
    brier_uncalibrated: float,
    brier_calibrated: float,
    binning_method: str = "uniform",
    is_uncalibrated: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    bin_centers = np.array(plot_data["bin_centers"])
    mean_predicted = np.array(plot_data["mean_predicted"])
    fraction_positives = np.array(plot_data["fraction_positives"])
    counts = np.array(plot_data["counts"])

    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.5, alpha=0.5, label="Perfect Calibration")
    ax.plot(
        mean_predicted,
        fraction_positives,
        "o-",
        linewidth=2,
        markersize=8,
        color="#2166ac",
        label="Calibration Curve",
        zorder=3,
    )

    ax2 = ax.twinx()
    ax2.bar(
        bin_centers,
        counts,
        width=0.08,
        alpha=0.2,
        color="gray",
        label="Sample Count",
        zorder=1,
    )
    ax2.set_ylabel("Number of Samples", color="gray", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_yscale("log")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect("equal")

    if is_uncalibrated:
        brier_text = f"Brier Score: {brier_uncalibrated:.6f}\nImprovement: 0.00%"
    else:
        if brier_uncalibrated > 0.0:
            improvement_pct = (brier_uncalibrated - brier_calibrated) / brier_uncalibrated * 100.0
        else:
            improvement_pct = 0.0
        brier_text = (
            f"Brier Score (Uncalibrated): {brier_uncalibrated:.6f}\n"
            f"Brier Score (Calibrated):   {brier_calibrated:.6f}\n"
            f"Improvement: {improvement_pct:.2f}%"
        )

    ax.text(
        0.05,
        0.95,
        brier_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Reliability curve plot saved: {output_path.name}")


def save_brier_score_summary(
    output_path: Path,
    cv_uncalibrated_metrics: Dict[str, float],
    cv_calibration_metrics: Dict[str, float],
    test_uncalibrated_metrics: Dict[str, float],
    test_calibrated_metrics: Dict[str, float],
    calibration_method: str,
) -> None:
    with output_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("BRIER SCORE SUMMARY - CALIBRATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Calibration Method: {calibration_method}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CROSS-VALIDATION SET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Brier Score (Uncalibrated): {cv_uncalibrated_metrics['brier_score']:.8f}\n")
        f.write(f"Brier Score (Calibrated):   {cv_calibration_metrics['brier_score']:.8f}\n")
        if cv_uncalibrated_metrics["brier_score"] > 0.0:
            improvement_cv = (
                (cv_uncalibrated_metrics["brier_score"] - cv_calibration_metrics["brier_score"])
                / cv_uncalibrated_metrics["brier_score"]
                * 100.0
            )
        else:
            improvement_cv = 0.0
        f.write(f"Improvement: {improvement_cv:.4f}%\n\n")

        f.write(
            f"ECE (Uncalibrated): {cv_uncalibrated_metrics['expected_calibration_error']:.8f}\n"
        )
        f.write(f"ECE (Calibrated):   {cv_calibration_metrics['expected_calibration_error']:.8f}\n")
        if cv_uncalibrated_metrics["expected_calibration_error"] > 0.0:
            ece_improvement_cv = (
                cv_uncalibrated_metrics["expected_calibration_error"]
                - cv_calibration_metrics["expected_calibration_error"]
            ) / cv_uncalibrated_metrics["expected_calibration_error"] * 100.0
        else:
            ece_improvement_cv = 0.0
        f.write(f"ECE Improvement: {ece_improvement_cv:.4f}%\n\n")

        f.write("-" * 70 + "\n")
        f.write("TEST SET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Brier Score (Uncalibrated): {test_uncalibrated_metrics['brier_score']:.8f}\n")
        f.write(f"Brier Score (Calibrated):   {test_calibrated_metrics['brier_score']:.8f}\n")
        if test_uncalibrated_metrics["brier_score"] > 0.0:
            improvement_test = (
                test_uncalibrated_metrics["brier_score"] - test_calibrated_metrics["brier_score"]
            ) / test_uncalibrated_metrics["brier_score"] * 100.0
        else:
            improvement_test = 0.0
        f.write(f"Improvement: {improvement_test:.4f}%\n\n")

        f.write(
            f"ECE (Uncalibrated): {test_uncalibrated_metrics['expected_calibration_error']:.8f}\n"
        )
        f.write(
            f"ECE (Calibrated):   {test_calibrated_metrics['expected_calibration_error']:.8f}\n"
        )
        if test_uncalibrated_metrics["expected_calibration_error"] > 0.0:
            ece_improvement_test = (
                test_uncalibrated_metrics["expected_calibration_error"]
                - test_calibrated_metrics["expected_calibration_error"]
            ) / test_uncalibrated_metrics["expected_calibration_error"] * 100.0
        else:
            ece_improvement_test = 0.0
        f.write(f"ECE Improvement: {ece_improvement_test:.4f}%\n\n")

        f.write("=" * 70 + "\n")

    print(f"Brier score summary saved: {output_path.name}")


__all__ = [
    "load_cv_predictions",
    "load_test_predictions",
    "verify_row_alignment_basic",
    "fit_isotonic_calibration",
    "fit_platt_calibration",
    "apply_isotonic_calibration",
    "apply_platt_calibration",
    "fit_calibration_per_fold",
    "compute_reliability_plot_data",
    "compute_calibration_metrics",
    "plot_reliability_curve",
    "save_brier_score_summary",
]

