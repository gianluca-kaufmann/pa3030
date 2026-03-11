from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from .calibration_core import (
    apply_isotonic_calibration,
    apply_platt_calibration,
    compute_calibration_metrics,
    compute_reliability_plot_data,
    fit_calibration_per_fold,
    fit_isotonic_calibration,
    fit_platt_calibration,
    load_cv_predictions,
    load_test_predictions,
    plot_reliability_curve,
    save_brier_score_summary,
    verify_row_alignment_basic,
)
from .config import ModelEvalConfig


@dataclass
class CalibrationRunResult:
    """Structured summary of a calibration run."""

    cv_predictions_file: Path
    test_predictions_file: Path
    calibrated_test_file: Path
    reliability_plot_json: Path
    calibration_metadata_json: Path
    brier_summary_txt: Path
    figures_dir: Path
    cv_uncalibrated_metrics: Dict[str, float]
    cv_calibrated_metrics: Dict[str, float]
    test_uncalibrated_metrics: Dict[str, float]
    test_calibrated_metrics: Dict[str, float]


def _find_latest_scored_file(
    config: ModelEvalConfig,
    split_version: str,
    pattern_override: Optional[str] = None,
) -> Path:
    """
    Find the most recent scored parquet file for the given model/algorithm.

    Training scripts save scored files to the root ml_models directory (not split-specific).

    LGBM: <model_id>_lgbm_scored_*.parquet
    RF:   <model_id>_rf_win5_scored_*.parquet
    """
    output_dir = config.metrics_root

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if pattern_override is not None:
        pattern = pattern_override
    else:
        if config.algorithm == "lgbm":
            pattern = f"{config.model_id}_lgbm_scored_*.parquet"
        elif config.algorithm == "rf":
            pattern = f"{config.model_id}_rf_win5_scored_*.parquet"
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

    files = list(output_dir.glob(pattern))
    files = [f for f in files if "calibrated" not in f.name]

    if not files:
        raise FileNotFoundError(
            f"No scored files found matching pattern {pattern} in {output_dir}\n"
            f"Expected pattern: {pattern}"
        )

    return max(files, key=lambda p: p.stat().st_mtime)


def _search_cv_files(
    config: ModelEvalConfig,
    split_version: str,
    require_temporal_cv: bool,
) -> Path:
    """
    Search for the appropriate CV / earlystop predictions file for calibration.

    Behaviour mirrors the original region-specific scripts but is parameterised
    by ModelEvalConfig.
    """
    ml_root = config.metrics_root
    split_dir = ml_root / split_version

    search_dirs = []
    if split_dir.exists():
        search_dirs.append(split_dir)
    search_dirs.append(ml_root)

    temporal_cv_files: list[Path] = []
    other_cv_files: list[Path] = []

    if config.algorithm == "lgbm":
        # 1) Temporal CV predictions (preferred)
        for d in search_dirs:
            if not d.exists():
                continue
            # Preferred pattern with win5
            temporal = list(d.glob(f"{config.model_id}_lgbm_win5_temporal_cv_predictions_*.parquet"))
            temporal_cv_files.extend(temporal)
            # Backwards-compatible pattern without win5
            temporal_alt = list(d.glob(f"{config.model_id}_lgbm_temporal_cv_predictions_*.parquet"))
            temporal_alt = [p for p in temporal_alt if "win5" not in p.name]
            temporal_cv_files.extend(temporal_alt)

        if temporal_cv_files:
            return max(temporal_cv_files, key=lambda p: p.stat().st_mtime)

        # 2) Fallback: any other CV predictions for this model
        if require_temporal_cv:
            searched = "\n".join(str(d) for d in search_dirs)
            raise FileNotFoundError(
                "ERROR: --require-temporal-cv flag is set, but no temporal CV predictions file found.\n"
                "Temporal CV predictions are required for calibration.\n"
                f"Expected pattern: {config.model_id}_lgbm_win5_temporal_cv_predictions_*.parquet\n"
                f"Searched in:\n{searched}"
            )

        for d in search_dirs:
            if not d.exists():
                continue
            files = list(d.glob(f"{config.model_id}_lgbm_win5_cv_predictions_*.parquet"))
            other_cv_files.extend(files)
            generic = list(d.glob("*cv_predictions*.parquet"))
            generic = [p for p in generic if p.name.startswith(f"{config.model_id}_lgbm_")]
            other_cv_files.extend(generic)

        if other_cv_files:
            return max(other_cv_files, key=lambda p: p.stat().st_mtime)

        searched = "\n".join(str(d) for d in search_dirs)
        raise FileNotFoundError(
            "CV predictions file not found for calibration.\n"
            f"Expected temporal CV pattern: {config.model_id}_lgbm_win5_temporal_cv_predictions_*.parquet\n"
            f"Searched in:\n{searched}"
        )

    # RF: use earlystop scored predictions as validation
    if config.algorithm == "rf":
        earlystop_files: list[Path] = []
        for d in search_dirs:
            if not d.exists():
                continue
            files = list(d.glob(f"{config.model_id}_rf_win5_earlystop_scored_*.parquet"))
            earlystop_files.extend(files)

        if earlystop_files:
            return max(earlystop_files, key=lambda p: p.stat().st_mtime)

        searched = "\n".join(str(d) for d in search_dirs)
        raise FileNotFoundError(
            "Earlystop predictions file not found for RF calibration.\n"
            f"Expected pattern: {config.model_id}_rf_win5_earlystop_scored_*.parquet\n"
            f"Searched in:\n{searched}"
        )

    raise ValueError(f"Unsupported algorithm for CV search: {config.algorithm}")


def run_calibration(
    config: ModelEvalConfig,
    split_version: str = "main",
    cv_predictions_path: Optional[Path] = None,
    test_predictions_path: Optional[Path] = None,
    calibration_method: Literal["isotonic", "platt"] = "platt",
    assume_oof: bool = False,
    per_fold: bool = False,
    require_temporal_cv: bool = False,
    plot_binning: Literal["uniform", "quantile", "both"] = "both",
) -> CalibrationRunResult:
    """
    Shared calibration runner used by region/model-specific wrapper scripts.
    """
    repo_root = config.repo_root
    output_dir = config.metrics_root
    split_output_dir = output_dir / split_version
    split_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"CALIBRATION: {config.model_id}_{config.algorithm.upper()} (Region: {config.region}, Split: {split_version})")
    print("=" * 70)
    print(f"\nRegion: {config.region}")
    print(f"Model id: {config.model_id}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Split version: {split_version}")
    print(f"Calibration method: {calibration_method}")
    print(f"Output directory: {split_output_dir}")

    # ------------------------------------------------------------------
    # STEP 1: Load CV / earlystop predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: LOAD CV / VALIDATION PREDICTIONS")
    print("=" * 70)

    if cv_predictions_path is None:
        print("\nSearching for CV/validation predictions file...")
        if config.algorithm == "lgbm":
            print("  Priority: temporal CV predictions (from training) > other CV predictions")
        else:
            print("  Using earlystop validation predictions for RF (no temporal CV available)")

        cv_predictions_path = _search_cv_files(
            config=config,
            split_version=split_version,
            require_temporal_cv=require_temporal_cv,
        )

    if not cv_predictions_path.exists():
        raise FileNotFoundError(f"CV/validation predictions file not found: {cv_predictions_path}")

    is_temporal_cv = "temporal_cv_predictions" in cv_predictions_path.name
    is_earlystop = "earlystop_scored" in cv_predictions_path.name
    is_spatial_cv = "spatial" in cv_predictions_path.name.lower() or (
        "cv_predictions" in cv_predictions_path.name
        and "temporal" not in cv_predictions_path.name
        and not is_earlystop
    )

    cv_type = (
        "temporal"
        if is_temporal_cv
        else ("earlystop" if is_earlystop else ("spatial" if is_spatial_cv else "unknown"))
    )

    if is_temporal_cv:
        print(f"  Using temporal CV predictions: {cv_predictions_path.name}")
    elif is_earlystop:
        print(f"  Using earlystop validation predictions: {cv_predictions_path.name}")
    elif is_spatial_cv:
        if require_temporal_cv:
            raise ValueError(
                "ERROR: --require-temporal-cv flag is set, but the CV file appears to be spatial CV.\n"
                "Temporal CV predictions are required for calibration."
            )
        print("  WARNING: Using spatial CV predictions for calibration; this is not ideal.")
    else:
        print(f"  Note: Could not determine CV type from filename: {cv_predictions_path.name}")

    return_fold_info_for_calib = not is_earlystop
    y_cv_true, y_cv_proba, fold_idx = load_cv_predictions(
        cv_predictions_path, return_fold_info=return_fold_info_for_calib
    )

    # ------------------------------------------------------------------
    # STEP 1.5: OOF requirements and policy
    # ------------------------------------------------------------------
    if is_earlystop:
        print("\n" + "=" * 70)
        print("STEP 1.5: SKIP OOF REQUIREMENTS (EARLYSTOP MODE)")
        print("=" * 70)
        print("\nUsing earlystop validation predictions (held-out, not CV).")
        fold_idx = None
    else:
        print("\n" + "=" * 70)
        print("STEP 1.5: CHECK OUT-OF-FOLD (OOF) REQUIREMENTS")
        print("=" * 70)

    fold_info: Optional[Dict[str, Any]] = None

    if not is_earlystop:
        if fold_idx is not None:
            unique_folds = sorted(set(int(f) for f in fold_idx))
            n_folds = len(unique_folds)
            print(f"\nFold information found ({n_folds} fold(s)): {unique_folds}")

            if per_fold:
                if n_folds < 2:
                    raise ValueError(
                        "ERROR: Insufficient folds for per-fold calibration "
                        f"({n_folds} fold(s) found). Use pooled calibration instead."
                    )
                print("Using per-fold calibration (--per-fold).")
                if calibration_method == "isotonic":
                    print("\n" + "!" * 70)
                    print("WARNING: Isotonic cannot be averaged across folds; switching to Platt.")
                    print("!" * 70 + "\n")
                    calibration_method = "platt"
            else:
                print("Using pooled calibration (fold info available, --per-fold not set).")
                if calibration_method == "isotonic" and not assume_oof:
                    raise ValueError(
                        "\nERROR: Isotonic calibration on pooled data requires --assume-oof.\n"
                        "If CV predictions are guaranteed OOF across all temporal folds, "
                        "re-run with --assume-oof."
                    )
                if calibration_method == "isotonic" and assume_oof:
                    print("\n" + "!" * 70)
                    print("WARNING: Isotonic with --assume-oof; ensure CV predictions are true OOF.")
                    print("!" * 70 + "\n")
        else:
            print("\nNo fold information found in CV predictions.")
            if calibration_method == "isotonic" and not assume_oof:
                raise ValueError(
                    "\nERROR: Cannot verify OOF property without fold information.\n"
                    "Isotonic requires --assume-oof or explicit fold info."
                )
            if calibration_method == "isotonic" and assume_oof:
                print("\n" + "!" * 70)
                print("WARNING: Isotonic with --assume-oof and no fold info; ensure CV is true OOF.")
                print("!" * 70 + "\n")

    # ------------------------------------------------------------------
    # STEP 2: Load test predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: LOAD TEST PREDICTIONS")
    print("=" * 70)

    if test_predictions_path is None:
        try:
            test_predictions_path = _find_latest_scored_file(config, split_version=split_version)
            print(f"\nFound latest test predictions file: {test_predictions_path.name}")
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            expected_pattern = (
                f"{config.model_id}_lgbm_scored_*.parquet"
                if config.algorithm == "lgbm"
                else f"{config.model_id}_rf_win5_scored_*.parquet"
            )
            print(f"Expected pattern: {expected_pattern}")
            raise

    if not test_predictions_path.exists():
        raise FileNotFoundError(f"Test predictions file not found: {test_predictions_path}")

    _, y_test_true, y_test_proba = load_test_predictions(
        test_predictions_path, load_minimal_cols=True
    )

    # ------------------------------------------------------------------
    # STEP 3: Fit calibration on CV / validation predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    if is_earlystop:
        print(f"STEP 3: FIT {calibration_method.upper()} CALIBRATION ON EARLYSTOP VALIDATION")
    else:
        print(f"STEP 3: FIT {calibration_method.upper()} CALIBRATION ON CV PREDICTIONS")
    print("=" * 70)

    if fold_idx is not None and per_fold:
        print("\nUsing per-fold calibration...")
        calibrator, fold_info = fit_calibration_per_fold(
            fold_idx, y_cv_true, y_cv_proba, calibration_method
        )
    else:
        if fold_idx is not None:
            print("\nUsing pooled calibration (fold info available, --per-fold not set)...")
        elif assume_oof:
            print("\nUsing pooled calibration (OOF assumed via --assume-oof)...")
        else:
            print("\nUsing pooled calibration (no fold information)...")

        if calibration_method == "isotonic":
            calibrator = fit_isotonic_calibration(y_cv_true, y_cv_proba)
        elif calibration_method == "platt":
            calibrator = fit_platt_calibration(y_cv_true, y_cv_proba)
        else:
            raise ValueError(f"Unknown calibration method: {calibration_method}")

        fold_info = {
            "method": "pooled",
            "assume_oof": assume_oof,
            "oof_assumption": (
                "user_asserted" if (calibration_method == "isotonic" and assume_oof) else None
            ),
        }

    if calibration_method == "isotonic":
        apply_calibration = lambda proba: apply_isotonic_calibration(calibrator, proba)
    elif calibration_method == "platt":
        if hasattr(calibrator, "_is_averaged_platt") and getattr(calibrator, "_is_averaged_platt"):
            apply_calibration = lambda proba: calibrator.calibrate(proba)
        else:
            apply_calibration = lambda proba: apply_platt_calibration(calibrator, proba)
    else:
        raise ValueError(f"Unknown calibration method: {calibration_method}")

    # Evaluate on CV / validation
    y_cv_calibrated = apply_calibration(y_cv_proba)
    cv_uncalibrated_metrics_uniform = compute_calibration_metrics(
        y_cv_true, y_cv_proba, binning_method="uniform"
    )
    cv_calibrated_metrics_uniform = compute_calibration_metrics(
        y_cv_true, y_cv_calibrated, binning_method="uniform"
    )

    print(
        f"  CV (uncalibrated): Brier={cv_uncalibrated_metrics_uniform['brier_score']:.6f}, "
        f"ECE={cv_uncalibrated_metrics_uniform['expected_calibration_error']:.6f}"
    )
    print(
        f"  CV (calibrated):   Brier={cv_calibrated_metrics_uniform['brier_score']:.6f}, "
        f"ECE={cv_calibrated_metrics_uniform['expected_calibration_error']:.6f}"
    )

    # ------------------------------------------------------------------
    # STEP 4: Apply calibration to test predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: APPLY CALIBRATION TO TEST PREDICTIONS")
    print("=" * 70)

    y_test_calibrated = apply_calibration(y_test_proba)
    print(f"\nApplied calibration to {len(y_test_calibrated):,} test predictions")

    test_uncalibrated_metrics_uniform = compute_calibration_metrics(
        y_test_true, y_test_proba, binning_method="uniform"
    )
    test_calibrated_metrics_uniform = compute_calibration_metrics(
        y_test_true, y_test_calibrated, binning_method="uniform"
    )

    print(
        f"  Test (uncalibrated): Brier={test_uncalibrated_metrics_uniform['brier_score']:.6f}, "
        f"ECE={test_uncalibrated_metrics_uniform['expected_calibration_error']:.6f}"
    )
    print(
        f"  Test (calibrated):   Brier={test_calibrated_metrics_uniform['brier_score']:.6f}, "
        f"ECE={test_calibrated_metrics_uniform['expected_calibration_error']:.6f}"
    )

    # Also compute quantile-binned metrics for metadata
    cv_uncalibrated_metrics_quantile = compute_calibration_metrics(
        y_cv_true, y_cv_proba, binning_method="quantile"
    )
    cv_calibrated_metrics_quantile = compute_calibration_metrics(
        y_cv_true, y_cv_calibrated, binning_method="quantile"
    )
    test_uncalibrated_metrics_quantile = compute_calibration_metrics(
        y_test_true, y_test_proba, binning_method="quantile"
    )
    test_calibrated_metrics_quantile = compute_calibration_metrics(
        y_test_true, y_test_calibrated, binning_method="quantile"
    )

    # ------------------------------------------------------------------
    # STEP 5: Reliability plot data & figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: COMPUTE RELIABILITY PLOT DATA AND PLOTS")
    print("=" * 70)

    cv_uncalibrated_plot_uniform = compute_reliability_plot_data(
        y_cv_true, y_cv_proba, n_bins=10, binning_method="uniform"
    )
    cv_calibrated_plot_uniform = compute_reliability_plot_data(
        y_cv_true, y_cv_calibrated, n_bins=10, binning_method="uniform"
    )
    test_uncalibrated_plot_uniform = compute_reliability_plot_data(
        y_test_true, y_test_proba, n_bins=10, binning_method="uniform"
    )
    test_calibrated_plot_uniform = compute_reliability_plot_data(
        y_test_true, y_test_calibrated, n_bins=10, binning_method="uniform"
    )

    cv_uncalibrated_plot_quantile = compute_reliability_plot_data(
        y_cv_true, y_cv_proba, n_bins=10, binning_method="quantile"
    )
    cv_calibrated_plot_quantile = compute_reliability_plot_data(
        y_cv_true, y_cv_calibrated, n_bins=10, binning_method="quantile"
    )
    test_uncalibrated_plot_quantile = compute_reliability_plot_data(
        y_test_true, y_test_proba, n_bins=10, binning_method="quantile"
    )
    test_calibrated_plot_quantile = compute_reliability_plot_data(
        y_test_true, y_test_calibrated, n_bins=10, binning_method="quantile"
    )

    figures_dir = (
        repo_root
        / f"outputs/{config.region}/results/{config.model_id}_{config.algorithm}"
        / split_version
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_paths: list[tuple[str, Path]] = []

    use_uniform = plot_binning in ("uniform", "both")
    use_quantile = plot_binning in ("quantile", "both")

    if use_uniform:
        # Test set
        p = figures_dir / "reliability_test_uncalibrated_uniform.png"
        plot_reliability_curve(
            test_uncalibrated_plot_uniform,
            f"Reliability Curve - Test (Uncalibrated, Uniform)\n{calibration_method.upper()}",
            p,
            test_uncalibrated_metrics_uniform["brier_score"],
            test_uncalibrated_metrics_uniform["brier_score"],
            binning_method="uniform",
            is_uncalibrated=True,
        )
        plot_paths.append(("Test (uncalibrated, uniform)", p))

        p = figures_dir / "reliability_test_calibrated_uniform.png"
        plot_reliability_curve(
            test_calibrated_plot_uniform,
            f"Reliability Curve - Test (Calibrated, Uniform)\n{calibration_method.upper()}",
            p,
            test_uncalibrated_metrics_uniform["brier_score"],
            test_calibrated_metrics_uniform["brier_score"],
            binning_method="uniform",
            is_uncalibrated=False,
        )
        plot_paths.append(("Test (calibrated, uniform)", p))

        # CV set
        p = figures_dir / "reliability_cv_uncalibrated_uniform.png"
        plot_reliability_curve(
            cv_uncalibrated_plot_uniform,
            f"Reliability Curve - CV (Uncalibrated, Uniform)\n{calibration_method.upper()}",
            p,
            cv_uncalibrated_metrics_uniform["brier_score"],
            cv_uncalibrated_metrics_uniform["brier_score"],
            binning_method="uniform",
            is_uncalibrated=True,
        )
        plot_paths.append(("CV (uncalibrated, uniform)", p))

        p = figures_dir / "reliability_cv_calibrated_uniform.png"
        plot_reliability_curve(
            cv_calibrated_plot_uniform,
            f"Reliability Curve - CV (Calibrated, Uniform)\n{calibration_method.upper()}",
            p,
            cv_uncalibrated_metrics_uniform["brier_score"],
            cv_calibrated_metrics_uniform["brier_score"],
            binning_method="uniform",
            is_uncalibrated=False,
        )
        plot_paths.append(("CV (calibrated, uniform)", p))

    if use_quantile:
        # Test set
        p = figures_dir / "reliability_test_uncalibrated_quantile.png"
        plot_reliability_curve(
            test_uncalibrated_plot_quantile,
            f"Reliability Curve - Test (Uncalibrated, Quantile)\n{calibration_method.upper()}",
            p,
            test_uncalibrated_metrics_uniform["brier_score"],
            test_uncalibrated_metrics_uniform["brier_score"],
            binning_method="quantile",
            is_uncalibrated=True,
        )
        plot_paths.append(("Test (uncalibrated, quantile)", p))

        p = figures_dir / "reliability_test_calibrated_quantile.png"
        plot_reliability_curve(
            test_calibrated_plot_quantile,
            f"Reliability Curve - Test (Calibrated, Quantile)\n{calibration_method.upper()}",
            p,
            test_uncalibrated_metrics_uniform["brier_score"],
            test_calibrated_metrics_uniform["brier_score"],
            binning_method="quantile",
            is_uncalibrated=False,
        )
        plot_paths.append(("Test (calibrated, quantile)", p))

        # CV set
        p = figures_dir / "reliability_cv_uncalibrated_quantile.png"
        plot_reliability_curve(
            cv_uncalibrated_plot_quantile,
            f"Reliability Curve - CV (Uncalibrated, Quantile)\n{calibration_method.upper()}",
            p,
            cv_uncalibrated_metrics_uniform["brier_score"],
            cv_uncalibrated_metrics_uniform["brier_score"],
            binning_method="quantile",
            is_uncalibrated=True,
        )
        plot_paths.append(("CV (uncalibrated, quantile)", p))

        p = figures_dir / "reliability_cv_calibrated_quantile.png"
        plot_reliability_curve(
            cv_calibrated_plot_quantile,
            f"Reliability Curve - CV (Calibrated, Quantile)\n{calibration_method.upper()}",
            p,
            cv_uncalibrated_metrics_uniform["brier_score"],
            cv_calibrated_metrics_uniform["brier_score"],
            binning_method="quantile",
            is_uncalibrated=False,
        )
        plot_paths.append(("CV (calibrated, quantile)", p))

    print(f"\nCreated {len(plot_paths)} reliability curve plots.")

    # ------------------------------------------------------------------
    # STEP 6: Save calibrated predictions & metadata
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: SAVE RESULTS")
    print("=" * 70)

    # Reload test with metadata for saving
    test_df_metadata, y_test_true_meta, y_test_proba_meta = load_test_predictions(
        test_predictions_path, load_minimal_cols=False
    )
    print("\nVerifying row alignment between initial and metadata loads...")
    verify_row_alignment_basic(
        y_true_1=y_test_true,
        y_true_2=y_test_true_meta,
        y_pred_1=y_test_proba,
        y_pred_2=y_test_proba_meta,
    )

    minimal_cols = []
    if test_df_metadata is not None:
        for col in ("id", "row", "col", "x", "y", "year"):
            if col in test_df_metadata.columns:
                minimal_cols.append(col)

    if test_df_metadata is not None and minimal_cols:
        test_df_minimal = test_df_metadata[minimal_cols].copy()
    else:
        test_df_minimal = pd.DataFrame(index=range(len(y_test_true)))

    test_df_minimal["y_true"] = y_test_true
    test_df_minimal["y_pred_proba_uncalibrated"] = y_test_proba
    test_df_minimal["y_pred_proba_calibrated"] = y_test_calibrated
    test_df_minimal["y_pred_proba"] = y_test_calibrated

    if config.algorithm == "lgbm":
        calibrated_test_path = (
            split_output_dir
            / f"{config.model_id}_lgbm_scored_calibrated_{calibration_method}_{timestamp}.parquet"
        )
    else:
        calibrated_test_path = (
            split_output_dir
            / f"{config.model_id}_rf_win5_scored_calibrated_{calibration_method}_{timestamp}.parquet"
        )

    test_df_minimal.to_parquet(calibrated_test_path, index=False)
    print(f"\nCalibrated test predictions saved: {calibrated_test_path.name}")

    # Reliability JSON
    reliability_plot_path = (
        split_output_dir
        / f"{config.model_id}_{config.algorithm}_win5_reliability_plot_{calibration_method}_{timestamp}.json"
    )
    reliability_plot_data: Dict[str, Any] = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "region": config.region,
            "model_id": config.model_id,
            "algorithm": config.algorithm,
            "calibration_method": calibration_method,
            "cv_predictions_file": cv_predictions_path.name,
            "cv_type": cv_type,
            "is_temporal_cv": is_temporal_cv,
            "is_spatial_cv": is_spatial_cv,
            "test_predictions_file": test_predictions_path.name,
            "fold_info_available": fold_idx is not None,
            "calibration_strategy": "per_fold" if (fold_idx is not None and per_fold) else "pooled",
            "per_fold_flag": per_fold,
            "assume_oof_flag": assume_oof if (fold_idx is None or not per_fold) else None,
            "oof_assumption": fold_info.get("oof_assumption") if fold_info else None,
        },
        "cv": {
            "uncalibrated": {
                "uniform": cv_uncalibrated_plot_uniform,
                "quantile": cv_uncalibrated_plot_quantile,
            },
            "calibrated": {
                "uniform": cv_calibrated_plot_uniform,
                "quantile": cv_calibrated_plot_quantile,
            },
        },
        "test": {
            "uncalibrated": {
                "uniform": test_uncalibrated_plot_uniform,
                "quantile": test_uncalibrated_plot_quantile,
            },
            "calibrated": {
                "uniform": test_calibrated_plot_uniform,
                "quantile": test_calibrated_plot_quantile,
            },
        },
    }
    reliability_plot_path.write_text(json.dumps(reliability_plot_data, indent=2))
    print(f"Reliability plot data saved: {reliability_plot_path.name}")

    # Calibration metadata JSON
    calibration_metadata_path = (
        split_output_dir
        / f"{config.model_id}_{config.algorithm}_win5_calibration_{calibration_method}_{timestamp}.json"
    )
    calibration_metadata: Dict[str, Any] = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "region": config.region,
            "model_id": config.model_id,
            "algorithm": config.algorithm,
            "calibration_method": calibration_method,
            "cv_predictions_file": cv_predictions_path.name,
            "cv_type": cv_type,
            "is_temporal_cv": is_temporal_cv,
            "is_spatial_cv": is_spatial_cv,
            "test_predictions_file": test_predictions_path.name,
            "calibrated_test_predictions_file": calibrated_test_path.name,
        },
        "oof_verification": {
            "fold_info_available": fold_idx is not None,
            "calibration_strategy": "per_fold" if (fold_idx is not None and per_fold) else "pooled",
            "per_fold_flag": per_fold,
            "assume_oof_flag": assume_oof if (fold_idx is None or not per_fold) else None,
            "oof_assumption": fold_info.get("oof_assumption") if fold_info else None,
            "fold_info": fold_info,
        },
        "calibration_metrics": {
            "cv": {
                "uncalibrated": {
                    "uniform": cv_uncalibrated_metrics_uniform,
                    "quantile": cv_uncalibrated_metrics_quantile,
                },
                "calibrated": {
                    "uniform": cv_calibrated_metrics_uniform,
                    "quantile": cv_calibrated_metrics_quantile,
                },
            },
            "test": {
                "uncalibrated": {
                    "uniform": test_uncalibrated_metrics_uniform,
                    "quantile": test_uncalibrated_metrics_quantile,
                },
                "calibrated": {
                    "uniform": test_calibrated_metrics_uniform,
                    "quantile": test_calibrated_metrics_quantile,
                },
            },
        },
        "data_info": {
            "n_cv": int(len(y_cv_true)),
            "n_cv_positives": int(y_cv_true.sum()),
            "n_cv_negatives": int((y_cv_true == 0).sum()),
            "n_test": int(len(y_test_true)),
            "n_test_positives": int(y_test_true.sum()),
            "n_test_negatives": int((y_test_true == 0).sum()),
        },
    }
    calibration_metadata_path.write_text(json.dumps(calibration_metadata, indent=2))
    print(f"Calibration metadata saved: {calibration_metadata_path.name}")

    # Brier/ECE summary text
    brier_summary_path = (
        split_output_dir
        / f"{config.model_id}_{config.algorithm}_win5_brier_summary_{calibration_method}_{timestamp}.txt"
    )
    save_brier_score_summary(
        brier_summary_path,
        cv_uncalibrated_metrics=cv_uncalibrated_metrics_uniform,
        cv_calibration_metrics=cv_calibrated_metrics_uniform,
        test_uncalibrated_metrics=test_uncalibrated_metrics_uniform,
        test_calibrated_metrics=test_calibrated_metrics_uniform,
        calibration_method=calibration_method,
    )

    # Final summary to stdout
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nRegion: {config.region}")
    print(f"Model id: {config.model_id}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Calibration method: {calibration_method}")
    print(f"CV predictions: {cv_predictions_path.name}")
    print(f"Test predictions: {test_predictions_path.name}")
    print(
        f"\nTest (uncalibrated): Brier={test_uncalibrated_metrics_uniform['brier_score']:.6f}, "
        f"ECE={test_uncalibrated_metrics_uniform['expected_calibration_error']:.6f}"
    )
    print(
        f"Test (calibrated):   Brier={test_calibrated_metrics_uniform['brier_score']:.6f}, "
        f"ECE={test_calibrated_metrics_uniform['expected_calibration_error']:.6f}"
    )
    print("\nOutput files:")
    print(f"  Calibrated predictions: {calibrated_test_path.name}")
    print(f"  Reliability plot data: {reliability_plot_path.name}")
    print(f"  Calibration metadata: {calibration_metadata_path.name}")
    print(f"  Brier score summary: {brier_summary_path.name}")
    print("\nReliability curve plots:")
    for label, path in plot_paths:
        print(f"  {label}: {path.name}")
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)

    return CalibrationRunResult(
        cv_predictions_file=cv_predictions_path,
        test_predictions_file=test_predictions_path,
        calibrated_test_file=calibrated_test_path,
        reliability_plot_json=reliability_plot_path,
        calibration_metadata_json=calibration_metadata_path,
        brier_summary_txt=brier_summary_path,
        figures_dir=figures_dir,
        cv_uncalibrated_metrics=cv_uncalibrated_metrics_uniform,
        cv_calibrated_metrics=cv_calibrated_metrics_uniform,
        test_uncalibrated_metrics=test_uncalibrated_metrics_uniform,
        test_calibrated_metrics=test_calibrated_metrics_uniform,
    )


__all__ = ["CalibrationRunResult", "run_calibration"]

