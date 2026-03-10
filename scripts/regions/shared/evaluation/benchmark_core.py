from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

from .config import ModelEvalConfig
from .dataset_hash import compute_dataset_hash


@dataclass
class BenchmarkResult:
    benchmark: Dict[str, Any]
    metrics_path: Path
    benchmark_path: Path
    dataset_hash: str


def _find_latest_metrics_file(
    config: ModelEvalConfig,
    model_type: Literal["lgbm", "rf"],
) -> Path:
    """
    Find the most recent metrics JSON file for a given region/model/algorithm.

    This unifies the SA `benchmark_1` and USA `benchmark_2` behaviours.
    """
    output_dir = config.metrics_root
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    if model_type == "lgbm":
        pattern = f"{config.model_id}_lgbm_metrics_*.json"
        ts_re = re.compile(rf"{re.escape(config.model_id)}_lgbm_metrics_(\d{{8}}_\d{{6}})\.json")
    elif model_type == "rf":
        pattern = f"{config.model_id}_rf_win5_metrics_*.json"
        ts_re = re.compile(rf"{re.escape(config.model_id)}_rf_win5_metrics_(\d{{8}}_\d{{6}})\.json")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    files = list(output_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No metrics files found in {output_dir} matching pattern {pattern}"
        )

    def key(path: Path):
        m = ts_re.match(path.name)
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                return (dt, 0.0)
            except ValueError:
                pass
        return (datetime.fromtimestamp(0), path.stat().st_mtime)

    return max(files, key=key)


def _load_metrics_file(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _create_benchmark_json(
    metrics_data: Dict[str, Any],
    dataset_hash: str,
    source_metrics_filename: str,
) -> Dict[str, Any]:
    """
    Construct the benchmark JSON structure from a training metrics JSON.

    This closely follows the structure used in the existing region-specific
    benchmark scripts while being region-agnostic.
    """
    metadata = metrics_data.get("metadata", {})
    cv_data = metrics_data.get("cross_validation", {})
    test_perf = metrics_data.get("test_performance", {})
    val_perf = metrics_data.get("validation_performance", {})
    model_params = metrics_data.get("model_parameters", {})
    temporal_split = metrics_data.get("temporal_split", {})
    data_info = metrics_data.get("data", {})

    metadata_timestamp = metadata.get("timestamp")

    benchmark = {
        "schema_version": "1.0.0",
        "benchmark_metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_metrics_file": source_metrics_filename,
            "source_metrics_timestamp": metadata_timestamp if metadata_timestamp else None,
            "model": metadata.get("model", "Unknown"),
            "task": metadata.get("task", "Unknown"),
            "target_column": metadata.get("target_column", "Unknown"),
            "lookahead_years": metadata.get("lookahead_years", 5),
        },
        "dataset": {
            "hash": dataset_hash,
            "temporal_split": temporal_split,
            "data_info": {
                "n_train_full": data_info.get("n_train_full"),
                "n_test": data_info.get("n_test"),
                "train_positives": data_info.get("train_positives"),
                "train_negatives": data_info.get("train_negatives"),
                "test_positives": data_info.get("test_positives"),
                "test_negatives": data_info.get("test_negatives"),
            },
        },
        "model_config": {
            "random_seed": metadata.get("random_state", model_params.get("random_state")),
            "n_features": metadata.get("n_features", 0),
            "features": metadata.get("features", []),
            "hyperparameters": model_params,
        },
    }

    cv_aggregated = cv_data.get("aggregated", {}) or {}
    # Note: we keep the existing note text for now; future work can make this
    # conditional on explicit cv_type metadata if present.
    benchmark["evaluative_metrics"] = {
        "cross_validation": {
            "aggregated": cv_aggregated,
            "note": "Temporal cross-validation across 3 folds"
            if cv_aggregated
            else "Not available",
        },
        "test_performance": test_perf,
    }

    earlystop_clean = {k: v for k, v in val_perf.items() if not k.startswith("precision_at_")}
    earlystop_note = (
        "In-sample earlystop on data used during training (NON-EVALUATIVE)"
        if earlystop_clean
        else "Not reported"
    )
    earlystop_clean["note"] = earlystop_note

    benchmark["non_evaluative_metrics"] = {
        "earlystop_performance": earlystop_clean,
    }

    benchmark["performance_summary"] = {
        "cv_pr_auc_mean": cv_aggregated.get("pr_auc_mean"),
        "cv_pr_auc_std": cv_aggregated.get("pr_auc_std"),
        "cv_roc_auc_mean": cv_aggregated.get("roc_auc_mean"),
        "cv_roc_auc_std": cv_aggregated.get("roc_auc_std"),
        "test_pr_auc": test_perf.get("pr_auc"),
        "test_roc_auc": test_perf.get("roc_auc"),
    }

    return benchmark


def run_benchmark(
    config: ModelEvalConfig,
    model_type: Literal["lgbm", "rf"],
    split_version: str = "main",
) -> BenchmarkResult:
    """
    Core benchmark routine used by all regions/models.
    """
    metrics_path = _find_latest_metrics_file(config, model_type)
    metrics_data = _load_metrics_file(metrics_path)

    dataset_hash = compute_dataset_hash(config, split_version=split_version)

    benchmark_dict = _create_benchmark_json(
        metrics_data=metrics_data,
        dataset_hash=dataset_hash,
        source_metrics_filename=metrics_path.name,
    )

    # Benchmarks are stored under results/ml_models/<split_version> to mirror
    # existing scripts.
    benchmark_output_dir = config.metrics_root / split_version
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_path = benchmark_output_dir / f"{config.model_id}_{model_type}_win5_benchmark_{timestamp}.json"

    with benchmark_path.open("w") as f:
        json.dump(benchmark_dict, f, indent=2)

    return BenchmarkResult(
        benchmark=benchmark_dict,
        metrics_path=metrics_path,
        benchmark_path=benchmark_path,
        dataset_hash=dataset_hash,
    )


__all__ = [
    "BenchmarkResult",
    "run_benchmark",
]

