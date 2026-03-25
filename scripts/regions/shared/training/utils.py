"""Shared training utilities.

Pure utility functions shared across all regional training scripts
(model1_LGBM, model1_RF, model2_LGBM, model2_RF, ...).

Import pattern in training scripts
------------------------------------
    _shared = Path(__file__).resolve().parents[3] / "shared"
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))
    del _shared
    from scripts.regions.shared.training.utils import Tee, get_repo_root, ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pyarrow as pa
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


class Tee:
    """Write to both stdout and a file."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)
        self.stdout.flush()
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def get_repo_root() -> Path:
    """Get repository root directory.

    Checks environment variables first, then searches upward for README.md.
    """
    env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    script_dir = Path(__file__).resolve().parent
    current = script_dir
    for _ in range(10):
        if (current / "README.md").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise RuntimeError(
        f"Repository root not found. Searched upward from {script_dir} for README.md.\n"
        "Set PROJECT_ROOT or REPO_ROOT environment variable, or ensure README.md exists in repo root."
    )


def report_memory_usage(label: str = "") -> None:
    """Report current memory usage."""
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / 1024**3
        print(f"  [Memory {label}] RSS: {mem_gb:.2f} GB")
    except ImportError:
        pass  # psutil not available


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(
        obj,
        (
            np.integer,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        try:
            return obj.item() if hasattr(obj, "item") else obj
        except (ValueError, TypeError, AttributeError):
            return str(obj)


def compute_precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float) -> float:
    """Compute precision at top k% of predictions."""
    n_top_k = max(1, int(len(y_true) * k / 100))
    # Use argpartition for efficiency: only need to separate top K from the rest
    # argpartition puts the n_top_k largest elements at the end
    top_k_idx = np.argpartition(y_proba, -n_top_k)[-n_top_k:]
    return y_true[top_k_idx].sum() / n_top_k


def compute_recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float) -> float:
    """Compute recall at top k% of predictions.

    "Of all actual PA designation events in the test set, what fraction fall
    within the top k% of model-predicted risk pixels?"

    This metric operates over the full test period (all years pooled), so the
    denominator is every PA transition event across the entire test window.
    It is policy-relevant: k% defines the share of pixels a planner would
    need to screen to capture this fraction of real future designations.

    A random model achieves recall_at_k ≈ k/100 (e.g. 0.05 at k=5).
    Values substantially above k/100 indicate that the model prioritises
    genuine future PA sites over unprotected background pixels.

    Args:
        y_true:  Binary ground-truth labels (0/1).
        y_proba: Predicted probabilities (higher = more likely to transition).
        k:       Percentage of top predictions to consider (e.g. k=5 → top 5%).

    Returns:
        Fraction of positives captured in top k% predictions.
        Returns 0.0 if no positive labels exist.
    """
    n_pos_total = int(y_true.sum())
    if n_pos_total == 0:
        return 0.0
    n_top_k = max(1, int(len(y_true) * k / 100))
    top_k_idx = np.argpartition(y_proba, -n_top_k)[-n_top_k:]
    return float(y_true[top_k_idx].sum() / n_pos_total)


def extract_features_pyarrow_to_numpy(feature_table: pa.Table, mask: np.ndarray) -> np.ndarray:
    """Extract feature columns from PyArrow table and convert to NumPy float32 array.

    Pure PyArrow→NumPy path (no pandas) for speed.

    Args:
        feature_table: PyArrow table containing feature columns
        mask: Boolean mask to filter rows (applied after conversion)

    Returns:
        Feature matrix (n_filtered, n_features) as float32
    """
    # Preallocate output array to avoid column_stack overhead
    n_rows = int(mask.sum())
    n_features = len(feature_table.column_names)
    X = np.empty((n_rows, n_features), dtype=np.float32)

    # Fill each column directly into preallocated array
    for col_idx, col_name in enumerate(feature_table.column_names):
        col_array = feature_table[col_name].to_numpy(zero_copy_only=False)
        # Apply mask and convert to float32, write directly to preallocated column
        X[:, col_idx] = col_array[mask].astype(np.float32)

    return X


def compute_year_weights(
    years: np.ndarray,
    min_year: int = 2001,
    max_year: int = 2014,
    min_weight: float = 0.5,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Compute sample weights based on year.

    Linear interpolation: min_year -> min_weight, max_year -> max_weight.

    Args:
        years: Array of years
        min_year: Earliest year (gets min_weight)
        max_year: Latest year (gets max_weight)
        min_weight: Weight for min_year (default 0.5)
        max_weight: Weight for max_year (default 1.0)

    Returns:
        Array of weights (float32)
    """
    # Guard against division by zero when min_year == max_year
    if max_year == min_year:
        return np.full(len(years), max_weight, dtype=np.float32)

    # Normalize years to [0, 1] range
    years_normalized = (years - min_year) / (max_year - min_year)
    # Clip to [0, 1] in case of out-of-range years
    years_normalized = np.clip(years_normalized, 0.0, 1.0)
    # Linear interpolation
    weights = min_weight + years_normalized * (max_weight - min_weight)
    return weights.astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute all metrics (used for earlystop/test sets)."""
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline_rate = y_true.mean()
    prec_at_k   = {k: compute_precision_at_k(y_true, y_proba, k) for k in [1, 5, 10]}
    recall_at_k = {k: compute_recall_at_k(y_true, y_proba, k)    for k in [1, 5, 10]}
    # Brier score: clip predictions to [0, 1] first so out-of-distribution transfer
    # scores (e.g. cross-continental) don't produce NaN or values > 1.
    y_proba_clipped = np.clip(y_proba, 0.0, 1.0)
    try:
        brier = float(brier_score_loss(y_true, y_proba_clipped))
    except Exception:
        brier = float("nan")
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_at_1pct":  float(prec_at_k[1]),
        "precision_at_5pct":  float(prec_at_k[5]),
        "precision_at_10pct": float(prec_at_k[10]),
        # Recall at top-k%: fraction of all actual PA events captured when
        # screening the top k% of predictions.  Complementary to precision:
        # precision measures purity of the shortlist, recall measures coverage
        # of real events.  Both are needed to characterise the precision-recall
        # tradeoff at a policy-actionable operating point.
        "recall_at_1pct":  float(recall_at_k[1]),
        "recall_at_5pct":  float(recall_at_k[5]),
        "recall_at_10pct": float(recall_at_k[10]),
        "baseline_rate": float(baseline_rate),
        "lift_at_1pct":  float(prec_at_k[1]  / baseline_rate) if baseline_rate > 0 else 0,
        "lift_at_5pct":  float(prec_at_k[5]  / baseline_rate) if baseline_rate > 0 else 0,
        "lift_at_10pct": float(prec_at_k[10] / baseline_rate) if baseline_rate > 0 else 0,
        "brier_score": brier,
    }


def check_year_overlap(
    train_years: tuple[int, int], eval_years: tuple[int, int], eval_name: str
) -> bool:
    """Check if evaluation years overlap with training years.

    Returns True if there's overlap, False otherwise.
    """
    train_start, train_end = train_years
    eval_start, eval_end = eval_years

    # Check for overlap: eval years should be strictly after training years
    if eval_start <= train_end:
        print(f"\n{'!' * 70}")
        print(
            f"WARNING: {eval_name} years ({eval_start}-{eval_end}) OVERLAP with"
            f" training years ({train_start}-{train_end})!"
        )
        print("  Evaluation years must be strictly after training years to avoid data leakage.")
        print(f"  SKIPPING {eval_name} evaluation for this fold.")
        print(f"{'!' * 70}\n")
        return True
    return False


def wandb_log_one_shot(
    *,
    project: str,
    run_name: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    """Single short W&B run: ``init`` → ``log`` → ``finish``.

    Mirrors ``scripts/regions/south_america/5_training/model1_LGBM`` ``main()``:
    ``wandb.init(project=..., entity=os.environ.get("WANDB_ENTITY"), name=..., config=...)``
    with no ``mode=`` keyword (same env behaviour as the training scripts).
    """
    try:
        import wandb
    except ImportError:
        print("W&B not available (module not installed)")
        return

    started = False
    try:
        wandb.init(
            project=project,
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_name,
            config=dict(config),
        )
        started = True
        wandb.log(dict(metrics))
        print("W&B connected")
    except Exception as err:
        print(f"W&B failed: {err}")
    finally:
        if started:
            try:
                wandb.finish()
            except Exception:
                pass
