from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def get_git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def get_dataset_fingerprint(dataset_path: Path) -> str:
    stat = dataset_path.stat()
    payload = f"{dataset_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_metadata(
    train_path: Path,
    target_col: str,
    feature_count: int,
    year_split: Dict[str, Any],
    timestamp: str,
    mode: str,
    optimizer: str,
    cv_strategy: str,
    repo_root: Path,
    seed: int,
) -> Dict[str, Any]:
    return {
        "dataset_fingerprint": get_dataset_fingerprint(train_path),
        "target_col": target_col,
        "feature_count": int(feature_count),
        "year_split": year_split,
        "git_commit": get_git_commit(repo_root),
        "timestamp": timestamp,
        "tuning_mode": mode,
        "optimizer": optimizer,
        "cv_strategy": cv_strategy,
        "seed": int(seed),
    }


def save_tuning_artifact(payload: Dict[str, Any], canonical_path: Path, timestamped_path: Path) -> None:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    timestamped_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = convert_numpy_types(payload)
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    with open(timestamped_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
