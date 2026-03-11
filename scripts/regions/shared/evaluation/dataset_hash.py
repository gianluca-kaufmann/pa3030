from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pyarrow.parquet as pq

from .config import ModelEvalConfig


def _parquet_candidates(
    config: ModelEvalConfig,
    filename: str,
    split_version: str,
) -> Sequence[Path]:
    """
    Build candidate paths for a parquet file for a given region/split, mirroring
    the logic used in the existing region-specific scripts.
    """
    candidates = []

    # Prefer SCRATCH-backed locations if metrics_root / data_root were pointed there
    # when the config was created.
    # We cannot see SCRATCH directly here; we rely on the config's roots.
    # Try results first, then data, both under the configured roots.
    # South America / USA current layout:
    #   outputs/<region>/results/<split_version>/<filename>
    #   data/<region>/ml/<split_version>/<filename>
    #
    # The config.metrics_root is .../results/ml_models; jump one level up to .../results.
    results_root = config.metrics_root.parent
    data_root = config.data_root

    candidates.append(results_root / split_version / filename)
    candidates.append(data_root / split_version / filename)

    return candidates


def resolve_parquet_file(
    config: ModelEvalConfig,
    filename: str,
    split_version: str = "main",
) -> Optional[Path]:
    """
    Locate a parquet file (train/test panel) for a given region + split.

    Returns the first existing candidate path, or None if not found.
    """
    for cand in _parquet_candidates(config, filename, split_version):
        if cand.exists():
            return cand
    return None


def _parquet_metadata_fingerprint(path: Path, num_sample_row_groups: int) -> str:
    """
    Build a deterministic, content-stable string fingerprint from parquet metadata.
    """
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow

    column_names = sorted(field.name for field in schema)
    total_rows = metadata.num_rows
    num_row_groups = metadata.num_row_groups

    num_to_sample = min(num_sample_row_groups, num_row_groups)
    row_group_samples: list[str] = []
    for i in range(num_to_sample):
        rg = metadata.row_group(i)
        # We avoid column-level stats here to keep it cheap but deterministic
        row_group_samples.append(f"rg{i}:rows={rg.num_rows}:cols={rg.num_columns}")

    parts = [
        f"columns={','.join(column_names)}",
        f"total_rows={total_rows}",
        f"num_row_groups={num_row_groups}",
        f"sample_row_groups={';'.join(row_group_samples)}",
    ]
    return "|".join(parts)


def compute_dataset_hash(
    config: ModelEvalConfig,
    split_version: str = "main",
    num_sample_row_groups: int = 3,
    train_filename: str = "train_win5.parquet",
    test_filename: str = "test_win5.parquet",
) -> str:
    """
    Compute a content-stable dataset hash for the train/test panels used in
    model training, using only parquet metadata.

    This unifies the previous region-specific implementations used by
    `benchmark_1` (South America, model1) and `benchmark_2` (USA, model2).

    The hash is based on:
      - sorted column names
      - total row count
      - number of row-groups
      - coarse row-group descriptors for the first N row-groups
    """
    files: Tuple[Tuple[str, str], ...] = (
        ("train", train_filename),
        ("test", test_filename),
    )

    hash_input: list[str] = []

    for role, filename in files:
        file_path = resolve_parquet_file(config, filename, split_version)
        if file_path is None:
            hash_input.append(f"{role}:{filename}:not_found")
            continue

        try:
            fp = _parquet_metadata_fingerprint(file_path, num_sample_row_groups)
            hash_input.append(f"{role}:{filename}:{fp}")
        except Exception as e:  # pragma: no cover - defensive
            hash_input.append(f"{role}:{filename}:parquet_error={str(e)}")

    hash_str = "|".join(hash_input)
    return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()


__all__ = [
    "compute_dataset_hash",
    "resolve_parquet_file",
]

