from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


def get_repo_root() -> Path:
    """
    Best-effort repository root resolver shared across regions.

    Prefer environment variables, then walk upwards from this file until a
    README.md is found. This mirrors the logic already used in several
    region-specific scripts.
    """
    env_root = (
        os.environ.get("PROJECT_ROOT")
        or os.environ.get("REPO_ROOT")
    )
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
        "Set PROJECT_ROOT or REPO_ROOT, or ensure README.md exists in the repo root."
    )


@dataclass(frozen=True)
class ModelEvalConfig:
    """
    Shared configuration for evaluation / benchmarking across regions and models.

    This is intentionally minimal and focused on filesystem layout and naming
    conventions. Statistical behaviour (e.g. calibration defaults) should live
    in the evaluation logic and accept this config as an input.
    """

    region: Literal["south_america", "usa", "se_asia"]
    model_id: str  # e.g. "model1", "model2"

    # Algorithm identifier used in filenames, e.g. "lgbm" or "rf"
    algorithm: Literal["lgbm", "rf"]

    # Repository root; if not provided, resolved via get_repo_root()
    repo_root: Path

    # Root where model metrics and scored files are written
    metrics_root: Path

    # Root where ML parquet panels live
    data_root: Path

    @property
    def region_results_root(self) -> Path:
        return self.repo_root / f"outputs/{self.region}/results"

    @property
    def ml_models_root(self) -> Path:
        return self.region_results_root / "ml_models"

    @property
    def model_prefix(self) -> str:
        # Filenames are typically e.g. "model1_lgbm_..." or "model2_rf_..."
        return f"{self.model_id}_{self.algorithm}"

    @staticmethod
    def for_region_model(
        region: Literal["south_america", "usa", "se_asia"],
        model_id: str,
        algorithm: Literal["lgbm", "rf"],
        repo_root: Optional[Path] = None,
    ) -> "ModelEvalConfig":
        """
        Convenience constructor that derives standard paths from region/model.
        """
        import os

        if repo_root is None:
            repo_root = get_repo_root()

        scratch_root = os.environ.get("SCRATCH")
        if scratch_root:
            scratch_root_path = Path(scratch_root)
            metrics_root = scratch_root_path / f"outputs/{region}/results/ml_models"
            data_root = scratch_root_path / f"data/{region}/ml"
        else:
            metrics_root = repo_root / f"outputs/{region}/results/ml_models"
            data_root = repo_root / f"data/{region}/ml"

        return ModelEvalConfig(
            region=region,
            model_id=model_id,
            algorithm=algorithm,
            repo_root=repo_root,
            metrics_root=metrics_root,
            data_root=data_root,
        )


__all__ = [
    "ModelEvalConfig",
    "get_repo_root",
]

