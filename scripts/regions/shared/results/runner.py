from __future__ import annotations

import importlib
import os
from pathlib import Path


def _resolve_core_script(region: str) -> Path:
    normalized = region.strip().lower()
    shared_dir = Path(__file__).resolve().parent
    if normalized not in {"south_america", "usa", "se_asia"}:
        raise ValueError("Unsupported region '{}'. Supported: south_america, usa, se_asia".format(region))
    return shared_dir / "results_core.py"


def run_region_results(region: str) -> None:
    normalized = region.strip().lower()
    core_script = _resolve_core_script(region)
    if not core_script.exists():
        raise FileNotFoundError(f"Missing shared results core: {core_script}")
    previous = os.environ.get("PA3030_RESULTS_REGION")
    os.environ["PA3030_RESULTS_REGION"] = normalized
    try:
        # Import as a package module so relative imports in results_core work.
        # Reload after setting PA3030_RESULTS_REGION to refresh region-specific config.
        importlib.invalidate_caches()
        importlib.reload(importlib.import_module("scripts.regions.shared.results.config"))
        core_module = importlib.reload(importlib.import_module("scripts.regions.shared.results.results_core"))
        core_module.main()
    finally:
        if previous is None:
            os.environ.pop("PA3030_RESULTS_REGION", None)
        else:
            os.environ["PA3030_RESULTS_REGION"] = previous
