from __future__ import annotations

import importlib
import os
from pathlib import Path

_VALID_STAGES = {
    "coverage_baseline",
    "features",
    "predict",
    "results",
    "backtest",
    "pathway",
    "clustering",
}

_STAGE_MODULE = {
    "coverage_baseline": "scripts.regions.shared.forward.coverage_core",
    "features":          "scripts.regions.shared.forward.features_core",
    "predict":           "scripts.regions.shared.forward.predict_core",
    "results":           "scripts.regions.shared.forward.results_core",
    "backtest":          "scripts.regions.shared.forward.backtest_core",
    "pathway":           "scripts.regions.shared.forward.pathway_analysis",
    "clustering":        "scripts.regions.shared.forward.clustering_analysis",
}


def run_region_forward(
    region: str,
    stage: str,
    model_type: str = "lgbm",
) -> None:
    """Run a forward-prediction pipeline stage for the given region.

    Parameters
    ----------
    region:
        "south_america", "usa", or "se_asia"
    stage:
        One of "coverage_baseline", "features", "predict", "results", "backtest"
    model_type:
        "lgbm" (default) or "rf"
    """
    normalized = region.strip().lower()
    if normalized not in {"south_america", "usa", "se_asia"}:
        raise ValueError("Unsupported region '{}'. Supported: south_america, usa, se_asia".format(region))
    stage = stage.strip().lower()
    if stage not in _VALID_STAGES:
        raise ValueError(
            "Unsupported stage '{}'. Supported: {}".format(stage, ", ".join(sorted(_VALID_STAGES)))
        )
    model_type = model_type.strip().lower()
    if model_type not in {"lgbm", "rf"}:
        raise ValueError("Unsupported model_type '{}'. Supported: lgbm, rf".format(model_type))

    prev_region     = os.environ.get("PA3030_FORWARD_REGION")
    prev_stage      = os.environ.get("PA3030_FORWARD_STAGE")
    prev_model_type = os.environ.get("PA3030_FORWARD_MODEL_TYPE")

    os.environ["PA3030_FORWARD_REGION"]     = normalized
    os.environ["PA3030_FORWARD_STAGE"]      = stage
    os.environ["PA3030_FORWARD_MODEL_TYPE"] = model_type

    try:
        importlib.invalidate_caches()
        # Reload config so region-specific values are picked up
        importlib.reload(importlib.import_module("scripts.regions.shared.forward.config"))
        core_module = importlib.reload(importlib.import_module(_STAGE_MODULE[stage]))
        core_module.main()
    finally:
        def _restore(key: str, prev_value):
            if prev_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev_value

        _restore("PA3030_FORWARD_REGION",     prev_region)
        _restore("PA3030_FORWARD_STAGE",      prev_stage)
        _restore("PA3030_FORWARD_MODEL_TYPE", prev_model_type)
