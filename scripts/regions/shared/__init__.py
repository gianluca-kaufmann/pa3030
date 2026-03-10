"""Shared utilities for regional pipelines.

Pipeline order: 1_preprocessing → 2_tuning → 3_training → 4_evaluation → 5_results.
Exposed as preprocessing, tuning, training, evaluation, results for imports
(since package names like 1_preprocessing are not valid Python identifiers).
"""
import importlib
import sys

def _load(name: str, numeric_module: str):
    m = importlib.import_module(numeric_module)
    sys.modules[f"scripts.regions.shared.{name}"] = m
    return m

preprocessing = _load("preprocessing", "scripts.regions.shared.1_preprocessing")
tuning = _load("tuning", "scripts.regions.shared.2_tuning")
training = _load("training", "scripts.regions.shared.3_training")
evaluation = _load("evaluation", "scripts.regions.shared.4_evaluation")
results = _load("results", "scripts.regions.shared.5_results")

__all__ = ["preprocessing", "tuning", "training", "evaluation", "results"]
