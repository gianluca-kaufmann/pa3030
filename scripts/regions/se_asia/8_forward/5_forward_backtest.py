#!/usr/bin/env python3
"""Stage 0c: South East Asia pseudo-forecast backtesting (thin wrapper).

Default model type: PA3030_FORWARD_MODEL_TYPE env (else lgbm).
Override: --model-type lgbm|rf
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regions.shared.forward import run_region_forward

if __name__ == "__main__":
    model_type = os.environ.get("PA3030_FORWARD_MODEL_TYPE", "lgbm").strip().lower()
    args = sys.argv[1:]
    if "--model-type" in args:
        idx = args.index("--model-type")
        if idx + 1 < len(args):
            model_type = args[idx + 1]
    run_region_forward("se_asia", stage="backtest", model_type=model_type)
