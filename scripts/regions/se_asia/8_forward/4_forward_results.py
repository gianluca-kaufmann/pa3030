#!/usr/bin/env python3
"""Stage 3: South East Asia forward results — maps, scenarios, breakdowns (thin wrapper).

Passes --model-type lgbm|rf from command line.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regions.shared.forward import run_region_forward

if __name__ == "__main__":
    model_type = "lgbm"
    args = sys.argv[1:]
    if "--model-type" in args:
        idx = args.index("--model-type")
        if idx + 1 < len(args):
            model_type = args[idx + 1]
    run_region_forward("se_asia", stage="results", model_type=model_type)
