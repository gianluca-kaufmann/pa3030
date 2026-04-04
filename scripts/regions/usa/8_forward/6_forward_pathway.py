#!/usr/bin/env python3
"""Stage: USA pathway analysis — coverage trajectories and milestone tables.

Reads coverage_baseline.json + country_breakdown.csv to produce:
  - Annual PA coverage trajectory figure (historical + BAU / moderate / 30x30)
  - Region-level milestone table (when does BAU reach 30%? years behind 2030?)
  - Country-level on-track/behind table

Override model type: --model-type lgbm|rf  (default: lgbm)
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
    run_region_forward("usa", stage="pathway", model_type=model_type)
