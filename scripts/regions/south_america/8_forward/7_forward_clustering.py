#!/usr/bin/env python3
"""Stage 7: South America forward clustering — thin wrapper.

Runs spatial DBSCAN clustering on the top-1% risk zone of the SA forward
predictions, producing cluster_summary.csv/.tex, cluster_hotspot_boxes.json,
and cluster_map.png/.pdf.
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
    run_region_forward("south_america", stage="clustering", model_type=model_type)
