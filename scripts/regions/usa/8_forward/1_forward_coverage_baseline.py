#!/usr/bin/env python3
"""Stage 0a: USA 2024 coverage baseline (thin wrapper)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.regions.shared.forward import run_region_forward

if __name__ == "__main__":
    run_region_forward("usa", stage="coverage_baseline")
