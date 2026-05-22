#!/usr/bin/env python3
"""W4: Stage 2 feature-group ablation driver (South America)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.regions.shared.training.stage2_ablation_groups import ABLATION_GROUPS  # noqa: E402

OUT_DIR = _ROOT / "outputs" / "south_america" / "results" / "ablation"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: dict = {"timestamp": ts, "groups": {}}

    train_script = _ROOT / "scripts/regions/south_america/5_training/model1_LGBM_stage2"
    for group_name in ABLATION_GROUPS:
        env = {**__import__("os").environ, "STAGE2_ABLATION_DROP": group_name}
        print(f"\n=== Ablation: drop {group_name} ===")
        try:
            subprocess.run(
                [sys.executable, str(train_script), "--ablation-drop", group_name],
                check=True,
                env=env,
                cwd=str(_ROOT),
            )
            results["groups"][group_name] = {"status": "submitted"}
        except subprocess.CalledProcessError as exc:
            results["groups"][group_name] = {"status": "failed", "error": str(exc)}

    out_path = OUT_DIR / f"stage2_ablation_manifest_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nAblation manifest: {out_path}")
    print(
        "Note: implement --ablation-drop in model1_LGBM_stage2 to drop feature "
        "groups by prefix; until then run manual feature subsets."
    )


if __name__ == "__main__":
    main()
