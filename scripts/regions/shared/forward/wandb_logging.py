#!/usr/bin/env python3
"""Single entry point for Weights & Biases logging in the forward pipeline.

Uses project ``forward`` and respects ``WANDB_MODE`` (default ``offline`` on Euler).
"""

from __future__ import annotations

import os
from typing import Any

FORWARD_WANDB_PROJECT = "forward"


def log_forward_wandb(
    *,
    stage: str,
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Init → log → finish one W&B run; non-fatal on any failure."""
    try:
        import wandb
    except ImportError:
        print("W&B: module not installed — skipping logging.")
        return

    mode = os.environ.get("WANDB_MODE", "offline")
    full_config = {**config, "forward_stage": stage}

    run = None
    try:
        run = wandb.init(
            project=FORWARD_WANDB_PROJECT,
            entity=os.environ.get("WANDB_ENTITY"),
            name=run_name,
            config=full_config,
            mode=mode,
        )
        wandb.log(metrics)
        print("W&B: metrics logged.")
    except Exception as err:
        print(f"W&B logging failed (non-fatal): {err}")
    finally:
        if run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
