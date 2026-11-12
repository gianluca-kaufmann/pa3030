"""Scenario pixel selection: which unprotected pixels each forward scenario designates.

Two allocation modes, set by PA3030_FORWARD_ALLOCATION:

  regional_pool   One area target for the whole region. Pixels are ranked by predicted
                  probability across all countries and selected until the regional gap is
                  covered. Kunming-Montreal Target 3 is a global target, so this is one
                  valid reading, but countries do not each reach 30%.
  national_quota  Each country closes its own gap. Pixels are ranked within the country and
                  selected until that country's gap is covered. Countries already at the
                  target receive nothing.

BAU uses the regional historical 2019-2024 designation volume in both modes, because the
coverage baseline records that volume only at regional level.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

ALLOCATION_MODES = ("regional_pool", "national_quota")
TARGET_PCT = 0.30


def select_by_area(
    proba: np.ndarray,
    area_km2: np.ndarray,
    target_km2,
    group: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mark the highest-probability pixels whose cumulative area first covers the target.

    Args:
        proba (np.ndarray): Predicted designation probability per pixel.
        area_km2 (np.ndarray): True pixel area in km2.
        target_km2 (float or dict): Area to cover. A dict maps each group to its own target.
        group (np.ndarray, optional): Group label per pixel. Ranking runs within each group.

    Returns:
        np.ndarray: Boolean selection, aligned with the inputs. The pixel that crosses the
        target is included. Pixels in a group without a target are never selected.
    """
    order = np.argsort(-proba, kind="stable")
    area_sorted = area_km2[order]
    if group is None:
        area_before = np.cumsum(area_sorted) - area_sorted
        target_sorted = float(target_km2)
    else:
        group_sorted = pd.Series(group[order])
        area_before = pd.Series(area_sorted).groupby(group_sorted.values).cumsum().values - area_sorted
        target_sorted = group_sorted.map(target_km2).fillna(0.0).values
    selected = np.zeros(len(proba), dtype=bool)
    selected[order] = area_before < target_sorted
    return selected


def national_targets(baseline: Dict) -> pd.DataFrame:
    """Per-country km2 still needed for the moderate and the 30% scenario.

    The moderate target is the midpoint between the country's 2024 coverage and 30%,
    the national analogue of the regional moderate target in coverage_core.
    """
    stats = pd.DataFrame(baseline["country_stats"])
    if stats.empty:
        raise ValueError("coverage_baseline.json has no country_stats; rerun the coverage baseline stage.")
    current = stats["protected_2024_km2"] / stats["total_km2"]
    stats["moderate_target_pct"] = (current + TARGET_PCT) / 2
    stats["moderate_needed_km2"] = (stats["moderate_target_pct"] * stats["total_km2"] - stats["protected_2024_km2"]).clip(lower=0.0)
    stats["full_needed_km2"] = (TARGET_PCT * stats["total_km2"] - stats["protected_2024_km2"]).clip(lower=0.0)
    return stats.set_index("iso_a3")


def add_scenario_selections(df: pd.DataFrame, baseline: Dict, allocation: str, proba_col: str) -> pd.DataFrame:
    """Add boolean bau_sel, moderate_sel and full_sel columns to the scored pixel frame.

    national_quota needs an iso_a3 column on df.
    """
    if allocation not in ALLOCATION_MODES:
        raise ValueError(f"Unknown allocation '{allocation}'. Use one of {ALLOCATION_MODES}.")
    proba = df[proba_col].values
    area = df["area_km2"].values
    if not baseline.get("bau_km2"):
        raise ValueError("coverage_baseline.json has no bau_km2 (2019-2024 designation volume); the WDPA 2019 raster is needed.")
    df["bau_sel"] = select_by_area(proba, area, baseline["bau_km2"])

    if allocation == "regional_pool":
        moderate_km2 = baseline.get("km2_needed_for_moderate") or baseline.get("km2_needed_for_25pct", 0.0)
        df["moderate_sel"] = select_by_area(proba, area, moderate_km2)
        df["full_sel"] = select_by_area(proba, area, baseline["km2_needed_for_30pct"])
    else:
        targets = national_targets(baseline)
        iso = df["iso_a3"].values
        df["moderate_sel"] = select_by_area(proba, area, targets["moderate_needed_km2"].to_dict(), group=iso)
        df["full_sel"] = select_by_area(proba, area, targets["full_needed_km2"].to_dict(), group=iso)
    return df


def national_allocation_table(df: pd.DataFrame, baseline: Dict) -> pd.DataFrame:
    """Per-country targets against what the selection delivered, including any shortfall.

    A shortfall above zero means the country has less unprotected land than its gap.
    """
    targets = national_targets(baseline)
    area = df["area_km2"]
    delivered = pd.DataFrame({
        "unprotected_km2": area,
        "moderate_selected_km2": area * df["moderate_sel"],
        "full_selected_km2": area * df["full_sel"],
    }).groupby(df["iso_a3"]).sum()
    table = targets.join(delivered, how="left").fillna({"unprotected_km2": 0.0, "moderate_selected_km2": 0.0, "full_selected_km2": 0.0})
    table["full_shortfall_km2"] = (table["full_needed_km2"] - table["full_selected_km2"]).clip(lower=0.0)
    table["projected_pct_30x30"] = (table["protected_2024_km2"] + table["full_selected_km2"]) / table["total_km2"]
    return table.reset_index()
