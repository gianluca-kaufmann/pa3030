#!/usr/bin/env python3
"""Stage: Forward pathway analysis — annual coverage trajectories and milestone tables.

Reads coverage_baseline.json + any available WDPA annual rasters to reconstruct
the historical PA coverage series, then projects three scenarios forward to 2030:

  BAU:      historical 5-year designation rate (2019→2024) applied annually
  Moderate: region-adaptive midpoint between current coverage and 30%
  30×30:    rate required to reach 30% by 2030

Outputs (written to outputs/{region}/results/forward/pathway/)
--------------------------------------------------------------
  pathway_coverage.png/.pdf     — annual coverage trajectory figure
  pathway_milestone_table.csv   — per-region milestone summary
  pathway_milestone_table.tex   — LaTeX version
  pathway_country_table.csv     — per-country on-track/behind table
  pathway_country_table.tex     — LaTeX version
  pathway_summary.json          — structured numbers for paper tables

Run after:
  1_forward_coverage_baseline.py   (produces coverage_baseline.json)
  4_forward_results.py             (produces country_breakdown.csv)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
del _repo_root
# ─────────────────────────────────────────────────────────────────────────────

from scripts.regions.shared.forward.config import (  # noqa: E402
    DATA_SUBDIR,
    ISO_CODES,
    MODEL_PREFIX,
    OUTPUTS_SUBDIR,
    REGION_LABEL,
    WDPA_2024_FILENAME,
    forward_dir_search_paths,
    get_repo_root,
)

# ── Styling ───────────────────────────────────────────────────────────────────
FIGURE_DPI = 300
SCENARIO_COLORS = {
    "historical": "#333333",
    "bau":        "#d62728",
    "moderate":   "#e87e2b",
    "30x30":      "#1a78c2",
}
TARGET_30PCT = 0.30


# =============================================================================
# Historical series
# =============================================================================

def _compute_raster_coverage(
    backbone_path: Path,
    wdpa_path: Path,
) -> float:
    """Return fraction of land pixels protected according to wdpa_path."""
    import rasterio
    from scripts.regions.shared.geo_utils import pixel_area_km2

    with rasterio.open(backbone_path) as src:
        backbone = src.read(1)
        nodata = src.nodata
        transform = src.transform
        pixel_size_m = abs(transform.a)

    valid_mask = (backbone == 1) if nodata is not None else (backbone > 0)
    rows_all, _ = np.where(valid_mask)
    y_all = transform.f + (rows_all + 0.5) * transform.e
    total_km2 = float(pixel_area_km2(y_all, pixel_size_m=pixel_size_m).sum())

    with rasterio.open(wdpa_path) as src_w:
        wdpa_data = src_w.read(1)

    h = min(valid_mask.shape[0], wdpa_data.shape[0])
    w = min(valid_mask.shape[1], wdpa_data.shape[1])
    protected_mask = valid_mask[:h, :w] & (wdpa_data[:h, :w] == 1)
    prot_rows, _ = np.where(protected_mask)
    y_prot = transform.f + (prot_rows + 0.5) * transform.e
    prot_km2 = float(pixel_area_km2(y_prot, pixel_size_m=pixel_size_m).sum())

    return prot_km2 / max(total_km2, 1.0)


def discover_wdpa_annual_points(
    repo_root: Path,
    data_subdir: str,
    wdpa_2024_filename: str,
) -> Dict[int, float]:
    """Scan WDPA data directory for annual rasters and compute coverage per year.

    Returns {year: coverage_fraction} for every raster found beyond the 2024
    point already in coverage_baseline.json.
    """
    wdpa_dir = repo_root / f"data/{data_subdir}/ready/WDPA"
    scratch = os.environ.get("SCRATCH", "")
    if scratch:
        scratch_wdpa = Path(scratch) / f"data/{data_subdir}/ready/WDPA"
        if scratch_wdpa.exists():
            wdpa_dir = scratch_wdpa

    if not wdpa_dir.exists():
        return {}

    # Derive filename stem e.g. "WDPA_SA_1km_2024.tif" → "WDPA_SA_1km"
    stem = wdpa_2024_filename.replace("_2024.tif", "").replace("_2024.TIF", "")
    tif_files = list(wdpa_dir.glob(f"{stem}_*.tif")) + list(wdpa_dir.glob(f"{stem}_*.TIF"))

    if not tif_files:
        return {}

    # Same resolution order as results_core.resolve_backbone_path_for_plot
    backbone_candidates: List[Path] = []
    if scratch:
        backbone_candidates += [
            Path(scratch) / f"data/{data_subdir}/ready/backbone.tif",
            Path(scratch) / f"data/{data_subdir}/ready/backbone/backbone.tif",
        ]
    backbone_candidates += [
        repo_root / f"data/{data_subdir}/ready/backbone.tif",
        repo_root / f"data/{data_subdir}/ready/backbone/backbone.tif",
    ]
    backbone_path = next((p for p in backbone_candidates if p.exists()), None)
    if backbone_path is None:
        print("  WARNING: backbone.tif not found — cannot compute historical coverage from rasters.")
        return {}

    points: Dict[int, float] = {}
    for tif in sorted(tif_files):
        # Extract year from filename e.g. "WDPA_SA_1km_2019.tif" → 2019
        name = tif.stem  # "WDPA_SA_1km_2019"
        try:
            year = int(name.split("_")[-1])
        except ValueError:
            continue
        if year == 2024:
            continue  # already in baseline
        try:
            cov = _compute_raster_coverage(backbone_path, tif)
            points[year] = cov
            print(f"  Raster coverage {year}: {cov:.2%}")
        except Exception as e:
            print(f"  WARNING: Could not compute coverage from {tif.name}: {e}")

    return points


def build_historical_series(
    baseline: Dict[str, Any],
    extra_raster_points: Dict[int, float],
    start_year: int = 2000,
    end_year: int = 2024,
) -> Dict[int, float]:
    """Combine all known data points and interpolate linearly between them.

    Known sources (merged in priority order):
      1. coverage_baseline.json: 2024 coverage
      2. bau_km2: implies 2019 coverage as protected_2024 - bau_5yr
      3. extra_raster_points: any additional years from WDPA rasters
    """
    coverage_2024 = baseline["coverage_pct_2024"]
    total_km2 = baseline["total_land_km2"]
    prot_2024_km2 = baseline["protected_2024_km2"]

    known: Dict[int, float] = {2024: coverage_2024}

    # Reconstruct 2019 from BAU km² if available
    bau_km2 = baseline.get("bau_km2")
    if bau_km2 is not None and bau_km2 > 0:
        prot_2019_km2 = prot_2024_km2 - bau_km2
        known[2019] = prot_2019_km2 / total_km2

    # Merge raster-derived points (override bau-derived 2019 if raster is available)
    known.update(extra_raster_points)

    # Interpolate to annual resolution
    sorted_years = sorted(known)
    result: Dict[int, float] = {}
    for i, yr in enumerate(sorted_years):
        result[yr] = known[yr]
        if i < len(sorted_years) - 1:
            yr_next = sorted_years[i + 1]
            for fill_yr in range(yr + 1, yr_next):
                t = (fill_yr - yr) / (yr_next - yr)
                result[fill_yr] = known[yr] + t * (known[yr_next] - known[yr])

    # Clip to requested range; extrapolate before earliest known point as flat
    earliest = min(sorted_years)
    for yr in range(start_year, end_year + 1):
        if yr not in result:
            result[yr] = result[earliest] if yr < earliest else result.get(end_year, coverage_2024)

    return {yr: result[yr] for yr in range(start_year, end_year + 1) if yr in result}


# =============================================================================
# Scenario projections
# =============================================================================

def build_scenario_series(
    baseline: Dict[str, Any],
    start_year: int = 2024,
    end_year: int = 2032,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Return (years, bau, moderate, full_30x30) coverage fractions.

    BAU:      historical annual rate (bau_km2 / 5) applied linearly
    Moderate: linear interpolation from current to moderate_target by end_year
    30×30:    linear interpolation from current to 30% by 2030
              (then flat, since 30% is the target)
    """
    cov_2024 = baseline["coverage_pct_2024"]
    total_km2 = baseline["total_land_km2"]
    bau_km2 = baseline.get("bau_km2") or 0.0
    bau_annual_pct = (bau_km2 / 5.0) / total_km2  # fraction per year

    moderate_target = baseline.get("moderate_target_pct", (cov_2024 + TARGET_30PCT) / 2)

    years = list(range(start_year, end_year + 1))
    bau, moderate, full_30x30 = [], [], []

    for yr in years:
        dt = yr - start_year
        bau.append(min(1.0, cov_2024 + bau_annual_pct * dt))

        if yr <= 2030:
            t = dt / (2030 - start_year) if 2030 > start_year else 1.0
            moderate.append(cov_2024 + t * (moderate_target - cov_2024))
            full_30x30.append(cov_2024 + t * (TARGET_30PCT - cov_2024))
        else:
            # Extrapolate moderate at its 2030 rate; 30×30 is flat after 2030
            rate_mod = (moderate_target - cov_2024) / (2030 - start_year)
            moderate.append(min(1.0, moderate_target + rate_mod * (yr - 2030)))
            full_30x30.append(TARGET_30PCT)

    return years, bau, moderate, full_30x30


def _bau_year_30pct(
    baseline: Dict[str, Any],
    max_year: int = 2200,
) -> Optional[float]:
    """Return the fractional year when BAU hits 30%, or None if > max_year."""
    cov_2024 = baseline["coverage_pct_2024"]
    if cov_2024 >= TARGET_30PCT:
        return 2024.0
    total_km2 = baseline["total_land_km2"]
    bau_km2 = baseline.get("bau_km2") or 0.0
    bau_annual_km2 = bau_km2 / 5.0
    if bau_annual_km2 <= 0:
        return None
    km2_needed = (TARGET_30PCT - cov_2024) * total_km2
    years_needed = km2_needed / bau_annual_km2
    target_year = 2024.0 + years_needed
    return target_year if target_year <= max_year else None


# =============================================================================
# Figure
# =============================================================================

def plot_pathway_coverage(
    region_label: str,
    historical: Dict[int, float],
    fwd_years: List[int],
    bau: List[float],
    moderate: List[float],
    full_30x30: List[float],
    baseline: Dict[str, Any],
    output_path: Path,
) -> None:
    """Annual PA coverage trajectory figure."""
    print(f"\n  Plotting pathway coverage figure…")

    bau_year = _bau_year_30pct(baseline)
    cov_2024 = baseline["coverage_pct_2024"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Historical line
    hist_yrs = sorted(historical)
    hist_cov = [historical[y] * 100 for y in hist_yrs]
    ax.plot(hist_yrs, hist_cov, color=SCENARIO_COLORS["historical"],
            lw=2.2, zorder=4, label="Historical (observed)")
    # Mark data anchor points (non-interpolated)
    ax.scatter(hist_yrs[0], hist_cov[0], color=SCENARIO_COLORS["historical"],
               s=40, zorder=5)
    ax.scatter(hist_yrs[-1], hist_cov[-1], color=SCENARIO_COLORS["historical"],
               s=40, zorder=5)

    # Scenario lines (2024 onward)
    ax.plot(fwd_years, [v * 100 for v in bau],
            color=SCENARIO_COLORS["bau"], lw=2.0, ls="--", zorder=4,
            label="BAU (historical rate continued)")
    ax.plot(fwd_years, [v * 100 for v in moderate],
            color=SCENARIO_COLORS["moderate"], lw=2.0, ls="--", zorder=4,
            label=f"Moderate (→{baseline.get('moderate_target_pct', 0.278):.0%})")
    ax.plot(fwd_years, [v * 100 for v in full_30x30],
            color=SCENARIO_COLORS["30x30"], lw=2.0, ls="--", zorder=4,
            label="30×30 (→30% by 2030)")

    # 30% threshold
    ax.axhline(30.0, color="#2c7bb6", lw=1.0, ls=":", zorder=2, alpha=0.7)
    ax.text(hist_yrs[0], 30.3, "30% target", fontsize=8.5,
            color="#2c7bb6", va="bottom")

    # Vertical line at 2024 (placed after data so ylim is set)
    ax.axvline(2024, color="#999999", lw=0.8, ls=":", zorder=2)
    y_lo, _ = ax.get_ylim()
    ax.text(2024.2, y_lo + 0.3, "2024", fontsize=8.5, color="#666666", va="bottom")

    # Annotate BAU milestone only if it falls within the plot x-range
    x_lo, x_hi = ax.get_xlim()
    if bau_year is not None and x_lo <= bau_year <= x_hi:
        ax.axvline(bau_year, color=SCENARIO_COLORS["bau"], lw=0.8, ls=":", alpha=0.6, zorder=2)
        ax.annotate(
            f"BAU: {int(round(bau_year))}",
            xy=(bau_year, 30.0), xytext=(bau_year + 0.3, 27.5),
            fontsize=8, color=SCENARIO_COLORS["bau"],
            arrowprops=dict(arrowstyle="->", color=SCENARIO_COLORS["bau"], lw=0.8),
        )
    elif bau_year is not None:
        # Add a text note in the legend area instead
        ax.text(0.98, 0.05, f"BAU reaches 30%: {int(round(bau_year))}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=SCENARIO_COLORS["bau"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("PA coverage (% of land area)", fontsize=12)
    ax.set_title(f"{region_label} — Protected Area Coverage Trajectory\n"
                 "Historical and three scenarios under the 30×30 target",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, framealpha=0.8, loc="upper left")
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        p = output_path.with_suffix(ext)
        fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {p.name}")
    plt.close(fig)


# =============================================================================
# Milestone and country tables
# =============================================================================

def _fmt_year(y: Optional[float]) -> str:
    if y is None:
        return ">2200"
    if y <= 2024:
        return "already met"
    return f"{int(round(y))}"


def _fmt_behind(y: Optional[float]) -> str:
    if y is None:
        return ">170"
    if y <= 2024:
        return "0 (met)"
    behind = int(round(y)) - 2030
    return f"+{behind}" if behind > 0 else str(behind)


def build_region_milestone_table(baseline: Dict[str, Any], region_label: str) -> pd.DataFrame:
    """Single-row DataFrame summarising region-level milestone numbers."""
    cov_2024 = baseline["coverage_pct_2024"]
    total_km2 = baseline["total_land_km2"]
    bau_km2 = baseline.get("bau_km2") or 0.0
    bau_annual_pct = (bau_km2 / 5.0) / total_km2 * 100  # %/yr
    bau_year = _bau_year_30pct(baseline)
    mod_target = baseline.get("moderate_target_pct", (cov_2024 + TARGET_30PCT) / 2)
    req_rate = max(0.0, (TARGET_30PCT - cov_2024) / 6.0) * 100  # %/yr over 6 years

    return pd.DataFrame([{
        "region":                    region_label,
        "current_pct_2024":          round(cov_2024 * 100, 1),
        "bau_annual_rate_pct_yr":    round(bau_annual_pct, 3),
        "moderate_target_pct":       round(mod_target * 100, 1),
        "projected_30pct_year_bau":  _fmt_year(bau_year),
        "years_behind_2030_bau":     _fmt_behind(bau_year),
        "required_rate_30x30_pct_yr": round(req_rate, 2),
    }])


def build_country_milestone_table(country_breakdown_path: Path) -> Optional[pd.DataFrame]:
    """Per-country on-track/behind table from forward country_breakdown.csv."""
    if not country_breakdown_path.exists():
        print(f"  country_breakdown.csv not found: {country_breakdown_path}")
        return None

    df = pd.read_csv(country_breakdown_path)
    required_cols = {"iso_a3", "country", "total_km2", "protected_2024_km2",
                     "current_pct_protected", "bau_new_km2", "gap_to_30pct_km2"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"  country_breakdown.csv missing columns: {missing}")
        return None

    rows = []
    for _, r in df.iterrows():
        cov = float(r["current_pct_protected"])
        total = float(r["total_km2"])
        bau_km2 = float(r.get("bau_new_km2", 0))  # 6-year total in forward results
        bau_annual_km2 = bau_km2 / 6.0
        bau_annual_pct = bau_annual_km2 / max(total, 1.0) * 100
        gap_km2 = float(r.get("gap_to_30pct_km2", max(0.0, 0.30 * total - float(r["protected_2024_km2"]))))

        if gap_km2 <= 0:
            year_30 = "already met"
            behind = "0 (met)"
        elif bau_annual_km2 > 0:
            yrs_needed = gap_km2 / bau_annual_km2
            yr = 2024.0 + yrs_needed
            year_30 = f"{int(round(yr))}"
            behind_val = int(round(yr)) - 2030
            behind = f"+{behind_val}" if behind_val > 0 else str(behind_val)
        else:
            year_30 = ">2200"
            behind = ">170"

        req_rate = max(0.0, gap_km2 / (6.0 * max(total, 1.0))) * 100

        rows.append({
            "iso_a3":                    r["iso_a3"],
            "country":                   r["country"],
            "current_pct_2024":          round(cov * 100, 1),
            "bau_annual_rate_pct_yr":    round(bau_annual_pct, 3),
            "projected_30pct_year_bau":  year_30,
            "years_behind_2030_bau":     behind,
            "required_rate_30x30_pct_yr": round(req_rate, 2),
        })

    return pd.DataFrame(rows)


def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Minimal LaTeX tabular from DataFrame."""
    cols = list(df.columns)
    header = " & ".join(c.replace("_", r"\_") for c in cols) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def save_tables(
    region_table: pd.DataFrame,
    country_table: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write CSV and LaTeX for milestone and country tables."""
    # Region milestone
    region_table.to_csv(output_dir / "pathway_milestone_table.csv", index=False)
    latex = _df_to_latex(
        region_table,
        caption="Region-level 30×30 milestone summary.",
        label="tab:pathway_milestone",
    )
    (output_dir / "pathway_milestone_table.tex").write_text(latex)
    print(f"  Saved: pathway_milestone_table.csv / .tex")

    # Country table
    if country_table is not None:
        country_table.to_csv(output_dir / "pathway_country_table.csv", index=False)
        latex_c = _df_to_latex(
            country_table,
            caption="Country-level 30×30 milestone summary (BAU scenario).",
            label="tab:pathway_country",
        )
        (output_dir / "pathway_country_table.tex").write_text(latex_c)
        print(f"  Saved: pathway_country_table.csv / .tex")


def save_pathway_summary(
    baseline: Dict[str, Any],
    region_table: pd.DataFrame,
    country_table: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save machine-readable pathway_summary.json for paper tables."""
    summary: Dict[str, Any] = {
        "region": OUTPUTS_SUBDIR,
        "coverage_pct_2024": baseline["coverage_pct_2024"],
        "total_land_km2": baseline["total_land_km2"],
        "protected_2024_km2": baseline["protected_2024_km2"],
        "bau_km2_5yr": baseline.get("bau_km2"),
        "km2_needed_for_30pct": baseline.get("km2_needed_for_30pct"),
        "moderate_target_pct": baseline.get("moderate_target_pct"),
        "region_milestone": region_table.to_dict(orient="records"),
        "country_milestones": country_table.to_dict(orient="records") if country_table is not None else [],
    }
    out_path = output_dir / "pathway_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: pathway_summary.json")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    from scripts.regions.shared.forward.config import (  # noqa: F401
        DATA_SUBDIR, OUTPUTS_SUBDIR, REGION_LABEL, WDPA_2024_FILENAME,
        forward_dir_search_paths, get_repo_root,
    )

    model_type = os.environ.get("PA3030_FORWARD_MODEL_TYPE", "lgbm").strip().lower()

    repo_root    = get_repo_root()
    forward_dirs = forward_dir_search_paths(repo_root, OUTPUTS_SUBDIR)
    output_dir   = forward_dirs[0] / "pathway"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FORWARD PATHWAY ANALYSIS — {REGION_LABEL.upper()}")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")

    # ── Load coverage baseline ────────────────────────────────────────────────
    baseline: Optional[Dict[str, Any]] = None
    for fd in forward_dirs:
        for name in ("coverage_baseline.json", "forward_coverage_baseline.json"):
            p = fd / name
            if p.exists():
                with open(p) as f:
                    baseline = json.load(f)
                print(f"  Loaded baseline: {p}")
                break
        if baseline:
            break
    if baseline is None:
        raise FileNotFoundError(
            "coverage_baseline.json not found. Run 1_forward_coverage_baseline.py first."
        )

    # ── Discover additional WDPA annual rasters ───────────────────────────────
    print("\n  Scanning for WDPA annual rasters…")
    extra_points = discover_wdpa_annual_points(repo_root, DATA_SUBDIR, WDPA_2024_FILENAME)
    if extra_points:
        print(f"  Found {len(extra_points)} additional year(s): {sorted(extra_points)}")
    else:
        print("  None found beyond 2024 — using baseline + bau_km2 for historical series.")

    # ── Build historical series ───────────────────────────────────────────────
    historical = build_historical_series(baseline, extra_points)
    print(f"\n  Historical series: {min(historical)}–{max(historical)} "
          f"({len(historical)} points), "
          f"{historical[min(historical)]:.1%} → {historical[2024]:.1%}")

    # ── Build forward scenarios ───────────────────────────────────────────────
    fwd_years, bau, moderate, full_30x30 = build_scenario_series(baseline, 2024, 2032)
    bau_year = _bau_year_30pct(baseline)
    bau_rate_pct_yr = (baseline.get("bau_km2", 0) / 5.0) / baseline["total_land_km2"] * 100
    print(f"\n  BAU annual rate:   {bau_rate_pct_yr:.4f}%/yr")
    print(f"  BAU → 30% year:    {_fmt_year(bau_year)} ({_fmt_behind(bau_year)} years behind 2030)")
    print(f"  Moderate target:   {baseline.get('moderate_target_pct', 0):.1%}")

    # ── Figure ────────────────────────────────────────────────────────────────
    plot_pathway_coverage(
        REGION_LABEL, historical, fwd_years, bau, moderate, full_30x30,
        baseline, output_dir / "pathway_coverage",
    )

    # ── Tables ────────────────────────────────────────────────────────────────
    region_table = build_region_milestone_table(baseline, REGION_LABEL)
    print("\n  Region milestone table:")
    print(region_table.to_string(index=False))

    # Load country breakdown (produced by 4_forward_results.py)
    country_table: Optional[pd.DataFrame] = None
    for fd in forward_dirs:
        cb_path = fd / model_type / "country_breakdown.csv"
        if cb_path.exists():
            country_table = build_country_milestone_table(cb_path)
            break

    if country_table is not None:
        print(f"\n  Country milestone table ({len(country_table)} countries):")
        print(country_table.to_string(index=False))
    else:
        print("\n  country_breakdown.csv not found — skipping country table.")
        print("  Run 4_forward_results.py first.")

    save_tables(region_table, country_table, output_dir)
    save_pathway_summary(baseline, region_table, country_table, output_dir)

    print("\n" + "=" * 70)
    print("PATHWAY ANALYSIS COMPLETE")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
