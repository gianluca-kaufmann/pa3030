"""Country-year panel aggregation for Stage 1 models."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.regions.shared.country_raster import country_ids_for_rows, load_country_raster

BATCH_SIZE = 500_000
LAG_YEARS = (1, 2, 3)

# Expected sub-path for WDPA CSVs relative to repo root.
# Polygon file (0) is the main source; point files (1,2) supplement.
WDPA_CSV_RELPATH_TEMPLATE = "data/shared/wdpa/WDPA_WDOECM_Public_{n}.csv"

# WDPA STATUS values accepted as "designated".
WDPA_DESIGNATED_STATUSES = {"Designated", "Established", "Inscribed"}

# Path to the full V-Dem CY Core dataset (relative to repo root).
# Contains v2eltype_0 (legislative) and v2eltype_6 (presidential) indicators.
# Source: Coppedge et al. (2024) V-Dem v15, https://doi.org/10.23696/vdemds24
_VDEM_CORE_RELPATH = "data/shared/VDem/V-Dem-CY-Core-v15.csv"


def compute_years_to_next_election(
    countries: list[str],
    year_range: tuple[int, int] = (2001, 2024),
    vdem_path: Path | None = None,
) -> pd.DataFrame:
    """Derive years_to_next_election from V-Dem v15 CY Core.

    Election-year rule:
      - Countries that have direct presidential/executive elections (v2eltype_6):
        use those years.
      - Countries with only legislative elections (v2eltype_0): use those as a
        proxy for executive accountability pressure.
      - Countries absent from V-Dem (e.g., Brunei — absolute monarchy): NaN.

    Returns DataFrame with columns: iso3, year, years_to_next_election
      0 = election in this year, N = N years until next, NaN = none in window.
    """
    if vdem_path is None:
        _root = Path(__file__).resolve().parents[3]
        vdem_path = _root / _VDEM_CORE_RELPATH

    if not vdem_path.exists():
        raise FileNotFoundError(f"V-Dem CY Core not found: {vdem_path}")

    df = pd.read_csv(
        vdem_path,
        usecols=["country_text_id", "year", "v2eltype_0", "v2eltype_6"],
    )
    y_min, y_max = year_range
    df = df[df["country_text_id"].isin(countries) & df["year"].between(y_min, y_max)]

    rows: list[dict] = []
    for country in countries:
        c = df[df["country_text_id"] == country].sort_values("year")
        if c.empty:
            for yr in range(y_min, y_max + 1):
                rows.append({"iso3": country, "year": yr, "years_to_next_election": np.nan})
            continue

        pres = c.loc[c["v2eltype_6"] == 1, "year"].values.astype(int)
        leg = c.loc[c["v2eltype_0"] == 1, "year"].values.astype(int)
        election_years = pres if len(pres) > 0 else leg

        for yr in range(y_min, y_max + 1):
            future = election_years[election_years >= yr]
            rows.append({
                "iso3": country,
                "year": yr,
                "years_to_next_election": float(future[0] - yr) if len(future) > 0 else np.nan,
            })

    result = pd.DataFrame(rows)
    result["years_to_next_election"] = result["years_to_next_election"].astype("float32")
    return result.sort_values(["iso3", "year"]).reset_index(drop=True)


def compute_pre2001_expansion(
    wdpa_csv_paths: Union[Path, list[Path]],
    iso3_to_id: dict[str, int],
    year_range: tuple[int, int] = (1990, 2000),
    status_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Compute country-year PA expansion (km²) from WDPA STATUS_YR.

    Used to extend Stage 1 panels back to pre-2001, improving training-set size
    for the Poisson GLM.  The WDPA CSV download from ProtectedPlanet.net (~250 MB)
    contains one polygon file and two point files; pass all three paths to maximise
    coverage.

    Args:
        wdpa_csv_paths: Path or list of paths to extracted WDPA CSV files.
            Typically ``data/shared/wdpa/WDPA_WDOECM_Public_{0,1,2}.csv``.
            Files are expected to have columns ISO3, STATUS, STATUS_YR, GIS_AREA
            (and REP_AREA as fallback for point records with GIS_AREA = 0).
        iso3_to_id: Region-specific mapping from ISO3 → integer country_id,
            matching the country raster encoding used in the feature parquet.
        year_range: Inclusive (start, end) years to extract.  Typically (1990, 2000)
            or (1996, 2000) depending on how far political covariates extend back.
        status_filter: Set of WDPA STATUS values to accept as "designated".
            Defaults to {"Designated", "Established", "Inscribed"}.

    Returns:
        DataFrame with columns [country_id, year, pa_expansion_pixels] where
        ``pa_expansion_pixels`` is the total GIS_AREA (km²) designated in that
        country-year.  At 1 km² pixel resolution this is approximately equal to
        the pixel-count used in the post-2001 panel.
    """
    if status_filter is None:
        status_filter = WDPA_DESIGNATED_STATUSES
    if isinstance(wdpa_csv_paths, Path):
        wdpa_csv_paths = [wdpa_csv_paths]

    needed_cols = ["ISO3", "STATUS", "STATUS_YR", "GIS_AREA"]
    optional_cols = ["REP_AREA"]

    parts: list[pd.DataFrame] = []
    for path in wdpa_csv_paths:
        if not path.exists():
            print(f"  WARN [compute_pre2001_expansion]: {path} not found — skipped")
            continue
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c in needed_cols + optional_cols,
                low_memory=True,
                encoding="latin-1",
            )
        except Exception as exc:
            print(f"  WARN [compute_pre2001_expansion]: could not read {path}: {exc}")
            continue
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["country_id", "year", "pa_expansion_pixels"])

    raw = pd.concat(parts, ignore_index=True)

    # Resolve area: prefer GIS_AREA; fall back to REP_AREA for point records.
    if "REP_AREA" in raw.columns:
        raw["area_km2"] = np.where(
            raw["GIS_AREA"].fillna(0) > 0,
            raw["GIS_AREA"].fillna(0),
            raw["REP_AREA"].fillna(0),
        )
    else:
        raw["area_km2"] = raw["GIS_AREA"].fillna(0)

    # Filter to designated PAs with known STATUS_YR and relevant countries/years.
    y_min, y_max = year_range
    countries = set(iso3_to_id.keys())
    mask = (
        raw["STATUS"].isin(status_filter)
        & (raw["STATUS_YR"] > 0)
        & raw["ISO3"].isin(countries)
        & raw["STATUS_YR"].between(y_min, y_max)
    )
    filtered = raw.loc[mask, ["ISO3", "STATUS_YR", "area_km2"]].copy()

    if filtered.empty:
        print(
            f"  WARN [compute_pre2001_expansion]: no rows matched for "
            f"countries={sorted(countries)}, years={year_range}"
        )
        return pd.DataFrame(columns=["country_id", "year", "pa_expansion_pixels"])

    grouped = (
        filtered.groupby(["ISO3", "STATUS_YR"], as_index=False)["area_km2"]
        .sum()
        .rename(columns={"ISO3": "iso3", "STATUS_YR": "year", "area_km2": "pa_expansion_pixels"})
    )
    grouped["country_id"] = grouped["iso3"].map(iso3_to_id)
    grouped = grouped.dropna(subset=["country_id"])
    grouped["country_id"] = grouped["country_id"].astype(int)
    grouped["pa_expansion_pixels"] = grouped["pa_expansion_pixels"].astype(np.float64)

    return (
        grouped[["country_id", "year", "pa_expansion_pixels"]]
        .sort_values(["country_id", "year"])
        .reset_index(drop=True)
    )


def extend_panel_with_wdpa(
    cy: pd.DataFrame,
    wdpa_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Prepend WDPA-based pre-panel rows to an existing country-year panel and
    recompute all momentum lags and the cumulative saturation feature.

    ``cy`` must be the direct output of ``build_country_year_panel()`` (columns:
    country_id, year, pa_expansion_pixels, pa_momentum_pixels_lag{1,2,3},
    pa_cumsum_lag1_pixels).

    ``wdpa_rows`` is the output of ``compute_pre2001_expansion()`` (columns:
    country_id, year, pa_expansion_pixels).  Rows with the same (country_id, year)
    as existing rows in ``cy`` are dropped (existing data takes precedence).
    """
    if wdpa_rows.empty:
        return cy

    # Drop any WDPA rows that overlap with the existing panel.
    existing_keys = set(zip(cy["country_id"], cy["year"]))
    mask = ~wdpa_rows.apply(
        lambda r: (int(r["country_id"]), int(r["year"])) in existing_keys, axis=1
    )
    new_rows = wdpa_rows.loc[mask, ["country_id", "year", "pa_expansion_pixels"]].copy()
    if new_rows.empty:
        return cy

    # Identify derived columns that will be recomputed vs. extra covariates to preserve.
    derived_cols = (
        [f"pa_momentum_pixels_lag{lag}" for lag in LAG_YEARS]
        + ["pa_cumsum_lag1_pixels"]
    )
    base_cols = ["country_id", "year", "pa_expansion_pixels"]
    extra_cols = [c for c in cy.columns if c not in base_cols + derived_cols]

    # Build combined timeline using only base columns (pre-2001 rows have no extra cols).
    combined = (
        pd.concat([new_rows[base_cols], cy[base_cols]], ignore_index=True)
        .sort_values(["country_id", "year"])
        .reset_index(drop=True)
    )

    # Recompute momentum lags and cumulative saturation on the full timeline.
    for lag in LAG_YEARS:
        combined[f"pa_momentum_pixels_lag{lag}"] = (
            combined.groupby("country_id")["pa_expansion_pixels"]
            .shift(lag)
            .fillna(0)
            .astype(np.float64)
        )
    combined["pa_cumsum_lag1_pixels"] = (
        combined.groupby("country_id")["pa_expansion_pixels"]
        .cumsum()
        .shift(1)
        .fillna(0)
        .astype(np.float64)
    )

    # Merge back extra covariates (iso3, political vars, forest_area_pct, …).
    # Pre-2001 rows will have NaN for these; the expansion model fills them with 0
    # (fillna on feature_cols) — acceptable given political data is available via
    # attach_political_covariates() when stage1_data_builder.py is re-run on Euler.
    if extra_cols:
        combined = combined.merge(
            cy[["country_id", "year"] + extra_cols],
            on=["country_id", "year"],
            how="left",
        )

    return combined.reset_index(drop=True)


def build_country_year_panel(
    panel_path: Path,
    region: str,
    year_range: tuple[int, int] = (2001, 2013),
) -> pd.DataFrame:
    """Aggregate transition_01 to country-year; add PA momentum lags 1-3."""
    raster = load_country_raster(region)
    schema_names = pq.ParquetFile(panel_path).schema_arrow.names
    has_country = "country_id" in schema_names

    accum: dict[tuple[int, int], int] = {}
    cols = ["row", "col", "year", "transition_01"]
    if has_country:
        cols.append("country_id")

    pf = pq.ParquetFile(panel_path)
    y_min, y_max = year_range
    for batch in pf.iter_batches(columns=cols, batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        if df.empty:
            continue
        df = df[(df["year"] >= y_min) & (df["year"] <= y_max)]
        if df.empty:
            continue
        if not has_country:
            df["country_id"] = country_ids_for_rows(df, raster)
        grouped = df.groupby(["country_id", "year"], as_index=False)["transition_01"].sum()
        for _, row in grouped.iterrows():
            cid = int(row["country_id"])
            if cid == 0:
                continue
            key = (cid, int(row["year"]))
            accum[key] = accum.get(key, 0) + int(row["transition_01"])

    if not accum:
        raise ValueError("No country-year rows aggregated.")

    rows = [
        {"country_id": k[0], "year": k[1], "pa_expansion_pixels": v}
        for k, v in accum.items()
    ]
    cy = pd.DataFrame(rows).sort_values(["country_id", "year"]).reset_index(drop=True)
    for lag in LAG_YEARS:
        cy[f"pa_momentum_pixels_lag{lag}"] = (
            cy.groupby("country_id")["pa_expansion_pixels"]
            .shift(lag)
            .fillna(0)
            .astype(np.float64)
        )
    # Cumulative PA expansion since panel start — proxy for frontier saturation.
    # Captures how much of the "easy" land a country has already designated.
    # Distinct from momentum lags (recent rate) — this is the running total.
    cy["pa_cumsum_lag1_pixels"] = (
        cy.groupby("country_id")["pa_expansion_pixels"]
        .cumsum()
        .shift(1)
        .fillna(0)
        .astype(np.float64)
    )
    return cy
