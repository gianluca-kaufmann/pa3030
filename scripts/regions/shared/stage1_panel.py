"""Country-year panel aggregation for Stage 1 models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.regions.shared.country_raster import country_ids_for_rows, load_country_raster

BATCH_SIZE = 500_000
LAG_YEARS = (1, 2, 3)

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
