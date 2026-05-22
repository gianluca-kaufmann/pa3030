"""Country-year panel aggregation for Stage 1 models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.regions.shared.country_raster import country_ids_for_rows, load_country_raster

BATCH_SIZE = 500_000
LAG_YEARS = (1, 2, 3)


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
    return cy
