#!/usr/bin/env python3
"""
Quantify train/test leakage in the SA pa3030 main split.

Main split (from scripts/regions/south_america/5_training/model1_splits):
  TRAIN years [2001, 2016], TEST years [2017, 2019], 5-year lookahead.

A test-positive pixel "leaks" if first_transition_year is in [2018, 2021] —
that pixel then also carries a positive label somewhere in TRAIN.

Usage on Euler:
  python quantify_leakage.py $SCRATCH/outputs/south_america/results/merged_panel_final.parquet \
                             leakage_summary.json
"""

import json
import sys
from pathlib import Path

import duckdb


def main(parquet_path: Path, out_json: Path) -> None:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=" + str(8))

    p = str(parquet_path).replace("'", "''")

    min_year = con.execute(f"SELECT MIN(year) FROM read_parquet('{p}')").fetchone()[0]
    print(f"min_year in panel: {min_year}")

    con.execute(f"""
        CREATE TEMP TABLE first_transitions AS
        SELECT row, col, MIN(year) AS first_transition_year
        FROM read_parquet('{p}')
        WHERE transition_01 = 1 AND year > {min_year}
        GROUP BY row, col
    """)

    print("\n=== first_transition_year distribution, 2018-2024 ===")
    dist_rows = con.execute("""
        SELECT first_transition_year, COUNT(*) AS n_pixels
        FROM first_transitions
        WHERE first_transition_year BETWEEN 2018 AND 2024
        GROUP BY first_transition_year
        ORDER BY first_transition_year
    """).fetchall()
    for y, n in dist_rows:
        print(f"  {y}: {n:,} pixels")

    print("\n=== test-positive pixel-year rows, bucketed ===")
    by_year = con.execute(f"""
        WITH test_pos AS (
            SELECT p.year, ft.first_transition_year
            FROM read_parquet('{p}') p
            JOIN first_transitions ft USING (row, col)
            WHERE p.year BETWEEN 2017 AND 2019
              AND p.year < ft.first_transition_year
              AND ft.first_transition_year <= p.year + 5
        )
        SELECT year AS test_year,
               COUNT(*) FILTER (WHERE first_transition_year BETWEEN 2018 AND 2021) AS leaking,
               COUNT(*) FILTER (WHERE first_transition_year BETWEEN 2022 AND 2024) AS clean,
               COUNT(*) AS total
        FROM test_pos
        GROUP BY year
        ORDER BY year
    """).fetchall()

    by_year_out, tot_leak, tot_clean = [], 0, 0
    for ty, leak, clean, total in by_year:
        share = leak / total if total else 0.0
        print(f"  test_year={ty}: total={total:,}  leaking={leak:,} ({share:.1%})  clean={clean:,}")
        by_year_out.append(dict(test_year=ty, total=total, leaking=leak, clean=clean, leak_share=share))
        tot_leak += leak
        tot_clean += clean
    grand = tot_leak + tot_clean
    grand_share = tot_leak / grand if grand else 0.0
    print(f"\nALL test positives: total={grand:,}  leaking={tot_leak:,} ({grand_share:.1%})  clean={tot_clean:,}")

    distinct = con.execute(f"""
        WITH test_pos_pixels AS (
            SELECT DISTINCT p.row, p.col, ft.first_transition_year
            FROM read_parquet('{p}') p
            JOIN first_transitions ft USING (row, col)
            WHERE p.year BETWEEN 2017 AND 2019
              AND p.year < ft.first_transition_year
              AND ft.first_transition_year <= p.year + 5
        )
        SELECT COUNT(*) FILTER (WHERE first_transition_year BETWEEN 2018 AND 2021) AS leaking,
               COUNT(*) FILTER (WHERE first_transition_year BETWEEN 2022 AND 2024) AS clean,
               COUNT(*) AS total
        FROM test_pos_pixels
    """).fetchone()
    d_leak, d_clean, d_tot = distinct
    d_share = d_leak / d_tot if d_tot else 0.0
    print(f"DISTINCT test-positive pixels: total={d_tot:,}  leaking={d_leak:,} ({d_share:.1%})  clean={d_clean:,}")

    summary = dict(
        split=dict(train_years=[2001, 2016], test_years=[2017, 2019], lookahead_years=5),
        first_transition_distribution_2018_2024={str(y): n for y, n in dist_rows},
        by_test_year=by_year_out,
        all_test_positive_rows=dict(total=grand, leaking=tot_leak, clean=tot_clean, leak_share=grand_share),
        distinct_test_positive_pixels=dict(total=d_tot, leaking=d_leak, clean=d_clean, leak_share=d_share),
    )
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nJSON written to {out_json}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python quantify_leakage.py PATH_TO_merged_panel_final.parquet [OUTPUT_JSON]")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("leakage_summary.json"))
