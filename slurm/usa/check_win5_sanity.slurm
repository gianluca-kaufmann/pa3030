#!/bin/bash
#SBATCH --job-name=check_win5_sanity
#SBATCH --partition=normal.24h
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/%u/logs/check_win5_sanity_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/check_win5_sanity_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/outputs/usa/tables"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

VENV_DIR="$HOME/venvs/master_thesis"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting win5 sanity checks..."
echo "PWD = $(pwd)"

python - <<'PY'
import os
import sys
from pathlib import Path
import duckdb

lookahead = 5
scratch = Path(os.environ.get("SCRATCH", "/cluster/scratch/gikaufmann"))
job_id = os.environ.get("SLURM_JOB_ID", "local")

# Paths used by your current split run.
raw_candidates = [
    Path("/cluster/scratch/gikaufmann/data/usa/ml/merged_panel_final.parquet"),
    scratch / "data/usa/ml/merged_panel_final.parquet",
    Path("/cluster/home/gikaufmann/master_thesis/outputs/usa/results/merged_panel_final.parquet"),
]
win5_candidates = [
    Path("/cluster/scratch/gikaufmann/data/usa/ml/main/merged_panel_final_win5.parquet"),
    scratch / "data/usa/ml/main/merged_panel_final_win5.parquet",
    Path("/cluster/home/gikaufmann/master_thesis/outputs/usa/results/main/merged_panel_final_win5.parquet"),
]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

raw_path = first_existing(raw_candidates)
win5_path = first_existing(win5_candidates)

if raw_path is None:
    print("ERROR: Could not find merged_panel_final.parquet in expected locations.")
    for p in raw_candidates:
        print(f"  checked: {p}")
    sys.exit(1)

if win5_path is None:
    print("ERROR: Could not find merged_panel_final_win5.parquet in expected locations.")
    for p in win5_candidates:
        print(f"  checked: {p}")
    sys.exit(1)

print(f"Raw panel:   {raw_path}")
print(f"Win5 panel:  {win5_path}")

con = duckdb.connect()
threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
mem_per_cpu = int(os.environ.get("SLURM_MEM_PER_CPU", "16384"))
mem_gb = max(1, (threads * mem_per_cpu) // 1024)
con.execute(f"SET threads={threads}")
con.execute(f"SET memory_limit='{mem_gb}GB'")
con.execute("SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{scratch}/duckdb_temp'")

raw = str(raw_path).replace("'", "''")
win5 = str(win5_path).replace("'", "''")

max_year = con.execute(f"SELECT MAX(year) FROM read_parquet('{raw}')").fetchone()[0]
cutoff = max_year - lookahead
print(f"\nmax_year={max_year}, lookahead={lookahead}, cutoff={cutoff}")

print("\nCHECK 1: transition_01 per year (raw panel, includes 2000 baseline)")
rows = con.execute(
    f"""
    SELECT year, COUNT(*) AS total, SUM(transition_01) AS pos
    FROM read_parquet('{raw}')
    GROUP BY year
    ORDER BY year
    """
).fetchall()
for y, total, pos in rows:
    ratio = (pos / total) if total else 0.0
    print(f"  {y}: total={total:,} pos={pos:,} pos_ratio={ratio:.5f}")

print("\nCHECK 2: Is year-2000 a baseline spike?")
year_2000 = con.execute(
    f"""
    SELECT COALESCE(SUM(transition_01), 0)
    FROM read_parquet('{raw}')
    WHERE year = 2000
    """
).fetchone()[0]
print(f"  transition_01 positives in 2000: {year_2000:,}")

print("\nCHECK 3: Label identity on win5 file (definition check)")
violations = con.execute(
    f"""
    WITH first_transitions AS (
        SELECT row, col, MIN(year) AS first_transition_year
        FROM read_parquet('{raw}')
        WHERE transition_01 = 1
        GROUP BY row, col
    ),
    expected AS (
        SELECT
            d.row, d.col, d.year, d.transition_01_win5,
            CASE
                WHEN ft.first_transition_year IS NOT NULL
                     AND d.year < ft.first_transition_year
                     AND ft.first_transition_year <= d.year + {lookahead}
                THEN 1 ELSE 0
            END AS expected_win5
        FROM read_parquet('{win5}') d
        LEFT JOIN first_transitions ft USING (row, col)
    )
    SELECT COUNT(*) FROM expected WHERE transition_01_win5 <> expected_win5
    """
).fetchone()[0]
print(f"  mismatched rows: {violations:,}")
print("  PASS" if violations == 0 else "  FAIL")

print("\nCHECK 4: How many rows have transition_01=1 but transition_01_win5=0 (by year)?")
rows = con.execute(
    f"""
    SELECT
        year,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN transition_01 = 1 THEN 1 ELSE 0 END) AS pos_t01,
        SUM(CASE WHEN transition_01_win5 = 1 THEN 1 ELSE 0 END) AS pos_win5,
        SUM(CASE WHEN transition_01 = 1 AND transition_01_win5 = 0 THEN 1 ELSE 0 END) AS t01_not_win5
    FROM read_parquet('{win5}')
    GROUP BY year
    ORDER BY year
    """
).fetchall()
for y, total_rows, pos_t01, pos_win5, t01_not_win5 in rows:
    print(
        f"  {y}: t01={pos_t01:,} win5={pos_win5:,} "
        f"t01_and_not_win5={t01_not_win5:,}"
    )

print("\nCHECK 5: 2019 win5 positives and share")
y2019 = con.execute(
    f"""
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN transition_01_win5 = 1 THEN 1 ELSE 0 END) AS pos_win5
    FROM read_parquet('{win5}')
    WHERE year = 2019
    """
).fetchone()
total_2019, pos_2019 = y2019
ratio_2019 = (pos_2019 / total_2019) if total_2019 else 0.0
print(f"  2019: total={total_2019:,} pos_win5={pos_2019:,} pos_ratio={ratio_2019:.6f}")

out_file = scratch / "outputs/usa/tables" / f"win5_sanity_summary_{job_id}.txt"
with out_file.open("w", encoding="utf-8") as f:
    f.write(f"raw_path={raw_path}\n")
    f.write(f"win5_path={win5_path}\n")
    f.write(f"max_year={max_year}\n")
    f.write(f"cutoff={cutoff}\n")
    f.write(f"year2000_transition_01={year_2000}\n")
    f.write(f"label_identity_mismatches={violations}\n")
    f.write(f"year2019_total={total_2019}\n")
    f.write(f"year2019_win5_pos={pos_2019}\n")
    f.write(f"year2019_win5_ratio={ratio_2019:.8f}\n")

print(f"\nWrote summary: {out_file}")
print("\nSanity check finished.")
PY

echo "check_win5_sanity finished."
