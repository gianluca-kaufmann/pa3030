#!/usr/bin/env python3
"""Select a representative subset of LOBO biomes via greedy farthest-point
sampling in normalised feature space, maximising ecological diversity while
minimising the number of SLURM array tasks.

The selection is data-driven and objective: per-biome mean feature vectors are
computed from a sample of the training parquet, z-score normalised, and then
the N biomes that are farthest apart in that space are selected greedily.

Run interactively on the Euler login node (fast, ~1-2 min), from the repo root:

    python scripts/regions/shared/evaluation/select_lobo_biomes.py \\
        --region south_america --n-biomes 5

    python scripts/regions/shared/evaluation/select_lobo_biomes.py \\
        --region usa --n-biomes 6

    python scripts/regions/shared/evaluation/select_lobo_biomes.py \\
        --region se_asia --n-biomes 6

The script writes the selected indices to slurm/{region}/lobo_biome_array.txt
so SLURM files do not need to be edited manually.  Submit directly with:

    sbatch --array=$(cat slurm/south_america/lobo_biome_array.txt) \\
        slurm/south_america/spatial_CV_1.slurm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path so the package import works the same
# way as the thin wrapper scripts (PYTHONPATH=$HOME/master_thesis on Euler).
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

from scripts.regions.shared.evaluation.spatial_cv_core import (  # noqa: E402
    _biome_raster_paths,
    build_biome_groups,
    BATCH_SIZE,
    EXCLUDE_COLS,
    load_biome_raster_and_mapping,
    resolve_parquet,
    TRAIN_YEARS,
)

_DEFAULT_SAMPLE_ROWS = 500_000  # rows to read; enough for stable means


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_biome_means(
    parquet_path: Path,
    feature_cols: list[str],
    biome_raster: np.ndarray,
    biome_groups: list,
    sample_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (means, counts) of shape (n_biomes, n_features).

    Reads up to *sample_rows* training-year rows from *parquet_path*,
    assigns each row to a biome via the raster lookup, and accumulates
    per-biome feature sums (NaN-safe).
    """
    H, W = biome_raster.shape
    n_b = len(biome_groups)
    n_f = len(feature_cols)

    # ecoregion int code → fold index
    int_to_fold: dict[int, int] = {}
    for fi, (_, ints) in enumerate(biome_groups):
        for code in ints:
            int_to_fold[code] = fi

    sums   = np.zeros((n_b, n_f), dtype=np.float64)
    counts = np.zeros(n_b, dtype=np.int64)
    total  = 0

    cols = feature_cols + ["row", "col", "year"]
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        if total >= sample_rows:
            break

        ya = batch["year"].to_numpy(zero_copy_only=False)
        ra = batch["row"].to_numpy(zero_copy_only=False)
        ca = batch["col"].to_numpy(zero_copy_only=False)

        keep = (ya >= TRAIN_YEARS[0]) & (ya <= TRAIN_YEARS[1])
        ra, ca = ra[keep], ca[keep]

        ib  = (ra >= 0) & (ra < H) & (ca >= 0) & (ca < W)
        eco = np.zeros(len(ra), dtype=np.int32)
        eco[ib] = biome_raster[ra[ib], ca[ib]].astype(np.int32)

        fold_idx = np.full(len(ra), -1, dtype=np.int32)
        for code, fi in int_to_fold.items():
            fold_idx[eco == code] = fi
        valid = fold_idx >= 0

        Xb = np.column_stack([
            batch[c].to_numpy(zero_copy_only=False)[keep][valid].astype(np.float64)
            for c in feature_cols
        ])
        fi_valid = fold_idx[valid]

        for b in range(n_b):
            mask = fi_valid == b
            if mask.any():
                sums[b]   += np.nansum(Xb[mask], axis=0)
                counts[b] += int(mask.sum())

        total += int(keep.sum())
        print(f"  Sampled {min(total, sample_rows):,} / {sample_rows:,} rows …", end="\r")

    print()
    means = np.where(
        counts[:, None] > 0,
        sums / np.maximum(counts[:, None], 1),
        0.0,
    )
    return means, counts


def greedy_farthest_point(means_norm: np.ndarray, n_select: int) -> list[int]:
    """Greedy farthest-point selection in normalised feature space.

    Seeds from the biome with the largest L2 norm (most extreme in feature
    space), then repeatedly picks the biome farthest from all already-selected
    biomes.  Returns sorted indices.
    """
    seed = int(np.argmax(np.linalg.norm(means_norm, axis=1)))
    selected = [seed]
    min_sq = np.full(len(means_norm), np.inf)

    while len(selected) < n_select:
        d = ((means_norm - means_norm[selected[-1]]) ** 2).sum(axis=1)
        min_sq = np.minimum(min_sq, d)
        min_sq[selected] = -1.0          # exclude already-selected
        selected.append(int(np.argmax(min_sq)))

    return sorted(selected)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select representative LOBO biomes by feature-space diversity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--region", required=True,
                        help="Region name, e.g. south_america, usa, or se_asia")
    parser.add_argument("--n-biomes", type=int, required=True,
                        help="Number of biomes to select")
    parser.add_argument("--sample-rows", type=int, default=_DEFAULT_SAMPLE_ROWS,
                        help=f"Max training rows to read (default {_DEFAULT_SAMPLE_ROWS:,})")
    args = parser.parse_args()

    raster_path, shp_path = _biome_raster_paths(args.region)
    if not raster_path.exists():
        print(f"ERROR: Biome raster not found: {raster_path}")
        sys.exit(1)

    print("Loading biome raster and mapping …")
    biome_raster, _, int_to_biome = load_biome_raster_and_mapping(raster_path, shp_path)
    biome_groups = build_biome_groups(int_to_biome, biome_raster)
    n_biomes = len(biome_groups)

    print(f"\nAll {n_biomes} biomes in {args.region}:")
    for idx, (name, _) in enumerate(biome_groups):
        print(f"  [{idx:2d}] {name}")

    if args.n_biomes >= n_biomes:
        print(f"\nRequested {args.n_biomes} ≥ available {n_biomes} — using all biomes.")
        indices = list(range(n_biomes))
    else:
        train_path = resolve_parquet(args.region, "train.parquet")
        schema = pq.ParquetFile(train_path).schema_arrow
        feature_cols = [
            name for name, field in zip(schema.names, schema)
            if (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
            and name not in EXCLUDE_COLS
        ]
        print(f"\nComputing per-biome feature means "
              f"({len(feature_cols)} features, up to {args.sample_rows:,} rows) …")
        means, counts = compute_biome_means(
            train_path, feature_cols, biome_raster, biome_groups, args.sample_rows,
        )

        zero_data = counts == 0
        if zero_data.any():
            zero_names = [biome_groups[i][0] for i in np.where(zero_data)[0]]
            print(f"  WARNING: {zero_data.sum()} biome(s) have no data in sample "
                  f"and cannot be selected: {zero_names}")

        # Z-score normalise across biomes that have data
        mu  = means[~zero_data].mean(axis=0)
        std = means[~zero_data].std(axis=0)
        std[std == 0] = 1.0
        means_norm = (means - mu) / std
        means_norm[zero_data] = 0.0   # zero-data biomes map to centroid; won't be seeded

        print(f"Running greedy farthest-point selection ({args.n_biomes} biomes) …")
        indices = greedy_farthest_point(means_norm, args.n_biomes)

    print("\n" + "=" * 60)
    print(f"Selected {len(indices)} biomes for --region {args.region}:")
    print("=" * 60)
    for idx in indices:
        name, _ = biome_groups[idx]
        print(f"  [{idx:2d}] {name}")

    array_str = ",".join(str(i) for i in indices)

    # Write array string to file so sbatch can consume it without editing SLURM files
    array_file = _REPO / f"slurm/{args.region}/lobo_biome_array.txt"
    array_file.parent.mkdir(parents=True, exist_ok=True)
    array_file.write_text(array_str + "\n")
    print(f"\nArray indices written to: {array_file}")

    print(f"\nSubmit with:")
    print(f"  sbatch --array=$(cat slurm/{args.region}/lobo_biome_array.txt) "
          f"slurm/{args.region}/spatial_CV_1.slurm")
    print(f"  sbatch --array=$(cat slurm/{args.region}/lobo_biome_array.txt) "
          f"slurm/{args.region}/spatial_CV_1_rf.slurm")


if __name__ == "__main__":
    main()
