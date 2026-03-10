# SA/USA Tuning Runbook

This runbook describes the new SA/USA tuning system with paper mode as default.

## Defaults

- `TUNING_MODE=paper` (default)
- `TUNING_OPTIMIZER=optuna` (default in paper mode)
- `CV_STRATEGY=rolling` (default in paper mode)
- `ROLLING_FOLDS=5`
- `TARGET_COL=transition_01_win5` (fixed in framework)
- Risk set filtering is always enforced: `WDPA_prev == 0`

## Quick local runs

### South America

```bash
python scripts/regions/south_america/4_tuning/model1_tuning_lgbm
python scripts/regions/south_america/4_tuning/model1_tuning_rf
```

### USA

```bash
python scripts/regions/usa/4_tuning/model2_tuning_lgbm
python scripts/regions/usa/4_tuning/model2_tuning_rf
```

## Fast debug mode (optional)

Use fast mode only for debugging and rapid iteration:

```bash
TUNING_MODE=fast TUNING_OPTIMIZER=randomized CV_STRATEGY=holdout \
python scripts/regions/usa/4_tuning/model2_tuning_lgbm
```

## Euler SLURM

Recommended entrypoints:

- `slurm/south_america/tuning_lgbm.slurm`
- `slurm/south_america/tuning_rf.slurm`
- `slurm/usa/tuning_lgbm.slurm`
- `slurm/usa/tuning_rf.slurm`

All scripts default to paper mode and expose environment-variable overrides.

## Artifacts

Each tuning run writes:

- Canonical artifact:
  - `scripts/regions/<region>/5_training/lgbm_best_params.json`
  - `scripts/regions/<region>/5_training/rf_best_params.json`
- Timestamped archive in corresponding `tuning/` directory.

Artifacts include metadata fields required by training loaders:

- `dataset_fingerprint`
- `target_col`
- `feature_count`
- `year_split`
- `git_commit`
- `timestamp`

## Sampling alignment (critical for SA)

### Problem

Without a positive cap, the 5-year lookahead target (`transition_01_win5`)
causes an extreme class imbalance inversion in the SA tuning dataset: with a
2.5M-row budget and ~14 years × ~140k positives/year, roughly 1.9M of the 2.5M
rows are positives (~75.8%). This is the inverse of the real distribution
(~0.3–0.5% positive in production) and breaks PR-AUC as a meaningful metric.

### Fix: `MAX_POS_PER_YEAR_PAPER`

The SA SLURM scripts set `MAX_POS_PER_YEAR_PAPER=5000`.  With ~14 active years
this yields ≈70k positives out of ≈2.1M rows (SPW ≈ 30x), matching the
production rare-event regime.

| Setting | Approx. pos | Approx. SPW | Regime |
|---------|-------------|-------------|--------|
| `MAX_POS_PER_YEAR=0` (no cap) | ~1.9M | ~0.3 | Majority-positive — broken |
| `MAX_POS_PER_YEAR=5000` | ~70k | ~30 | Rare-event — correct |

USA defaults to `MAX_POS_PER_YEAR_PAPER=0` (no cap) because USA positives are
naturally sparse; verify with a class-balance check before production runs.

The positive cap uses the same reservoir-sampling algorithm as the negative cap
so positives are drawn uniformly at random within each year.

## `scale_pos_weight` handling

`scale_pos_weight` is **not** a tunable hyperparameter.  It is computed
deterministically from the data and injected as a fixed value:

- **Optuna path:** computed per fold as `n_neg_fold / n_pos_fold`, so every
  trial sees the correct weighting regardless of sampled class balance.
- **Randomized search path:** computed from the training-fold union and
  passed as a fixed parameter to `LGBMClassifier` before the grid search.

This keeps tuning consistent with the production training convention and
ensures PR-AUC is evaluated in the correct rare-event regime.

## Search space changes (vs. original grid)

Three regularisation parameters were added and `colsample_bytree` was replaced
with `colsample_bynode` to match the production training configuration:

| Parameter | Change | Rationale |
|-----------|--------|-----------|
| `scale_pos_weight` | **Removed** | Computed deterministically (see above) |
| `colsample_bytree` | **Replaced** by `colsample_bynode` | Production uses `colsample_bynode` (per-split column subsampling) |
| `min_split_gain` | **Added** | Regularises leaf creation; was doing meaningful work in production |
| `path_smooth` | **Added** | Smooths leaf scores along the tree path; reduces over-optimisation on rare events |
