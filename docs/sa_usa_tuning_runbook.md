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
python scripts/regions/south_america/4_ml/2_tuning/model1_tuning_lgbm
python scripts/regions/south_america/4_ml/2_tuning/model1_tuning_rf
```

### USA

```bash
python scripts/regions/usa/4_ml/2_tuning/model2_tuning_lgbm
python scripts/regions/usa/4_ml/2_tuning/model2_tuning_rf
```

## Fast debug mode (optional)

Use fast mode only for debugging and rapid iteration:

```bash
TUNING_MODE=fast TUNING_OPTIMIZER=randomized CV_STRATEGY=holdout \
python scripts/regions/usa/4_ml/2_tuning/model2_tuning_lgbm
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
  - `scripts/regions/<region>/4_ml/3_training/lgbm_best_params.json`
  - `scripts/regions/<region>/4_ml/3_training/rf_best_params.json`
- Timestamped archive in corresponding `tuning/` directory.

Artifacts include metadata fields required by training loaders:

- `dataset_fingerprint`
- `target_col`
- `feature_count`
- `year_split`
- `git_commit`
- `timestamp`
