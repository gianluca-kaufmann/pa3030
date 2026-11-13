# Mouillot et al. (2024) reproducibility audit (R0)

Paper: Mouillot D. et al., The socioeconomic and environmental niche of protected areas reveals global
conservation gaps and opportunities, Nature Communications 15:9007 (2024).

## Package

- Zenodo record 11183846, version v1.0, published 2024-05-13, CC-BY-4.0.
- One file, `FRBCesab/pelagic-v1.0.zip`, 384,451,203 bytes, md5 `87bd229f508019e7bcedc74588773db2` (matches Zenodo).
- Mirrors GitHub `FRBCesab/pelagic` tag v1.0, commit `f4a2b42`.
- Unpacked locally at `~/Files/PA/mouillot_pelagic_v1.0/src/FRBCesab-pelagic-f4a2b42/` (outside any repo).
- Contents: `data/` 29 files (415 MB), `analyses/` 20 R scripts, `figures/` 11 files, an R package skeleton (`R/`, `man/`).
- `_make.R` runs the ENFA analysis and Figures 1, S2, S3 only.

## Does the package refit the published model?

- No. The package holds no code for the steps that produced the published model and its scores:
  - no `blockCV` / `spatialBlocks` (the 1,000 km spatial cross-validation),
  - no `missForest` (gap filling),
  - no GAM calibration of the RF scores,
  - no `caret::confusionMatrix` (accuracy, TSS).
- `ranger` is called only in `analyses/Fig3.R` and `analyses/FigS6.R`, to draw partial-dependence plots. Those calls fit on the full balanced sample, with default settings and no seed.
- The reported spatial-CV land accuracy (0.74 ± 0.02) cannot be reproduced from the package.
- The stored `score.ATP.All.RData` lists 10 folds with accuracy 0.852 to 0.856 and kappa 0.704 to 0.712. Those values do not match the paper's 0.74. Unresolved: they may come from a non-spatial split or from another model.

## What the package stores

| File | Rows | Content |
|---|---|---|
| `IUCN.I.VI.ATP.RData` | 517,450 | Training sample: 258,725 protected cells (IUCN I–VI) + 258,725 unprotected background cells, with all predictor values |
| `res_prob_terrestrial.Rdata` | 301,972 | `corr.rf.response`: 258,725 background cells + 43,247 IUCN Ia/Ib cells, with `ISO3`, `long`, `lat` |
| `final_predictions_terrestrial.Rdata` | 301,972 | Same cells, `PredictedProtection` |
| `datZLand.RData` | 258,725 | Conservation-priority ranking (`RankZ`) for the background cells |
| `importance_permutation_*.Rdata` | — | Permutation importances (Fig. 2) |
| `CellsLand.tif` | 517,450 cells | Cell ids of the sampled cells only (10 km Mollweide), not the full land grid |

- The published maps (Figs. 4–6, "potential" vs "unrealistic" gains) read `res_prob_*.corr.rf.response` (`fig-4_maps.R`, `fig-X_maps.R`).
- Because the protected rows are IUCN Ia/Ib only, `corr.rf.response` is the score of the strict IUCN-I model.
- The all-category (IUCN I–VI) model's scores are not stored.
- The mean of `corr.rf.response` (0.1432) equals the protected share of those rows (43,247 / 301,972).

## Can the stored score serve as a frozen benchmark?

- Methods: the background cells are "randomly selected [...] among the remaining unprotected cells", matched in number to the protected cells.
- So the 258,725 background cells are a uniform random sample of land cells unprotected in April 2020. A post-2020 designation test on them keeps the true event rate in expectation.
- Caveats:
  - **Strict model only.** The stored score is the IUCN-I model. A test of the all-category model needs a refit from the stored training data, which is no longer the authors' exact model.
  - **In-sample scores ruled out (check below).** The background cells were training negatives, but neither stored score behaves like an in-sample prediction.
  - **The two stored scores disagree.** `corr.rf.response` is not a monotone transform of `PredictedProtection` (different AUCs, Spearman 0.745 on background cells). The published maps use `corr.rf.response`, so it is the primary benchmark and `PredictedProtection` the sensitivity check. Which model and split produced each score is unresolved.
  - **Sampling fraction unknown.** The package does not give the total number of unprotected land cells, so it does not give the share the sample represents.
  - **No WDPA snapshot.** The April 2020 WDPA snapshot itself is not included, only each cell's protected flag and IUCN category.

## In-sample check (2026-11-13)

- Script: `scripts/regions/world/mouillot_in_sample_check.R`, ranger 0.18.0, R 4.5.2, seed 42, 46 s on 16 cores.
- What it does: refits the strict IUCN-I land model on the stored training sample (301,972 rows, formula and log transforms from `Fig3.R`). Every stored row matched a training row on longitude, latitude and ISO3.

| Score | AUC, Ia/Ib vs background | Background median | Background 95th pct |
|---|---|---|---|
| Refit, in-sample | 0.9994 | 0.0045 | 0.171 |
| Refit, out-of-bag | 0.9847 | 0.0065 | 0.268 |
| Stored `PredictedProtection` | 0.9704 | 0.0029 | 0.286 |
| Stored `corr.rf.response` | 0.9076 | 0.0225 | 0.357 |

- Spearman correlation on background cells:
  - `corr.rf.response` vs refit: 0.853 (in-sample), 0.837 (out-of-bag).
  - `PredictedProtection` vs refit: 0.655 (in-sample), 0.637 (out-of-bag).
- Both stored scores separate the classes worse than random out-of-bag predictions. In-sample predictions reach 0.999.
- Interpretation, not proof: the stored scores look like held-out predictions. Held-out cells from 1,000 km spatial blocks would score below random out-of-bag predictions. What would overturn this: a stored score that equals in-sample predictions of a weaker forest (fewer or shallower trees). The package records no forest settings that could settle it.

## Predictor dates

- 14 land factors (`list_variables_terrestrial.csv`).
  - Environment: altitude, distance to ocean, island, precipitation, freshwater, NDVI, temperature.
  - Socioeconomic: local GDP, human footprint, conflict, HDI, NGO, accessibility, natural resources.
- Sources, versions and reference years are in the paper's Supplementary Data 1, which is not in the package. Each `allowed_cutoff` stays open until that file is read.

## R0 verdict

- A frozen benchmark exists for the strict IUCN-I land model, on a uniform random sample of 258,725 cells unprotected in April 2020.
- The published spatial-CV performance and the all-category model cannot be reproduced exactly from the package.

## Open items before T1

1. Get Supplementary Data 1 and record source, version and reference year for each of the 14 factors.
2. Done 2026-11-13: the stored scores are not in-sample (see the in-sample check). Still open: which model and split produced each of the two scores. Ask the authors (Casajus or Loiseau, listed in `DESCRIPTION`) if this matters for the protocol.
3. Count the unprotected 10 km land cells in April 2020 on the GADM 3.6 grid, to get the sampling fraction.
4. Build the post-2020 event ledger from archived WDPA releases (April 2020 onward), separating new designations from late reporting, renaming and boundary changes.
5. Write the T1 protocol (cutoff, horizon, event definition, metric, baselines, go/no-go) before scoring any designation.
