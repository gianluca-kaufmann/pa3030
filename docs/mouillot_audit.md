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
  - **Provenance unresolved (check below).** The background cells were training negatives. Neither stored score matches the in-sample predictions of the refit, but the check cannot say whether the authors' scores are in-sample or held-out.
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
- Interpretation, not proof: the stored scores are consistent with held-out predictions, since cells held out in 1,000 km spatial blocks would score below random out-of-bag predictions. The check does not establish this. A less flexible forest (larger minimum node size, smaller `mtry`, fewer or shallower trees) predicted in-sample could also score below 0.98, and the package records no forest settings.
- What would settle it: the authors' score-generation code, fold membership or out-of-bag records, or the authors' description of how each stored score was produced.
- A post-April 2020 designation test is held out in time whatever the spatial split, because no post-2020 outcome entered the fit. If the scores were fitted on the background cells, every one of them was a training negative, which can lower their scores and change their ranking. The T1 protocol must state which case applies, or that it is unknown.

## Predictor dates

- 14 land factors (`list_variables_terrestrial.csv`).
  - Environment: altitude, distance to ocean, island, precipitation, freshwater, NDVI, temperature.
  - Socioeconomic: local GDP, human footprint, conflict, HDI, NGO, accessibility, natural resources.
- Source: Supplementary Data 1, `41467_2024_53241_MOESM4_ESM.xlsx` (15,248 bytes, Springer static content, downloaded 2026-11-13 to `~/Files/PA/mouillot_pelagic_v1.0/supplementary/`).
- Release dates for the freshwater and island sources were checked from Zenodo and Crossref metadata. Every other row repeats the sheet's period and database, and their release dates are not yet verified.

| Factor | Database (as listed) | Reference period | Scale | Observable by April 2020? |
|---|---|---|---|---|
| Altitude | GEBCO-2019 grid | 2019 | 15 arc s | Yes, static |
| Distance to ocean | GADM 3.6 | — | polygons | Yes, static |
| Island | Mouillot et al. 2020 island database | — | polygons | Release 7 Sep 2020 (Crossref) is after the cutoff. Island status is physical, so content is unchanged |
| Mean temperature | WorldClim 2 | 1970–2000 | 30 arc s | Yes |
| Precipitation | WorldClim 2 | 1970–2000 | 30 arc s | Yes |
| Freshwater | Copernicus Global Land Service land cover 100 m, collection 3, epoch 2015 (zenodo 3939038, V3.0.1) | 2015 | 100 m | **No.** V3.0.1 was published 8 Sep 2020 (Zenodo). It describes 2015, but this release postdates the cutoff |
| NDVI | Copernicus Global Land Service long-term NDVI statistics | 1999–2017 | ~1 km | Yes by period, release unverified |
| Local GDP | Kummu et al. (dryad dk1j0) | 2010–2015 | 5 arc min | Yes by period, release unverified |
| Human footprint | SEDAC 2009 Human Footprint, 2018 release | 2009 | ~1 km | Yes |
| Accessibility | Malaria Atlas Project travel time to cities | 2015 | ~1 km | Yes by period, release unverified |
| Conflict | UCDP/PRIO Armed Conflict Dataset v19.1 (sum of conflict years) | 1946–2018 | country | Yes by period, release unverified |
| HDI | UNDP Human Development Report 2019 | 2018 | country | Yes by period, release unverified |
| NGO | Union of International Associations, count per country | 2019 | country | Yes by period, release unverified |
| Natural resources | World Bank (no source link in the sheet) | 2019 | country | Yes by period, release unverified |

- **April 2020 cutoff.** Every reference period ends by 2019. One input release postdates the cutoff: freshwater (V3.0.1, Sep 2020). The island database is also a later release, but island status does not change. So the frozen score is prospective for April 2020 except for the freshwater layer. Freshwater ranks 13th of 14 in mean permutation importance (0.029). The T1 protocol still needs a sensitivity run with freshwater dropped or replaced by a pre-2020 release.
- **Earlier cutoffs (T2 rolling backtests).** HDI (2018), NGO (2019), natural resources (2019), conflict (to 2018), GEBCO 2019 and NDVI (to 2017) are fixed at their latest years. Any cutoff before 2018 would use later values, so backtests need re-dated versions of these factors. The frozen score cannot serve them.
- **Country-level factors.** Conflict, HDI, NGO and natural resources take one value per country. In the stored permutation importances (`importance_permutation_terrestrial.Rdata`, mean over 10 folds), HDI ranks 1st (0.130), NGO 5th (0.101), natural resources 7th (0.100) and conflict 8th (0.094) of 14. Country-level values repeated across cells are not local evidence, so held-out-country tests carry the forecasting claim.

## R0 verdict

- A frozen benchmark exists for the strict IUCN-I land model, on a uniform random sample of 258,725 cells unprotected in April 2020.
- The published spatial-CV performance and the all-category model cannot be reproduced exactly from the package.

## Open items before T1

1. Done 2026-11-13: predictor dates recorded (see Predictor dates). Still open: release dates for the rows marked unverified.
2. Partly done 2026-11-13: neither stored score matches the in-sample predictions of the refit (see the in-sample check). Still open: which model and split produced each of the two scores. This needs the authors' code, fold records or a description from the authors (Casajus or Loiseau, listed in `DESCRIPTION`).
3. Count the unprotected 10 km land cells in April 2020 on the GADM 3.6 grid, to get the sampling fraction.
4. Build the post-2020 event ledger from archived WDPA releases (April 2020 onward), separating new designations from late reporting, renaming and boundary changes.
5. Write the T1 protocol (cutoff, horizon, event definition, metric, baselines, go/no-go) before scoring any designation.
