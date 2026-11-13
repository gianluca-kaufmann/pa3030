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
- The reported spatial-CV land performance cannot be reproduced from the package. The paper reports accuracy 0.74 ± 0.02 and TSS 0.47 ± 0.04 for all categories, and accuracy 0.91 ± 0.02 and TSS 0.46 ± 0.06 for IUCN I.
- The stored cross-validation files do not match either model:

| File | Accuracy (10 folds) | TSS | Paper |
|---|---|---|---|
| `score.ATP.All.RData` | 0.854 ± 0.001 | 0.708 ± 0.003 | 0.74 ± 0.02, TSS 0.47 |
| `score.ATP.I.RData` | 0.945 ± 0.001 | 0.722 ± 0.006 | 0.91 ± 0.02, TSS 0.46 |

- Both files list a different predictor set from the paper's 14: they include `Voice` and `logPopDensity` and lack accessibility, local GDP, human footprint and NDVI. They come from another model version. They are read only by `analyses/figure_3.R`. The `importance_permutation_*` files (Fig. 2) use the paper's 14 predictors.

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
- The methods do not describe the sampling design further. If the draw was simple random, the 258,725 background cells represent land cells unprotected in April 2020, and a post-2020 designation test on them keeps the true event rate in expectation. This is an assumption to state in the protocol, not a verified property.
- Caveats:
  - **Strict model only.** The stored score is the IUCN-I model. A test of the all-category model needs a refit from the stored training data, which is no longer the authors' exact model.
  - **Provenance unresolved (check below).** The background cells were training negatives. Neither stored score matches the in-sample predictions of the refit, but the check cannot say whether the authors' scores are in-sample or held-out.
  - **The two stored scores disagree.** `corr.rf.response` is not a monotone transform of `PredictedProtection` (different AUCs, Spearman 0.745 on background cells). The published maps use `corr.rf.response`, so it is the primary benchmark and `PredictedProtection` the sensitivity check. Which model and split produced each score is unresolved.
  - **Sampling fraction unknown.** The package does not give the total number of unprotected land cells, so it does not give the share the sample represents.
  - **No WDPA snapshot.** The April 2020 WDPA snapshot itself is not included, only each cell's protected flag and IUCN category. The paper does not name its exact release. See the WDPA agreement check below.

## In-sample check (2026-11-13)

- Script: `scripts/regions/world/mouillot_in_sample_check.R`, ranger 0.18.0, R 4.5.2, seed 42, 46 s on 16 cores.
- What it does: refits the strict IUCN-I land model on the stored training sample (301,972 rows, formula and log transforms from `Fig3.R`). Every stored row matched a training row on longitude, latitude and ISO3.
- The refit is not the authors' specification. `Fig3.R` uses untransformed HDI and freshwater, while the Fig. 2 importance files list `logHDI` and `logFreshWater`. Trees split on the order of a predictor, so a monotone transform does not change which partitions are available. The refit also skips the gap filling (0.7% of land cells per the methods) and the GAM calibration (monotone, so it leaves AUC unchanged). Its forest settings beyond 500 trees are ranger defaults, which the paper does not confirm.

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
- The methods state that the protected flags come from the April 2020 WDPA. If the stored scores come from that fit, no post-2020 outcome entered them, and a post-April 2020 designation test is held out in time whatever the spatial split. This rests on the paper's description, since the score-generation code is not in the package. If the scores were fitted on the background cells, every one of them was a training negative, which can lower their scores and change their ranking. The T1 protocol must state which case applies, or that it is unknown.

## WDPA agreement check (2026-11-13)

- Script: `scripts/regions/world/mouillot_wdpa_agreement.py`. It draws 1,000 stored land cells from each stratum (background, IUCN Ia/Ib, IUCN II–VI, seed 42), rebuilds each as a 10 km Mollweide square around its stored centre, and lists the polygons of an Earth Engine WDPA release that intersect it. The stored centres sit on a regular 10 km Mollweide grid.
- Release `WCMC/WDPA/202004`:

| Stratum | Any polygon | Any IUCN I–VI polygon |
|---|---|---|
| Background | 0.087 | 0.001 |
| IUCN Ia/Ib | 0.941 | 0.920 |
| IUCN II–VI | 0.889 | 0.882 |

- Where a protected cell meets an IUCN polygon, its strictest category equals the stored category in every one of the 1,802 cases.
- Background cells overlap non-IUCN polygons (not reported, not applicable, not assigned) in 8.6% of cases but an IUCN I–VI polygon in 1 of 1,000. So the paper counted only IUCN I–VI sites as protected. Cells covered only by non-IUCN sites sit in the background, which matters for a 30x30 risk set.
- The 198 protected cells with no IUCN polygon in `202004` are 184 in Russia, 12 in Colombia and 2 elsewhere. The Earth Engine `202004` release holds 55 Russian polygons and 414 Colombian ones. Releases `201907` to `202003` hold 2,206 and 996, `202005` and `202006` hold about 50 and 1,211, and `202009` holds 2,206 and 1,211.
- On release `202003`, 196 of those 198 cells (0.99) meet a polygon of the stored category, and 1 of 200 resampled background cells meets an IUCN polygon. The 1,802 cells that matched on `202004` were not rechecked on `202003`.
- Conclusion: the stored flags are consistent with a WDPA release that still held Russia and Colombia in full. The Earth Engine `202004` asset lacks most of both. `202003` is the closest complete pre-cutoff release in Earth Engine. Whether the public April 2020 download differs from the Earth Engine asset is not checked.
- Consequence for the event ledger: whole countries drop out of single releases and return (Russia also grows from 2,206 to 7,785 polygons by `202102`). A site first seen after a dropout is not a new designation. Eligibility must use the union of releases up to the cutoff, and new events need a designation year after the cutoff, not only a first appearance.

## Predictor dates

- 14 land factors (`list_variables_terrestrial.csv`).
  - Environment: altitude, distance to ocean, island, precipitation, freshwater, NDVI, temperature.
  - Socioeconomic: local GDP, human footprint, conflict, HDI, NGO, accessibility, natural resources.
- Source: Supplementary Data 1, `41467_2024_53241_MOESM4_ESM.xlsx` (15,248 bytes, Springer static content, downloaded 2026-11-13 to `~/Files/PA/mouillot_pelagic_v1.0/supplementary/`).
- Release dates for the freshwater and island sources were checked from Zenodo and Crossref metadata. Every other row repeats the sheet's period and database, and their release dates are not yet verified.

| Factor | Database (as listed) | Reference period | Scale | Release before April 2020? |
|---|---|---|---|---|
| Altitude | GEBCO-2019 grid | 2019 | 15 arc s | Not checked. Static |
| Distance to ocean | GADM 3.6 | — | polygons | Not checked. Static |
| Island | Mouillot et al. 2020 island database | — | polygons | No. Released 7 Sep 2020 (Crossref). Island status is physical, so the content should not depend on the release |
| Mean temperature | WorldClim 2 | 1970–2000 | 30 arc s | Not checked |
| Precipitation | WorldClim 2 | 1970–2000 | 30 arc s | Not checked |
| Freshwater | Copernicus Global Land Service land cover 100 m, collection 3, epoch 2015 (zenodo 3939038, V3.0.1) | 2015 | 100 m | **No.** V3.0.1 was published 8 Sep 2020 (Zenodo). It describes 2015, and earlier collection-3 releases may differ |
| NDVI | Copernicus Global Land Service long-term NDVI statistics | 1999–2017 | ~1 km | Not checked |
| Local GDP | Kummu et al. (dryad dk1j0) | 2010–2015 | 5 arc min | Not checked |
| Human footprint | SEDAC 2009 Human Footprint, 2018 release | 2009 | ~1 km | Sheet says 2018 release. Not checked |
| Accessibility | Malaria Atlas Project travel time to cities | 2015 | ~1 km | Not checked |
| Conflict | UCDP/PRIO Armed Conflict Dataset v19.1 (sum of conflict years) | 1946–2018 | country | Not checked |
| HDI | UNDP Human Development Report 2019 | 2018 | country | Not checked |
| NGO | Union of International Associations, count per country | 2019 | country | Not checked |
| Natural resources | World Bank (no source link in the sheet) | 2019 | country | Not checked |

- **April 2020 cutoff.** Every reference period ends by 2019, but a reference period before the cutoff does not show that the data were available then. Two releases are verified to postdate the cutoff: freshwater (V3.0.1, Sep 2020) and the island database (Sep 2020). The other release dates are not checked. So the frozen score cannot yet be called an operational April 2020 forecast. It can be evaluated retrospectively against later designations, which is a different claim.
- **Freshwater.** Freshwater ranks 13th of 14 in mean permutation importance in the strict model (`importance_permutation_terrestrial_strict.Rdata`, 0.014). Low mean importance does not show that the layer has no effect on the ranking of individual cells or on temporal performance. Dropping freshwater or swapping in a pre-2020 release requires a refit, which gives a separate benchmark from the published frozen score. The protocol must report the two separately.
- **Earlier cutoffs (T2 rolling backtests).** HDI (2018), NGO (2019), natural resources (2019), conflict (to 2018), GEBCO 2019 and NDVI (to 2017) are fixed at their latest years. Any cutoff before 2018 would use later values, so backtests need re-dated versions of these factors. The frozen score cannot serve them.
- **Country-level factors.** Conflict, HDI, NGO and natural resources take one value per country. In the strict model's stored permutation importances (`importance_permutation_terrestrial_strict.Rdata`, mean over 10 folds), HDI ranks 1st (0.099), NGO 3rd (0.079), conflict 4th (0.063) and natural resources 6th (0.061) of 14. Country-level values repeated across cells are not local evidence, so held-out-country tests carry the forecasting claim.

## R0 verdict

- Stored scores exist for the strict IUCN-I land model on 258,725 cells the paper describes as randomly selected among cells unprotected in April 2020. They can serve as a frozen retrospective benchmark, with three conditions stated in the protocol: score provenance is unresolved, the sampling design is assumed, and the input release dates are only partly verified.
- The published spatial-CV performance, the stored cross-validation files and the all-category model cannot be reproduced from the package.

## Open items before T1

1. Done 2026-11-13: predictor dates recorded (see Predictor dates). Still open: release dates for the rows marked unverified.
2. Partly done 2026-11-13: neither stored score matches the in-sample predictions of the refit (see the in-sample check). Still open: which model and split produced each of the two scores. This needs the authors' code, fold records or a description from the authors (Casajus or Loiseau, listed in `DESCRIPTION`).
3. Count the unprotected 10 km land cells in April 2020 on the GADM 3.6 grid, to get the sampling fraction.
4. Build the post-2020 event ledger from archived WDPA releases, separating new designations from late reporting, renaming and boundary changes. Earth Engine holds monthly `WCMC/WDPA/YYYYMM` releases from 2017-07 to 2026-11 (listed 2026-11-13), with gaps including 2020-10 to 2021-01, 2021-08, 2021-11, 2022-08 and 2026-02 to 2026-06. Use the union of releases up to the cutoff for eligibility (see the WDPA agreement check).
5. Write the T1 protocol (cutoff, horizon, event definition, metric, baselines, go/no-go) before scoring any designation.
