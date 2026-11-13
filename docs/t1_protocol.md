# T1 protocol: retrospective test of the Mouillot et al. (2024) protection niche

Fixed before any designation outcome is scored. Items marked OPEN must be settled, and this file committed, before scoring. Any later change is listed under Deviations with its date and reason.

## Question

Did the 2020 social-environmental niche of strict protected areas (Mouillot et al. 2024, Nature Communications 15:9007) rank the places that received new protection during 2021 to 2024, within each country? What does the answer imply for where protection expands under the 30x30 target?

- This is a retrospective evaluation of frozen scores. It is not an operational April 2020 forecast, because two inputs (freshwater CGLS LC100 V3.0.1, the island database) were released after April 2020 and ten input release dates are unchecked.
- Evaluating the frozen model country by country does not make its predictions leave-country-out. The model was trained on all countries.

## Scores

- Primary: `corr.rf.response` in `res_prob_terrestrial.Rdata` (Zenodo 11183846, v1.0), the strict IUCN I land model, on the 258,725 stored background cells (10 km Mollweide).
- Sensitivity: `PredictedProtection` in `final_predictions_terrestrial.Rdata`, same cells. Every outcome and metric is repeated with it.
- Both are presence-background relative likelihoods. They are rankings, not designation probabilities.
- Stated limits:
  - Whether the stored scores are in-sample or held-out predictions for these cells is unknown. Every background cell was a training negative.
  - The paper says the background cells were "randomly selected". Prevalence in the sample estimates prevalence among unprotected land cells only if the draw was simple random. The sampling fraction is unknown, so the test reports no population totals or coverage shares.

## Outcomes

- Primary: new IUCN Ia or Ib protection after the cutoff.
- Secondary, fixed in advance: new IUCN I to VI protection, scored with the same strict-model score. This tests transfer of the strict niche to broader protection. It does not validate the all-category model, whose scores are not stored.
- Neither outcome covers all of 30x30: non-IUCN sites and OECMs are outside both.

## Populations

- Sample population: the stored background cells, which the paper defines as cells outside IUCN I to VI protection in April 2020.
- Expansion population: the sample population minus cells overlapped at the cutoff by a qualifying protected area of any IUCN status (Designated, Inscribed or Established, PA_DEF 1). This removes about 8.6% of the sample (agreement check, release `202004`). The primary analysis uses the expansion population. The sample population is reported as a sensitivity analysis.
- Exclusion counts by reason and by country are reported, along with how the excluded cells change the population.

## Cutoff, window and observation

- Cutoff: April 2020, the paper's snapshot date. OPEN: the release that stands for it. Release `202003` recovers 196 of 198 cells that `202004` misses, but the 1,802 cells that matched on `202004` have not been rechecked on `202003`. That check and a per-country completeness check of `201907` to `202005` come first.
- Designation window: legal designation in calendar years 2021 to 2024.
- Observation date: OPEN, at least 18 months after the window ends. The latest available release (`202611`) gives 23 months.
- The first release in which a site appears is recorded separately from its legal designation year (STATUS_YR).

## Event ledger

- Built from historical releases with their own geometries. The current-only extraction (`scripts/regions/world/fetch_wdpa_global.py`) cannot serve as the event dataset unchanged.
- Each release is checked for completeness by country. A country-release with failed or missing chunks is marked incomplete, never treated as complete.
- Each candidate site is followed across releases by WDPAID, WDPA_PID, name, STATUS, STATUS_YR, IUCN_CAT and geometry. ID changes, multipart sites, reappearances, removals and degazettement are reconciled.
- The union of earlier releases is used to detect prior records and omissions. It does not automatically count as protection at the cutoff.
- Every candidate gets one class:
  1. Verified new designation: not recorded before the cutoff, STATUS_YR 2021 to 2024.
  2. Late reporting: first seen after the cutoff, STATUS_YR before 2020.
  3. Ambiguous 2020 timing: STATUS_YR 2020.
  4. Change to an existing site: reappearance, ID change, category change or upgrade, boundary extension.
  5. Unresolved: missing or conflicting information.
- Only class 1 sites are positive events in the primary analysis. Class 4 is analysed separately from new sites.

## Cell labels

- A cell is positive if a class 1 event polygon of the outcome's categories overlaps its 10 km square, the paper's own rule.
- Cells overlapped by a class 3 or class 5 site, and by no class 1 site, are removed from the primary evaluation and never labelled negative. Bounds: once treated as positive, once as negative.
- Each positive cell is assigned to the class 1 event with the largest area in the cell.

## Metrics

- Primary, within each country:
  - average precision (AP),
  - observed prevalence,
  - AP divided by prevalence,
  - Recall@5%, the share of positive cells in the top 5% of that country's eligible cells.
- Secondary: Recall@1% within each country, and pooled global AP and Recall@5% with one global budget. The pooled metrics let differences between countries contribute.
- Reported per country with eligible-cell, positive-cell and event counts.
- Summaries give every country equal weight and include only countries with at least one positive cell. Countries with zero positives are listed separately, with no recall or AP and no imputed zero.
- OPEN: the minimum number of eligible cells for a country to enter the summaries.
- Ties: AP treats tied scores as one threshold. A tie at a Recall budget boundary gets fractional credit, the expected recall under random ordering of the tied cells.

## Baselines

All baselines use only information available at the cutoff.

- Within country:
  - random ranking;
  - distance from the cell centre to the nearest protected area at the cutoff (closer ranks higher);
  - an environment and accessibility ranking, the mean rank of travel time (higher ranks higher), human footprint (lower ranks higher) and altitude (higher ranks higher), from the stored predictor values.
- Between countries (pooled metrics only): each country's designation rate before the cutoff, from sites with STATUS_YR 2015 to 2019.
- OPEN: if the paper evaluates which countries expand most, define the country outcome (for example, newly protected share of eligible cells) and the aggregation of cell scores before scoring.

## Uncertainty

- Paired country-clustered bootstrap for every model-minus-baseline difference: resample countries with replacement, 1,000 replicates, percentile 95% intervals.
- No bootstrap treats cells as independent.
- Concentration is reported: the number of events, and the share of positive cells held by the 10 largest events. Results are repeated without those 10 events.

## Sensitivity analyses

1. `PredictedProtection` in place of `corr.rf.response`, for both outcomes.
2. Sample population in place of expansion population.
3. Class 3 cells counted as positive, then as negative.
4. Class 2 and class 4 sites counted as events.
5. A cell is positive only if events cover at least 10% of it.
6. The 10 largest events removed.
7. Countries with any incomplete release in the window removed.
8. Designation window 2021 to 2023.
9. Freshwater dropped, by refitting the strict model on the stored training sample. This is a separate refitted benchmark and never replaces the frozen score.

## Interpretation

- A comparison with the baselines tests whether the 2020 niche adds predictive information. It does not decide whether results are reported or whether the paper proceeds. Every result listed above is reported, whatever its sign.
- Four predictors (HDI, NGO count, conflict, natural resources) take one value per country, and HDI has the highest importance in the strict model. Importance alone does not show what drives temporal performance.

## Deviations

None yet.
