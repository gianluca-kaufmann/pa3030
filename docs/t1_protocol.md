# T1 protocol: retrospective test of the Mouillot et al. (2024) protection niche

Version 1.1, frozen 19 November 2026 before any designation outcome was scored. The commit holding this version and the file's sha256 are recorded with the freeze. Any later change is listed under Deviations with its date and reason, and a changed rule gives a new version.

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

- Primary: new IUCN Ia or Ib protection, established 2021 to 2024, of land unprotected at the cutoff.
- Secondary, fixed in advance: the same for IUCN I to VI, scored with the same strict-model score. This tests transfer of the strict niche to broader protection. It does not validate the all-category model, whose scores are not stored.
- Neither outcome covers all of 30x30: non-IUCN sites and OECMs are outside both.
- New land is the part of an event outside the union of all cutoff baseline polygons. Extensions of earlier sites count for their new land. Designation of land already protected at the cutoff (upgrades, stricter zones, re-registration) is analysed separately and is never a positive.
- Overlap with the baseline is diagnostic. High overlap does not prove an upgrade in law, and no overlap threshold is used or tuned.
- Governance: governmental, private, NGO and Indigenous sites all count when the record is Designated, Inscribed or Established with a 2021 to 2024 year. A government-governance-only sensitivity is reported. It does not imply that other sites are invalid.
- Footprint: a record's footprint at the window end comes from the last release up to `202412` holding it as a qualifying polygon. Where its first footprint differs, the established footprint is the union of the two: land protected during the window and removed or cut before its end still counts as established. The removed part is kept separately. Records first reported after `202412` have only later footprints, which may include later additions that cannot be separated. Land held only in releases between the first and the window-end footprint is not reconstructed. It is a stated limitation, not part of the established footprint.
- Release timing is not legal timing. A release shows that a record or boundary was reported by that date. It does not show when the act took effect, so a change seen in a 2021 to 2024 release may rest on an earlier act, and an act of 2024 may first appear in 2025.
- Category timing, from the release the footprint comes from:
  - recorded in window: the category is recorded in a release up to `202412`. Primary.
  - recorded after window: the record was first reported after `202412`, so its category is known only retrospectively, even with a 2021 to 2024 year. Primary, flagged, with a sensitivity removing these events. Leaving them out of the primary would favour countries that report quickly.
  - assigned later: the record was "Not Assigned" or "Not Reported" and received a category only in a later release. A later assignment does not show that the category held during the window. These events are removed from the primary evaluation of the category they were later given, and bounded once as positive and once as negative.
- The legal year is the latest STATUS_YR, since later releases correct years. A sensitivity uses the category at the observation release.

## Populations

- Sample population: the stored background cells, which the paper defines as cells outside IUCN I to VI protection in April 2020.
- Expansion population: the sample population minus cells overlapped at the cutoff by a qualifying protected area of any IUCN status (Designated, Inscribed or Established, PA_DEF 1). This removed about 8.6% of the sample on raw release `202004`. The share is recomputed on the reconstructed baseline before scoring. The primary analysis uses the expansion population. The sample population is reported as a sensitivity analysis.
- On the reconstructed baseline, with polygons split at the antimeridian, the expansion population excludes 18,133 of 258,725 background cells (7.01%). Only 61 of them touch an IUCN I to VI polygon.
- Exclusion counts by reason and by country are reported, along with how the excluded cells change the population.
- Baseline point records (20,857) have no boundary and are not used for exclusion in the primary analysis, so earlier protection can be present in some eligible cells. Two scenario sensitivities exclude cells containing a baseline point (1,814 eligible cells) and cells touched by a circle of the point's reported area around it (7,655 eligible cells). Neither footprint is the real boundary, so neither is an upper or lower bound on the true effect.

## Cutoff, window and observation

- Cutoff: April 2020, the paper's snapshot date. The stored protection flags match April 2020 content: 186 Colombian protected cells are covered only by sites first present in Colombia's April 2020 update.
- Cutoff baseline: a reconstructed April 2020 release, because the Earth Engine copies of WDPA omit whole countries in single releases. Sources by country:
  - Russia from `202003` (absent from `202004` to `202006`, identical IDs and attributes on return in `202007`);
  - Colombia, Ecuador and Mozambique from `202005` (truncated in `202004`. The May 2020 release notes report no change for them, `202005` and `202006` are identical for them, and no substituted record has STATUS_YR after 2019);
  - every other country from `202004`, plus the 2 sites absent only from `202004`.
- The reconstruction totals 227,326 polygons against the official April 2020 total of 227,321, and matches the stored category in 1,998 of 2,000 sampled protected cells. Unverified: identity with the official file, boundaries (compared by GIS_AREA only), Russia's April content, the 5 extra polygons, and whether May 2020 rather than April matches the stored flags.
- The Earth Engine asset `202003` holds February 2020 content and does not stand for March.
- No post-cutoff record enters the baseline as new protection. The `202005` substitutions supply only sites that the official April release reported.
- Designation window: legal designation in calendar years 2021 to 2024.
- Observation date: release `202610`, 22 months after the window ends (the minimum is 18). The later release `202611` is not used: it holds 4,651 Estonian, 14,646 German and 1,215 Polish records against 21,681, 23,609 and 3,217 in `202610`, the pattern of the 2020 truncations, and as the last release it cannot be screened for transient omissions.
- The first release in which a site appears is recorded separately from its legal designation year (STATUS_YR).

## Event ledger

- Built from historical releases with their own geometries by `scripts/regions/world/wdpa_event_ledger.py`. The current-only extraction (`scripts/regions/world/fetch_wdpa_global.py`) cannot serve as the event dataset unchanged.
- Each release is checked for completeness by country. A country-release with failed or missing chunks is marked incomplete, never treated as complete.
- Every release used is screened at site-ID level: a WDPA_PID absent from one release but present in the releases before and after marks that country-release as a transient omission. Count thresholds are not used, since they missed omissions of 3,157 German and 933 Swedish sites in `202003`.
- Each candidate site is followed across releases by WDPAID, WDPA_PID, name, STATUS, STATUS_YR, IUCN_CAT and geometry. ID changes, multipart sites, reappearances, removals and degazettement are reconciled.
- Parcel lineage: a parcel record leaving the archive for good and a record of the same WDPAID first appearing in the next release, of the same kind and IUCN category and with areas within 1%, are one parcel renamed (release `202512` renamed suffixes _A, _B to _1, _2). Ties go to the same suffix position, and remaining ambiguous pairs stay separate. A pair is linked only if its polygons in the two releases share at least half of their union (intersection over union 0.5). Point records have no polygon and keep the attribute match. The successor takes the predecessor's ID. Pairs are listed with their release. Legitimate multipart sites keep one record per parcel.
- The union of earlier releases is used to detect prior records and omissions. It does not automatically count as protection at the cutoff.
- Every candidate gets one class:
  1. Candidate new designation: not recorded before the cutoff, STATUS_YR 2021 to 2024. Archive-derived, not legally verified.
  2. Late reporting: first seen after the cutoff, STATUS_YR before 2020.
  3. Ambiguous 2020 timing: STATUS_YR 2020.
  4. Change to an existing site: reappearance, ID change, category change or upgrade, boundary extension.
  5. Unresolved: missing or conflicting information.
- Expansion role of each record, all archive-derived:
  - new site: class 1;
  - candidate extension: a new parcel of a site protected before the cutoff, with its own STATUS_YR 2021 to 2024. Its new land is expansion on the same evidence as a new site;
  - uncertain extension: a re-entered record of a site protected before the cutoff whose STATUS_YR now falls in 2021 to 2024. The year may be a correction or a re-registration;
  - undated change: a category change or area increase of a baseline record. Its new land has no legal date.
- New sites and candidate extensions are the positive events. The new land of uncertain extensions and undated changes has uncertain timing and is handled like class 3 and class 5. Upgrades of land protected at the cutoff are analysed separately.
- STATUS_YR above 2024: not an event in the window. Their cells stay negative, and a sensitivity removes them. STATUS_YR missing (0): class 5.
- Point records (no boundary): class 5. For each outcome, the cell containing a point record of the outcome's categories with STATUS_YR 2020 to 2024 or missing is removed. Point records dated before 2020 are late reports and, like baseline points, exclude no cell in the primary evaluation. Point records dated after 2024 leave their cells negative. A point has no geometry release, so its category timing is not checked, and the removal applies whatever the timing. A sensitivity removes cells touched by a circle of the reported area. Both are approximations.
- Source validation: 57 events in a stratified sample drawn before checking were compared with national records (`PA/audit/event_validation_results.csv`). The check is not extended to unsampled events, which are not described as legally verified. Two general rules follow from it and are applied to all events: International designations (for example Ramsar listings) are not new legal protection and become class 5, and the new-land rule above covers the upgrades found.
- D1: every event established 2021 to 2024 counts, whether or not it is present in the observation release, including land later removed or cut. The question is where designation happened during the window. A survivor-only outcome would mix designation with later reporting changes. Renamed parcels are reconciled by lineage. Absences are reported by type (land still mapped under another ID, site present under another parcel, partly mapped, unresolved). Unresolved absences are bounded: once counted as positive, once removed.

## Cell labels

- A cell is positive if the new land of a positive event (new site or candidate extension) of the outcome's categories overlaps its 10 km square, the paper's own overlap rule. New land is the established footprint outside the union of cutoff baseline polygons. In the expansion population every eligible cell lies outside the baseline, so this equals any overlap.
- Eligible cells overlapped by the new land of a class 3 or class 5 site (International designations included, sites designated after 2024 excepted), an uncertain extension, an undated change, or an event whose category was assigned later, and by no positive event, are removed from the primary evaluation and never labelled negative. Cells outside the expansion population are not eligible and are never counted as removed. Bounds: once treated as positive, once as negative.
- Each positive cell is assigned to the positive event with the largest area in the cell.

## Metrics

- Primary, within each country:
  - average precision (AP),
  - observed prevalence,
  - AP divided by prevalence,
  - Recall@5%, the share of positive cells in the top 5% of that country's eligible cells.
- Secondary: Recall@1% within each country, and pooled global AP and Recall@5% with one global budget. The pooled metrics let differences between countries contribute.
- Reported per country with eligible-cell, positive-cell and event counts.
- Summaries give every country equal weight and include only countries with at least one positive cell. Countries with zero positives are listed separately, with no recall or AP and no imputed zero.
- Summaries include countries with at least 100 eligible cells and at least 5 positive cells. A sensitivity uses at least 1 positive cell. Every country is reported with an estimability flag.
- Concentration across countries is reported with the share of positive cells per country and the effective number of countries (inverse Herfindahl of positive cells).
- Ties: AP treats tied scores as one threshold. A tie at a Recall budget boundary gets fractional credit, the expected recall under random ordering of the tied cells.

## Baselines

All baselines use only information available at the cutoff.

- Within country:
  - random ranking;
  - distance from the cell centre to the nearest protected area at the cutoff (closer ranks higher);
  - an environment and accessibility ranking, the mean rank of travel time (higher ranks higher), human footprint (lower ranks higher) and altitude (higher ranks higher), from the stored predictor values.
- Between countries (pooled metrics only): each country's designation rate before the cutoff, from sites with STATUS_YR 2015 to 2019.
- No country-level expansion outcome is evaluated in T1. National counts are descriptive, and no national or global totals are inferred, since the sampling fraction is unknown.

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
7. Countries with any release in the window flagged by the ID-level transient screen removed.
8. Designation window 2021 to 2023.
9. Russia, Colombia, Ecuador and Mozambique, the countries whose cutoff content is reconstructed, removed from the expansion population and the distance baseline.
10. Government-governance events only.
11. Category at the observation release in place of the category at the footprint release. Events whose category was recorded only after the window removed.
12. Post-2024 events removed rather than negative. Cells containing or touched by baseline or event point records removed (two scenario footprints).
13. Land removed or cut during the window not counted as established (window-end footprint only).
14. Candidate extensions not counted as events (new sites only).
15. Freshwater dropped, by refitting the strict model on the stored training sample. This is a separate refitted benchmark and never replaces the frozen score.

## Biodiversity comparison

Specified separately in `docs/biodiversity_comparison_protocol.md`, which is not part of this frozen T1 protocol.

## Interpretation

- A comparison with the baselines tests whether the 2020 niche adds predictive information. It does not decide whether results are reported or whether the paper proceeds. Every result listed above is reported, whatever its sign.
- The test asks whether expansion after 2020 followed the 2020 placement pattern, then what that expansion represents relative to equal-area alternatives. It does not show future feasibility, causal biodiversity gains, or that one algorithm beats another.
- Four predictors (HDI, NGO count, conflict, natural resources) take one value per country, and HDI has the highest importance in the strict model. Importance alone does not show what drives temporal performance.

## Deviations

- 19 November 2026, version 1.1, before any score was examined. Clarifies which point records remove cells (year and outcome category), that International records remove cells as class 5, that sites designated after 2024 stay negative, and that only eligible cells are counted as removed. Prompted by a review of the label counts, which applied an all-category point mask built before parcel lineage, removed post-2024 sites, and left International records in neither mask. No rule on positive events changed.

Protocol status: FROZEN, version 1.1. The biodiversity comparison has its own protocol and does not hold this one open.
