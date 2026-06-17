# PA3030 — Publication Roadmap

**Updated**: 2026-06-17 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

**Story**: The 30×30 agreement forces countries to roughly double protected area coverage by 2030. We predict which land areas will be designated — giving investors, policymakers, and conservation planners a data-driven transition risk tool. Stage 1 predicts *when and how much* each country expands; Stage 2 predicts *which unprotected land patches* are selected. Combined output: annual designation probability for every unprotected connected land block in South America, 2025–2030.

**Target journal**: Nature Sustainability (primary) → One Earth / GEC → JEEM.

---

## Research Questions

**RQ1 (Stage 1 — solved)**: Which countries will expand their PA networks under 30×30, and how much area will they add each year?

**RQ2 (Stage 2 — in progress)**: Within an expansion event, which contiguous land areas does a government select — and can we learn the spatial, ecological, and institutional patterns of past designations well enough to predict future ones?

Governments designate **polygons** (connected areas), not individual pixels. Patch-level ranking directly models this decision unit. The methodological contribution: we show that changing the ranking unit from pixels to connected land patches resolves the fundamental framing mismatch and makes the task tractable.

**RQ3 (applied)**: Where does predicted PA expansion diverge from conservation priority — and what does the gap reveal about systemic bias in how 30×30 will be met in practice?

---

## Architecture

```
P(patch j designated in year t)
  = P(country C expands in year t)        ← Stage 1: Poisson GLM, country-year panel
  × S(patch j | expansion in C, t)        ← Stage 2: LightGBM LambdaRank on patches
```

**Stage 1** — Poisson GLM, LASSO α=100, 9 features. Metric: D² OOS. Train 2001–2016, test 2017–2023.

**Stage 2** — LightGBM LambdaRank. Ranking unit: connected-component patches of unprotected land (WDPA_prev==0). One row per patch per country-year; positive if any pixel inside was designated. Grouped by `(country_id, year)`. Train 2001–2013, early-stop 2014–2016, test 2017–2024.

**Primary metrics**: Lift@1% + Recall@5% within expansion groups (patch level). Secondary: weighted Recall (weighted by n_pos), size-stratified Recall.

**Forward output** (Phase 4): Stage 1 budget × Stage 2 calibrated patch scores → fill from top-ranked patches until budget exhausted → cumulative risk 1−∏(1−p_t). BAU / 30×30 / NGFS scenarios.

**Lift metric note**: `lift_at_k_within_groups` = macro_precision@k / global_positive_rate. Mixing macro numerator and micro baseline is consistent across all experiments and does not affect rankings, but must be noted in the paper.

---

## Current State (2026-06-17)

### What is proven

| Claim | Evidence |
|---|---|
| Patch-level framing is correct | H12: Recall@5% 18.96% → 87.7% by changing ranking unit alone |
| Patch-level architecture is clean of leakage | Full audit: dist_wdpa uses WDPA_prev; transition_01 is annual; trends use [t-4,t]; all lag features use .shift(k) |
| Pixel-level approaches are fundamentally limited | Tiers 1+2 ceilinged at ~19% Recall regardless of training-side fixes |

### What is NOT yet proven (gaps to Nature)

| Gap | Why it matters |
|---|---|
| Model ≈ naive size-sort | Trained model (81.32×, 87.7%) barely beats size-only (80.38×, 88.8%). Reviewers will reject if no added value shown. |
| 66.4% of gain from one feature | `log_patch_size_km2` dominates; model converges at iter 20. Not learned ecological pattern recognition. |
| Brazil performance poor | On Brazil (the most important SA country), trained model (Lift=2.67×, Recall=11.9%) is WORSE than size-only (7.47×, 49.4%). |
| Metrics inflated by trivial groups | 43/61 test groups have exactly 1 positive patch in small countries. Trivially solved by size ranking. |
| Missing AGB/carbon feature | REDD+ is a primary SA designation driver. AGB exists on scratch but has NaN/scaling issue. |
| No shape/heterogeneity features | Current patch features are all means. Variance, compactness, perimeter not included. |
| No naive baseline in paper | Required for any top-tier journal submission. |
| No cross-regional validation at patch level | Stage 2 not tested on SE Asia or USA at patch level. |
| Stage 1+2 not integrated | Model published as ranking exercise, not full conditional probability. |
| RQ3 undeveloped | Representation gap analysis not started. |
| No CIs on metrics | Required for statistical rigor. |

---

## Publication Bar

| Metric | Nature bar | Current (H12) | Naive size-sort |
|---|---|---|---|
| Lift@1% | ≥ 15× | 81.32× ✅ | 80.38× |
| Recall@5% (macro) | ≥ 90% | 87.7% | 88.8% |
| Recall@5% (weighted by n_pos) | TBD — must add | not computed | not computed |
| Size-stratified Recall (medium patches) | Must show >random | not computed | not computed |
| Recall on Brazil alone | Must show >size-sort | 11.9% ✗ | 49.4% |

**The headline numbers look strong but do not yet constitute a Nature-level result.** The model must demonstrably outperform naive size-only ranking — especially on the large, ecologically important cases like Brazil. Until that is achieved, the paper cannot be submitted.

---

## Guiding Principles (updated 2026-06-17)

1. **Patch-level framing is correct and locked.** Never go back to pixel ranking. The methodological contribution (matching ranking unit to government decision unit) is real and will survive peer review.

2. **The current model is a size-sorter.** This is the central scientific problem to solve. The fix is not tuning — it is richer features that enable within-size discrimination.

3. **Brazil is the hard case and the scientific core.** Brazil accounts for the largest and most policy-relevant designations in SA. Any model that cannot outperform size-sort on Brazil cannot be published in Nature. Brazil performance is the primary quality gate.

4. **Structure before tuning.** Do not run Optuna until the feature set is locked (AGB, shape, percentile features all added and tested). Tuning the wrong feature space is wasted compute.

5. **No ablation of log_patch_size_km2 from the model.** Patch size IS a legitimate, scientifically meaningful predictor. The goal is to show what the model learns BEYOND size, not to remove size. Size stays in the feature set; we add more features alongside it and show they contribute.

6. **Every result must beat a documented naive baseline.** For every metric we report in the paper, we must show: (a) random, (b) size-only, (c) our model. No exceptions.

7. **Proxy experiments are directionally useful, not quantitatively reliable.** Mini-sample experiments guide direction. Full SA results decide.

8. **No hyperparameter retuning until feature set locked.** Established: early retune at pixel level was catastrophic (2.85× → 2.06×).

---

## Nature-Ready Plan

### Phase 1 — Fix the size-sort problem (immediate; ~1 week)

**Goal**: Show that the model learns real ecological/institutional signal beyond patch size.

#### P1.1 — Run no-size ablation on full SA
Remove `log_patch_size_km2` from the feature set and retrain H12. This is the most critical diagnostic.

- If Recall@5% >> random (say >40%): there IS signal beyond size; add enriched features to unlock it.
- If Recall@5% ≈ random: current 83 features carry almost no within-size signal; need fundamentally better features.
- Script: re-run `model1_LGBM_stage2_patch` with STAGE2_FEATURE_EXCLUDE=log_patch_size_km2 env var, OR edit feature_cols at load time.
- SLURM: can run on existing patch panel. ~30 min job.

#### P1.2 — Enrich patch aggregation: shape + percentile features
Modify `build_patch_panel.py` to add:

| New feature | Description | Why |
|---|---|---|
| `log_patch_perimeter_km` | log(perimeter in pixels ×1000m) | Management cost proxy; compact vs elongated |
| `patch_compactness` | 4π × area / perimeter² | Closer to 1 = circle; lower = fragmented, harder to fence |
| `patch_std_gsn_b2` | Std of biodiversity within patch | Heterogeneous patches may be more/less attractive |
| `patch_std_ndvi` | Std of NDVI within patch | Mix of forest/open land vs uniform |
| `patch_max_gsn_b2` | Max biodiversity within patch | Peak biodiversity matters, not just mean |
| `patch_max_deforestation_b1` | Max deforestation pressure within patch | Threatened areas may be prioritised |
| `patch_std_elevation` | Std of elevation | Mountain vs flat; ecoregional diversity |
| `patch_frac_high_ndvi` | Fraction of pixels with NDVI > 0.7 | Dense forest cover fraction |

Perimeter computation: during CC labelling, count boundary pixels (pixels adjacent to protected or non-patch pixels). This is an O(n) operation within the existing scipy CC pass.

Rebuild patch panel with enriched features after implementing. ~6h Euler job (same as original build).

#### P1.3 — Fix AGB and add to patch panel
AGB TIF exists at `data/south_america/ready/AGB/agb_sa.tif`. Previous pixel-level attempt failed (best_iter=5, likely NaN or extreme scale). Steps:
1. Inspect AGB TIF: check value range, NaN fraction, projection alignment.
2. If NaN > 20%: clip to SA land mask and fill ocean/water pixels with 0.
3. If scale issue: values in Mg/ha — check range is 0–500 (reasonable). Clip outliers.
4. Add to merge pipeline OR add directly to patch panel via `add_feature_to_splits.py` pattern.
5. Patch-level aggregation: mean AGB, max AGB per patch (both informative — mean = average carbon density, max = presence of high-carbon forest).
6. Test on mini-sample before full SA rebuild.

AGB is the single most important missing feature: REDD+ payments and VCM credits are tied to protecting high-carbon-stock forests. Brazil's Amazon designations are heavily influenced by carbon finance. This likely explains a significant part of the model's Brazil failure.

#### P1.4 — Weighted group evaluation metric
Add to `stage2_metrics.py` and evaluation scripts:

```python
def weighted_recall_at_k(y_true, y_score, group_sizes, k_pct, weights):
    # weights[g] = n_pos in group g (or n_pos_pixels)
    # Computes recall weighted by group size, not macro average
```

Macro recall treats Brazil 2018 (148 positives) the same as Ecuador 2020 (1 positive). Weighted recall gives Brazil proportional influence. Report BOTH in paper.

**Decision rule after P1.1 and P1.2+1.3 results are in:**
- No-size ablation Recall >40% AND enriched features improve Brazil >20%: genuine signal exists → proceed to Phase 2.
- No-size ablation Recall <20%: rethink feature engineering approach before proceeding.

---

### Phase 2 — Feature engineering enrichment (~2–3 weeks)

**Goal**: Build a feature set rich enough that the model genuinely outperforms size-sort, especially on Brazil.

#### P2.1 — Indigenous territory overlap
RAISG data (RAISG.socioambiental.org) provides indigenous territory polygons for SA. Indigenous territories are a major pathway for PA designation — politically easier, internationally recognised.

- Feature: `frac_indigenous_area` = fraction of patch pixels overlapping indigenous territory
- Feature: `in_indigenous_territory` = binary (any overlap)
- Requires: RAISG shapefile download → rasterise → add to panel
- Expected value: very strong predictor for Brazil, Bolivia, Peru where many large PAs overlap indigenous lands

#### P2.2 — REDD+ eligibility proxy
Computable from existing features (no new data needed):
- `redd_eligible` = (AGB_mean > 50 Mg/ha) AND (deforestation_b1 > threshold) — patch has carbon value AND is under threat
- `carbon_threat_score` = AGB_mean × deforestation_b1_smooth64 — joint signal of carbon × threat
- This captures the REDD+ financial incentive for designation without requiring the REDD TIF (which already exists at `data/south_america/ready/REDD/redd_sa.tif`)

#### P2.3 — KBA overlap (if shapefile available)
Key Biodiversity Areas (BirdLife International). If shapefile downloadable:
- `frac_kba` = fraction of patch overlapping any KBA
- `is_kba` = binary
- Blocked on BirdLife data agreement — prioritise after RAISG.

#### P2.4 — Patch isolation features
Computable from patch panel (no new data):
- `dist_nearest_large_patch_km` = distance to nearest patch > 1000 km²
  (corridor potential — isolated patches less likely to be designated than corridor-filling ones)
- `n_patches_within_100km` = count of patches within 100km (landscape context)
- Implementation: build KD-tree over patch centroids per year; query at each patch

#### P2.5 — Patch temporal evolution features
- `patch_size_trend` = log(size_t) − log(size_{t-1}): shrinking patch (being consumed by other PAs?) vs stable
- `patch_age` = consecutive years this patch has existed without designation (older = established boundary)
- Requires: year-to-year CC comparison, tracking patches across years by spatial overlap

#### P2.6 — Country-level governance interaction
Already have policy_b1–b4 (CBD ratification, NBI signals) and pa_momentum_pixels_lag{1,2,3}. Add interaction:
- `momentum_×_proximity` = pa_momentum_pixels_lag1 × (1/dist_wdpa): high momentum near PAs = active expansion corridor
- `country_gdp_per_km2` (if data available): richer countries can afford enforcement, poor countries may protect paper parks

---

### Phase 3 — Evaluation robustness (~1 week)

**Goal**: Every metric claim in the paper is statistically defensible and compared against naive baselines.

#### P3.1 — Naive baseline comparison table (required for paper)
For all test set metrics, report:

| Model | Lift@1% | Recall@5% (macro) | Recall@5% (weighted) | CI |
|---|---|---|---|---|
| Random | — | — | — | — |
| Size-only sort | — | — | — | — |
| Proximity-only sort (-dist_wdpa) | — | — | — | — |
| Size + Proximity | — | — | — | — |
| **Ours (H12 enriched)** | — | — | — | — |

Script: `stage2_patch_baselines.py` (new), outputs table as LaTeX.

#### P3.2 — Size-stratified evaluation
Bin test patches into quartiles by `log_patch_size_km2`. Report Lift@1% and Recall@5% within each bin. This shows whether the model adds value within size classes (not just across them).

| Size quartile | n patches | n positive | Lift@1% model | Lift@1% size-sort | Added value? |
|---|---|---|---|---|---|
| Q1 (smallest) | — | — | — | — | — |
| Q2 | — | — | — | — | — |
| Q3 | — | — | — | — | — |
| Q4 (largest) | — | — | — | — | — |

If model outperforms size-sort within Q1–Q3, that IS a Nature-level result.

#### P3.3 — Bootstrap confidence intervals
1000 bootstrap resamples of the 61 test groups (sample 61 with replacement). Report 95% CIs for all primary metrics and all baseline comparisons. Required for statistical rigor.

#### P3.4 — Per-country evaluation table
Report metrics per country in the test set (all 13 SA countries with at least one test expansion). Specifically highlight Brazil vs rest. Shows where the model generalises well and honestly discloses limitations.

#### P3.5 — Temporal stability
Report Recall@5% year by year (2017, 2018, ..., 2024). Does performance degrade toward 2024? Temporal drift is a legitimate reviewer concern.

#### P3.6 — Cross-regional transfer (patch level)
Build patch panel for SE Asia splits. Score with SA-trained model (zero-shot transfer). Report transfer Lift@1% and Recall@5%.

If patterns transfer: the model is learning universal spatial ecology of PA designation.
If they don't: explicitly discuss as a limitation and explain why (different governance systems, landscape structures).

Required scripts: `build_patch_panel.py` for SE Asia (same script, different STAGE2_DATA_ROOT), score with SA model.

---

### Phase 4 — Hyperparameter tuning (~1 week compute, after P1+P2 locked)

**Only run after the feature set from Phases 1+2 is finalised.** Tuning the wrong feature space wastes compute and produces hyperparameters that need re-tuning after feature changes.

- ≥100 Optuna trials on patch-level LambdaRank
- Search space: num_leaves (31–511), max_depth (-1, 4–8), learning_rate (0.01–0.1), min_child_samples (5–50), subsample (0.5–1.0), colsample_bytree (0.5–1.0), reg_alpha/lambda, lambdarank_truncation_level (50–500)
- Objective: Recall@5% within groups on earlystop set
- Output: new `model1_stage2_patch_best_params.json`
- Expected gain: close remaining gap to 90% Recall after feature enrichment

---

### Phase 5 — Stage 1 + Stage 2 integration (~1 week)

**Goal**: Produce the full conditional probability model and forward predictions.

#### P5.1 — Stage 1 validation
Confirm Stage 1 (Poisson GLM) D² metrics are current. Stage 1 is solved — verify outputs are accessible.

#### P5.2 — Platt calibration of patch scores
Fit logistic regression on logit(patch_scores) from the training or earlystop set. Output: calibrated probabilities P(patch designated | country expands). Required for integration and forward maps.

#### P5.3 — Stage 1 × Stage 2 integration
```
For each year t (2025-2030):
  For each country C:
    expansion_budget = Stage1.predict(C, t)  ← km² predicted to be designated
    patch_probs = Stage2.predict_calibrated(C, t)  ← suitability per patch
    Fill patches from highest score downward until budget exhausted
    → Designated patches for C in t
  Cumulative risk[patch] = 1 - ∏(1 - P(designated, t))
```

Output: cumulative designation probability per patch, 2025-2030. Three scenarios: BAU (historical designation rates), moderate (midpoint to 30%), 30×30 (meets target by 2030).

#### P5.4 — Forward maps
Produce:
- Continental map: cumulative designation probability per patch, coloured by scenario
- Top-100 highest-risk patches in SA: who, where, what ecological value
- Per-country table: how many km² predicted by 2030 per scenario vs how far from 30% target
- Economic exposure in top-risk zones (HNTL × patch probability)

---

### Phase 6 — Science story & RQ3 (~2 weeks)

**Goal**: Answer RQ3 and provide the Nature Sustainability hook.

#### P6.1 — Representation gap analysis (RQ3)
Obtain biodiversity priority layers:
- Option A: GSN_b2 (already have) — use as priority proxy. What is the mean GSN_b2 of predicted-to-be-protected patches vs all unprotected patches?
- Option B: IUCN range maps / global biodiversity hotspots — download and rasterise
- Option C: Protected Planet's Key Biodiversity Area coverage

Analysis: among the top-predicted patches for 2025-2030, what is the distribution of biodiversity value? If high-carbon/low-biodiversity patches dominate, that IS the representation gap finding.

**Key finding to demonstrate**: "Predicted PA expansion under 30×30 is biased toward large, remote, low-human-pressure land blocks. Biodiversity hotspots with high economic pressure (agricultural frontier zones) are systematically under-predicted to be protected — consistent with historical patterns of protecting 'cheap' land rather than ecologically critical land."

#### P6.2 — SHAP analysis on final model
Run SHAP on the enriched, tuned patch model (after Phases 1–4).
- Beeswarm plot: feature effects on patch designation probability
- Interaction plot: size × biodiversity; size × AGB
- Country comparison: do drivers differ between Brazil and Andean countries?
- Temporal SHAP: are recent designations driven by different features than early 2000s?

#### P6.3 — Literature comparison
Benchmark against:
- Venter et al. 2014 (Nature): global PA expansion drivers (regression-based)
- Adams et al. 2019 (Nature Communications): PA effectiveness and selection bias
- Any pixel-level PA prediction paper with reported Lift/Recall

Show that patch-level LambdaRank achieves higher coverage metrics than pixel-level approaches, and explain why (framing mismatch).

---

### Phase 7 — Paper writing (after all above)

**Manuscript gate**: Do not begin writing until:
- [ ] Enriched patch model outperforms size-sort on at least Q1–Q3 size strata
- [ ] Brazil Recall@5% > size-sort baseline
- [ ] Weighted Recall@5% ≥ 75%
- [ ] Cross-regional transfer completed
- [ ] Stage 1+2 integration working
- [ ] RQ3 representation gap analysed
- [ ] All naive baselines documented

**Suggested structure (Nature Sustainability ~3500 words main text)**:
1. **Introduction** (600w): 30×30 urgency, prediction vs prescription gap, framing mismatch problem, this paper's contribution
2. **Results** (1800w):
   - S1: Country-year expansion (Stage 1, 200w)
   - S2: Patch-level selection — what drives designation (600w, SHAP story)
   - S3: Model performance vs baselines — size-stratified (500w)
   - S4: Forward predictions + scenario maps (300w)
   - S5: Representation gap — where 30×30 will and won't protect (200w)
3. **Discussion** (600w): drivers of designation, representation gap implications, limitations (Brazil, data gaps), policy relevance
4. **Methods** (500w): data, Stage 1, Stage 2, patch construction, evaluation, forward pipeline
5. **Extended Data** (~10 figures): per-country metrics, temporal stability, SE Asia transfer, SHAP, forward maps by scenario

---

## Experiment Queue (current priorities)

| Priority | Task | Status | Expected output |
|---|---|---|---|
| **1** | **No-size ablation: train H12 without log_patch_size_km2** | ⬜ Next | Diagnostic: does any signal exist beyond size? |
| **2** | **Add shape + percentile features to build_patch_panel.py** | ⬜ Next | Compactness, perimeter, std_gsn_b2, max_gsn_b2, std_ndvi, frac_high_ndvi |
| **3** | **Fix AGB and add to patch panel** | ⬜ Next | Inspect AGB TIF → fix NaN/scale → add mean+max AGB to patch panel |
| **4** | **Add weighted Recall metric** | ⬜ Next | New metric in stage2_metrics.py |
| **5** | **Rebuild enriched patch panel on Euler** | ⬜ After P1.2+P1.3 | patch_{train,earlystop,test}.parquet with 8+ new patch features |
| **6** | **Retrain with enriched features** | ⬜ After 5 | Full SA training job. Primary gate: does Brazil >size-sort? |
| **7** | **Add RAISG indigenous territories** | ⬜ After data download | frac_indigenous_area, in_indigenous_territory features |
| **8** | **Naive baseline comparison script** | ⬜ After 6 | LaTeX table for paper |
| **9** | **Size-stratified evaluation script** | ⬜ After 6 | Per-quartile Lift/Recall table |
| **10** | **Bootstrap CIs** | ⬜ After 6 | 95% CIs on all primary metrics |
| **11** | **Hyperparameter tuning (≥100 Optuna trials)** | ⬜ After feature set locked | New best_params JSON for patch model |
| **12** | **Build SE Asia patch panel + transfer eval** | ⬜ After 11 | Cross-regional validation |
| **13** | **Platt calibration at patch level** | ⬜ Phase 5 | Calibrated P(patch designated) |
| **14** | **Stage 1 + Stage 2 integration** | ⬜ Phase 5 | Full conditional model |
| **15** | **Forward maps 2025-2030** | ⬜ Phase 5 | BAU/30×30/NGFS scenario maps |
| **16** | **RQ3 representation gap analysis** | ⬜ Phase 6 | Biodiversity vs predicted designation comparison |
| **17** | **SHAP analysis on final enriched model** | ⬜ Phase 6 | Feature importance story for paper |
| **18** | **Literature comparison** | ⬜ Phase 6 | Benchmark table |
| KBA features | dist_kba_km, is_kba | ⏸ Blocked — BirdLife shapefile | High value if available |
| H4 extended training | Extend train to 2016 | ⏸ Discuss with supervisor | Pipeline change required |

---

## Full SA Experiment History

**Pixel-level locked baseline**: H6+H1b+H5. Lift=6.46×, Recall=15.7%, best_iter=136.
**Patch-level baseline (H12)**: Lift=81.32×, Recall=87.7%, best_iter=20. BUT essentially equal to naive size-sort. Feature enrichment required.

| Experiment | Lift@1% | Recall@5% | iter | Verdict |
|---|---|---|---|---|
| Baseline (79 feat, default params) | 2.85× | 14.0% | 149 | starting point |
| 20-trial Optuna retune | 2.06× | 8.4% | 7 | ✗ catastrophic — never retune early |
| 67 feat + temporal year weights | 2.64× | 11.4% | 113 | ✗ both negative |
| H6+H1b (Recall stop + inv_sqrt_npos weights) | 3.73× | 18.1% | 89 | ✓ first both-improved |
| **H6+H1b+H5 (no rank-norm)** | **6.46×** | **15.7%** | **136** | ✓ **pixel locked baseline** |
| H6+H8+H10 (temporal weights + combined stop) | 4.17× | 18.96% | 112 | ✗ proxy 20× did not transfer |
| H6+H1b+H5+H7 (train 2010–2013 only) | 1.51× | 10.5% | — | ✗ catastrophic — too few groups |
| H6+H1b+H5+H11 (patch-context features) | 5.56× | 16.6% | 50 | ✗ pixel unit wrong; patch features constant within patch |
| Tier 1: spatial diffusion (steps=10, α=2.0) | 6.15× | 17.2% | — | ✗ Recall ceiling confirmed |
| **H12: patch-level LambdaRank** | **81.32×** | **87.7%** | **20** | ✓ **framing fix works; BUT ≈ size-sort; feature enrichment next** |
| *Naive size-sort (log_patch_size_km2 only)* | *80.38×* | *88.8%* | *—* | *reference baseline — model must beat this* |

**Locked architectural decisions (all confirmed on full SA patch model):**
- Ranking unit: connected-component patches of WDPA_prev==0 land
- H6: Recall@5% early stopping
- H1b: inv_sqrt_npos group-norm weights
- H5: rank normalisation OFF (absolute features carry cross-country signal)
- W9a eco sub-groups: OFF at patch level (not applicable)

---

## Settled Decisions

| Decision | Value | Rationale |
|---|---|---|
| Ranking unit | Connected-component patches (WDPA_prev==0) | Framing fix: governments designate polygons not pixels |
| Engine | LightGBM LambdaRank | Validated |
| Early stopping | Recall@5% within groups (H6) | Directly optimises publication metric |
| Sample weights | inv_sqrt_npos (H1b) | Gradient deconcentration |
| Rank normalisation | Off (H5) | Absolute features carry global information |
| Primary metrics | Lift@1% + Recall@5% within expansion groups | — |
| Secondary metrics | Weighted Recall@5%, size-stratified Recall | Needed for Nature |
| Temporal split | Train 2001–2013, earlystop 2014–2016, test 2017–2024 | H4 (extend to 2016) needs supervisor sign-off |
| Proxy screening | Direction only | Quantitatively unreliable |
| Ensembles | Forbidden | Supervisor directive |
| Hyperparameter retuning | Phase 4 only, after features locked | Early retune catastrophic |
| Naive baselines | Must be documented and reported | Required for top-journal submission |

---

## Data Paths

| Dataset | Location |
|---|---|
| SA pixel splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet` |
| SA patch splits | `euler:$SCRATCH/data/south_america/ml/main/patch_{train,earlystop,test}.parquet` |
| SA mini-sample | `data/south_america/mini_sample.parquet` |
| H12 patch model | `data/south_america/ml/models/model1_lgbm_stage2_patch_20260617_154235.pkl` |
| H12 metrics | `outputs/south_america/results/ml_models/model1_lgbm_stage2_patch_metrics_20260617_154235.json` |
| H12 scored test | `outputs/south_america/results/ml_models/model1_lgbm_stage2_patch_scored_20260617_154235.parquet` |
| Pixel model (H6+H1b+H5) | `data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl` |
| Patch best params | `scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json` (reuse pixel params for now) |
| AGB TIF | `data/south_america/ready/AGB/agb_sa.tif` |
| REDD TIF | `data/south_america/ready/REDD/redd_sa.tif` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` |

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)` — backbone CRS is LOCAL_CS, `crs.to_epsg()` returns None.

---

## Paused

- SE Asia Stage 2 — resume at Phase 3 cross-regional transfer
- USA Stage 2 — deprioritised (may serve as second cross-regional hold-out)
- Forward pipeline — Phase 5

## Out of Scope (Paper 1)

- Ensemble methods, sub-models, neural networks (Paper 2)
- Survival/competing-risks framing
- Tropical Africa
- Sub-national governance data (Paper 2 with richer institutional features)
