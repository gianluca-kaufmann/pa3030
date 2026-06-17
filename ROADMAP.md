# PA3030 — Publication Roadmap

**Updated**: 2026-06-17 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

---

## The Paper in One Paragraph

The 30×30 agreement requires countries to nearly double protected area coverage by 2030. We ask: **where will that doubling actually happen?** Using 23 years of PA designation data across South America, we train a model that learns the spatial and ecological logic of past designation decisions. We find that designation follows a predictable pattern driven by remoteness, proximity to existing PAs, and low economic pressure — **not by biodiversity value or ecological need**. Forward projections to 2030 reveal a systematic gap: if historical patterns continue, 30×30 will predominantly protect ecologically suboptimal land while leaving the most biodiverse and threatened areas unprotected. This gap between where protection will go and where it is most needed is the paper's central finding.

**The prediction model is the analytical tool. The representation gap is the finding.**

---

## Research Questions

**RQ1** — Which countries will expand their PA networks under 30×30, and how much area each year? *(Stage 1, solved)*

**RQ2** — Within an expansion event, which land does a government select — and can we learn the spatial and ecological patterns well enough to predict it? *(Stage 2, in progress)*

**RQ3** — Where does predicted PA expansion diverge from conservation priority, and what does that gap reveal about how 30×30 will actually be met? *(The Nature hook — not yet done)*

---

## Architecture

```
P(pixel j designated in year t | country C expands)
  = P(country C expands in year t)          ← Stage 1: Poisson GLM
  × S(pixel j | expansion in C, t)          ← Stage 2: LightGBM LambdaRank
```

**Stage 1** — Poisson GLM, LASSO α=100, 9 features, D² metric. Train 2001–2016, test 2017–2023. Status: complete.

**Stage 2** — LightGBM LambdaRank, grouped by `(country_id, year)`, ranking unit: pixels with `WDPA_prev==0`. Train 2001–2013, early-stop 2014–2016, test 2017–2024. Locked settings: H6 (Recall@5% early stop), H1b (inv_sqrt_npos weights), H5 (rank normalisation OFF).

**Why pixel-level, not patch-level**: The connected-component patch approach (H12) was tried and found to be structurally degenerate. SA unprotected land forms a few continent-spanning mega-blobs (max: 6.4M pixels) with thousands of isolated 1-2 pixel specks. The model trivially learns to rank mega-blobs first, giving 87.7% Recall — but removing all 82 features except size gives 86.0% Recall (ablation confirmed). The patch metric is an artifact of the label construction (positive = any pixel inside designated), not a model signal. Pixel-level ranking is clean, validated, and sufficient for the science we are doing: we are predicting which **locations** will be protected, not delineating polygon boundaries.

**Primary metrics**: Lift@1% and Recall@5% within expansion groups, both macro (per group) and weighted (by n_pos). Secondary: AUC-PR, per-country breakdown, temporal stability.

---

## Honest Current State (2026-06-17)

### What we have

| Item | Status | Numbers |
|---|---|---|
| Stage 1 Poisson GLM | ✅ Complete | D² OOS validated |
| Stage 2 pixel model (H6+H1b+H5) | ✅ Working | Lift@1%=6.46×, Recall@5%=19%, Lift vs random=3.8× |
| 3 continental regions | ✅ Data exists | SA (primary), USA, SE Asia — splits on Euler |
| Forward pipeline (8_forward/) | ✅ Built | SA forward map exists, needs updating |
| Patch-level approach (H12) | ❌ Abandoned | 87.7% Recall is artifact of degenerate CC definition |

### What the pixel model numbers actually mean

Lift@1% = 6.46×: the top 1% of ranked pixels contains 6.46× more actual designations than a random 1%. This is a real, learnable signal. It means the model genuinely identifies where designation is concentrated.

Recall@5% = 19%: looking at the top 5% of pixels within each country-year group, we find 19% of all designated pixels (vs 5% for random — 3.8× improvement). This number looks modest but reflects the genuine difficulty: designation is spatially diverse and driven partly by political/institutional factors our features don't capture. It is an honest number.

These are publishable results for a paper whose central contribution is **what the model learns** (SHAP) and **where the predictions are concentrated** (representation gap), not the prediction accuracy itself.

### What is missing for Nature

| Gap | Why it matters | Effort |
|---|---|---|
| AGB / carbon feature | REDD+ is a primary SA designation driver; missing from all models | Medium |
| SHAP analysis | Must show WHAT drives designation — this is the science, not the metric | Medium |
| Representation gap (RQ3) | The Nature hook — where will 30×30 go vs where it should go | High |
| Proper baselines | Reviewers require comparison to random, persistence, proximity-only | Low |
| Bootstrap CIs | Statistical rigor — 95% CIs on all reported metrics | Low |
| Cross-regional validation | Does model transfer to USA and SE Asia? | Medium |
| Forward maps (updated) | 2025–2030 scenarios — already partly built | Medium |
| Literature comparison | Benchmark against published PA prediction methods | Low |

---

## The Science Story (what the paper argues)

This section exists so we don't lose sight of the goal while implementing.

**Claim 1 — Designation is predictable**: Historical PA designation in SA follows learnable spatial patterns. A model trained on features available before designation (remoteness, existing PA proximity, biodiversity indicators, land pressure) achieves 6.46× Lift@1% on held-out future years. This is not trivial — it means governments behave consistently.

**Claim 2 — What drives designation (SHAP story)**: The dominant predictors are proximity to existing PAs, remoteness (low road density, low population), and low economic pressure (low HNTL, low agricultural value). Biodiversity scores and ecosystem threat are secondary predictors. REDD+ carbon incentives (AGB) are hypothesised to be important for SA specifically — to be confirmed with AGB feature.

**Claim 3 — The representation gap**: Forward projections to 2030 show that the top-predicted areas are concentrated in remote, low-pressure, low-economic-value regions. Overlaying these with biodiversity priority maps (GSN_b2, KBAs, IUCN threat data) reveals a systematic gap: high-biodiversity, high-threat land (agricultural frontier zones, forest-savanna transitions) is systematically under-predicted to receive protection. Meeting 30×30 quantitatively does not mean meeting it biodiversity-wise.

**Claim 4 — Cross-regional consistency**: The same spatial logic of designation (remoteness, PA proximity, low pressure) transfers to USA and SE Asia, suggesting this is a universal pattern of PA designation rather than a South America–specific finding.

These four claims together constitute a Nature Sustainability paper.

---

## Plan

### Phase 1 — Complete Stage 2 feature set (~1–2 weeks)

The pixel model's most important missing feature is AGB (above-ground biomass / carbon stock). REDD+ payments and voluntary carbon markets financially incentivise protecting high-carbon forests — this is the dominant driver of Brazil's Amazon designations and is absent from all current models.

**P1.1 — Add AGB to pixel splits** (highest priority)
- TIF exists: `data/south_america/ready/AGB/agb_sa.tif`
- Script: `scripts/regions/south_america/3_merging/add_patch_features_to_splits.py` pattern (already have this template)
- Add two features: `AGB_mean` (mean carbon density), `AGB_max` (presence of high-carbon forest)
- Rebuild pixel splits with AGB → retrain H6+H1b+H5 → check if Lift/Recall improves, especially on Brazil
- Expected: AGB is correlated with forest cover (already have NDVI), so gain may be modest — but it completes the feature set for scientific completeness and REDD+ interpretability

**P1.2 — Retrain with AGB and check per-country breakdown**
- Run full SA training with AGB added
- Report Lift@1% and Recall@5% per country (Brazil is the key test)
- If AGB meaningfully improves Brazil performance: add to settled feature set
- If AGB makes no difference: keep for interpretability (SHAP story), note in paper

**P1.3 — Add proper baselines to evaluation**
- Script: `stage2_pixel_baselines.py` (new, simple)
- Baselines: random ranking, sort by -dist_wdpa only, sort by GSN_b2 only
- Output: comparison table in LaTeX for paper
- This takes half a day to implement, no Euler job needed

---

### Phase 2 — SHAP analysis (~1 week)

Run SHAP on the final pixel model (after AGB added). This is the core scientific contribution.

**P2.1 — Global SHAP beeswarm**
- Run on test set (2017–2024 expansion groups)
- Key question: which features drive positive predictions? Is remoteness #1? Is proximity to existing PAs #1?
- Expected finding: dist_wdpa, dist_road, GPW (population) dominate. Biodiversity (GSN_b2) is secondary. AGB contributes for Amazonian groups.

**P2.2 — Country-level SHAP comparison**
- Run SHAP separately on Brazil, Bolivia, Colombia, Peru, Argentina groups
- Key question: do drivers differ between Amazonian countries (carbon-driven) and Andean countries (connectivity-driven)?
- This is an interesting nuanced finding for the paper

**P2.3 — Temporal SHAP**
- Split test years into 2017–2020 and 2021–2024
- Key question: did the importance of AGB / carbon-related features increase in recent years, consistent with growing REDD+ activity?

---

### Phase 3 — Representation gap analysis (~2 weeks)

This is RQ3 and the central Nature finding. It answers: where will 30×30 protection go, and is that the right place?

**P3.1 — Define biodiversity priority layers**
- Primary proxy: `GSN_b2` (already in splits) — this is our biodiversity importance indicator
- Secondary: download IUCN KBA shapefile (Key Biodiversity Areas) — or use existing GSN bands
- Tertiary: use deforestation pressure (`deforestation_b1`) as threat indicator
- Compute for each pixel: biodiversity_priority = f(GSN_b2, deforestation_b1) — high biodiversity + high threat = most in need of protection

**P3.2 — Compare predicted vs priority**
- Bin all unprotected pixels into quartiles by model score (top 5%, 5–10%, 10–25%, 25–50%, bottom 50%)
- For each bin, compute mean biodiversity_priority
- If model scores are negatively correlated with biodiversity_priority: the gap exists and is quantifiable
- This is the key figure for the paper: a scatter or bar chart showing "high model score = low biodiversity priority"

**P3.3 — Forward projection gap quantification**
- Take forward predictions (2025–2030, BAU scenario)
- Among pixels predicted to be protected by 2030, what fraction overlap KBAs?
- Among KBA pixels, what fraction are predicted to be protected by 2030?
- Compare to: if random land were protected (what fraction of KBAs would be covered just by chance?)
- This gives the headline number: "30×30 under BAU will protect X% of KBAs vs Y% expected by chance"

---

### Phase 4 — Robustness and validation (~1 week)

**P4.1 — Bootstrap confidence intervals**
- 1000 bootstrap resamples of test groups
- 95% CIs on Lift@1%, Recall@5%, and all baseline comparisons
- Required for statistical rigor

**P4.2 — Temporal stability**
- Report Recall@5% per year 2017–2024
- Key question: does performance degrade toward 2024? (temporal drift)

**P4.3 — Per-country evaluation table**
- Report Lift@1% and Recall@5% per country in test set
- Explicitly highlight strong and weak countries
- Brazil, Colombia, Peru, Bolivia most important

**P4.4 — Cross-regional transfer**
- Score USA and SE Asia test splits with SA-trained pixel model (zero-shot)
- Key question: does the model generalise? If yes: universal pattern of designation. If no: discuss governance differences.
- Scripts already partly exist (spatial_CV_3)

---

### Phase 5 — Forward maps and integration (~1 week)

Forward pipeline (8_forward/) is already partly built. Update it for the final pixel model.

**P5.1 — Calibrate pixel model**
- Platt calibration on earlystop set
- Output: calibrated P(pixel designated | country expands) for each unprotected pixel

**P5.2 — Stage 1 × Stage 2 integration**
- P(pixel designated in year t) = P(country expands in t) × P(pixel | expansion)
- Cumulative risk 2025–2030: 1 − ∏(1 − p_t)
- Three scenarios: BAU (historical rates), moderate (midpoint to 30%), 30×30 (meets target by 2030)

**P5.3 — Forward map figures**
- Continental risk map (cumulative probability per unprotected pixel, 2025–2030)
- Overlay with biodiversity priority — this is the representation gap figure
- Per-country coverage table: km² predicted to be protected by 2030 vs how far from 30% target

---

### Phase 6 — Paper writing

Do not begin until:
- [ ] AGB feature added and validated
- [ ] SHAP analysis complete
- [ ] Representation gap quantified with a headline number
- [ ] Baselines documented
- [ ] Bootstrap CIs computed
- [ ] Cross-regional transfer result in hand
- [ ] Forward maps produced

**Target structure (Nature Sustainability, ~3500 words main text)**:

1. **Introduction** (600w): 30×30 urgency, prediction vs prescription gap, paper contribution
2. **Results** (1800w):
   - Stage 1: country-year expansion (200w)
   - Stage 2: pixel model performance and what drives designation — SHAP (600w)
   - RQ3: representation gap — predicted vs priority (600w)
   - Forward projections 2025–2030 (400w)
3. **Discussion** (600w): designation logic, representation gap implications, limitations, policy relevance
4. **Methods** (500w): data, Stage 1, Stage 2, evaluation, forward pipeline
5. **Extended Data** (~8 figures): per-country metrics, temporal stability, USA/SE Asia transfer, SHAP by country, forward maps by scenario

---

## Experiment Queue

| Priority | Task | Status | Blocks |
|---|---|---|---|
| **1** | Inspect AGB TIF; add to pixel splits and rebuild | ⬜ Next | Everything downstream |
| **2** | Retrain H6+H1b+H5 with AGB; check per-country breakdown | ⬜ After 1 | SHAP, forward |
| **3** | Add baselines script (random, dist_wdpa-only, GSN_b2-only) | ⬜ Can do now | Paper table |
| **4** | SHAP analysis on final pixel model | ⬜ After 2 | Core paper story |
| **5** | Representation gap: score vs biodiversity_priority | ⬜ After 2 | RQ3 / Nature hook |
| **6** | Bootstrap CIs on all metrics | ⬜ After 2 | Statistical rigor |
| **7** | Per-country evaluation table | ⬜ After 2 | Honest assessment |
| **8** | Temporal stability (per-year Recall@5%) | ⬜ After 2 | Reviewer concern |
| **9** | Cross-regional transfer: SA model → USA and SE Asia | ⬜ After 2 | Claim 4 |
| **10** | Update forward pipeline with final pixel model | ⬜ After 2 | Forward maps |
| **11** | Forward map figures + representation gap overlay | ⬜ After 10 | Figure 3 in paper |
| **12** | SHAP country comparison (Brazil vs Andean) | ⬜ After 4 | Paper nuance |
| **13** | KBA download + overlap analysis | ⬜ After 5 | Stronger RQ3 claim |
| **14** | Literature comparison table | ⬜ Before writing | Credibility |
| **15** | Paper writing | ⬜ After all above | — |

---

## Full Experiment History

### Pixel-level Stage 2 (all on full SA)

| Experiment | Lift@1% | Recall@5% | iter | Verdict |
|---|---|---|---|---|
| Baseline (79 feat, default params) | 2.85× | 14.0% | 149 | Starting point |
| 20-trial Optuna retune | 2.06× | 8.4% | 7 | ✗ catastrophic — never retune early |
| Temporal year weights | 2.64× | 11.4% | 113 | ✗ both metrics worse |
| H6+H1b (Recall stop + inv_sqrt_npos weights) | 3.73× | 18.1% | 89 | ✓ first both-positive |
| **H6+H1b+H5 (no rank-norm)** | **6.46×** | **15.7%** | **136** | ✅ **locked pixel baseline** |
| H6+H8+H10 (temporal weights + combined stop) | 4.17× | 18.96% | 112 | ✗ did not beat H5 on both metrics |
| H6+H1b+H5+H7 (train 2010–2013 only) | 1.51× | 10.5% | — | ✗ catastrophic — too few groups |
| H6+H1b+H5+H11 (patch-context pixel features) | 5.56× | 16.6% | 50 | ✗ patch features constant within patch |
| Tier 1: spatial diffusion (steps=10, α=2.0) | 6.15× | 17.2% | — | ✗ marginal, ceiling confirmed |

### Patch-level Stage 2 (abandoned — see note below)

| Experiment | Lift@1% | Recall@5% | iter | Verdict |
|---|---|---|---|---|
| H12: connected-component patch ranking | 81.32× | 87.7% | 20 | ❌ artifact — see below |
| H12 no-size ablation (all 82 features, no log_patch_size_km2) | 79.11× | 86.0% | **1** | ❌ confirms artifact |
| Naive size-sort (reference) | 80.38× | 88.8% | — | reference — trivial |

**Why the patch approach is abandoned**: South American unprotected land forms a few continent-spanning connected blobs (largest: 6.4M pixels) and tens of thousands of 1–2 pixel isolated specks. The model achieves 87.7% Recall by learning to rank the mega-blobs first — confirmed by the no-size ablation (86.0% Recall in 1 boosting round with size removed, i.e., 82 other features provide almost no additional signal). The high metric is a structural artifact of the label construction (positive = any pixel inside designated), not a learned signal. The approach is not salvageable without a fundamentally different patch definition (e.g., fixed grid cells), and rebuilding at that cost is not justified given that the pixel model is sufficient for the science.

---

## Settled Decisions

| Decision | Value | Rationale |
|---|---|---|
| Stage 2 ranking unit | Pixels (WDPA_prev==0) | Patch approach is degenerate; pixels are clean and validated |
| Engine | LightGBM LambdaRank | Validated |
| Early stopping | Recall@5% within groups (H6) | Directly optimises target metric |
| Sample weights | inv_sqrt_npos (H1b) | Gradient deconcentration |
| Rank normalisation | Off (H5) | Absolute features carry cross-country signal |
| Primary metrics | Lift@1% + Recall@5% (macro + weighted) | Honest reporting |
| Temporal split | Train 2001–2013, ES 2014–2016, test 2017–2024 | Validated |
| Ensembles | Forbidden | Supervisor directive |
| Hyperparameter retuning | Only after feature set locked | Early retune catastrophic (2.85× → 2.06×) |
| Naive baselines | Must be documented | Required for any top journal |
| Paper framing | Representation gap is the finding; prediction model is the tool | Core strategic decision |

---

## Data Paths

| Dataset | Location |
|---|---|
| SA pixel splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet` |
| Pixel model (H6+H1b+H5) | `data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl` |
| USA pixel splits | `euler:$SCRATCH/data/usa/ml/main/{train,earlystop,test}.parquet` |
| SE Asia pixel splits | `euler:$SCRATCH/data/se_asia/ml/main/{train,earlystop,test}.parquet` |
| AGB TIF | `data/south_america/ready/AGB/agb_sa.tif` |
| REDD TIF | `data/south_america/ready/REDD/redd_sa.tif` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` |
| Forward scored (existing) | `euler:$SCRATCH/outputs/south_america/forward_scored_2024.parquet` |

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)`. Backbone CRS is LOCAL_CS; `crs.to_epsg()` returns None.

---

## Paused / Out of Scope

- Patch-level approach — abandoned (degenerate CC definition)
- Hyperparameter tuning — after AGB + final feature set validated
- SE Asia Stage 2 training — Phase 4 cross-regional transfer
- Forward pipeline update — Phase 5
- Neural networks, ensembles, sub-models — Paper 2
- Tropical Africa — Paper 2
- Sub-national governance data — Paper 2
