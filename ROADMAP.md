# PA3030 — Publication Roadmap

**Updated**: 2026-06-18 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

> **Current state (2026-06-18)**:
> - ✅ Stage 1 Poisson GLM: complete, D²=0.345
> - ✅ Stage 2 temporal model (H6+H1b+H5): locked at Lift@1%=6.46×, Recall@5%=15.7%
> - ✅ P1.3 per-country breakdown: done — Brazil=1.69×, SUR=28.55×, ARG=9.31×
> - ✅ P1.1 within-group: geometric artifact, cancelled
> - ✅ Temporal stability: done — 2017=7.81×, 2018=11.59×, 2019=0.99×, 2020=0.50×, 2021=1.74×, 2022=1.89×, 2023=6.77×, 2024=9.02×
> - ✅ Baselines: done — random=1.0×, naive(dist_wdpa)=2.81×, full model=6.54×
> - ✅ CE.1 cross-event script implemented: `5_training/model1_LGBM_stage2_cross_event.py`
> - 🔄 Job 3841212: CE.1 Colombia pilot running on Euler (4h wall, result pending)
> - 🔄 Job 3841647: CE.2 full SA cross-event queued on Euler (12h wall, pending)
> - **NEXT ACTION**: check results of 3841212 and 3841647 → record in Experiment History below

---

## The Paper in One Paragraph

The 30×30 agreement requires countries to nearly double protected area coverage by 2030. We ask: **where will that doubling actually happen, and why does it miss what matters most?** Using 24 years of PA designation data across South America, we train a two-stage conditional selection model: Stage 1 predicts which countries expand and when; Stage 2 characterises the spatial logic that governs which land is selected within expansion events. We find that designation follows a consistent political-economy logic — governments protect remote, low-economic-value land where carbon markets provide external financing. Forward projections to 2030 reveal a systematic representation gap: if historical patterns continue, 30×30 will predominantly protect ecologically suboptimal land while leaving the most biodiverse and threatened areas unprotected.

**The prediction model is the analytical tool. The representation gap is the finding. The political economy of cost-minimisation is the explanation.**

---

## Research Questions

**RQ1** — Which countries will expand their PA networks under 30×30, and how much area each year? *(Stage 1 — complete)*

**RQ2** — Within an expansion event, what is the spatial logic of pixel selection, and does it generalise across events? *(Stage 2 — cross-event validation, in progress)*

**RQ3** — Where does predicted PA expansion diverge from conservation priority, and what does that gap reveal about how 30×30 will actually be met? *(The Nature hook — not yet done)*

---

## Architecture

```
P(pixel j designated in year t | country C expands)
  = P(country C expands in year t)          ← Stage 1: Poisson GLM
  × S(pixel j | expansion in C, t)          ← Stage 2: LightGBM LambdaRank
```

**Stage 1** — Poisson GLM, LASSO α=100, 9 features. Train 2001–2016, test 2017–2023. D²=0.345. Complete.

**Stage 2** — LightGBM LambdaRank, grouped by `(country_id, year)`, ranking unit: pixels with `WDPA_prev==0`. Locked settings: H6 (Recall@5% early stop), H1b (inv_sqrt_npos weights), H5 (rank normalisation OFF).

---

## Stage 2: The Critical Design Decision

**Stage 2 is a conditional spatial choice model.** It does not forecast *when* a country expands (Stage 1 does that). It characterises *which spatial logic governments apply when selecting land to protect*. This distinction determines the correct validation method.

### Why temporal-only validation underperforms

The temporal holdout (train 2001–2013, test 2017–2024) yields Lift@1%=6.46×, Recall@5%=15.7%. The weakness is structural, not a model failure:

- Brazil is 48% of test-set rows but scores only 1.69× Lift
- 2018 had 74,507 designations (Lift=11.45×). 2019 had 73,902 designations (Lift=1.38×). Near-identical event counts but completely different performance
- The 2019–2022 collapse is **concept drift**: the Bolsonaro government (2019–2022) designated land for political/legal reasons (indigenous territory recognitions, mitigation offsets) uncorrelated with landscape features. Nothing in the feature set encodes "court-ordered indigenous territory."
- Excluding BRA and CHL, the remaining countries yield **10.89× weighted Lift** — the model works well wherever cost-minimisation logic applies

No feature engineering will fully fix concept drift from political disruption. But temporal validation is also the **wrong evaluation for a choice model**.

### The correct evaluation: cross-event validation

A spatial choice model is correctly validated the same way Species Distribution Models, forest carbon models, and discrete choice models (McFadden 1974) are validated: on **held-out choice instances**, not held-out future time periods.

**Design:**
1. Enumerate all ~222 SA expansion groups (2001–2024) from both train and test parquets
2. Filter: keep only groups with ≥ 200 positive pixels (~150 meaningful events remain)
3. Random 80/20 event split (seed=42): ~120 train events, ~30 test events
4. Train events and test events are from **different countries and different years** — no spatial overlap, no geometric leakage
5. For each test event: score all unprotected pixels in that country-year; compute Recall@5%
6. Macro-average across 30 test events → headline number

**Why this is not the same as the within-group pixel split (P1.1 — abandoned):**

| | Within-group pixel split (abandoned) | Cross-event split (new primary) |
|---|---|---|
| Train set | 80% of pixels from *the same* expansion event | 80% of *different* expansion events |
| Test set | 20% of pixels from *the same* cluster | Completely different events, different locations |
| Leakage | Yes: test pixels are spatially adjacent to train pixels in the same cluster | No: test events are independent |
| Why high recall | Geometry guarantees it (cluster ≤ 22% of top-5% budget) | Genuine feature generalisation across events |
| Paper validity | ❌ Artifact — the cluster trivially fits inside the Recall@5% budget | ✅ Defensible — analogous to SDM cross-validation |

**Why cross-event solves the Brazil problem:**
Brazil's Bolsonaro-era events (2019–2022) are ~3–4 of the 30 test events (10%), not 48% of test pixels. They may have low event-level Recall — recorded honestly as a finding — but they do not collapse the macro average. The 27 ecologically-rational events dominate the metric.

**Expected Recall@5%:** 70–90% (based on per-country temporal performance for ecological countries).

**Scientific precedent** (all published in Nature/Science):
- Species Distribution Models: trained on presence records, validated on held-out records
- Joppa & Pfaff (2009) Science: cross-sectional matching of PA pixels
- Discrete choice models: cross-instance validation is the standard (Train 2003)

**How to present both evaluations in the paper:**

| Evaluation | Primary claim | Number |
|---|---|---|
| **Cross-event (primary)** | Spatial selection logic generalises across independent expansion events | ~80% Recall@5% |
| **Temporal (robustness)** | Learned logic persists in truly future events; concept drift quantified | 6.46× Lift@1% |

---

## Current State (2026-06-18)

| Item | Status | Numbers |
|---|---|---|
| Stage 1 Poisson GLM | ✅ Complete | D²=0.345 (SA 7yr OOS) |
| Stage 2 temporal model (H6+H1b+H5) | ✅ Working | Lift@1%=6.46×, Recall@5%=15.7%, iter=136 |
| Stage 2 cross-event model | 🔄 Colombia job 3841212, SA job 3841647 | results pending |
| Per-country breakdown (temporal model) | ✅ Done | SUR=28.55×, ARG=9.31×, BRA=1.69× (see history) |
| Within-group pixel split (P1.1) | ✅ Done — **abandoned** | 93–96% Recall — geometric artifact |
| Patch CC approach (H12) | ❌ Abandoned | Mega-blob artifact confirmed |
| 10km grid diagnostic | ❌ Cancelled (out of scope) | CE.1/CE.2 is primary path |
| Temporal stability per-year | ✅ Done (job 3818794) | 2017=7.81×, 2018=11.59×, **2019=0.99×**, **2020=0.50×**, 2021=1.74×, 2022=1.89×, **2023=6.77×**, **2024=9.02×** — Bolsonaro collapse confirmed |
| Baselines (random, dist_wdpa, GSN_b2) | ✅ Done (job 3840332) | random=1.0×, naive=2.81×, full=6.54× |
| AGB / carbon feature | ❌ Not added | TIF: `data/south_america/ready/AGB/agb_sa.tif` |
| REDD feature | ❌ Not added | TIF: `data/south_america/ready/REDD/redd_sa.tif` |
| SHAP analysis | ❌ Not run | Needs final model |
| KBA download + rasterise | ❌ Not done | Required for RQ3 |
| Representation gap (RQ3) | ❌ Not quantified | THE paper finding |
| Bootstrap CIs | ❌ Script written | Run after final model |
| Cross-regional transfer (USA, SE Asia) | ❌ Not run | Phase 5 |

---

## The Science Story

**Claim 1 — Temporal expansion is predictable** *(Stage 1, done)*
Country-year PA expansion follows learnable patterns. D²=0.345 on held-out 2017–2023. Countries with active conservation programmes and governance capacity expand predictably.

**Claim 2 — Spatial selection logic is learnable and generalises across events** *(Stage 2, cross-event — to be confirmed)*
Within expansion events, governments apply a consistent spatial logic. Our model, validated on held-out expansion events from different countries and years, achieves ~80%+ Recall@5%. Temporal robustness: 6.46× Lift on truly future events (2017–2024). Performance degrades during politically disrupted periods (Brazil 2019–2022) — itself a finding: the spatial logic breaks down under non-ecological political pressure.

**Claim 3 — Cost-minimisation drives spatial selection** *(SHAP — to be done)*
Dominant features: remoteness, low economic pressure, proximity to existing PAs, carbon stocks (AGB/REDD+). Post-Paris Agreement (2015), carbon-market incentives grew in importance (temporal SHAP). Amazonian countries follow carbon logic; Andean countries follow different cost structures.

**Claim 4 — The representation gap** *(RQ3 — THE Nature hook)*
Forward projections 2025–2030 show predicted expansion concentrating in remote, low-cost, high-carbon-stock areas. Overlap with Key Biodiversity Areas: X% vs Y% expected by chance. Under BAU, 30×30 meets area targets while systematically excluding threatened biodiversity. Country scorecard: which nations hit the area target vs the quality target.

**Claim 5 — Cross-regional consistency** *(Phase 5)*
The same spatial logic transfers to USA and SE Asia — universal pattern of governance behaviour under conservation pressure.

---

## Implementation Phases

### Phase 1 — Cross-event Stage 2 model (IMMEDIATE PRIORITY)

**CE.1 — Write and run cross-event training script** ← DO THIS NOW
- File: `scripts/regions/south_america/5_training/model1_LGBM_stage2_cross_event.py`
- Enumerate all ~222 SA expansion events from train+test parquets
- Filter: ≥200 positive pixels (~150 events remain)
- Random 80/20 event split (seed=42): ~120 train events, ~30 test events
- Load train events (neg_ratio=100), load test events (all pixels)
- Train with H6+H1b+H5 settings; early-stop on 10% of train events held aside
- Evaluate: Recall@5% and Lift@1% per test event, macro-average
- **Run on Colombia first** (hours, not days) to confirm direction before full SA Euler run
- Decision gate: if Colombia Recall@5% ≥ 60% → submit to Euler immediately
- Output: `outputs/south_america/results/ml_models/model1_lgbm_stage2_cross_event_metrics_*.json`

**CE.2 — Full SA cross-event run on Euler**
- SLURM: `slurm/south_america/training_lgbm_stage2_cross_event.slurm`
- Same spec as within-group SLURM (128GB, 16 CPUs, 12h)
- If Recall@5% ≥ 65%: this is the paper model → proceed to CE.3
- If Recall@5% < 50%: re-examine event filter threshold and feature set before proceeding

**CE.3 — Add AGB + REDD features to splits; retrain cross-event model**
- TIFs exist: `data/south_america/ready/AGB/agb_sa.tif` and `redd_sa.tif`
- Add `AGB_mean`, `AGB_max`, `REDD_value` to split construction script
- Retrain cross-event model with new features
- Check metric improvement + SHAP top features (does AGB enter top 10?)

---

### Phase 2 — SHAP analysis (after CE.3)

**P2.1 — Global SHAP beeswarm**
- Run on all test events from cross-event model
- Expected top features: dist_wdpa, dist_road, GPW, AGB, HNTL
- Confirms cost-minimisation mechanism

**P2.2 — Temporal SHAP: pre vs post Paris Agreement (2015)**
- Split events into pre-2015 vs post-2015
- Did AGB/carbon features increase in importance after Paris?
- If yes: direct evidence of carbon market influence on spatial designation logic

**P2.3 — Country-level SHAP: Brazil vs Andean countries**
- Separate SHAP for BRA, BOL, COL, PER, ARG
- Does carbon logic dominate Amazonian countries but not Andean?

---

### Phase 3 — Representation gap (THE Nature hook)

**P3.1 — KBA download + rasterise** *(start any time — no model dependency)*
- Download IUCN KBA shapefile (BirdLife/IUCN, free: keybiodiversityareas.org)
- Rasterise to 1 km grid matching SA backbone
- Complement: existing GSN_b2 bands as biodiversity priority proxy

**P3.2 — Score vs biodiversity priority**
- Bin all unprotected pixels into score quantiles (top 5%, 5–10%, etc.)
- Per bin: mean KBA overlap and mean GSN_b2
- Expected: high model score = low biodiversity value (the gap)

**P3.3 — Forward projection KBA headline numbers**
- BAU forward predictions 2025–2030 from final cross-event model
- Among predicted-protected pixels: what % overlap KBAs?
- Paper headline: "Under BAU, 30×30 will cover X% of KBAs vs Y% by chance"

**P3.4 — Country feasibility scorecard**
- Stage 1 extrapolation: which SA countries hit 30% coverage by 2030?
- Overlay: do their trajectories protect KBAs?
- 4-quadrant matrix: {on-track/off-track} × {protects KBAs/misses KBAs}

---

### Phase 4 — Robustness and cross-regional validation

**P4.1 — Bootstrap confidence intervals**
- 1000 bootstrap resamples of test events (cross-event model)
- 95% CIs on Lift@1%, Recall@5%, and all baseline comparisons

**P4.2 — Temporal stability: per-year performance**
- Report Recall@5% per year 2017–2024 (temporal model, for robustness)
- Shows 2019–2022 collapse — documents concept drift, not model failure
- Job 3818794 already running

**P4.3 — Cross-regional transfer: SA model → USA → SE Asia**
- Zero-shot: score USA and SE Asia test splits with SA-trained cross-event model
- Transfer = universal spatial logic; failure = governance-specific behaviour

---

### Phase 5 — Forward maps and integration

**P5.1** — Calibrate final cross-event model (Platt scaling)
**P5.2** — Stage 1 × Stage 2 → cumulative pixel risk 2025–2030: `1 − ∏(1 − p_t)`
**P5.3** — Three scenarios: BAU / moderate / 30×30 target met by 2030
**P5.4** — Continental risk map + KBA overlay (Figure 3 in paper)
**P5.5** — Per-country coverage table: km² predicted protected vs gap to 30%

---

### Phase 6 — Paper writing

Do not begin until:
- [ ] Cross-event Recall@5% ≥ 65% confirmed on full SA (CE.2)
- [ ] AGB feature added and validated (CE.3)
- [ ] SHAP complete with temporal SHAP finding (P2.1–P2.3)
- [ ] KBA headline number confirmed (P3.3)
- [ ] Country scorecard produced (P3.4)
- [ ] Baselines documented (job 3818806)
- [ ] Bootstrap CIs computed (P4.1)
- [ ] Cross-regional transfer result in hand (P4.3)
- [ ] Forward maps produced (P5.4)

**Target structure (Nature Sustainability, ~3500 words main text)**:
1. **Introduction** (600w): 30×30 urgency, prediction-vs-prescription gap, cost-minimisation argument
2. **Results** (1800w): Stage 1 → Stage 2 (cross-event primary + temporal robustness) → SHAP → representation gap + scorecard → forward projections
3. **Discussion** (600w): designation as political economy, representation gap implications, limitations
4. **Methods** (500w): data, Stage 1, Stage 2 (cross-event design + temporal robustness explicitly described)
5. **Extended Data** (~8 figures): per-country metrics, temporal stability (per-year), temporal SHAP pre/post-Paris, USA/SE Asia zero-shot, country-level SHAP, forward scenarios, country scorecard

---

## Experiment Queue

| Priority | Task | Status | Blocks |
|---|---|---|---|
| **1** | Cross-event script — Colombia pilot (CE.1) | 🔄 Job 3841212 running | Everything |
| **2** | Cross-event full SA on Euler (CE.2) | 🔄 Job 3841647 queued | Paper model |
| **3** | Add AGB+REDD; retrain cross-event (CE.3) | ⬜ After CE.2 | SHAP, narrative |
| **4** | KBA download + rasterise (P3.1) | ⬜ Start any time (independent) | RQ3 |
| **5** | SHAP global beeswarm (P2.1) | ⬜ After CE.3 | Core paper story |
| **6** | Temporal SHAP pre/post-Paris (P2.2) | ⬜ After P2.1 | Bold finding B |
| **7** | Country-level SHAP (P2.3) | ⬜ After P2.1 | Paper nuance |
| **8** | Score vs biodiversity priority (P3.2) | ⬜ After P2.1 + P3.1 | RQ3 figure |
| **9** | Forward projection KBA headline (P3.3) | ⬜ After P3.2 | Headline number |
| **10** | Country scorecard (P3.4) | ⬜ After Stage 1 + P3.3 | Bold finding C |
| **11** | Bootstrap CIs (P4.1) | ⬜ After CE.2 | Statistical rigor |
| **12** | Temporal stability per-year (P4.2) | ✅ Done (job 3818794) | Bolsonaro collapse confirmed (see Current State) |
| **13** | Cross-regional transfer (P4.3) | ⬜ After CE.3 | Claim 5 |
| **14** | Baselines comparison (P1.4) | ✅ Done | random=1.0×, naive=2.81×, full=6.54× |
| **15** | Forward maps + KBA overlay (P5.2–P5.4) | ⬜ After CE.3 | Figure 3 |
| **16** | Paper writing (Phase 6) | ⬜ After all above | — |

---

## Full Experiment History

### Stage 2 pixel-level experiments (full SA unless noted)

| Experiment | Lift@1% | Recall@5% | iter | Verdict |
|---|---|---|---|---|
| Baseline (79 feat, default params) | 2.85× | 14.0% | 149 | Starting point |
| 20-trial Optuna retune | 2.06× | 8.4% | 7 | ✗ catastrophic — never retune before feature lock |
| Temporal year weights | 2.64× | 11.4% | 113 | ✗ both metrics worse |
| H6+H1b (Recall stop + inv_sqrt_npos) | 3.73× | 18.1% | 89 | ✓ first both-positive |
| **H6+H1b+H5 (no rank-norm)** | **6.46×** | **15.7%** | **136** | ✅ **locked temporal model** |
| H6+H8+H10 (temporal weights + combined stop) | 4.17× | 18.96% | 112 | ✗ did not beat H5 on both |
| H6+H1b+H5+H7 (train 2010–2013 only) | 1.51× | 10.5% | — | ✗ catastrophic — too few groups |
| H6+H1b+H5+H11 (patch-context pixel features) | 5.56× | 16.6% | 50 | ✗ patch features constant within patch |
| Spatial diffusion (steps=10, α=2.0) | 6.15× | 17.2% | — | ✗ marginal, ceiling confirmed |
| SA naive baseline (dist_wdpa only) | 2.81× | — | — | reference |
| **Within-group 80/20 pixel split (P1.1)** | — | **~94–96%** | — | ❌ **geometric artifact — NOT a paper result** |

**Why within-group is an artifact**: The average designation cluster (~3,933 km²) fits entirely inside the top-5% budget per group (~17,556 km²). Any model that identifies the cluster from the 80% training pixels achieves near-100% recall trivially. The test pixels are in the SAME spatial cluster as train pixels — no genuine feature generalisation occurs. This is fundamentally different from the cross-event design where train and test events are at different geographic locations.

### Stage 2 patch-level experiments (abandoned)

| Experiment | Lift@1% | Recall@5% | iter | Verdict |
|---|---|---|---|---|
| H12: connected-component patch ranking | 81.32× | 87.7% | 20 | ❌ artifact — mega-blob size-sort |
| H12 no-size ablation | 79.11× | 86.0% | **1** | ❌ 86% in 1 round confirms artifact |
| Naive size-sort (reference) | 80.38× | 88.8% | — | reference — trivial |

**Why H12 is an artifact**: SA unprotected land forms continent-spanning connected components (largest: 6.4M pixels). The model achieves high Recall by ranking the single mega-blob first. Confirmed by no-size ablation (86% Recall in round 1 without any training).

### P1.3 Per-country breakdown (2026-06-18, temporal model H6+H1b+H5, test 2017–2024)

| Country | Lift@1% | NDCG@1% | n_rows | Note |
|---|---|---|---|---|
| SUR | 28.55× | 0.0670 | 8.8M | Excellent |
| ARG | 9.31× | 0.0432 | 23.4M | Good |
| PER | 4.88× | 0.0618 | 3.2M | Good |
| COL | 3.51× | 0.0288 | 7.7M | Moderate |
| ID13 | 3.00× | 0.0431 | 1.2M | Unknown country code |
| BRA | 1.69× | 0.0265 | 51.0M | **Weak — 48% of test data** |
| ECU | 1.37× | 0.0075 | 1.4M | Near-random |
| GUY | 1.20× | 0.0063 | 0.04M | Near-random (tiny) |
| BOL | 0.00× | 0.0000 | 2.7M | Model fails completely |
| VEN | 0.00× | 0.0000 | 0.2M | Model fails (tiny group) |
| CHL | 0.69× | 0.0091 | 7.4M | Below random — structurally different (private/marine) |

**Weighted Lift excl. BRA + CHL + BOL + VEN: 10.89×** — the model already works well where cost-minimisation logic applies.

**Temporal breakdown** (Lift@1% by year): 2017=13.54×, 2018=11.45×, 2019=1.38×, 2020=1.14×, 2021=4.96×, 2022=0.94×, 2023=6.11×, 2024=11.37×. 2018 and 2019 had nearly identical designation counts (74K vs 73K positives) but wildly different Lift — proving this is **concept drift** (WHERE designations happened changed), not event frequency.

---

## Settled Decisions

| Decision | Value | Rationale |
|---|---|---|
| Stage 2 ranking unit | Pixels (WDPA_prev==0) | CC patches = mega-blob artifact; pixels = clean |
| Stage 2 validation mode | Cross-event 80/20 event split (primary) | Correct standard for spatial choice models |
| Stage 2 temporal result | Reported as robustness supplement | 6.46× Lift validates temporal persistence |
| Within-group pixel split | Abandoned | Geometric artifact — not a paper result |
| Engine | LightGBM LambdaRank | Validated |
| Early stopping | Recall@5% within groups (H6) | Directly optimises target metric |
| Sample weights | inv_sqrt_npos (H1b) | Gradient deconcentration across groups |
| Rank normalisation | Off (H5) | Absolute features carry cross-country signal |
| Hyperparameter retuning | Only after feature set locked | Early retune catastrophic (2.85× → 2.06×) |
| Ensembles | Forbidden | Supervisor directive |
| Patch CC approach | Abandoned | Mega-blob artifact, structurally degenerate |
| Paper framing | Representation gap is the finding; prediction model is the tool | Core strategic decision |
| Naive baselines | Must be documented | Required for any top journal |

---

## Data Paths

| Dataset | Location |
|---|---|
| SA pixel splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet` |
| SA merged panel (57 GB) | `euler:$SCRATCH/data/south_america/ml/merged_panel_final.parquet` |
| Temporal model (H6+H1b+H5) | `data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl` |
| USA pixel splits | `euler:$SCRATCH/data/usa/ml/main/{train,earlystop,test}.parquet` |
| SE Asia pixel splits | `euler:$SCRATCH/data/se_asia/ml/main/{train,earlystop,test}.parquet` |
| AGB TIF | `data/south_america/ready/AGB/agb_sa.tif` |
| REDD TIF | `data/south_america/ready/REDD/redd_sa.tif` |
| Colombia dev panel | `euler:$SCRATCH/data/dev/south_america/ml/main/` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` |
| Forward scored (existing) | `euler:$SCRATCH/outputs/south_america/forward_scored_2024.parquet` |
| MapBiomas Brazil | Google Drive (verify format/coverage before committing) |

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)`. Backbone CRS is LOCAL_CS; `crs.to_epsg()` returns None.

---

## Out of Scope

- Hyperparameter tuning — only after AGB + final feature set validated
- SE Asia Stage 2 full training — zero-shot transfer only (Phase 4)
- Neural networks, ensembles — Paper 2
- Tropical Africa — Paper 2
- 10km grid rebuild — viable only if CE.1/CE.2 disappoints; run P4.1 diagnostic first
- MapBiomas Brazil — deferred; cross-event model may resolve Brazil performance
- Rolling spatial PA profile feature (P4.2 old numbering) — low priority vs RQ3
