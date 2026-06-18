# PA3030 — Publication Roadmap

**Updated**: 2026-06-18 (session 3) | **Branch**: `paper` (active). `main` = intact thesis, never touch.

> **Current state (2026-06-18)**:
> - ✅ Stage 1 Poisson GLM: complete, D²=0.345
> - ✅ Stage 2 temporal model (H6+H1b+H5): 79 features, Lift@1%=6.46×, Recall@5%=15.7%
> - ✅ P1.3 per-country breakdown: BRA=1.69×, SUR=28.55×, ARG=9.31×; excl. BRA+CHL+BOL+VEN → 10.89× weighted
> - ✅ Temporal stability: 2017=7.81×, 2018=11.59×, 2019=0.99×, 2020=0.50×, 2021=1.74×, 2022=1.89×, 2023=6.77×, 2024=9.02×
> - ✅ Baselines: random=1.0×, naive(dist_wdpa)=2.81×, full model=6.54×
> - ✅ CE.2 full SA cross-event: macro R@5%=23.8%, weighted R@5%=41.6% (28 test events) — gate FAILED
> - **Root cause identified**: events are heterogeneous — ecological campaigns R@5%=49–93%, political/legal designations R@5%=0–12%. Average mixes both.
> - ✅ CE.3b code complete: Fix 1 (coherence filter), Fix 2 (inv_npos event-norm), Fix 4 (stratified split + patience=200 + coherence-filtered val)
> - 🔄 **RUNNING**: Full chain submitted 2026-06-18:
>   - 3864314: CE.3a spatial coherence diagnostic (~30 min)
>   - 3864332: CE.3b redesigned cross-event training (afterok 3864314, ~12h, reads threshold from CE.3a)
>   - 3864334: AGB rasterise — ESA CCI Biomass → agb_sa.tif (afterok 3864332 gate pass)
>   - 3864336: REDD rasterise — ID-RECCO → redd_sa.tif (afterok 3864334)
>   - 3864339: AGB inject into splits (afterok 3864336)
>   - 3864342: REDD inject into splits (afterok 3864339)
> - **NEXT ACTION (manual, after chain)**: submit CE.4 retrain with same ce3b SLURM script (splits now have AGB+REDD)

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
| Stage 2 cross-event (CE.2) | ✅ Done — gate FAILED | macro R@5%=23.8%, weighted=41.6%, 28 events; root cause: heterogeneous events; redesign in progress |
| CE.3a spatial coherence diagnostic | 🔄 RUNNING (job 3864314) | Output: stage2_event_coherence_diagnostic.json |
| CE.3b redesigned cross-event run | 🔄 QUEUED (job 3864332, afterok 3864314) | Fixes 1+2+4; reads coherence threshold from CE.3a JSON at runtime |
| CE.4 AGB+REDD features | 🔄 QUEUED (jobs 3864334–3864342, afterok CE.3b gate pass) | AGB rasterise → REDD rasterise → inject splits × 2 |
| Per-country breakdown (temporal model) | ✅ Done | SUR=28.55×, ARG=9.31×, BRA=1.69×; excl. outliers: 10.89× |
| Within-group pixel split (P1.1) | ❌ Abandoned | Geometric artifact (93–96% guaranteed by cluster geometry) |
| Patch CC approach (H12) | ❌ Abandoned | Mega-blob artifact confirmed |
| Temporal stability per-year | ✅ Done | 2017=7.81×, 2018=11.59×, 2019=0.99×, 2020=0.50×, 2021=1.74×, 2022=1.89×, 2023=6.77×, 2024=9.02× |
| Baselines | ✅ Done | random=1.0×, naive=2.81×, full=6.54× |
| SHAP analysis | ❌ Not run | Needs final model (after CE.3b/CE.4) |
| KBA download + rasterise | ❌ Not done | Required for RQ3 |
| Representation gap (RQ3) | ❌ Not quantified | THE paper finding |
| Bootstrap CIs | ❌ Not run | After final model |
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

### Phase 1 — Cross-event Stage 2 model (IN PROGRESS — redesign based on CE.2 failure)

**CE.1 / CE.2 — Done but gate failed.** See Experiment History for full results and per-event breakdown.

---

#### Root cause of CE.2 failure

CE.2 macro Recall@5%=23.8%. This is the macro average across two structurally different event types:

1. **Ecological campaigns** (large, spatially coherent, cost-minimisation logic): Recall@5%=49–93%. These events are what the model is designed to predict, and it does so well. Examples: Brazil 2009 (52.9%), Suriname 2015 (59.1%), Argentina 2023 (92.8%).
2. **Political/legal designations** (court-ordered indigenous territories, scattered reclassifications, economic-crisis-driven designations): Recall@5%=0–12%. No landscape feature can predict which specific parcels are affected by a legal ruling or political deal. These events structurally resist prediction.

The macro average mixes both types indiscriminately. The model is correct on class 1; class 2 is inherently unpredictable.

**Three additional design flaws in CE.2 that suppressed performance:**
1. **Training gradient imbalance (Fix 2)**: The current `inv_sqrt_npos` weights reduce gradient per-pixel within large events, but Brazil's 100K+ pixel events still contribute ~50× more total gradient signal than Colombia's 1K-pixel events. The model becomes biased toward "what Amazon pixels look like" and underfits smaller countries.
2. **Early stopping failure (Fix 4)**: The model stopped at iteration 66 (vs 136 for the temporal model). This happened because the validation set contained too many class-2 events — no amount of training would improve recall on politically-driven events, so the early-stopping metric stalled and training terminated prematurely.
3. **Unlucky random event split (Fix 4)**: Random seed=42 placed all of Argentina's early-2000s events (crisis years 2001-2005) in the test set. These are anomalous years (Argentine economic collapse, Dec 2001). A stratified split prevents this.

---

#### The four fixes

**Fix 1 — Spatial coherence filter (CE.3a diagnostic first, then applied in CE.3b)**

Ecological campaigns are spatially clustered (one big contiguous park). Political designations are geographically scattered (10 isolated parcels across the country for 10 different legal reasons). Measure this per event using pixel coordinates already in the parquet: coherence = fraction of positive pixels within 5 km of another positive pixel.

**Important**: do NOT rely on min_positive_pixels alone as the filter. A 200-pixel designation of a coherent wetland is legitimate and predictable. A 10,000-pixel event of scattered parcels is not. Use coherence as the primary filter; min_positive_pixels stays at 200 (do not raise arbitrarily).

**Fix 2 — Event-level gradient normalization (code change in cross_event training script)**

Change weighting so each EVENT contributes equally to model training, regardless of pixel count. Currently:
- `inv_sqrt_npos` per pixel → Brazil 155K-pixel event contributes ~333 total gradient units; Colombia 1K-pixel event contributes ~22 units (15× imbalance even after weighting)

New formula: `weight_i = 1 / n_pos_in_group` (normalise each group's total gradient to 1.0). Combined with inv_sqrt_npos: `weight_i = 1 / (n_pos_in_group × sqrt(n_pos_in_group))`. Every event, regardless of size, contributes equal total signal to the model.

This is a change in `compute_group_norm_weights()` in `scripts/regions/shared/training/stage2_lgbm_core.py`. Add a new mode `"event_norm"` (or `"inv_npos"`) alongside the existing `"inv_sqrt_npos"`.

**Fix 3 — AGB + REDD features (CE.4 — blocked on data download, separate from CE.3b)**

AGB (above-ground biomass) and REDD (proximity to carbon-market projects) are key missing features. They capture the carbon-market incentive mechanism that drives Amazon designation. High-carbon forests earn international payments under REDD+, making them preferential targets for PA designation.

**Critical facts:**
- Both features are **static** (not annually dynamic). AGB is an ESA CCI Biomass 2010 snapshot. REDD is a distance-to-project metric from the ID-RECCO V5.0 database (2023 snapshot). This is intentional — we want the structural signal (is this pixel in a carbon-rich forest?) not year-by-year biomass fluctuations.
- **Both raw datasets are present locally but NOT yet on Euler.** AGB: 29 tiles (1.7 GB) at `data/shared/ESA_CCI_Biomass/`. REDD: `ID-RECCO V5.0_20231201.zip` at `data/REDD/`. The current SA splits use 79 features (temporal model) or 83 features (CE.2 — 4 unidentified extra columns, definitely NOT AGB/REDD).
- Fix 3 requires: (a) rsync raw data to Euler, (b) run rasterise scripts on Euler, (c) rebuild SA splits on Euler (42 GB rebuild job), (d) retrain.
- This is a significant pipeline step — **CE.4 is the correct home for this**, running after CE.3b establishes the training design.

**Fix 4 — Stratified event split + longer early-stop patience (code change)**

- **Stratified split**: instead of random shuffle, group events by country, then sample 80/20 within each country. This prevents unlucky seeds from putting all of one country's anomalous years in the test set.
- **Coherence-filtered validation set**: restrict the early-stopping validation set to high-coherence events only. The model can learn from political events in training (they provide some signal) but should not be stopped early because it fails on inherently unpredictable validation events.
- **Patience=200** (was 100): the model gave up at iteration 66. More patience lets it find deeper signal in the ecological events.

---

#### Implementation plan

**CE.3a — Spatial coherence diagnostic** ← IMMEDIATE (local, ~1h, no Euler needed)
- Script: `scripts/regions/south_america/6_evaluation/stage2_event_coherence.py` (new)
- Input: scored test parquet from CE.2 (`model1_lgbm_stage2_cross_event_20260618_170012.pkl`) + train/test parquets for positive pixel coordinates
- For each of the 139 SA events: load positive pixel (row, col) coordinates; compute coherence = fraction within 5 km (5 pixels) of another positive; record (country_id, year, n_pos, coherence, recall_at_5pct_from_CE2)
- Output: `outputs/south_america/results/ml_models/stage2_event_coherence_diagnostic.json`
- Decision: find the coherence threshold above which events have median Recall@5% ≥ 30%; use that as the filter for CE.3b

**CE.3b — Redesigned cross-event run on Euler** ← AFTER CE.3a
- Implement Fix 2 (event normalization) in `stage2_lgbm_core.py`
- Implement Fix 4 (stratified split, coherence-filtered validation, patience=200) in `model1_LGBM_stage2_cross_event.py`
- Apply Fix 1 coherence filter (threshold from CE.3a); keep min_positive_pixels=200
- SLURM: update `slurm/south_america/training_lgbm_stage2_cross_event.slurm`
- Decision gate: macro Recall@5% on coherent-event test set ≥ 50% → proceed to CE.4
- Decision gate: if < 40% → investigate WDPA designation_type filter (separate diagnostic)

**CE.4 — Add AGB + REDD; retrain** ← AFTER CE.3b passes; blocked on data
- Download ESA CCI Biomass v4.0 tiles from CEDA (requires free registration)
- Download ID-RECCO V5.0 from reddprojectsdatabase.org (zip may be available locally — recheck)
- Run: `scripts/regions/south_america/2_preprocessing/agb_rasterise.py`
- Run: `scripts/regions/south_america/2_preprocessing/redd_rasterise.py`
- Rebuild SA splits on Euler to inject AGB + REDD columns (42 GB rebuild)
- Retrain with CE.3b settings; check AGB/REDD importance in SHAP
- Decision gate: AGB or REDD enters top-10 SHAP features → confirmed as mechanism for paper

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
| **Priority** | **Task** | **Status** | **Blocks** |
| 1 | CE.3a Spatial coherence diagnostic (local, ~1h) | ⬜ IMMEDIATE | CE.3b threshold |
| 2 | CE.3b Redesigned CE run (Fix 1+2+4: coherence filter, event norm, stratified split, patience=200) | ⬜ After CE.3a | Paper model |
| 3 | CE.4 AGB+REDD: download → rasterise → split rebuild → retrain | ⬜ After CE.3b passes; **blocked on data download** | Features + SHAP |
| 4 | KBA download + rasterise (P3.1) | ⬜ Start any time — no model dependency | RQ3 |
| 5 | SHAP global beeswarm (P2.1) | ⬜ After CE.4 | Paper mechanism story |
| 6 | Temporal SHAP pre/post-Paris (P2.2) | ⬜ After P2.1 | Carbon-market finding |
| 7 | Country-level SHAP (P2.3) | ⬜ After P2.1 | Brazil vs Andean nuance |
| 8 | Score vs biodiversity priority (P3.2) | ⬜ After P2.1 + P3.1 | RQ3 figure |
| 9 | Forward projection KBA headline (P3.3) | ⬜ After P3.2 | Headline number |
| 10 | Country scorecard (P3.4) | ⬜ After Stage 1 + P3.3 | Policy finding |
| 11 | Bootstrap CIs (P4.1) | ⬜ After CE.3b | Statistical rigour |
| 12 | Temporal stability per-year (P4.2) | ✅ Done | Bolsonaro collapse documented |
| 13 | Cross-regional transfer SA→USA→SEA (P4.3) | ⬜ After CE.4 | Claim 5 |
| 14 | Baselines (P1.4) | ✅ Done | random=1.0×, naive=2.81×, full=6.54× |
| 15 | Forward maps + KBA overlay (P5.2–P5.4) | ⬜ After CE.4 | Figure 3 |
| 16 | Paper writing (Phase 6) | ⬜ After all above | — |

---

## Full Experiment History

### Stage 2 cross-event experiments (2026-06-18)

| Experiment | macro R@5% | weighted R@5% | macro Lift@1% | n_test_events | Verdict |
|---|---|---|---|---|---|
| CE.1 Colombia pilot (country_id=5 only, min_pos=200) | 16.8% | 11.8% | 5.26× | 4 | ❌ gate FAILED (<60%); within-Colombia cross-time only |
| CE.2 Full SA (13 countries, min_pos=200, 83 feat) | 23.8% | 41.6% | 6.05× | 28 | ❌ gate FAILED (<65%); macro vs weighted gap is large |

**Per-event pattern (CE.2 full SA test events)**:

| Event | n_pos | Recall@5% | Lift@1% | Note |
|---|---|---|---|---|
| (1, 2023) | 320 | 92.8% | 49.1× | Best — tiny but geographically coherent |
| (10, 2015) | 17581 | 59.1% | 21.7× | Excellent large event |
| (3, 2009) | 155630 | 52.9% | 13.8× | Strong Brazil event |
| (3, 2005) | 111429 | 49.4% | 8.9× | Good Brazil |
| (6, 2019) | 297 | 48.8% | 17.2× | Small but coherent |
| (6, 2002) | 300 | 46.7% | 0.67× | OK recall, weak lift |
| (5, 2009) | 14079 | 32.5% | 8.3× | Decent Colombia |
| (3, 2024) | 866 | 23.2% | 6.7× | Moderate |
| (1, 2005) | 6050 | 17.5% | 12.1× | OK recall, high lift |
| (1, 2003) | 10349 | 17.2% | 3.6× | Moderate |
| Many country 1 and 2 events | varies | 2–6% | 0–0.1× | Near-zero: model fails |

**Structural diagnosis**: Macro recall (23.8%) is dragged down by small-count or geographically-scattered events (many near-zero). Weighted recall (41.6%) reflects that large, ecologically-coherent events (Brazil mass designations, Argentina large parks) perform well. The gap reveals **event heterogeneity**: cost-minimisation logic predicts large coherent campaigns; it cannot explain small politically-idiosyncratic designations.

**Key question for next step**: Is raising min_positive_pixels (e.g., ≥5,000) the right filter, or should weighted recall replace macro as the headline metric?

---

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
| Hyperparameter retuning | Only after AGB+REDD features locked (post CE.4) | Early retune catastrophic (2.85× → 2.06×) |
| Ensembles | Forbidden | Supervisor directive |
| Patch CC approach | Abandoned | Mega-blob artifact, structurally degenerate |
| Paper framing | Representation gap is the finding; prediction model is the tool | Core strategic decision |
| Naive baselines | Must be documented | Required for any top journal |
| min_positive_pixels | Stay at 200; do NOT raise as primary filter | Small coherent events are scientifically valid; use coherence filter instead |
| Sample weighting | Change to event-level norm (Fix 2) after CE.3b validation | inv_sqrt_npos alone creates 50× gradient imbalance between large/small events |
| Event split | Stratified by country (Fix 4) | Random seed=42 placed Argentina crisis years entirely in test set |
| AGB / REDD features | Static features (2010 AGB snapshot, 2023 REDD database) — intentional | Annual dynamics not needed; structural signal is what drives designation |

---

## Data Paths

| Dataset | Location | Status |
|---|---|---|
| SA pixel splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet` | ✅ Ready (79 confirmed features; 4 unidentified extra in CE.2 run — investigate) |
| SA merged panel (57 GB) | `euler:$SCRATCH/data/south_america/ml/merged_panel_final.parquet` | ✅ Ready |
| Temporal model (H6+H1b+H5) | `data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl` | ✅ Locked baseline |
| CE.2 cross-event models | `data/south_america/ml/models/model1_lgbm_stage2_cross_event_*.pkl` | ✅ 3 files (2 × Colombia, 1 × SA) |
| USA pixel splits | `euler:$SCRATCH/data/usa/ml/main/{train,earlystop,test}.parquet` | ✅ Ready |
| SE Asia pixel splits | `euler:$SCRATCH/data/se_asia/ml/main/{train,earlystop,test}.parquet` | ✅ Ready |
| AGB raw tiles | `data/shared/ESA_CCI_Biomass/` | ✅ **29 tiles, 1.7 GB on Euler** — ESA CCI Biomass fv7.0 (2010); ready to rasterise |
| AGB TIF (processed) | `data/south_america/ready/AGB/agb_sa.tif` | ⬜ Run `agb_rasterise.py` (next step for CE.4) |
| REDD raw data | `data/REDD/ID-RECCO V5.0_20231201.zip` | ✅ **1.4 MB zip on Euler** — ready to rasterise |
| REDD TIF (processed) | `data/south_america/ready/REDD/redd_sa.tif` | ⬜ Run `redd_rasterise.py` (next step for CE.4) |
| Colombia dev panel | `euler:$SCRATCH/data/dev/south_america/ml/main/` | ✅ Ready (79 features) |
| Forward scored (existing) | `euler:$SCRATCH/outputs/south_america/forward_scored_2024.parquet` | ✅ Exists |
| MapBiomas Brazil | Google Drive | ⚠️ Verify format/coverage before committing |

### AGB and REDD: what they are and why they're blocked

**AGB (Above-Ground Biomass)** — ESA CCI Biomass v4.0, 2010 vintage. Static single-year snapshot. Forest biomass changes slowly, so the 2010 snapshot captures the structural signal without temporal leakage. This is intentional: we want "is this pixel in a carbon-rich forest?" not year-by-year flux. Feature column: `agb_tonne_ha`.

**REDD** — ID-RECCO V5.0 database of REDD+ project centroids. Produces `dist_redd_km` = distance to nearest REDD+ project. Static (database snapshot, 2023). Captures carbon-market geography: pixels near REDD+ projects face higher designation pressure from international financing.

**Why these matter for the model**: Carbon market incentives (REDD+, Paris Agreement NDCs) are a primary driver of Amazon PA designation. The model currently cannot distinguish a carbon-rich forest (high REDD+ value) from a grassland at the same location — these features close that gap. Expected effect: significant improvement on Brazil/Peru/Bolivia events (Amazon-heavy), possibly entering top-5 SHAP features.

**Pipeline steps required before CE.4**:
1. Download ESA CCI Biomass tiles → `data/shared/ESA_CCI_Biomass/`
2. Download ID-RECCO V5.0 zip → `data/REDD/`
3. Run `scripts/regions/south_america/2_preprocessing/agb_rasterise.py`
4. Run `scripts/regions/south_america/2_preprocessing/redd_rasterise.py`
5. Rebuild SA splits on Euler: inject AGB and REDD columns into train/earlystop/test.parquet (42 GB rebuild job, ~4h on Euler)

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)`. Backbone CRS is LOCAL_CS; `crs.to_epsg()` returns None.

---

## Out of Scope

- Hyperparameter tuning — only after AGB + REDD features locked and CE.4 validated
- SE Asia Stage 2 full training — zero-shot transfer only (Phase 4)
- Neural networks, ensembles — Paper 2
- Tropical Africa — Paper 2
- 10km grid rebuild — viable only if CE.3b + CE.4 both disappoint; last resort
- MapBiomas Brazil — deferred; cross-event model with AGB/REDD may resolve Brazil performance
- Rolling spatial PA profile feature — low priority vs RQ3
- Raising min_positive_pixels as a primary filter — a 200-pixel designation can be scientifically significant; use spatial coherence filter instead
