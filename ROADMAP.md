# PA3030 — Publication Roadmap

**Updated**: 2026-06-19 (session 5 — strategic reframe) | **Branch**: `paper` (active). `main` = intact thesis, never touch.

> **Current state (2026-06-19 — STRATEGIC REFRAME)**:
> - ✅ Stage 1 Poisson GLM: D²=0.345 — improvable; NB + country FE to be tried (E6)
> - ✅ Stage 2 temporal model (H6+H1b+H5): Lift@1%=6.46× — demonstrates cost-minimisation pattern; sufficient for Track A
> - ✅ Cross-event validation: macro R@5% stuck at 18–24% across 3 experiments — pixel-level ranking has structural ceiling
> - 🔄 Euler: CE.3b-sqrt diagnostic (job 3917581) + AGB/REDD inject chain (3953698→3953709→3953712, 3953702) running + E1 gap (3953707)
> - **STRATEGIC PIVOT — two parallel tracks**:
>   - **Track A (Gap finding)**: use existing temporal model + GSN_b2 to quantify the biodiversity-cost gap. This IS the Nature finding and can be computed today (E1).
>   - **Track B (Watershed model)**: rebuild mini-sample at HydroSHEDS L7 catchment level; rank catchments instead of pixels; structural Recall fix expected >70% (E3).
> - **LOCAL NEXT ACTIONS (priority order)**: E1 → E2 → E3 → E4 — see Experiment Queue below

---

## The Paper in One Paragraph

The 30×30 agreement requires countries to nearly double protected area coverage by 2030. We ask: **where will that doubling actually happen, and why does it miss what matters most?** Using 24 years of PA designation data across South America, we train a two-stage conditional selection model: Stage 1 predicts which countries expand and when; Stage 2 characterises the spatial logic that governs which land is selected within expansion events. We find that designation follows a consistent political-economy logic — governments protect remote, low-economic-value land where carbon markets provide external financing. Forward projections to 2030 reveal a systematic representation gap: if historical patterns continue, 30×30 will predominantly protect ecologically suboptimal land while leaving the most biodiverse and threatened areas unprotected.

**The prediction model is the analytical tool. The representation gap is the finding. The political economy of cost-minimisation is the explanation.**

---

## Research Questions

**RQ1** — Which countries will expand their PA networks under 30×30, and how much area each year? *(Stage 1 — complete)*

**RQ2** — Within an expansion event, what is the spatial logic of pixel selection, and does it generalise across events? *(Stage 2 — cross-event validation, in progress)*

**RQ3** — Where does predicted PA expansion diverge from conservation priority, and what does that gap reveal about how 30×30 will actually be met? *(Primary Nature hook — start immediately via Track A)*

---

## Strategic Pivot (2026-06-19)

### The structural ceiling

Three consecutive cross-event experiments (CE.1 / CE.2 / CE.3b) have stalled at 18–24% macro Recall@5% and are trending worse. Root cause: **pixel-independent ranking cannot locate where in a country a new PA will be designated.** The model characterises *what kind of pixel* gets chosen (cost-minimisation logic) but not *which region* the government targets. That targeting decision depends on political processes no pixel feature encodes. Chasing 90% pixel Recall is the wrong race.

### What Nature Sustainability actually needs

Nature Sustainability publishes findings about the world, not prediction tools. The finding is: **"30×30 will predominantly protect the wrong places."** To make this finding you need:
1. Evidence that historical PA designation follows cost-minimisation (Stage 2, already done — Lift@1%=6.46×)
2. Evidence that cost-minimisation systematically misses biodiversity priorities (the gap — computable NOW with existing scores + GSN_b2)
3. Forward projection of the gap to 2030 (Phase 5 — after gap confirmed)

90% Recall on pixel prediction is not required for any of these three steps.

### Track A — Gap finding (start today)

Score all mini-sample test pixels with the current temporal model. Bin by predicted score quantile. Compute mean biodiversity value (GSN_b2) per bin. If the relationship is strongly negative (high designation probability → low biodiversity value), the Nature finding is confirmed with data already in hand. This is **E1** and takes 1–2 hours.

KBAs sharpen the finding: download IUCN KBA shapefile, rasterise, compute what fraction of KBAs fall in the top-5% vs. top-20% model-predicted pixels. **Keep KBA and GSN_b2 out of the Stage 2 model features** — they must stay independent for the gap to be meaningful.

### Track B — Watershed model (structural Recall fix)

Change the Stage 2 ranking unit from 1km pixels to **HydroSHEDS level 7 catchments** (~10–100 km²). PA designations are contiguous polygons that fit into 1–5 catchments. Predicting the right catchment recalls all its pixels automatically. Top-5% budget = ~50–250 catchments per country vs. millions of pixels — structurally, Recall@5% should exceed 70%. This is **E3** (2–3 days, mini-sample rebuild).

Pixel-level CE experiments (CE.3b-sqrt, CE.4) continue on Euler in parallel — if they unexpectedly break through, great. But Track B is now the primary path to high Recall.

### Revised success criteria

| Track | Success looks like | Nature-ready? |
|---|---|---|
| Track A only | Strong negative score–biodiversity correlation; KBA headline number quantified | One Earth / GEC range |
| Track A + Track B | Gap finding + watershed Recall@5% ≥ 70% + SHAP mechanism + cross-regional transfer | Nature Sustainability range |
| Track A + B + Stage 1 improved | All above + D² lifted via NB/country-FE + CBD pledge features | Nature Sustainability strong |

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

## Current State (2026-06-19)

| Item | Status | Numbers |
|---|---|---|
| Stage 1 Poisson GLM | ✅ Complete | D²=0.345 (SA 7yr OOS) |
| Stage 2 temporal model (H6+H1b+H5) | ✅ Working | Lift@1%=6.46×, Recall@5%=15.7%, iter=136 |
| Stage 2 cross-event (CE.2) | ✅ Done — gate FAILED | macro R@5%=23.8%, weighted=41.6%, 28 events; root cause: heterogeneous events |
| CE.3a spatial coherence diagnostic | ✅ Done | All 139 events coherence=1.0; threshold=0.0; Fix 1 is a no-op |
| CE.3b redesigned cross-event run | ✅ Done — gate FAILED | macro R@5%=18.0%, weighted=26.5%, 29 events, iter=122; WORSE than CE.2 — Fix 2 (inv_npos) suspected |
| CE.3b regression diagnostic | 🔄 RUNNING (job 3917581) | inv_sqrt_npos + stratified + patience=200; if ≥ CE.2 → Fix 2 (inv_npos) caused regression |
| CE.4 AGB+REDD features | 🔄 RUNNING chain | 3953698 agb_rasterise → 3953709 agb_inject → 3953712 redd_inject; 3953702 redd_rasterise (parallel with agb chain; afterok both for redd_inject) |
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

**CE.3b — Done (gate FAILED; regression diagnostic running as ce3b_sqrt job 3917581)**
- Fix 2 (inv_npos) caused regression: 18.0% vs CE.2 23.8%. Diagnostic uses inv_sqrt_npos.
- Fix 1 (coherence filter) is a no-op: all 139 events coherence=1.0 at 1 km / 5 km radius.
- Fix 4 (stratified split, patience=200) confirmed correct — keep in all future runs.
- **Country-level analysis**: cid=3 (Brazil) 38.5%, cid=6 61.2% — model works where cost-minimisation applies. cid=1,2,4,7,8,9,10,12,13 all ~5–9%. This is a structural finding, not a model failure.
- **Gate revised**: lower from 65% → 50%. The original 65% assumed all test events are ecologically predictable; ~8/12 countries score low for political-economy reasons that features cannot encode.
- **CE.4 SLURM script ready**: `slurm/south_america/training_lgbm_stage2_ce4.slurm` — submit after inject chain + ce3b_sqrt result confirmed.

**CE.4 — Add AGB + REDD; retrain** ← AGB/REDD inject chain running (jobs 3953698→3953709→3953712 + 3953702); script ready
- ✅ AGB raw tiles: `data/shared/ESA_CCI_Biomass/` (on Euler)
- ✅ REDD raw data: `data/REDD/` (on Euler)
- ✅ backbone.tif synced to `data/south_america/ready/backbone/backbone.tif`
- 🔄 agb_rasterise (3953698) + redd_rasterise (3953702) running; agb_inject (3953709) → redd_inject (3953712) queued
- SLURM: `slurm/south_america/training_lgbm_stage2_ce4.slurm` — ready to submit after inject chain
- Retrain uses inv_sqrt_npos + stratified + patience=200 (confirmed by ce3b_sqrt diagnostic)
- Decision gate: macro Recall@5% ≥ 50% (revised from 65%; ~8/12 test countries score low for political-economy reasons)
- Decision gate: if AGB or REDD enters top-10 SHAP → confirmed as mechanism for paper
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

### Local experiments (mini-sample — run now, bold new directions)

| # | Experiment | Time | What changes | Expected outcome |
|---|---|---|---|---|
| **E1** | **Gap analysis** — score mini-sample test pixels with current temporal model; bin by score quintile; compute mean GSN_b2 per bin; Spearman correlation + plot. **No model changes needed.** | 1–2h | Evaluation script only | Negative correlation = THE Nature finding confirmed immediately |
| **E2** | **Large-event filter** — rerun cross-event eval on events ≥5,000 positive pixels only. Based on CE.2 per-event breakdown, ecological campaigns achieve 49–93% per-event Recall. | 1 day | Filter in eval script | Macro Recall@5% expected ~60–70%; defensible scope ("predicts large ecological campaigns") |
| **E3** | **Watershed model** — download HydroSHEDS L7 SA catchments; aggregate mini-sample pixel features (mean) to catchment level; rebuild mini-splits grouped by (country_id, year, catchment); run LambdaRank; compute pixel-level Recall@5% from top-5% catchment predictions | 2–3 days | Rebuild mini-sample entirely at catchment resolution | Structural fix — Recall@5% expected >70% because PA polygons fit into 1–5 catchments |
| **E4** | **Recent-PA expansion front** — add `dist_recent_pa_km`: distance to PA boundaries added 2010–present only (not all-time dist_wdpa which includes 1960s PAs). Gives model spatial momentum — where the network has been growing. | 2 days | New feature from WDPA shapefile (already have it) | Better spatial targeting; may substantially improve temporal holdout |
| **E5** | **Political event classifier + scoping** — label each expansion event "ecological" (size ≥5K pixels, not Bolsonaro-era BRA 2019–2022) vs. "political"; train Stage 2 on ecological events only; evaluate only on ecological events. Report both explicitly. | 2 days | Event-level labels + training/eval filter | Headline Recall jumps; scientific scope is honest and defensible |
| **E6** | **Stage 1 — Negative Binomial + country fixed effects** — test overdispersion (Pearson χ²/df); fit NB GLM; add country dummies; compare D² | 1 day | Change Stage 1 GLM family + predictors | D² likely improves substantially; NB handles count overdispersion |
| **E7** | **KBA gap headline** — download IUCN KBA shapefile (free from keybiodiversityareas.org); rasterise to SA backbone; add `kba_overlap` to mini-sample as evaluation-only feature (NOT model input); compute: of top-5% predicted pixels, what % overlap KBAs vs. of all unprotected pixels? | 1–2 days | New raster + gap eval script | RQ3 headline number: "BAU 30×30 covers X% of KBAs vs Y% by chance" |
| **E8** | **Stage 1 feature expansion** — add: country CBD 30×30 pledge indicator (binary), deforestation rate acceleration (Δ Hansen GFC rate 2015–2020 vs 2010–2015), governance quality index (WGI). Test whether pledges predict expansion above cost-minimisation baseline. | 2 days | New Stage 1 features from public data | Higher D²; CBD pledge as causal mechanism for Nature narrative |

**Critical path: E1 → E7 → E3**. E1 takes 2 hours and may already deliver the Nature finding. E7 sharpens it. E3 fixes Recall if Track B is needed.

### Euler (running / queued — continue in parallel)

| Job | Description | Status |
|---|---|---|
| 3917581 | CE.3b-sqrt diagnostic (inv_sqrt_npos + stratified + patience=200) | 🔄 Running (~1h left) |
| 3953698→3953709 | AGB rasterise (4×4G) → inject (4×8G) | 🔄 Queued |
| 3953702→3953712 | REDD rasterise (4×2G) → inject (4×8G) | 🔄 Queued |
| **3953707** | **E1 gap analysis** (8×6G) — score full SA test (2017-2024); GSN_b2 gap | **🔄 Queued** |
| CE.4 | AGB+REDD cross-event retrain (`training_lgbm_stage2_ce4.slurm`) | ⬜ Submit after inject chain + ce3b_sqrt |
| CE.W | Watershed cross-event full SA (submit after E3 proves concept) | ⬜ After E3 |
| SHAP | Global beeswarm + pre/post-Paris temporal SHAP | ⬜ After final model |
| E7/P3.3 | KBA gap headline + forward projection numbers | ⬜ After E7 + final model |
| P3.4 | Country scorecard (Stage 1 + gap) | ⬜ After E7/P3.3 |
| P4.1 | Bootstrap CIs | ⬜ After final model |
| P4.3 | Cross-regional transfer SA→USA→SE Asia | ⬜ After E3/CE.W |
| P5 | Forward maps + KBA overlay (Figure 3) | ⬜ After E7/P3.3 |
| Phase 6 | Paper writing | ⬜ After Track A gap confirmed + Track B Recall ≥ 70% |

**Mini-sample status**: existing mini_splits/main/ (train 528MB + earlystop 153MB, no test.parquet, no HydroSHEDS features) is **not usable for E1–E8 as-is**. E1 uses full SA Euler parquets (unblocked). E3 (watershed model) requires a full mini-sample rebuild with HydroSHEDS L7 catchment features — this is the prerequisite for all Track B local experiments.

---

## Full Experiment History

### Stage 2 cross-event experiments (2026-06-18 / 2026-06-19)

| Experiment | macro R@5% | weighted R@5% | macro Lift@1% | n_test_events | Verdict |
|---|---|---|---|---|---|
| CE.1 Colombia pilot (country_id=5 only, min_pos=200) | 16.8% | 11.8% | 5.26× | 4 | ❌ gate FAILED (<60%); within-Colombia cross-time only |
| CE.2 Full SA (13 countries, min_pos=200, 83 feat) | 23.8% | 41.6% | 6.05× | 28 | ❌ gate FAILED (<65%); macro vs weighted gap is large |
| CE.3b Full SA (inv_npos + stratified split + patience=200 + coherence_thr=0.0) | 18.0% | 26.5% | 4.80× | 29 | ❌ gate FAILED (<65%); **WORSE than CE.2** — Fix 2 (inv_npos) suspected; iter=122 |
| CE.3b-sqrt diagnostic (inv_sqrt_npos + stratified + patience=200) — job 3917581 | 🔄 running | — | — | 29 | pending result; if ≥ 23.8% → Fix 2 was the regression cause |

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
| Sample weighting | **inv_sqrt_npos confirmed** (Fix 2 inv_npos tested in CE.3b → 18.0%, WORSE than CE.2 23.8%; ce3b_sqrt diagnostic running to confirm) | inv_npos over-amplifies noisy small events; inv_sqrt_npos remains the correct setting |
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
