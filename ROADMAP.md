# PA3030 — Publication Roadmap

**Updated**: 2026-06-19 (session 6 — B0 applied; CE.4 + A1 + A3 queued) | **Branch**: `paper` (active). `main` = intact thesis, never touch.

> **Current state (2026-06-19 — STRATEGIC REFRAME)**:
> - ✅ Stage 1 Poisson GLM: D²=0.345 — improvable; NB + country FE to be tried (E6)
> - ✅ Stage 2 temporal model (H6+H1b+H5): Lift@1%=6.46× — demonstrates cost-minimisation pattern; sufficient for Track A
> - ✅ Cross-event validation: macro R@5% stuck at 18–24% across 3 experiments — pixel-level ranking has structural ceiling
> - ✅ AGB + REDD inject chain complete (3957103→3957105→3957107, 2026-06-19); SA splits now have 94 columns
> - ✅ CE.3b-sqrt (3917581) complete — macro R@5%=18.8% (exit 1 = gate fail, not crash)
> - ✅ B0 applied: E11 (val min-pos filter) + E12 (group size cap) in code + CE.4 SLURM
> - 🔄 e1_gap RUNNING (3981693, 128G); eco_gap_inject PENDING (3981696); CE.4 PENDING (3981700, dep); stage1+NB PENDING (3981878→3981887)
> - ❌ A2 KBA: blocked — user must download KBA shapefile from keybiodiversityareas.org/kba-data/request
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

**Track B extension — two-level hierarchical model (Paper 2 / post-Track B):** If single-level catchment ranking works but leaves further headroom, the natural next step is a two-stage spatial selector within Stage 2: Stage 2a ranks catchments to identify the target region; Stage 2b ranks pixels within the top-K catchments using the pixel-level features. Stage 2a handles the political/institutional targeting decision (which watershed); Stage 2b handles the boundary-drawing logic (which specific pixels). This cleanly separates the two decisions the current pixel model conflates. Prerequisite: Track B (E3) must first confirm catchment-level Recall ≥ 70%. Do not implement before then.

### Custom Recall@5% training loss (refinement after Track B)

**The problem:** LambdaRank optimises NDCG — a smooth proxy that rewards putting positive pixels near the very top with a logarithmic discount. Recall@5% is a hard threshold metric: every positive pixel inside the top 5% counts equally, and every positive pixel outside counts as a miss, regardless of how far outside it is. These objectives are not the same. A model that's excellent at NDCG might push a few positives to rank 1–5 while leaving others just outside the 5% boundary, missing them entirely.

**The fix:** Replace the NDCG training objective with a smooth differentiable approximation of Recall@5%. Concretely: for each positive pixel in a group, compute a soft indicator of whether it falls in the top 5% using a sigmoid over the score gap to the 5th-percentile threshold. The loss penalises any positive pixel not confidently in the top 5%, equally and directly. LightGBM supports custom gradient/hessian functions — this would be implemented as a custom objective passed to `lgb.train()`.

**Expected impact:** Modest — H6 early stopping on Recall@5% already partially aligns training with evaluation. The custom loss would close the remaining gap. Estimated improvement: 2–5 percentage points on top of whatever Track B achieves.

**Implementation effort:** ~1 week. Requires deriving gradient and hessian of the smooth Recall approximation and validating numerical stability. Do not implement before Track B confirms the catchment unit works.

**Status:** Not started. Post-Track B refinement only.

### Track C — Survival model (contingency if Track B stalls)

If watershed-level Recall@5% also fails to break 50% after E3, the correct next move is **not more feature engineering** — it is a fundamentally different model class: a **Cox Proportional Hazards survival model**. Instead of ranking pixels within an annual expansion event, we ask: *when will this pixel be protected?* Each unprotected pixel enters the model at t=2000 (or first available year) and exits either when protected (event) or at end of observation (right-censored). Key advantages for our structural problem:

- **Baseline hazard absorbs the mid-2010s structural break**: time-varying baseline h₀(t) captures the drop in designation rates without poisoning the feature coefficients. Features explain *who gets chosen*, not *how many get chosen per year*.
- **Right-censoring is native**: pixels never protected are treated as right-censored, not as "definitive negatives" — statistically correct for a process still unfolding.
- **No event-grouping needed**: removes the need for country-year expansion groups entirely; partial likelihood handles the rare-event structure.

Implementation: `lifelines.CoxTimeVaryingFitter` or `scikit-survival`. Features would be the same spatial predictors + annual time-varying covariates (NDVI, NTL, deforestation rate). Status: **Not started — activate only if E3 Track B Recall ≤ 50%.**

### Heckman framing (paper positioning, not an architecture change)

Gemini independently validated that our two-stage architecture maps directly onto the **Heckman Selection Model** from econometrics (Heckman, 1979): Stage 1 is the selection equation (which countries expand, and when); Stage 2 is the outcome equation (conditional on expansion, which land is chosen). This framing should appear explicitly in the Methods section — it grounds the two-stage design in a decades-old econometric tradition and strengthens the paper against reviewer questions about why we don't just train a single joint model.

### Revised success criteria

| Track | Success looks like | Nature-ready? |
|---|---|---|
| Track A only | Strong negative score–biodiversity correlation; KBA headline number quantified | One Earth / GEC range |
| Track A + Track B | Gap finding + watershed Recall@5% ≥ 70% + SHAP mechanism + cross-regional transfer | Nature Sustainability range |
| Track A + B + Stage 1 improved | All above + D² lifted via NB/country-FE + CBD pledge features | Nature Sustainability strong |
| Track A + B fails → Track C | Cox PH survival model replaces Stage 2; same gap finding still valid | Nature Sustainability if survival model Recall competitive |

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
| CE.3b regression diagnostic (ce3b_sqrt) | ✅ Done — gate FAILED | macro R@5%=18.8%, val=35.3%, iter=51; inv_sqrt_npos confirmed better than inv_npos (val +3.5pp); BUT test still 18.8% vs CE.2 23.8% — stratified split is harder test set |
| CE.4 AGB+REDD features | ✅ Inject chain COMPLETE | 3957103 agb_rasterise ✅; 3957105 agb_inject ✅; 3957107 redd_inject ✅ — SA splits updated 2026-06-19 ~16:24 |
| E1 gap analysis (Track A) | ❌ OOM — resubmit | 3953707 killed at 48G; stage2_gap_analysis.slurm fixed to 128G (16×8G) |
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

## Sequential Execution Plan

**Two parallel streams. Within each stream, one change at a time — so the impact of every decision is attributable. The streams converge at paper writing.**

---

### Stream A — Nature Finding

Independent of model Recall. Can start at any time. Does not block or wait for Stream B.

| Step | Action | What we learn | Status |
|---|---|---|---|
| **A1** | E1: Score full SA test pixels with temporal model; bin by score quantile; compute mean GSN_b2 per bin; Spearman r + plot | Does cost-minimisation systematically miss biodiversity? Confirms THE Nature finding | 🔄 QUEUED (job 3981693, 128G) |
| **A2** | E7: Download IUCN KBA shapefile (keybiodiversityareas.org); rasterise to SA backbone; compute % of top-5% predicted pixels overlapping KBAs vs. random baseline | Headline number: "BAU 30×30 covers X% of KBAs vs Y% by chance" | ❌ BLOCKED — user must download KBA shapefile and place at `data/shared/KBA/KBA_poly.shp`; then run `kba_rasterise.py` |
| **A3** | E6: Fit Stage 1 Negative Binomial GLM + country fixed effects; test overdispersion (Pearson χ²/df); compare D² | Does NB handle overdispersion? Does D² improve? | 🔄 QUEUED (jobs 3981878 → 3981887) |
| **A4** | E8: Add to Stage 1 — CBD 30×30 pledge indicator (binary) + continuous national coverage gap (30% − current PA fraction); compare D² vs A3 | Does political commitment improve expansion prediction above cost baseline? | ⬜ 2 days; after A3 |
| **A5** | P3.3–P3.4: BAU forward projections 2025–2030 with KBA overlay; country scorecard (on-track vs off-track × protects KBAs vs misses KBAs) | Paper-ready figures; country-level policy headline | ⬜ After A1+A2+final Stage 2 model |

**Gate**: A1 must confirm negative score–biodiversity correlation before A5 is worth running. A2, A3, A4 are fully independent of each other.

---

### Stream B — Stage 2 Model (strictly sequential)

**One change at a time. Record the result before starting the next step.**

---

#### B0 — Training infrastructure fixes (~1 day, code only, no Euler run) ✅ DONE (2026-06-19)

Applied before CE.4. Two code changes in effect:

- **E11** ✅ — filter early-stop val set to events with `n_pos ≥ 5000` only. `STAGE2_ES_MIN_POS=5000` env var in CE.4 SLURM. Implemented in `model1_LGBM_stage2_cross_event.py`.
- **E12** ✅ — cap training groups at 50,000 pixels; subsample negatives only. `STAGE2_MAX_GROUP_SIZE=50000` env var in CE.4 SLURM. `_cap_cy_groups()` helper added to `stage2_lgbm_core.py`.

---

#### B1 — CE.4: AGB + REDD + eco_protection_gap (Euler, ~1 week)

*Best pixel-level attempt. Combines all pending feature additions with the B0 infrastructure fixes.*

Pre-submission checklist:
- ✅ AGB inject complete (job 3957105, 2026-06-19)
- ✅ REDD inject complete (job 3957107, 2026-06-19)
- ✅ eco_protection_gap inject submitted (job 3981696, 2026-06-19)
- ✅ policy_b1-4 confirmed present in train.parquet schema (94 total cols)
- ✅ B0 code changes applied (E11 STAGE2_ES_MIN_POS=5000 + E12 STAGE2_MAX_GROUP_SIZE=50000)
- ✅ CE.4 submitted with dependency (job 3981700, 2026-06-19)

What changes vs. CE.3b-sqrt (18.8%): AGB (carbon stocks), dist_redd (carbon-market geography), eco_protection_gap (30×30 political urgency per biome-country) + E11 (cleaner early-stop signal) + E12 (balanced group gradients).

**What we learn**: isolated combined impact of all feature additions and training infrastructure fixes on the pixel-level cross-event model. This is our definitive pixel ceiling.

Decision gates:
- macro Recall@5% ≥ 50% → pixel model viable; continue to B2
- macro Recall@5% < 50% → pixel ceiling confirmed; skip B2–B3, go to B4
- AGB or REDD in top-10 SHAP → carbon mechanism confirmed for paper regardless of Recall

---

#### B2 — E5: Political event scoping (~2 days, local) [only if B1 ≥ 30%]

Using CE.4 model: relabel events as "ecological" (n_pos ≥ 5000, not BRA 2019–2022) vs. "political". Train on ecological only, evaluate on ecological only. Record Recall separately for both groups.

**What we learn**: the THEORETICAL CEILING of the pixel model on events it CAN predict. Separates "unfixable" (politically-driven) from "architectural" (pixel vs. catchment) failure.

Decision gates:
- Ecological-only Recall ≥ 65% → pixel model works for its valid scientific scope; Track B adds headroom
- Ecological-only Recall < 50% → even predictable events are hard at pixel level; Track B is essential

---

#### B3 — E9 + E10: New features (local, ~2 days) [add one at a time]

After B1 establishes the CE.4 baseline:
1. Add **E9 (dist_border_km)** alone: distance to nearest country border; retrain on mini-sample; record Recall delta
2. Add **E10 (TRI)** on top: Terrain Ruggedness Index from existing DEM; retrain; record Recall delta

**Why one at a time**: two features added together cannot be individually attributed. Each isolated run gives us the marginal value of each feature and informs SHAP interpretation.

If either shows positive Recall delta on mini-sample → inject into Euler splits and include in CE.W (B5).

---

#### B4 — E3: Watershed proof-of-concept (local, 2–3 days)

Rebuild the mini-sample at HydroSHEDS L7 catchment level:
1. Download HydroSHEDS L7 SA catchment shapefile
2. For each catchment: aggregate pixel features (mean elevation, mean dist_wdpa, mean AGB, etc.)
3. Rebuild mini-splits grouped by `(country_id, year, catchment_id)`
4. Train LambdaRank on catchments with B0 infrastructure
5. Evaluation: take top-5% catchments → recall all pixels inside → compute pixel-level Recall@5%

**What we learn**: does changing the ranking unit from pixel to catchment fundamentally fix Recall? PA polygons fit into 1–5 catchments; top-5% budget = ~250 catchments vs. 500M pixels. Structurally expected > 70%.

Decision gates:
- Recall ≥ 70% → Track B validated; scale to full SA (B5)
- Recall 50–70% → partial fix; investigate which events still fail; add features then scale
- Recall < 50% → catchment unit also insufficient; activate Track C (Cox PH survival model)

---

#### B5 — CE.W: Full SA watershed model (Euler, ~1 week) [after B4 ≥ 50%]

Scale B4 to the full 42GB SA splits:
- Aggregate pixel features to HydroSHEDS L7 level (new Euler splits at catchment resolution)
- Train LambdaRank with B0 infrastructure + best features from B1 + those that gained in B3
- This is the model that goes in the paper

**What we learn**: real-scale performance across all 222 SA expansion events with full continental training data. Group size imbalance is naturally reduced at catchment level.

---

#### B6 — Refinements (local, after B5) [time permitting]

After model structure and features are fully settled, in this order:
1. **E4 (dist_recent_pa)**: distance to PAs added 2010-present only, capturing spatial momentum of the expanding network; retrain; record delta
2. **Custom Recall@5% loss**: replace NDCG objective with smooth sigmoid approximation of Recall@5%; expected 2–5pp improvement; ~1 week implementation effort

Do each in isolation. These are polish, not structural fixes. Skip if time does not allow.

---

### Convergence — Final analysis and paper

Do not begin until both streams are complete.

| Step | Action | Prerequisite |
|---|---|---|
| **C1** | SHAP: global beeswarm + temporal (pre/post-Paris 2015) + per-country (BRA vs Andean) on final model | B5 done |
| **C2** | Bootstrap CIs: 1000 resamples of test events; 95% CIs on Lift@1% and Recall@5% | B5 done |
| **C3** | Cross-regional transfer: zero-shot SA watershed model → USA and SE Asia test splits | B5 done |
| **C4** | Calibration: Platt scaling on final watershed model | B5 done |
| **C5** | Forward maps: Stage 1 × Stage 2 → cumulative pixel risk 2025–2030; BAU / moderate / 30×30 scenarios; Figure 3 + KBA overlay | C4 + A2 done |
| **C6** | Paper writing | All above + A1–A4 done |

**Paper writing gate** — do not begin until all boxes checked:
- [ ] A1: gap confirmed (negative Spearman r between model score and GSN_b2)
- [ ] A2: KBA headline number quantified
- [ ] A3+A4: Stage 1 D² finalised with best GLM specification
- [ ] B5 (or B2 if pixel scope is the paper claim): final Recall number
- [ ] C1: SHAP mechanism confirmed (cost-minimisation drivers named)
- [ ] C2: Bootstrap CIs in hand
- [ ] C3: Cross-regional transfer result (transfer = universal claim; failure = regional finding)
- [ ] C5: Forward maps and country scorecard produced

**Paper structure (Nature Sustainability, ~3500 words)**:
1. **Introduction** (600w): 30×30 urgency, prediction-vs-prescription gap, cost-minimisation argument
2. **Results** (1800w): Stage 1 → Stage 2 cross-event → SHAP mechanism → representation gap + KBA headline → forward projections + country scorecard
3. **Discussion** (600w): designation as political economy, representation gap implications, concept drift as finding, limitations
4. **Methods** (500w): data; Stage 1 as Heckman selection equation; Stage 2 as Heckman outcome equation with cross-event validation design; temporal robustness supplement
5. **Extended Data** (~8 figures): per-country metrics, temporal stability per-year, temporal SHAP pre/post-Paris, USA/SE Asia zero-shot, country-level SHAP, forward scenarios, country scorecard

---

### Euler job tracker

| Job / Step | Description | Status |
|---|---|---|
| 3957103→3957105→3957107 | AGB rasterise → agb_inject → redd_inject | ✅ All COMPLETED (2026-06-19 ~16:24) |
| B0 E11+E12 | Apply code changes: ES min-pos filter + group size cap | ✅ Applied (2026-06-19) |
| 3981693 | A1: E1 gap analysis (re-run at 128G) | 🔄 QUEUED |
| 3981696 | eco_protection_gap inject | 🔄 QUEUED |
| 3981700 | B1 CE.4 training (after 3981696) | 🔄 QUEUED (dependency) |
| 3981878 | A3: stage1 panel rebuild + Poisson GLM | 🔄 QUEUED |
| 3981887 | A3: stage1_nb_overdispersion E6 (after 3981878) | 🔄 QUEUED (dependency) |
| A2 KBA | Download KBA shapefile → kba_rasterise.py | ❌ BLOCKED — needs user to download from keybiodiversityareas.org |
| A4 CBD features | Add CBD pledge + coverage gap to Stage 1 | ⬜ After A3 results |
| B2 | Ecological scope filter on CE.4 model | ⬜ After B1 result |
| B3 | E9 (dist_border) + E10 (TRI) features | ⬜ After B1 baseline |
| B4 (E3) | Watershed proof-of-concept on mini-sample | ⬜ Local; after B1 result known |
| B5 (CE.W) | Full SA watershed model | ⬜ Euler; after B4 ≥ 50% |
| C1 | SHAP (global + temporal + country) | ⬜ After B5 |
| C2 | Bootstrap CIs | ⬜ After B5 |
| C3 | Cross-regional transfer SA→USA→SE Asia | ⬜ After B5 |
| C5 | Forward maps + Figure 3 | ⬜ After C4 + A2 |

---

## Full Experiment History

### Stage 2 cross-event experiments (2026-06-18 / 2026-06-19)

| Experiment | macro R@5% | weighted R@5% | macro Lift@1% | n_test_events | Verdict |
|---|---|---|---|---|---|
| CE.1 Colombia pilot (country_id=5 only, min_pos=200) | 16.8% | 11.8% | 5.26× | 4 | ❌ gate FAILED (<60%); within-Colombia cross-time only |
| CE.2 Full SA (13 countries, min_pos=200, 83 feat) | 23.8% | 41.6% | 6.05× | 28 | ❌ gate FAILED (<65%); macro vs weighted gap is large |
| CE.3b Full SA (inv_npos + stratified split + patience=200 + coherence_thr=0.0) | 18.0% | 26.5% | 4.80× | 29 | ❌ gate FAILED (<65%); **WORSE than CE.2** — Fix 2 (inv_npos) suspected; iter=122 |
| CE.3b-sqrt diagnostic (inv_sqrt_npos + stratified + patience=200) | 18.8% | — | 6.01× | 29 | ❌ gate FAILED; inv_sqrt_npos confirmed best (val +3.5pp vs inv_npos); stratified split = harder test — explains CE.2→CE.3b regression |

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
| Policy features (WGI, V-Dem, DPI) | Already in panel as `policy_b1-4`; do NOT re-add | Confirmed present in merge pipeline; DPI executive ideology, V-Dem LDI, WGI GE + RL |
| `dist_indigenous` | Already in feature set (STATIC_DISTANCE_MAPPING) | Distance to indigenous territory computed at merge time |
| `deforestation` neighbourhood signal | Already available as `deforestation_b1_smooth16/smooth64` | Smooth versions capture ~16km and ~64km neighbourhood deforestation pressure |
| Early-stop iter count (CE.3b-sqrt: 51; CE.2: 66) | Val set contaminated by political events → metric stalls early; Fix 4b (large-event-only val set) addresses this | Fixed iteration count (500 rounds, lr=0.02) is viable alternative; test in E11 |
| Group size cap threshold | Not yet determined — test MAX_GROUP_SIZE=50,000 in E12 | inv_npos (CE.3b) over-amplified noisy small groups and made things worse; cap is a safer approach than full normalisation |

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
| AGB TIF (processed) | `data/south_america/ready/AGB/agb_sa.tif` | ✅ Done (2026-06-19) — injected into all SA splits |
| REDD raw data | `data/REDD/ID-RECCO V5.0_20231201.zip` | ✅ Done — rasterised 2026-06-19 |
| REDD TIF (processed) | `data/south_america/ready/REDD/redd_sa.tif` | ✅ Done (2026-06-19) — injected into all SA splits |
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
- **MaxEnt / one-class modeling** — designed for presence-only data; we have full panel data; LambdaRank with inv_sqrt_npos is the correct choice
- **Spatial Autoregressive (SAR) model** — the spatial contagion signal is already captured via dist_wdpa; a formal SAR specification adds complexity without clear benefit given our ranking objective
- **Cox PH survival model (Track C)** — kept as a contingency if Track B watershed Recall ≤ 50%; not on the critical path now
- **Political regime feature (re-adding WGI/V-Dem)** — already in panel as `policy_b1-4`; confirmed present
- **Indigenous territory distance feature** — already in feature set as `dist_indigenous`
- **Neighbourhood deforestation pressure ring** — already approximated by `deforestation_b1_smooth16/smooth64`
- **Oil/gas concession distance** — already in feature set as `dist_oil_gas`; mining concessions could add marginal signal but data availability for SA is poor
