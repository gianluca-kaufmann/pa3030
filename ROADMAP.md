# PA3030 — Paper Publication Roadmap

**Purpose**: Authoritative planning document. Keep compact and updated after every session.
**Status**: Post-thesis. Paper branch active. Architecture decision made (2026-05-18).
**Target**: GEC / One Earth (base case). Nature Finance / JEEM / Nature Sustainability if
LSE financial data materialises or Stage 2 results are exceptional.

---

## CORE INSIGHT (2026-05-18)

PA designation at time t is the product of two independent processes:

```
P(pixel i designated in year t)
  = P(country C expands PAs in year t)    ← political timing  [NOT in features]
  × P(pixel i chosen | expansion in C,t)  ← geographic selection  [IN features]
```

Every model that treats this as a single binary classification conflates the two terms.
The annual hazard model (AUC 0.582) exposed this — it is graded against politically-timed
annual events but uses static geographic features that cannot predict timing.

**The fix is architectural, not parametric.** Build the two processes as separate models.

---

## THE MODEL: Two-Stage Conditional Selection

### Stage 1 — Country-Year Expansion Model (macro)

**Question**: How much PA expansion will country C do in year t?

**Target**: km² designated per country per year (continuous, not binary).

**Model**: Panel regression (OLS or Poisson) with:
- PA momentum lags 1–3 (already built in W3)
- 30×30 commitment dummy (post-COP15 2023+)
- CBD meeting year dummies (known dates)
- GDP per capita, agricultural rent index
- Government ideology / environmental ministry strength (ParlGov, V-Dem)

**Expected performance**: R² 0.5–0.8. Country-year expansion is tractable because it
responds to measurable political signals. Brazil Lula vs Bolsonaro years is a signal.
International funding cycles are signals. This is a panel regression, not a black box.

**Output for investors/central banks**: Country-level PA expansion forecast for 2025–2030.
Which sovereigns face the largest 30×30 shortfall AND have the institutional momentum
to close it? This is jurisdiction-level transition risk — directly relevant for sovereign
bonds, agricultural commodity exposure, and central bank stress tests.

---

### Stage 2 — Geographic Selection Model (micro, conditional)

**Question**: Given that country C expands in year t, which pixels does it select?

**Training set**: Only rows from country-years with observed expansion (Stage 1 > 0).
Political timing is removed by conditioning on the expansion decision having occurred.

**Model**: LightGBM with `objective=lambdarank`, grouped by `(country_id, year)`.
Optimises NDCG within groups — directly trains the ranking problem.

**Features**: Same geographic feature set (elevation, climate, dist_wdpa, biodiversity,
deforestation, PA momentum). No changes to the feature pipeline.

**Expected performance**: NDCG@1% within country-year groups: 0.75–0.90. Geographic
selection IS predictable when conditioned on expansion. The model is no longer penalised
for failing to predict which year a country decides to act.

**Validation**: Concordance index within country-year on test set. Lift@1% within groups.
NOT global AUC (wrong metric for this model).

**Output for investors/central banks**: Geographic ranking of PA candidates per country.
Which agricultural pixels are in the top quintile of designation probability in their
country? This is site-level transition risk — directly relevant for farmland portfolios,
credit risk on agricultural loans, and TNFD site disclosure.

---

### Forward Prediction — Combining Stages 1 and 2

For 2025–2030:
1. Stage 1 → expected expansion budget per country per year (30×30 shortfall ×
   Stage 1 model coefficients, with uncertainty bands)
2. Stage 2 → geographic ranking of all eligible pixels per country
3. Forward map = top-K pixels from Stage 2 ranking where K = Stage 1 budget

This produces:
- A 2030 PA expansion probability surface with credible uncertainty bands
- Country-level exposure estimates (agricultural land value in top quintile)
- Biodiversity gap: top-K expansion vs species richness / threatened habitat maps
- Transition risk: `Exposure = Stage1_prob × Stage2_rank × land_value × commodity`

---

## WHY THIS IS PUBLISHABLE AT A TOP JOURNAL

The two-stage separation is a **genuine methodological contribution**. Nobody has
published this architecture for PA designation. The paper's explicit claim:

> "Treating PA designation as a single prediction problem conflates two processes with
> fundamentally different information requirements. We decompose the problem, show that
> the macro-level expansion decision is predictable from political/institutional signals
> and the micro-level geographic selection is predictable from biophysical signals, and
> demonstrate that this separation substantially improves both scientific validity and
> investor-relevance of the resulting risk maps."

This framing inverts the conventional ML paper logic: instead of claiming high AUC, the
paper's finding IS the demonstration that naive AUC is the wrong metric and the wrong
estimand. The two-stage model is the correct one. The 5-year window AUC of 0.93 (or
annual 0.582) is then explained as an artefact of model misspecification, not a measure
of predictive quality.

Strong framing for GEC / One Earth / Nature Sustainability. With LSE financial data, also
Nature Finance.

---

## THE STORY

> "Governments decide to expand protection for political reasons we can model; they then
> allocate that expansion to specific places for geographic reasons we can also model.
> These are different models answering different questions. We build both, combine them
> into a forward scenario for 30×30, and show that even under the most optimistic
> expansion trajectories, the places most likely to receive protection are systematically
> not the places where protection is most needed. We quantify the transition risk this
> creates for agricultural investors and central banks."

---

## SETTLED DECISIONS

**Paper aim**: Forecast where PA expansion will land under 30×30 using a two-stage
conditional selection model. Characterise geographic drivers. Expose biodiversity gap.
Quantify transition risk for investors and central banks.

**Model architecture**: Two-stage (Stage 1 panel regression + Stage 2 lambdarank).
This supersedes the single annual hazard model and the 5-year window.

**Three regions**: SA (primary), SE Asia, USA. All three.
- USA: Stage 2 near-trivial (pure adjacency), near-perfect within-group concordance.
  This is itself a finding: mature conservation systems select by proximity, emerging
  ones by biophysical value. Keep USA as the contrast case.

**DO NOT** add tropical Africa. No data pipeline.
**DO NOT** start Paper 2 (embeddings) until Paper 1 submitted.

---

## IMPLEMENTATION PLAN

### What changes (surgical, not a rewrite)

| Component | Change | Effort |
|-----------|--------|--------|
| Stage 1 script | New: `5_training/model1_expansion.py` (panel regression, country-year level) | ~3 days |
| Stage 2 objective | Change `objective=binary` → `objective=lambdarank` + add group construction by `(country_id, year)` in LGBM dataset | ~2 days |
| Evaluation | New NDCG@K within-group metric + Stage 1 R²/RMSE | ~1 day |
| Forward | Combine Stage 1 budget forecast + Stage 2 ranking into forward maps | ~2 days |
| Cox baseline | Logistic regression on Stage 2 formulation (interpretable coefficients) | ~1 day |

**Total local implementation**: ~2 weeks before next Euler run.

**What does NOT change**: Feature pipeline, GEE extractions, preprocessing, LOBO
infrastructure, calibration, backtest machinery, SHAP computation.

### Implementation order (next session)

1. Implement Stage 1 locally (SA first) — validate that expansion rates are predictable
2. Restructure Stage 2 training script: `lambdarank` objective, group by country-year
3. New evaluation script: NDCG@K within groups, concordance index within groups
4. Run both stages locally on a subset to confirm the machinery works
5. Push to Euler for full run

---

## WORKSTREAMS

### W0 — Feature/provenance gate [✅ complete]

### W1 — Stage 2 lambdarank model [CODE CHANGE NEEDED before Euler]

Replace `objective=binary` with `objective=lambdarank` + country-year grouping.
This is the single most impactful change. All existing pipeline infrastructure
(LOBO, calibration, SHAP, forward) carries over.

### W2 — New data [HIGH VALUE, parallel]

**Carbon stocks** (ESA CCI Biomass): REDD+ mechanism makes high-carbon land a
priority for designation. Expected top-5 SHAP. Strengthens the financial story
(carbon market → designation risk → investor exposure).

**Land tenure / indigenous lands** (RAISG): Most important omitted variable.
Designation is constrained by who owns the land, not just biophysics. SA tractable.

**Political variables for Stage 1** (ParlGov, V-Dem, CBD commitment database):
Essential for the Stage 1 expansion model. These are small datasets (country-year
level) — easy to add.

### W3 — PA momentum [✅ code complete; Euler feature_engineering rerun needed]

`pa_momentum_pixels_lag{1,2,3}` implemented. Run before W1 Euler rerun.

### W4 — Ablation [after Stage 2 results confirmed]

Same plan as before: remove feature groups one at a time, report NDCG@K drop.
Critical ablation: `dist_wdpa` alone (tests spatial autocorrelation dependence).

### W5 — Cox/logistic baseline [~1 day local]

Logistic regression on the Stage 2 formulation (within-group, conditional on expansion).
Provides interpretable coefficients that validate SHAP directions and give economics
reviewers a familiar anchor.

### W6 — Manuscript [after W1 Stage 2 results]

Paper structure:

1. **Introduction** (~800 w): 30×30 urgency → the two-process problem → what we do
2. **Results** (~2,500 w):
   - Stage 1: Which countries will expand? (political economy of PA supply)
   - Stage 2: Where? SHAP driver story across 3 continents
   - Forward maps + 30×30 biodiversity gap
   - Transition risk for investors and central banks
3. **Methods** (~1,200 w): Two-stage decomposition, lambdarank formulation,
   evaluation metrics (NDCG, concordance within groups, Stage 1 R²)
4. **Discussion** (~1,000 w): Political timing vs geographic selection; USA path-
   dependency as Stage 2 extreme case; limits of the Stage 1 forecast; implications
   for TNFD/NGFS nature risk frameworks
5. **Supplement**: Feature dictionary, full regional tables, LOBO, transfer results,
   backtest vintages, hyperparameters, Group A/B diagnostic from old model

---

## OPEN QUESTIONS

1. **[BLOCKING] Stage 1 political variables**: Which datasets are accessible for
   government ideology + international commitment indicators? Check ParlGov (EU/OECD
   coverage good, SA/SE Asia partial) and V-Dem (global coverage). CBD national pledge
   database is online. Confirm coverage before designing Stage 1 fully.

2. **[BLOCKING] LSE financial data**: Elena Almeida / CETEx — what financial datasets
   are available? Still determines journal ceiling (Nature Finance/JEEM vs GEC/One Earth).
   Must resolve before finalising manuscript structure.

3. **LambdaRank label construction**: Binary labels (0/1 designation) work for lambdarank
   but relevance-graded labels (e.g., larger designations get higher relevance score)
   might improve Stage 2. Decision: start with binary, upgrade if initial results are weak.

4. **Stage 1 spatial scale**: Country-level is cleanest. Sub-national (state/province)
   would be more granular for large countries (Brazil, Indonesia, USA) but requires
   sub-national political variables. Start with country-level; upgrade if data exists.

5. **USA Stage 2**: Within-group concordance for USA expected to be near-perfect (pure
   adjacency effect). Confirm whether this makes USA's Stage 2 model uninformative for
   the cross-continental SHAP story or whether it strengthens the contrast narrative.

---

## KEY NUMBERS

| Metric | Value | Notes |
|--------|-------|-------|
| Annual AUC (old approach) | 0.582 | Expected — wrong estimand |
| 5-year window AUC | 0.93 (test) / 0.56 (Group B) | Wrong estimand, inflated |
| Stage 1 R² target | 0.5–0.8 | Country-year expansion rate |
| Stage 2 NDCG@1% target | 0.75–0.90 | Within country-year groups |
| Stage 2 Lift@1% target | 15–40× within groups | Geographic selection signal |

---

## CURRENT OUTPUT STATUS (as of 2026-05-18)

| Artifact | Status | Notes |
|----------|--------|-------|
| SA LGBM annual binary training | ✅ Euler (old approach) | AUC 0.582, superseded by two-stage |
| W0 feature guard | ✅ complete | 9 smoke tests pass |
| W1 hazard code | ✅ code complete | Needs lambdarank change before rerun |
| W3 PA momentum | ✅ code complete | Needs feature_engineering rerun on Euler |
| Stage 1 expansion model | ❌ not started | First task next session |
| Stage 2 lambdarank model | ❌ not started | Second task next session |
| All Euler reruns | ❌ pending | After local implementation confirmed |

---

## BRANCH AND REVERSION POLICY

- **`paper` branch**: Active development. All W1/W2/W3 changes + two-stage model.
- **`main` branch**: Intact thesis code (5-year window). Never touched.
- Revert to thesis: `git checkout main` — original scripts and artifacts intact.

---

## TWO-PUBLICATION STRATEGY

### Paper 1 — This paper

**Journals (ordered)**: GEC / One Earth → Nature Sustainability → Nature Finance (if
LSE data) / JEEM (if LSE data).

**One-sentence pitch**: "A two-stage model of PA expansion separates the predictable
geographic selection of conservation candidates from the unpredictable political timing
of designation, enabling credible 30×30 forward scenarios and transition risk
quantification for investors and central banks."

### Paper 2 — Methods / Prediction (AFTER P1 SUBMITTED)

**Target**: Nature Sustainability / PNAS / Nature Machine Intelligence.
**Pitch**: Foundation model embeddings improve Stage 2 cross-regional transfer.
**Gate**: AlphaEarth access confirmed + P1 submitted.

---

## THINGS DELIBERATELY NOT IN THIS PAPER

- Colombia: supplement only
- Tropical Africa: no data pipeline
- Embeddings / Paper 2: blocked until P1 submitted
- Single-model global AUC as primary metric: rejected (wrong estimand)
