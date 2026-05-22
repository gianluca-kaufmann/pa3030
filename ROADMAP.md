# PA3030 — Paper Publication Roadmap

**Purpose**: Authoritative planning document. Keep compact and updated after every session.
**Status**: Post-thesis. Paper branch active. Architecture decision made (2026-05-18).
**Target**: Nature/Science if Stage 1 R² ≥ 0.55 and Stage 2 NDCG@1% ≥ 0.80.

---

## JUSTIFICATION FOR THE METHODOLOGICAL PIVOT

This section documents why the thesis model is inadequate and why the two-stage architecture
is the scientifically correct replacement. Use this when presenting to supervisors.

### Why the thesis model is wrong (not just weak)

The thesis model achieved test AUC = 0.93 (SA) and 0.986 (SE Asia). These numbers are
artefacts of model misspecification, confirmed by the Group A/B leakage diagnostic
(run 2026-04-25):

| Pixel group | Definition | SA RF AUC | Interpretation |
|-------------|-----------|-----------|----------------|
| Group A | Designated 2018–2019 (label overlap with training features) | 0.9994 | Memorised — features include post-designation state |
| Ambiguous | Designated 2020–2022 (partial overlap) | 0.8623 | Upper bound on honest short-horizon performance |
| Group B | Designated 2023–2024 (genuinely unseen) | 0.5587 | Near-random — true OOS performance |

The high AUC is driven almost entirely by Group A pixels whose features already reflect
their protected status. The model learned a tautology: protected-looking pixels get
protected. Group B (the only genuinely unseen pixels) performs at AUC = 0.56 —
barely above random.

The annual hazard model (AUC = 0.582) made the deeper problem explicit: static geographic
features cannot predict the *year* a country decides to expand. The model was graded
against politically-timed annual events it had no information to anticipate.

### The root cause: model misspecification, not data quality

PA designation is the product of two independent processes:

```
P(pixel i designated in year t)
  = P(country C expands PAs in year t)    ← political timing  [NOT in features]
  × P(pixel i chosen | expansion in C,t)  ← geographic selection  [IN features]
```

A single binary classifier conflates both terms. It is trained to predict the
*joint* event but only has features relevant to the second term. No amount of
feature engineering or hyperparameter tuning can fix this — it is a structural
specification error.

**The fix is architectural, not parametric.**

### Why the two-stage model is the correct specification

- Stage 1 estimates the first term: how much expansion will country C do in year t?
  This is modelled with political/institutional variables (V-Dem, WGI, GDP, CBD) that
  are genuinely informative about political timing.
- Stage 2 estimates the second term: given expansion, which pixels are selected?
  This is a ranking problem within country-year groups, using geographic features
  that are genuinely informative about geographic selection.
- Conditioning Stage 2 on the expansion event having occurred removes the political
  noise that drove the old model's poor performance.

The methodological contribution is the decomposition itself. The paper's explicit claim
is that treating PA designation as a single prediction problem is wrong, and we prove
this empirically (Group A/B diagnostic) before presenting the correct model.

---

## THE MODEL: Two-Stage Conditional Selection

### Stage 1 — Country-Year Expansion Model (macro)

**Question**: How much PA expansion will country C do in year t?

**Target**: km² designated per country per year (continuous).

**Model**: Panel regression (Poisson or negative binomial — NOT OLS; expansion data
has many country-years at zero and occasional large spikes) with:
- PA momentum lags 1–3 (already built in W3)
- 30×30 commitment dummy (post-COP15 2023+)
- CBD meeting year dummies (known dates)
- GDP per capita, agricultural land % (WB WDI)
- Democracy index (V-Dem v16: `v2x_polyarchy`, annual, 1900–2025)
- Government effectiveness (WB WGI: `GOV_WGI_GE.EST`, annual, 1996–2024)
- Note: ParlGov DROPPED — EU/OECD only, no coverage for SA or SE Asia

**Honest performance expectation**: R² 0.40–0.65. Country-year PA expansion is
partially tractable from political signals, but lumpy international funding cycles
and individual political events create irreducible noise. The autoregressive baseline
(PA momentum lags alone) must be computed first; V-Dem + WGI are valuable only if
they improve substantially on that baseline. If Stage 1 R² < 0.40, forward scenario
uncertainty bands become too wide to claim investor-relevance — Stage 1 is then
reframed as illustrative macro context rather than a forecasting model. Stage 2
results stand independently either way (see below).

**Output for investors/central banks**: Country-level PA expansion likelihood for
2025–2030. Which sovereigns face the largest 30×30 shortfall AND have the
institutional capacity to close it? Jurisdiction-level transition risk: directly
relevant for sovereign bond exposure and central bank stress tests.

---

### Stage 2 — Geographic Selection Model (micro, conditional)

**Question**: Given that country C expands in year t, which pixels does it select?

**Training set**: Only rows from country-years with observed expansion (Stage 1 > 0).
Political timing noise is removed by conditioning on the expansion event.

**Model**: LightGBM `objective=lambdarank`, grouped by `(country_id, year)`.
Optimises NDCG within groups — directly trains the ranking problem.

**Features**: Existing geographic feature set (elevation, climate, dist_wdpa,
biodiversity, deforestation, PA momentum). No changes to feature pipeline.

**Honest performance expectation**: NDCG@1% within country-year groups: 0.72–0.88.
Geographic selection IS predictable when conditioned on expansion. The model is no
longer penalised for failing to predict which year a country acts.

**Critical baseline**: The `dist_wdpa`-only model must be computed as the naïve
baseline (W4 ablation, run early). If Stage 2 NDCG@1% cannot beat this baseline
by ≥ 5 percentage points, the 60-feature pipeline adds nothing beyond spatial
autocorrelation — that would be a finding in itself, but a much narrower paper.

**Validation**: NDCG@K within country-year groups, concordance index within groups,
Lift@1% within groups. NOT global AUC (wrong metric for a conditional ranking model).

**Stage 2 is the primary empirical contribution of the paper.** It stands alone
regardless of Stage 1 performance. The SHAP driver story (which geographic factors
drive selection across 3 continents), the biodiversity gap (top-K expansion vs
priority habitat maps), and the USA adjacency contrast are all Stage 2 outputs.

**Output for investors/central banks**: Site-level transition risk. Which agricultural
pixels are in the top quintile of designation probability in their country? Directly
relevant for farmland portfolios, agricultural credit risk, and TNFD site disclosure.

---

### Forward Prediction — Combining Stages 1 and 2

For 2025–2030:
1. Stage 1 → expected expansion budget per country (30×30 shortfall × Stage 1
   coefficients, with bootstrapped uncertainty bands). Three scenarios: BAU,
   moderate (midpoint), 30×30-compliant.
2. Stage 2 → geographic ranking of all eligible pixels per country.
3. Forward map = top-K pixels from Stage 2 ranking where K = Stage 1 budget.

**Spatial aggregation**: Top-K individual pixels produce scattered 1km² patches.
Run a hexagonal binning or kernel density step before producing map figures to show
spatial concentration of high-probability pixels. Acknowledge explicitly in Methods
that this is a pixel-level likelihood surface, not a contiguous PA boundary proposal.
Show empirically that high-probability pixels cluster geographically (if they do) —
this validates the map without requiring contiguity.

This produces:
- 2030 PA expansion probability surface with scenario uncertainty bands
- Country-level shortfall + capacity table (sovereign bond / stress test input)
- Biodiversity gap: top-K expansion vs species richness / threatened habitat maps
- Transition risk formula: `Exposure = Stage1_prob × Stage2_rank × land_value`

---

## WHY THIS IS PUBLISHABLE AT A TOP JOURNAL

The two-stage decomposition is a **genuine methodological contribution**. No prior
PA prediction paper has built this architecture. The paper's argument:

> "Treating PA designation as a single prediction problem conflates two processes
> with fundamentally different information requirements. We prove this empirically
> using a Group A/B leakage diagnostic, decompose the problem into its constituent
> parts, and show that the micro-level geographic selection process is strongly
> predictable once political timing noise is removed. We use the resulting model to
> produce credible 30×30 forward scenarios and to quantify the gap between where
> protection will go and where it is most needed."

This framing inverts conventional ML paper logic: the finding IS the demonstration
that naive AUC is the wrong metric and wrong estimand. The 0.93 AUC and the 0.582
AUC are both explained as model misspecification artefacts, not measures of
predictive quality. The two-stage model is the correct specification.

**Why GEC / One Earth**: policy-facing, quantitative methods accepted, biodiversity
gap + transition risk narrative is precisely on-scope. 30×30 is the defining
conservation policy of the decade.

**Why Nature Sustainability is reachable**: if Stage 1 R² ≥ 0.55 and Stage 2
NDCG@1% ≥ 0.80, the combined forward scenario is credible enough for their
standards. The three-continent scope and TNFD/NGFS relevance are consistent with
their recent published work.

---

## THE STORY (one paragraph for co-authors and supervisors)

> "Governments decide to expand protection for political reasons we can partially
> model; they then allocate that expansion to specific places for geographic reasons
> we can model well. These are structurally different problems requiring different
> models. We prove the misspecification of the conventional single-model approach
> using a leakage diagnostic, then build both stages separately. The combined model
> produces 30×30 forward scenarios and shows that even under optimistic expansion
> trajectories, the places most likely to receive protection are systematically not
> the places where protection is most needed. We quantify the agricultural transition
> risk this creates."

---

## PRE-FLIGHT CHECKS (blocking — complete before writing Stage 1 or Stage 2 code)

These two checks take ≤ 1 day and determine whether the design requires adjustment
before implementation begins. Do not skip them.

### Check A — Within-group sample sizes (Stage 2 viability)

Query the existing parquet panels and count, per `(country_id, year)`, the number
of positive transitions. Report: distribution of group sizes, number of groups with
≥ 5 positives, ≥ 10 positives, median group size. LambdaRank requires multiple
positives per group to learn meaningful rankings.

**Decision rule**:
- Median group positives ≥ 5 across all regions → proceed as designed (annual groups)
- Median group positives 2–4 → aggregate to 2–3 year windows before grouping
- Median group positives < 2 → Stage 2 group definition needs rethinking; escalate

Script: `scripts/regions/shared/stage2_group_size_check.py` (to be written, ~30 lines).

### Check B — Stage 1 autoregressive baseline

Before building the full political panel model, fit a simple Poisson regression with
PA momentum lags 1–3 only (no political variables). Report pseudo-R² and RMSE.
This is the baseline Stage 1 must beat; it also calibrates R² expectations.

**Decision rule**:
- Momentum-only pseudo-R² ≥ 0.50 → adding V-Dem + WGI can push total to 0.55–0.65;
  Stage 1 is a credible macro model
- Momentum-only pseudo-R² 0.30–0.50 → Stage 1 framed as illustrative context,
  not a forecast; paper lead stays on Stage 2
- Momentum-only pseudo-R² < 0.30 → Stage 1 expansion is dominated by noise;
  drop Stage 1 from main results, include as supplement or Discussion

Script: `scripts/regions/south_america/5_training/stage1_ar_baseline.py` (~50 lines).

---

## SETTLED DECISIONS

**Paper aim**: Forecast where PA expansion will land under 30×30 using a two-stage
conditional selection model. Characterise geographic drivers. Expose biodiversity gap.
Quantify transition risk for investors and central banks.

**Model architecture**: Two-stage (Stage 1 panel regression + Stage 2 lambdarank).
Supersedes single annual hazard model and 5-year window. Neither can be reverted to
as a primary model — they are shown to be misspecified.

**Stage 2 is primary, Stage 1 is context**: The paper's empirical spine is Stage 2.
Stage 1 provides the macro expansion budget needed for the forward map. If Stage 1
performs poorly, the paper narrows to "here is where protection will go, conditional
on a country choosing to act" — still publishable. Do not gate the whole paper on
Stage 1 performance.

**Three regions**: SA (primary), SE Asia, USA. All three.
- USA: near-perfect within-group concordance expected (pure adjacency effect). This
  IS a finding: mature conservation systems select by proximity, emerging ones by
  biophysical value. USA is the contrast case, not a validation failure.

**COP15 structural break**: Training runs to 2019; 30×30 commitment post-2022.
Stage 1 is extrapolating to a qualitatively different political regime. The COP15
dummy partially addresses this but does not resolve it. Address explicitly in Methods
as a design limitation, not an afterthought. Frame it as: "Stage 1 estimates the
pre-30×30 expansion rate; the COP15 dummy scales the forecast upward under the
assumption that the commitment is partially binding. We present sensitivity scenarios."

**Literature positioning**: Stage 1 must add something existing political ecology /
PA supply papers do not already do. Confirmed targets for literature check before
submission: Joppa & Pfaff (2009), Baldi et al. (2010), Nolte et al. (2010),
and recent country-year PA supply models. Stage 1 covariates (V-Dem + WGI at
country-year) are not used in prior PA prediction papers — this is the gap.

**LambdaRank labels**: Binary (0/1 designation). Standard for NDCG with binary
relevance. Upgrade to graded relevance (larger designations = higher score) only
if initial NDCG@1% < 0.70.

**Stage 1 spatial scale**: Country-level. Sub-national only if Check A reveals
that country-year groups are too sparse and aggregating to 3-year windows is
insufficient — then state-level grouping for Brazil/Indonesia/USA may help.

**DO NOT** add tropical Africa. No data pipeline.
**DO NOT** start Paper 2 (embeddings) until Paper 1 submitted.

---

## IMPLEMENTATION PLAN

### What changes (surgical, not a rewrite)

| Component | Change | Effort |
|-----------|--------|--------|
| Pre-flight Check A | Group size diagnostic script | ~0.5 days |
| Pre-flight Check B | Stage 1 AR baseline script | ~0.5 days |
| Stage 1 script | New: `5_training/model1_expansion.py` (Poisson panel regression) | ~3 days |
| Stage 2 objective | `objective=binary` → `objective=lambdarank` + country-year grouping | ~2 days |
| Stage 2 calibration | Platt/isotonic on lambdarank scores before transition risk formula | ~0.5 days |
| Evaluation | NDCG@K within groups, concordance index, Stage 1 pseudo-R²/RMSE | ~1 day |
| Naïve baseline (W4 early) | `dist_wdpa`-only lambdarank (required reference point) | ~0.5 days |
| Forward | Stage 1 budget × Stage 2 ranking → forward maps + hex-binned figures | ~2 days |
| Logistic baseline (W5) | Logistic regression on Stage 2 formulation (interpretable coefficients) | ~1 day |

**Total local implementation**: ~2 weeks before next Euler run.

**What does NOT change**: Feature pipeline, GEE extractions, preprocessing, LOBO
infrastructure, calibration, backtest machinery, SHAP computation, existing splits.

### Implementation order

0. ~~**Verify Stage 1 political data coverage**~~ **[DONE 2026-05-19]**
   V-Dem v16 + WB WGI confirmed for all three regions. ParlGov dropped.
   See `outputs/data_checks/stage1_political_coverage.json`.

0a. **[BLOCKING] Pre-flight Check A** — within-group sample sizes. Must run before
    Stage 2 design is locked.

0b. **[BLOCKING] Pre-flight Check B** — Stage 1 AR baseline (momentum lags only).
    Sets realistic R² expectations before building full political model.

1. Implement Stage 1 locally (SA first) — full political panel model

2. Restructure Stage 2: `lambdarank` objective, group by `(country_id, year)`

3. **Run naïve `dist_wdpa`-only baseline immediately** (before full Stage 2 run).
   This is the benchmark Stage 2 must beat. Do not wait for W4.

4. New evaluation script: NDCG@K within groups, concordance index within groups

5. Run both stages locally on a subset to confirm machinery works end-to-end

6. Push to Euler for full run across all three regions

---

## WORKSTREAMS

### W0 — Feature/provenance gate [✅ complete]

### W1 — Stage 2 lambdarank model [CODE CHANGE NEEDED before Euler]

Replace `objective=binary` with `objective=lambdarank` + country-year grouping.
Single most impactful change. LOBO, calibration, SHAP, forward all carry over.

### W2 — New data [HIGH VALUE, parallel with W1]

**Carbon stocks** (ESA CCI Biomass): REDD+ mechanism makes high-carbon land a
priority for designation. Expected top-5 SHAP. Directly strengthens the financial
story (carbon market → designation incentive → investor transition exposure).
Do SA first; SE Asia second.

**Land tenure / indigenous lands** (RAISG): Biggest omitted variable for SA.
Designation is constrained by land ownership, not just biophysics. Public and
indigenous lands are the path of least political resistance. SA tractable via RAISG.

**Political variables** (V-Dem v16, WB WGI, WDI): Already confirmed. Download and
format as country-year panel for Stage 1 script. Small datasets, low effort.

### W3 — PA momentum [✅ code complete; Euler feature_engineering rerun needed]

`pa_momentum_pixels_lag{1,2,3}` implemented. Run before W1 Euler rerun.

### W4 — Ablation [run naïve baseline early; full ablation after Stage 2 confirmed]

**Naïve baseline (run immediately)**: `dist_wdpa`-only lambdarank model. This is
the critical reference point. Stage 2 must beat it by ≥ 5pp NDCG@1% to claim the
full feature set adds meaningful signal beyond spatial autocorrelation.

**Full ablation** (after Stage 2 results confirmed): Remove feature groups one at a
time (climate, biodiversity, deforestation, PA momentum, terrain). Report NDCG@K
drop per group. Generates the feature importance table for Methods section.

### W5 — Logistic baseline [~1 day local, after W1]

Logistic regression on Stage 2 formulation (within-group, conditional on expansion).
Interpretable coefficients anchor the SHAP directions for economics reviewers and
satisfy the "show us a simple model first" reviewer request.

### W6 — Manuscript [after W1 Stage 2 results, with Stage 1 in whatever state it is]

**Paper structure**:

1. **Introduction** (~800 w): 30×30 urgency → the two-process misspecification
   problem (with Group A/B diagnostic as evidence) → what we do
2. **Results** (~2,500 w):
   - Stage 2: Where will expansion go? SHAP driver story across 3 continents.
     USA adjacency contrast. Biodiversity gap.
   - Stage 1: Which countries will expand? (political economy of PA supply —
     or: macro context, depending on Check B outcome)
   - Forward maps + 30×30 scenarios (BAU / moderate / compliant)
   - Transition risk exposure estimates for investors
3. **Methods** (~1,200 w): Group A/B diagnostic. Two-stage decomposition.
   LambdaRank formulation. NDCG within groups. Stage 1 Poisson panel.
   COP15 structural break handling. Forward spatial aggregation.
4. **Discussion** (~1,000 w): Political timing vs geographic selection; USA
   path-dependency as Stage 2 extreme case; COP15 extrapolation limits;
   TNFD/NGFS implications; what a prescriptive model would look like vs this
   descriptive model.
5. **Supplement**: Feature dictionary, full regional tables, LOBO, cross-continental
   transfer, backtest vintages, hyperparameters, Group A/B full diagnostic,
   `dist_wdpa`-only baseline comparison.

**Note on paper structure**: Stage 2 leads Results, not Stage 1. This protects the
paper against the scenario where Stage 1 performs modestly — the lead finding is
still strong regardless.

---

## OPEN QUESTIONS

1. **[RESOLVED 2026-05-19] Stage 1 political data coverage**: V-Dem v16 +
   WB WGI confirmed for all three regions. ParlGov dropped (EU/OECD only).
   See `outputs/data_checks/stage1_political_coverage.json`.

2. **[RESOLVED] LSE financial data**: Dropped. Journal target fixed at GEC / One Earth.

3. **[BLOCKING — Pre-flight Check A] Within-group sample sizes**: How many
   country-year groups have ≥ 5 positive transitions? Determines whether annual
   grouping is viable or whether 2–3 year aggregation is needed for SE Asia.

4. **[BLOCKING — Pre-flight Check B] Stage 1 AR baseline R²**: Momentum-only
   pseudo-R² sets realistic expectations before the full political model is built.

5. **USA Stage 2**: Near-perfect concordance expected (adjacency). Confirm in
   local run. If concordance > 0.95, USA Stage 2 is the contrast case; if lower
   than expected, investigate what other features matter in the USA context.

6. **Graded relevance for LambdaRank**: Start with binary labels. Upgrade to
   area-weighted relevance (larger designations = higher score) only if initial
   NDCG@1% < 0.70 after first Euler run.

7. **Sub-national Stage 1**: Country-level is the default. If Check A reveals
   group sparsity requiring 2–3 year aggregation AND Check B shows low AR R²,
   investigate whether state-level grouping for Brazil/Indonesia/USA improves
   Stage 1 fit. This is a fallback, not the plan.

---

## KEY NUMBERS

| Metric | Value | Notes |
|--------|-------|-------|
| Thesis 5-year AUC (Group A) | 0.9994 | Leakage — memorised |
| Thesis 5-year AUC (Group B) | 0.5587 | Genuinely unseen — near-random |
| Annual AUC (old approach) | 0.582 | Correct outcome for a misspecified model |
| Stage 1 R² target (honest) | 0.40–0.65 | Depends on Check B AR baseline |
| Stage 2 NDCG@1% target | 0.72–0.88 | Within country-year groups |
| Stage 2 naïve baseline | TBD (dist_wdpa only) | Must beat by ≥ 5pp |
| Stage 2 Lift@1% target | 10–35× within groups | Geographic selection signal |

---

## CURRENT OUTPUT STATUS (as of 2026-05-19)

| Artifact | Status | Notes |
|----------|--------|-------|
| SA LGBM annual binary | ✅ Euler | AUC 0.582 — misspecified, superseded |
| Group A/B leakage diagnostic | ✅ complete | SA Group B AUC 0.5587 confirmed |
| W0 feature guard | ✅ complete | 9 smoke tests pass |
| W1 hazard code | ✅ code complete | Needs lambdarank change before rerun |
| W3 PA momentum | ✅ code complete | Needs feature_engineering rerun on Euler |
| Stage 1 political data coverage | ✅ complete | V-Dem + WGI confirmed |
| **Pre-flight Check A** (group sizes) | ✅ code ready | `scripts/regions/shared/stage2_group_size_check.py` — run on Euler panels |
| **Pre-flight Check B** (AR baseline) | ✅ code ready | `scripts/regions/south_america/5_training/stage1_ar_baseline.py` |
| Stage 1 expansion model | ✅ code ready | `stage1_data_builder.py`, `model1_expansion.py` — needs political CSVs + panel |
| Stage 2 lambdarank model | ✅ code ready | `model{1,2,3}_LGBM_stage2` + `shared/training/stage2_lgbm_core.py` |
| dist_wdpa naïve baseline | ✅ code ready | `model1_LGBM_stage2_naive` |
| `country_id` in panels | ✅ code ready | feature_engineering keeps column; **Euler FE rerun required** |
| Stage 2 tuning / SLURM | ✅ code ready | `tuning_lgbm_stage2.slurm` × 3 regions |
| Two-stage forward predict | ✅ code ready | `two_stage_predict_core.py`; set `PA3030_FORWARD_TWO_STAGE=1` |
| All Euler reruns | ❌ pending | FE rerun + Stage 2 train/tune after Checks A/B pass |

---

## BRANCH AND REVERSION POLICY

- **`paper` branch**: Active development. All W1/W2/W3 changes + two-stage model.
- **`main` branch**: Intact thesis code (5-year window). Never touched.
- Revert to thesis: `git checkout main` — original scripts and artifacts intact.

---

## TWO-PUBLICATION STRATEGY

### Paper 1 — This paper

**Journals (ordered)**: GEC / One Earth → Nature Sustainability → JEEM.

**One-sentence pitch**: "A two-stage model of protected area designation separates
the predictable geographic selection of conservation candidates from the politically-
timed expansion decision, enabling credible 30×30 forward scenarios and transition
risk quantification for agricultural investors and central banks."

### Paper 2 — Methods / Embeddings (AFTER P1 SUBMITTED)

**Target**: Nature Sustainability / PNAS / Nature Machine Intelligence.
**Pitch**: Foundation model embeddings improve Stage 2 cross-regional transfer.
**Gate**: AlphaEarth access confirmed + P1 submitted.

---

## THINGS DELIBERATELY NOT IN THIS PAPER

- Colombia: supplement only
- Tropical Africa: no data pipeline
- Embeddings / Paper 2: blocked until P1 submitted
- Single-model global AUC as primary metric: rejected (wrong estimand, proven empirically)
- Graded LambdaRank labels: fallback only if binary NDCG@1% < 0.70
- Sub-national Stage 1: fallback only if country-level Check B baseline is very weak
