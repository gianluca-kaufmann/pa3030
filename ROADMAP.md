# PA3030 — Paper Publication Roadmap

**Purpose**: Authoritative planning document. Keep compact and updated after every session.
**Status**: Post-thesis. Paper branch active. Annual hazard results back from Euler.
**Target**: GEC / One Earth (primary). Nature Finance / JEEM if LSE financial data materialises.

---

## STRATEGIC CONTEXT (updated 2026-05-17)

### The fundamental insight from the annual hazard results

SA LGBM annual hazard (Euler, 2026-05-17):

| Metric   | Old model (win5, SE Asia) | New model (annual, SA) |
|----------|--------------------------|------------------------|
| ROC-AUC  | 0.980                    | 0.582                  |
| PR-AUC   | 0.538                    | 0.005                  |
| Lift@1%  | 74×                      | 2.7×                   |

This collapse is **not a code bug**. It reveals a structural problem with the learning task.

PA designation at time t is the product of two independent processes:

```
P(pixel i designated in year t)
  = P(country C expands PAs in year t)    ← political timing  [NOT in features]
  × P(pixel i chosen | expansion in C,t)  ← geographic selection  [IN features]
```

The feature set — elevation, climate, biodiversity, dist_wdpa — carries signal only for the
**second term**. The first term (does Brazil designate in 2018 vs 2019?) is driven by
election cycles, CBD meetings, and donor funding, all invisible to geographic features.

The 5-year window was accidentally aggregating over enough political cycles that the
timing noise averaged out and the geographic selection signal dominated. The annual
model forces prediction of both terms simultaneously and is dominated by the
unpredictable first term.

AUC 0.582 is near-random because the model produces a stable geographic ranking
(mostly static features) graded against politically-timed annual events.

### What the paper needs vs what it does NOT need

The paper's three deliverables are **geographic ranking problems**, not annual precision problems:

1. **Forward risk maps** — Where is PA expansion most likely by 2030? This is a geographic
   screening question. A model with moderate discriminative power that correctly clusters
   risk in the Amazon frontier, Cerrado edge, and SE Asian deforestation fronts is
   actionable, regardless of annual AUC.

2. **30×30 gap finding** — Where does expansion go vs where biodiversity needs protection?
   This is a spatial overlay. Fully independent of AUC.

3. **Transition risk for investors/central banks** — What agricultural exposure sits in
   high-designation-probability zones? This is also a spatial overlay:
   `Exposure = P(designation by 2030) × land value × commodity sensitivity`.
   Investors and central banks need geographic risk stratification, not precise annual
   timing. Even a rough quintile ranking of land-by-designation-risk is actionable.

None of these require annual AUC > 0.80. They require geographically sensible probability
surfaces and stable geographic rankings.

### The one diagnostic that determines everything

Before any architectural decision: **do the 5-year cumulative predictions from the annual
model look geographically sensible?**

Compute `1 − (1 − ĥ)^5` from the Euler output and check:
- Do high-probability pixels cluster in the Amazon frontier, Atlantic Forest edge,
  Cerrado transition zone, SE Asian deforestation fronts?
- Does the backtest (train-to-2015, predict-2016–2020) show concentration in the
  right geographic zones?

If YES → the model is working correctly as a geographic screening tool. Annual AUC 0.582
is the expected result when grading stable geographic rankings against annually-noisy
political events. Proceed with cumulative evaluation as the primary metric.

If NO (forward map looks like noise) → deeper data or feature problem needs diagnosis
before any compute is invested.

**Run this diagnostic first, before the next Euler job.**

---

## METHODOLOGY DECISION (pending diagnostic)

Three options, in order of preference:

### Option A — Annual hazard + cumulative evaluation [CURRENT PATH]

Keep the annual hazard model. Report `1 − (1 − ĥ)^5` as the primary forward output.
Evaluate against 5-year cumulative ground truth (already implemented in backtest_core).
The annual AUC (0.582) is reported honestly in Methods as the expected result given that
the model captures geographic selection but not political timing — this is itself a
finding. Primary paper metric = forward backtest lift@K%, not annual AUC.

Viable if: the cumulative maps look geographically sensible.

### Option B — Lambdarank reframe [METHODOLOGICAL UPGRADE, ~2 weeks]

Reformulate as a conditional ranking problem within `(country_id, year)` groups.
LightGBM `objective=lambdarank`, evaluation metric = NDCG@K within groups.

This directly models the second term only:
*Given that country C designated some area in year t, which pixels did it select?*

Expected NDCG improvement is substantial because political timing is removed from the
loss function entirely. The forward maps are identical in interpretation: rank all
eligible pixels per country, top-ranked = highest PA expansion probability.

Implement if: Option A maps look sensible but reviewers are likely to push back on
annual AUC 0.582 and the two-process explanation needs a stronger methodological anchor.

### Option C — Revert to 5-year window with honest Group A/B reporting [FALLBACK]

Reinstate `transition_01_win5` as target. Report test AUC 0.93 WITH the explicit
decomposition: Group A (path-dependent, structural overlap) AUC ≈ 0.99 vs
Group B (genuinely unseen) AUC ≈ 0.56. Frame Group A AUC as a finding about
path-dependency, not a leak. The Group A/B contrast is publishable at GEC/One Earth.

Use only if: Option A cumulative maps fail the geographic sensibility check AND the
lambdarank reframe (Option B) is not feasible within the paper timeline.

---

## SETTLED DECISIONS

**Paper aim**: Forecast where PA expansion will land under 30×30, characterise the
geographic drivers, expose the biodiversity gap, and quantify transition risk for
agricultural assets. The model is a screening tool, not a precise prediction engine.

**Three regions**: SA (primary), SE Asia, USA. Keep all three.
- USA path-dependency (AUC ≈ 0.99 that collapses OOS) is a positive finding about
  mature conservation systems — feature in Discussion, not Limitations.
- SA→SE Asia transfer was AUC 0.796 under 5-year window; re-evaluate under hazard/lambdarank.

**DO NOT** add tropical Africa. No data pipeline exists.
**DO NOT** implement prediction-vintage snapshots.
**DO NOT** start Paper 2 (embeddings) until Paper 1 is submitted.

---

## CURRENT OUTPUT STATUS (as of 2026-05-17)

| Artifact | Status | Notes |
|----------|--------|-------|
| W0 feature guard | ✅ complete | 9 smoke tests pass |
| W1 hazard code | ✅ complete | All scripts updated, guard in place |
| W3 PA momentum | ✅ code complete | Needs feature_engineering rerun on Euler |
| SA LGBM annual training | ✅ Euler complete | AUC 0.582 — diagnostic pending |
| SA RF annual training | ❌ pending | |
| SE Asia annual training | ❌ pending | |
| USA annual training | ❌ pending | |
| Cumulative eval (all regions) | ❌ pending | **FIRST PRIORITY** |
| Calibration, LOBO, transfer | ❌ pending | After cumulative eval passes |
| Forward maps | ❌ pending | After cumulative eval passes |
| Cox/logistic baseline | ❌ not started | ~2 days local, no Euler |
| Ablation study | ❌ not started | After main results |
| Manuscript | ❌ not started | After results |

---

## WORKSTREAMS

### IMMEDIATE — Cumulative evaluation diagnostic

**Before any further Euler compute**, run locally or on Euler:

1. Load the SA LGBM annual hazard model output (scored test parquet from Euler).
2. Compute `y_pred_proba_5yr_cumulative = 1 − (1 − h)^5` — already in `predict_core.py`.
3. Plot a geographic probability map of SA. Check visually:
   - High-probability zones: Amazon frontier, Cerrado edge, Atlantic Forest remnants?
   - Or noise / implausible pattern?
4. Run the forward backtest (`backtest_core.py`) with cumulative predictions.
   Report Lift@1%, Lift@5%, PR-AUC against 5-year cumulative ground truth.
5. **Decision gate**: If maps look sensible → proceed with Option A (annual + cumulative).
   If maps look like noise → diagnose before any further compute.

Additionally, compute **country-year stratified AUC**: within each `(country, year)` group
in the test set, compute concordance between model scores and designations. This isolates
the geographic selection signal from political timing noise. If stratified AUC ≈ 0.75–0.85,
the model is working correctly and the 0.582 global AUC is explained by the two-process
decomposition.

### W1 — Hazard model Euler reruns [partially complete]

SA LGBM done. Remaining:
- `training_rf.slurm` × SA
- `tuning_lgbm.slurm` + `training_lgbm.slurm` × SE Asia
- `tuning_lgbm.slurm` + `training_lgbm.slurm` × USA
- Same for RF all regions
- Then: calibration → benchmark → LOBO → transfer → forward (all 3 regions)

**Hold** further Euler jobs until the cumulative evaluation diagnostic passes.

### W2 — New data [HIGH VALUE, parallel with W1]

**Carbon stocks** (ESA CCI Biomass, GEE): REDD+ and carbon markets make high-carbon
land attractive to protect. Expected top-5 SHAP in SA and SE Asia. Strengthens the
financial story (carbon market → designation risk → investor exposure).

**Land tenure / indigenous lands** (RAISG, LandMark): Strongest missing omitted variable.
Designation is constrained by governance and land rights, not just biophysics. USA federal
land is tractable; SA/SE Asia requires coverage/licensing check.

Add only if feasible within timeline. Carbon stocks first.

### W3 — PA momentum [code ✅; Euler feature_engineering rerun needed]

`pa_momentum_pixels_lag{1,2,3}` implemented in feature_engineering. Run
`feature_engineering.slurm` × 3 regions before re-kicking W1 training. These lags
are the only time-varying governance signal currently in the feature set.

### W4 — Ablation study [after W1 results confirmed]

Remove feature groups one at a time, report PR-AUC drop. Critical ablation:
`dist_wdpa` alone — if removing it collapses the model, the model is mostly learning
spatial autocorrelation. Other groups: terrain, climate, biodiversity, infrastructure,
governance, economic value.
~15–20 SLURM jobs. Results go into §Methods.

### W5 — Cox/logistic duration baseline [~2 days local, no Euler]

Logistic regression on the same annual-hazard target, same features.
Provides: (a) coefficient estimates that economics reviewers trust; (b) baseline showing
what LGBM adds over a well-specified linear model; (c) coefficient signs that validate
SHAP directions. SA first, then SE Asia.

### W6 — Manuscript [after W1 + diagnostic pass]

Paper structure (high-impact, ~5,000 words main + supplement):

1. **Introduction** (~800 w): 30×30 urgency → the two-process insight → what we do
2. **Results** (~2,500 w): Driver story (SHAP across 3 continents) → forward maps →
   30×30 biodiversity gap → transition risk quantification
3. **Methods** (~1,200 w): Two-process decomposition, hazard model formulation,
   conditional ranking interpretation, data, evaluation (cumulative AUC + Lift + NDCG)
4. **Discussion** (~1,000 w): Geographic selection vs political timing; opportunity-cost
   bias in designation; path-dependency in USA; SA→SE Asia transfer; limitations
5. **Supplement**: Feature dictionary, hyperparameters, full regional tables, LOBO,
   transfer, backtest vintages, annual AUC with explanation

Key narrative:
*"PA expansion follows predictable geographic patterns — shaped by existing conservation
geography, biophysical value, and land-use pressure. The political decision of when to
act is not forecastable from geographic features, but which areas will be targeted when
action happens is. We characterise these drivers across three continents, forecast where
30×30 will land, expose the biodiversity gap between where protection goes and where it
is needed most, and quantify the transition risk for agricultural assets in
high-designation-probability zones."*

---

## OPEN QUESTIONS

1. **[BLOCKING] Diagnostic result**: Do the SA LGBM cumulative maps look geographically
   sensible? Determines which methodology option to pursue. Run before next session.

2. **[BLOCKING] LSE financial data**: What land price, credit spread, farmland REIT, or
   concession value data can Elena Almeida / CETEx contribute? Determines journal target
   (Nature Finance/JEEM vs GEC/One Earth) and whether the transition risk section is
   evidential (asset pricing anomaly) or illustrative (exposure quantification).

3. **Methodology choice**: Option A (annual + cumulative) vs Option B (lambdarank)?
   Decide after diagnostic result. If lambdarank, implementation is ~2 weeks locally.

4. **Backtesting metric**: Annual hazard predictions aggregated to 5-year cumulative
   before comparing against 5-year ground truth (primary). Annual-only backtest as
   secondary diagnostic. Both already in `backtest_core.py`.

5. **USA policy features**: Federal governance differs structurally from SA/SE Asia
   country-level indicators. Check whether excluding USA from the governance SHAP
   narrative makes the cross-continental comparison cleaner.

---

## KEY NUMBERS

Current (Euler, annual hazard, SA LGBM):
| Metric | Value | Notes |
|--------|-------|-------|
| Annual ROC-AUC | 0.582 | Expected given two-process decomposition |
| Annual PR-AUC | 0.005 | Base-rate collapse (annual events ~5× rarer than 5-year) |
| Annual Lift@1% | 2.7× | Graded against politically-timed annual events |
| Cumulative AUC | **TBD** | Run diagnostic — this is the primary metric |
| Country-year stratified AUC | **TBD** | Run diagnostic — tests geographic selection signal |

Reference (5-year window, post-fix, SA RF):
| Metric | Value |
|--------|-------|
| Test AUC | 0.9268 |
| PR-AUC | 0.6337 |
| Lift@1% | 67.6× |
| Forward backtest AUC (T=2015) | 0.748 |
| Group B AUC (genuinely unseen) | 0.5587 |

Target (cumulative hazard, primary metric):
- Forward backtest Lift@1% ≥ 5× → maps are actionable for investor risk screening
- Country-year stratified AUC ≥ 0.70 → geographic selection signal confirmed

---

## BRANCH AND REVERSION POLICY

- **`paper` branch**: All W1/W2/W3 hazard model changes. Active development here.
- **`main` branch**: Intact thesis code (5-year window, original splits). Never touched.
- To revert to thesis settings: `git checkout main` — all original scripts, splits,
  and model artifacts are preserved there.
- To recover specific thesis outputs: they remain in `outputs/` on main, untouched.

---

## TWO-PUBLICATION STRATEGY

### Paper 1 — Finance / Economics (THIS PAPER)

**Journals**: GEC / One Earth (base case). Nature Finance / JEEM if LSE data confirms
the "unpriced risk" test is feasible.

**One-sentence pitch**: "Geographic screening of protected-area expansion reveals where
30×30 will land and quantifies the transition risk for agricultural investors and central
banks."

**Core contributions**:
1. Cross-continental characterisation of PA designation drivers (SHAP, 3 regions, 20 years)
2. Forward maps of PA expansion under 30×30 with biodiversity gap analysis
3. Transition risk quantification for agricultural assets (requires LSE financial data
   for the evidential version; illustrative version does not)
4. The two-process insight: geographic selection is forecastable; political timing is not

### Paper 2 — Methods / Prediction (AFTER P1 SUBMITTED)

**Target**: Nature Sustainability / PNAS / Nature Machine Intelligence

**One-sentence pitch**: "Satellite foundation model embeddings reveal a universal visual
signature of conservation-attractive landscape, enabling cross-regional PA prediction
that tabular features cannot achieve."

**Gate**: Requires AlphaEarth (or equivalent) access confirmed + Paper 1 submitted.
Do not start until P1 is submitted.

---

## THINGS DELIBERATELY NOT IN THIS PAPER

- Colombia: supplement only
- Tropical Africa: no data pipeline, not feasible
- Embeddings / Paper 2 model: blocked until P1 submitted
- Prediction-vintage snapshots: rejected
- Bayesian network interpretability layer: dropped
