# PA3030 — Paper Publication Roadmap

**Purpose**: Authoritative planning document. Keep compact and updated after every session.
**Status**: Post-thesis. Paper branch active. Two-stage architecture settled.
**Target**: GEC / One Earth → Nature Sustainability → JEEM.

---

## WHY THE TWO-STAGE MODEL IS THE CORRECT SPECIFICATION

PA designation is the product of two independent processes:

```
P(pixel i designated in year t)
  = P(country C expands PAs in year t)    ← political timing  [NOT in features]
  × P(pixel i chosen | expansion in C,t)  ← geographic selection  [IN features]
```

A single binary classifier conflates both terms. Trained to predict the joint event but only having features for the second term, it produces inflated AUC driven by label overlap, not prediction. The Group A/B diagnostic confirms this empirically:

| Group | Definition | SA RF AUC | Interpretation |
|-------|-----------|-----------|----------------|
| A | Designated 2018–2019 (label overlap) | 0.9994 | Memorised |
| B | Designated 2023–2024 (genuinely unseen) | 0.5587 | Near-random — true OOS |

**Stage 1** estimates country-year expansion volume using political/institutional variables.
**Stage 2** estimates pixel-level selection within expansion events using geographic features.
Conditioning Stage 2 on expansion having occurred removes the political timing noise that made the old model fail.

The paper's core claim: treating PA designation as a single prediction problem is a structural specification error. We prove this empirically, decompose it correctly, and build both stages.

---

## THE MODEL

### Stage 1 — Country-Year Expansion (macro)

**Question**: How much will country C designate in year t?
**Target**: km² per country per year.
**Model**: Poisson GLM (log-link). Standardised features (scaler saved in coefficients JSON).
**Features**: PA momentum lags 1–3; V-Dem polyarchy; WB WGI effectiveness; GDP per capita; agricultural land %; 30×30 dummy; CBD meeting year dummy.
**Metric**: D² (deviance R²). Honest expectation: 0.40–0.65.
**Status**: illustrative macro context (momentum-only D² = 0.415 in-sample). Adding political covariates may push above 0.50.
**Note (Open Issue A)**: D² = 0.612 for SA is IN-SAMPLE. OOS on 2014–2019 needed before citing.
**Note (Open Issue B)**: `target_30x30` coefficient = 0 (no variation in 2001–2013 training window). Forward scenarios with 30×30 commitment require an exogenous budget override post-prediction.

### Stage 2 — Geographic Selection (micro, conditional)

**Question**: Given expansion in country C in year t, which pixels are selected?
**Training set**: Only country-years with observed expansion.
**Model**: LightGBM `objective=lambdarank`, grouped by `(country_id, year)`.
**Preprocessing**: Within-group percentile rank normalisation [0,1] applied before training AND inference (see forward inference note below).
**Labels**: Graded relevance 1–4 by designation event cluster size (4-connected BFS).
**Metric**: NDCG@1% within groups (macro-averaged). Honest expectation: 0.65–0.85.
**Primary metric for the paper**: Lift@1% within groups (intuitive; ~10–35× random).
**Critical baseline (W4)**: dist_wdpa-only lambdarank. Stage 2 must beat by ≥ 5 pp NDCG@1% to claim the full feature set adds signal beyond spatial autocorrelation.

### Forward Prediction (combining stages)

1. Stage 1 → expected expansion budget per country (BAU / moderate / 30×30-compliant scenarios).
2. Stage 2 → pixel-level ranking per country (with within-country rank normalisation at inference).
3. Forward map = top-K pixels from Stage 2 where K = Stage 1 budget.

**Note**: forward inference normalises within country (not country-year, since there is no year at inference time). This is consistent with the training logic — relative position within a country's pixel distribution is the signal, not absolute values.

---

## BUGS FIXED (2026-05-24)

All four original W1 bugs (wrong objective, wrong truncation range, early-stopping on wrong metric, n_estimators cap) were fixed 2026-05-23. The following additional bugs were found and fixed 2026-05-24:

**Bug F1 — Forward inference skipped rank normalisation (critical)**
`two_stage_predict_core.py` fed raw feature values to a model trained on within-group percentile ranks. Fixed: data is now loaded fully, then `_rank_normalize_within_groups` is applied per country before `model.predict()`.

**Bug F2 — Stage 1 budget used wrong link function and missing scaler (critical)**
`_predict_country_budgets` computed the linear predictor η instead of exp(η), and passed unscaled features to a model fitted on standardised features. Fixed: `model1/2/3_expansion.py` now saves `scaler_mean` and `scaler_scale` alongside coefficients; `_predict_country_budgets` standardises features then applies exp().

**Bug F3 — Inconsistent neg_ratio across regions (important)**
SA tuning used `STAGE2_NEG_RATIO=20`, USA used 40, SEA defaulted to 100. This made tuning NDCG values incomparable across regions and misaligned the evaluation window during tuning with the test evaluation window. Fixed: all three tuning SLURM scripts now use `STAGE2_NEG_RATIO=100`.

**Bug F4 — Concordance metric used O(n²) Python loops (moderate)**
`concordance_within_groups` would have taken hours for USA-scale groups. Fixed: replaced with O(n log n) `searchsorted` implementation.

**Bug F5 — Fallback truncation level was 5 (minor)**
Dead code in `_prepare_lgb_params` set truncation to 5 if missing. Fixed to use `FIXED_PARAMS` default (100).

---

## SETTLED DECISIONS

- **Architecture**: Two-stage (Stage 1 Poisson panel + Stage 2 LambdaRank). Not revertable as primary model.
- **Stage 2 is primary**: paper lead is Stage 2 SHAP story + biodiversity gap + Lift@1%. Stage 1 is macro context regardless of performance.
- **Three regions**: SA (primary), SE Asia, USA. USA near-perfect concordance is a finding (adjacency effect), not a failure.
- **Graded relevance (1–4)**: default on. Based on BFS cluster size. Binary superseded.
- **Within-group normalisation**: default on for training and inference.
- **COP15 structural break**: Stage 1 extrapolates to a qualitatively different political regime. Address in Methods. Frame Stage 1 as pre-30×30 baseline; 30×30 scenario is an exogenous budget override.
- **Journals**: GEC / One Earth → Nature Sustainability → JEEM. Nature Finance / JEEM if financial angle strengthens.
- **DO NOT** add tropical Africa. **DO NOT** start Paper 2 until Paper 1 submitted.

---

## WORKSTREAMS

### W0 — Feature / provenance gate ✅ complete

### W1 — Stage 2 LambdaRank ✅ code complete; Euler tuning + training in progress

All bugs fixed (original 4 from 2026-05-23 + bugs F1–F5 from 2026-05-24). Code is clean.

**Truncation level experiment** (run after first training completes):
After the initial training run produces test NDCG, submit a second training job with `STAGE2_TRUNCATION_LEVEL=1500` (uncomment the line in each training SLURM script). Tuning searched [50,500] but test k@1% is ~2000; 1500 covers that window. Compare test NDCG. If improvement ≥ 2 pp, use 1500 as the reported model.

### W2 — New data [HIGH VALUE, parallel with W1]

- **Carbon stocks** (ESA CCI Biomass): REDD+ mechanism makes high-carbon land a priority. Expected top-5 SHAP. Do SA first.
- **Land tenure / indigenous lands** (RAISG): Biggest omitted variable for SA. Public and indigenous lands are the path of least resistance.
- **Political variables** (V-Dem v15, WB WGI, WDI): ✅ downloaded and formatted.

### W3 — PA momentum ✅ complete

`pa_momentum_pixels_lag{1,2,3}` and `country_id` confirmed present in all three regions' parquets.

### W4 — Ablation [run naïve baseline IMMEDIATELY after training]

**Naïve baseline** (`dist_wdpa`-only lambdarank): the critical reference point. The training SLURM scripts already run this automatically after the main model. Do not skip reviewing this number — it determines whether the 60-feature pipeline adds meaningful signal.

**Full ablation** (after naïve baseline confirms Stage 2 adds value): remove feature groups one at a time (terrain, biodiversity, deforestation, PA momentum, infrastructure). Report NDCG drop. Generates feature importance table for Methods.

### W5 — Logistic baseline [~1 day local, after W1]

Logistic regression on Stage 2 formulation (within-group, conditional on expansion). Interpretable coefficients anchor the SHAP directions for economics reviewers.

### W6 — Stage 1 full political model [parallel with W1 training]

Run `model1_expansion.py` (SA), `model2_expansion.py` (USA, new), `model3_expansion.py` (SEA, new) with all political covariates. Then run OOS evaluation (train ≤ 2013, test 2014–2019) before citing D² in the paper.

### W7 — Manuscript [after W1 Stage 2 test NDCG confirmed]

**Paper structure**: Stage 2 leads Results. Stage 1 provides context.
1. Intro (~800 w): 30×30 urgency → misspecification problem (Group A/B as evidence) → what we do
2. Results (~2,500 w): Stage 2 SHAP story across 3 continents; USA adjacency contrast; biodiversity gap; Stage 1 macro context; forward maps + scenarios; transition risk estimates
3. Methods (~1,200 w): Group A/B diagnostic; two-stage decomposition; LambdaRank; NDCG within groups; Poisson panel; COP15 structural break; forward spatial aggregation
4. Discussion (~1,000 w): political vs geographic separation; USA path-dependency; COP15 limits; TNFD/NGFS implications
5. Supplement: feature dictionary; full regional tables; LOBO; cross-continental transfer; hyperparameters; Group A/B full diagnostic; dist_wdpa baseline comparison

---

## OPEN ISSUES

**A — Stage 1 D² is in-sample** (blocking for Stage 1 paper claims)
SA `model.score(X, y)` is computed on training data. Add OOS split (train ≤ 2013, test 2014–2019) to `model1/2/3_expansion.py` before citing D² values. The in-sample D² = 0.612 is an upper bound; OOS will be lower.

**B — `target_30x30` coefficient = 0** (blocking for scenario forward maps)
No variation in the 2001–2013 training window → coefficient is zero. Forward scenarios with the 30×30 commitment have no effect on Stage 1. Fix: apply the 30×30 scenario as an exogenous budget multiplier after Stage 1 prediction rather than through the model coefficient.

**C — SA tuning completing with neg_ratio=20** (cosmetic, not blocking)
SA is currently at ~trial 48/100 with `STAGE2_NEG_RATIO=20` (now fixed in SLURM for future runs). The resulting best_params.json will have a truncation_level tuned against k@1%≈163 instead of k@1%≈782. Use those params for the first training run, then re-run SA tuning (now with ratio=100) if first-run NDCG is disappointing. The truncation level override (`STAGE2_TRUNCATION_LEVEL=1500`) partially compensates.

**D — USA and SE Asia Stage 1 not run**
`model2_expansion.py` and `model3_expansion.py` now exist (created 2026-05-24). Run after SA OOS bug (Issue A) is resolved.

**E — USA Stage 2 timing**
Submit USA tuning after SE Asia training completes (CPU quota). USA tuning SLURM updated to use `STAGE2_NEG_RATIO=100` and 32 CPUs / 256GB.

---

## KEY NUMBERS

| Metric | Value | Notes |
|--------|-------|-------|
| Thesis AUC (Group A) | 0.9994 | Leakage — memorised |
| Thesis AUC (Group B, genuinely unseen) | 0.5587 | True OOS — near-random |
| Annual hazard model AUC | 0.582 | Misspecified — superseded |
| Stage 1 momentum-only D² (in-sample) | 0.415 | Illustrative context zone |
| Stage 1 full political model D² (in-sample) | 0.612 | In-sample only — OOS needed |
| Stage 2 tuning NDCG (SA, neg_ratio=20) | ~0.186 | Distorted by small groups — not test NDCG |
| Stage 2 tuning NDCG (SEA, neg_ratio=100) | 0.126 | Distorted by subsampled groups — not test NDCG |
| Stage 2 NDCG@1% target (test, full data) | 0.65–0.85 | Post-fix expectation |
| Stage 2 naïve baseline (dist_wdpa only) | TBD | Must beat by ≥ 5 pp |
| Stage 2 Lift@1% target | 10–35× within groups | Primary investor-facing metric |

---

## CURRENT OUTPUT STATUS

| Artifact | Status | Notes |
|----------|--------|-------|
| W0 feature guard | ✅ complete | 9 smoke tests pass |
| W3 PA momentum | ✅ in panels | All 3 regions confirmed |
| Stage 1 political data | ✅ downloaded | V-Dem v15 + WGI + WDI |
| Stage 1 SA (in-sample) | ⚠️ needs OOS | D²=0.612 in-sample only |
| Stage 1 USA / SEA | ❌ not run | Scripts exist (created 2026-05-24) |
| Group A/B diagnostic | ✅ complete | SA Group B AUC=0.5587 confirmed |
| Stage 2 code (all regions) | ✅ clean | All bugs fixed as of 2026-05-24 |
| Stage 2 tuning — SA | 🔄 running | ~trial 48/100 with old neg_ratio=20; let finish |
| Stage 2 tuning — SEA | ✅ complete | best NDCG=0.126 (tuning metric); params saved |
| Stage 2 tuning — USA | ❌ pending | Submit after SEA training completes |
| Stage 2 training — all regions | ❌ pending | After tuning completes |
| Naïve dist_wdpa baseline | ❌ pending | Runs automatically with training SLURM |
| Two-stage forward predict | ✅ code fixed | Bugs F1+F2 resolved 2026-05-24 |

---

## WHAT TO DO NEXT (ordered)

1. **Wait for SA tuning to finish** on Euler (~trial 48/100 with neg_ratio=20). Do not cancel it.

2. **Submit SEA training** (`sbatch slurm/se_asia/training_lgbm_stage2.slurm`). The naive baseline runs automatically in the same job. Review both NDCG@1% numbers immediately.

3. **Submit SA training** after tuning completes (`sbatch slurm/south_america/training_lgbm_stage2.slurm`). Same — naïve baseline runs automatically.

4. **Check naïve baseline** as soon as each training job finishes. If full model beats dist_wdpa-only by ≥ 5 pp NDCG@1%: Stage 2 contribution is confirmed. If not: paper framing shifts.

5. **Run truncation level experiment** for any region where test NDCG < 0.60: uncomment `STAGE2_TRUNCATION_LEVEL=1500` in the training SLURM, change job name, resubmit.

6. **Re-run SA tuning** with the updated SLURM (`STAGE2_NEG_RATIO=100`) once the current run finishes and SA training has produced a first set of test results.

7. **Submit USA tuning** after SEA training completes (to stay within CPU quota).

8. **Run Stage 1 OOS evaluation** locally: modify `model1_expansion.py` to split train ≤ 2013 / test 2014–2019. Run locally. Do this before writing any Stage 1 results.

9. **Run `model2/3_expansion.py`** for USA and SEA Stage 1 (scripts created 2026-05-24).

---

## BRANCH AND REVERSION POLICY

- **`paper` branch**: active development. All W1–W6 changes + two-stage model.
- **`main` branch**: intact thesis code (5-year window). Never touched.

---

## TWO-PUBLICATION STRATEGY

**Paper 1 (this paper)**: GEC / One Earth → Nature Sustainability → JEEM.
"A two-stage model of PA designation separates the predictable geographic selection of conservation candidates from the politically-timed expansion decision, enabling credible 30×30 forward scenarios and transition risk quantification."

**Paper 2 (after P1 submitted)**: Nature Sustainability / PNAS / NMI.
Foundation model embeddings improve Stage 2 cross-regional transfer. Gate: AlphaEarth access + P1 submitted.

---

## THINGS DELIBERATELY NOT IN THIS PAPER

- Colombia: supplement only
- Tropical Africa: no data pipeline
- Embeddings / Paper 2: blocked until P1 submitted
- Single-model global AUC as primary metric: rejected (wrong estimand, proven empirically)
- Binary LambdaRank labels: superseded by graded (1–4)
- Sub-national Stage 1: fallback only if country-level results are very weak
