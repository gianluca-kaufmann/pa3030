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
**Metric**: NDCG@1% within groups (macro-averaged).

**NDCG@1% target — corrected derivation (2026-05-24):**
When n_pos > k@1% (true for all three regions), NDCG@1% ≈ precision@1% = binary_positive_rate × binary_lift.
With binary positive rates of ~1% (SE Asia), ~0.8% (SA), and lift of 10–25×:
→ SE Asia: NDCG ≈ 0.10–0.25. SA: ≈ 0.08–0.20.
The earlier target of 0.65–0.85 was wrong: it would require lift of 65–85×, far above the
10–35× stated target. **Revised honest expectation: 0.15–0.35** (SE Asia, with the
truncation and eval_at fixes applied). The lift@1% metric remains the primary paper metric
since it is more interpretable and independent of group size.

**Primary metric for the paper**: Lift@1% (binary: precision@k / binary_positive_rate; ~10–25× random).
**Critical baseline (W4)**: dist_wdpa-only lambdarank. Stage 2 must beat by ≥ 5 pp NDCG@1% to claim the full feature set adds signal beyond spatial autocorrelation.

### Forward Prediction (combining stages)

1. Stage 1 → expected expansion budget per country (BAU / moderate / 30×30-compliant scenarios).
2. Stage 2 → pixel-level ranking per country (with within-country rank normalisation at inference).
3. Forward map = top-K pixels from Stage 2 where K = Stage 1 budget.

**Note**: forward inference normalises within country (not country-year, since there is no year at inference time). This is consistent with the training logic — relative position within a country's pixel distribution is the signal, not absolute values.

---

## BUGS FIXED

All four original W1 bugs (wrong objective, wrong truncation range, early-stopping on wrong metric, n_estimators cap) were fixed 2026-05-23. Additional bugs found and fixed 2026-05-24:

**Bug F1 — Forward inference skipped rank normalisation (critical)**
`two_stage_predict_core.py` fed raw feature values to a model trained on within-group percentile ranks. Fixed: data is now loaded fully, then `_rank_normalize_within_groups` is applied per country before `model.predict()`.

**Bug F2 — Stage 1 budget used wrong link function and missing scaler (critical)**
`_predict_country_budgets` computed the linear predictor η instead of exp(η), and passed unscaled features to a model fitted on standardised features. Fixed: `model1/2/3_expansion.py` now saves `scaler_mean` and `scaler_scale` alongside coefficients; `_predict_country_budgets` standardises features then applies exp().

**Bug F3 — Inconsistent neg_ratio across regions (important)**
SA tuning used `STAGE2_NEG_RATIO=20`, USA used 40, SEA defaulted to 100. Fixed: all three tuning SLURM scripts now use `STAGE2_NEG_RATIO=100`.

**Bug F4 — Concordance metric used O(n²) Python loops (moderate)**
Fixed: replaced with O(n log n) `searchsorted` implementation.

**Bug F5 — Fallback truncation level was 5 (minor)**
Fixed to use `FIXED_PARAMS` default (100).

Additional bugs and improvements applied 2026-05-25:

**Improvement I1 — Random within-group shuffle before sub-window splitting (major, structural)**
Training panels are spatially sorted (row, col order), so LightGBM's consecutive 9K sub-windows
contained spatially homogeneous patches. LambdaRank gradients compared only locally-similar pixels,
teaching "rank vs 9K neighbours" rather than "rank vs all 500K country-year pixels".
Fix: `_shuffle_within_groups` called in `load_stage2_arrays` after rank normalisation (and before
lgb.Dataset construction). Each 9K window is now a random sample of the full country-year
distribution, aligning training gradients with the test evaluation task. Column-wise X shuffle
keeps peak memory O(n × 4B) per column, safe for USA-scale groups (~50M rows/country-year).
Applied to both training and tuning (tuning also calls `load_stage2_arrays`).

**Bug F9 — Stage 2 temporal weighting always uniform (important)**
`run_stage2_training` passed `np.full(len(y_tr), cfg.train_years[1])` (constant year 2013) to
`compute_year_weights`, making all weights 1.0. The actual `years_tr` were discarded with `_`.
Fixed: capture `years_tr` and pass it to `compute_year_weights(years_tr, min_year=..., max_year=...)`.
Samples from 2001 now receive weight 0.5; samples from 2013 receive weight 1.0, focusing gradient
on patterns most relevant to the 2017–2019 test period.

**Memory fix applied 2026-05-25** (see W1 section above for full description).

Additional bugs found and fixed 2026-05-24 (second pass):

**Bug F6 — eval_at not set → early stopping monitored NDCG@1 (single position) (important)**
LightGBM default `eval_at=[1,2,3,4,5]` means early stopping fires on whether the #1 ranked
pixel in each 9 000-row split sub-group is a transition — a binary, very noisy signal.
Groups are split to ≤9 000 rows by `_split_large_groups`; k@1% of 9 000 = 90.
Fixed: added `eval_at=[90]` to `FIXED_PARAMS` in `stage2_lgbm_core.py` and to
`get_lgbm_stage2_fixed_params` in `search_spaces.py`. Early stopping now monitors ndcg@90,
matching the training objective scale.

**Bug F7 — Lift metric used graded mean as denominator instead of binary positive rate (important)**
`lift_at_k_within_groups` divided precision@k by `y_true.mean()` (mean of graded labels 1–4),
not by the binary positive rate (fraction of pixels with any transition). Since positives have
average relevance > 1, the denominator was inflated, making the naive baseline appear
sub-random (0.94×) when it is actually ~1.5× binary lift. The full model reported 5× when
true binary lift ≈ 8.7× for SE Asia. Fixed: `stage2_metrics.py` now uses `(y_true > 0).mean()`
for both lift and `baseline_rate` in the reported metrics dict.

**Bug F8 — Truncation search range ceiling at 500 was hit by SA tuning (important)**
SA tuning found `truncation_level=499` (the upper bound of [50, 500]), signalling that higher
is better. USA tuning has not yet run and would also hit this ceiling. Fixed: upper bound
raised to 3 000 in `get_lgbm_stage2_optuna_bounds`. Both SA and SE Asia training SLURMs now
set `STAGE2_TRUNCATION_LEVEL=3000` (overrides the tuned value for the first training run;
re-tune after if results disappoint).

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

### W1 — Stage 2 LambdaRank ✅ code complete; SA training pending re-run

All bugs fixed (original 4 from 2026-05-23 + F1–F8 from 2026-05-24). Code is clean.

**Memory fix applied 2026-05-25**: `load_stage2_arrays` in `stage2_lgbm_core.py` now separates
metadata (5 cols, ~4 GB) from feature arrays (float32 numpy) during batch streaming. This cuts
peak memory from ~190 GB (old: pd.concat of full DataFrames) to ~120 GB (new: 2× feature array
during np.concatenate and sort). SA training previously OOM-killed at 128 GB; this fix should allow
it to run within 128 GB.

**SEA tuning–training inconsistency found 2026-05-25**: SEA tuning ran 2026-05-23 (before Bug F6
fix), using default `eval_at=[1,2,3,4,5]`, and found `n_estimators=1397`. Full training uses
`eval_at=[90]` (from FIXED_PARAMS post-F6), which peaked at iteration 52 and fired early stopping.
Result: SEA re-run (565173) trained only 52 trees → NDCG@1%=0.1125, well below the ~0.20+ that
1397 trees might achieve. SA tuning ran after Bug F6 fix with `eval_at=[90]`, finding n_estimators=955;
SA training will be consistent. SEA must be re-tuned before re-training.

SEA re-run (565173): NDCG@1%=0.1125, lift@1%=10.1×, concordance=0.789 (52 trees only — under-trained).
SA training (567921): ✅ DONE. NDCG@1%=0.028, lift@1%=6.0×, concordance=0.643. Beats naive (0.017) by only 1.1 pp — below 5 pp target. Large train/test gap vs CV NDCG=0.186 warrants investigation.
SA naive baseline: NDCG@1%=0.017, lift@1%=2.7×.

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

**C — SA tuning with neg_ratio=20** ✅ resolved
SA tuning re-ran with neg_ratio=100 and completed 100 trials (timestamp 20260524_202135).
Best truncation=499 hit the old ceiling of 500 → confirms Bug F8. Training now uses
`STAGE2_TRUNCATION_LEVEL=3000` override.

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
| Stage 2 tuning NDCG (SA, neg_ratio=100) | 0.186 | CV metric on training data; eval_at=[90] ✓ |
| Stage 2 tuning NDCG (SEA, neg_ratio=100) | 0.126 | CV metric — pre-F6 fix, eval_at=[1-5]; re-tune needed |
| Stage 2 SEA test NDCG@1% (first run, trunc=285) | 0.106 | Binary lift ≈ 8.7× |
| Stage 2 SEA test NDCG@1% (re-run, trunc=3000, eval_at=[90]) | 0.1125 | Lift=10.1×, concordance=0.789 — **52 trees only** |
| Stage 2 naïve baseline SEA (dist_wdpa only) | 0.015 NDCG / 2.9× lift | Beaten by ≥9 pp ✓ |
| Stage 2 naïve baseline SA (dist_wdpa only) | 0.017 NDCG / 2.7× lift | Full model beats by only 1.1 pp — below 5 pp target |
| Stage 2 SA test NDCG@1% (final, all fixes) | **0.028** | Lift=6.0×, concordance=0.643 — below 10× target; large train/test gap vs CV NDCG=0.186 |
| Stage 2 NDCG@1% target (test, corrected) | 0.15–0.35 | SE Asia; see derivation in Stage 2 section |
| Stage 2 Lift@1% target (binary) | 10–25× | Primary paper metric |

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
| Stage 2 code (all regions) | ✅ clean | All bugs F1–F8 fixed as of 2026-05-24 |
| Stage 2 tuning — SA | ✅ complete | 100 trials, best NDCG=0.186, trunc=499 (old ceiling) |
| Stage 2 tuning — SEA | ✅ complete | 100 trials, best NDCG=0.126, trunc=285 |
| Stage 2 tuning — USA | ❌ pending | Submit now (SEA training done); range now [50,3000] |
| Stage 2 training — SEA (first run) | ✅ done | NDCG=0.106, lift≈8.7×; trunc=285, no eval_at fix |
| Stage 2 training — SA | ✅ done (job 567921) | NDCG=0.028, lift=6.0×, concordance=0.643 — below target; beats naive by 1.1 pp only |
| Stage 2 training — SEA (re-run trunc=3000) | ✅ done (job 565173) | NDCG=0.1125, lift=10.1× — under-trained (52 trees); re-tune queued |
| Stage 2 tuning — SEA (re-run with eval_at=[90]) | 🔄 job 567952 RUNNING | 6h26m of 24h; shuffle + F9 fix applied |
| Stage 2 training — SEA (final) | 🔄 job 567979 → afterok:567952 | Will use new n_estimators from re-tune |
| Stage 2 tuning — USA | 🔄 job 568012 → afterok:567979 | Range [50,3000], all fixes applied |
| Stage 2 training — USA | 🔄 job 568045 → afterok:568012 | trunc=3000 + all fixes |
| Naïve baseline SEA | ✅ done | NDCG=0.014, lift≈1.5× binary (full model beats by 9 pp) |
| Naïve baseline SA | ✅ done | NDCG=0.017, lift=2.7× |
| Two-stage forward predict | ✅ code fixed | Bugs F1+F2 resolved 2026-05-24 |

---

## WHAT TO DO NEXT (ordered)

**Active job chain (submitted 2026-05-25):**
- 567921: SA training — ✅ DONE (NDCG=0.028, lift=6.0×)
- 567952: SEA tuning — RUNNING (6h26m of 24h)
- 567979: SEA training — PENDING (afterok:567952)
- 568012: USA tuning — PENDING (afterok:567979)
- 568045: USA training — PENDING (afterok:568012)

1. **Investigate SA underperformance**: lift=6× vs 10× target; beats naive by only 1.1 pp NDCG.
   CV NDCG=0.186 → test=0.028 is a large drop. Candidates: test-period distribution shift (2017–2019
   has more large PA events?), group size imbalance (26 groups, some very large), or grouping noise.
   Defer ablation until SEA/USA results are in — if SA is a systematic outlier, structural cause
   is more likely than a tuning issue.

2. **After SEA training (567979) finishes**: confirm NDCG@1% improved over 0.1125 (52-tree result).

3. **After USA training (568045) finishes**: check for the USA "near-perfect concordance" finding
   (adjacency effect — described in SETTLED DECISIONS).

4. **After all three regions**: compare metrics table, confirm SEA and USA beat naive by ≥ 5 pp.
   If SA remains the outlier, investigate group structure differences.

5. **Run Stage 1 OOS evaluation** locally: modify `model1_expansion.py` to split train ≤ 2013 /
   test 2014–2019 before citing D² in the paper. (Open Issue A.)

6. **Run `model2/3_expansion.py`** for USA and SEA Stage 1. (Open Issue D.)

7. **Full ablation (W4)** after all three regions have final training results.

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
