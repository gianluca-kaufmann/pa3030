# PA3030 — Paper Publication Roadmap

**Purpose**: Authoritative planning document. Updated after every session.
**Target**: GEC / One Earth → Nature Sustainability → JEEM.
**Branch**: `paper` (active development). `main` = intact thesis code, never touched.

---

## Architecture

PA designation is the product of two independent processes:

```
P(pixel i designated in year t)
  = P(country C expands PAs in year t)    ← Stage 1: political timing  [NOT in pixel features]
  × P(pixel i chosen | expansion in C,t)  ← Stage 2: geographic selection  [pixel features]
```

A single classifier conflates both terms. The Group A/B diagnostic proves this empirically:

| Group | Definition | SA AUC | Meaning |
|---|---|---|---|
| A | Designated 2018–2019 (label overlap) | 0.9994 | Memorised |
| B | Designated 2023–2024 (genuinely unseen) | 0.5587 | Near-random — single model fails OOS |

**Stage 1** — Poisson GLM, country-year panel. Target: `pa_expansion_pixels` (km²). Metric: D² OOS (train 2001–2016, test 2017–2024).
**Stage 2** — LightGBM LambdaRank grouped by `(country_id, year)`, trained only on expansion country-years. Within-group percentile rank normalisation. Graded relevance 1–4 by BFS designation event cluster size. Metric: NDCG@1% and Lift@1% within groups.
**Temporal split**: Stage 2: Train 2001–2013 | Early-stop 2014–2016 | Test 2017–2024 | Predict 2024→2029. Stage 1: Train 2001–2016 | Test 2017–2024.
**Forward**: Stage 1 budget (BAU / moderate / 30×30 scenarios) × Stage 2 pixel ranking → top-K designation maps per country.

---

## Key Numbers

| Metric | Value | Notes |
|---|---|---|
| Stage 1 D² — SA (in-sample only) | 0.612 | **OOS needed before citing** (Issue A) |
| Stage 1 D² — momentum-only baseline | 0.415 | Illustrative lower bound |
| Stage 2 tuning NDCG — SA | 0.186 | CV on training data |
| Stage 2 test NDCG@1% — SA | **0.028** | Lift=6.0×; naive lift=0.747× (< 1) — see below |
| Stage 2 naive baseline — SA | NDCG=0.016 / lift=0.747× | dist_wdpa ANTI-predictive in test period |
| Stage 2 test NDCG@1% — SEA (re-tuned, eval_at=[90]) | **0.091** | Lift@1%=12.7×; naive lift=2.7× — beats SA significantly |
| Stage 2 naive baseline — SEA | 0.014 / ~1.5× lift | Full model beats by ~9 pp |
| NDCG@1% target (corrected) | 0.15–0.35 | SEA/SA; see derivation below |
| Lift@1% target | 10–25× | Primary paper metric |

**NDCG@1% derivation**: When n_pos > k@1%, NDCG@1% ≈ precision@1% = binary_positive_rate × lift. With positive rates ~0.8–1% and lift 10–25×, honest expectation is 0.08–0.25.

**SA underperformance — two compounding causes**:
1. **Frontier exhaustion (structural)**: Annual designations collapsed 2001–2009 (150–320K km²/yr) → 2010+ (~20–60K km²/yr) — *before* Bolsonaro. Large easy Amazon blocks were designated first; 2017–2019 pixels are smaller and qualitatively different from training-era positives. Fixed by: `country_pa_cumsum_lag1_pixels` saturation feature (re-run panel merge on Euler) + accept as paper finding.
2. **Bolsonaro structural break hypothesis — FALSIFIED (Issue C)**: 2019 is actually the best test year (Lift=9.64×). 2017 drives the shortfall. Issue D (9K sub-window mismatch) is now the primary suspect.

**Naive baseline finding**: Within expansion groups, `dist_wdpa`-only model achieves lift@1%=0.747 < 1.0 — proximity to existing PAs is ANTI-predictive for 2017–2019. Test-period designations are new areas, not extensions. **Paper finding: two-stage model provides genuine added value over naive heuristic.**

**USA LambdaRank chain cancelled (2026-05-25)**: Jobs 568012/568045 (USA Stage 2 tune/train) cancelled to prioritise SA FE re-run (Issues H+I) and W8 binary. USA Stage 2 will be re-submitted after SA results confirm the approach.

**STAGE2_TRUNCATION_LEVEL override removed (2026-05-25)**: `training_lgbm_stage2.slurm` no longer overrides truncation to 3000. The SA re-tune with [50,3000] will find the true optimum, which will then be read from `best_params.json` by the chained training job.

**Three tuning/training inconsistencies fixed (2026-05-25)**:
1. `neg_ratio` mismatch: training now reads `STAGE2_NEG_RATIO` env var (same as tuning); SLURM training scripts updated with `STAGE2_NEG_RATIO=100`.
2. Year weights absent from Optuna CV: `compute_year_weights` now applied in CV fold datasets in `stage2_optuna_runner.py`.
3. SA re-tune needed: best tuned trunc=499 hit old ceiling; training overrides to 3000. Re-tune SA with `[50,3000]` range after current job chain completes.

**Local work completed (2026-05-25)**:
- **Trend features**: `_ols_slope_vectorized()` + `compute_trend_features()` added to all three `feature_engineering` scripts. Adds `NDVI_b1_trend5`, `deforestation_b1_trend5`, `HNTL_b1_trend5` per pixel. Leak-free (uses data only up to year t). Runs automatically on Euler when `feature_engineering` is re-run.
- **Election cycle**: `years_to_next_election` is now computed **directly from V-Dem CY Core** inside `stage1_panel.compute_years_to_next_election()` — no intermediate CSV. All three `stage1_data_builder.py` files call this function; V-Dem CY Core is the single source of truth. `build_election_cycle.py` is a standalone verification/inspection script (prints election years per country).
- **REDD+**: `redd_plus.csv` generated by `build_redd_plus.py` from named FCPF/UN-REDD enrollment events with source URLs cited per country. Entries marked `# TO VERIFY` need final cross-check against cited FCPF/UN-REDD pages before submission.
- **stage1_data_builder.py for USA + SE Asia**: Created (`scripts/regions/usa/5_training/stage1_data_builder.py` and `scripts/regions/se_asia/5_training/stage1_data_builder.py`). Parallel structure to SA; loads V-Dem/WGI/WDI/election/REDD+.
- **W6 political covariates**: `v2x_corr`, `v2cseeorgs`, `gov_wgi_rl_est`, `years_to_next_election`, `redd_plus_enrolled` added to `POLITICAL_COLS` in `model2_expansion.py` (USA) and `model3_expansion.py` (SE Asia). Columns are optional — silently skipped if absent from panel.
- **W8 binary scripts for USA + SE Asia**: Python (`model2/3_LGBM_stage2_binary`, `model2/3_tune_stage2_binary`) and SLURM (`tuning/training_lgbm_stage2_binary.slurm`) created for both regions.

**LambdaRank scale mismatch** (Issue D): LightGBM enforces a 9K-row per-query limit. SA median group = 413K rows → model optimises "rank within 9K" but is evaluated on "rank within 413K." This partially explains the CV=0.186 → test=0.028 gap. Addressed by W8.

---

## Current Output Status

| Artifact | Status |
|---|---|
| W0 feature guard | ✅ 9 smoke tests pass |
| W3 PA momentum (lags 1–3, all regions) | ✅ Confirmed |
| Stage 1 political data (V-Dem v15, WGI, WDI) | ✅ Downloaded |
| Stage 1 SA | ⚠️ Code complete (OOS split + all W2 political vars + election/REDD+ from official sources). Re-run `stage1_data_builder.py` then `model1_expansion.py` on Euler. |
| Stage 1 USA / SEA | ⚠️ Code complete (stage1_data_builder.py + model2/3_expansion.py with full political vars). Re-run on Euler after SA. |
| Group A/B diagnostic | ✅ SA confirmed (Group B AUC=0.5587) |
| Stage 2 code — all bugs fixed (F1–F9, I1) | ✅ Clean |
| Stage 2 tuning — SA | ✅ 100 trials, NDCG=0.186, trunc=3000 |
| Stage 2 training — SA | ✅ Job 567921. NDCG=0.028, lift=6.0× (see diagnosis above) |
| Stage 2 naive baseline — SA | ✅ NDCG=0.017, lift=2.7× |
| Stage 2 tuning — SEA (re-run, eval_at=[90]) | ✅ Job 567952 done. NDCG=0.091, Lift=12.7× |
| Stage 2 training — SEA (final) | ✅ Job 567979 done |
| Stage 2 tuning — USA | ❌ Job 568012 cancelled (deprioritised) |
| Stage 2 training — USA | ❌ Job 568045 cancelled (deprioritised) |
| Issue C: SA year/country breakdown | 🔄 Job 628878 running |
| Stage 1 SA (data builder + Poisson expansion) | 🔄 Job 628893 queued |
| Stage 1 USA (data builder + Poisson expansion) | 🔄 Job 628895 queued |
| Stage 1 SEA (data builder + Poisson expansion) | 🔄 Job 628897 queued |
| W8: SA binary Stage 2 tune | 🔄 Job 628923 queued |
| W8: SA binary Stage 2 train | 🔄 Job 628926 (afterok:628923) |
| SA feature engineering (Issues H+I: saturation + trend features) | 🔄 Job 628941 queued |
| SA LambdaRank Stage 2 re-tune (corrected [50,3000] range) | 🔄 Job 628943 (afterok:628941) |
| SA LambdaRank Stage 2 retrain (uses re-tuned best_params.json) | 🔄 Job 628945 (afterok:628943) |
| Forward prediction pipeline | ✅ Bugs F1+F2 fixed; probability output is uncalibrated (Issue G) |

**Active chains**: 628923→628926 (W8 binary SA) | 628941→628943→628945 (SA FE → re-tune → retrain). Do not modify Stage 2 architecture or SA panel parquets while these run.

---

## Open Issues

**H — Panel merge re-run required for saturation feature** ← code done; Euler re-run pending
`country_pa_cumsum_lag1_pixels` added to `_build_country_pa_momentum_table` in all three region `feature_engineering` scripts. Re-run `feature_engineering` on Euler to rebuild the panel parquets, then re-train Stage 2. This is the single highest-expected-value new feature.

**I — SA Stage 2 re-tune with corrected range required**
Best tuned `lambdarank_truncation_level=499` hit the old [50,500] ceiling. Current training overrides to 3000 via `STAGE2_TRUNCATION_LEVEL`. Re-run SA Stage 2 tuning (`tuning_lgbm_stage2.slurm`) after job chain completes to find the true optimum in [50,3000] and then **remove** the override from the training SLURM.

**A — Stage 1 D² is in-sample** ← code complete; re-run on Euler to get numbers
OOS split (train 2001–2016, test 2017–2024) + v2x_corr/v2cseeorgs/gov_wgi_rl_est + election_cycle + redd_plus all in code. Requires `stage1_data_builder.py` rebuild then `model1_expansion.py` on Euler. Data files: `data/shared/election_cycle.csv` (V-Dem-derived) and `redd_plus.csv` (FCPF/UN-REDD sourced) ready; REDD+ TO VERIFY entries need final check before paper submission.

**B — `target_30x30` coefficient = 0**
No variation in 2001–2013 training window. Fix: apply 30×30 scenario as exogenous budget multiplier post-prediction, not through the model coefficient.

**C — SA year breakdown ✅ (job 628878, 2026-05-25)**
Result: 2017 Lift=1.47× | 2018 Lift=7.90× | 2019 Lift=9.64× (best!). **Bolsonaro hypothesis falsified** — 2019 is strongest. The 2017 collapse (baseline_rate=0.0017, half of 2018/2019) drags the 3-year average to Lift=6×. Likely cause: fewer designations in 2017 make NDCG highly sensitive to misclassification of rare events. **Implication: Issue D (9K sub-window mismatch) is now primary unexplained residual; W8 binary is the key test.** Country breakdown unavailable (scored parquet lacks country_id; re-run with new code will include it).

**D — LambdaRank 9K sub-window vs. full-group evaluation mismatch**
Training optimises ranking within 9K-row sub-windows; evaluation is over full country-year groups (up to 1.2M rows). Addressed by W8 (binary LightGBM Stage 2). Decision pending cross-region metric comparison after chain completes.

**E — USA Stage 2 timing**
Job 568012/568045 queued (afterok chain). No action needed until they complete.

**F — WDPA label quality: no STATUS or IUCN filter** ← important methodological gap
`1_extraction/WDPA_export` paints ALL WDPA polygons (no `STATUS == 'Designated'` filter, no IUCN category filter). This may include "Proposed" and "Inscribed" areas in the positive labels — potential leakage if those areas are known-pending designations. Fix requires adding `WDPA.filter(ee.Filter.eq('STATUS', 'Designated'))` and optionally `IUCN_CAT` filter to the GEE export, then re-running preprocessing for all regions. **Assess impact first**: estimate what fraction of positive labels come from non-Designated polygons before deciding whether to re-export.

**G — Forward "probability" is not calibrated**
`y_pred_proba_5yr_cumulative` in `two_stage_predict_core.py` is min-max normalised LambdaRank score — not a statistical probability. Cannot be interpreted as P(designated). For investor/risk framing this is misleading. Fix: W8 binary LightGBM produces proper probabilities; apply Platt scaling calibration on top. Until then, label this output as "designation risk index", not "probability."

---

## Workstreams

### W1 — Stage 2 LambdaRank ✅ code complete; SA done; SEA/USA running

After chain completes:
1. Resolve Issue C (year/country NDCG breakdown for SA).
2. Cross-region comparison: if SA uniquely underperforms → Bolsonaro structural break (paper finding). If all regions poor → escalate to W8.
3. Confirm SEA NDCG improved over 0.1125 (52-tree baseline).
4. USA: expect high concordance (adjacency effect — settled finding).

### W2 — New Data [HIGH VALUE — next data sprint]

**Requires GEE export + preprocessing (SA first):**
- **Carbon stocks** (ESA CCI Biomass / GEDI AGBD): REDD+ mechanism makes high-carbon forests profitable to protect. Expected top-5 SHAP.
- **Indigenous territory area fraction** (RAISG): % of pixel covered by indigenous territory. Biggest omitted SA variable. Complements existing `dist_indigenous`.
- **Trend features**: ✅ Code complete in all three `feature_engineering` scripts. Adds `NDVI_b1_trend5`, `deforestation_b1_trend5`, `HNTL_b1_trend5`. **Runs automatically on Euler when `feature_engineering` is re-run for Issues H+I.**

**No GEE needed — ✅ ALL DONE locally:**
- ✅ `v2x_corr`, `v2cseeorgs`, `gov_wgi_rl_est` added to all three expansion scripts
- ✅ Election cycle: computed directly from V-Dem CY Core in `stage1_panel.compute_years_to_next_election()` — no CSV. `build_election_cycle.py` is a read-only inspection script.
- ✅ REDD+ enrollment CSV: generated by `build_redd_plus.py` from named FCPF/UN-REDD sources with per-country URL citations. TO VERIFY entries need final check before submission.
- ✅ `alpha=0.1` already set in all three expansion scripts
- ✅ `stage1_data_builder.py` for USA and SE Asia created
- ✅ `gdp_growth_lag1`: derived in `stage1_data_builder.py` from existing WDI `gdp_per_capita` (1990–2024). No new download. Added to all three expansion models.
- ✅ `pa_cumsum_lag1_pixels`: running total of designated pixels — proxy for frontier saturation. Added to `stage1_panel.build_country_year_panel()` and LAG_COLS in all three expansion models.

**Stage 1 sample size — structural ceiling and options:**
- SA train: 12 countries × 16 years = **192 obs**, 16 features → ~12:1 ratio. Ridge (alpha=0.1) helps.
- SEA train: 11 countries × 16 years = **176 obs** — similar.
- USA train: 1 country × 16 years = **16 obs**. Underdetermined for 16 features. Coefficients are unreliable; treat USA Stage 1 as a time-series trend only, not cross-country evidence.
- **Option — extend to pre-2001**: WDI (1990+) and V-Dem are available. WGI starts 1996 with biennial gaps (1997/1999/2001 missing → forward-fill). Blocker: `build_country_year_panel()` reads from the feature_engineering parquet (starts 2001). Pre-2001 PA expansion counts need reading from WDPA `STATUS_YR` directly — a new function, ~60 extra SA obs. Worth considering for journal revision.

**Medium priority (backlog):**
- Accessibility to cities: Weiss et al. 2018 travel time raster (better than `dist_road` alone)
- Key Biodiversity Areas (KBA) coverage fraction per pixel (BirdLife)

### W3 — PA Momentum ✅ complete

### W4 — Ablation [run after W1 all three regions complete]

`stage2_ablation.py` exists for SA. Run for all three regions. Feature groups: terrain, biodiversity, deforestation, pa_momentum, infrastructure, policy. Critical test: does removing `pa_momentum` collapse to naive-level NDCG? Report NDCG@1% drop per group for Methods table.

### W5 — Logistic Stage 2 Baseline [~1 day local, after W1]

Script exists: `model1_logistic_stage2.py`. Run for SA. Interpretable coefficients anchor SHAP directions for economics reviewers. Report alongside LambdaRank and (if run) binary LightGBM.

### W6 — Stage 1 Full Political Model [code complete; run on Euler]

1. ✅ OOS split in all three expansion scripts.
2. ✅ All W2 political vars (v2x_corr, v2cseeorgs, gov_wgi_rl_est, election_cycle, redd_plus) added to all three scripts.
3. ✅ stage1_data_builder.py for USA + SE Asia created.
4. **On Euler**: `python stage1_data_builder.py && python model1_expansion.py` (SA), then USA, then SEA.
5. Cross-region Stage 1 comparison: are political drivers (VDem, election cycle) consistent across regions?

### W7 — Manuscript [after W1 all regions + Stage 1 OOS confirmed]

**Paper structure:**
1. Intro (~800 w): 30×30 urgency → single-model misspecification (Group A/B) → two-stage solution
2. Results (~2,500 w): Stage 2 SHAP + biodiversity gap + lift@1%; USA adjacency contrast; Stage 1 macro context; forward maps + scenarios; Bolsonaro structural break as political-economy finding
3. Methods (~1,200 w): two-stage decomposition; LambdaRank; NDCG within groups; Poisson GLM; COP15 structural break; forward spatial aggregation; WDPA label limitations
4. Discussion (~1,000 w): political vs. geographic separation; USA path-dependency; BAU extrapolation limits; TNFD/NGFS implications
5. Supplement: feature dictionary; full regional tables; LOBO; cross-continental transfer; hyperparameters; Group A/B diagnostic; dist_wdpa baseline comparison

### W8 — Binary LightGBM Stage 2 ✅ code complete all regions; SA tuning + training pending

`variant="binary"` fully implemented in `stage2_lgbm_core.py`. No neg_ratio (scale_pos_weight handles imbalance). No 9K sub-window constraint. Year weights applied in both training and Optuna CV.

**Scripts ready for all three regions** (SA scripts existed; USA + SE Asia created 2026-05-25):
- SA: `model1_LGBM_stage2_binary`, `model1_tune_stage2_binary`, `slurm/south_america/training/tuning_lgbm_stage2_binary.slurm`
- USA: `model2_LGBM_stage2_binary`, `model2_tune_stage2_binary`, `slurm/usa/training/tuning_lgbm_stage2_binary.slurm`
- SEA: `model3_LGBM_stage2_binary`, `model3_tune_stage2_binary`, `slurm/se_asia/training/tuning_lgbm_stage2_binary.slurm`

**SA run order**: (1) `sbatch tuning_lgbm_stage2_binary.slurm` → (2) `sbatch training_lgbm_stage2_binary.slurm`.

**Decision rule**: if binary Lift@1% > LambdaRank Lift@1% on SA test set → switch primary. Either way, binary produces calibrated forward probabilities (resolves Issue G).

---

## Next Actions (Ordered)

1. ✅ **Issue C** — Done. 2019 is the BEST year (Lift=9.64×); 2017 is the worst (1.47×). Bolsonaro hypothesis falsified. Issue D (9K sub-window) is now primary suspect for SA underperformance.
2. ✅ **Issues H+I** — SA FE job 628941 queued → re-tune 628943 → retrain 628945. Will rebuild panel with `country_pa_cumsum_lag1_pixels` + trend features then find optimal truncation in [50,3000].
3. ✅ **W8 SA binary** — Jobs 628923 → 628926 queued. Compare binary Lift@1% vs LambdaRank 6.0× once done.
4. ✅ **Stage 1 SA/USA/SEA** — Jobs 628893/895/897 queued. Will give OOS D² + full political coefficients.
5. **Cross-region comparison** (after jobs complete): SEA Lift=12.7× vs SA Lift=6.0×. If SA uniquely underperforms after H+I re-train → Bolsonaro structural break (paper finding). Then re-queue USA Stage 2.
6. **REDD+ final verification** (before paper submission, not blocking): Open each `# TO VERIFY` URL in `build_redd_plus.py` (FCPF and UN-REDD country pages) and confirm the enrollment year. Election cycle is fully verified via V-Dem v15 — no manual check needed.
7. **Next data sprint**: GEE export for ESA CCI Biomass (carbon stocks) and RAISG indigenous territory area fraction. SA first.
8. **W5**: Run logistic Stage 2 baseline for SA on Euler (needs rebuilt panel from SA FE, job 628941).
9. **USA Stage 2 LambdaRank**: Re-submit once SA binary and SA LambdaRank re-train results are in hand.
10. **Issue F**: Audit WDPA label quality (estimate % non-Designated in positives); decide whether GEE re-export is warranted.
11. **W7**: Start manuscript after Stage 1 OOS + SA SHAP confirmed + cross-region metrics in hand.

---

## Settled Decisions

- **Architecture**: Two-stage (Stage 1 Poisson + Stage 2 LambdaRank). Binary LightGBM (W8) is a comparison; replaces primary only if it outperforms on Lift@1%.
- **Stage 2 is primary**: Paper leads with Stage 2 SHAP + biodiversity gap + Lift@1%. Stage 1 is macro context regardless of D².
- **Three regions**: SA primary; SE Asia and USA as robustness checks. USA high concordance = adjacency/path-dependency finding, not failure.
- **Graded relevance 1–4**: BFS cluster size. Binary labels superseded.
- **Within-group normalisation**: on for training and inference.
- **COP15 structural break**: Stage 1 extrapolates pre-30×30 patterns. 30×30 scenario = exogenous budget override (Issue B).
- **SA 2019 underperformance**: Bolsonaro hypothesis **falsified** (Issue C). 2019 is best year; 2017 is worst. Issue D (9K sub-window) is primary suspect. W8 binary result will confirm or rule out.
- **Journals**: GEC / One Earth → Nature Sustainability → JEEM.
- **DO NOT** add tropical Africa. **DO NOT** start Paper 2 until Paper 1 submitted.

---

## Out of Scope for Paper 1

- Tropical Africa (no data pipeline)
- Colombia (supplement only if requested by reviewers)
- Embeddings / Paper 2 (gate: AlphaEarth access + P1 submitted)
- Single-model global AUC as primary metric (wrong estimand, proven empirically)
- Sub-national Stage 1 (fallback only if country-level results collapse)
- Marine PAs (no consistent feature coverage)
- DPI political institutions (superseded by V-Dem + WGI)
- ±2 year STATUS_YR sensitivity analysis (mention as Methods limitation; run only if reviewer requests)

---

## Two-Publication Strategy

**Paper 1 (this paper)**: GEC / One Earth → Nature Sustainability → JEEM.
Two-stage decomposition separates geographic suitability from political timing; credible 30×30 forward scenarios; transition risk quantification.

**Paper 2 (after P1 submitted)**: Nature Sustainability / PNAS / NMI.
Foundation model embeddings improve Stage 2 cross-regional transfer.
Gate: AlphaEarth access + P1 submitted.
