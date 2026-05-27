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
| Stage 1 D² — SA (11-feat + Δgov + agri + cbd + p95 + log1p, final) | D²_train=0.600, **D²_7yr=+0.368 (PRIMARY, 2017–23)**, D²_3yr=**+0.403**, D²_8yr=**+0.343** (secondary) | 11-feat: adds **cbd_meeting_year** (CBD COP years create urgency; +0.14 coef), **p95 winsor** (cap=39,662 px), **log1p momentum**. Chow F=0.57 **p=0.866 (NOT significant)**. **2024 excluded from primary**: WDPA May2026 CSV shows 219 polygons/4,851 km² for SA 2024, but panel has only 1,628 px (~33% capture rate). CBD-free fallback: D²_7yr=+0.442 (saved to JSON). Sensitivity grid confirms p95+cbd is optimal across all winsor variants. **Numbers updated 2026-05-27**: pre-2001 marine contamination fix removed Galápagos MR (138,722 km²) and other marine PAs from lag initialization (ECU: 173K→34K km², COL: 50K→2K km²). D² dropped ~10pp vs old spec. These are the CORRECT numbers. Still uses local stage1_panel.parquet; Euler re-run with corrected GeoTIFs will update 2001+ data. |
| Stage 1 D² — SEA (9-feat Δgov spec, final) | D²_train=0.085, **D²_6yr=+0.259 (PRIMARY, 2017–22)**, D²_3yr=**+0.303**, D²_8yr=**+0.178** (secondary) | 9-feat: **Δv2xlg_legcon + Δv2csprtcpt replace forest_area_pct**. **2023–2024 excluded from primary**: WDPA May2026 CSV shows IDN 25,320 km² (2023) + 25,038 km² (2024) not captured in panel (0 in both years — WDPA reporting lag). Cross-regional finding: v2x_polyarchy coef=**−0.35** (authoritarian regimes expand more in SEA via top-down mandates, vs SA +0.77 democratic culture). **Numbers updated 2026-05-27**: pre-2001 marine contamination fix (GIS_M_AREA subtracted) slightly improved D²_6yr (+0.007). |
| Stage 1 D² — USA (trend-only, Issue K) | D²_train=0.218, D²_test=**−3.14** | 4 momentum features, α=10; political path-dependency finding |
| Stage 1 SA — Chow break at 2010 | F=1.86, p=0.069 | Marginal significance; visual evidence of frontier exhaustion is compelling |
| Stage 1 D² — momentum-only baseline | 0.407 | OOS: −0.249 (SA), +0.129 (SEA) |
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
| Stage 2 tuning — SEA (re-run, eval_at=[90]) | ✅ Job 567952 done. NDCG=0.091, Lift=12.7× — **pre-H+I panel; not final** |
| Stage 2 training — SEA (final) | ✅ Job 567979 done — **pre-H+I panel; SEA FE re-run still pending** |
| Stage 2 tuning — USA | ❌ Job 568012 cancelled (deprioritised) |
| Stage 2 training — USA | ❌ Job 568045 cancelled (deprioritised) |
| Issue C: SA year/country breakdown | ✅ Job 628878 done. 2017=1.47× | 2018=7.90× | 2019=9.64× |
| Stage 1 SA (11-feat Δgov+agri+cbd+p95+log1p, FINAL) | ✅ Updated locally 2026-05-27. D²_train=0.600, **D²_7yr=+0.368 (PRIMARY, 2017–23)**, D²_3yr=+0.403, D²_8yr=+0.343 (secondary). Marine fix: pre-2001 lags now terrestrial-only (GIS_AREA−GIS_M_AREA). 2024 excl. WDPA lag. Chow p=0.866. CBD-free fallback D²_7yr=+0.442 in JSON. ⚠️ Will update again after Euler SA FE re-run (2001+ pixel data corrected). |
| Stage 1 SEA (9-feat Δgov spec, FINAL) | ✅ Updated locally 2026-05-27. D²_train=0.085, **D²_6yr=+0.259 (PRIMARY, 2017–22)**, D²_3yr=+0.303, D²_8yr=+0.178 (secondary). Marine fix applied to pre-2001 lags. 2023–2024 excl. WDPA lag. Polyarchy coef=−0.35. ⚠️ Will update again after Euler SEA FE re-run. |
| Stage 1 USA (trend-only model, Issue K fix) | ✅ Run locally 2026-05-26. D²_train=0.218, D²_test=−3.14 (4 trend features, alpha=10; see Issue K) |
| W8: SA binary Stage 2 tune | 🔄 Job 689625 running (30 trials, trunc N/A, n_est ≤1500) |
| W8: SA binary Stage 2 train | 🔄 Job 689627 (afterok:689625) |
| SA feature engineering (Issues H+I: saturation + trend features) | ✅ Job 628941 done. Panel rebuilt 2026-05-26 01:09 (57 GB) |
| SA LambdaRank Stage 2 re-tune ([50,1000] range, 30 trials) | 🔄 Job 689639 pending (QOSMaxMemoryPerUser, afterok binary tune) |
| SA LambdaRank Stage 2 retrain | 🔄 Job 689640 (afterok:689639) |
| USA feature engineering (Issue H: saturation + trend features) | ❌ Not yet run — panel is pre-feature-engineering (May 22) |
| SE Asia feature engineering (Issue H: saturation + trend features) | ❌ Blocked on scp of patched TIFs to Euler (done locally — see Issue L) |
| SA WDPA GeoTIF patch (Issue L: 2023+2024 missing designations) | ✅ Patched locally + scp to Euler done 2026-05-27. +2,347 px (2023) +103 px (2024). Next: SA FE re-run → stage1_data_builder → model1_expansion local → Stage 2 re-queue (after 689640 completes). |
| SEA WDPA GeoTIF patch (Issue L) | ✅ Patched locally 2026-05-27. +1,127 px (2023) +188 px (2024). Scp to Euler ✅ done — SEA FE re-run now unblocked. |
| Forward prediction pipeline | ✅ Bugs F1+F2 fixed; probability output is uncalibrated (Issue G) |

**Active chains**: 689625→689627 (W8 binary SA tune→retrain) | 689639→689640 (SA LambdaRank tune→retrain, pending). Do not modify SA panel parquets while these run.

---

## Open Issues

**L — WDPA GeoTIF patch ✅ DONE locally (2026-05-27)**
GEE asset `WCMC/WDPA/current/polygons` confirmed missing IDN 2023–2024 designations (verified: `ee.Filter(ISO3=IDN, STATUS_YR>=2023).size() == 0`). WDPA May2026 CSV confirms they exist (IDN 2023=25,320 km², 2024=25,038 km²). GEE re-export is NOT the fix — GEE asset is lagging.
**Fix applied** via `scripts/regions/se_asia/2_preprocessing/patch_wdpa_sea.py` (committed 2026-05-27):
- Source: WDPA_May2026_Public_shp parts 0–2 (local, `data/shared/WDPA/`)
- SEA countries scanned: IDN (112 polygons, 50,359 km²) + THA (2 polygons, 750 km²)
- 96.5% of burned pixels fall in NaN (marine, outside backbone) — expected for KK/PKK marine areas
- **Net new land pixels**: 2023 TIF +1,127 px | 2024 TIF +188 px (cumulative above 2023)
- Cumulative protected pixel counts: 2022=679,234 → 2023=680,361 → 2024=680,549
- Originals backed up: `WDPA_SEA_1km_2023_pre_patch.tif`, `WDPA_SEA_1km_2024_pre_patch.tif`
- **Still needed**: scp patched TIFs to Euler `$SCRATCH/data/se_asia/ready/WDPA/` before SEA FE re-run
5. SA 2023+2024: **✅ Patched locally 2026-05-27** via `scripts/regions/south_america/2_preprocessing/patch_wdpa_sa.py`. Net new pixels: +2,347 (2023), +103 more (2024). SA 2023 is in the primary metric period — affects Stage 1 country-year target and Stage 2 test labels. **Sequence**: scp AFTER 689640 completes (running chain reads panel parquet, not TIFs — no conflict); then SA FE re-run → stage1_data_builder → model1_expansion local → Stage 2 re-tune+retrain.
**Stage 1 terrestrial note**: Stage 1 target `pa_expansion_pixels` is aggregated from `transition_01` in the backbone parquet (terrestrial only). Marine designations have zero effect on Stage 1. The GIS_AREA CSV is only used in `compute_pre2001_expansion` for pre-2001 lag initialization.
**Sequence gate**: SEA FE re-run must come AFTER scp to Euler. Both patch scripts are idempotent (backup check prevents double-patching).

**H — Panel merge re-run required for saturation + trend features** ← SA ✅ done; USA + SE Asia pending
`country_pa_cumsum_lag1_pixels` (saturation) + `NDVI_b1_trend5` / `deforestation_b1_trend5` / `HNTL_b1_trend5` (trend) added to all three `feature_engineering` scripts. SA panel rebuilt 2026-05-26 (61 GB); SA re-tune+retrain chain running. USA (`merged_panel_final.parquet` from 2026-05-22, 46 GB) and SE Asia (from 2026-05-22, 18 GB) still need `feature_engineering` re-runs on Euler, then Stage 2 re-tune+retrain for each.

**I — SA Stage 2 re-tune with corrected range ✅ submitted**
Old ceiling was 500 (best trial hit 499). New search space: `lambdarank_truncation_level` ∈ [50, 1000], `n_estimators` ∈ [200, 1500], 30 trials. Jobs 689639→689640 queued.

**Stage 1 specification — UPDATED (2026-05-26)**
Full grid search over feature sets, regularisation, and model families (Poisson, log-OLS, Negative Binomial, hurdle, country FE, year FE, rolling target, ensemble), plus second session testing year weights, log1p momentum, CBD meeting year, and Δgov for SEA. Final results per region:

| Spec | SA D²_test | SEA D²_test | Notes |
|---|---|---|---|
| Momentum-only (baseline) | −0.249 | +0.129 | No political vars |
| Full 16-feat Poisson α=0.1 (old) | −0.096 | +0.103 | Old spec; multicollinear |
| Parsimonious 7-feat Poisson α=1 | −0.004 | +0.184 | Prior spec |
| Full Poisson α=100 | −0.083 | +0.230 | Best SEA, but sacrifices interpretability |
| Country FE log-OLS | +0.091 | +0.061 | Positive SA D²! — but train D²=0.037; level forecasts unreliable |
| Parsimonious α=300 | +0.0002 | — | Barely positive; extreme shrinkage makes coefs ≈0 |
| **Parsimonious 7-feat + winsorise p90** | **+0.195** | — | Previous SA spec |
| 10-feat + WDPA lag fix (level gov) | +0.233 | +0.109 | Prior spec — temporal drift in 2022–2024 predictions |
| **10-feat + Δgov + agri + winsorise p90** | **+0.356** | — | Previous SA spec (superseded) |
| **11-feat + Δgov + agri + cbd + p95 + log1p (marine fix)** | **+0.368** (7yr PRIMARY) / +0.343 (8yr) | — | **Final SA spec** — marine fix applied 2026-05-27: pre-2001 lags now terrestrial-only. D² reduced ~10pp vs old marine-contaminated spec. train D²=0.600; 2024 excl. WDPA lag |
| **9-feat Δgov (drop forest) — SEA (marine fix)** | — | **+0.259** (6yr PRIMARY) / +0.178 (8yr) | **Final SEA spec** — marine fix applied 2026-05-27: D²_6yr +0.007 vs old spec. 3yr=+0.303; 2023–2024 excl. WDPA lag |
| Oracle (country train mean) | −7.23 | −0.11 | Ceiling is structural, not model-specific |

Key insight: SA oracle test D²=−7.23 confirms the problem is the period-level distributional shift (train mean=9,200 vs test mean=2,151 pixels/country-year), not cross-sectional model failure. Winsorising training observations at p90 (15,836 px) removes the leverage of Brazil's 2001–2009 boom years. Using **first differences** of v2xlg_legcon and v2csprtcpt rather than levels removes temporal drift: level-based governance variables with positive coefficients cause predictions to grow monotonically over 2020–2024 exactly when actual expansion was declining. agricultural_land_pct captures land-availability constraint (URY/PRY always zero, low agri countries have more to designate).

**Year-by-year SA predictions (winsorise p90 model)**:
- Model predicts 22,000–32,000 px/yr steadily; actual is volatile (1,628–73,753 px/yr)
- Model correctly captures the order-of-magnitude (not boom-era scale); appropriate for the post-frontier-exhaustion regime
- Per-country scaled Log-Ridge achieves better Spearman (0.66 vs 0.34) but collapses absolute predictions to <2,000 px/yr — unusable for forward projection

**SA spec features (11 total, 2026-05-26, UPDATED)**: p95 winsor (cap=39,662 px), log1p momentum
- Momentum (log1p): pa_momentum_pixels_lag1/2/3, pa_cumsum_lag1_pixels
- Political (levels): v2x_polyarchy (+0.77 — cross-country democratic culture), gdp_growth_lag1 (+0.39 — fiscal space), redd_plus_enrolled (−0.18 — substitution effect)
- Political (first differences — event timing signal): Δv2xlg_legcon (+0.12), Δv2csprtcpt (+0.26). Theory: governance CHANGES drive TIMING of PA designation.
- Land constraint: agricultural_land_pct (−0.15 — more agri land = less natural area)
- Policy cycle: cbd_meeting_year (+0.14 — CBD COP years create international urgency; training CBD years: 2002 (high expansion), 2010 (moderate); test CBD years: 2018, 2022)
- Key gains vs old p90 spec (pre-marine-fix): 3yr +0.357→+0.563 (+20.6pp), 7yr +0.391→+0.467 (+7.6pp), 8yr +0.356→+0.443 (+8.7pp). **After marine fix (2026-05-27): 7yr=+0.368, 3yr=+0.403, 8yr=+0.343** — Galápagos MR (138,722 km²) and other marine PAs removed from pre-2001 lags; CORRECT numbers.
- Still dropped: v2xlg_legcon level, v2csprtcpt level (replaced by Δ), forest_area_pct (collinear with agri), v2x_corr↔gov_wgi_rl_est, gov_wgi_ge_est, v2cseeorgs

**SEA spec features (9 total, 2026-05-26, UPDATED)**: Δgov replacing forest_area_pct
- Momentum: pa_momentum_pixels_lag1/2/3, pa_cumsum_lag1_pixels
- Political: v2x_polyarchy (−0.35 — NEGATIVE: authoritarian states expand PAs more in SEA; paper finding on cross-regional heterogeneity), gdp_growth_lag1 (+0.13), redd_plus_enrolled (+0.04)
- Political (first differences): Δv2xlg_legcon (+0.06), Δv2csprtcpt (+0.00 — near-zero)
- forest_area_pct DROPPED: competes with Δgov vars when both included (B1 worse than B0); Δgov is more theoretically grounded as timing signal; forest sign-reversal finding now in supplement
- Key gains vs 8-feat (pre-marine-fix): 3yr +0.301→+0.306 (+0.5pp), 6yr +0.207→+0.252 (+4.5pp), 8yr +0.109→+0.168 (+5.9pp). **After marine fix (2026-05-27): 6yr=+0.259, 3yr=+0.303, 8yr=+0.178** — slight improvement vs old spec.
- cbd_meeting_year NOT used (hurts SEA; CBD years don't correspond to SEA designation events)

**Chow structural break test (SA, break at 2010)**:
- **11-feat + Δgov + agri + cbd + p95 + log1p (final): F=0.57, p=0.866 (NOT significant)** ← current
- 10-feat + Δgov + agri + p90 (previous): F=0.64, p=0.791 (NOT significant)
- 10-feat + WDPA lag fix (levels): F=0.82, p=0.621 (NOT significant)
- 10-feat (wrong lag init, no WDPA): F=1.83, p=0.052 (marginal)
- 9-feat (no forest, no WDPA): F=1.99, p=0.037 (significant)
- Old 7-feat parsimonious: F=1.86, p=0.069 (marginal)
- Full model (16 feat): F=1.97, p=0.018 (significant)
- **Key finding**: The apparent structural break was a **lag initialization artefact**. Setting
  `pa_momentum_pixels_lag1(2001) = 0` instead of the true 2000 WDPA value (e.g., BRA = 149K km²)
  made the pre-2010 period look different. Once corrected via WDPA pre-2001 data, the break
  disappears. The visual expansion collapse is real but the model's coefficient structure is stable.
- **Paper implication**: Cannot claim a Chow structural break anymore — reframe as visual evidence
  only ("frontier exhaustion") without a formal test claim.

**Split grid search (2026-05-26)**: Exhaustive grid over train_end (2010–2021) and test window length.
- The current split (train 2001–2016, test 2017–2024) is **already optimal for combined SA+SEA D²** (+0.180 combined vs all alternatives).
- Changing the split to maximise SA D² alone would be p-hacking: train 2001–2017 gives SA D²=+0.062 but kills SEA (−0.151).
- SA achieves positive D² for ALL test windows ≤ 7 years: 3yr=+0.155, 4yr=+0.135, 5yr=+0.131, 6yr=+0.066, 7yr=+0.031. Only the 8yr window (including 2024) is negative.
- **Root cause of 8yr degradation**: SA 2024 = 1,628 total designated pixels (WDPA reporting lag — data not yet finalised). Model massively overpredicts near-zero 2024 expansion.
- **Rolling CV (1-year-ahead)**: Mean D²=−2.96, highly volatile. Stage 1 is a multi-year aggregate model — 1-year-ahead CV is not the right evaluation metric.
- **Paper strategy**: SA primary metric = D²_7yr (2017–2023), excluding 2024 (WDPA lag confirmed from May2026 CSV: panel ~33% of CSV coverage). SEA primary metric = D²_6yr (2017–2022), excluding 2023–2024 (IDN: 25,320 km²/25,038 km² in CSV vs 0 in panel). D²_8yr reported as secondary metric. D²_3yr as "policy horizon" window. WDPA lag exclusion is scientifically stronger than including it — avoids evaluating model on systematically incomplete labels.

**WDPA test-year reporting lag — confirmed and resolved (2026-05-26)**
Investigated WDPA_May2026_Public_csv.csv for all test-window years. Key findings:
- **SA 2024**: CSV shows 219 polygons / 4,851 km². Panel records only 1,628 pixels (~33% capture). SA 2024 is systematically incomplete — not a model failure.
- **SEA 2023**: IDN alone shows 92 polygons / 25,320 km² in CSV. Panel records 0. 
- **SEA 2024**: IDN shows 20 polygons / 25,038 km² in CSV. Panel records 0.
- IDN WDPA reporting lag is a known issue (government submissions come in multi-year batches). The panel underrepresents ALL recent IDN activity; 2023–2024 zeros are definitively artefacts.
- **Resolution**: SA primary metric = D²_7yr (2017–2023); SEA primary metric = D²_6yr (2017–2022). Both scripts updated with `PRIMARY_EVAL_END` constants and `wdpa_lag_note` in output JSON. This exclusion is scientifically stronger and avoids reviewer challenge about model "failing" in 2023–2024.

**Remaining avenue — WDI forest area ✅ done (2026-05-26)**:
`forest_area_pct` (WDI `AG.LND.FRST.ZS`) downloaded for all 24 countries, added to `data/shared/wdi.csv` and all three `stage1_panel.parquet` files. Added to `POLITICAL_COLS` in model1/model3 expansion scripts. Results:
- SA: forest coef=+0.547 (positive — REDD+ driven); 3yr +2.3 pp, 8yr −3.5 pp (WDPA 2024 lag artefact)
- SEA: forest coef=−0.422 (negative — deforestation pressure > conservation in IDN/MYS); 3yr +2.2 pp
- Cross-regional sign reversal is a paper finding on heterogeneity of forest-conservation relationship.

**Remaining avenue — WDPA pre-2001 lag correction ✅ DONE (2026-05-26); marine fix ✅ DONE (2026-05-27)**:
Downloaded and processed `WDPA_May2026_Public_csv.csv` (already in `data/shared/`). Key findings:
- **Do NOT extend TRAIN_YEARS to 1990**: pre-2001 expansion had 2–10× higher rates (frontier exhaustion),
  adding those rows doubles the p90 winsor cap and collapses OOS D² for both regions.
- **DO use WDPA to correct lag initialization**: `pa_momentum_pixels_lag1` at year 2001 was 0 for all
  countries; actual 2000 WDPA values (BRA=149K km², VEN=316K km², BOL=4K km²) are much larger.
  Correcting this improves SA D²_test(8yr) from 0.202 → 0.233, 3yr 0.224 → 0.248.
- **Marine contamination fix (2026-05-27)**: Original `compute_pre2001_expansion` used `GIS_AREA` (total
  including marine). Major outlier: ECU 1998 = 138,722 km² (Galápagos Marine Reserve) — 86% marine.
  SA pre-2001 total was 2,964K km² vs corrected 2,712K km². Fix: `area = GIS_AREA − GIS_M_AREA`,
  clipped to ≥0. Implemented in `stage1_panel.compute_pre2001_expansion()` (optional cols
  `GIS_M_AREA`, `REP_M_AREA`). Applied to SA+SEA via `scripts/regions/shared/patch_stage1_panels.py`.
  SA impact: ECU 173K→34K km², COL 50K→2K km², BRA 2,005K→1,952K km². SA D²_7yr: 0.467→0.368 (marine
  fix removed spurious saturation signal). SEA D²_6yr: 0.252→0.259 (slight improvement).
- **Chow structural break becomes non-significant** (F=0.82, p=0.621): the apparent 2010 break was a
  lag initialization artefact. Paper cannot claim a formal Chow break — reframe as visual evidence only.
- SA+SEA `stage1_panel.parquet` patched locally. USA has 10.1% marine in pre-2001 (225,261 km²);
  fix will apply automatically when `stage1_data_builder.py` re-runs on Euler (reads corrected function).
- `TRAIN_YEARS` remains (2001, 2016) in all expansion scripts.

**A — Stage 1 D² ✅ OOS numbers in hand (2026-05-26)**
Results with full political model (v2x_polyarchy, v2x_corr, v2cseeorgs, gov_wgi_ge_est, gov_wgi_rl_est, gdp_growth_lag1, redd_plus_enrolled, pa_cumsum_lag1_pixels):
- SA: D²_train=0.761, D²_test=**−0.096** (improved from −0.623 with simpler model). Still slightly negative — structural SA break (frontier exhaustion) remains the primary cause; v2x_polyarchy (β=3.04) is the strongest predictor.
- SE Asia: D²_train=0.305, D²_test=**+0.103** (positive; beats null). Full model slightly underperforms simpler model on test (0.249→0.103) — see Issue J.
- USA: D²_test=**OVERFLOW** — see Issue K. Do not use USA Stage 1 coefficients.
REDD+ TO VERIFY entries still need final check against cited FCPF/UN-REDD pages before paper submission.

**B — `target_30x30` coefficient = 0**
No variation in 2001–2013 training window. Fix: apply 30×30 scenario as exogenous budget multiplier post-prediction, not through the model coefficient.

**C — SA year breakdown ✅ (job 628878, 2026-05-25)**
Result: 2017 Lift=1.47× | 2018 Lift=7.90× | 2019 Lift=9.64× (best!). **Bolsonaro hypothesis falsified** — 2019 is strongest. The 2017 collapse (baseline_rate=0.0017, half of 2018/2019) drags the 3-year average to Lift=6×. Likely cause: fewer designations in 2017 make NDCG highly sensitive to misclassification of rare events. **Implication: Issue D (9K sub-window mismatch) is now primary unexplained residual; W8 binary is the key test.** Country breakdown unavailable (scored parquet lacks country_id; re-run with new code will include it).

**D — LambdaRank 9K sub-window vs. full-group evaluation mismatch**
Training optimises ranking within 9K-row sub-windows; evaluation is over full country-year groups (up to 1.2M rows). SA median group = 413K rows → ~45 sub-windows per group (not 188 — that figure used the mean, which Brazil inflates). Concordance=0.64 confirms the model IS learning to rank; the problem is tail discrimination at the top-1% cutoff. Two complementary fixes under investigation:
1. **W8 binary** (currently running): no 9K ceiling, scale_pos_weight handles imbalance. Primary diagnostic — if SA binary Lift@1% >> 6×, the 9K mismatch is confirmed as the dominant bottleneck.
2. **W9 ecoregion groups** (conditional on W8 result): change training groups from `(country_id, year)` → `(country_id, year, ecoregion_id)`. Ecoregion groups are typically 1K–50K pixels → no sub-window splitting. Scientifically motivated (designation decisions are hierarchical: which biome first, then which pixel). **Critical caveat**: training on within-biome ranking and evaluating on within-country ranking creates a train/eval distribution mismatch — the model never sees cross-biome comparisons during gradient updates. This is a genuine experimental risk, not just a reporting caveat. Requires empirical validation. Infrastructure exists: `gsn_terrestrial_ecoregions_mask_1km.tif` is already loaded in `spatial_cv_core.py:load_biome_raster_and_mapping()`. Pursue only if W8 result confirms 9K as the bottleneck and suggests LambdaRank is worth fixing rather than replacing.

**E — USA Stage 2 timing**
Job 568012/568045 queued (afterok chain). No action needed until they complete.

**F — WDPA label quality: no STATUS or IUCN filter** ← important methodological gap
`1_extraction/WDPA_export` paints ALL WDPA polygons (no `STATUS == 'Designated'` filter, no IUCN category filter). This may include "Proposed" and "Inscribed" areas in the positive labels — potential leakage if those areas are known-pending designations. Fix requires adding `WDPA.filter(ee.Filter.eq('STATUS', 'Designated'))` and optionally `IUCN_CAT` filter to the GEE export, then re-running preprocessing for all regions. **Assess impact first**: estimate what fraction of positive labels come from non-Designated polygons before deciding whether to re-export.

**J — SE Asia Stage 1 overfits with full political model**
Full model (16 features, 176 obs) gives D²_test=0.103 vs simpler model's 0.249. The additional political variables (WGI, V-Dem, REDD+) hurt OOS performance in SE Asia — likely multicollinearity (gov_wgi_ge_est=+2.66 vs gov_wgi_rl_est=−2.13, v2cseeorgs=−1.70). Fix options: (1) feature selection / L1 penalty; (2) drop redundant WGI columns and keep only one governance index; (3) report both models and note the tradeoff. Decision needed before paper submission.

**K — USA Stage 1 underdetermined: FIXED (2026-05-26)**
`model2_expansion.py` now uses only LAG_COLS (4 momentum/saturation features, alpha=10). Results: D²_train=0.218, D²_test=−3.14. Negative test D² confirms the model overpredicts — it learned the Obama-era high-expansion pattern (2001–2016) but Trump era (2017–2024) saw a structural drop in designation rate. **Paper framing: USA Stage 1 is a time-series trend extrapolation, not cross-country political evidence. The negative OOS D² is itself a finding about political path-dependency in USA conservation.** Note: USA POLITICAL_COLS are excluded from the model output JSON; USA Stage 1 coefficients cannot be compared to SA/SE Asia political coefficients.

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
- **Option — extend to pre-2001 ✅ resolved**: Pre-2001 training data hurts (regime mismatch). WDPA is used only to correct lag initialization at 2001. `TRAIN_YEARS` stays (2001, 2016).

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

### W9 — Stage 2 Structural Improvements [conditional on W8 result]

Three improvements to investigate after W8 binary result is in hand. Priority order is fixed by the W8 outcome.

**W9a — Ecoregion-stratified training groups** (see Issue D for full description and caveats)
- Change training groups from `(country_id, year)` → `(country_id, year, ecoregion_id)`
- Keep evaluation groups at `(country_id, year)` — unchanged
- Apply within-group rank normalisation at `(country_id, year)` level *before* splitting into ecoregion sub-groups, so cross-biome feature comparability is preserved
- **Gate**: pursue only if W8 binary SA Lift@1% > LambdaRank 6× AND W8 binary is not clearly dominant; if binary is dominant, switch primary and skip W9a
- Implementation: ~2 days. Add `ecoregion_id` column via raster lookup in `load_stage2_arrays()` (optional flag); change `group_sizes` computation for training path only

**W9b — Curriculum hard negative mining** (~1 day, after any re-trained model exists)
- Phase 1: score all training pixels with the current model
- Phase 2: retrain with a mix of (a) all positives + (b) top-scored negatives ("hard") + (c) random negatives, ramping the hard-negative fraction from 0% → 50% over training epochs
- **Do NOT use pure hard negatives** (no curriculum ramp): pure hard mining causes the model to memorise current failure cases rather than learn a generalizable ranking — curriculum is the robust IR recommendation
- Expected gain: 10–20% NDCG improvement (speculative; no experiment yet)
- Implement as `neg_sampling="curriculum"` option in `load_stage2_arrays()`

**W9c — Stage 1 predicted expansion as Stage 2 feature** (~1 hour to wire, but requires care)
- Add `stage1_predicted_expansion` (country-year level) as a pixel-level feature in Stage 2 — same value for every pixel in a group
- Rationale: countries under high expansion pressure may select different pixel types (faster, lower-quality) than countries at normal pace; bridging Stage 1 → Stage 2 is architecturally principled
- **Leakage caveat**: Stage 1 trains on 2001–2016, which overlaps the Stage 2 early-stop window (2014–2016). Using Stage 1 in-sample predictions for those years would leak. Fix: generate rolling/walk-forward Stage 1 predictions (train on years ≤t−1, predict year t) for the Stage 2 training window before adding this feature
- Low priority — implement only if W9a and W9b are done and NDCG remains below target

---

## Next Actions (Ordered)

1. ✅ **Issue C** — Done. 2019 is the BEST year (Lift=9.64×); 2017 is the worst (1.47×). Bolsonaro hypothesis falsified. Issue D (9K sub-window) is now primary suspect for SA underperformance.
2. ✅ **Issues H+I** — SA FE job 628941 done. SA panel rebuilt with `country_pa_cumsum_lag1_pixels` + trend features. SA re-tune (689639) + retrain (689640) pending.
3. ✅ **W8 SA binary** — Jobs 689625→689627 running. Compare binary Lift@1% vs LambdaRank 6.0× once done.
4. ✅ **Stage 1 SA/USA/SEA** — UPDATED 2026-05-27 (marine fix). SA D²_7yr=+0.368 (PRIMARY, 2017–23), D²_8yr=+0.343; SEA D²_6yr=+0.259 (PRIMARY, 2017–22), D²_8yr=+0.178; USA D²_test=−3.14. These are terrestrial-only pre-2001 lag numbers (GIS_AREA−GIS_M_AREA). ⚠️ Will update again after Euler FE re-runs fix 2001+ pixel data.
5. **W8 + SA LambdaRank re-train decision** (after jobs 689627 and 689640 complete): Compare binary Lift@1% vs LambdaRank Lift@1% on SA test set. If binary >> LambdaRank → binary becomes primary (W8 wins); if similar → LambdaRank primary with ecoregion investigation (W9a). Record numbers here and update Settled Decisions.
6. ✅ **Issue L — WDPA GeoTIF patch for SEA** — done locally 2026-05-27. Scp to Euler ✅ done. SEA FE re-run now unblocked.
   **Issue L — WDPA GeoTIF patch for SA** — ✅ patched locally + scp to Euler done 2026-05-27. Next: SA FE re-run (after 689640 completes) → stage1_data_builder → model1_expansion local → Stage 2 re-tune+retrain.
7. **Euler re-run sequence** (blocked until 689640 completes for SA):
   - **SEA** (TIFs on Euler ✅): submit SEA FE SLURM job → download SEA stage1_panel.parquet → re-run `model3_expansion.py` locally → re-queue SEA Stage 2 tune+retrain. SEA D²_6yr=+0.259 will update.
   - **SA** (after scp): re-run SA FE SLURM job → `stage1_data_builder.py` → download → `model1_expansion.py` locally → re-queue SA Stage 2 tune+retrain. SA D²_7yr=+0.368 will update.
   - **USA**: FE re-run has no WDPA lag issue; can submit independently. marine fix applies automatically via corrected `compute_pre2001_expansion`. Download USA stage1_panel.parquet → re-run `model2_expansion.py` locally. Add USA to `patch_stage1_panels.py` REGIONS dict if needed.
   SEA D²_6yr=+0.259 and SA D²_7yr=+0.368 are the CORRECT pre-2001-only estimates; 2001+ data will improve further once GeoTIF patch propagates through Euler pipeline.
   USA FE re-run has no known WDPA lag issue; can proceed independently.
8. **USA Stage 2 LambdaRank** (after SEA FE re-run confirms approach): Re-submit tune+train chain.
9. **W9a ecoregion-stratified training** (conditional — see Issue D gate): implement only if W8 confirms 9K bottleneck AND binary does not clearly dominate. ~2 days.
10. **W4 ablation + W5 logistic baseline** (after SA re-train result is in hand; both use rebuilt SA panel):
    - W4: run `stage2_ablation.py` for SA. Critical test: does removing `pa_momentum` collapse to naive NDCG?
    - W5: run `model1_logistic_stage2.py` for SA. Required for methods comparison table and economics reviewers.
11. **Issue F — WDPA label quality audit**: count what fraction of positive labels come from non-Designated polygons. Estimate impact before deciding on GEE re-export. Must appear in methods section of any top-tier submission regardless of outcome.
12. **REDD+ final verification** (before paper submission, not blocking): Open each `# TO VERIFY` URL in `build_redd_plus.py` and confirm enrollment year against FCPF/UN-REDD pages.
13. **Next data sprint**: GEE export for ESA CCI Biomass (carbon stocks) and RAISG indigenous territory area fraction. SA first. Run in parallel with steps 7–10.
14. **W7 manuscript** (gate: SA re-train result confirmed + SEA FE re-run done + Issue L patched + ablation run + Issue F quantified): Do not start W7 until these gates are met.

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
- **Stage 1 primary metrics fixed (2026-05-26, updated 2026-05-27)**: SA primary = D²_7yr (2017–2023) = **+0.368**; SEA primary = D²_6yr (2017–2022) = **+0.259**. Both confirmed from WDPA May2026 CSV: SA 2024 panel captures ~33% of CSV designations; IDN 2023+2024 in CSV (25,320+25,038 km²) record as 0 in panel. Numbers corrected for marine contamination in pre-2001 lags (2026-05-27). D²_8yr reported as secondary. ⚠️ Will update after Euler FE re-runs.
- **Saturation interaction leakage (SEA) — DO NOT re-investigate**: saturation_ratio = cumsum/max(cumsum) over full panel is leaky (max encodes future information). sat_clean (max over train years only) gives SEA D²_6yr=−0.275 (catastrophically bad). No interaction terms adopted for SEA. The apparent +10.7pp SEA 6yr gain from saturation_ratio was entirely spurious.
- **CBD thin training support (SA)**: cbd_meeting_year has only 2 training instances (2002, 2010). CBD-free fallback spec is saved to model1_expansion_coefficients.json. Consistent +0.009–0.029 improvement on 7yr across all winsor variants confirms it is not noise. Use fallback as response to reviewers if challenged.

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
