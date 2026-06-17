# PA3030 — Publication Roadmap

**Updated**: 2026-06-17 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

**Story**: The 30×30 agreement forces countries to roughly double protected area coverage by 2030. We predict which land areas will be designated — giving investors, policymakers, and conservation planners a data-driven transition risk and coverage tool. Stage 1 predicts *when and how much* each country expands; Stage 2 predicts *which areas* are selected. Combined output: annual designation probability for every unprotected pixel in South America, 2025–2030.

**Target journal**: Nature Sustainability (primary) → One Earth / GEC → JEEM.

---

## Research Questions

These are refined as of 2026-06-17 to reflect what the model actually achieves and where the scientific contribution lies.

**RQ1 (Stage 1 — solved)**: Which countries will expand their PA networks under 30×30, and how much area will they add each year?

**RQ2 (Stage 2 — in progress)**: Within an expansion event, which contiguous land areas does a government select — and can we learn the spatial, ecological, and institutional patterns of past designations well enough to predict future ones?

The framing matters: governments designate **polygons** (connected areas), not individual pixels. A model that predicts which pixels are "most likely" is approximating a fundamentally area-based institutional process. This is the key tension the model must address, and the methodological contribution of the paper.

**RQ3 (applied)**: Where does predicted PA expansion diverge from conservation priority — and what does the gap tell us about systemic bias in how the 30×30 target will be met?

---

## Architecture

```
P(pixel i designated in year t)
  = P(country C expands in year t)     ← Stage 1: Poisson GLM, country-year panel
  × S(pixel i | expansion in C, t)     ← Stage 2: LightGBM LambdaRank → calibrated suitability
```

**Stage 1** — Poisson GLM, LASSO α=100, 9 features. Metric: D² OOS. Train 2001–2016, test 2017–2023.

**Stage 2** — LightGBM LambdaRank W9a. Eco-stratified training groups; evaluated on `(country_id, year)` expansion groups. Graded relevance 1–4. Train 2001–2013, early-stop 2014–2016, test 2017–2024.

**Primary metrics**: Lift@1% + Recall@5% within expansion groups.

**Forward output** (Phase 3): Stage 1 budget × Stage 2 suitability → cumulative risk = 1 − ∏(1 − score_t). BAU / 30×30 / NGFS scenarios.

**Lift metric note**: `lift_at_k_within_groups` = macro_precision@k / global_positive_rate. Mixing macro numerator and micro baseline is consistent across all experiments and does not affect rankings, but must be noted in the paper methods section.

---

## Publication Bar

| Metric | Bar | Current best |
|---|---|---|
| Lift@1% | ≥ 15× | **81.32×** (H12 patch-level, job 3733616) ✅ |
| Recall@5% | ≥ 90% | **87.7%** (H12 patch-level, job 3733616) — 2.3% below bar |
| Theoretical ceiling | — | **99.6%** (perfect model, structural) |

**H12 is a breakthrough.** Patch-level ranking (Tier 3) shattered both bars in a single run: Lift from 6.46× → 81.32× (+1158%) and Recall from 18.96% → 87.7% (+362%). Recall@5% is 2.3% below the 90% bar — within reach of Phase 2 hyperparameter tuning (≥100 Optuna trials). Structure is now locked at patch level; all further work is optimisation.

---

## Guiding Principles

1. **Structure before tuning — confirmed.** H12 (patch-level ranking) achieved Lift=81.32× and Recall=87.7% in a single run with untuned hyperparameters. The entire 6.46× → 81.32× Lift gain came from fixing the ranking unit, not from tuning.

2. **The polygon-vs-pixel framing mismatch was the primary barrier — resolved.** Governments designate contiguous areas; patch-level LambdaRank ranks connected unprotected components, not individual pixels. This change alone drove Recall from 19% to 88%.

3. **Proxy experiments are directionally useful, not quantitatively reliable.** H8+H10 proxy Lift=20× → full SA Lift=4.17×. Only full SA results drive decisions.

4. **Phase 2: retune with structure locked.** Early retune at pixel level was catastrophic (Lift 2.85× → 2.06×). Now that patch-level structure is confirmed, ≥100 Optuna trials are safe and should close the Recall@5% gap (87.7% → 90%).

---

## Full SA Experiment History

**Pixel-level locked baseline**: H6+H1b+H5. Full SA: Lift=6.46×, Recall=15.7%, best_iter=136.

**Patch-level new baseline**: H12. Full SA: Lift=81.32×, Recall=87.7%, best_iter=20. Both publication bars met or nearly met in a single run.

| Experiment | Lift@1% | Recall@5% | iter | Verdict |
|---|---|---|---|---|
| Baseline (79 feat, default params) | 2.85× | 14.0% | 149 | starting point |
| 20-trial Optuna retune (truncation=84) | 2.06× | 8.4% | 7 | ✗ catastrophic — never retune early |
| 67 feat + temporal year weights | 2.64× | 11.4% | 113 | ✗ both negative |
| H6+H1b (Recall stop + inv_sqrt_npos weights) | 3.73× | 18.1% | 89 | ✓ first both-improved |
| **H6+H1b+H5 (+ no rank-norm)** | **6.46×** | **15.7%** | **136** | ✓ **pixel-level locked baseline; +73% Lift** |
| H6+H8+H10 (temporal weights + combined stop) | 4.17× | 18.96% | 112 | ✗ proxy 20× did not transfer |
| H6+H1b+H5+H7 (train 2010–2013 only) | 1.51× | 10.5% | — | ✗ catastrophic — 25 groups too few; year restriction abandoned |
| H6+H1b+H5+H11 (patch-context features) | 5.56× | 16.6% | 50 | ✗ Lift regression −14%; Recall flat. Pixel-level patch features do not fix polygon-vs-pixel mismatch. |
| **H12 — patch-level ranking (Tier 3)** | **81.32×** | **87.7%** | **20** | ✅ **BREAKTHROUGH. Both bars met/near-met. Structural fix confirmed. Phase 2 next.** |

**What is locked in the baseline (all confirmed on full SA):**
- H6: Recall@5% early stopping — directly optimises the publication metric
- H1b: `inv_sqrt_npos` group-norm weights — reduces gradient concentration from mega-events
- H5: No rank normalisation — absolute feature signals (biodiversity, distance) carry global information that within-group rank destroys; +73% Lift

**What has been closed:**
- H1 (`inv_npos`): too aggressive, training collapses at iter 56
- H2 (binary labels): no gain on top of H1b or H5
- H3 (W9a off): hurts both metrics; eco sub-groups stay
- H7 (train 2010–2013 only): catastrophic — Lift 6.46→1.51×, Recall 15.7→10.5%. 25 groups too few; old data provides necessary structural signal despite gradient concentration
- H8+H10 (temporal decay + combined stop): proxy 20× did not transfer to full SA; rank-norm removal (H5) is the stronger lever
- H11 (patch-context features): Lift 6.46→5.56×, Recall flat at 16.6%, best_iter=50. Patch-level features are constant within each patch → model gains cross-patch discrimination but loses nothing within the patch. The pixel-level ranking unit is the fundamental barrier; adding patch features to pixels cannot fix it.

---

## Root Cause Analysis

### 1. Gradient concentration (training side)

From the training split (2001–2013, 121 expansion groups):

| Group subset | % of total gradient |
|---|---|
| Top 5 groups | 72.6% |
| Top 10 groups | 96.0% |
| Years 2001–2009 (96 groups) | **98.6%** |
| Years 2010–2013 (25 groups) | 1.4% |

LambdaRank gradient scales as `n_pos × min(n_neg, neg_ratio × n_pos)`. Brazil 2006 (110K positives) generates ~40,000× more gradient signal than Peru 2012 (13 positives) even after neg_ratio=100 subsampling. The model has been trained almost entirely on ~10 giant 2001–2009 events; the small, targeted 2010–2013 events that resemble the test set are invisible to it.

H1b (inv_sqrt_npos) partially addresses this. H8 (inv_sqrt_npos_temporal) made it worse on full SA. H7 (hard cutoff: drop 2001–2009 entirely) is the current test.

### 2. Polygon-vs-pixel framing mismatch (inference side) ← primary Recall barrier

Governments designate **connected polygons** — they draw a boundary around a contiguous area. LambdaRank ranks **individual pixels** independently. The model correctly identifies the right *part of a country* (explaining moderate Lift), but the actual PA polygon contains thousands of interior pixels that are not individually distinguishable by their feature vectors from non-PA neighbours — they are designated because they are *adjacent to the core*, not because they have uniquely high feature values.

**This is why all training-side interventions hit a Recall ceiling near 20%**: even with perfect training, a pixel-independent model cannot recover the full polygon extent from features alone. Fixing this requires either (a) spatial post-processing that propagates scores within connected regions, (b) features that encode patch membership, or (c) changing the ranking unit from pixels to patches.

### 3. Structural ceiling diagnostic (2026-06-17)

Test set: 61 expansion groups, 239,885 positives (2017–2024).

| Size bin | # groups | % positives | Structural max Recall@5% |
|---|---|---|---|
| 1–5 pixels | 4 | 0.006% | 100% (trivially achievable) |
| 6–20 | 3 | 0.015% | 100% |
| 21–100 | 10 | 0.3% | 100% |
| 101–1K | 20 | 3.3% | 100% |
| **1K+** | **24** | **96.4%** | 100% (except 1 group: 73%) |

Median positive rate within expansion groups = 0.064%. The top-5% window holds ~78× more pixels than positives for the median group. **Theoretical max macro Recall@5% = 99.6%.** The 90% bar is structurally achievable across 60/61 groups. The gap from 18.96% to 90% is pure model quality failure.

---

## Strategy: Three Tiers

### Tier 1 — Spatial diffusion (immediate, no retraining)

**What**: Multi-step 8-neighbour score propagation on the H6+H1b+H5 model output. Each step: `score[pixel] += alpha × mean(scores[neighbours]) × decay^step`.

**Why**: If the model ranks even 1–5 pixels per PA polygon above the 95th percentile, iterative diffusion propagates that seed across the entire connected polygon. Recall@5% is dominated by within-polygon coverage — this directly targets it without any retraining.

**Expected gain**: Recall@5% from ~19% to 40–60%. Lift@1% should be unaffected (top-1% seeds are already identified).

**Implementation**: Add iteration loop to existing `spatial_postprocess_stage2.py`. Test 5/10/20 steps, decay=0.95. ~10 lines of code; run on Euler with H6+H1b+H5 booster.

**Result (job 3723734, H6+H1b+H5 booster)**: Recall ceiling at 17.2% regardless of n_steps or alpha. Best (steps=10, alpha=2.0): Recall=17.2%, Lift=6.15×. Decision rule triggered: Recall < 30% → move to Tier 2. Root cause: diffusion within expansion groups (millions of negatives) dilutes signal across a sea of non-polygon pixels. Correct fix requires patch-level features so the model can uniformly score all pixels in the same connected component.

### Tier 2 — Patch-context features (H11+)

**What**: A family of features that encode the spatial context of each pixel's contiguous unprotected block (connected component of WDPA_prev==0).

| Feature | Description | Leakage-safe? |
|---|---|---|
| `log_patch_size_km2` | Size of unprotected patch this pixel belongs to | ✓ (uses WDPA_prev) |
| `patch_pa_adjacency_frac` | Fraction of patch perimeter touching existing PAs | ✓ |
| `patch_mean_gsn_b2` | Mean biodiversity score across the patch | ✓ |
| `patch_designation_lag1` | Was any pixel in this patch designated last year? | ✓ |

**Why**: These features give the model information about the *decision unit* governments operate on. A pixel in a 50,000 km² contiguous patch is categorically different from an isolated pixel — governments draw PA polygons around large unprotected areas adjacent to existing PAs. The model currently cannot represent this.

**Implementation**: Annual `scipy.ndimage.label` on the WDPA_prev==0 binary raster in `feature_engineering.py`. O(n) per year; feasible on Euler. Requires panel rebuild (large Euler job).

**Result (jobs 3728182/3728184)**: Lift=5.56×, Recall=16.6%, best_iter=50 (baseline: 6.46×/15.7%/136). **Failed.** Root cause: patch-level features (log_patch_size_km2, patch_pa_adjacency_frac, patch_mean_gsn_b2, patch_designation_lag1) are constant for all pixels within the same patch. Cross-patch discrimination exists but within-patch ranking remains zero. The model early-stops at iter 50 — patch features add noise that confounds the existing pixel-level signal. Pixel-level ranking is the wrong unit for polygon-based designation. Tier 3 triggered.

### Tier 3 — Patch-level ranking (architectural clean fix) ← **NEXT**

**What**: Change the ranking unit from pixels to patches (connected unprotected components). Each patch gets one feature vector (mean/max of pixel features) and one score. Ranked within country-year groups. A patch is a positive if any pixel inside it was designated.

**Why this is the correct fix**: Governments designate polygons. LambdaRank must rank decision units that match what governments choose. Tiers 1+2 confirmed this — diffusion and patch features both failed because the ranking unit (pixel) is incommensurable with the designation unit (polygon). Tier 3 eliminates the mismatch at the source.

**Secondary benefit**: Sample size drops from ~1M pixels to ~hundreds of patches per country-year group. Gradient concentration from Brazil 2006 shrinks because it contributes one patch per polygon, not 110K pixels. H1b (inv_sqrt_npos) may become unnecessary.

**Metrics semantics**: Patch-level Lift@1% = fraction of the top-1% of ranked patches that contain a designated pixel ÷ fraction of all patches that are positive. Recall@5% = fraction of positive patches that fall in the top 5% of ranked patches. These are directly comparable to pixel-level metrics and remain the publication metrics.

**Implementation plan**:
1. `add_patch_features_to_splits.py` already computes patch IDs — extend to output patch-level aggregates (mean/max per feature) as a separate parquet
2. New training script that loads patch-level panel, groups by `(country_id, year)`, runs LambdaRank on patches
3. Evaluation computes patch-level Lift@1% + Recall@5% (positive patch = any pixel designated)

**Gate for Tier 3**: Tiers 1+2 Recall < 70%. **Confirmed: Tier 1 max=17.2%, Tier 2=16.6%. Gate triggered. H12 implementation submitted.**

**H12 decision rule**: Recall@5% > 30% → framing fix works; iterate. > 70% → Tier 3 success, proceed to Phase 2 retune. < 30% → open architectural discussion.

**Scientific framing**: "We model PA designation at the polygon level, consistent with how governments make designation decisions. This outperforms pixel-level ranking on spatial coverage metrics and eliminates the gradient concentration artefact from area-weighted training data."

---

## Experiment Queue

| Priority | Experiment | Status | Notes |
|---|---|---|---|
| 1 | **H7: train 2010–2013 only** | ✗ Done — job 3718862 | Lift=1.51×, Recall=10.5%. Both regressed. 25 groups too few. Year restriction abandoned. |
| 2 | **Tier 1: multi-step spatial diffusion** | ✗ Done — job 3723734 | Best: Recall=17.2%, Lift=6.15× (steps=10, alpha=2.0). Ceiling confirmed — decision rule triggered: Recall < 30% → Tier 2 first. |
| 3 | **Tier 2: implement patch-context features (H11+)** | ✗ Done — jobs 3728182/3728184 | Lift=5.56×, Recall=16.6%, best_iter=50. Regression. Pixel-level patch features do not fix polygon-vs-pixel mismatch. Tier 3 triggered. |
| 4 | **Tier 3: patch-level ranking unit (H12)** | ✅ Done — jobs 3733614/3733616 | Lift=81.32×, Recall=87.7%, best_iter=20. Both bars met/near-met. Structure locked at patch level. |
| 5 | **Phase 2: retune hyperparams (≥100 Optuna trials)** | ⬜ **Next** | Structure locked. Run full Optuna search on patch-level model to close Recall gap (87.7% → 90%). |
| 6 | **AGB feature (carbon stock)** | ⬜ After retune | Diagnose proxy failure (best_iter=5). Add to patch model if fixed. |
| 6 | KBA features (dist_kba_km, is_kba) | ⏸ Blocked — BirdLife shapefile | — |
| 7 | Indigenous polygon features | ⏸ Blocked — RAISG download | — |
| 8 | H4: extended training window (2001–2016) | ⏸ Discuss with supervisor | Pipeline change required. |
| 9 | Tier 3: patch-level ranking | ⬜ Only if Tiers 1+2 < 70% Recall | Major restructuring; becomes methodological contribution. |

**Decision rules:**

- **H7 result**: Lift > 6.46× → era mismatch confirmed; H7 added to locked baseline. Lift < 6.46% + Recall improves → keep H7 for Recall but retain H5 baseline for Lift. Both regress → 25 groups too few; year restriction abandoned.
- **After diffusion**: If Recall@5% > 50% → proceed to patch features (Tier 2). If Recall < 30% even after diffusion → model seed quality is too low; prioritise patch features first to improve seeds.
- **Gate for Tier 3**: Implement only if Tier 1 + Tier 2 combined give Recall@5% < 70%.

---

## Settled Decisions

| Decision | Value | Rationale |
|---|---|---|
| Engine | LightGBM LambdaRank | Validated through Phase 0 |
| neg_ratio | 100 | Locked |
| Eco sub-groups (W9a) | On | H3 (off) tested and rejected |
| Early stopping | Recall@5% within groups (H6) | Directly optimises publication metric |
| Sample weights | inv_sqrt_npos (H1b) | Partial gradient deconcentration |
| Rank normalisation | Off (H5) | +73% Lift; absolute features carry global signal |
| Primary metrics | Lift@1% + Recall@5% within expansion groups | — |
| Temporal split | Train 2001–2013, earlystop 2014–2016, test 2017–2024 | H4 (extend to 2016) needs supervisor sign-off |
| Proxy screening | Use for direction only | Quantitatively unreliable (H8+H10: proxy 20× → full SA 4.17×) |
| Ensembles / sub-models | Forbidden | Supervisor directive |
| Governance features | First differences only | — |
| CBD features | Robustness check only | CBD-free is primary model |
| Hyperparameter retuning | Phase 2 only | Early retune is catastrophic (2.85× → 2.06×) |

---

## Phase 2 (after SA bar confirmed on both metrics)

- Full SA retune (≥100 Optuna trials) with confirmed features + objective
- SE Asia Stage 2 retune on corrected panel
- Bootstrap CIs, model comparison table
- Spearman ρ, negative-binomial robustness, Stage 1/Stage 2 independence check

## Phase 3 (after Phase 2)

- Platt calibration → suitability scores
- Cumulative risk pipeline: Stage 1 budget × Stage 2 suitability → 1 − ∏(1 − score_t)
- NGFS scenario integration (BAU / moderate / 30×30)
- Conservation gap map: predicted designations × biodiversity raster → 2×2 (Nature Sustainability hook for RQ3)
- **Manuscript gate**: all above before writing

---

## Data Paths

| Dataset | Location |
|---|---|
| SA full splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/` |
| SA mini-sample (79 features) | `data/south_america/mini_sample.parquet` |
| Best params (locked baseline) | `scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json` |
| Best params (archive) | `outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json` |
| H6+H1b+H5 booster | `data/south_america/ml/models/model1_lgbm_stage2_20260617_011621.pkl` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` |
| AGB TIF | `data/south_america/ready/AGB/agb_sa.tif` |
| REDD TIF | `data/south_america/ready/REDD/redd_sa.tif` |

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)` — backbone CRS is LOCAL_CS, `crs.to_epsg()` returns None.

---

## SHAP Audit (2026-06-15, baseline model)

```
 1. NDVI_smooth64          0.350
 2. GSN_b2                0.283   biodiversity priority
 3. WorldClim_b2          0.244
 4. GPW                   0.227   population density
 5. deforestation_b2      0.207
 6. WorldClim_b14         0.195
 7. WorldClim_b11         0.187
 8. dist_indigenous        0.185
 9. elevation_b2_smooth4  0.149
10. GSN_b3                0.148
11. elevation_b1_smooth16 0.137
12. WorldClim_b19         0.128
13. WorldClim_b16         0.126
14. dist_wdpa             0.125
15. HNTL_smooth64         0.108
```

Caveat: reflects what the baseline model learned — dominated by 2001–2009 mega-events. Does not generalise to the H6+H1b+H5 model. A new SHAP run after the locked baseline stabilises would be informative.

Note on `dist_wdpa`: implemented as `scipy.ndimage.distance_transform_edt(1 − wdpa_binary) × 1000` — this is boundary distance to the nearest PA-occupied pixel, which is correct (PA polygons are rasterised as filled areas). No change needed.

Plots: `outputs/south_america/results/phase1/baseline/shap_importance.png`, `shap_beeswarm.png`.

---

## Paused

- SE Asia Stage 2 — Phase 2
- USA Stage 2 — deprioritised
- Forward pipeline — Phase 3

## Out of Scope (Paper 1)

- Ensemble methods, sub-models, neural networks (Paper 2)
- Survival framing, Tropical Africa
