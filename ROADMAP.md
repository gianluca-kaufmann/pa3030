# PA3030 — Publication Roadmap

**Updated**: 2026-06-16 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

**Story**: The 30×30 agreement forces countries to double protected area coverage by 2030. We predict which pixels will be designated — giving investors, central banks, and policymakers a transition risk tool. Stage 1 predicts *when* countries expand; Stage 2 predicts *which pixels* are chosen. Stage 2 output = calibrated suitability score (annual designation probability per pixel).

**Target journal**: Nature Sustainability (primary) → One Earth / GEC → JEEM. Nature Finance if NGFS integration strong.

---

## Architecture

```
P(pixel i designated in year t)
  = P(country C expands in year t)     ← Stage 1: Poisson GLM, country-year panel
  × S(pixel i | expansion in C, t)     ← Stage 2: LightGBM LambdaRank → calibrated suitability
```

**Stage 1** — Poisson GLM, LASSO α=100, 9 features. Metric: D² OOS. Train 2001–2016, test 2017–2023.

**Stage 2** — LightGBM LambdaRank W9a. Eco-stratified training groups; eval on `(country_id, year)` groups. Graded relevance 1–4. Train 2001–2013, early-stop 2014–2016 (full), test 2017–2024. Primary metrics: Lift@1% + Recall@5% within expansion groups.

**Forward output** (Phase 3): Stage 1 budget × Stage 2 suitability → cumulative risk = 1 − ∏(1 − score_t). BAU / 30×30 / NGFS scenarios.

---

## Publication Bar

| Metric | Bar | Full SA best |
|---|---|---|
| Lift@1% | ≥ 15× | 2.85× (baseline) |
| Recall@5% | ≥ 90% | 14.0% (baseline) |

---

## Data Audit Findings (2026-06-16)

Full analysis of SA mini-sample (4M rows, 2001–2013). These findings drive the Phase 1 strategy.

### 1. Temporal Designation Collapse — The Core Problem

PA designation rates in South America collapsed by a factor of ~100× across the training window:

| Period | Rate | Share of training positives |
|---|---|---|
| 2001–2009 | 0.5–2.2% per year | **96.1%** of all positives |
| 2010–2013 | 0.003–0.31% per year | **3.9%** of all positives |
| 2014–2016 (earlystop) | ~0.04% | — |
| 2017+ (test) | ~0.16%+ | — |

**Consequence**: The model trains almost exclusively on the pre-2010 high-designation regime. The 2013 year (end of training) contributes only 48 positives in the mini-sample vs. 6,765 in 2006. The earlystop (2014–2016) and test (2017–2024) represent a different policy era — the 30×30 and REDD+ period — where designation volumes partially recover but with different geographic targets.

The default temporal weighting (min_weight=0.5, linear 2001→2013) barely addresses this: 2001 gets half the weight of 2013, but 2013 itself has almost no positives. **The model is dominated by 2001–2009 gradient signal regardless.**

### 2. Small Events Are Essentially Unpredictable

79 expansion groups in the mini-sample split into:
- **34 groups with 1–5 positives**: All feature z-scores near zero (|z| < 0.2). Positives are indistinguishable from negatives in any measured feature. These events are driven by institutional/political decisions (REDD+ agreement, land tenure deal, NGO acquisition) not captured in ecological or physical features.
- **76 groups with ≥ 6 positives**: Key features show meaningful z-scores (0.4–1.2). These are learnable.

**Critical implication**: No feature-based model can reliably predict the ~31% of expansion events driven by small, targeted designations. The publication bar of Recall@5% ≥ 90% requires either capturing institutional drivers (KBA, REDD+, indigenous poly) or accepting that small events set a hard floor. Current features are insufficient for small events; adding KBA + indigenous polygon features is the highest-priority structural intervention.

### 3. Feature Redundancy and Noise

**Near-zero discriminators** (dropped from model as of 2026-06-16, 79 → 67 features):
| Feature(s) | Issue |
|---|---|
| `elevation_b2` + all smooths | slope — |z| ≈ 0 across all analyses |
| `elevation_b1_smooth4` | r = 0.999 with raw b1; zero additional information |
| `powerplants_b2` | |z| = 0.016, 0.73% null, no discriminative value |
| `GSN_b1` + smooths | |z| < 0.20; biodiversity signal carried by GSN_b2/b3 |
| `GSN_b5` + smooths | |z| < 0.03, near-constant values |

**WorldClim collinearity** (19 bands, 14+ pairs with |r| > 0.9):
- b1 ↔ b2,b3,b16,b18,b19: r > 0.92
- b4 ↔ b5,b8: r > 0.92
- b6 ↔ b9: r = 0.993
- b13 ↔ b14, b14 ↔ b17: r > 0.90

Only ~7 independent dimensions in the 19-band space. Best discriminators: b14 (z=−0.89), b8 (z=0.87), b4 (z=0.85), b11 (z=0.67), b16 (z=0.74). **Reducing to 8 targeted bands is Phase 1B.**

**Top 10 discriminating features** (|z| from pos/neg pooled analysis):
1. `landcover` (−1.04)
2. `WorldClim_b14` precipitation driest month (−0.89)
3. `WorldClim_b8` temp wettest quarter (0.87)
4. `NDVI_smooth64` regional vegetation (0.87)
5. `WorldClim_b4` temp seasonality (0.85)
6. `WorldClim_b5` max temp warmest month (0.84)
7. `dist_road` (0.83)
8. `WorldClim_b17` precip driest quarter (−0.82)
9. `WorldClim_b16` precip wettest quarter (0.74)
10. `dist_indigenous` (−0.66)

### 4. Event-Size × Feature-Signal Interaction

| Event size | Key features z-score (avg) | Structural recall ceiling |
|---|---|---|
| 1–5 positives (34 groups) | ~0.05 | 100% (positives fit in top 1%) |
| 6–100 positives (44 groups) | ~0.4 | 100% |
| >100 positives (32 groups) | ~0.8–1.2 | 100% |

Even for small events, a perfect ranker could achieve Recall@5%=100% (positives fit within top 1% by count). The gap between current 14% and ceiling is purely model quality for large events, and partially unpredictability for small events.

### 5. What Drives the Publication-Bar Gap (2.85× vs. 15×)

In order of estimated impact:
1. **Missing institutional features** — KBA, indigenous polygons, REDD+ hotspots. Small events (33% of test groups) are driven by these; current features can't see them. *~30–40% of the recall gap.*
2. **Training distribution mismatch** — 96% of gradient signal from 2001–2009. Model learns "cheap remote land" pattern, not modern targeted designation. *~20–30% of the gap.*
3. **Feature noise** — 12 near-zero features dilute splits; WorldClim collinearity wastes tree capacity. *~5–10% gain from pruning.*
4. **Hyperparameter instability** — low truncation level (84 in failed retune) caused early stopping noise. *Fixed by reverting to baseline params.*
5. **Earlystop sensitivity** — stopping at iter 149 may be premature; noisy NDCG@1% on small earlystop sets. *Longer patience (150) may help.*

---

## Current Situation

**Best result to date (baseline, 2026-06-15):** Full SA Lift@1%=2.85×, Recall@5%=14.0%, test 2017–2024, 61 groups, 107M rows.

**Failed retune (2026-06-16):** 20-trial Optuna on full SA found truncation_level=84 (vs baseline 741). Best_iter=7, essentially untrained. Lift@1%=2.06×, Recall@5%=8.4%. **Reverted to baseline params.**

**Key realisations from data audit:**
- Proxy is definitively abandoned (5× overoptimistic on Lift@1%)
- All future experiments run on full SA only
- Small events are a structural limitation — KBA and indigenous polygon features are the highest-priority interventions
- Feature reduction from 79→67 implemented (near-zero features dropped)
- Temporal weighting made more aggressive (STAGE2_YEAR_WEIGHT_MIN=0.2 env var)

**Next job queued:** 3561527 (pending) — baseline params + 67 features + YEAR_WEIGHT_MIN=0.2

---

## Phase 1A — Currently Running / Immediate

| Step | Status | Detail |
|---|---|---|
| Revert to baseline params (truncation=741, lr=0.081) | ✅ Done | `5_training/model1_stage2_lgbm_best_params.json` restored |
| Feature pruning 79→67 | ✅ Done | 12 near-zero features added to `STAGE2_EXCLUDE_COLS` |
| Temporal weighting YEAR_WEIGHT_MIN=0.2 | ✅ Done | env var in `stage2_lgbm_core.py` + SLURM script |
| Full SA retrain | ⬜ Queued (job 3561527) | Record Lift@1% + Recall@5% in `feature_ablation_sa.json` |

Expected outcome: modestly better than 2.85× Lift@1% (feature pruning removes noise; temporal weighting focuses on recent years). If result is worse than baseline, YEAR_WEIGHT_MIN=0.5 (revert to default).

---

## Phase 1B — Feature Experiments (after 1A result confirmed)

Run each on full SA (submit `training_lgbm_stage2_phase1.slurm`). No per-experiment retuning — use baseline params throughout. Record in `feature_ablation_sa.json`.

| Priority | Experiment | Rationale | Expected impact |
|---|---|---|---|
| 1 | **`is_kba` + `dist_kba_km`** | ⏸ Blocked — awaiting BirdLife shapefile | Strongest intent signal. KBA = IUCN "should be protected." Small events highly likely overlap KBAs. **High impact.** |
| 2 | **`in_indigenous_poly` + `dist_indigenous_poly_km`** | ⏸ Blocked — awaiting RAISG download | Resguardo→national park pathway; small events often overlap indigenous territories. **High impact.** |
| 3 | **WorldClim reduction: 19→8 bands** | Drop b2,b3,b5,b6,b7,b9,b15,b17,b18. Keep b1,b4,b8,b11,b12,b14,b16,b19. | Removes 9 redundant bands (r>0.9 with kept bands). Frees tree capacity. **Medium impact.** |
| 4 | **`year_normalized` as non-rank-normalized feature** | Add year as group-level context (constant within group, varies across groups). Requires excluding year from `_rank_normalize_within_groups`. Lets model shift decision boundary across time. **Medium impact.** |
| 5 | **`dist_redd_km` (revisit)** | Previously dropped on proxy (Recall 24.4%→13.9%) — that was proxy noise. Re-evaluate on full SA. | Unknown on full SA |
| 6 | **`agb_tonne_ha`** (AGB carbon stocks) | Previously dropped (best_iter=5 on proxy). Re-test on full SA with longer patience. TIF at `data/south_america/ready/AGB/agb_sa.tif`. | Unknown on full SA |

**KBA download**: https://www.keybiodiversityareas.org/kba-data/request → `data/shared/KBA/`
**RAISG download**: https://www.raisg.org/ → `data/shared/RAISG/`

---

## Phase 1C — Model Improvements (after 1B, or in parallel)

These are architectural changes that go beyond feature injection:

| Priority | Change | Rationale |
|---|---|---|
| 1 | **Increase earlystop patience 100→150** | Current best_iter=149 hits at the patience limit. May be stopping too early. Low risk. |
| 2 | **Group-size balancing weights** | Weight each (country_id, year) group by `1/log2(n_pos + 2)`. Balances large-event (2001–2009) vs. small-event (2010–2013) gradient contributions. Addresses training distribution mismatch. |
| 3 | **Extended training window 2001–2016** | Include earlystop years in training; use 2017–2019 for early stopping. More recent training examples (post-2014 policy regime). Risk: less earlystop data. |
| 4 | **Full SA retune (100 trials)** | After feature set is locked via 1A+1B, run proper 100-trial Optuna. Previous 20-trial retune found bad params (truncation=84). **Only after feature set is stable.** |
| 5 | **Separate large/small event models** | Train one model on groups with n_pos≥10 and one on n_pos<10. Ensemble predictions. Risk: too few small-event groups in training (~34 in SA). Supervisor directive against ensembles — check first. |

---

## Phase 2 (locked until SA bar confirmed on full test)

- SA full retune (100 trials) with confirmed features + objective — see Phase 1C item 4
- SEA Stage 2 retune on corrected panel
- Bootstrap CIs (Issue Q), model comparison table (Issue GG)
- Spearman ρ, NB robustness, independence check (Issues EE, R, T)

## Phase 3 (after Phase 2)

- Platt calibration → suitability scores (Issue DD)
- Cumulative risk pipeline (Issues G+M)
- NGFS scenario integration (Issue FF)
- Conservation gap: predictions × biodiversity raster → 2×2 map (Issue HH) — Nature Sustainability hook
- **Manuscript gate**: all above before writing

---

## Settled Decisions

- **Engine**: LightGBM LambdaRank W9a, neg_ratio=100
- **Primary metrics**: Lift@1% + Recall@5% within expansion groups (not global AUC)
- **Temporal split**: Train 2001–2013, early-stop 2014–2016, test 2017–2024
- **Graded relevance 1–4**: LambdaRank >> Binary (5.99× vs 3.04×)
- **neg_ratio=100**: Best result on full SA
- **Governance features**: first differences only
- **CBD**: robustness only; CBD-free is primary
- **SEA/USA D²<0**: regime-switching / path-dependency findings, not failures
- **No ensembles, no sub-models** (supervisor directive) — revisit only if bar is unreachable otherwise
- **Simplicity principle**: single Stage 2 model
- **Proxy abandoned**: all evaluation on full SA only (proxy 5× overoptimistic)
- **Full SA retuning with low trial count is harmful**: Optuna needs ≥100 trials; 20-trial run found truncation_level=84 (catastrophic). Only retune after feature set is locked.

---

## Phase 1 — Feature Ablation Log

| Feature / Experiment | Full SA Lift@1% | Full SA Recall@5% | best_iter | Kept |
|---|---|---|---|---|
| baseline (79 features, truncation=741) | 2.85× | 14.0% | 149 | ✓ reference |
| bad retune (truncation=84, 20 trials) | 2.06× | 8.4% | 7 | ✗ reverted |
| 67 features + YEAR_WEIGHT_MIN=0.2 | — | — | — | ⬜ running (job 3561527) |

---

## Spatial Post-Processing (closed)

```
alpha=0.0  Lift@1%=3.33×  Recall@5%=14.8%   ← correct baseline (rank-normalized)
alpha=0.1  Lift@1%=3.36×  Recall@5%=14.8%
alpha=1.0  Lift@1%=3.43×  Recall@5%=14.6%
alpha=2.0  Lift@1%=3.38×  Recall@5%=14.6%
```
No meaningful effect. The model doesn't score "core" pixels well enough for propagation to work. Drop until model quality improves substantially. Script kept at `scripts/regions/south_america/6_evaluation/spatial_postprocess_stage2.py`.

---

## Data Paths

| Dataset | Location |
|---|---|
| SA full splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/` |
| SA mini-sample (67 features after pruning) | `data/south_america/mini_sample.parquet` |
| Baseline best params | `scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json` |
| Baseline best params (Phase 1 archive) | `outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json` |
| Baseline booster | `outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_booster.txt` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` |
| AGB TIF (tried on proxy, dropped — revisit on full SA) | `data/south_america/ready/AGB/agb_sa.tif` |
| REDD TIF (tried on proxy, dropped — revisit on full SA) | `data/south_america/ready/REDD/redd_sa.tif` |

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)` — backbone CRS is LOCAL_CS, `crs.to_epsg()` returns None.

---

## SHAP Audit (2026-06-15)

Top 15 features by mean |SHAP| (baseline model, 79 features):
```
 1. NDVI_smooth64        0.350   (64km-scale vegetation — coarse landscape signal)
 2. GSN_b2              0.283   (biodiversity priority)
 3. WorldClim_b2        0.244   (climate — now dropped post-feature-pruning)
 4. GPW                 0.227   (population density)
 5. deforestation_b2    0.207
 6. WorldClim_b14       0.195
 7. WorldClim_b11       0.187
 8. dist_indigenous     0.185
 9. elevation_b2_smooth4 0.149  (now dropped — near-zero discriminator)
10. GSN_b3              0.148
11. elevation_b1_smooth16 0.137
12. WorldClim_b19       0.128
13. WorldClim_b16       0.126
14. dist_wdpa           0.125
15. HNTL_smooth64       0.108
```
`dist_wdpa` (#14) is already annual (WDPA_prev at t-1). Low SHAP rank is genuine behaviour, not a bug.

Note: SHAP uses baseline 79-feature model. Items 3 and 9 are dropped in the 67-feature model; their SHAP mass will redistribute to correlated features.

SHAP plots: `outputs/south_america/results/phase1/baseline/shap_importance.png`, `shap_beeswarm.png`.

---

## Paused

- SEA Stage 2 — Phase 2
- USA Stage 2 — deprioritised
- Forward pipeline — Phase 3

## Out of Scope (Paper 1)

- Ensemble methods, sub-models, neural networks (Paper 2), survival framing, tropical Africa
