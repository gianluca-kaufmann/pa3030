# PA3030 — Publication Roadmap

**Updated**: 2026-06-14 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

**Story**: The 30×30 agreement forces countries to double protected area coverage by 2030. We predict which pixels will be designated — giving investors, central banks, and policymakers an actionable transition risk tool. Stage 1 predicts *when* countries will expand; Stage 2 predicts *which pixels* will be chosen. Stage 2 output is a **calibrated suitability score**: every pixel gets a value interpretable as annual designation probability.

**Target journal**: Nature Sustainability (primary) → One Earth / GEC → JEEM. If NGFS integration is strong (Issue FF), Nature Finance is an alternative.

---

## Architecture

```
P(pixel i designated in year t)
  = P(country C expands in year t)         ← Stage 1: Poisson GLM, country-year panel
  × S(pixel i | expansion in C, t)         ← Stage 2: LambdaRank → calibrated suitability score
```

**Stage 1** — Poisson GLM with LASSO-selected features (Issue CC). Target: `pa_expansion_pixels`. Metrics: D² OOS (primary) + Spearman ρ country totals (robustness, Issue EE). Train 2001–2016, test 2017–2023 (SA primary 7yr).

**Stage 2** — LightGBM LambdaRank W9a OR XGBoost `rank:ndcg` (decided by Issue BB). Grouped by `(country_id, year)`. Graded relevance 1–4. Output: Platt-scaled suitability score → P(designated | pixel, year) (Issue DD). Metrics: Lift@1% + Recall@5% within expansion groups. Train 2001–2013, early-stop 2014–2016, test 2017–2024.

**Forward output**: Stage 1 budget × Stage 2 suitability → cumulative risk = 1 − ∏(1 − score_t). Scenarios: BAU / 30×30 / NGFS-aligned (Issue FF). Phase 3 only.

**Simplicity principle** (supervisor directive): single model for Stage 2. Add complexity only if bar cannot be met without it.

---

## Current Key Numbers ⚠️ All preliminary

| Region | Stage 1 D² | 95% CI | Stage 2 Lift@1% | Status |
|---|---|---|---|---|
| SA | **+0.345** (7yr PRIMARY, CBD-free, corrected panel) | JK [−0.155, +0.817] SE=0.199 | 5.99× (Colombia dev, LambdaRank W9a) | Phase 0 pending |
| SEA | **−1.001** (6yr PRIMARY, corrected panel) | — | 12.7× (old panel, stale) | Regime-shift finding; Phase 2 |
| USA | −3.14 | — | TBD | Path-dependency finding; deprioritised |

SA naive baseline (dist_wdpa only): 2.81×. W9a impact: LambdaRank 2.99× → **5.99×**. Binary = 3.04× → **LambdaRank confirmed** (Issue W closed 2026-05-29).

**SA Stage 1 spec** (12 features, CBD-free primary, Poisson α=1, p95 winsor, log1p momentum, pre-2010 decay=0.6):
Momentum: lag1/2/3, cumsum_lag1 | Political: v2x_polyarchy, gdp_growth_lag1, redd_plus_enrolled | Governance Δ: Δv2xlg_legcon, Δv2csprtcpt, Δv2xlg_legcon_lag1 | Interaction: legcon_x_cspart | Land: agricultural_land_pct

**SEA regime-shift finding**: KHM 226→26,691→0 px/yr; LAO 0→9,898 spike; MYS 27K train→0 test. One-off political decisions dominate. Same as USA D²<0 — defines limits of predictability from momentum. Paper contribution, not a model failure.

---

## Publication Bar

| Metric | Bar | Current | Notes |
|---|---|---|---|
| Recall@5% | ≥ 90% | ~13% (Colombia dev, 8 test groups) | Primary bar — supervisors confirmed achievable |
| Lift@1% | ≥ 15× | 5.99× | ~2.5× gap — feature problem, not tuning |
| Lift@1% 95% CI | must exclude 1× | Not computed | Bootstrap after final model |

**Why 6× is not enough**: Lift@1% = 6× → Recall@1% ≈ 5–18%. Reviewer will ask "you miss 80–95% of designations." The 90% Recall@5% answer is: "our top 5% slice captures 90% of future designations."

**Most likely unlock**: KBA features — IUCN's formal "should be protected" list encodes conservation intent directly. First feature to add in Phase 1.

---

## Mini-Sample Strategy (iteration engine for Phase 1)

The full SA panel is 57 GB. Every Euler job is 6h. The Phase 1 feature sprint (9 features × tune+retrain) would take months on Euler alone. The mini-sample collapses this to local, same-day experiments.

**Design:**
- Sample ~3–5M rows from the SA training set (enough for stable Lift@1% per group)
- Preserve natural neg/pos ratio (~0.3–0.5%) — no artificial resampling
- Stratify by `(country_id, year)` so all 156 training groups are represented
- Keep the SA test set 100% untouched — always evaluate on full SA test

**Constraint**: LambdaRank needs enough positives per group. At 3–5M rows × 0.4% positive = ~12K–20K positives across ~156 groups ≈ 75–130 positives per group. Workable.

**Validation gate (do once before trusting mini-sample)**: Train on mini-sample → eval on full SA test. If Lift@1% rank order across configs correlates with full-SA results → mini-sample is valid for all subsequent experiments.

**Workflow:**
```
Euler (one-time):   build mini-sample → sync to Desktop
Local (daily):      add feature / tune / retrain → check Lift@1% on full SA test
Weekly gate:        if mini-sample shows improvement → run full SA retune on Euler
Continental:        only once SA bar is confirmed
```

Local retrain at 3–5M rows: seconds. 30-trial Optuna: ~5–10 min. Feature experiments become same-day.

---

## Issues Tracker

### Closed ✅
| ID | Issue | Resolution |
|---|---|---|
| F | WDPA label quality | Audited 2026-05-28: SA <5%, USA <0.15%; SEA corrected TIFs on SCRATCH |
| O | Binary SPW bug | `_scan_true_class_counts()` pre-scan; neg_ratio=100 for binary jobs |
| D | LambdaRank 9K ceiling | W9a: eco-stratified training groups; NDCG eval on full cy groups via `_TrueNdcg1PctEarlyStop` |
| N | Early stopping wrong metric | `_TrueNdcg1PctEarlyStop` callback; applied to training + Optuna |
| H | All-region panel rebuild | SA+SEA+USA merge+FE all complete |
| V | CBD in primary spec | CBD-free is primary (D²=+0.345); CBD-inclusive = robustness (D²=+0.321) |
| J | SEA Stage 1 parsimony | 10-feat parsimonious spec selected |
| P | Comparison metric for W8 | Lift@1% used; LambdaRank selected over binary |
| W | Graded vs binary labels | LambdaRank 5.99× >> Binary 3.04× — graded labels confirmed 2026-05-29 |
| CC | Stage 1 LASSO feature selection | LOYO-CV alpha=100. 9-feat selected (D²_7yr=0.346): lag1, cumsum_lag1, polyarchy, gdp_lag1, redd, d_csprtcpt, agri_land, d_legcon_lag1, cbd. Zeroed: lag2, lag3, d_legcon, legcon_x_cspart. 9-feat marginally better than 12-feat (0.346 vs 0.345). CBD survival disregarded (2 training instances). Interaction term dropped by LASSO but retained in primary spec on theoretical grounds. |

### Open — Phase 0 (do before any feature work)
| ID | Issue | Action |
|---|---|---|
| ~~**CC**~~ | Stage 1 LASSO feature selection | ✅ Done 2026-06-14. See `outputs/south_america/results/stage1_lasso.json`. |
| **BB** | XGBoost `rank:ndcg` vs LambdaRank W9a | ⚠️ First training (job 3354097) invalid: early-stop set had only 3 Colombia expansion groups → NDCG noisy → stopped at iter 4 → reported Lift@1%=2.33× with 4 trees. Bug fixed in `stage2_xgb_core.py` (falls back to fixed-iter when <5 expansion groups). Retraining submitted job 3391479 with full 739 iterations. Await new Lift@1% to compare vs LambdaRank 5.99×. |
| **X** | neg_ratio sensitivity | neg_ratio=200 ✅ NDCG@1%=0.1275. neg_ratio=full ✅ NDCG@1%=0.1061 (worst). Baseline neg_ratio=100 NDCG@1%=0.1070. Winner: neg_ratio=200 on tuning metric. Submit LGBM training with neg_ratio=200 after BB engine decision. |
| **MS** | Build SA mini-sample | ✅ Built. Validation gate: first run (job 3356942) OOM killed (32 GB → not enough for full SA test). Fixed SLURM to 16×8G=128 GB. Resubmitted job 3391648. |

→ After Phase 0: engine (XGBoost or LightGBM) + neg_ratio locked. Mini-sample validated. No further architecture changes in Phase 1.

### Open — Stage 2 Quality (Phase 1–2)
| ID | Issue | Action |
|---|---|---|
| Q | No CIs on Stage 2 metrics | Bootstrap over (country_id, year) groups after final model. Phase 2. |
| S | W9a train/eval group mismatch | If BB→XGBoost: dissolved. If LightGBM retained: Methods paragraph + SHAP-by-biome. |
| Y | Macro Lift@1% weights groups equally | Compute area-weighted Lift@1% alongside macro. Phase 2. |
| Z | 2024 test-set labels incomplete | Exclude 2024 from primary metrics or report delta. Phase 2. |
| AA | Spatial autocorrelation | Moran's I on scored test parquet. Phase 2. |
| GG | No model comparison table | LR / RF / Binary LGBM / LambdaRank on SA (same features). Methods Table 1. Phase 2. |

### Open — Stage 1 Quality (Phase 0–2)
| ID | Issue | Action |
|---|---|---|
| R | Poisson over-dispersion | NB robustness spec on Euler. If NB D² ≈ Poisson D²: document. Phase 2. |
| EE | Stage 1 CI wide | Add Spearman ρ of country-level totals (2017–23, N=12 countries). Phase 2. |
| T | Independence assumption | Correlate Stage 1 residuals with Stage 2 mean pixel chars. State in Methods. Phase 2. |

### Open — Forward Pipeline (Phase 3)
| ID | Issue | Action |
|---|---|---|
| DD | Stage 2 output not calibrated | Platt-scale LambdaRank/XGBoost scores → calibrated P(designated). |
| G+M | Forward pipeline incomplete | Cumulative risk = 1 − ∏(1 − score_t) over 2025–2029. Requires DD. |
| FF | NGFS integration | Map IIASA scenarios to Stage 1 budget multipliers. Net Zero 2050 → 30×30; Current Policies → BAU. |
| HH | No conservation gap analysis | Cross Stage 2 predictions with biodiversity raster → 2×2 quadrant map. Nature Sustainability hook. |

---

## Phase 1 Feature Sprint (after Phase 0 settled)

**Workflow per feature**: rasterise → add to SA mini-sample → 30-trial Optuna retune locally → retrain → check Lift@1% + Recall@5% on full SA test. If promising → validate on full SA on Euler. Track in `outputs/south_america/results/feature_ablation_sa.json`. Keep if SHAP rank top-5 AND Lift@1% improves.

| Priority | Feature | Source | Notes |
|---|---|---|---|
| 1 | `is_kba`, `dist_kba` | BirdLife International (free shapefile) | IUCN formal "should be protected" list. Direct intent signal. Most likely bar-clearing feature. |
| 2 | `in_redd`, `dist_redd` | Verra registry (public shapefiles) | Carbon credit projects tie designation to revenue. |
| 3 | `in_indigenous`, `dist_indigenous_poly` | RAISG AMAZONAS (public shapefile) | Many Colombia PAs are resguardos → PNNs. Separate from existing point-based `dist_indigenous`. |
| 4 | `agb_tonne_ha` | ESA CCI Biomass v4 (300m, free) | Carbon stocks. High-biomass pixels are REDD+ targets. |
| 5 | `pa_connectivity_gap` | Derived from WDPA raster | Binary: pixel bridges two PA clusters within 5km. |
| 6 | `in_runap_proposal` | datos.gov.co (Colombia SIAC/PNN) | Government expansion pipeline — literal designation intent. Colombia-specific. |
| 7 | `in_priority_watershed` | datos.gov.co (IDEAM) | Hydrological corridors in national policy targets. |
| 8 | `dist_deforestation_frontier` | Hansen GFC tree-cover loss (lag 1–3yr) | Pixels adjacent to recent deforestation near PAs. |
| 9 | `ecoregion_protection_gap` | Derived from WDPA + GSN | Under-protected ecoregions get faster designation. |

---

## Next Actions (ordered)

### Phase 0 — Engine + Mini-Sample (in progress)

1. ~~**Issue CC**~~ ✅ Done 2026-06-14.

2. **Issue BB** — XGBoost tuning ✅. First training invalid (early-stop bug, 4 trees). Fixed + resubmitted (job 3391479, 739 trees). Once done: compare Lift@1% vs LambdaRank 5.99× → lock engine.

3. **Issue X** — All ratios done: neg_full=0.1061 (worst), baseline=0.1070, neg200=0.1275 (best). Once BB engine locked: submit LGBM training with neg_ratio=200 → confirm Lift@1% on test.

4. **Issue MS** — mini-sample ✅ built. Validation gate resubmitted (job 3391648, 128 GB). Once done: check Lift@1% > 2.81× (naive baseline) → gate passes → sync to Desktop.

→ Outcome: engine locked + neg_ratio locked + mini-sample validated. Ready for Phase 1.

### Phase 1 — Feature Sprint (local, daily iterations)

5. **KBA features**: Download BirdLife shapefile → rasterise → add to mini-sample → retune locally → validate on full SA if Lift@1% improves. Highest expected impact.
6. **REDD+**: Verra Colombia shapefiles → `in_redd` + `dist_redd`.
7. **Indigenous territories**: RAISG shapefile → `in_indigenous` + `dist_indigenous_poly`.
8. **ESA CCI Biomass**: 300m → resample 1km → `agb_tonne_ha`.
9. **PA connectivity gap**: Derived from WDPA.
10. **Bar check**: After Tier 1 features → measure Recall@5% + Lift@1% on full SA test. If bar met (Recall@5% ≥ 90%, Lift@1% ≥ 15×) → Phase 2. If not → Tier 2 features (priority 6–9).
11. **Ablation study**: Remove feature groups one at a time. Required for Methods section.

### Phase 2 — Continental Scale-Up (locked until SA bar confirmed)

12. SA full retune + retrain (100 trials, all confirmed features, chosen engine).
13. SEA Stage 2 retune + retrain on corrected panel.
14. Issues Q (bootstrap CIs), EE (Spearman ρ), R (NB robustness), T (independence), S (W9a justification if retained), Y+Z+AA.
15. Issue GG: model comparison table.

### Phase 3 — Forward Pipeline + Paper

16. Issue DD: Platt calibration → suitability scores.
17. Issues G+M: cumulative risk pipeline.
18. Issue FF: NGFS scenario integration.
19. Issue HH: conservation gap analysis (2×2 map).
20. **Manuscript gate**: SA/SEA bar confirmed + all Phase 2 issues resolved + ablation done + DD/G+M/HH done. Do not start writing before all gates.

---

## Data Paths

| Dataset | Location |
|---|---|
| SA merged_panel_final.parquet (57 GB) | `euler:$SCRATCH/data/south_america/ml/merged_panel_final.parquet` |
| SA Stage 2 splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/{train,earlystop,test}.parquet` |
| SA mini-sample (~200 MB) | `euler:$SCRATCH/data/south_america/ml/mini_sample.parquet` → sync to `data/south_america/mini_sample.parquet` |
| Colombia Stage 2 panel (3.9 GB) | `euler:$SCRATCH/data/dev/south_america/ml/main/{train,earlystop,test}.parquet` |
| Stage 1 panels (~35 KB each) | `data/{south_america,se_asia,usa}/stage1_panel.parquet` (in repo) |
| Stage 2 eco raster | `$SCRATCH/data/south_america/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` (create) |
| Stage 1 LASSO output | `outputs/south_america/results/stage1_lasso.json` (create) |

---

## Dev Environment

**SLURM scripts** (`slurm/south_america/`):
- `dev_colombia_panel.slurm` — create Colombia splits from SA splits
- `tuning_lgbm_stage2_colombia.slurm` — LambdaRank 30 trials, 6h, 32 GB
- `training_lgbm_stage2_colombia.slurm` — LambdaRank training, 3h, 32 GB
- `tuning_lgbm_stage2_binary_colombia.slurm` — binary / GG comparison
- `tuning_xgboost_stage2_colombia.slurm` — XGBoost rank:ndcg tuning, Issue BB
- `training_xgboost_stage2_colombia.slurm` — XGBoost training + test eval, Issue BB
- `tuning_lgbm_stage2_neg200_colombia.slurm` — LambdaRank neg_ratio=200, Issue X
- `tuning_lgbm_stage2_neg_full_colombia.slurm` — LambdaRank neg_ratio=full, Issue X
- `build_mini_sample.slurm` — SA mini-sample, Issue MS

**Key env vars**: `STAGE2_DATA_ROOT`, `STAGE2_OUTPUT_DIR`, `STAGE2_ECO_GROUPS=1` (W9a), `STAGE2_NEG_RATIO` (100/200/full).

**Old legacy pipeline** (`south_america/colombia/`): delete — fully superseded.

---

## Settled Decisions (Do Not Revisit)

- **Architecture**: Two-stage. Single-model AUC is proven leakage: Group A AUC=0.9994 → Group B AUC=0.5587.
- **Primary metrics**: Lift@1% + Recall@5% within expansion groups (Stage 2); D² OOS (Stage 1). Not global AUC.
- **LambdaRank > Binary**: Colombia 5.99× vs 3.04×. Confirmed 2026-05-29 (Issue W closed).
- **Graded relevance 1–4**: Labels used as-is. Binary labels discarded.
- **Temporal split**: Train 2001–2016, test 2017–2024. Do not change.
- **WDPA reporting lag**: SA 2024 excluded (~33% capture). SA primary = 7yr (2017–23); SEA = 6yr (2017–22).
- **Governance features**: First differences only — levels cause temporal drift in forward predictions.
- **No pre-2001 training data**: Frontier-exhaustion regime mismatch.
- **CBD**: Robustness check only. CBD-free is primary.
- **Suitability score framing**: Stage 2 output = calibrated 0–1 probability throughout the paper.
- **SEA/USA D²<0 = regime-switching finding**: Defines limits of predictability, not a model failure.
- **SEA saturation leakage**: Do not re-investigate.
- **Simplicity principle**: Single Stage 2 model. No ensembles or sub-models for Paper 1.
- **Regions**: SA primary; SEA + USA as robustness/contrast. Tropical Africa: out of scope.
- **Development order**: SA mini-sample → SA full → SEA → continental.

---

## Paused (until SA bar confirmed)

- SA continental Stage 2 retune — do not submit
- SEA Stage 2 retune — old 12.7× on old panel; Phase 2
- USA Stage 2 — deprioritised
- Forward pipeline (G+M, DD, FF) — Phase 3
- NGFS integration — Phase 3

## Out of Scope (Paper 1)

- KBA-conditioned sub-models, ensemble methods
- Neural networks / deep learning (Paper 2)
- Survival / discrete-time hazard framing
- Random effects / mixed Poisson for Stage 1
- Sub-national Stage 1, tropical Africa, marine PAs
- Embeddings / Paper 2 (gate: Paper 1 submitted)
