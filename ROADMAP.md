# PA3030 — Publication Roadmap

**Updated**: 2026-05-29 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

**Story**: The 30×30 agreement forces countries to double protected area coverage by 2030. We predict which pixels will be designated — giving investors, central banks, and policymakers an actionable transition risk tool. Stage 1 predicts *when* countries will expand; Stage 2 predicts *which pixels* will be chosen. Stage 2 output is a **calibrated suitability score** (not a raw rank): every pixel gets a value interpretable as annual designation probability.

**Target journal**: Nature Sustainability (primary) → One Earth / GEC → JEEM. If NGFS integration is strong (Issue FF), Nature Finance is an alternative first submission.

---

## Architecture

```
P(pixel i designated in year t)
  = P(country C expands in year t)         ← Stage 1: Poisson GLM, country-year panel
  × S(pixel i | expansion in C, t)         ← Stage 2: LambdaRank/XGBoost → calibrated suitability score
```

**Stage 1** — Poisson GLM with LASSO-selected features (Issue CC). Target: `pa_expansion_pixels`. Metrics: D² OOS (primary) + Spearman rank correlation of country-level totals (robustness, Issue EE). Train 2001–2016, test 2017–2023 (SA primary 7yr).

**Stage 2** — LightGBM LambdaRank W9a OR XGBoost `rank:ndcg` — decided by Issue BB. Grouped by `(country_id, year)` for evaluation. Graded relevance 1–4 (or binary — decided by Issue W). Output: Platt-scaled suitability score → P(designated | pixel, year) (Issue DD). Metrics: Lift@1% + Recall@5% within expansion groups. Train 2001–2013, early-stop 2014–2016, test 2017–2024.

**Forward output** (transition risk product): Stage 1 budget × Stage 2 suitability → cumulative risk per pixel = 1 − ∏(1 − score_t). Scenarios: BAU / 30×30 / NGFS-aligned (Issue FF). Requires Issues G+M+DD.

**Simplicity principle** (supervisor directive): single model for Stage 2 (no ensembles, no sub-models). Add complexity only if the publication bar cannot be met without it. Paper framing must be intuitively understandable.

---

## Current Key Numbers ⚠️ All preliminary

| Region | Stage 1 D² | 95% CI | Stage 2 Lift@1% | Status |
|---|---|---|---|---|
| SA | **+0.345** (7yr PRIMARY, CBD-free, corrected panel) | JK [−0.155, +0.817] SE=0.199 | 5.99× (Colombia dev, LambdaRank W9a) | Stage 2 re-tune paused (Phase 2) |
| SEA | **−1.001** (6yr PRIMARY, corrected panel) | — | 12.7× (old panel, stale) | Regime-shift finding; Stage 2 Phase 2 |
| USA | −3.14 | — | TBD | Path-dependency finding; Stage 2 deprioritised |

SA naive baseline (dist_wdpa only): 2.81×. W9a impact: LambdaRank 2.99× → **5.99×**. Binary = 3.04× → LambdaRank selected.

**SA Stage 1 spec** (12 features, CBD-free primary, Poisson α=1, p95 winsor, log1p momentum, pre-2010 decay=0.6):
Momentum: lag1/2/3, cumsum_lag1 | Political levels: v2x_polyarchy, gdp_growth_lag1, redd_plus_enrolled | Governance Δ: Δv2xlg_legcon, Δv2csprtcpt, Δv2xlg_legcon_lag1 | Interaction: legcon_x_cspart | Land: agricultural_land_pct
CBD robustness: D²=+0.321. Country FE sensitivity: D²=−0.373 (FE collapses OOS → no-FE justified).

**SEA Stage 1 regime-shift finding**: KHM 226→26,691→0 px/yr; LAO 0→9,898 spike; MYS 27K training → 0 test. One-off political decisions dominate — momentum features cannot predict this. Same interpretation as USA D²<0. Neither is a model failure; both define the *limits of predictability* from momentum, which is a paper contribution.

---

## Publication Bar

| Metric | Bar | Current (Colombia) | Notes |
|---|---|---|---|
| Recall@5% | ≥ 90% | Not yet measured | Primary bar — supervisors confirmed achievable |
| Lift@1% | ≥ 15× | 5.99× | ~2.5× gap |
| Lift@1% 95% CI | must exclude 1× | Not computed | Bootstrap after final model |

**Why 6× is not enough**: Lift@1% = 6× at ~1-3% positive rate → Recall@1% ≈ 5–18%. A reviewer will ask "you miss 80–95% of designations — how is this a transition risk tool?" The 90% Recall@5% bar is the answer: "our top 5% slice captures 90% of future designations."

**Most likely unlock**: KBA features. IUCN's Key Biodiversity Area list encodes *conservation intent* — places actively targeted for designation. If ≥60% of Colombia's test-set designations fall inside KBA boundaries, `is_kba` alone may clear the Recall@5% bar. This is the first feature to add after Phase 0 engine decisions.

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
| J | SEA Stage 1 parsimony | 10-feat parsimonious spec selected (D²>0 on old panel; D²=−1.001 on corrected = regime shift) |
| P | Comparison metric for W8 | Lift@1% used (not NDCG); LambdaRank selected over binary |

### Open — Phase 0 (settle engine before feature sprint)
| ID | Issue | Action |
|---|---|---|
| **BB** | XGBoost `rank:ndcg` vs LightGBM W9a | 1 Euler job: tune XGBoost rank:ndcg on Colombia (30 trials, same features). If XGBoost wins: replace LightGBM; Issues S dissolved cleanly. If LightGBM wins: keep W9a. **Do first.** |
| **X** | neg_ratio gradient distortion | Run LambdaRank neg_ratio ∈ {100, 200, full} on Colombia. Full groups (3.9 GB / 32 GB node) likely fit. Run in parallel with BB. |
| **W** | Graded vs binary relevance labels | LambdaRank with binary 0/1 labels vs current graded 1–4. If Lift@1% similar, use binary (simpler, more defensible). Run in parallel with BB+X. |
| **CC** | Stage 1 LASSO feature selection | Run LASSO path (Poisson, L1) on `stage1_panel.parquet` with LOYO-CV alpha selection. Surviving features = primary spec. Replaces manual Issues U+V. Local, ~10 min. **Do today.** |

### Open — Stage 2 Quality (Phase 1–2)
| ID | Issue | Action |
|---|---|---|
| Q | No CIs on Stage 2 metrics | Bootstrap over (country_id, year) expansion groups after final model. Add 95% CI to all metrics. |
| S | W9a train/eval group mismatch | If BB → XGBoost wins: dissolved. If LightGBM retained: write Methods paragraph framing eco-stratification as ecological hypothesis (selection operates within strata); provide SHAP-by-biome evidence. |
| W | (see Phase 0 above) | — |
| Y | Macro-averaged Lift@1% weights small/large groups equally | Compute area-weighted Lift@1% alongside macro. Report both; explain policy-relevant metric. |
| Z | 2024 test-set labels incomplete (~33% WDPA capture) | Exclude 2024 from primary Stage 2 metrics (consistent with Stage 1), or report with/without delta. |
| AA | Spatial autocorrelation in Stage 2 residuals | Moran's I on scored test parquet. If near zero: documented and done. Phase 2. |
| GG | No model comparison in paper | LR / RF / Binary LGBM / LambdaRank on Colombia (same features). Becomes Methods Table 1 — justifies model choice. |

### Open — Stage 1 Quality (Phase 0–2)
| ID | Issue | Action |
|---|---|---|
| CC | (see Phase 0 above) | LASSO feature selection |
| R | Poisson over-dispersion unaddressed | Add NB robustness spec on Euler (statsmodels). If NB D²_7yr ≈ Poisson D²_7yr: document and move on. If materially different: NB becomes primary. |
| EE | Stage 1 metric unstable (8 test years, JK CI wide) | Add Spearman rank correlation of country-level totals over 2017–2023 as a robustness metric. N=12 countries ranked by predicted vs actual total expansion. More stable than year-by-year D². |
| T | Independence assumption never stated or tested | Correlate Stage 1 residuals with Stage 2 mean pixel characteristics. State assumption explicitly in Methods. Phase 2. |

### Open — Reviewer Blockers (resolve before submission)
| ID | Issue | Action |
|---|---|---|
| Q | CIs missing on all metrics | Stage 1: jackknife done. Stage 2: bootstrap after final model (Phase 2). |
| R | Over-dispersion (see above) | Euler, Phase 2. |
| S | W9a justification (see above) | Depends on BB outcome. |
| T | Independence assumption (see above) | Phase 2. |

### Open — Forward Pipeline + Paper Contribution (Phase 3)
| ID | Issue | Action |
|---|---|---|
| G+M | Forward pipeline incomplete | Platt-scale Stage 2 → P(designated); implement cumulative risk = 1 − ∏(1 − score_t). Requires Stage 2 final model. |
| DD | Stage 2 output not calibrated | Platt scaling on LambdaRank/XGBoost scores. Reframe output as suitability score in paper. Enables G+M accumulation. |
| FF | NGFS scenario integration | Map NGFS IIASA land-use scenarios (Net Zero 2050, Current Policies, etc.) to Stage 1 budget multipliers. Replaces/augments custom BAU/moderate/30×30 scenarios. Makes maps compatible with central bank stress tests. |
| HH | No conservation gap analysis | Cross Stage 2 predictions with biodiversity raster → 2×2 (high/low predicted risk × high/low biodiversity). "Where designation will go vs where it should go." ~2h on existing scored outputs. Central Nature Sustainability finding. |

---

## Colombia Feature Sprint (Phase 1 — after engine settled)

For each feature: rasterise to 1km EPSG:3857 → add to Colombia panel → 30-trial Optuna retune → retrain → report Lift@1% + Recall@5% delta. Track in `outputs/south_america/results/feature_ablation_colombia.json`. Keep if SHAP rank top-5 AND Lift@1% improves.

| Priority | Feature | Source | Notes |
|---|---|---|---|
| 1 | `is_kba`, `dist_kba` | BirdLife International (free shapefile) | IUCN's formal "should be protected" list. Direct intent signal. Most likely bar-clearing feature. |
| 2 | `in_redd`, `dist_redd` | Verra registry (public shapefiles) | Financial incentive signal. Carbon credit projects tie designation to revenue. |
| 3 | `in_indigenous`, `dist_indigenous_poly` | RAISG AMAZONAS (public shapefile) | Many Colombia PAs are resguardos → PNNs. Separate from existing point-based `dist_indigenous`. |
| 4 | `agb_tonne_ha` | ESA CCI Biomass v4 (300m, free) or GEDI L4A | Carbon stocks. High-biomass pixels are REDD+ targets. |
| 5 | `pa_connectivity_gap` | Derived from WDPA raster | Binary: pixel bridges two PA clusters within 5km. Simple proxy for network gap designation. |
| 6 | `in_runap_proposal` | datos.gov.co (Colombia SIAC/PNN) | Government expansion pipeline — literal designation intent. Colombia-specific. |
| 7 | `in_priority_watershed` | datos.gov.co (IDEAM) | Hydrological corridors in national policy targets. |
| 8 | `dist_deforestation_frontier` | Hansen GFC tree-cover loss (lag 1–3yr) | Pixels adjacent to recent deforestation near PAs are frequently designated. |
| 9 | `ecoregion_protection_gap` | Derived from WDPA + GSN | Under-protected ecoregions get faster designation. |

---

## Data Paths

| Dataset | Location |
|---|---|
| Colombia Stage 2 panel (3.9 GB) | `euler:$SCRATCH/data/dev/south_america/ml/main/{train,earlystop,test}.parquet` |
| SA merged_panel_final.parquet (57 GB) | `euler:$SCRATCH/data/south_america/ml/merged_panel_final.parquet` |
| Stage 1 panels (~35 KB) | `data/{south_america,se_asia,usa}/stage1_panel.parquet` (in repo) |
| Stage 2 eco raster | `$SCRATCH/data/south_america/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_colombia.json` (create) |
| Stage 1 LASSO output | `outputs/south_america/results/stage1_lasso.json` (create) |

---

## Next Actions

### Phase 0 — Engine Decision (do now, before any feature work)

**All Phase 0 jobs can run in parallel.**

1. **Issue CC — LASSO Stage 1** (local, ~10 min): Run LASSO path (Poisson + L1) on `data/south_america/stage1_panel.parquet` with LOYO-CV alpha selection. LASSO-selected features replace current 12-feature manual spec as primary. Save surviving features to `stage1_lasso.json`. Re-run D²_7yr and JK CI on selected spec.

2. **Issue BB — XGBoost `rank:ndcg` Colombia** (Euler, 6h, 30 trials): Tune on Colombia with same features as current LambdaRank. Compare Lift@1% vs 5.99×. Decision rule: if XGBoost ≥ LambdaRank → switch (simpler, Issue S dissolved); if LambdaRank wins → keep W9a and schedule Issue S write-up.

3. **Issue X — neg_ratio sensitivity** (Euler, ~6h, submit alongside BB): LambdaRank neg_ratio ∈ {100, 200, full}. Full groups should fit (3.9 GB panel, 32 GB node). Best-performing ratio used going forward.

4. **Issue W — Binary vs graded label test** (Euler, ~6h, submit alongside BB+X): LambdaRank with binary 0/1 relevance vs current graded 1–4. If Lift@1% is within 10%, use binary (scientifically simpler; Issue W closed).

→ After Phase 0: commit to (engine: XGBoost or LightGBM) + (neg_ratio) + (label scheme). No further architecture changes in Phase 1.

### Phase 1 — Colombia Feature Sprint (after Phase 0 settled)

5. **KBA features** (Tier 1): Download BirdLife KBA shapefile → rasterise to 1km EPSG:3857 → add `is_kba` + `dist_kba` → 30-trial retune + retrain → report Lift@1% + Recall@5%. Highest expected impact.

6. **REDD+ project boundaries** (Tier 1): Verra registry Colombia shapefiles → `in_redd` + `dist_redd` → retune + retrain.

7. **Indigenous territory polygons** (Tier 1): RAISG shapefile → `in_indigenous` + `dist_indigenous_poly` → retune + retrain.

8. **ESA CCI Biomass** (Tier 1): 300m raster → resample 1km → `agb_tonne_ha` → retune + retrain.

9. **PA connectivity gap** (Tier 1): Derive from WDPA raster → binary bridging indicator → retune + retrain.

10. **Issue GG — Model comparison table** (parallel with feature sprint): LR / RF / Binary LGBM / LambdaRank on Colombia (final feature set). Each model: same features, same train/test split. Becomes Methods Table 1.

11. **Evaluate Colombia bar**: After Tier 1 features, measure Recall@5% and Lift@1% on test 2017–2024. If bar met (Recall@5% ≥ 90%, Lift@1% ≥ 15×) → advance to Phase 2. If not → continue Tier 2 features (priority 6–9 in Feature Sprint table).

12. **Issue W4 — Ablation study**: Remove feature groups one at a time. Measures contribution of each group. Required for paper Methods section.

### Phase 2 — Continental Scale-Up (locked until Colombia bar confirmed)

13. SA full re-tune + retrain (chosen engine, 100 trials, all confirmed features).
14. SEA Stage 2 re-tune + retrain on corrected panel.
15. **Issue Q — Bootstrap CIs**: Bootstrap over (country_id, year) expansion groups for SA + SEA. Add 95% CI to all reported metrics.
16. **Issue EE — Spearman rank test**: Compute Spearman ρ between predicted and actual country-level PA expansion totals over 2017–2023. N=12 countries.
17. **Issue R — NB robustness**: Run on Euler (statsmodels). Compare NB D²_7yr vs Poisson D²_7yr for SA.
18. **Issue T — Independence assumption**: Correlate Stage 1 residuals with Stage 2 mean pixel characteristics. State and test in Methods.
19. **Issue S** (if LightGBM/W9a retained): Write Methods paragraph justifying eco-stratification ecologically; include SHAP-by-biome figure.
20. **Issues Y + Z + AA**: Area-weighted Lift@1%, 2024 exclusion test, Moran's I on Stage 2 residuals.

### Phase 3 — Forward Pipeline + Paper (after Phase 2 complete)

21. **Issue DD — Suitability score / Platt calibration**: Platt-scale Stage 2 scores → calibrated P(designated | pixel, year). Rewrite paper and all outputs to use "suitability score" framing throughout.
22. **Issues G+M — Forward pipeline**: Cumulative risk = 1 − ∏(1 − score_t) over 2025–2029.
23. **Issue FF — NGFS scenario integration**: Map NGFS IIASA scenarios to Stage 1 budget multipliers. Download from https://data.ene.iiasa.ac.at/ngfs/#/downloads. Link Net Zero 2050 → 30×30 scenario; Current Policies → BAU.
24. **Issue HH — Conservation gap analysis**: Cross-tabulate Stage 2 risk map with a biodiversity raster (e.g. KBA density, IUCN species richness). Produce 2×2 quadrant map. "Where designation will go vs where it is most needed" is the Nature Sustainability hook.
25. **REDD+ URL verification**: Open all `# TO VERIFY` entries in `build_redd_plus.py` before submission.
26. **Manuscript gate**: Colombia bar confirmed + SA/SEA Stage 2 final + Issues Q/R/S/T/W/GG all resolved + ablation complete + DD/G+M/HH done. Do not start writing before all gates.

---

## Dev Environment (Colombia)

**SLURM scripts** (`slurm/south_america/`):
- `dev_colombia_panel.slurm` — create Colombia splits from SA splits
- `tuning_lgbm_stage2_colombia.slurm` — LambdaRank 30 trials, 6h, 32 GB
- `training_lgbm_stage2_colombia.slurm` — LambdaRank training, 3h, 32 GB
- `tuning_lgbm_stage2_binary_colombia.slurm` — binary / GG comparison
- `training_lgbm_stage2_binary_colombia.slurm` — binary training

**Key env vars**: `STAGE2_DATA_ROOT`, `STAGE2_OUTPUT_DIR`, `STAGE2_ECO_GROUPS=1` (W9a, LambdaRank only), `STAGE2_NEG_RATIO` (100/200/full).

**Old legacy pipeline** (`south_america/colombia/`): delete — fully superseded.

---

## Settled Decisions (Do Not Revisit)

- **Architecture**: Two-stage. Single-model AUC is proven leakage: Group A AUC=0.9994 → Group B AUC=0.5587.
- **Primary metrics**: Lift@1% + Recall@5% within expansion groups (Stage 2); D² OOS (Stage 1). Not global AUC.
- **Temporal split**: Train 2001–2016, test 2017–2024. Do not change.
- **WDPA reporting lag**: SA 2024 excluded from primary (~33% capture). SA primary = 7yr (2017–23); SEA primary = 6yr (2017–22).
- **Governance features**: First differences only — level variables cause temporal drift in forward predictions.
- **Pre-2001 lags**: Terrestrial-only; marine contamination fix applied (ECU Galápagos).
- **No pre-2001 training data**: Frontier-exhaustion regime mismatch.
- **30×30 as exogenous budget**: Post-Stage-1 multiplier, not a model coefficient.
- **CBD**: Robustness check only. CBD-free is primary (2-instance coefficient indefensible).
- **LambdaRank > Binary classifier**: Colombia 5.99× vs 3.04×. LambdaRank is the training objective regardless of suitability-score output framing. These are separable: training = ranking loss; output = calibrated probability.
- **Suitability score framing**: Stage 2 output is presented as calibrated suitability score throughout the paper. The ranking objective is an implementation detail; the output is a 0–1 probability.
- **SEA D²<0 = regime-switching finding**: KHM/LAO sporadic designations are structurally unpredictable from momentum. Not a failure — defines limits of predictability. USA D²<0 = same interpretation.
- **SEA saturation leakage**: Do not re-investigate. sat_clean gain was pure future-info leakage.
- **Simplicity principle**: Single Stage 2 model. No ensembles, no KBA-conditioned sub-models for Paper 1. Add complexity only if bar cannot be met without it.
- **Regions**: SA primary; SEA + USA as robustness/contrast. Tropical Africa: out of scope.
- **Development**: Colombia first. Continental locked until Colombia Recall@5% ≥ 90%.

---

## Paused (until Colombia bar met)

- SA Stage 2 continental re-tune (`tuning_lgbm_stage2.slurm`) — do not submit
- SEA Stage 2 continental re-tune — old 12.7× is on old panel; will rerun in Phase 2
- USA Stage 2 — path-dependency finding reduces urgency; deprioritised
- Forward pipeline (G+M, DD, FF) — depends on final Stage 2 model
- NGFS integration (FF) — Phase 3
- Moran's I (AA) — Phase 2

## Out of Scope (Paper 1)

- KBA-conditioned sub-models (complexity — test only if bar unreachable without them)
- Ensemble methods (complexity)
- Neural networks / deep learning (Paper 2, requires AlphaEarth access)
- Survival / discrete-time hazard framing
- Random effects / mixed Poisson for Stage 1 (note as future extension only)
- Sub-national Stage 1
- Tropical Africa, marine PAs
- Single-model global AUC as primary metric
- Embeddings / Paper 2 (gate: Paper 1 submitted)
