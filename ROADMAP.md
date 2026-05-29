# PA3030 — Paper Publication Roadmap

**Updated**: 2026-05-29 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

**Story**: The 30×30 biodiversity agreement is coming. We predict which land will be designated as protected area — giving investors and central banks an actionable transition risk tool for portfolio exposure to PA designation events.

**Target journal**: One Earth / GEC → Nature Sustainability → JEEM

---

## STRATEGY: Colombia-First Until Publication Bar Met

**ALL development happens on Colombia (~7M pixel-years) until the model hits the publication bar.
No continental re-tunes, no SA/SEA/USA reruns, no new regions — until Colombia is done.**

Rationale: 6× Lift@1% is not publishable in a tier-A journal. A reviewer will ask why they should believe the model can identify where 30×30 expansion will actually land. We need a result that speaks for itself. Continental runs take 2 days per iteration; Colombia runs in hours. Every architectural decision, every new feature, every issue resolution gets proved on Colombia first, then replicated.

### Publication Bar (non-negotiable before continental scale)

**Primary target: Recall@5% ≥ 90% within Colombia expansion groups (test years 2017–2024)**

Operational meaning: Take the top 5% of ranked pixels within each Colombia expansion country-year. That slice must contain ≥90% of all pixels that actually became protected areas. This is the claim that matters for a transition-risk tool — *"our model identifies a small area of land that captures 90% of future PA designations."*

Supporting thresholds (all must hold before scale-up):
- Lift@1% ≥ 15× (currently 5.99× — must roughly 2.5× improve)
- Recall@5% ≥ 90% (primary new bar)
- Recall@1% ≥ 50% (stretch: capturing half of all PAs in top 1% of ranked pixels)
- Bootstrap 95% CI on Lift@1% must not include 1× (currently wide — CI must tighten)

These numbers define "done." If we cannot reach them on Colombia's 7M rows with the right features, the model cannot be published in a tier-A journal.

### Why 6× Lift Is Not Enough

6× Lift@1% means: in the top 1% of ranked pixels (within expansion groups), we capture ~6× more PAs than random. At ~1-3% positive rate within expansion groups, this translates to Precision@1% ≈ 6–18%. Recall@1% ≈ 5–18% of all PAs. A reviewer will ask: *"You miss 80–95% of actual designations — how is this a transition risk tool?"* The answer must be a much stronger recall at a policy-relevant threshold.

### Data Paths (Local Sync Reference)

- **Colombia Stage 2 panel** (3.9 GB, syncable): `euler:/cluster/scratch/gikaufmann/data/dev/south_america/ml/main/{train,earlystop,test}.parquet`
- **SA merged_panel_final.parquet** (57 GB, Euler-only): `/cluster/scratch/gikaufmann/data/south_america/ml/merged_panel_final.parquet`
- **Stage 1 panel** (35 KB, already in repo): `data/south_america/stage1_panel.parquet` → run `model1_expansion.py` locally to verify D² numbers without syncing the 57 GB file. If you distrust the expansion counts themselves, sync the 57 GB file and re-run `stage1_data_builder.py` first.

---

## Architecture

```
P(pixel i designated in year t)
  = P(country C expands PAs in year t)       ← Stage 1: political timing  [country-year features]
  × P(pixel i chosen | expansion in C, t)    ← Stage 2: geographic selection  [pixel features]
```

Single-model AUC is a proven leakage artefact: Group A (label overlap) AUC=0.9994 → Group B (genuinely unseen) AUC=0.5587. Two-stage is the correct architecture.

**Stage 1** — Poisson GLM, country-year panel. Target: `pa_expansion_pixels`. Metric: D² OOS.
Train 2001–2016 | Test 2017–2024 (SA primary: 2017–2023; SEA primary: 2017–2022 — WDPA reporting lag confirmed).

**Stage 2** — LightGBM, grouped by `(country_id, year)`, trained on expansion country-years only. Within-group percentile rank normalisation. Graded relevance 1–4 by BFS cluster size. Metric: **Lift@1%** within expansion groups (primary paper metric).
Train 2001–2013 | Early-stop 2014–2016 | Test 2017–2024.

**Forward output** (transition risk): Stage 1 budget (BAU / moderate / 30×30 scenarios) × Stage 2 calibrated P(designated) → cumulative risk per pixel = 1 − ∏_{t=2025}^{2029}(1 − stage1_share_t × p_i). Requires Issues G+M resolved.

---

## Current Key Numbers ⚠️ SA Stage 1 updated on corrected panel (2026-05-29); Stage 2 re-tune in progress

| Region | Stage 1 D² | Stage 2 Lift@1% | Status |
|---|---|---|---|
| SA | **+0.345** (7yr PRIMARY 2017–23, CBD-free, corrected panel) / +0.448 (3yr) / +0.309 (8yr). JK CI: [−0.155, +0.817] (SE=0.199). CBD robustness D²=+0.334. Country-FE sensitivity D²=−0.373 (FE collapses OOS → no-FE justified). | 5.99× LambdaRank (Colombia dev, W9a, Issue N+D fix). Full SA re-tune pending. | LambdaRank selected; submit SA re-tune 2026-05-29 |
| SEA | **−1.001** (6yr PRIMARY 2017–22, corrected panel) ⚠️ FINDING: regime shift (VNM/MMR dominate training → KHM/LAO dominate test). KHM: 226→26,691→0 px/yr. Old +0.290 was on contaminated labels. Paper: SEA Stage 1 is path-dependent (like USA). | **12.7×** LambdaRank (old panel) — update after FE re-run | SEA Stage 2 re-tune pending; Stage 1 = path-dependency finding |
| USA | −3.14 (8yr, trend-only) | TBD | Path-dependency finding; Stage 2 deprioritised |

SA naive baseline (dist_wdpa only): 2.81× on 8yr test (2017–24).

**W9a impact on Colombia dev panel**: LambdaRank 2.99× (no eco) → 5.99× (W9a eco groups). Binary 4.38× (no W9a) → 3.04× (W9a irrelevant for binary). LambdaRank with W9a is the decisive winner.

**SEA Stage 1 regime-shift finding**: KHM expansion 226 px (2015) → 26,691 px (2016) → near-zero by 2018. LAO: ~0 in training → 9,898 px (2019) spike. MYS: 27K training → 0 test. These one-off political decisions are structurally unpredictable from momentum features, explaining D²<0.

---

## Colombia Feature Sprint — Path to 90%+ Recall

**Current features (~60) are insufficient.** 6× Lift@1% tells us the model captures geographic patterns of past protection but cannot reliably identify the specific parcels that will be designated next. We need features that encode *intent* and *incentive* — not just geography. Below is the priority order for new Colombia-specific features and architecture experiments.

### Tier 1 — Highest Expected Impact (run first, all on Colombia)

1. **Key Biodiversity Areas (KBAs)** — IUCN's formal list of sites that qualify for PA designation. These are the places conservation organisations are actively trying to protect. `is_kba` binary + `dist_kba`. Source: BirdLife International KBA shapefile (free download). Expected: top-1 SHAP on Colombia.

2. **REDD+ project boundaries (Verra / Gold Standard)** — Active carbon credit projects tied to avoided deforestation. These are precise polygon boundaries from the Verra registry (public). Designation of these pixels is financially incentivised. `dist_redd_project`, `in_redd_project`. Source: Verra API / Verra project search for Colombia.

3. **Indigenous territory boundaries (RAISG)** — RAISG AMAZONAS has polygon boundaries for all formally recognised and proposed indigenous territories in SA. Many of Colombia's new PAs are indigenous resguardos converted to PNNs. `dist_indigenous_territory`, `in_indigenous_territory`. Source: RAISG (already partially available as `dist_indigenous` but polygon overlap is missing).

4. **ESA CCI Biomass / GEDI AGBD (carbon stocks)** — Above-ground biomass in tonnes/ha. High-carbon-stock pixels are financially valuable to protect via REDD+/VCM. Already in Data Sprint. Source: ESA Climate Office (CCI Biomass v4 at 300m) or GEDI L4A.

5. **PA network connectivity gap score** — For each unprotected pixel, compute the increase in PA network connectivity if it were designated (graph distance reduction in PA network). Pixels that would "bridge" PA clusters are disproportionately targeted. Simple version: count of unprotected pixels connecting two PA clusters within 5km. Source: derived from WDPA spatial index.

### Tier 2 — High Impact (Colombia after Tier 1 proven)

6. **Colombia RUNAP / PNN expansion proposals** — Colombia's Registro Único Nacional de Áreas Protegidas and Parques Nacionales Naturales have formal expansion boundaries (buffer zones, corridor plans). These are the literal government pipeline. Source: SIAC/IDEAM open data portal (datos.gov.co).

7. **Deforestation threat proximity** — Distance to active deforestation frontier (Hansen GFC tree-cover loss, most recent 3 years). Pixels adjacent to deforestation fronts near existing PAs are frequently designated to "protect" the edge. Source: Hansen GFC (already partially used but not as a lag feature).

8. **Watershed / water tower importance** — Colombia's Estrategia Nacional de Cuencas prioritises hydrological corridors. `in_priority_watershed` binary from IDEAM. Source: IDEAM open GIS.

9. **Ecoregion protection gap** — Within each WWF ecoregion, fraction of area already protected. Underprotected ecoregions that intersect with existing policy targets get faster designation. `ecoregion_protection_gap` = 1 - current_pct_protected.

10. **IUCN species richness / endemism** — Number of threatened species with confirmed occurrence in each pixel. Source: IUCN Red List spatial data (requires licence; GBIF as proxy).

### Tier 3 — Architecture Experiments (after features exhausted)

11. **neg_ratio=full for Colombia training** — Issue X: with Colombia panel at 3.9 GB, LambdaRank may be trainable on full groups without neg_ratio subsampling. Test neg_ratio ∈ {100, 200, full}. Gradient distortion at neg_ratio=100 may be masking 2–3× lift improvement.

12. **Survival model framing** — Replace binary per-year label with time-to-designation. Use Cox proportional hazards or discrete-time hazard. This collapses the label noise from "unprotected in year t but will be in t+3" into a proper right-censored outcome. LightGBM supports Cox via `objective=mape` on survival — or use scikit-survival.

13. **Spatial lag features** — For each pixel, compute fraction of pixels within 1km, 5km, 10km rings that became PA in the prior year. This is a rolling spatial contagion signal that the current `spatial_smoothing` feature approximates but doesn't capture at multiple scales.

14. **XGBoost rank:ndcg (no 9K ceiling)** — LightGBM's 9K query ceiling forced the W9a eco-group workaround (Issue D+S). XGBoost `rank:ndcg` has no ceiling; training on full `(country_id, year)` groups gives true gradient signals. If Lift@1% improves on Colombia, this resolves Issue S scientifically.

### Feature Engineering Process

For each Tier 1–2 feature on Colombia:
1. Extract raster → resample to 1km EPSG:3857 → add column to Colombia panel
2. Re-run Stage 2 tuning (30 trials) → retrain → report Lift@1% and Recall@5%
3. SHAP importance check: does the new feature rank top-5?
4. If yes → keep. If no → document and move on.

Track progress in a `feature_ablation_colombia.json` in `outputs/south_america/results/`.

---

## Stage 1 Current Specs

**SA — 12 features (CBD-free primary, Issue V), Poisson α=1, p95 winsor, log1p momentum, pre-2010 decay=0.6**
Momentum (log1p): lag1/2/3, cumsum_lag1 | Political levels: v2x_polyarchy, gdp_growth_lag1, redd_plus_enrolled | First differences: Δv2xlg_legcon, Δv2csprtcpt, Δv2xlg_legcon_lag1 | Interaction: legcon_x_cspart | Land constraint: agricultural_land_pct
Corrected panel (2026-05-29): D²_7yr=**+0.345** PRIMARY / D²_3yr=+0.448 / D²_8yr=+0.309.
Jackknife 7yr CI: [−0.155, +0.817] (SE=0.199) — wide CI reflects genuine year-to-year variability.
CBD robustness D²=+0.334 (CBD-free marginally better). No-interact D²=+0.347 (≈same — interaction marginal on new panel).
Country FE sensitivity D²=−0.373 (FE collapses OOS; no-FE justified for cross-regional transfer).
NB robustness: statsmodels not installed in venv (run on Euler when available).

**SEA — 10 features (same momentum + v2x_polyarchy + Δgov), Poisson α=1, no decay**
Corrected panel (2026-05-29): D²_6yr=**−1.001** PRIMARY ⚠️. Regime shift finding (see key numbers above).
Old +0.290 was on contaminated labels (4.2–5.9% non-Designated in training years 2000–2015).

**USA — 4 momentum features, α=10, trend-only**
Negative OOS D² = political path-dependency finding (Obama-era expansion pattern collapses under Trump).

---

## Open Issues (Priority Order)

### CRITICAL — Blocking paper or results validity

**F — WDPA label quality** ✅ **Fully closed 2026-05-28 — all three regions audited, labels acceptable**
GEE export confirmed to have no `STATUS='Designated'` filter (paints all WDPA statuses). Full shapefile audit run against WDPA May2026 (job 1076907, completed 13:07):
- **USA**: 0.12–0.14% contamination across all years. Excellent — well under 5% threshold.
- **SA**: 1.8–2.7% contamination across all years. Under 5% threshold — labels acceptable.
- **SEA**: 4.2–5.9% contamination. 2016–2024 under 5% (acceptable). 2000–2015 exceed 5% in ORIGINAL TIFs — expected and already resolved: Designated-only corrected TIFs uploaded to SCRATCH on 2026-05-27. New panel (SEA merge in queue) will use corrected TIFs.
- Non-Designated pixels are primarily "Inscribed" (UNESCO WHSites) and "Established" — legally valid PAs.
- Reporting lag confirmed: SA 2023=~78% capture (old panel), SEA 2023/2024=0% (old panel).
- Audit script: `scripts/regions/shared/audit_wdpa_status.py`. JSON output in `outputs/<region>/results/wdpa_audit_<region>.json`.
- GEE re-export NOT required. Issue F fully closed.

**G+M — Forward pipeline incomplete** ← Core transition risk contribution
Stage 2 output is an uncalibrated rank score, not P(designated). Stage 1 budget is single-year only. Together these block the 2025–2029 cumulative risk product. Fix: W8 binary + Platt calibration → true P(designated | expansion); then implement multi-year accumulation. Depends on W8 completing.

**O — Binary scale_pos_weight bug** ✅ **Fixed 2026-05-28**
W8 binary job 689625 OOM'd: `neg_ratio=None` loaded full parquet; SPW computed on subsampled ratio (~100) instead of true ratio (~300–500). Fix applied: `_scan_true_class_counts()` pre-scans expansion groups before loading; binary now uses `STAGE2_NEG_RATIO=100` for memory; true SPW passed through to both tuning (`optimize_lgbm_stage2_binary_optuna(true_scale_pos_weight=...)`) and training (`scale_pos_weight = true_spw`). All 6 binary SLURM scripts updated with `export STAGE2_NEG_RATIO=100`. W8 resubmitted (after new SA splits from corrected panel) — see Active Euler Jobs.

### HIGH — Affects Stage 2 quality

**D — LambdaRank 9K sub-window mismatch** ✅ **Fixed 2026-05-29 (W9a implemented)**
LightGBM enforces 9K-row per-query ceiling. SA median group = 413K rows → training optimised within 9K but evaluation is over full groups. Fix (W9a): training groups are now `(country_id, year, eco_id)` ecoregion sub-groups (Colombia estimate: ~2,680 rows/group, well below 9K); NDCG evaluation still uses full `(country_id, year)` groups via `_TrueNdcg1PctEarlyStop`. Eco rasters loaded from `$SCRATCH/data/{region}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif`. Rank normalisation and graded relevance are computed within full cy groups before eco sub-sort. Binary (W8) has no 9K ceiling — eco groups not used for binary. Enabled via `STAGE2_ECO_GROUPS=1` env var in Colombia SLURM scripts.

**N — Early stopping monitors wrong metric** ✅ **Fixed 2026-05-29**
`eval_at=[90]` within 9K sub-windows caused LambdaRank to stop at iter 2 (Colombia); binary AP stopping caused stop at iter 5. Fix: `_TrueNdcg1PctEarlyStop` custom callback in `stage2_lgbm_core.py` calls `model.predict(X_val)` every iteration and stops on true NDCG@1% within full groups (patience=100 for final training, patience=50 for Optuna folds). Applied to both LambdaRank and binary in training AND Optuna — both models now tune and stop on the actual paper metric. Colombia re-run submitted (jobs 1227274–1227280).

**H — All-region panel rebuild** ⏳ **SA done; SEA+USA in queue**
- ✅ SA merge completed 2026-05-28 13:56 → `merged_panel_2000_2024.parquet` (40 GB)
- ✅ SA FE completed 2026-05-28 16:34 → `merged_panel_final.parquet` (57 GB)
- ⏳ SEA merge (job 1076912), SEA FE (1076913), USA merge (1076915), USA FE (1076916) — sequential chain in queue.

### REVIEWER BLOCKERS — Required before tier-A submission (identified 2026-05-29)

**Q — No confidence intervals on any reported metric**
Every number in the results tables (D²=+0.399, Lift@1%=4.95×, etc.) is a point estimate. This is not publishable in a tier-A journal. SA test-year Lift@1% ranges 1.47× (2017) to 9.64× (2019) — a 6.5× spread that makes the macro-average meaningless without variance. Fix: bootstrap over test years (2017–2023 for Stage 1; expansion year sub-samples for Stage 2). Add 95% CI to all reported metrics before submission. Stage 1 has only 8 test years so use all-subsets jackknife. Stage 2 bootstrap over (country_id, year) expansion groups. This is the single most impactful missing piece.

**R — Stage 1 Poisson over-dispersion unaddressed**
PA expansion pixel counts are wildly over-dispersed (Brazil 2001–2009: 100–320K px/yr). Poisson assumes Var = Mean, violated by orders of magnitude. Winsorisation + log1p are heuristic fixes, not a distributional solution. Fix: add Negative Binomial regression as a robustness spec (same 13 features, same train/test split). If NB D²_7yr ≈ Poisson D²_7yr, over-dispersion is not distorting results and both specs can be reported. If NB materially differs, it becomes the primary model. A statistical reviewer from JEEM or GEC will flag the missing NB spec immediately.

**S — W9a train/eval group mismatch needs scientific reframing, not engineering justification**
W9a trains LambdaRank on `(country_id, year, eco_id)` sub-groups but evaluates on `(country_id, year)` full groups. The pairwise gradients come from within-ecoregion comparisons; the paper metric is within-country-year. "We did this because LightGBM has a 9K row ceiling" is an engineering explanation that a Methods reviewer will reject. Fix: reframe in the Methods section as an ecological hypothesis — "geographic selection operates within ecological strata; designations in the Amazon are not directly comparable to designations in the Andes even within the same country-year." Provide supporting evidence (e.g., SHAP values differ by biome). If this framing cannot be defended, investigate XGBoost `rank:ndcg` (no query-size ceiling) as an alternative.

**T — Two-stage independence assumption is implicit, never stated**
The product `P(expansion | country) × P(pixel | expansion)` assumes Stage 1 and Stage 2 are conditionally independent. In reality a government may expand *because* a specific high-value forest block exists (REDD+ projects are site-specific, not country-level commitments). Fix: (1) state the independence assumption explicitly in the Methods; (2) test it by correlating Stage 1 residuals with the average ranked pixel characteristics from Stage 2 — if high-value pixels drive expansion timing, the residuals will correlate with Stage 2 feature distributions; (3) acknowledge the violation as a limitation if found.

### HIGH — Affects Stage 2 quality

**D — LambdaRank 9K sub-window mismatch** ✅ **Fixed 2026-05-29 (W9a implemented)**
LightGBM enforces 9K-row per-query ceiling. SA median group = 413K rows → training optimised within 9K but evaluation is over full groups. Fix (W9a): training groups are now `(country_id, year, eco_id)` ecoregion sub-groups (Colombia estimate: ~2,680 rows/group, well below 9K); NDCG evaluation still uses full `(country_id, year)` groups via `_TrueNdcg1PctEarlyStop`. Eco rasters loaded from `$SCRATCH/data/{region}/ready/GSN/gsn_terrestrial_ecoregions_mask_1km.tif`. Rank normalisation and graded relevance are computed within full cy groups before eco sub-sort. Binary (W8) has no 9K ceiling — eco groups not used for binary. Enabled via `STAGE2_ECO_GROUPS=1` env var in Colombia SLURM scripts.

**N — Early stopping monitors wrong metric** ✅ **Fixed 2026-05-29**
`eval_at=[90]` within 9K sub-windows caused LambdaRank to stop at iter 2 (Colombia); binary AP stopping caused stop at iter 5. Fix: `_TrueNdcg1PctEarlyStop` custom callback in `stage2_lgbm_core.py` calls `model.predict(X_val)` every iteration and stops on true NDCG@1% within full groups (patience=100 for final training, patience=50 for Optuna folds). Applied to both LambdaRank and binary in training AND Optuna — both models now tune and stop on the actual paper metric. Colombia re-run submitted (jobs 1227274–1227280).

**H — All-region panel rebuild** ⏳ **SA done; SEA+USA in queue**
- ✅ SA merge completed 2026-05-28 13:56 → `merged_panel_2000_2024.parquet` (40 GB)
- ✅ SA FE completed 2026-05-28 16:34 → `merged_panel_final.parquet` (57 GB)
- ⏳ SEA merge (job 1076912), SEA FE (1076913), USA merge (1076915), USA FE (1076916) — sequential chain in queue.

**U — Stage 1 sample size too small for 13 parameters; no country fixed effects**
SA Stage 1 trains on ~192 observations (12 countries × 16 years) with 13 features including a sparse interaction term. Effective independent N ≈ 12 (countries). The interaction `legcon_x_cspart` fires on 25 instances with 89.5% bootstrap sign stability — borderline. A reviewer from econometrics will ask why there are no country fixed effects (absorbs baseline expansion rate heterogeneity; standard in panel data). Fix: (1) run a sensitivity spec with country fixed effects and compare D²_7yr; (2) if fixed effects collapse OOS performance (expected — they cannot generalize to new countries), use that as the explicit justification for excluding them; (3) consider dropping `legcon_x_cspart` from the primary spec given marginal bootstrap stability, demoting it to a robustness table.

**V — CBD meeting year coefficient based on 2 training instances**
`cbd_meeting_year` fires on 2001–2016 training data only for years 2002 and 2010. A coefficient from 2 instances is statistically indistinguishable from noise. The improvement over CBD-free fallback (D²_7yr=+0.399 vs +0.389) is 0.010 — well within model uncertainty at N=192. Fix: drop `cbd_meeting_year` from the primary spec; report CBD-free as the primary spec and CBD-inclusive as a robustness check (currently reversed). The CBD-free fallback is more defensible as primary because it has no 2-instance coefficients.

**W — Graded relevance labels encode a debatable scientific hypothesis**
Labels 1–4 based on BFS cluster size mean a 5,000px park designation is trained as 4× more relevant than a 10px IUCN II patch. This embeds a judgment (large parks matter more) that is debatable for transition risk (a single farmer adjacent to a 10px proposal faces the same risk as one adjacent to a 5,000px park). Cluster size is also spatially autocorrelated — pixels next to other transitioning pixels tend to form larger clusters, biasing labels toward spatially clustered events. Fix: (1) run Colombia binary-label sensitivity (graded vs binary labels, same LambdaRank architecture) — if Lift@1% is similar, the graded labels add complexity without benefit; (2) document the scientific justification for the label scheme in the Methods regardless of outcome.

**X — LambdaRank neg-ratio subsampling distorts gradient signals**
LambdaRank pairwise gradients depend on the *rank* of a positive item among all negatives. With neg_ratio=100 (true ratio ~300–500), removing 2/3 of negatives changes the rank of every positive — a pixel at rank 500/1500 becomes rank 167/500 — producing different gradient magnitude and direction. Unlike binary scale_pos_weight, there is no scalar correction for this rank distortion. Fix: on the Colombia panel, run neg_ratio ∈ {50, 100, 200, full} for LambdaRank and compare Lift@1% on the validation set. If results are stable, document this as a sensitivity check. If unstable, use the smallest ratio that fits in memory or switch to a memory-efficient streaming approach for LambdaRank.

### LOWER — Required before submission

**J — SEA Stage 1 overfits with full political model**: Parsimonious 10-feat spec (D²=+0.290) beats full model (D²=+0.103). Settled: use parsimonious.

**P — Use Lift@1% (not NDCG) for W8 comparison**: LambdaRank trained on graded labels has structural NDCG advantage. Lift@1% is fair.

**Y — Lift@1% macro-averaging gives equal weight to large and small expansion events**
Macro-averaged Lift@1% weights Brazil 2012 (~413K pixels) identically to Ecuador 2008 (~2K pixels). A randomly good prediction on Ecuador inflates the metric as much as a genuinely good prediction on Brazil, which accounts for most actual land area. Fix: compute and report both macro-averaged and area-weighted (weighted by cy group size) Lift@1%. If they diverge substantially, the paper must explain which is the policy-relevant metric and why.

**Z — WDPA 2024 labels in Stage 2 test set are systematically incomplete**
Stage 1 excludes 2024 from primary metrics due to ~33% WDPA panel capture rate (WDPA reporting lag). Stage 2 `test_years = (2017, 2024)` includes 2024, meaning Stage 2 Lift@1% is evaluated against incomplete labels for the most recent year. Fix: either (a) exclude 2024 from Stage 2 test set for primary metrics (consistent with Stage 1), or (b) quantify the impact — run test metrics with and without 2024 and report the difference. If 2024 has few expansion groups it may not matter, but this must be checked and documented.

**AA — Spatial autocorrelation in Stage 2 residuals not tested**
Stage 2 features (dist_wdpa, spatial smoothing) and labels (cluster-based relevance) are spatially autocorrelated. The LOBO cross-validation tests biome-level generalization but not within-biome residual autocorrelation. Fix: compute Moran's I on the Stage 2 residuals (predicted score minus realised label) for the SA test set. If Moran's I is near zero, spatial autocorrelation is not inflating performance estimates. If it is large, spatial block cross-validation is needed. This is a half-day analysis on the scored test parquet.

**REDD+ verification**: Open each `# TO VERIFY` URL in `build_redd_plus.py` and confirm enrollment year against FCPF/UN-REDD pages before submission.

---

## Dev Environment (Colombia — Primary Development Environment)

**Purpose**: Colombia (~7M pixel-years, 3.9 GB panel) is now the PRIMARY development environment, not just a fast-iteration debugging aid. All new features, architectural experiments, and issue resolutions are built and validated on Colombia first. SA/SEA/USA continental runs are paused until Colombia hits the publication bar (Recall@5% ≥ 90%).

Full SA Stage 2 tuning takes ~2 days per run — Colombia runs in hours. This iteration speed is essential for the Feature Sprint (Tier 1–3 above).

**Design**: Filter SA Stage 2 splits (train/earlystop/test) to Colombia `country_iso3=COL` → `data/dev/south_america/ml/main/`. Set `STAGE2_DATA_ROOT` to redirect data loading and `STAGE2_OUTPUT_DIR` to isolate best_params.json. Zero production code changes; model/metrics outputs go to standard paths with timestamps (no collision risk).

**New env vars** (added 2026-05-28):
- `STAGE2_DATA_ROOT`: directory containing `main/{train,earlystop,test}.parquet` — set by Colombia SLURM scripts.
- `STAGE2_OUTPUT_DIR`: redirect best_params.json from tuning → used by training to avoid overwriting SA production params.

**Use for**: Issues D/N/O — LambdaRank vs binary comparison, SPW fix validation, group-size diagnostics. Signals are directional only; hyperparameters do not transfer to full SA. Full SA re-tunes once architecture is settled.

**Prerequisite**: Verify Colombia has ≥5 expansion country-years in 2001–2013 training window before relying on results (checked automatically by `create_colombia_panel.py`).

**SLURM scripts** (`slurm/south_america/`):
- `dev_colombia_panel.slurm` — create Colombia splits (depends on SA splits job)
- `tuning_lgbm_stage2_colombia.slurm` — LambdaRank tuning, 30 trials, 6h, 32GB
- `training_lgbm_stage2_colombia.slurm` — LambdaRank training, 3h, 32GB
- `tuning_lgbm_stage2_binary_colombia.slurm` — binary tuning, 30 trials, 6h, 32GB
- `training_lgbm_stage2_binary_colombia.slurm` — binary training, 3h, 32GB

**Old Colombia pipeline** (`south_america/colombia/`): Delete — non-standard legacy structure, fully superseded.

---

## Active Euler Jobs

| Chain | Status |
|---|---|
| SA LambdaRank re-tune → retrain (689639 → 689640) | ✅ Done 2026-05-28 (ran on old panel — re-train on new panel pending after splits) |
| WDPA audit (1076907) | ✅ Done 2026-05-28 13:07 — all 3 regions |
| SA merge (1076908) → SA FE (1076910) | ✅ Done 2026-05-28 13:56 / 16:34 |
| SEA merge (1076912) → SEA FE (1076913) → USA merge (1076915) → USA FE (1076916) | ✅ All done (SEA merge 5m, SEA FE 22m, USA merge 11m, USA FE 57m) |
| SA splits (1106055) | ✅ Done (3m 53s) |
| col_panel (1106062) | ❌ Failed silently — `country_iso3` absent + WDPA_b2 leakage bug |
| col_tune_lr (1106410) → col_train_lr (1106411) | ❌ Failed — ran on full SA panel; WDPA_b2 blocked both |
| col_panel (1122578) | ❌ Failed — `RecordBatch` passed to `write_table()` (needs `Table`); partial file written |
| col_tune_lr (1122580) → col_train_lr (1122581) → col_tune_bin (1122582) → col_train_bin (1122583) | ❌ All failed — empty Colombia panel → no expansion samples |
| W8 tune (1122585) → W8 train (1122586) | ❌ Cancelled 2026-05-29 — ran prematurely (silent Colombia failures released afterok dependency). Must wait for Colombia Issue D decision first. |
| col_panel (1141209) → col_tune_lr (1141210) → col_train_lr (1141211) → col_tune_bin (1141212) → col_train_bin (1141213) | ✅ Done 2026-05-29 — but early stopping fired at iter 2 (LR) / iter 5 (bin); Issue N unresolved |
| col_tune_lr (1227274) → col_train_lr (1227276) → col_tune_bin (1227278) → col_train_bin (1227280) | ❌ Cancelled 2026-05-29 — 1227274 ran 1h20m without W9a; cancelled to resubmit with W9a |
| col_tune_lr (1242056) → col_train_lr (1242057) → col_tune_bin (1242058) → col_train_bin (1242059) | ✅ Done 2026-05-29 14:06/15:18. LR Lift@1%=5.99× (W9a), Bin=3.04×. **LambdaRank selected.** |
| SA Stage 1 panel rebuild (stage1_data_builder.py) | ✅ Done 2026-05-29 locally — new panel from corrected merged_panel_final.parquet |
| SEA Stage 1 panel rebuild (stage1_data_builder.py) | ✅ Done 2026-05-29 locally — D²_6yr=−0.619 (regime shift finding) |
| SA Stage 2 LambdaRank re-tune (100 trials, W9a) → retrain | ⏳ Ready to submit — run `sbatch slurm/south_america/tuning_lgbm_stage2.slurm` then chain training |

---

## Next Actions (Ordered)

0. ~~**Issue F audit**~~ ✅ Done 2026-05-28.
0b. ~~**Issue O fix**~~ ✅ Done 2026-05-28. `_scan_true_class_counts()`; binary neg_ratio=100; 6 SLURM scripts updated.
0c. ~~**Colombia dev panel script**~~ ✅ Done 2026-05-28.
0d. ~~**Submit Colombia pipeline + W8**~~ ✅ Done 2026-05-28.
0e. ~~**Issue N fix**~~ ✅ Done 2026-05-29. `_TrueNdcg1PctEarlyStop` in training + Optuna.
0f. ~~**Issue D fix (W9a)**~~ ✅ Done 2026-05-29. Eco-stratified groups; STAGE2_ECO_GROUPS=1.
0g. ~~**Panel rebuild chain**~~ ✅ All done (SA, SEA, USA FE complete).
0h. ~~**Colombia comparison + model decision**~~ ✅ Done 2026-05-29. LambdaRank 5.99× >> Binary 3.04×. W9a: LR 2.99×→5.99×. LambdaRank selected.
0i. ~~**Stage 1 re-run (Issues V+R+Q+U)**~~ ✅ Done 2026-05-29. CBD-free primary (V). Jackknife CI (Q). Country FE sensitivity (U, FE→−0.44, no-FE justified). Panels rebuilt from corrected merged panels. SEA D²=−0.619 = path-dependency finding. NB robustness code added (statsmodels absent locally; run on Euler when needed).
### Phase 1: Colombia Feature Sprint (CURRENT — do not advance to Phase 2 until bar met)

1. **Verify Stage 1 locally** — sync `data/south_america/stage1_panel.parquet` (35 KB, already in repo) and run `model1_expansion.py` locally. Compare D²=+0.345 against local run. If you distrust the expansion counts, then sync 57 GB: `rsync euler:/cluster/scratch/gikaufmann/data/south_america/ml/merged_panel_final.parquet .` and re-run `stage1_data_builder.py`.

2. **Issue X — neg_ratio sensitivity (Colombia, Tier 3 arch experiment)**: Run LambdaRank neg_ratio ∈ {100, 200, full} on Colombia. 3.9 GB panel may fit full groups. Gradient distortion at 100 may be masking significant lift. Do this first — it costs one Euler job and may give 2–3× improvement for free.

3. **Issue W — Graded vs binary label sensitivity (Colombia)**: LambdaRank with binary labels vs current graded 1–4. Compare Recall@5% and Lift@1%. Required for Methods justification (Issue W).

4. **Tier 1 Feature Sprint — KBAs (Colombia)**: Extract KBA polygons → rasterise to 1km EPSG:3857 → compute `is_kba` + `dist_kba` → add to Colombia panel → retune + retrain → report Recall@5% and Lift@1% delta. Source: BirdLife International (free download).

5. **Tier 1 Feature Sprint — REDD+ project boundaries (Colombia)**: Verra registry shapefiles for Colombia projects → `dist_redd_project`, `in_redd_project`. Source: Verra API or verra.org project search.

6. **Tier 1 Feature Sprint — Indigenous territory boundaries (Colombia)**: RAISG polygon overlap → `dist_indigenous_territory`, `in_indigenous_territory`. Separate from existing `dist_indigenous` (which is point-based).

7. **Tier 1 Feature Sprint — ESA CCI Biomass (Colombia)**: Carbon stocks. Download from ESA Climate Office (300m) → resample to 1km → add `agb_tonne_ha` to Colombia panel.

8. **Tier 1 Feature Sprint — PA connectivity gap score (Colombia)**: For each pixel, compute increase in PA network connectivity if designated. Simple proxy: binary indicator for pixels that lie in a gap between two PA clusters within 5km.

9. **Evaluate Colombia bar**: After Tier 1 features, compute Recall@5% and Lift@1% on test set (2017–2024). If bar met (Recall@5% ≥ 90%, Lift@1% ≥ 15×) → advance to Phase 2. If not → continue Tier 2 features.

10. **Issue Q — Bootstrap CI on Recall@5% and Lift@1% (Colombia)**: Once bar is met, add bootstrap 95% CI to all reported metrics. This is the evidence needed for a reviewer.

11. **W4 ablation study (Colombia)**: Remove feature groups one at a time. Required for paper Methods: "what drives protection decisions."

### Phase 2: Continental Scale-Up (LOCKED until Phase 1 bar met)

**Do not run these until Colombia Recall@5% ≥ 90% is confirmed.**

12. **SA full re-tune + retrain (LambdaRank W9a, 100 trials, with all Tier 1+ features)**:
    `sbatch slurm/south_america/tuning_lgbm_stage2.slurm`
    Then chain: `sbatch --dependency=afterok:<tune_job_id> slurm/south_america/training_lgbm_stage2.slurm`
13. **SEA Stage 2 full re-tune + retrain** on corrected panel with same feature set.
14. **Issue T — Independence assumption**: Correlate Stage 1 residuals with Stage 2 mean ranked pixel characteristics. Document in Methods.
15. **Issue S — W9a scientific reframing**: Methods paragraph justifying eco sub-groups as ecological hypothesis. SHAP-by-biome evidence.
16. **Issue R — NB robustness**: Run when statsmodels available on Euler. Add NB D²_7yr comparison for SA paper table.
17. **Issues Y + Z + AA**: Area-weighted Lift@1%, Stage 2 2024 exclusion, Moran's I.
18. **Forward pipeline — Issues G+M**: Platt calibration + multi-year accumulation. After Stage 2 finalised.
19. **REDD+ verification**: Open TO VERIFY URLs before final manuscript.
20. **Manuscript gate**: Colombia bar confirmed + SA + SEA Stage 2 final + Issues Q/R/S/T/W resolved + ablation + G+M. Do not start writing before all gates.

---

## Data Sprint (Colombia first; SA/SEA/USA after Phase 1 bar met)

**GEE / external exports — Colombia priority order:**
- KBAs: BirdLife International shapefile (free, no GEE needed) → rasterise locally
- REDD+ project polygons: Verra registry (public) → download → rasterise locally
- RAISG indigenous territories: Direct download (public shapefile) → rasterise locally
- ESA CCI Biomass: ESA Climate Office download (300m, free) → resample → add to panel
- PA network connectivity: Derived from WDPA (no export needed); compute locally from existing WDPA rasters
- RUNAP/PNN proposals: datos.gov.co (Colombia national open data) → download → rasterise
- IDEAM priority watersheds: datos.gov.co → download → binary `in_priority_watershed`

**GEE exports (after Colombia bar met, SA/SEA/USA):**
- ESA CCI Biomass (already downloaded for Colombia above; GEE for continental scale)
- RAISG at continental scale (already available as shapefile; GEE not needed)

---

## Settled Decisions (Do Not Revisit)

- **Architecture**: Two-stage. Single-model AUC is proven leakage. Group A/B diagnostic is definitive.
- **Primary metric**: Lift@1% within expansion groups (Stage 2); D² OOS (Stage 1). Not global AUC.
- **Temporal split**: Train 2001–2016, test 2017–2024 — already optimal for SA+SEA combined D². Do not change.
- **WDPA reporting lag**: SA 2024 (~33% panel capture) and SEA 2023–2024 (IDN=0) excluded from primary metrics. Primary = SA 7yr (2017–23), SEA 6yr (2017–22). D²_8yr is secondary.
- **Governance: first differences only** (Δ, not levels) — level variables cause monotonic temporal drift in forward predictions.
- **Pre-2001 lags: terrestrial-only** — GIS_AREA−GIS_M_AREA; marine contamination fix applied (ECU Galápagos was 86% marine).
- **No pre-2001 training data** — frontier exhaustion regime mismatch. WDPA used for lag init at 2001 only.
- **30×30 as exogenous budget**: post-Stage-1 multiplier, not a model coefficient (zero training variation).
- **CBD = robustness check, not primary**: 2 training instances (2002, 2010). CBD-free primary D²_7yr=+0.334 (corrected panel 2026-05-29). CBD-inclusive is retained as a robustness check (D²=+0.321).
- **SEA Stage 1 D²<0 = path-dependency finding**: KHM/LAO sporadic one-off designations are structurally unpredictable from momentum features. Same interpretation as USA D²<0. Old D²=+0.290 was on contaminated labels (4.2–5.9% non-Designated, now corrected).
- **SEA saturation leakage**: DO NOT re-investigate. sat_clean gives D²=−0.275 — spurious gain was entirely from future-info leakage.
- **Bolsonaro hypothesis falsified**: 2019 is SA's best test year (Lift=9.64×). 2017 is worst (1.47×). Issue D (9K mismatch) is primary suspect for SA underperformance.
- **USA**: Stage 1 D²<0 = path-dependency finding, not failure. Stage 2 deprioritised until SA+SEA settled.
- **Regions**: SA primary; SEA + USA as robustness. DO NOT add tropical Africa. DO NOT start Paper 2 until Paper 1 submitted.
- **Development scale**: Colombia first. All architecture and feature decisions made on Colombia (~7M rows, hours per run). Continental SA/SEA/USA locked until Colombia Recall@5% ≥ 90%.
- **Performance bar**: Recall@5% ≥ 90% within Colombia expansion groups is the minimum to publish in a tier-A journal. 6× Lift@1% alone is not sufficient — a reviewer will ask how many PAs we miss. The answer must be "fewer than 10% in our top 5% slice."
- **neg_ratio + year weights**: training reads `STAGE2_NEG_RATIO` env var (same as tuning). Year weights active in both training and Optuna CV.

---

## Paused (until Colombia Phase 1 bar met)

These are not cancelled — they are waiting for Colombia to prove the model.

- **SA Stage 2 continental re-tune**: paused. Do not submit `tuning_lgbm_stage2.slurm` until Colombia Recall@5% ≥ 90%.
- **SEA Stage 2 continental re-tune**: paused. Old 12.7× is on old panel; will re-evaluate after Phase 1.
- **USA Stage 2**: deprioritised; path-dependency finding reduces urgency.
- **Forward pipeline (Issues G+M)**: paused until Stage 2 finalised on Colombia.
- **Issue AA (Moran's I)**: paused until continental Stage 2 results available.

## Out of Scope (Paper 1)

- Tropical Africa, marine PAs
- Colombia as a supplement (Colombia IS the primary development environment now — it will appear in the paper as the proving ground, not a supplement)
- Embeddings / Paper 2 (gate: AlphaEarth access + P1 submitted)
- Sub-national Stage 1 (fallback only if country-level results collapse)
- Curriculum hard negative mining, Stage 1 predictions as Stage 2 feature (defer to Paper 2)
- ±2 year STATUS_YR sensitivity (methods footnote only)
- Single-model global AUC as primary metric
