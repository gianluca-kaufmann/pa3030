# PA3030 — Paper Publication Roadmap

**Updated**: 2026-05-28 (evening) | **Branch**: `paper` (active). `main` = intact thesis, never touch.

**Story**: The 30×30 biodiversity agreement is coming. We predict which land will be designated as protected area — giving investors and central banks an actionable transition risk tool for portfolio exposure to PA designation events.

**Target journal**: One Earth / GEC → Nature Sustainability → JEEM

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

## Current Key Numbers ⚠️ Preliminary — SA re-train on new panel pending; SEA+USA FE in queue

| Region | Stage 1 D² | Stage 2 Lift@1% | Status |
|---|---|---|---|
| SA | **+0.399** (7yr PRIMARY 2017–23) / +0.428 (3yr) / +0.377 (8yr) | 4.95× LambdaRank (8yr 2017–24, intermediate) | Improvements incoming |
| SEA | **+0.290** (6yr PRIMARY 2017–22) / +0.399 (3yr) / +0.208 (8yr) | **12.7×** LambdaRank | FE re-run pending; SEA is currently the stronger Stage 2 result |
| USA | −3.14 (8yr, trend-only) | TBD | Path-dependency finding; Stage 2 deprioritised |

SA naive baseline (dist_wdpa only): 2.81× on 8yr test (2017–24). All numbers will update after next improvements + FE re-runs.

---

## Stage 1 Current Specs

**SA — 13 features, Poisson α=1, p95 winsor, log1p momentum, pre-2010 decay=0.6**
Momentum (log1p): lag1/2/3, cumsum_lag1 | Political levels: v2x_polyarchy (+0.62), gdp_growth_lag1 (+0.33), redd_plus_enrolled (−0.16) | First differences: Δv2xlg_legcon (≈0), Δv2csprtcpt (+0.09), Δv2xlg_legcon_lag1 (−0.11, 1-yr policy lag) | Interaction: legcon_x_cspart = Δlegcon×Δcspart (+0.05) | Land constraint: agricultural_land_pct (−0.16) | Policy cycle: cbd_meeting_year (+0.12, robustness check only — 2 training instances)

CBD-free fallback D²_7yr=+0.389. No-interact fallback D²_7yr=+0.380. Chow p=0.858 (break non-significant — was lag-init artefact).

**SEA — 10 features, Poisson α=1, no decay**
Same momentum | v2x_polyarchy (−0.35 — authoritarian states expand more in SEA; sign reversal vs SA is a paper finding) | gdp_growth_lag1, redd_plus_enrolled | Δv2xlg_legcon, Δv2csprtcpt | legcon_x_cspart

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

**D — LambdaRank 9K sub-window mismatch**
LightGBM enforces 9K-row per-query ceiling. SA median group = 413K rows → training optimises within 9K but evaluation is over full groups. CV=0.186 → test=0.028 gap is partly this. Binary (W8) has no ceiling. Decision rule: binary Lift@1% vs LambdaRank Lift@1% on SA test set. If binary wins → switch primary, activate Issues G+M fix. If LambdaRank wins → pursue W9a ecoregion groups.

**N — Early stopping monitors wrong metric** ✅ **Fixed 2026-05-29**
`eval_at=[90]` within 9K sub-windows caused LambdaRank to stop at iter 2 (Colombia); binary AP stopping caused stop at iter 5. Fix: `_TrueNdcg1PctEarlyStop` custom callback in `stage2_lgbm_core.py` calls `model.predict(X_val)` every iteration and stops on true NDCG@1% within full groups (patience=100 for final training, patience=50 for Optuna folds). Applied to both LambdaRank and binary in training AND Optuna — both models now tune and stop on the actual paper metric. Colombia re-run submitted (jobs 1227274–1227280).

**H — All-region panel rebuild** ⏳ **SA done; SEA+USA in queue**
- ✅ SA merge completed 2026-05-28 13:56 → `merged_panel_2000_2024.parquet` (40 GB)
- ✅ SA FE completed 2026-05-28 16:34 → `merged_panel_final.parquet` (57 GB)
- ⏳ SEA merge (job 1076912), SEA FE (1076913), USA merge (1076915), USA FE (1076916) — sequential chain in queue.

### LOWER — Required before submission

**J — SEA Stage 1 overfits with full political model**: Parsimonious 10-feat spec (D²=+0.290) beats full model (D²=+0.103). Settled: use parsimonious.

**P — Use Lift@1% (not NDCG) for W8 comparison**: LambdaRank trained on graded labels has structural NDCG advantage. Lift@1% is fair.

**REDD+ verification**: Open each `# TO VERIFY` URL in `build_redd_plus.py` and confirm enrollment year against FCPF/UN-REDD pages before submission.

---

## Dev Environment (Colombia Fast-Iteration Panel)

**Purpose**: Full SA Stage 2 tuning takes ~2 days per run — too slow for architectural iteration. Colombia-only subset (~7M vs 350M rows) runs in hours.

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
| col_tune_lr (1227274) → col_train_lr (1227276) → col_tune_bin (1227278) → col_train_bin (1227280) | ⏳ In queue 2026-05-29 — Issue N fix applied: _TrueNdcg1PctEarlyStop replaces ndcg@90/AP stopping |

---

## Next Actions (Ordered)

0. ~~**Issue F audit**~~ ✅ Done 2026-05-28. All 3 regions audited; labels acceptable.
0b. ~~**Issue O fix**~~ ✅ Done 2026-05-28. `_scan_true_class_counts()` added; binary neg_ratio=100 with true SPW override; 6 SLURM scripts updated.
0c. ~~**Colombia dev panel script**~~ ✅ Done 2026-05-28. `scripts/regions/south_america/dev/create_colombia_panel.py` + `STAGE2_DATA_ROOT`/`STAGE2_OUTPUT_DIR` env vars.
0d. ~~**Submit Colombia pipeline + W8**~~ ✅ Done 2026-05-28. SA splits → Colombia panel → LambdaRank + binary chains; W8 SA binary in parallel. See Active Euler Jobs.
1. ~~**Submit panel rebuild chain**~~ ✅ SA done; SEA+USA in queue (jobs 1076912–1076916).
2. **Colombia results**: When Colombia LambdaRank + binary training complete, compare Lift@1% in SLURM logs. Record here. This is the Issue D decision point.
3. **W8 decision**: When W8 SA binary training completes, compare binary Lift@1% vs LambdaRank Lift@1% on full SA test set. Record here.
   - Binary wins → fix forward pipeline (G+M), binary becomes Stage 2 primary.
   - LambdaRank wins → pursue W9a ecoregion-stratified training groups.
5. **Re-run Stage 1** locally after SEA + USA FE jobs complete → update key numbers table above.
6. **W4 ablation study** (SA first, after re-train result confirmed): does removing pa_momentum collapse Lift? Required for paper Methods.
7. **Forward pipeline — Issues G+M**: Platt calibration + multi-year accumulation (2025–2029). Implement after W8 settled.
8. **W9a ecoregion groups** (conditional on W8 outcome — see Issue D): groups `(country_id, year, ecoregion_id)` for training; evaluate at `(country_id, year)`. Only if LambdaRank primary and 9K is confirmed bottleneck.
9. **REDD+ verification**: open TO VERIFY URLs before final manuscript.
10. **Manuscript gate**: SA + SEA Stage 2 final results confirmed + Issue F resolved + ablation done + Issues G+M implemented. Do not start W7 before all gates.

---

## Data Sprint (parallel to steps 2–7)

**GEE exports** (high scientific value — start SA first):
- ESA CCI Biomass / GEDI AGBD (carbon stocks): REDD+ mechanism makes high-carbon forests financially valuable to protect — expected top-5 SHAP
- RAISG indigenous territory area fraction: largest omitted SA variable; complements existing dist_indigenous

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
- **CBD = robustness check, not primary**: 2 training instances (2002, 2010). CBD-free fallback D²_7yr=+0.389 is the primary reported model.
- **SEA saturation leakage**: DO NOT re-investigate. sat_clean gives D²=−0.275 — spurious gain was entirely from future-info leakage.
- **Bolsonaro hypothesis falsified**: 2019 is SA's best test year (Lift=9.64×). 2017 is worst (1.47×). Issue D (9K mismatch) is primary suspect for SA underperformance.
- **USA**: Stage 1 D²<0 = path-dependency finding, not failure. Stage 2 deprioritised until SA+SEA settled.
- **Regions**: SA primary; SEA + USA as robustness. DO NOT add tropical Africa. DO NOT start Paper 2 until Paper 1 submitted.
- **neg_ratio + year weights**: training reads `STAGE2_NEG_RATIO` env var (same as tuning). Year weights active in both training and Optuna CV.

---

## Out of Scope (Paper 1)

- Tropical Africa, marine PAs, Colombia (supplement only if reviewer requests)
- Embeddings / Paper 2 (gate: AlphaEarth access + P1 submitted)
- Sub-national Stage 1 (fallback only if country-level results collapse)
- Curriculum hard negative mining, Stage 1 predictions as Stage 2 feature (defer to Paper 2)
- ±2 year STATUS_YR sensitivity (methods footnote only)
- Single-model global AUC as primary metric
