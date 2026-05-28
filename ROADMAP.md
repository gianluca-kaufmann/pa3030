# PA3030 — Paper Publication Roadmap

**Updated**: 2026-05-27 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

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

## Current Key Numbers ⚠️ Preliminary — Euler re-runs pending

| Region | Stage 1 D² | Stage 2 Lift@1% | Status |
|---|---|---|---|
| SA | **+0.399** (7yr PRIMARY 2017–23) / +0.428 (3yr) / +0.377 (8yr) | **6.0×** LambdaRank | Stage 2 re-tune+retrain pending (jobs 689639→689640) |
| SEA | **+0.290** (6yr PRIMARY 2017–22) / +0.399 (3yr) / +0.208 (8yr) | **12.7×** LambdaRank | FE re-run pending; SEA is currently the stronger Stage 2 result |
| USA | −3.14 (8yr, trend-only) | TBD | Path-dependency finding; Stage 2 deprioritised |

SA naive baseline: dist_wdpa Lift=0.747× < 1 (anti-predictive — test designations are novel areas, not extensions to existing PAs).
All numbers will update after Euler FE re-runs (WDPA GeoTIF patch + saturation + trend features).

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

**F — WDPA label quality** ← **Do first, before trusting any results**
GEE export has no `STATUS='Designated'` filter. "Proposed" / "Inscribed" areas may be in positive labels — affects all metrics. Quick audit: count `transition_01=1` pixels where WDPA STATUS ≠ 'Designated'. If contamination >5% → GEE re-export required and all downstream results are suspect.

**G+M — Forward pipeline incomplete** ← Core transition risk contribution
Stage 2 output is an uncalibrated rank score, not P(designated). Stage 1 budget is single-year only. Together these block the 2025–2029 cumulative risk product. Fix: W8 binary + Platt calibration → true P(designated | expansion); then implement multi-year accumulation. Depends on W8 completing.

**O — Binary scale_pos_weight bug** ← Blocking W8
W8 binary job 689625 OOM'd: `neg_ratio=None` loaded full parquet; SPW computed on subsampled ratio (~100) instead of true ratio (~300–500), inflating positives. Fix: pre-scan parquet for true n_pos/n_neg before loading; use neg_ratio=100 for memory; pass true SPW as override to both tuning and training. Resubmit after 689640 completes.

### HIGH — Affects Stage 2 quality

**D — LambdaRank 9K sub-window mismatch**
LightGBM enforces 9K-row per-query ceiling. SA median group = 413K rows → training optimises within 9K but evaluation is over full groups. CV=0.186 → test=0.028 gap is partly this. Binary (W8) has no ceiling. Decision rule: binary Lift@1% vs LambdaRank Lift@1% on SA test set. If binary wins → switch primary, activate Issues G+M fix. If LambdaRank wins → pursue W9a ecoregion groups.

**N — Early stopping monitors wrong metric**
`eval_at=[90]` within 9K sub-windows; paper metric is NDCG@1% within 413K-row groups. Binary avoids this entirely.

**H — USA + SEA FE re-runs pending**
SA panel rebuilt (✅). SEA TIFs now on Euler (✅) — FE re-run not yet submitted. USA panel from 2026-05-22 pre-dates saturation + trend features. Both unblocked.

### LOWER — Required before submission

**J — SEA Stage 1 overfits with full political model**: Parsimonious 10-feat spec (D²=+0.290) beats full model (D²=+0.103). Settled: use parsimonious.

**P — Use Lift@1% (not NDCG) for W8 comparison**: LambdaRank trained on graded labels has structural NDCG advantage. Lift@1% is fair.

**REDD+ verification**: Open each `# TO VERIFY` URL in `build_redd_plus.py` and confirm enrollment year against FCPF/UN-REDD pages before submission.

---

## Dev Environment (Colombia Fast-Iteration Panel)

**Purpose**: Full SA Stage 2 tuning takes ~2 days per run — too slow for architectural iteration. Colombia-only subset (~7M vs 350M rows) reduces this to minutes.

**Design**: Filter SA Stage 2 panel to Colombia `country_id` → `data/dev/sa_panel_dev.parquet`. Run existing Stage 2 scripts unchanged via `STAGE2_PANEL_PATH=data/dev/sa_panel_dev.parquet`. Zero code changes.

**Use for**: Issues D/N/O — LambdaRank vs binary comparison, SPW fix validation, group-size diagnostics. Signals are directional only; hyperparameters do not transfer. Full SA re-tunes once architecture is settled.

**Prerequisite**: Verify Colombia has ≥5 expansion country-years in 2001–2013 training window before relying on results.

**Old Colombia pipeline** (`south_america/colombia/`): Delete — non-standard legacy structure, fully superseded.

---

## Active Euler Jobs

| Chain | Status |
|---|---|
| SA LambdaRank re-tune → retrain (689639 → 689640) | 🔄 Pending (QOSMaxMemoryPerUser) |
| W8 SA binary tune → train | ❌ Needs resubmit after Issue O fix + after 689640 |
| SEA FE re-run | ⏳ Unblocked — submit now |
| USA FE re-run | ⏳ Independent — submit now |

**Do not modify SA panel parquets while 689639–689640 run.**

---

## Next Actions (Ordered)

0. **Set up Colombia dev panel**: filter SA Stage 2 parquet to Colombia `country_id`, verify ≥5 expansion training groups, confirm tuning runs in <30 min. Then use it to resolve Issues D/O before touching Euler.
1. **Issue F — WDPA label audit** (run locally, fast): count STATUS≠Designated in positive labels. If >5% contamination → GEE re-export takes priority over everything else.
2. **Submit SEA FE re-run** on Euler (TIFs already there — unblocked).
3. **Submit USA FE re-run** on Euler (independent).
4. **Await 689640** → resubmit W8 binary with Issue O fixes (true SPW, neg_ratio=100).
5. **W8 decision**: binary Lift@1% vs LambdaRank Lift@1%. Record result here.
   - Binary wins → fix forward pipeline (G+M), binary becomes Stage 2 primary.
   - LambdaRank wins → pursue W9a ecoregion-stratified training groups.
6. **Download updated panels** (SEA + USA after FE re-runs) → re-run Stage 1 locally → update key numbers above.
7. **W4 ablation study** (SA first, after re-train result confirmed): does removing pa_momentum collapse Lift? Required for paper Methods.
8. **Forward pipeline — Issues G+M**: Platt calibration + multi-year accumulation (2025–2029). Implement after W8 settled.
9. **W9a ecoregion groups** (conditional on W8 outcome — see Issue D): groups `(country_id, year, ecoregion_id)` for training; evaluate at `(country_id, year)`. Only if LambdaRank primary and 9K is confirmed bottleneck.
10. **REDD+ verification**: open TO VERIFY URLs before final manuscript.
11. **Manuscript gate**: SA + SEA Stage 2 final results confirmed + Issue F resolved + ablation done + Issues G+M implemented. Do not start W7 before all gates.

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
