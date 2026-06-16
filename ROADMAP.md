# PA3030 — Publication Roadmap

**Updated**: 2026-06-17 (H5 mini done) | **Branch**: `paper` (active). `main` = intact thesis, never touch.

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

**Stage 2** — LightGBM LambdaRank W9a. Eco-stratified training groups; eval on `(country_id, year)` groups. Graded relevance 1–4. Train 2001–2013, early-stop 2014–2016, test 2017–2024. Primary metrics: Lift@1% + Recall@5% within expansion groups.

**Forward output** (Phase 3): Stage 1 budget × Stage 2 suitability → cumulative risk = 1 − ∏(1 − score_t). BAU / 30×30 / NGFS scenarios.

---

## Publication Bar

| Metric | Bar | Full SA best |
|---|---|---|
| Lift@1% | ≥ 15× | **3.73×** (H6+H1b, 2026-06-16) |
| Recall@5% | ≥ 90% | **18.1%** (H6+H1b, 2026-06-16) |

---

## Guiding Principle

**The publication bar (Lift@1%≥15×, Recall@5%≥90%) is achievable. We haven't found the right approach yet.**

Before tweaking hyperparameters or dropping features, ask: *Is the fundamental framing correct?* When performance is far below bar, the problem is almost certainly foundational. Small tweaks will not close a 5× gap in Lift@1%. Be bold and creative. Treat every proposed fix as a hypothesis with uncertainty, not a confirmed solution.

---

## Current Situation (2026-06-16)

**All full SA experiments so far:**

| Experiment | Lift@1% | Recall@5% | best_iter | Verdict |
|---|---|---|---|---|
| Baseline (79 feat, truncation=741) | 2.85× | 14.0% | 149 | previous best |
| 20-trial retune → truncation=84 | 2.06× | 8.4% | 7 | ✗ catastrophic |
| 67 feat + YEAR_WEIGHT_MIN=0.2 | 2.64× | 11.4% | 113 | ✗ both negative |
| **H6+H1b (Recall@5% earlystop + inv_sqrt_npos)** | **3.73×** | **18.1%** | **89** | ✓ **new best — both improved** |

H6+H1b is the first experiment to improve both metrics simultaneously. +31% Lift, +29% Recall vs baseline. This is the new locked baseline for future experiments.

**What worked and why:**
- **H6 (Recall@5% early stopping)**: Directly optimises the publication metric instead of NDCG@1%. More stable signal (aggregates over 5× more data points per group). Main driver.
- **H1b (inv_sqrt_npos group weights)**: Down-weights large events (Brazil 2001–2009), gives small events more influence. Adds ~+1% Recall and improves Lift vs H6 alone.
- **H3 (W9a off)**: Tested and rejected — hurts both metrics. Eco sub-groups stay.

**What didn't work:**
- H1 (inv_npos, 1/n_pos): Too aggressive, destabilises training, stops at iter 56
- H1+H6 without H1b not tested directly (H6_only: Lift=8.85×, Recall=27% on proxy — H1b improved on this)

---

## Root Cause Analysis: The Gradient Concentration Problem

**This is the most important finding.** From scanning the full SA training split (2001–2013, 121 expansion groups):

| Group subset | # groups | % of total gradient |
|---|---|---|
| Top 5 groups | 5 | **72.6%** |
| Top 10 groups | 10 | **96.0%** |
| Groups with n_pos ≤ 100 | 27 | **0.003%** |
| Groups with n_pos ≤ 20 | 12 | **0.000%** |
| Years 2001–2009 | 96 groups | **98.6%** |
| Years 2010–2013 | 25 groups | **1.4%** |

The LambdaRank gradient from a group scales as `n_pos × min(n_neg, neg_ratio × n_pos)`. A single group with 110,000 positives generates ~820 billion pairs. A group with 13 positives generates 16,900 pairs — a ratio of ~50,000,000:1. After neg_ratio=100 subsampling, the ratio is still ~40,000:1.

**The model has been trained almost exclusively on ~10 giant designation events from 2001–2009. Everything else — including all the small, post-2010 events that resemble the test set — is invisible to it.**

This is why temporal weighting failed: even at YEAR_WEIGHT_MIN=0.2, the huge 2001–2009 groups still dominate because their absolute pair count dwarfs recent small events. You'd need to downweight 2006 by a factor of ~50,000,000 to equalise it with a 13-positive group.

---

## Hypotheses to Test (Ordered by Expected Impact)

These are hypotheses. Each needs a controlled experiment before committing Euler time.

### H1: Group-normalised weights (highest priority)

**Idea**: Weight every sample in group g by `1 / n_pos(g)`. This makes each expansion group contribute equal total gradient regardless of its size.

With this weighting:
- Group with 110,000 positives: weight per sample = 0.000009
- Group with 13 positives: weight per sample = 0.077
- Both groups contribute the same total gradient signal

**Expected effect**: The model is forced to learn from small events. If small-event positives do have learnable feature signals (within their group — which the pooled z-score analysis cannot tell us), the model will now learn them.

**Risk**: Small events may be genuinely noise at the feature level. If so, forcing the model to learn from them could degrade performance on large events without improving it on small ones. This is an empirical question.

**Implementation**: compute `group_weight = 1 / n_pos_for_group` per row, pass as `weights=` to `lgb.Dataset`. Modify `load_stage2_arrays` to return group_weights or compute them in `run_stage2_training`.

**Variant**: `1 / sqrt(n_pos)` is less aggressive — large events downweighted but not as severely. Test both.

### H2: Binary labels instead of graded relevance

**Idea**: Use 0/1 labels instead of 1–4 graded relevance by cluster size.

**Rationale**: Graded relevance amplifies the gradient from large clusters (relevance=4) relative to isolated pixels (relevance=1). Combined with the gradient concentration above, this makes the model doubly biased toward large-block events. Binary labels remove this second-order bias.

**Note**: Previously tested on proxy (Colombia dev): "Graded relevance 1–4: LambdaRank >> Binary (5.99× vs 3.04×)." But that was on Colombia dev data and likely dominated by large events too. On full SA with group normalisation, the comparison may be different. Test H2 jointly with H1.

### H3: Drop eco sub-groups (W9a off)

**Idea**: Train with standard (country_id, year) groups instead of (country_id, year, eco_id) sub-groups.

**Rationale**: W9a was validated on the proxy (22 large groups). On full SA with 121 diverse groups including many small events, splitting large groups into eco sub-groups may fragment the gradient signal in ways that hurt small-event learning. The model learns "which pixel is best within this ecoregion" but is tested on "which pixel is best within this entire country."

**Risk**: Without W9a, the 9K sub-window fragmentation problem returns for large groups. Evaluate whether large groups in training are actually >9K after eco splitting — if all eco sub-groups are already <9K, W9a may simply be benign but not necessary.

**Implementation**: Set `STAGE2_ECO_GROUPS=0` in SLURM script. Fast to test.

### H4: Extended training window 2001–2016

**Idea**: Include 2014–2016 in training; use 2017–2019 as early-stop window.

**Rationale**: 2014–2016 represents the post-2013 low-designation regime. Training on those years gives the model direct examples of the policy era the test set comes from. The 2013 cutoff was originally chosen to avoid leakage into the test period; extending to 2016 moves the gap from 4 years to 1 year.

**Risk**: Requires redefining the splits parquet files (currently test.parquet starts at 2017, earlystop is 2014–2016). This is a data pipeline change, not just a training change. Confirm with supervisor before doing.

### H5: Remove rank normalisation (or exclude it for key features)

**Idea**: Stop replacing feature values with within-group percentile ranks. Or at minimum, exclude features where absolute value matters (e.g. biodiversity score, distance to PA) from normalisation.

**Rationale**: Rank normalisation removes absolute feature information. If a pixel's biodiversity score is "objectively high" (top 5% globally), that absolute signal is lost when replaced by its rank within a country-year group where most pixels happen to have low biodiversity. The model cannot learn "protect the most biodiverse land globally" — only "protect the most biodiverse land within this particular country-year group."

Top SHAP features (biodiversity GSN_b2, population GPW, distance dist_wdpa) are ALL globally comparable — the absolute values carry real signal that within-group normalisation destroys. This is the strongest argument for H5.

**Risk**: Rank normalisation was introduced to handle cross-country scale differences. Without it, Brazil's feature distribution (large country, high variance) might overwhelm smaller countries. LambdaRank already compares within-group, so scale differences between countries should not matter as long as features are internally consistent.

**Implementation**: `STAGE2_NORMALIZE_WITHIN_GROUPS=0` — coded 2026-06-17.

### H6: Metric–objective alignment

**Idea**: We train on NDCG@1% (graded relevance, early stopping) but evaluate Recall@5% (binary, all designated pixels equal). These reward different things. Consider whether the early-stopping metric should be binary Recall@5% instead.

**Challenge**: Implementing a custom early-stopping callback on Recall@5% (binary, within-group) is straightforward — it's the same `_TrueNdcg1PctEarlyStop` structure but calling `compute_stage2_metrics` and extracting `recall_at_5pct_within_groups`. Try this alongside H1.

### H7: Restrict training to post-2010 events only

**Idea**: Set `STAGE2_TRAIN_YEAR_MIN=2010` to drop all 2001–2009 training groups and train only on events from 2010–2013 (25 groups).

**Rationale**: The gradient concentration analysis shows 2001–2009 = 98.6% of gradient. Even with H1b (inv_sqrt_npos), these ancient mega-events still dominate because their absolute pair count dwarfs recent small events. The test set (2017–2024) consists almost entirely of small, targeted post-2010 style events. Training exclusively on 2010–2013 events (same regime as test) may generalise far better, at the cost of fewer training groups.

**Risk**: Only 25 training groups. Training may be less stable; possible overfitting to those 25 groups. Cannot be tested on mini-sample (only 1 year of post-2009 data in mini train). Must go directly to full SA.

**Implementation**: `STAGE2_TRAIN_YEAR_MIN=2010` env var in `run_stage2_training()` — coded 2026-06-17. Full SA only.

### H8: Combined size + temporal gradient weights

**Idea**: `STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos_temporal`. Weight = `(1/sqrt(n_pos)) × exp(-0.2 × (max_year - year))`. Addresses BOTH gradient concentration sources simultaneously: event size AND event age.

**Rationale**: H1b (inv_sqrt_npos) only corrects for SIZE. But large ancient events are doubly damaging: large (monopolises gradient) AND old (most unlike test era). Multiplying by temporal decay further penalises 2001–2009 events while rewarding 2010–2013 events. Example: Brazil 2006 (n_pos=110K) gets ~2000× less weight than Peru 2012 (n_pos=13), compared to ~92× for pure H1b.

**Risk**: More aggressive than H1b. Could destabilise training (similar risk to H1 inv_npos). Should test with H6 (Recall@5% earlystop) to buffer instability.

**Implementation**: `mode="inv_sqrt_npos_temporal"` in `compute_group_norm_weights()` — coded 2026-06-17. Test on mini-sample first.

### H9: Partial rank normalisation (absolute features only)

**Idea**: Disable rank normalisation for features where absolute value is globally meaningful (biodiversity GSN_b2, population GPW, distance features), keep it for contextual features (climate, elevation, NDVI).

**Rationale**: H5 (full no-norm) loses the benefit of rank normalisation for features where it helps. Partial normalisation captures the best of both: absolute global signals where they exist, relative within-group ranking where absolute values are country-specific.

**Risk**: Requires defining the split manually. Test H5 (full no-norm) first — if H5 works, partial norm is refinement. If H5 fails, partial norm is the next experiment.

**Implementation**: `STAGE2_ABS_FEATURES="GSN_b2,GPW,dist_wdpa,dist_indigenous"` env var; apply rank normalisation only to the complement. Not yet coded.

---

## Experiment Queue

**Locked baseline**: H6+H1b (Recall@5% earlystop + inv_sqrt_npos). All future experiments build on this.

Run on mini-sample first. If proxy Recall@5% > 20% (current best) → promote to full SA.

| Priority | Experiment | Status | Notes |
|---|---|---|---|
| 1 | H5 full SA | 🔄 Running — job 3655195 | `training_lgbm_stage2_h6_h1b_h5.slurm`; ~8h |
| 2 | H10+H8 mini (4 runs) | 🔄 Queued — job 3655220, after 3655195 | `mini_h10_h8.slurm`; tests ndcg×recall stop + temporal weights |
| 3 | H8: inv_sqrt_npos_temporal weights | ⬜ Not started | combined size+temporal — `STAGE2_GROUP_WEIGHT_MODE=inv_sqrt_npos_temporal` (coded) |
| 4 | H7: train only 2010–2013 | ⬜ Not started — full SA only | `STAGE2_TRAIN_YEAR_MIN=2010`; cannot test on mini (only 1 post-2009 year) |
| 5 | Retune hyperparams (≥100 trials) | ⬜ Not started | only after structure is locked |
| 6 | KBA features | ⏸ Blocked — awaiting BirdLife shapefile | add dist_kba_km, is_kba |
| 7 | Indigenous polygon features | ⏸ Blocked — awaiting RAISG download | add dist_raisg_km, is_indigenous_territory |
| 8 | H4: extended training window (2001–2016) | ⏸ Needs pipeline change — discuss first | include 2014–2016 in train; use 2017–2019 as earlystop |

**Decision rule for H5 full SA**: If full SA Recall@5% > 18.1% (H6+H1b) → lock H5 as new baseline. If full SA Lift@1% also drops, investigate combined-metric stopping (H10) next.

**Closed experiments:**
- H1 inv_npos: ✗ too aggressive, training collapses
- H1b inv_sqrt_npos alone: ✗ NDCG@1% early stop fires at iter 8 — needs H6
- H3 W9a off: ✗ hurts both metrics
- H6+H1b+H3: ✗ W9a off degrades H6+H1b
- H2 binary labels (on top of H6+H1b): ✗ no change — binary labels add nothing when H1b already corrects gradient bias; Recall=27.9%, Lift=12.86× = identical to H6+H1b
- H5+H2: ✗ H2 adds nothing on top of H5; result identical to H5 alone

---

## Lift vs Recall Trade-off Diagnosis (2026-06-17)

**Critical finding**: Every intervention so far improves one metric at the cost of the other.

| Stopping metric | Proxy Lift@1% | Proxy Recall@5% | Direction |
|---|---|---|---|
| NDCG@1% (baseline, year_weights) | **20.32×** | 13.3% | high lift, low recall |
| Recall@5% (H6+H1b) | 12.86× | 27.9% | lower lift, better recall |
| Recall@5% (H6+H1b+H5) | 11.52× | **34.8%** | lowest lift, best recall |

The publication bar requires BOTH ≥15× Lift AND ≥90% Recall simultaneously. Neither early stopping metric achieves this. Current experiments optimise ONE metric while degrading the other.

**Root cause**: NDCG@1% stopping teaches the model to rank the best few pixels perfectly (high Lift) but ignores the bottom half of positives (low Recall). Recall@5% stopping teaches the model to find ALL positives in the top 5% (high Recall) but doesn't concentrate the best pixels in the very top 1% (lower Lift).

**H10 hypothesis**: Early stopping on `ndcg_at_1pct × recall_at_5pct` (product). This combined metric is maximised only when BOTH are high. Training under this objective might avoid the trade-off by forcing the model to be simultaneously a good ranker AND a good retriever.

---

## What We Know From Experiments So Far

**On proxy (22 large groups — now abandoned as unreliable):**
- Baseline: Lift@1%=15.34×, Recall@5%=24.4%
- Graded relevance >> binary (5.99× vs 3.04× on Colombia dev)
- eco_protection_gap: −1.3× Lift
- dist_redd_km: −1.1× Recall
- NDCG@5% earlystop: Lift collapsed
- agb_tonne_ha: best_iter=5 (model fails to train)
- pruned30 (49 feat): Lift+1.8× but Recall−2%

**On full SA (61 groups, 107M rows, test 2017–2024):**
- Baseline (79 feat, truncation=741, default params): **Lift=2.85×, Recall=14.0%** — current best
- 20-trial Optuna retune → truncation=84: Lift=2.06×, Recall=8.4% — catastrophic
- 67 feat + YEAR_WEIGHT_MIN=0.2: Lift=2.64×, Recall=11.4% — both negative

**Lesson**: Proxy experiments are noise. All decisions must be validated on full SA. Incremental tweaks (feature drops, temporal weighting) made things worse. The gap from 2.85× to 15× requires a structural fix.

---

## Settled Decisions

- **Engine**: LightGBM LambdaRank, neg_ratio=100
- **Early stopping**: Recall@5% within (country_id, year) groups (H6, locked 2026-06-16)
- **Sample weights**: inv_sqrt_npos group-norm weights (H1b, locked 2026-06-16)
- **Eco sub-groups**: W9a on (H3 tested and rejected)
- **Primary metrics**: Lift@1% + Recall@5% within expansion groups
- **Temporal split**: Train 2001–2013, early-stop 2014–2016, test 2017–2024 — *potentially revisit H4*
- **Proxy Recall@5% is directionally reliable**: proxy baseline 13.3% ≈ full SA 14.0%. Use proxy to screen experiments.
- **Full SA retuning with < 100 trials is harmful**: Only retune after structure is locked.
- **No ensembles, no sub-models** (supervisor directive)
- **Governance features**: first differences only
- **CBD**: robustness only; CBD-free is primary

---

## Phase 2 (locked until SA bar confirmed)

- Full SA retune (100 trials) with confirmed features + objective
- SEA Stage 2 retune on corrected panel
- Bootstrap CIs, model comparison table
- Spearman ρ, NB robustness, independence check

## Phase 3 (after Phase 2)

- Platt calibration → suitability scores
- Cumulative risk pipeline (Stage 1 × Stage 2)
- NGFS scenario integration
- Conservation gap: predictions × biodiversity raster → 2×2 map (Nature Sustainability hook)
- **Manuscript gate**: all above before writing

---

## Data Paths

| Dataset | Location |
|---|---|
| SA full splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/` |
| SA mini-sample (79 features) | `data/south_america/mini_sample.parquet` |
| Baseline best params | `scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json` |
| Baseline best params (archive) | `outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json` |
| Baseline booster | `outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_booster.txt` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` |
| AGB TIF | `data/south_america/ready/AGB/agb_sa.tif` |
| REDD TIF | `data/south_america/ready/REDD/redd_sa.tif` |

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)` — backbone CRS is LOCAL_CS, `crs.to_epsg()` returns None.

---

## SHAP Audit (2026-06-15, baseline 79-feature model)

```
 1. NDVI_smooth64        0.350
 2. GSN_b2              0.283   biodiversity priority
 3. WorldClim_b2        0.244
 4. GPW                 0.227   population density
 5. deforestation_b2    0.207
 6. WorldClim_b14       0.195
 7. WorldClim_b11       0.187
 8. dist_indigenous     0.185
 9. elevation_b2_smooth4 0.149
10. GSN_b3              0.148
11. elevation_b1_smooth16 0.137
12. WorldClim_b19       0.128
13. WorldClim_b16       0.126
14. dist_wdpa           0.125
15. HNTL_smooth64       0.108
```
Caveat: SHAP reflects what the baseline model learned, which is dominated by large 2001–2009 events. It does not reflect what a group-normalised model would use.

Plots: `outputs/south_america/results/phase1/baseline/shap_importance.png`, `shap_beeswarm.png`.

---

## Spatial Post-Processing (closed, revisit if model improves)

```
alpha=0.0  Lift@1%=3.33×  Recall@5%=14.8%
alpha=1.0  Lift@1%=3.43×  Recall@5%=14.6%
```
No meaningful effect at current model quality.

---

## Paused

- SEA Stage 2 — Phase 2
- USA Stage 2 — deprioritised
- Forward pipeline — Phase 3

## Out of Scope (Paper 1)

- Ensemble methods, sub-models, neural networks (Paper 2), survival framing, tropical Africa
