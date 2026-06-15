# PA3030 — Publication Roadmap

**Updated**: 2026-06-15 | **Branch**: `paper` (active). `main` = intact thesis, never touch.

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

**Stage 2** — LightGBM LambdaRank W9a. Eco-stratified training groups; eval on `(country_id, year)` groups. Graded relevance 1–4. Train 2001–2013, early-stop 2011–2013 (proxy) / 2014–2016 (full), test 2017–2024. Primary metrics: Lift@1% + Recall@5% within expansion groups.

**Forward output** (Phase 3): Stage 1 budget × Stage 2 suitability → cumulative risk = 1 − ∏(1 − score_t). BAU / 30×30 / NGFS scenarios.

---

## Publication Bar

| Metric | Bar | Proxy (22 groups) | Full SA |
|---|---|---|---|
| Lift@1% | ≥ 15× | **15.34×** ✓ | 3.82× (old baseline gate) |
| Recall@5% | ≥ 90% | **24.4%** ✗ | — |

---

## Current Situation

**Euler baseline gate complete (2026-06-15).** Full SA results (61 groups, 107M rows, test 2017–2024):

| Metric | Proxy (22 groups) | Full SA (61 groups) | Bar |
|---|---|---|---|
| Lift@1% | 15.34× | **2.85×** | ≥ 15× |
| Recall@5% | 24.4% | **14.0%** | ≥ 90% |

Both bars are missed by a wide margin on full SA. The proxy was overoptimistic: its 22 groups are the largest designation events where the model happens to do well. Full SA includes all 61 expansion country-years in the test period, including smaller/harder events. Early stopping hit at iter 149 (vs 446), suggesting proxy-tuned params don't fully transfer.

**Proxy is unreliable and overoptimistic.** Proxy Lift@1% (15.34×) is 5× higher than full SA (2.85×). Proxy Recall@5% (24.4%) is 1.7× higher than full SA (14.0%). Phase 1 feature experiments on the proxy were measuring noise, not signal.

Five Phase 1 experiments run on proxy, all now moot against full SA:
- `eco_protection_gap`: Proxy Recall 24.4%→23.4% — dropped
- `dist_redd_km`: Proxy Recall 24.4%→13.9% — dropped
- `ndcg5pct_earlystop`: Proxy Recall→29.1% but Lift@1% collapsed — reverted
- `agb_tonne_ha`: Proxy Lift@1%→14.07× (below proxy bar) — dropped
- `pruned30` (79→49 features): Proxy Lift@1%→17.10×, Recall→22.6%. mini_sample currently at 49-feature state.

**Structural ceiling hypothesis**: PA designation creates contiguous polygons. A pixel-independent ranking model correctly identifies "core" pixels (remote, biodiverse) but the remaining pixels in the polygon look ordinary in isolation — no feature makes them special. This is an area-process problem. The fix is spatial neighbourhood propagation of scores after training (no retraining needed). See queue below.

**Two problems now confirmed**: (1) Proxy is not representative — must redesign or abandon proxy-first workflow. (2) Absolute performance on full SA is far below bar on both metrics — spatial post-processing and/or retuning on full SA needed.

**SHAP audit done (2026-06-15).** Top 15 features by mean |SHAP|:
```
 1. NDVI_smooth64        0.350   (64km-scale vegetation — coarse landscape signal)
 2. GSN_b2              0.283   (biodiversity priority)
 3. WorldClim_b2        0.244   (climate)
 4. GPW                 0.227   (population density)
 5. deforestation_b2    0.207
 6. WorldClim_b14       0.195
 7. WorldClim_b11       0.187
 8. dist_indigenous     0.185   (distance to indigenous territories)
 9. elevation_b2_smooth4 0.149
10. GSN_b3              0.148
11. elevation_b1_smooth16 0.137
12. WorldClim_b19       0.128
13. WorldClim_b16       0.126
14. dist_wdpa           0.125   (distance to existing PAs)
15. HNTL_smooth64       0.108
```
Model dominated by coarse landscape signals. `dist_wdpa` (#14) is already annual (WDPA_prev at t-1, `feature_engineering:322`) — low rank is genuine behaviour, not a bug. SHAP plots at `outputs/south_america/results/phase1/baseline/shap_importance.png`, `shap_beeswarm.png`.

---

## Next Session — First Action

**Both Euler jobs are running. Wait for results, then interpret.**

| Job | SLURM ID | Status | ETA | Output |
|---|---|---|---|---|
| Spatial post-processing | 3495807 | PENDING→RUNNING | ~1–1.5h | `outputs/south_america/results/spatial_postprocess_alpha_sweep.json` |
| Full SA retuning (100 trials) | 3495808 | PENDING (afterok:3495807) | ~2–4 days | `scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json` |

Monitor:
```bash
squeue -u gikaufmann
tail $SCRATCH/logs/spatial_postprocess_3495807.out
tail $SCRATCH/logs/model1_tune_stage2_3495808.out
```

**When spatial PP finishes**: read `spatial_postprocess_alpha_sweep.json`. Look for the alpha where Recall@5% peaks without Lift@1% collapsing. That alpha is the post-processing parameter to use on top of all future models.

**When tuning finishes**: new `best_params.json` is in `scripts/regions/south_america/5_training/`. Submit `slurm/south_america/training_lgbm_stage2_phase1.slurm` to retrain with the new params, then apply the best alpha from the spatial sweep. Record full SA Lift@1% + Recall@5% in `feature_ablation_sa.json` (baseline entry, update `full_sa_*` fields).

**Note on spatial PP script**: `scripts/regions/south_america/6_evaluation/spatial_postprocess_stage2.py` replicates the exact preprocessing of `load_stage2_arrays` — WDPA_prev==0 filter + within-group rank normalization — so scores are valid. An earlier broken version (missing these two steps) was cancelled.

---

## Phase 1 — Queue

| Priority | Action | Status | Rationale |
|---|---|---|---|
| 1 | **Spatial post-processing** | 🔄 Running — job 3495807, ETA ~1–1.5h | Alpha sweep on existing model; read `spatial_postprocess_alpha_sweep.json` when done |
| 2 | **Full SA retuning (100 trials)** | 🔄 Queued — job 3495808, ETA ~2–4 days | 100 Optuna trials on full 42 GB SA splits; fixes hyperparameter mismatch |
| 3 | **Retrain + apply spatial PP** | ⬜ After tuning | Retrain with new params → apply best alpha → record full SA metrics |
| 4 | `is_kba`, `dist_kba_km` | ⏸ Blocked — awaiting BirdLife shapefile | Strongest intent signal (IUCN "should be protected" list) |
| 5 | `in_indigenous_poly`, `dist_indigenous_poly_km` | ⏸ Blocked — awaiting RAISG download | Resguardo → national park pathway |
| — | **Euler baseline gate** | ✅ Done 2026-06-15 | Full SA: Lift@1%=2.85×, Recall@5%=14.0%; proxy confirmed overoptimistic |
| — | SHAP audit | ✅ Done 2026-06-15 | See plots + findings above |
| — | `dist_wdpa` construction | ✅ Checked 2026-06-15 | Already annual. Low rank is genuine. |
| — | `agb_tonne_ha` | ✗ Dropped — best_iter=5 | AGB TIF still at `data/south_america/ready/AGB/agb_sa.tif` |
| — | `pruned30` (49 features) | ⏸ Parked — proxy Lift improves, Recall regresses | mini_sample currently at 49-feature state; revisit after full SA retuning |

**KBA download**: https://www.keybiodiversityareas.org/kba-data/request → `data/shared/KBA/`

**Spatial post-processing** (implemented, running):
```
final_score(pixel) = model_score(pixel) + α × max(model_score(8-neighbours))
```
Script: `scripts/regions/south_america/6_evaluation/spatial_postprocess_stage2.py`
Alpha sweep: [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]. Pick the alpha where Recall@5% peaks without Lift@1% collapsing.

**Mini-sample state**: currently 49 features (58 cols including metadata) after `pruned30` experiment. To restore to 88 cols for future local experiments: re-inject the 30 dropped features from their source TIFs. For now, Euler uses the full SA splits (all features) independently.

**Workflow per feature** (no per-feature retuning — use baseline params):
```
inject: python scripts/regions/south_america/3_merging/add_feature_to_mini_sample.py \
            --tif data/south_america/ready/<X>/x_sa.tif --cols col1 [col2]
split:  python scripts/regions/south_america/3_merging/prepare_mini_splits.py
train:  python scripts/regions/south_america/5_training/model1_phase1_local_train \
            --feature-tag <name> --notes "..."
```

## Phase 1 — Feature Ablation Log

| Feature / Experiment | Proxy Lift@1% | Proxy Recall@5% | best_iter | Kept |
|---|---|---|---|---|
| baseline (79 features, 88 cols) | 15.34× | 24.4% | 124 | ✓ reference |
| eco_protection_gap | 13.08× | 23.4% | 63 | ✗ |
| dist_redd_km | 15.29× | 13.9% | 113 | ✗ |
| ndcg5pct_earlystop (objective) | 5.38× | 29.1% | 20 | ✗ Lift collapse |
| agb_tonne_ha | 14.07× | 19.3% | 5 | ✗ |
| pruned30 (79→49 features) | **17.10×** | 22.6% | 94 | ⏸ Parked |

---

## Phase 2 (locked until SA bar confirmed on full test)

- SA full retune (100 trials) with confirmed features + objective
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
- **neg_ratio=100**: 200 lost on test Lift@1%
- **Governance features**: first differences only
- **CBD**: robustness only; CBD-free is primary
- **SEA/USA D²<0**: regime-switching / path-dependency findings, not failures
- **No ensembles, no sub-models** (supervisor directive)
- **Simplicity principle**: single Stage 2 model

---

## Data Paths

| Dataset | Location |
|---|---|
| SA full splits (42 GB) | `euler:$SCRATCH/data/south_america/ml/main/` |
| SA mini-sample (49 features, 58 cols) | `data/south_america/mini_sample.parquet` |
| SA mini-splits | `data/south_america/mini_splits/main/{train,earlystop}.parquet` |
| Baseline best params (training dir) | `scripts/regions/south_america/5_training/model1_stage2_lgbm_best_params.json` |
| Baseline best params (Phase 1 archive) | `outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_best_params.json` |
| Baseline booster | `outputs/south_america/results/phase1/baseline/model1_stage2_lgbm_booster.txt` |
| Feature ablation log | `outputs/south_america/results/feature_ablation_sa.json` |
| AGB TIF (tried, dropped) | `data/south_america/ready/AGB/agb_sa.tif` |
| REDD TIF (dropped) | `data/south_america/ready/REDD/redd_sa.tif` |

**CRS note**: All rasterise scripts must use `gdf.to_crs(epsg=3857)` and `CRS.from_epsg(3857)` — backbone CRS is LOCAL_CS, `crs.to_epsg()` returns None.

---

## Paused

- SA full retune — Phase 2
- SEA Stage 2 — Phase 2
- USA Stage 2 — deprioritised
- Forward pipeline — Phase 3

## Out of Scope (Paper 1)

- Ensemble methods, sub-models, neural networks (Paper 2), survival framing, tropical Africa
