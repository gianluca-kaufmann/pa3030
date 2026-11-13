# Protected-Area Designation Forecasting — Strategy & Status

## Purpose and maintenance

This is the project’s single strategy, status, and memory document. Read it before acting.

1. Keep **Current state**, **Next actions**, and **Decision gates** current; they are the operational source of truth.
2. Record settled results once in the **Evidence ledger**. Git history is the detailed archive; do not create narrative archive files.
3. Add one compact line to the **Session index** after each session. Remove obsolete operational detail once it no longer affects the next action.
4. A result changes direction only through a predeclared gate. Do not promote a suggestive exploratory result into a claim.
5. Preserve user changes and unrelated work. Never write to or run jobs in the frozen PA3030 thesis repository.

---

## North star

**Non-negotiable ambition:** produce a Nature-level publication related to protected areas. Everything else—question, geography, data, method, and collaborators—may change.

**Approved primary direction (2026-08-25):** reopen prediction as a genuinely new problem, not a continuation of the closed flat pixel-classification search.

> **Can the static socioeconomic–environmental niche of protected areas forecast future legal designations, and does conservation finance shift which high-biodiversity places become politically feasible to protect?**

Working scientific frame: **from a static conservation niche to a dynamic conservation-feasibility frontier**.

The ideal full claim has three parts:

1. **Forecasting:** a model frozen at time *t* predicts protected-area designations after *t* under real-world prevalence.
2. **Mechanism:** lagged conservation finance or policy pipelines add reproducible predictive value and, where identification permits, causally shift designation patterns.
3. **Consequence:** the shift changes whether high-biodiversity but initially “unrealistic” areas receive protection.

The Nature-level contribution would be the discovery and validation of a dynamic global conservation frontier—not a new machine-learning algorithm. Main *Nature* would require a large, general, policy-changing effect; *Nature Communications*, *Nature Sustainability*, or an equivalent top journal remains a strong outcome for a rigorous global result.

---

## Current state — 2026-08-25

**Active:** the **Mouillot Temporal Challenge**, a time-boxed attempt to falsify or validate prospective PA forecasting using Mouillot et al. (2024, *Nature Communications* 15:9007) as a frozen benchmark.

**Why this is a real reopening of Question 1:** Mouillot predicts static April-2020 PA occurrence on a global 10 km grid using 14 socioeconomic/environmental factors and spatial cross-validation. It does **not** test later designations, true event prevalence, temporal transfer, or held-out-country transfer. Our new estimand, risk set, validation, and finance mechanism are materially different from the closed work.

**What the paper establishes:** existing PAs are spatially distinguishable from sampled unprotected background. It explicitly estimates relative likelihood under presence–background sampling and invokes a space-for-time assumption; it does not prove that future designations are predictable.

**Other direction:** Question 2 (“Two Logics”) is **paused, not erased**. Its descriptive/causal results and the ongoing global dataset build are retained as supporting infrastructure and possible mechanism evidence. Do not continue regime-model elaboration unless a result directly serves the active question or the temporal challenge fails.

| ID | Active experiment | Status | Decision produced |
|---|---|---|---|
| **R0** | Mouillot package/data/date reproducibility audit | **NEXT** | Whether a defensible frozen benchmark exists |
| **T1** | April-2020 score → verified 2021–2024/25 designations | Blocked by R0/protocol | Stop, repair once, or build rolling forecasts |
| **T2** | Rolling-origin and held-country confirmation | Blocked by T1 | Whether forecasting remains a primary claim |
| **F1** | Incremental lagged finance/policy test | Blocked by T2 | Whether finance predicts a shifting frontier |
| **B1** | Biodiversity consequence + causal identification | Blocked by F1 | Whether the complete result is Nature-shaped |

### Immediate next actions

1. **Reproducibility audit (2–3 working days):** obtain the frozen Mouillot v1.0 Zenodo package; inventory data, scripts, saved models/outputs, package versions, and predictor dates; determine whether it fully refits the RF or mainly replays stored results/figures.
2. **Freeze the temporal-test protocol before seeing results:** cutoff, horizon, event definition, candidate universe, exclusions, primary metric, baselines, folds, uncertainty, and go/no-go criteria.
3. **Build the post-2020 event audit:** identify genuinely new designations rather than late WDPA reporting, renaming, geometry updates, or expansions of old sites.
4. **Run the frozen-score test:** among cells unprotected in April 2020, test whether the authors’ score predicts verified 2021–2024/25 designations.
5. **Decide at Gate T1:** stop, repair data, or proceed to rolling-origin backtests. Do not add model complexity before this decision.

---

## Research design contract

### Estimand and prediction target

- **Operational question:** which currently unprotected locations will receive legal PA status within the next three years?
- **Initial spatial unit:** the paper’s 10 × 10 km cell, to permit an exact benchmark.
- **Independence unit:** the legal designation event/site, not each covered cell. One large PA must not count as hundreds of independent successes.
- **Risk set:** all eligible cells unprotected at cutoff *t*, subject only to rules fixed before model evaluation.
- **Initial cutoff:** April 2020, matching the paper’s WDPA snapshot; initial outcome window is 2021–2024/25, depending on audited completeness.
- **Confirmation design:** multiple rolling cutoffs and untouched countries. A single 2020 test is a feasibility result, not final evidence.

### Label contract

A positive must be a plausibly new legal designation after the cutoff. Build an event ledger using, where available:

- legal designation year/status year;
- first observed appearance in archived WDPA releases;
- stable identifiers, names, geometry, and parent-site relationships;
- explicit flags for late reporting, renaming, recategorisation, boundary change, and expansion.

Primary results must exclude ambiguous events; include them only in sensitivity analyses. Report event counts, countries, area distribution, reporting delays, and concentration before modeling.

### Temporal feature contract

- Every predictor must have been observable by the forecast cutoff. Record `source`, `version`, `reference_year`, `release_date`, `processing_date`, and `allowed_cutoff`.
- Static physical layers may be reused after documentation. HDI, GDP, conflict, NGOs, human footprint, accessibility changes, finance, PA adjacency/density, and policy variables must be lagged or frozen.
- Never reconstruct an old forecast with a later covariate snapshot without marking it as a non-prospective sensitivity analysis.
- Fit preprocessing, imputation, winsorisation, feature selection, and calibration on training data only.

### Sampling and prevalence

- **Training:** use all positive events plus controlled negative sampling, class weighting, or a ranking objective as computationally appropriate.
- **Evaluation:** always score the complete eligible candidate universe at the real positive rate.
- Correct probability calibration to true prevalence; sampled training probabilities are not operational probabilities.
- Never headline accuracy. Extreme rarity is part of the scientific problem, not an inconvenience to remove from evaluation.

### Validation and metrics

**Primary metric:** average precision on the full true-prevalence risk set, accompanied by prevalence and AP/prevalence lift.

**Operational secondary metrics:** Recall@1% and Recall@5% candidate area, precision among the top-ranked candidates, calibration/Brier score, and macro country performance.

**Required validation:**

- rolling-origin temporal backtests;
- held-out-country transfer;
- spatial blocking within sufficiently large countries;
- uncertainty clustered/resampled by designation event and country;
- performance by region, realm, designation size, and reporting quality.

No random cell split can support the forecasting claim. Do not pool cell-level uncertainty across thousands of cells belonging to one designation.

---

## Model and evidence ladder

Complexity is earned only when the previous rung is understood.

| Rung | Model | Purpose |
|---|---|---|
| **B0** | Country historic rate / global prevalence | Non-spatial base rate |
| **B1** | Distance/adjacency to existing PA | Expansion-of-existing-network baseline |
| **B2** | Accessibility, human footprint, elevation, cheap-land variables | Classic residual-reserve baseline |
| **M1** | Exact Mouillot 14-variable, 500-tree RF and calibration | Frozen published static-niche benchmark |
| **M1t** | Temporally valid version of M1 | Test whether the niche transfers through time |
| **M2a** | Country-year hurdle/count model | Predict whether and how much a country protects |
| **M2b** | Within-country conditional ranker/hazard model | Predict which eligible locations are selected |
| **M3** | M2 + lagged finance/policy pipeline | Test incremental finance signal |
| **M4** | M3 + biodiversity opportunity frontier | Test whether finance changes conservation value |

Start with RF/gradient boosting, transparent count/hazard models, and ranking models. Do not start another deep-learning/model-zoo search. Neural point processes remain untested, but are justified only if a validated temporal signal and adequate event volume exist.

### Scale separation

Country-level finance or governance can explain **whether/how much** expansion occurs; it cannot by itself locate a 10 km cell. Local siting requires geographically resolved projects, proposals, land costs, threats, adjacency, or other site-level information. Never let repeated country-level values masquerade as local spatial evidence.

---

## Phased implementation and decision gates

### Phase R — Reproduce and audit the published benchmark

**Time box:** 2–3 working days.

**Deliverables:** immutable source archive/checksum; environment manifest; script/data map; reproduced terrestrial metrics or an exact discrepancy log; frozen scores; predictor-date audit.

**Gate R:** proceed only if we can regenerate or faithfully recover the model scores and explain every input used in the temporal test. If a full refit is unavailable but published frozen scores are recoverable, the initial out-of-time test may proceed; document the limitation.

### Phase T1 — Fast temporal falsification

**Time box:** 5–7 working days after Gate R.

1. Audit the number and geographic distribution of verified post-2020 events.
2. Score the full April-2020 unprotected risk set with M1.
3. Compare M1 with B0–B2 without tuning on the outcome window.
4. Report real-prevalence metrics and event/country-blocked uncertainty.

**Gate T1 outcomes:**

- **GO:** M1 shows positive, practically useful lift over all simple baselines with no obvious concentration/leakage explanation → build rolling backtests.
- **REPAIR:** event ledger or predictor timing is inadequate but fixable → repair only the identified defect and rerun once.
- **STOP:** M1 fails to beat simple baselines or signal disappears after event/reporting controls → static niche is descriptive, not prospective; forecasting becomes a secondary result.

T1 is deliberately diagnostic. Do not claim success from one favorable cutoff.

### Phase T2 — Confirm temporal and geographic generalisation

**Expected time:** 1–3 weeks, conditional on T1.

- Reconstruct several historical PA estates and contemporaneous features.
- Run rolling train-at-*t*, predict-*t*+1…3 forecasts.
- Hold out complete countries/regions for final confirmation.
- Compare one-stage global RF with the two-stage country-year + within-country architecture.
- Freeze one discovery set and one untouched confirmation set before feature exploration.

**Gate T2:** the forecasting direction survives only if uplift is positive in most temporal folds, the pooled event/country-blocked confidence interval excludes zero on the confirmation set, and performance is not driven by one country, large PA, adjacency alone, or reporting artefacts.

### Phase F — Finance mechanism

**Conditional on T2.** Add only strictly lagged sources with documented coverage.

- Country scale: OECD CRS commitments/disbursements, GEF/other multilateral portfolios, domestic/environmental budgets where globally comparable, NGO capacity, carbon/REDD+ entry.
- Site scale: geocoded project locations/footprints and dated policy/proposal pipelines only when spatial meaning is defensible.
- Test M3 against the frozen M2 specification; do not retune M2 after viewing finance results.

**Gate F:** finance must add out-of-time value on untouched data, remain directionally stable across folds/regions, and survive controls for country propensity, development, governance, and designation history. Country-only lift supports timing/amount prediction, not a local-siting claim.

### Phase B — Biodiversity consequence and causal evidence

Test whether finance is associated with realised protection of locations that are simultaneously:

- high biodiversity/conservation priority;
- low baseline M1 feasibility (“unrealistic high gains”);
- genuinely newly designated.

Separate prediction from causation. A predictive finance lift is not proof that finance caused designation. Causal language requires a defensible shock, discontinuity, staggered intervention, or other identification strategy with pre-trend and placebo tests.

**Publication gate:** a Nature-level submission requires a coherent result across forecasting, mechanism, and consequence. Static replication, a modest AP gain, or descriptive maps alone are not sufficient.

---

## Anti-fishing and stopping rules

1. Freeze outcomes, exclusions, cutoffs, horizons, baselines, primary metric, folds, and confirmation data before model comparison.
2. Use exploration data freely to find leads, but record every tested feature family/model family in the experiment ledger.
3. Promote a lead only after untouched temporal/country confirmation; one favorable region or fold is not a result.
4. Permit one targeted repair after a failed gate when a specific data or implementation defect is demonstrated. Otherwise stop or pivot.
5. Report nulls and reversals plainly. Do not change the unit, horizon, metric, and sample repeatedly until something becomes significant.
6. A real pivot changes the estimand, information set, or architecture—not a single pipeline parameter.

---

## Live infrastructure: 234-country polygon dataset

The full WDPA event-level build remains useful for label auditing, rolling histories, finance linkage, and mechanism tests. It must not delay Phase R/T1, which can start from the authors’ package and a focused post-2020 ledger.

**KILLED 2026-08-26 (this session)**: both supervisor loops (PIDs 58700/58703) and their local Python drivers (compute_wdpa_zonal_stats.py, PIDs 58705/58707) were killed, and all outstanding GEE server-side tasks (44 on Project A, 22 on Project C, all READY/RUNNING) were explicitly cancelled via `ee.data.cancelTask`. **Reason: both projects share GEE noncommercial free-tier quota with the EUDR production app, which was getting locked out** — EUDR is the priority (real customer-facing product), so this research job was stopped rather than paying for a commercial GEE tier. At kill time both projects were still reporting "restricted mode" (quota exceeded) per EE's own warning. **To resume**: the pipeline is resumable via per-side asset-existence checks (confirmed session 2026-08-26 — a restart came back `PRE-CACHED` for one chunk), so relaunching later costs nothing lost, just re-run the two supervisor scripts once EUDR's GEE usage no longer needs the headroom. No progress was lost, only queued/in-flight tasks were discarded (they'll need to be resubmitted, not recomputed from scratch, since completed assets persist).

**Last recorded operational state before kill (2026-08-26):**

- Both local drivers silently crashed 2026-08-25 ~15:57 on an uncaught transient EE error (`ee.ee_exception.EEException: Not signed up for Earth Engine or project is not registered` — a real but transient 403 during a routine `.status()` poll inside `_poll_and_retry`, not a real deregistration; EE access was confirmed healthy again on the next direct check). GEE-side tasks kept completing for hours with nobody polling, retrying failures, or running the final read-back/write step — this went undetected until an explicit progress check found `ps aux` empty for both PIDs with no matching reboot. **Fix: both are now launched via a supervisor loop** (`/tmp/run_project_a_loop.sh`, `/tmp/run_project_c_loop.sh`) that restarts the script automatically 30s after any nonzero exit; resumability (per-side asset-existence checks) means a restart costs nothing (confirmed: one chunk came back `PRE-CACHED` on the 2026-08-26 restart). If checking progress in a future session, verify the supervisor is still alive with `pgrep -f run_project_a_loop.sh` / `run_project_c_loop.sh`, not just the inner python PID (that will legitimately churn across restarts) — logs at `/tmp/project_a_restart.log` / `/tmp/project_c_restart2.log`.
- Project A (`master-thesis-473316`): 26-country pilot buckets, noncommercial quota exhausted/restricted; 1990–2010 largely complete, 2015/2020 pending at last check. Supervisor PID 58700.
- Project C (`project-637f1965-0089-4a26-86a`): remaining 208 countries, quota exhausted/restricted; running with `tileScale=8` on `main` and pre-chunked ~10.5k–17k-event batches. Supervisor PID 58703.
- Project B (`pa-model-wdpa-zonal-2`): idle spare; do not duplicate A’s pilot work.
- C writes `zonal_<iso3>_partC.parquet`; A writes the pilot outputs. Verify the final writer is additive before trusting overlapping bucket runs.

**Before the global merge:** verify processes/tasks and asset completeness; inspect whether the builder unions suffixed files; merge without overwriting; run:

```bash
python3 pipeline/4_features/build_wdpa_global_dataset.py
python3 pipeline/4_features/audit_dataset.py \
  --path regions/global/wdpa_polygon_dataset.parquet \
  --group-col iso3 --key-cols wdpaid --strict
```

Expected current pilot schema: 51,832 designation events, 26 countries, 1990–2023, 58 columns. The `marine_frac < 0.1` rule makes it terrestrial-only and severely reduces archipelagic/coastal countries; state this limitation or revise the sampling frame deliberately.

---

## Evidence ledger — settled results

### Published benchmark: Mouillot et al. 2024

- Global 10 km cells; April-2020 WDPA; 14 factors; presence–background sampling; 500-tree RF; 1,000 km spatial blocks.
- All-category performance: land accuracy 0.74/TSS 0.47; sea 0.88/0.76. These are static classification results, not future-event forecasts.
- Public Zenodo/GitHub package contains processed data, predictions, importance outputs, and analysis/figure code; full refit reproducibility still needs audit.
- Scientific opportunity: temporalise their “potential vs unrealistic conservation gains” framework and test finance as a frontier-shifting mechanism.

### Question 1 — closed 2026-08-22: accuracy-focused global prediction

**Do not resume the old formulation without a materially new reason.** It was closed after precommitted tests, not abandoned mid-run.

| Test | Result | Durable implication |
|---|---:|---|
| BC/Brazil/Indonesia best LOO AP | 0.2731 / 0.1817 / 0.2021 | Some regional signal exists; ceilings differ materially |
| 32-region pooled flat model | AP 0.108 | Global flat pooling loses local structure |
| TabPFN v2 | No-go, p=0.33 | Few-shot specialization did not help |
| Admin-adjacency GNN | No-go, p=0.85; ΔAP +0.0025 | Graph architecture alone did not help |
| Competing-risks hazard | No-go; BC ΔAP −0.0599 | Hazard reframing did not rescue the existing data |
| Hierarchical meta-regression | No-go, p=0.92; ΔAP −0.0011 | Region-level correction insufficient |
| Conservation-finance features | **Only confirmed gain:** political-block R² 0.0251→0.0374; pooled AP 0.094→0.108 | Finance is the strongest grounded lead |

Not tested/therefore not ruled out: neural spatiotemporal point process; B5H within-country hierarchy; final Recall@5% after finance enrichment. Reopening as Q1b is justified by a new published benchmark, true-prevalence temporal estimand, event audit, and two-stage architecture—not by trying more algorithms.

### Question 2 — paused: “Two Logics” / designation regimes

- Pilot: 51,832 events, 26 countries; GMM BIC did not plateau through k=30, arguing against crisp discrete global regimes.
- Weighted vs unweighted clustering ARI 0.465: regime labels are sensitive to reporting granularity/area weighting.
- REDD+/carbon entry DiD: regime/frontier outcome ATT +0.135 (SE 0.106; inconclusive); direct forest-loss-pre outcome +0.004 (SE 0.022; null).
- Country regime classifier: LOO accuracy 0.462, equal to majority baseline; no-go at pilot scale.
- Pillar 2 paper-park script exists but awaited post/ring forest-loss data.

Implication: do not build a Nature claim around discrete regimes without new evidence. Continuous niche structure is currently more defensible. Reuse the event dataset, DiD machinery, and forest-loss outcomes if they support the active finance-frontier question.

### PA3030 — frozen read-only reference

South America two-stage model: country expansion Poisson D²=0.455; within-country LambdaRank ceiling ~20% macro Recall@5% despite oracle 97.5%. It independently found heterogeneous siting logics (adjacent expansion versus carbon/REDD+-associated siting). B5H was designed but never built. Never edit, commit to, or launch jobs in `/Users/gianluca/Desktop/thesis/code/`.

### Superseded claim

The earlier “governments protect cheap land, creating unpriced 30×30 financial risk” framing is not active. Useful retained fact: BC mining assets were 1.85× overrepresented in high predicted-PA-risk zones; agricultural/mineral-tenure land was below random. Treat as a possible secondary result only.

---

## Hard-won rules — do not relearn these

### Scientific design

- Governments designate polygons/sites, not independent pixels. Evaluate and resample by legal event/site.
- True prevalence belongs in evaluation. Controlled negative sampling is acceptable only for training.
- Static PA occurrence, future designation, amount of country expansion, and within-country siting are different estimands.
- Flat pooling across heterogeneous regions failed badly; separate country propensity from local choice.
- Mean-aggregating pixels into polygons/watersheds erased sparse extreme siting signals; retain directional summaries (`min`/`max`) and test the aggregation choice.
- Administrative reporting granularity is not event intensity (e.g. many tiny German records versus fewer huge Brazilian sites); report event-, area-, and country-weighted sensitivities.
- Coordinates, PA adjacency, current PA density, and country identity are powerful leakage/confounding proxies. Always test performance without them and against them as explicit baselines.
- Country-level variables repeated across cells do not constitute local evidence. Use held-out-country tests.
- For forecasting, temporal splits are mandatory. For nonforecast event typologies, use event/country-aware splits; never random cells.
- Biodiversity cannot be both the prioritisation input and validating outcome without an independent consequence test.
- Prediction is not causation; partial-response plots and feature importance identify associations, not levers.

### Data and implementation

- Run `audit_dataset.py --group-col <region>` after every build/pool. Global missingness summaries missed region-identity leaks.
- Enforce feature availability with `pipeline/leakage_guard.py`; a real AEF future leak once improved-looking results falsely.
- VIF-prune governance/institutional indices; WGI/V-Dem/DPI collinearity reached extreme levels.
- Inspect repeated sentinel/fill values before clustering; one constant generated a false GMM cluster.
- Clip/winsorise unstable ratio/CV variables using training data only; `ndvi_cv` reached 150,882 and dominated distance models.
- Use reproducible hashes (`hashlib.sha256`), not Python `hash()`, for stable splits.
- Audit merge keys/casing and global site-ID duplicates after cross-country pulls.
- Avoid account-gated/manual sources unless no defensible open alternative exists.

### Data storage (2026-09-30)

- Desktop is iCloud-synced; big data must not live there (iCloud + hidden Drive/iCloud caches filled the Mac disk). Code, STATUS, docs stay on Desktop + GitHub; hard-to-recreate data lives on Google Drive "Meine Ablage" (Drive app must be in Stream mode, Desktop NOT in "Mein Mac" backup).
- Moved to Drive `pa_model_local_data/` and symlinked back (paths unchanged, need Drive running): `regions/british_columbia/data` (AEF/GEE exports, `covariate_stack.tif`, `pa_dataset.parquet`, target, raw vectors), `regions/british_columbia/outputs/maps`, `regions/shared/KBA_Data.zip` (KBA is registration-gated).
- Frozen thesis data moved to Drive `thesis_local_data/` (data/south_america, se_asia, usa, shared/KBA, WDPA May 2026 csv, untracked outputs >5 MB); thesis repo untouched apart from deleting a dead partial pack in `.git`.
- Deleted as re-downloadable: GRIP4 shapefiles (`1_fetch/grip4_density.py`), HydroBASINS zips (`build_hydrobasins_catalog.py`), HydroRIVERS zips (`rivers.py`), BC raw ESA CCI AGB tiles + `hybas_na` (`agb.py`), thesis ESA biomass 2010 tiles, V-Dem v15, ecoregions. File list and sources: `MANIFEST.md` in the Drive folders.
- Rebuild from scratch: re-run `1_fetch/run_all.sh`; put fetched downloads in `regions/shared/downloads` only temporarily and delete or move to Drive after use.

### GEE and compute

- Use batch exports, not large interactive `getInfo()` loops; batch was materially faster and more reliable.
- Binary global vector rasters: prefer `ee.Image(0).byte().paint(fc, 1)` over global `reduceToImage()`.
- Simplify polygons at working scale before buffering; use `tileScale=8` where reducers hit memory ceilings.
- Pre-chunk using empirically viable event counts; do not discover batch size through repeated giant failures.
- Check paired output assets independently and recognise already-completed depth-0 assets before retrying.
- Set a socket timeout for `getInfo()`; client-side concurrency can be slower than sequential work.
- Fresh GEE projects may need one Code Editor visit to provision the asset root.
- Bounds rectangles can pull unrelated territory for elongated, border-adjacent, archipelagic, or antimeridian countries; verify every fallback against the true country geometry and globally deduplicate site IDs.
- Separate Google identities require isolated `HOME` credential stores; ETH policy blocks service-account key creation, so Project C needs notebook OAuth again if its ephemeral credentials/process are lost.
- Noncommercial quota exhaustion causes restricted-mode throttling regardless of nominal task concurrency.
- Euler: jobs >10 min or >12 GB; 128 GB is the user-wide concurrent ceiling. Compute nodes need `eth_proxy` for internet and cannot push Git.
- Never write git-tracked outputs inside scratch symlinks; use real report directories.

---

## Reusable assets

| Asset | Role |
|---|---|
| `pipeline/4_features/audit_dataset.py` | Missingness, duplicates, per-group integrity |
| `pipeline/leakage_guard.py` | Temporal feature-availability contract |
| `pipeline/5_model/vif_prune.py` | Governance/covariate collinearity control |
| `pipeline/1_fetch/fetch_wdpa_global.py` | Global WDPA attributes/event inputs |
| `pipeline/1_fetch/compute_wdpa_zonal_stats.py` | Event-level GEE zonal features |
| `pipeline/4_features/build_wdpa_global_dataset.py` | Global designation-event dataset |
| `pipeline/5_model/discover_regimes.py` | Continuous/discrete siting structure diagnostic |
| `pipeline/5_model/pillar1_redd_did.py` | Staggered finance/carbon entry analysis |
| `pipeline/5_model/pillar2_paper_park.py` | Post-designation forest-loss/ring outcome |
| `regions/global/pillar1_report.txt` | Existing DiD results |

Stable repository layout, regional feature catalogue, Euler environment, and symlink rules remain in `CLAUDE.md`; do not duplicate them here.

---

## Session index — compressed

- **S1–11:** BC proof of concept; initial test AP 0.399 before later leakage-safe LOO correction.
- **S12–17:** Brazil built; coordinate-dominated models failed cross-country transfer (AP ≈0.05).
- **S18–37:** AEF and covariate expansion; final leakage-safe ceilings BC 0.2731, Brazil 0.1817, Indonesia 0.2021; many spatial proxies/features ruled out.
- **S38–41:** horizon, watershed, and global-tile knob-turns failed; naive multi-country pooling made BC worse.
- **S42–50:** 32-region pool and architecture bake-off; hierarchical, TabPFN, GNN, and hazard approaches all no-go; AEF future leak found and fixed.
- **S50–51:** conservation finance produced the only confirmed gain (pooled AP 0.094→0.108); old Question 1 closed by rule.
- **S52–56:** Question 2 designed; 26-country/51,832-event dataset and robust GEE batch pipeline built; multiple scale, duplication, and leakage bugs fixed.
- **S57–58:** REDD+ DiD, paper-park pipeline, and regime discovery run; discrete regimes fragile, causal pilot inconclusive, country classifier no-go; full 234-country expansion launched.
- **S59 (2026-08-25):** Mouillot et al. audited conceptually; approved the temporal true-prevalence challenge and finance/biodiversity frontier as the new primary strategy; `STATUS.md` rebuilt around its execution and gates.
