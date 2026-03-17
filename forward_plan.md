Forward Predictions to 2030 — Implementation Plan

The Two-Model Separation

The key methodological principle is a clean evaluation/deployment split:





Evaluation model (already exists, untouched): trained 2001-2016, tested 2017-2019. All reported metrics, PR curves, SHAP plots, risk maps, calibration diagrams stay exactly as-is. Nothing changes.



Deployment model (new): trained on all non-censored labeled years (2001-2019) using the already-locked hyperparameters and best_iteration. Used exclusively for the 2030 forward projection.

This is standard ML practice: evaluate honestly on held-out data, then train the best possible model for actual deployment using all labeled data. The deployment model benefits from 3 additional years of post-Paris-Agreement (2015) designation patterns, which are the most relevant for projecting post-COP15 (2022) behavior.

Forecast vs. Scenario — A Critical Distinction

These are two epistemologically different things and must be presented separately in the paper:





BAU Forecast: The model's genuine probabilistic prediction of where PA establishments are most likely under historical designation patterns. Output: a probability map of P(protection by ~2029). This is a model claim with measurable uncertainty, validated through backtesting.



30x30 Scenario Analysis: A normative policy exercise. Takes the model's spatial ranking of unprotected pixels and asks: "if governments needed to designate enough land to reach 30% coverage, which pixels would be designated first?" The scenario changes the quantity designated, not the probabilities themselves. It makes no predictive claim about government behavior — it projects the existing spatial preference pattern to a higher designation intensity.

The paper should explicitly state: "We present a probabilistic BAU forecast and a separate scenario analysis quantifying what meeting the 30x30 pledge would imply spatially if historical designation preferences were maintained."

Pipeline Overview

flowchart TD
    subgraph eval [Evaluation — Unchanged]
        A["Train 2001-2016"] --> B["Test 2017-2019"]
        B --> C["Metrics / Maps / SHAP"]
    end
    subgraph deploy [Deployment + Forward — New]
        D["Train 2001-2019\nlocked params + best_iteration"] --> E["Backtest\n4 historical origins"]
        D --> F["2024 Inference Set\nWDPA_prev==0"]
        E --> G["Backtest precision table + plot"]
        F --> H["forward_scored_2024.parquet"]
        H --> I["BAU Forecast Map"]
        H --> J["30x30 Scenario Maps"]
        H --> K["Gap Analysis"]
    end

Stage −1: Shared Utility — pixel_area_km2()

Location: scripts/regions/shared/geo_utils.py (new file)

All km² numbers in the forward pipeline must use latitude-corrected pixel areas. In EPSG:3857 (Web Mercator), a 1000m × 1000m projected pixel represents cos²(lat) km² of actual ground area. The distortion is significant for southern SA:





Equator (0°): 1.000 km²/pixel



10°S (central Brazil): 0.970 km²/pixel



30°S (Uruguay): 0.750 km²/pixel



50°S (Patagonia): 0.413 km²/pixel — naive counting overestimates by 2.4×

def pixel_area_km2(y_epsg3857: np.ndarray, pixel_size_m: float = 1000.0) -> np.ndarray:
    """Return ground area in km² for each EPSG:3857 pixel centre.

    Args:
        y_epsg3857: Array of EPSG:3857 northing coordinates (metres).
        pixel_size_m: Actual projected pixel side length in metres, extracted
                      from the raster transform (NOT assumed to be 1000).
                      Retrieve once via:
                          import rasterio
                          with rasterio.open(backbone_path) as src:
                              pixel_size_m = src.transform.a   # cell width
    """
    R = 6378137.0  # WGS84 semi-major axis
    lat_rad = 2.0 * np.arctan(np.exp(y_epsg3857 / R)) - np.pi / 2.0
    pixel_km = pixel_size_m / 1000.0
    return (pixel_km ** 2) * np.cos(lat_rad) ** 2  # actual ground area in km²

# Usage pattern (call once at script start, pass pixel_size_m everywhere):
#   with rasterio.open(backbone_path) as src:
#       PIXEL_SIZE_M = abs(src.transform.a)   # metres per pixel (cell width)
#   areas = pixel_area_km2(y_array, pixel_size_m=PIXEL_SIZE_M)

The y column (EPSG:3857 northing in metres, confirmed range −7,666,000 to +1,505,000 m) is present in all relevant parquet files. This function is called for every area sum: coverage baseline, scenario thresholds, country/biome breakdowns, and the gap analysis.

IMPORTANT: Do NOT hardcode pixel_size_m=1000. Extract it from the backbone raster transform. At typical Web Mercator resolutions the actual cell width can differ materially from 1000 m; naively assuming 1000 m would introduce a systematic area bias in every downstream km² figure.

Stage 0a: SA Coverage Baseline

Script: scripts/regions/south_america/8_forward/forward_coverage_baseline.py

merged_panel_final.parquet contains only unprotected pixels (risk-set filter WDPA_prev == 0 applied during feature engineering). It cannot be used alone for coverage calculations.

The correct source: backbone raster + WDPA 2024 raster, both at data/south_america/ready/.

# backbone.tif  → defines total SA land pixels (20,785,671 pixels confirmed)
# WDPA_2024.tif → WDPA status for all SA pixels in 2024

# For each backbone pixel, extract its EPSG:3857 y coordinate from the raster transform
# total_sa_km2 = sum(pixel_area_km2(y)) over all backbone==1 pixels
# protected_2024_km2 = sum(pixel_area_km2(y)) where WDPA_2024==1 AND backbone==1
# coverage_pct_2024 = protected_2024_km2 / total_sa_km2

Output: outputs/south_america/results/forward/forward_coverage_baseline.json

{
  "total_sa_pixels": 20785671,
  "total_sa_km2": "<latitude-corrected sum>",
  "protected_2024_pixels": "<count>",
  "protected_2024_km2": "<latitude-corrected sum>",
  "coverage_pct_2024": "<float>",
  "km2_needed_for_25pct": "<float>",
  "km2_needed_for_30pct": "<float>"
}

These numbers drive all scenario thresholds downstream.

Stage 0b: Deployment Model Training

Script: scripts/regions/south_america/5_training/model1_LGBM_deployment.py

Thin variation of the existing [model1_LGBM](scripts/regions/south_america/5_training/model1_LGBM) with three changes:





TRAIN_YEARS = (2001, 2019) — full non-censored labeled window



--cv-mode none — skip CV; num_boost_round taken from lgbm_best_params.json best_params["n_estimators"] (= 2555, already tuned). The existing training code reads this as: chosen_iteration = best_params.get("num_boost_round", best_params.get("n_estimators", 3000)).



No test-set scoring

IMPORTANT — scale_pos_weight: The existing training code ALWAYS pops scale_pos_weight from the loaded JSON and recomputes it as n_neg / n_pos from the actual training data. The deployment script must follow the same pattern. Do NOT lock scale_pos_weight=30.0 from the JSON; that value was computed for the subsampled tuning data, not the full 2001–2019 panel.

Output: data/south_america/ml/models/model1_lgbm_deployment_{timestamp}.pkl

Calibration — Deployment Calibration Split (not "Tuning Fold 4"):

Train an auxiliary model on 2001–2016 (locked hyperparameters, same n_estimators=2555, scale_pos_weight recomputed from data), score 2017–2019 pixels as a held-out calibration set. Fit Platt + isotonic calibrators on these predictions.

These predictions are out-of-sample with respect to the deployment model's extra training years (2017–2019). However, they are predictions from an AUXILIARY model (trained on 2001–2016), not from the deployment model itself. The calibrator parameters are thus fitted on a proxy model's score distribution. This is unavoidable — a model trained on all available labeled data has no proper held-out calibration set — but it is a known approximation that must be disclosed in the paper (see Key Methodological Notes).

Naming: call this the "deployment calibration split" in all code, comments, and documentation. Do NOT call it "OOF Fold 4". Tuning Fold 4 in lgbm_best_params.json is a different split entirely (train ≤ 2013, val 2014–2016) and the names would be misleading to any reader comparing the two files.

Stage 0c: Pseudo-Forecast Backtesting

Script: scripts/regions/south_america/8_forward/forward_backtest.py

Validates the forward prediction methodology by simulating the analysis at four historical time points:

For each origin year T ∈ {2013, 2015, 2017, 2019}:





Train a historical deployment model on 2001–(T−1) with locked hyperparameters + n_estimators=2555 (scale_pos_weight recomputed from data, same pattern as deployment model)



Score T-feature rows for WDPA_prev==0 AND WDPA==0 pixels at year T (from merged_panel_final.parquet) — see Stage 1 for why the WDPA==0 filter is required



Evaluate against actual establishments in (T, T+5] — Precision@1/5/10%, Lift@1/5/10%, Forecast Capture Rate

IMPORTANT — constructing the (T, T+5] actuals: merged_panel_final.parquet contains only transition_01 (the 1-year target). It does NOT contain transition_01_win5. The backtest script must reconstruct the 5-year window target on the fly:

    For each pixel (row, col) with WDPA_prev==0 AND WDPA==0 at year T, read the WDPA column for years T+1 … T+5 from merged_panel_final.parquet and assign backtest_label=1 if WDPA==1 in any of those years (pixel first transitioned within the 5-year window).

    Pixels where T+5 > LAST_LABEL_YEAR (=2019) must be excluded from quantitative evaluation to avoid right-censoring bias. This means only T ∈ {2013, 2015} have fully clean 5-year windows; T=2017 has its window cut at 2019 (only 2 of 5 years observable) and should be flagged; T=2019 has no evaluable window and can only be used as a consistency check against the existing Future Capture Rate metric.

Note: T=2019 closely mirrors the existing Future Capture Rate metric, providing a consistency check.

Key claim this enables: "At each historical origin, the methodology correctly identified [X]% of subsequent PA establishments in the top 5% of predicted pixels, demonstrating consistent and validated performance."

Outputs:





forward_backtest_results.json



forward_backtest_precision_over_time.pdf — Precision@K% vs. forecast origin year

Stage 1: Inference Set Extraction

Script: scripts/regions/south_america/8_forward/forward_features.py





Read data/south_america/ml/merged_panel_final.parquet, filter to year==2024, WDPA_prev==0 AND WDPA==0

CRITICAL — why both filters are required:
- WDPA_prev==0: pixel was not protected as of 2023 (the risk-set condition consistent with training).
- WDPA==0: pixel was not protected in 2024 either (i.e., not newly designated during 2024).
  Without this second filter, pixels that became protected in 2024 pass through. Those pixels have
  WDPA=1 and dist_wdpa≈0 in their year-2024 feature row, causing the model to assign them
  inflated predicted probabilities. They would appear as "top predicted future PAs" despite already
  being protected. The WDPA==0 filter ensures the inference set represents "unprotected as of
  end-2024," matching the Stage 0a coverage baseline which uses the WDPA_2024 raster.

Expected size after both filters: ~14–15M pixels.



Select the exact 73 training features (from lgbm_best_params.json feature_list_used) + retain y coordinate for area correction



Output: outputs/south_america/results/forward/forward_features_2024.parquet

Feature note: dist_wdpa in year-2024 rows reflects 2023 WDPA boundaries (computed from WDPA_prev per the feature engineering design, to be consistent with training-time feature construction). This is correct and expected — do not treat it as an error.

Stage 2: Calibrated Inference

Script: scripts/regions/south_america/8_forward/forward_predict.py





Load model1_lgbm_deployment_*.pkl + deployment calibrators



Batch-stream through forward features; apply model.predict() + calibration



Output: outputs/south_america/results/forward/forward_scored_2024.parquet





Columns: row, col, x, y, y_pred_proba_raw, y_pred_proba_calibrated

Stage 3: Results, Maps, and Analysis

Script: scripts/regions/south_america/8_forward/forward_results.py

BAU Forecast

The model's actual output: P(protection by ~2029) for each currently-unprotected pixel under historical patterns.





forward_probability_map.pdf — continuous probability (reuses create_probability_map() from results_core.py)



forward_risk_map_bau.pdf — binary map of the top X% of unprotected area (by y_pred_proba_calibrated, ranked by descending probability, with area-weighted cutoff at the historical 5-year designation volume)

Historical designation volume: derived from coverage baseline (new km² protected 2019→2024 / 5 years × 5 years). This is the expected BAU designation under current rates.

30x30 Scenario Analysis

Scenario threshold computation:





Sort unprotected 2024 pixels by y_pred_proba_calibrated descending



Compute cumulative pixel_area_km2(y) along the ranked list



Find cutoff R where cumulative area first reaches the target from forward_coverage_baseline.json







Scenario



Target area



Output





30x30 Moderate



km2_needed_for_25pct



forward_scenario_moderate.pdf





30x30 Full



km2_needed_for_30pct



forward_scenario_30x30.pdf (headline figure)

Country and Biome Breakdowns

All area totals use pixel_area_km2(y). Country boundaries reuse boundaries.py. Biome assignment reuses the GSN spatial join from spatial_CV_2.

Per-country columns: current_protected_km2, current_coverage_pct, bau_new_km2, 30x30_required_km2, 30x30_shortfall_km2

Outputs: forward_country_breakdown.csv/.tex, forward_biome_breakdown.csv

Gap Analysis (Core Academic Contribution)

Compares where protection will go (BAU forecast) vs. where it should go (biodiversity priority):





"Should protect" proxy: GSN_b1 == 1 pixels (GlobalSafetyNet high-priority areas, already in features)



Biodiversity Capture Rate (BCR) = sum(area_km2 of top-K ∩ GSN_b1==1) / sum(area_km2 of all unprotected GSN_b1==1)



Compare: BCR under BAU forecast, BCR under 30x30 full scenario, BCR under random baseline



All area-weighted via pixel_area_km2(y)

This quantifies the thesis's central claim: does 30x30 expansion following historical patterns actually protect the most biodiverse land?

Output: forward_gap_analysis.pdf — 4-panel map: (BAU predicted designations) | (GSN-priority unprotected) | (overlap) | (gap: high-biodiversity left unprotected)

Full Output Deliverables

All outputs to outputs/south_america/results/forward/:





forward_coverage_baseline.json



forward_backtest_results.json + forward_backtest_precision_over_time.pdf



forward_probability_map.pdf/.png — BAU forecast, continuous



forward_risk_map_bau.pdf — BAU forecast, binary



forward_scenario_moderate.pdf — 30x30 moderate scenario



forward_scenario_30x30.pdf — 30x30 full scenario (headline thesis figure)



forward_gap_analysis.pdf — 4-panel biodiversity gap map



forward_country_breakdown.csv/.tex — area-corrected country-level breakdown



forward_biome_breakdown.csv — area-corrected biome-level breakdown



forward_scenario_summary.json — all structured numbers for paper tables

Key Methodological Notes for the Paper





Evaluation vs. deployment: All performance metrics use the evaluation model (2001-2016 / 2017-2019). The deployment model section: "trained on all non-censored labeled years with locked hyperparameters."



Forecast vs. scenario: Explicitly separate. BAU forecast = probabilistic model output. 30x30 scenarios = policy exercise using model rankings.



Validation chain: Test-set AUC → Future Capture Rate (2020-2024) → pseudo-forecast backtesting (T=2013, 2015, clean 5-year windows) → 2024 forward projection.



Area correction: All km² figures use pixel_area_km2() with pixel size extracted from the backbone raster transform. State: "Pixel areas were corrected for the Web Mercator scale factor using the EPSG:3857 northing coordinate and the actual projected pixel dimensions from the backbone raster."



Coverage source: SA land area and 2024 protection coverage derived from backbone raster + WDPA 2024 raster directly, not from the risk-set-filtered panel.



Calibrated probabilities for thresholds: Scenario cutoffs are area-accumulated along the calibrated-probability-ranked pixel list.



Deployment model calibration disclosure: "Probability calibration for the deployment model used a Platt/isotonic calibrator fitted on out-of-sample predictions from an auxiliary model trained on 2001–2016 data, scored on 2017–2019 held-out data (the deployment calibration split). Because the deployment model itself is trained on all labeled years, no directly held-out calibration set exists; the auxiliary model's predictions serve as a practical approximation. The calibration transfer assumption — that the score-to-probability mapping is similar between the two models — is reasonable given identical hyperparameters and training data distribution, but cannot be verified without independent data."

Repo Structure

scripts/regions/shared/
└── geo_utils.py                         # pixel_area_km2() and related helpers

scripts/regions/south_america/
├── 5_training/
│   └── model1_LGBM_deployment.py        # deployment model (2001-2019)
└── 8_forward/
    ├── __init__.py
    ├── forward_coverage_baseline.py     # Stage 0a
    ├── forward_backtest.py              # Stage 0c
    ├── forward_features.py              # Stage 1
    ├── forward_predict.py               # Stage 2
    └── forward_results.py               # Stage 3

slurm/south_america/
├── forward_deployment.slurm            # Stage 0b deployment training + Stage 1-2 inference (~64GB RAM)
└── forward_backtest.slurm             # Stage 0c backtest — SLURM array job, one task per origin year
                                        # (4 models × ~deployment training time; do NOT chain sequentially
                                        # in forward_deployment.slurm — use #SBATCH --array=0-3)


To dos:
- Add pixel_area_km2(y_epsg3857, pixel_size_m) helper to shared/geo_utils.py: converts EPSG:3857 northings to ground area in km² using cos²(lat) correction scaled by actual pixel size; pixel_size_m MUST be extracted from the backbone raster transform (rasterio src.transform.a), NOT hardcoded to 1000; use everywhere area/coverage is calculated

- Compute SA coverage baseline from backbone.tif + WDPA 2024 raster (NOT from merged_panel_final.parquet alone): total_sa_km2, protected_2024_km2, coverage_pct_2024 — store in forward_coverage_baseline.json

- Build model1_LGBM_deployment.py: train on 2001-2019 with locked hyperparameters + n_estimators=2555 (cv-mode none); scale_pos_weight recomputed from data (NOT locked from JSON); calibrate via deployment calibration split (auxiliary model train 2001-2016, score 2017-2019) — do NOT call this "OOF Fold 4"

- Build forward_backtest.py: for each origin year T in {2013, 2015, 2017, 2019}, train a historical deployment model on 2001-(T-1), score year-T features for WDPA_prev==0 AND WDPA==0 pixels, reconstruct 5-year window actuals from yearly WDPA column in merged_panel_final.parquet (NOT from transition_01_win5 — that column does not exist there), measure Precision@K%/Lift@K% against (T, T+5] actuals; exclude pixels where T+5 > 2019 from evaluation; run as SLURM array job (one task per T)

- Build forward_features.py: filter merged_panel_final.parquet to year==2024, WDPA_prev==0 AND WDPA==0 (both filters required — see Stage 1), output inference parquet (~14–15M rows × 73 features + y coordinate for area correction)

- Build forward_predict.py: load deployment model + deployment calibrators, run batched inference on 2024 features, output forward_scored_2024.parquet (row, col, x, y, y_pred_proba_raw, y_pred_proba_calibrated)

- Build forward_results.py: BAU forecast probability map + binary map; 30x30 scenario maps (moderate + full); gap analysis 4-panel map; all using area-corrected km² thresholds; use results_core.py functions with proper SA region config (X_LIMITS, Y_LIMITS, etc.)

- Country and biome breakdown tables with latitude-corrected km² and coverage percentages; forward_scenario_summary.json for paper tables

- Implement Biodiversity Capture Rate for BAU forecast and 30x30 Full scenario vs. random baseline using GSN_b1; all area-weighted

- Add slurm/south_america/forward_deployment.slurm for Euler (Stage 0b deployment training + Stage 1-2 inference, ~64GB RAM) and separate forward_backtest.slurm as array job (Stage 0c, one task per origin year); Stage 3 maps can run locally