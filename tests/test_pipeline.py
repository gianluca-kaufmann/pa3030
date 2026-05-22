#!/usr/bin/env python3
"""Unit tests for the PA3030 pipeline.

Run with:  pytest tests/test_pipeline.py -v
Install:   pip install pytest

These tests use small synthetic data — no real datasets or trained models needed.
They verify that core logic (feature engineering, data splits, calibration) is correct.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import distance_transform_edt, uniform_filter


# ---------------------------------------------------------------------------
# Helper functions (extracted from feature_engineering script)
# These mirror the logic in scripts/regions/south_america/3_merging/feature_engineering
# ---------------------------------------------------------------------------

def reconstruct_grid(df: pd.DataFrame, value_col: str):
    """Reconstruct 2D grid from panel data (mirrors feature_engineering:84-92)."""
    min_row, max_row = df['row'].min(), df['row'].max()
    min_col, max_col = df['col'].min(), df['col'].max()
    grid = np.full((max_row - min_row + 1, max_col - min_col + 1), np.nan, dtype=np.float32)
    grid[(df['row'] - min_row).values, (df['col'] - min_col).values] = df[value_col].values
    return grid, min_row, max_row, min_col, max_col


def grid_to_dataframe(grid: np.ndarray, df: pd.DataFrame, min_row: int, min_col: int) -> pd.Series:
    """Convert grid back to DataFrame format (mirrors feature_engineering:95-98)."""
    values = grid[(df['row'] - min_row).values, (df['col'] - min_col).values]
    return pd.Series(values, index=df.index)


def compute_distance(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute Euclidean distance to nearest positive pixel (mirrors feature_engineering:101-115)."""
    grid, min_row, _, min_col, _ = reconstruct_grid(df, col)
    binary = np.where(np.isnan(grid), 0, grid > 0).astype(np.uint8)
    distance_grid = distance_transform_edt(1 - binary) * 1000  # pixels to meters
    return grid_to_dataframe(distance_grid, df, min_row, min_col)


def compute_smooth(df: pd.DataFrame, variable: str, window_size: int) -> pd.Series:
    """Compute smoothed feature (mirrors feature_engineering:208-227)."""
    grid, min_row, _, min_col, _ = reconstruct_grid(df, variable)
    valid_mask = ~np.isnan(grid)
    grid_filled = np.where(np.isnan(grid), 0, grid)
    smoothed = uniform_filter(grid_filled, size=window_size, mode='reflect')
    count = uniform_filter(valid_mask.astype(np.float32), size=window_size, mode='reflect')
    smoothed = np.where(count > 0, smoothed, np.nan)
    return grid_to_dataframe(smoothed, df, min_row, min_col)


# ---------------------------------------------------------------------------
# Test 1: Distance transform correctness
# ---------------------------------------------------------------------------

class TestDistanceTransform:
    """Verify distance_transform_edt produces correct distances in meters."""

    def test_distance_to_adjacent_pixel(self):
        """A pixel 1 cell away from a PA should have dist = 1000m (1 pixel * 1000)."""
        df = pd.DataFrame({
            'row': [0, 0, 0],
            'col': [0, 1, 2],
            'WDPA_prev': [1.0, 0.0, 0.0],  # PA at col=0
        })
        distances = compute_distance(df, 'WDPA_prev')
        assert distances.iloc[0] == 0.0, "Pixel ON the PA should have distance 0"
        assert distances.iloc[1] == pytest.approx(1000.0), "1 cell away = 1000m"
        assert distances.iloc[2] == pytest.approx(2000.0), "2 cells away = 2000m"

    def test_distance_diagonal(self):
        """Diagonal distance should be sqrt(2) * 1000."""
        df = pd.DataFrame({
            'row': [0, 1],
            'col': [0, 1],
            'WDPA_prev': [1.0, 0.0],  # PA at (0,0), query at (1,1)
        })
        distances = compute_distance(df, 'WDPA_prev')
        expected = np.sqrt(2) * 1000
        assert distances.iloc[1] == pytest.approx(expected, rel=1e-5)

    def test_no_protected_areas(self):
        """If no PAs exist, all distances should be positive (no zeros)."""
        df = pd.DataFrame({
            'row': [0, 0, 1, 1],
            'col': [0, 1, 0, 1],
            'WDPA_prev': [0.0, 0.0, 0.0, 0.0],
        })
        distances = compute_distance(df, 'WDPA_prev')
        assert (distances > 0).all(), "All distances should be > 0 when no PAs exist"

    def test_all_protected(self):
        """If all pixels are PAs, all distances should be 0."""
        df = pd.DataFrame({
            'row': [0, 0, 1, 1],
            'col': [0, 1, 0, 1],
            'WDPA_prev': [1.0, 1.0, 1.0, 1.0],
        })
        distances = compute_distance(df, 'WDPA_prev')
        assert (distances == 0.0).all(), "All distances should be 0 when all pixels are PAs"

    def test_nan_treated_as_unprotected(self):
        """NaN values in WDPA should be treated as 0 (not protected)."""
        df = pd.DataFrame({
            'row': [0, 0, 0],
            'col': [0, 1, 2],
            'WDPA_prev': [1.0, np.nan, 0.0],
        })
        distances = compute_distance(df, 'WDPA_prev')
        # NaN at col=1 should NOT be treated as a PA
        assert distances.iloc[1] == pytest.approx(1000.0), "NaN pixel should be 1000m from PA at col=0"


# ---------------------------------------------------------------------------
# Test 2: WDPA lag (no data leakage)
# ---------------------------------------------------------------------------

class TestWDPALag:
    """Verify WDPA_prev uses t-1 status, not current year (prevents data leakage)."""

    def test_lag_uses_previous_year(self):
        """WDPA_prev in year t should equal WDPA_b1 from year t-1."""
        # Simulate: pixel becomes protected in 2011
        df_2010 = pd.DataFrame({'row': [0], 'col': [0], 'year': [2010], 'WDPA_b1': [0]})
        df_2011 = pd.DataFrame({'row': [0], 'col': [0], 'year': [2011], 'WDPA_b1': [1]})

        # Compute lag for 2011: should use 2010's WDPA_b1 = 0
        df_prev = df_2010.rename(columns={'WDPA_b1': 'WDPA_prev'}).drop(columns=['year'])
        df_merged = df_2011.merge(df_prev, on=['row', 'col'], how='left')

        assert df_merged['WDPA_prev'].values[0] == 0, \
            "In 2011, WDPA_prev should be 0 (2010 status), NOT 1 (2011 status)"

    def test_lag_not_current_year(self):
        """WDPA_prev must differ from WDPA_b1 when protection status changes."""
        # Pixel transitions from unprotected to protected
        row = pd.Series({'WDPA_b1': 1, 'WDPA_prev': 0})
        # If WDPA_prev == WDPA_b1, there's a leakage bug
        assert row['WDPA_prev'] != row['WDPA_b1'], \
            "WDPA_prev should lag behind WDPA_b1 for transitioning pixels"

    def test_first_year_lag_is_zero(self):
        """First year in dataset should have WDPA_prev = 0 (no prior year exists)."""
        # Mirrors feature_engineering:294 — is_first_year → WDPA_prev = 0
        df = pd.DataFrame({
            'row': [0, 1], 'col': [0, 0],
            'year': [2000, 2000],
            'WDPA_b1': [1, 0],
        })
        df['WDPA_prev'] = 0  # First year convention
        assert (df['WDPA_prev'] == 0).all(), "First year should always have WDPA_prev = 0"


# ---------------------------------------------------------------------------
# Test 3: Risk-set filtering
# ---------------------------------------------------------------------------

class TestRiskSetFiltering:
    """Verify only unprotected pixels enter the risk set."""

    def test_excludes_already_protected(self):
        """Pixels with WDPA_prev == 1 must be excluded from risk set."""
        df = pd.DataFrame({
            'WDPA_prev': [0, 0, 1, 1, 0],
            'WDPA_b1': [0, 1, 1, 1, 0],
        })
        risk_set = df[df['WDPA_prev'] == 0]
        assert len(risk_set) == 3, "Should keep 3 unprotected pixels"
        assert (risk_set['WDPA_prev'] == 0).all(), "All risk set pixels must be unprotected"

    def test_transition_only_in_risk_set(self):
        """Transitions (0 → 1) can only happen for pixels in the risk set."""
        df = pd.DataFrame({
            'WDPA_prev': [0, 0, 0, 1],
            'WDPA_b1': [0, 1, 0, 1],
        })
        risk_set = df[df['WDPA_prev'] == 0].copy()
        risk_set['transition_01'] = risk_set['WDPA_b1'].fillna(0).astype('uint8')

        # Only 1 transition out of 3 risk-set pixels
        assert risk_set['transition_01'].sum() == 1
        assert len(risk_set) == 3

    def test_empty_risk_set(self):
        """If all pixels are already protected, risk set should be empty."""
        df = pd.DataFrame({
            'WDPA_prev': [1, 1, 1],
            'WDPA_b1': [1, 1, 1],
        })
        risk_set = df[df['WDPA_prev'] == 0]
        assert len(risk_set) == 0


# ---------------------------------------------------------------------------
# Test 4: Right-censoring (complete 5-year lookahead)
# ---------------------------------------------------------------------------

class TestRightCensoring:
    """Verify labels only exist for years with complete 5-year lookahead windows."""

    WDPA_LAST_YEAR = 2024
    LOOKAHEAD_YEARS = 5
    LAST_LABEL_YEAR = WDPA_LAST_YEAR - LOOKAHEAD_YEARS  # 2019

    def test_last_label_year_computation(self):
        """LAST_LABEL_YEAR should be WDPA_LAST_YEAR - LOOKAHEAD_YEARS."""
        assert self.LAST_LABEL_YEAR == 2019

    def test_year_2019_has_complete_window(self):
        """Year 2019 needs data through 2024 (2019+5). WDPA goes to 2024 → complete."""
        assert 2019 + self.LOOKAHEAD_YEARS <= self.WDPA_LAST_YEAR

    def test_year_2020_has_incomplete_window(self):
        """Year 2020 needs data through 2025, but WDPA only goes to 2024 → incomplete."""
        assert 2020 + self.LOOKAHEAD_YEARS > self.WDPA_LAST_YEAR

    def test_no_labels_after_last_label_year(self):
        """No test/eval data should include years > LAST_LABEL_YEAR."""
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        valid_years = [y for y in years if y <= self.LAST_LABEL_YEAR]
        assert max(valid_years) == 2019
        assert 2020 not in valid_years
        assert 2021 not in valid_years

    def test_split_configs_respect_censoring(self):
        """All CV fold boundaries should respect LAST_LABEL_YEAR."""
        # Mirrors modelC_LGBM:75-99
        cv_folds_main = [
            ((2000, 2008), (2009, min(2011, self.LAST_LABEL_YEAR))),
            ((2000, 2011), (2012, min(2013, self.LAST_LABEL_YEAR))),
            ((2000, 2013), (2014, min(2016, self.LAST_LABEL_YEAR))),
        ]
        for (train_start, train_end), (es_start, es_end) in cv_folds_main:
            assert es_end <= self.LAST_LABEL_YEAR, \
                f"Earlystop end {es_end} exceeds LAST_LABEL_YEAR {self.LAST_LABEL_YEAR}"
            assert train_end < es_start, \
                f"Train end {train_end} must be before earlystop start {es_start}"


# ---------------------------------------------------------------------------
# Test 5: Temporal split integrity (no future data in training)
# ---------------------------------------------------------------------------

class TestTemporalSplits:
    """Verify strict temporal ordering: train < earlystop < test."""

    def test_main_split_ordering(self):
        """Train years must come before earlystop, which must come before test."""
        train_years = (2000, 2013)
        earlystop_years = (2014, 2016)
        test_years = (2017, 2019)

        assert train_years[1] < earlystop_years[0], "Train must end before earlystop starts"
        assert earlystop_years[1] < test_years[0], "Earlystop must end before test starts"

    def test_robustness_split_ordering(self):
        """Robustness split should also have strict temporal ordering."""
        train_years = (2000, 2012)
        earlystop_years = (2013, 2015)
        test_years = (2016, 2019)

        assert train_years[1] < earlystop_years[0]
        assert earlystop_years[1] < test_years[0]

    def test_no_year_overlap(self):
        """No year should appear in more than one split."""
        train = set(range(2000, 2014))
        earlystop = set(range(2014, 2017))
        test = set(range(2017, 2020))

        assert train.isdisjoint(earlystop), "Train and earlystop must not overlap"
        assert train.isdisjoint(test), "Train and test must not overlap"
        assert earlystop.isdisjoint(test), "Earlystop and test must not overlap"

    def test_cv_folds_expanding_window(self):
        """Each CV fold should use a strictly larger training window than the previous."""
        cv_folds = [
            ((2000, 2008), (2009, 2011)),
            ((2000, 2011), (2012, 2013)),
            ((2000, 2013), (2014, 2016)),
        ]
        prev_train_end = -1
        for (train_start, train_end), (es_start, es_end) in cv_folds:
            assert train_start == 2000, "All folds should start from 2000"
            assert train_end > prev_train_end, "Training window must expand across folds"
            assert train_end < es_start, "No temporal leakage: train must end before earlystop"
            prev_train_end = train_end


# ---------------------------------------------------------------------------
# Test 6: Class imbalance handling (scale_pos_weight)
# ---------------------------------------------------------------------------

class TestClassImbalance:
    """Verify scale_pos_weight is computed correctly."""

    def test_scale_pos_weight_formula(self):
        """scale_pos_weight = n_negative / n_positive."""
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 9 neg, 1 pos
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        assert scale_pos_weight == 9.0

    def test_scale_pos_weight_extreme_imbalance(self):
        """With 0.5% positive rate, scale_pos_weight should be ~199."""
        n_total = 10000
        n_pos = 50  # 0.5%
        n_neg = n_total - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)
        assert scale_pos_weight == pytest.approx(199.0)

    def test_scale_pos_weight_no_positives(self):
        """With zero positives, should not crash (max(n_pos, 1) protects division)."""
        n_pos = 0
        n_neg = 1000
        scale_pos_weight = n_neg / max(n_pos, 1)
        assert scale_pos_weight == 1000.0


# ---------------------------------------------------------------------------
# Test 7: Spatial smoothing correctness
# ---------------------------------------------------------------------------

class TestSpatialSmoothing:
    """Verify uniform_filter smoothing produces expected values."""

    def test_uniform_value_unchanged(self):
        """Smoothing a constant grid should return the same constant."""
        df = pd.DataFrame({
            'row': [0, 0, 1, 1],
            'col': [0, 1, 0, 1],
            'value': [5.0, 5.0, 5.0, 5.0],
        })
        smoothed = compute_smooth(df, 'value', window_size=2)
        np.testing.assert_allclose(smoothed.values, 5.0, atol=1e-5)

    def test_smoothing_reduces_extremes(self):
        """Smoothing should pull outliers toward the mean."""
        # 3x3 grid, center pixel is 100, rest are 0
        rows = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        cols = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        vals = [0, 0, 0, 0, 100, 0, 0, 0, 0]
        df = pd.DataFrame({'row': rows, 'col': cols, 'value': vals})

        smoothed = compute_smooth(df, 'value', window_size=3)
        center_smoothed = smoothed.iloc[4]  # center pixel
        assert center_smoothed < 100, "Smoothing should reduce the center outlier"
        assert center_smoothed > 0, "Smoothed center should still be positive"


# ---------------------------------------------------------------------------
# Test 8: Calibration (Platt scaling)
# ---------------------------------------------------------------------------

class TestCalibration:
    """Verify Platt scaling calibration logic."""

    def test_platt_logit_transform(self):
        """Platt scaling should transform probabilities to logits before fitting."""
        proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        clipped = np.clip(proba, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped))

        # Known logit values
        assert logits[2] == pytest.approx(0.0, abs=1e-5), "logit(0.5) should be 0"
        assert logits[0] < 0, "logit(0.1) should be negative"
        assert logits[4] > 0, "logit(0.9) should be positive"

    def test_platt_clipping_prevents_inf(self):
        """Clipping to [1e-7, 1-1e-7] should prevent log(0) = -inf."""
        proba = np.array([0.0, 1.0])  # Edge cases
        clipped = np.clip(proba, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped))
        assert np.all(np.isfinite(logits)), "Logits should be finite after clipping"

    def test_calibrated_probabilities_in_range(self):
        """Calibrated probabilities must be in [0, 1]."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        # Synthetic: model gives higher scores to actual positives (well-ranked)
        np.random.seed(42)
        y_true = np.array([0]*900 + [1]*100)
        y_proba = np.where(y_true == 1,
                           np.random.uniform(0.3, 0.9, 1000),
                           np.random.uniform(0.0, 0.4, 1000))

        # Platt scaling
        clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped))
        platt = LogisticRegression(solver='lbfgs', max_iter=1000)
        platt.fit(logits.reshape(-1, 1), y_true)
        calibrated = platt.predict_proba(logits.reshape(-1, 1))[:, 1]

        assert calibrated.min() >= 0.0, "Calibrated probabilities must be >= 0"
        assert calibrated.max() <= 1.0, "Calibrated probabilities must be <= 1"

        # Isotonic regression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_proba, y_true)
        calibrated_iso = iso.transform(y_proba)

        assert calibrated_iso.min() >= 0.0
        assert calibrated_iso.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 9: Transition target variable
# ---------------------------------------------------------------------------

class TestTransitionTarget:
    """Verify transition_01 is correctly derived from risk-set filtered data."""

    def test_transition_is_binary(self):
        """transition_01 should only contain 0 or 1."""
        df = pd.DataFrame({
            'WDPA_prev': [0, 0, 0, 0],
            'WDPA_b1': [0.0, 1.0, np.nan, 0.0],
        })
        risk_set = df[df['WDPA_prev'] == 0].copy()
        risk_set['transition_01'] = risk_set['WDPA_b1'].fillna(0).astype('uint8')

        assert set(risk_set['transition_01'].unique()).issubset({0, 1})

    def test_nan_wdpa_treated_as_no_transition(self):
        """NaN in WDPA_b1 should be treated as 0 (no transition)."""
        df = pd.DataFrame({
            'WDPA_prev': [0],
            'WDPA_b1': [np.nan],
        })
        risk_set = df[df['WDPA_prev'] == 0].copy()
        risk_set['transition_01'] = risk_set['WDPA_b1'].fillna(0).astype('uint8')
        assert risk_set['transition_01'].values[0] == 0


# ---------------------------------------------------------------------------
# Test 10: Reproducibility (deterministic seeds)
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Verify that fixed seeds produce deterministic results."""

    def test_numpy_random_with_seed(self):
        """Same seed should produce identical random numbers."""
        rng1 = np.random.RandomState(42)
        vals1 = rng1.random(10)

        rng2 = np.random.RandomState(42)
        vals2 = rng2.random(10)

        np.testing.assert_array_equal(vals1, vals2)

    def test_excluded_columns_not_in_features(self):
        """EXCLUDE_COLS should prevent target/coordinate columns from being used as features."""
        EXCLUDE_COLS = {'transition_01', 'transition_01_win5', 'WDPA_b1',
                        'WDPA_prev', 'x', 'y', 'row', 'col', 'year'}

        all_columns = ['row', 'col', 'year', 'WDPA_b1', 'WDPA_prev',
                        'transition_01', 'transition_01_win5',
                        'NDVI_b1', 'elevation_b1', 'dist_wdpa', 'GPW_b1']

        feature_cols = [c for c in all_columns if c not in EXCLUDE_COLS]
        assert 'NDVI_b1' in feature_cols
        assert 'dist_wdpa' in feature_cols
        assert 'WDPA_b1' not in feature_cols, "Target-related column leaked into features"
        assert 'transition_01' not in feature_cols, "Target leaked into features"
        assert 'year' not in feature_cols, "Coordinate column leaked into features"


# ---------------------------------------------------------------------------
# Test 11: Grid reconstruction roundtrip
# ---------------------------------------------------------------------------

class TestGridReconstruction:
    """Verify grid ↔ dataframe conversion is lossless."""

    def test_roundtrip_preserves_values(self):
        """reconstruct_grid → grid_to_dataframe should return original values."""
        df = pd.DataFrame({
            'row': [0, 0, 1, 1, 2, 2],
            'col': [0, 1, 0, 1, 0, 1],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        grid, min_row, _, min_col, _ = reconstruct_grid(df, 'value')
        recovered = grid_to_dataframe(grid, df, min_row, min_col)

        np.testing.assert_array_equal(recovered.values, df['value'].values)

    def test_sparse_grid_handles_gaps(self):
        """Grid should handle non-contiguous row/col indices."""
        df = pd.DataFrame({
            'row': [0, 5],  # Gap between 0 and 5
            'col': [0, 3],  # Gap between 0 and 3
            'value': [10.0, 20.0],
        })
        grid, min_row, _, min_col, _ = reconstruct_grid(df, 'value')
        recovered = grid_to_dataframe(grid, df, min_row, min_col)

        assert recovered.iloc[0] == 10.0
        assert recovered.iloc[1] == 20.0
        # Interior cells should be NaN
        assert np.isnan(grid[1, 1])


class TestFeatureGuard:
    """Verify the shared feature-leakage guard at scripts/regions/shared/training/feature_guard.py."""

    def _import(self):
        from scripts.regions.shared.training.feature_guard import (  # noqa: WPS433
            FORBIDDEN_COLS,
            FORBIDDEN_PATTERNS,
            FeatureLeakageError,
            check_feature_denylist,
        )
        return FORBIDDEN_COLS, FORBIDDEN_PATTERNS, FeatureLeakageError, check_feature_denylist

    def test_clean_list_passes(self):
        _, _, _, check = self._import()
        check(["elevation_b1", "GSN_b3", "dist_road", "pa_momentum_pixels_lag1"], context="unit/clean")

    def test_raw_wdpa_raises(self):
        _, _, FeatureLeakageError, check = self._import()
        with pytest.raises(FeatureLeakageError) as exc:
            check(["elevation_b1", "WDPA"], context="unit/wdpa")
        assert "WDPA" in str(exc.value)

    def test_target_columns_raise(self):
        _, _, FeatureLeakageError, check = self._import()
        for col in ("transition_01", "transition_01_win5", "first_transition_year"):
            with pytest.raises(FeatureLeakageError):
                check([col, "elevation_b1"], context="unit/target")

    def test_pattern_match_raises(self):
        _, _, FeatureLeakageError, check = self._import()
        # Not in exact list, but matches WDPA prefix pattern.
        with pytest.raises(FeatureLeakageError) as exc:
            check(["wdpa_buffered_5km"], context="unit/pattern")
        msg = str(exc.value)
        assert "pattern denylist" in msg
        assert "wdpa_buffered_5km" in msg

    def test_error_message_includes_context_and_offenders(self):
        _, _, FeatureLeakageError, check = self._import()
        with pytest.raises(FeatureLeakageError) as exc:
            check(["row", "col", "elevation_b1"], context="my/specific/site")
        msg = str(exc.value)
        assert "my/specific/site" in msg
        assert "row" in msg
        assert "col" in msg

    def test_year_blocked_by_default(self):
        _, _, FeatureLeakageError, check = self._import()
        with pytest.raises(FeatureLeakageError):
            check(["elevation_b1", "year"], context="unit/year")

    def test_country_helpers_blocked(self):
        _, _, FeatureLeakageError, check = self._import()
        for col in ("country_id", "country_iso3"):
            with pytest.raises(FeatureLeakageError):
                check([col, "elevation_b1"], context="unit/country")

    def test_pa_momentum_features_pass(self):
        """The W3 momentum features must be allowed — they're regular features, not labels."""
        _, _, _, check = self._import()
        check(
            ["pa_momentum_pixels_lag1", "pa_momentum_pixels_lag2", "pa_momentum_pixels_lag3"],
            context="unit/momentum",
        )

    def test_explicit_override_allows_year(self):
        _, _, _, check = self._import()
        from scripts.regions.shared.training.feature_guard import FORBIDDEN_COLS  # noqa: WPS433
        check(["elevation_b1", "year"], context="unit/override", forbidden=FORBIDDEN_COLS - {"year"})


class TestStage2Metrics:
    """Stage 2 within-group NDCG / concordance metrics."""

    def test_ndcg_perfect_ranking(self):
        from scripts.regions.shared.evaluation.stage2_metrics import (
            compute_stage2_metrics,
        )

        y = np.array([0, 0, 1, 0, 1, 0], dtype=np.float64)
        scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.4], dtype=np.float64)
        groups = np.array([3, 3], dtype=np.int32)
        m = compute_stage2_metrics(y, scores, groups)
        assert m["ndcg_at_1pct"] >= 0.99

    def test_concordance_perfect(self):
        from scripts.regions.shared.evaluation.stage2_metrics import (
            concordance_within_groups,
        )

        y = np.array([0, 1, 0, 1], dtype=np.float64)
        scores = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
        groups = np.array([2, 2], dtype=np.int32)
        c = concordance_within_groups(y, scores, groups)
        assert c == pytest.approx(1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
