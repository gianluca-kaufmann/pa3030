import numpy as np
import pandas as pd
import pytest

from scripts.regions.world import paths
from scripts.regions.world.leakage_guard import (
    assert_dynamic_state_is_prior_only,
    assert_no_forbidden_columns,
)
from scripts.regions.world.vif_prune import iterative_prune


def test_country_year_tables_are_tracked_inputs():
    for path in [paths.GOVERNANCE_PATH, paths.FINANCE_PATH, paths.REDD_PATH]:
        assert path.exists(), path
        assert "country" in pd.read_csv(path, nrows=1).columns


def test_forbidden_column_raises():
    assert_no_forbidden_columns(["dist_to_pa_px", "ndvi"])
    with pytest.raises(RuntimeError):
        assert_no_forbidden_columns(["ndvi", "pa_year"])


def test_prior_state_mask_must_exclude_year_t():
    pa_year = np.array([0, 2005, 2010, 2012])
    assert_dynamic_state_is_prior_only(pa_year.astype(bool) & (pa_year < 2010), pa_year, 2010)
    with pytest.raises(RuntimeError):
        assert_dynamic_state_is_prior_only(pa_year.astype(bool) & (pa_year <= 2010), pa_year, 2010)


def test_vif_prune_drops_a_collinear_column():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=500), "a_copy": a + rng.normal(scale=0.01, size=500)})
    vifs, dropped = iterative_prune(X, threshold=10)
    assert len(dropped) == 1 and dropped[0][0] in {"a", "a_copy"}
    assert "b" in vifs
