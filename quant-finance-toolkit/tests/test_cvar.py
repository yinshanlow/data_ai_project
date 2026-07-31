import numpy as np
import pandas as pd
import pytest

from risk.cvar import apply_historical_scenario, historical_cvar, parametric_cvar
from risk.var import historical_var, parametric_var


@pytest.fixture
def returns():
    rng = np.random.default_rng(3)
    return pd.Series(rng.normal(0.0003, 0.018, 10_000))


def test_cvar_exceeds_var(returns):
    """CVaR is the average loss *beyond* VaR, so it must be at least as large as VaR."""
    assert historical_cvar(returns, 0.95) >= historical_var(returns, 0.95)
    assert parametric_cvar(returns, 0.95) >= parametric_var(returns, 0.95)


def test_cvar_increases_with_confidence(returns):
    for fn in (historical_cvar, parametric_cvar):
        assert fn(returns, 0.99) > fn(returns, 0.95)


def test_apply_historical_scenario_weighted_correctly():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    scenario = pd.Series({"A": -0.20, "B": 0.05})
    result = apply_historical_scenario(weights, scenario)
    expected = 0.6 * -0.20 + 0.4 * 0.05
    assert result.portfolio_shock == pytest.approx(expected)


def test_apply_historical_scenario_requires_overlapping_assets():
    weights = pd.Series({"A": 1.0})
    scenario = pd.Series({"Z": -0.1})
    with pytest.raises(ValueError):
        apply_historical_scenario(weights, scenario)
