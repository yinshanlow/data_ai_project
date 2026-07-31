import numpy as np
import pandas as pd
import pytest

from risk.portfolio import efficient_frontier, max_sharpe_portfolio, min_variance_portfolio


@pytest.fixture
def synthetic_returns():
    """Three assets with a known covariance structure — low-vol A, high-vol B, and C
    correlated with B — so the min-variance solution should tilt heavily toward A."""
    rng = np.random.default_rng(42)
    n = 1000
    mean = np.array([0.0003, 0.0008, 0.0006])
    cov = np.array([
        [0.0001, 0.0000, 0.0000],
        [0.0000, 0.0009, 0.0006],
        [0.0000, 0.0006, 0.0006],
    ])
    data = rng.multivariate_normal(mean, cov, n)
    return pd.DataFrame(data, columns=["A_low_vol", "B_high_vol", "C_correlated_with_B"])


def test_weights_sum_to_one_and_long_only(synthetic_returns):
    for result in (min_variance_portfolio(synthetic_returns), max_sharpe_portfolio(synthetic_returns)):
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)
        assert (result.weights >= -1e-6).all()


def test_min_variance_tilts_toward_low_vol_asset(synthetic_returns):
    result = min_variance_portfolio(synthetic_returns)
    assert result.weights["A_low_vol"] > result.weights["B_high_vol"]
    assert result.weights["A_low_vol"] > result.weights["C_correlated_with_B"]


def test_min_variance_beats_equal_weight_on_volatility(synthetic_returns):
    mv = min_variance_portfolio(synthetic_returns)
    equal_weights = pd.Series(1 / 3, index=synthetic_returns.columns)
    cov_annual = synthetic_returns.cov() * 252
    equal_vol = float(np.sqrt(equal_weights @ cov_annual @ equal_weights))
    assert mv.volatility <= equal_vol


def test_efficient_frontier_spans_target_returns(synthetic_returns):
    frontier = efficient_frontier(synthetic_returns, n_points=10)
    assert len(frontier) >= 8  # allow a couple of solver failures at the extremes
    assert frontier["volatility"].min() > 0
    assert frontier["return"].is_monotonic_increasing
