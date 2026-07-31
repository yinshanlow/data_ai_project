import numpy as np
import pytest

from advanced_models.delta_hedging import hedging_frequency_study, simulate_discrete_hedge


def test_pnl_mean_close_to_zero():
    """With no vol mismatch between simulation and hedging, the hedge should be
    unbiased — mean P&L near zero, regardless of rebalancing frequency."""
    result = simulate_discrete_hedge(100, 100, 1.0, 0.05, 0.2, rebalances_per_year=52, n_paths=30_000, seed=1)
    assert abs(result.pnl_mean) < 0.15


def test_pnl_variance_shrinks_as_rebalancing_frequency_increases():
    """The whole point of this module: more frequent rebalancing must reduce hedging
    P&L variance, monotonically, toward the continuous-hedging (zero-risk) limit."""
    results = hedging_frequency_study(100, 100, 1.0, 0.05, 0.2, frequencies=(12, 52, 252, 1000), n_paths=20_000, seed=0)
    stds = [r.pnl_std for r in results]
    assert stds == sorted(stds, reverse=True), f"expected monotonically decreasing P&L std, got {stds}"


def test_pnl_std_scales_roughly_as_inverse_sqrt_n():
    """Theoretical result: discrete-hedging P&L variance is O(dt), so std should scale
    roughly as 1/sqrt(n_steps). Quadrupling the rebalancing frequency should roughly
    halve the std — checked loosely, since this is a Monte Carlo estimate, not exact."""
    low_freq = simulate_discrete_hedge(100, 100, 1.0, 0.05, 0.2, rebalances_per_year=63, n_paths=30_000, seed=2)
    high_freq = simulate_discrete_hedge(100, 100, 1.0, 0.05, 0.2, rebalances_per_year=252, n_paths=30_000, seed=2)
    ratio = low_freq.pnl_std / high_freq.pnl_std
    assert 1.5 < ratio < 2.7, f"expected ~2x std reduction for 4x rebalancing frequency, got {ratio:.2f}x"


def test_high_frequency_hedge_has_small_variance_relative_to_premium():
    from pricing.black_scholes import bs_price

    premium = bs_price(100, 100, 1.0, 0.05, 0.2, "call")
    result = simulate_discrete_hedge(100, 100, 1.0, 0.05, 0.2, rebalances_per_year=1000, n_paths=20_000, seed=3)
    assert result.pnl_std < 0.5 * premium
