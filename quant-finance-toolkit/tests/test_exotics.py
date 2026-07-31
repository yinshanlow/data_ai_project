import pytest

from pricing.black_scholes import bs_price
from pricing.exotics import asian_arithmetic_call, barrier_down_and_out_call


def test_down_and_out_call_cheaper_than_vanilla():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    vanilla = bs_price(S, K, T, r, sigma, "call")
    barrier = barrier_down_and_out_call(S, K, T, r, sigma, barrier=80, n_paths=200_000, n_steps=252, seed=1)
    assert barrier.price < vanilla


def test_lower_barrier_means_less_knockout_risk_so_higher_price():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    near_barrier = barrier_down_and_out_call(S, K, T, r, sigma, barrier=95, n_paths=200_000, n_steps=252, seed=1)
    far_barrier = barrier_down_and_out_call(S, K, T, r, sigma, barrier=60, n_paths=200_000, n_steps=252, seed=1)
    assert far_barrier.price > near_barrier.price


def test_barrier_requires_barrier_below_spot():
    with pytest.raises(ValueError):
        barrier_down_and_out_call(100, 100, 1.0, 0.05, 0.2, barrier=110)


def test_asian_call_cheaper_than_vanilla():
    """Averaging dampens volatility, so an arithmetic Asian call is always worth less than
    the vanilla call on the same terminal-only payoff."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    vanilla = bs_price(S, K, T, r, sigma, "call")
    asian = asian_arithmetic_call(S, K, T, r, sigma, n_paths=200_000, n_steps=100, seed=1)
    assert asian.price < vanilla


def test_asian_call_payoff_nonnegative():
    result = asian_arithmetic_call(100, 100, 1.0, 0.05, 0.2, n_paths=10_000, seed=2)
    assert result.price >= 0
