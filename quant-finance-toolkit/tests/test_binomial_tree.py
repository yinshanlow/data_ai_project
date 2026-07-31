import pytest

from pricing.binomial_tree import binomial_price, convergence_to_bs
from pricing.black_scholes import bs_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_convergence_to_black_scholes(option_type):
    """The whole point of the tree: as n_steps grows, it should converge to the closed form."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    bs_ref = bs_price(S, K, T, r, sigma, option_type)
    results = convergence_to_bs(S, K, T, r, sigma, option_type, step_counts=(10, 50, 200, 800))
    errors = [abs(price - bs_ref) for _, price in results]

    assert errors[-1] < 0.01, f"800-step tree should be within 1 cent of BS, got error={errors[-1]}"
    assert errors == sorted(errors, reverse=True), "error should shrink monotonically as step count grows"


def test_american_put_worth_at_least_european_put():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    euro = binomial_price(S, K, T, r, sigma, 200, "put", "european")
    amer = binomial_price(S, K, T, r, sigma, 200, "put", "american")
    assert amer >= euro - 1e-9


def test_american_call_equals_european_call_with_no_dividends():
    """Textbook result: with q=0, early exercise of an American call is never optimal."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    euro = binomial_price(S, K, T, r, sigma, 200, "call", "european")
    amer = binomial_price(S, K, T, r, sigma, 200, "call", "american")
    assert amer == pytest.approx(euro, abs=1e-6)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        binomial_price(100, 100, 1.0, 0.05, 0.2, 100, "straddle")
