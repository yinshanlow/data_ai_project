import pytest

from pricing.binomial_tree import binomial_price
from pricing.black_scholes import bs_price
from pricing.pde_solver import crank_nicolson_price


def test_european_pde_matches_black_scholes():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    bs_ref = bs_price(S, K, T, r, sigma, "put")
    pde = crank_nicolson_price(S, K, T, r, sigma, "put", "european", M=300, N=300)
    assert pde.price == pytest.approx(bs_ref, abs=0.02)


def test_american_pde_matches_binomial_tree():
    """Two independent numerical methods for the same American option should agree closely."""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    tree_ref = binomial_price(S, K, T, r, sigma, 500, "put", "american")
    pde = crank_nicolson_price(S, K, T, r, sigma, "put", "american", M=300, N=300)
    assert pde.price == pytest.approx(tree_ref, abs=0.05)


def test_american_put_worth_at_least_european_put():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    euro = crank_nicolson_price(S, K, T, r, sigma, "put", "european", M=300, N=300)
    amer = crank_nicolson_price(S, K, T, r, sigma, "put", "american", M=300, N=300)
    assert amer.price >= euro.price - 1e-6


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        crank_nicolson_price(100, 100, 1.0, 0.05, 0.2, "straddle", "american", M=50, N=50)
