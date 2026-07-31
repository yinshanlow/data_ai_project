import numpy as np
import pytest

from pricing.black_scholes import bs_greeks, bs_price


def test_known_textbook_value():
    # Hull, "Options, Futures, and Other Derivatives" — S=42, K=40, r=10%, sigma=20%, T=0.5y
    price = bs_price(42, 40, 0.5, 0.10, 0.20, "call")
    assert price == pytest.approx(4.76, abs=0.01)


def test_put_call_parity():
    S, K, T, r, sigma, q = 100, 95, 0.75, 0.03, 0.25, 0.01
    call = bs_price(S, K, T, r, sigma, "call", q)
    put = bs_price(S, K, T, r, sigma, "put", q)
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_deep_itm_call_converges_to_intrinsic_forward():
    price = bs_price(1000, 100, 1.0, 0.05, 0.2, "call")
    forward_intrinsic = 1000 - 100 * np.exp(-0.05)
    assert price == pytest.approx(forward_intrinsic, rel=1e-3)


def test_deep_otm_put_near_zero():
    price = bs_price(1000, 10, 1.0, 0.05, 0.2, "put")
    assert price < 1e-6


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_finite_difference(option_type):
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    analytical = bs_greeks(S, K, T, r, sigma, option_type)

    h = 1e-4
    delta_fd = (bs_price(S + h, K, T, r, sigma, option_type) - bs_price(S - h, K, T, r, sigma, option_type)) / (2 * h)
    gamma_fd = (
        bs_price(S + h, K, T, r, sigma, option_type) - 2 * bs_price(S, K, T, r, sigma, option_type)
        + bs_price(S - h, K, T, r, sigma, option_type)
    ) / h**2
    vega_fd = (bs_price(S, K, T, r, sigma + h, option_type) - bs_price(S, K, T, r, sigma - h, option_type)) / (2 * h)

    assert analytical.delta == pytest.approx(delta_fd, abs=1e-3)
    assert analytical.gamma == pytest.approx(gamma_fd, abs=1e-2)
    assert analytical.vega == pytest.approx(vega_fd, abs=1e-2)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1.0, 0.05, 0.2, "straddle")
