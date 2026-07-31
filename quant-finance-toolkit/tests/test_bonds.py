import pytest

from fixed_income.bonds import bond_price, bond_risk_metrics, estimate_price_change


def test_par_bond_prices_at_face_value():
    price = bond_price(100, 0.05, 0.05, 10, freq=2)
    assert price == pytest.approx(100.0, abs=1e-6)


def test_premium_and_discount_bonds():
    premium = bond_price(100, 0.06, 0.05, 10, freq=2)  # coupon > yield
    discount = bond_price(100, 0.04, 0.05, 10, freq=2)  # coupon < yield
    assert premium > 100
    assert discount < 100


def test_price_falls_as_yield_rises():
    p_low_yield = bond_price(100, 0.05, 0.03, 10, freq=2)
    p_high_yield = bond_price(100, 0.05, 0.07, 10, freq=2)
    assert p_low_yield > p_high_yield


def test_duration_and_convexity_approximate_small_yield_moves_well():
    """The duration+convexity estimate should be very close to the exact repriced
    value for a small (25bp) yield shift."""
    metrics = bond_risk_metrics(100, 0.05, 0.05, 10, freq=2)
    dy = 0.0025
    estimated = estimate_price_change(metrics, dy)
    exact = bond_price(100, 0.05, 0.05 + dy, 10, freq=2) - metrics.price
    assert estimated == pytest.approx(exact, abs=0.01)


def test_approximation_error_grows_with_the_size_of_the_yield_move():
    """Duration+convexity is a local (Taylor) approximation — its error should grow
    as the yield move gets larger, which is the whole reason convexity matters."""
    metrics = bond_risk_metrics(100, 0.05, 0.05, 10, freq=2)

    small_dy, large_dy = 0.005, 0.03
    small_error = abs(estimate_price_change(metrics, small_dy) - (bond_price(100, 0.05, 0.05 + small_dy, 10, freq=2) - metrics.price))
    large_error = abs(estimate_price_change(metrics, large_dy) - (bond_price(100, 0.05, 0.05 + large_dy, 10, freq=2) - metrics.price))
    assert large_error > small_error


def test_longer_maturity_has_higher_duration():
    short = bond_risk_metrics(100, 0.05, 0.05, 2, freq=2)
    long = bond_risk_metrics(100, 0.05, 0.05, 20, freq=2)
    assert long.modified_duration > short.modified_duration
