import pytest

from fixed_income.short_rate import VasicekParams, monte_carlo_bond_price, vasicek_bond_price, vasicek_yield


@pytest.fixture
def params():
    return VasicekParams(kappa=0.5, theta=0.04, sigma=0.02)


def test_closed_form_matches_monte_carlo(params):
    """Two independent ways of computing the same zero-coupon bond price — the closed
    form and a full Monte Carlo simulation of the short rate — should agree closely."""
    r0, T = 0.03, 5.0
    closed_form = vasicek_bond_price(r0, T, params)
    mc_price, mc_se = monte_carlo_bond_price(r0, T, params, n_paths=100_000, n_steps=200, seed=1)
    assert mc_price == pytest.approx(closed_form, abs=6 * mc_se)


def test_bond_price_decreases_with_maturity(params):
    r0 = 0.03
    p1 = vasicek_bond_price(r0, 1, params)
    p5 = vasicek_bond_price(r0, 5, params)
    p10 = vasicek_bond_price(r0, 10, params)
    assert p1 > p5 > p10


def test_bond_price_between_zero_and_one(params):
    for T in [0.5, 1, 5, 10, 30]:
        price = vasicek_bond_price(0.03, T, params)
        assert 0 < price < 1


def test_yield_curve_upward_sloping_when_rate_below_long_run_mean(params):
    """r0=3% < theta=4% (long-run mean) — mean reversion should pull future rates up,
    producing an upward-sloping yield curve."""
    r0 = 0.03
    y_short = vasicek_yield(r0, 1, params)
    y_long = vasicek_yield(r0, 10, params)
    assert y_long > y_short


def test_yield_curve_downward_sloping_when_rate_above_long_run_mean(params):
    r0 = 0.06  # above theta=0.04
    y_short = vasicek_yield(r0, 1, params)
    y_long = vasicek_yield(r0, 10, params)
    assert y_long < y_short
