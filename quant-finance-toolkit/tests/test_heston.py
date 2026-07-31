import pytest

from advanced_models.heston import HestonParams, heston_price_mc, heston_smile
from pricing.black_scholes import bs_price


@pytest.fixture
def params():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.7)


def test_heston_atm_price_reasonably_close_to_bs_at_long_run_vol(params):
    """Not an exact match (Heston has vol-of-vol and correlation effects even ATM), but
    should be in the same neighborhood as BS priced at sigma = sqrt(long-run variance)."""
    S0, K, T, r = 100, 100, 1.0, 0.05
    price, se = heston_price_mc(S0, K, T, r, params, n_paths=100_000, seed=1)
    bs_ref = bs_price(S0, K, T, r, 0.2, "call")
    assert price == pytest.approx(bs_ref, abs=0.5)


def test_heston_smile_is_downward_sloping_with_negative_correlation():
    """rho < 0 (the standard equity 'leverage effect' assumption) should produce a
    downward-sloping skew: implied vol falls as strike rises."""
    S0, T, r = 100, 1.0, 0.05
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.7)
    strikes = [80, 90, 100, 110, 120]
    smile = heston_smile(S0, strikes, T, r, params, n_paths=100_000, seed=5)
    ivs = [iv for _, _, iv in smile]

    assert all(iv is not None for iv in ivs)
    assert ivs == sorted(ivs, reverse=True), f"expected monotonically decreasing implied vol, got {ivs}"


def test_heston_smile_is_not_flat_unlike_black_scholes():
    """The whole point of this module: Heston's implied vols must actually vary across
    strikes — if they came out flat, this wouldn't be demonstrating anything Black-Scholes
    doesn't already do."""
    S0, T, r = 100, 1.0, 0.05
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    strikes = [70, 100, 130]
    smile = heston_smile(S0, strikes, T, r, params, n_paths=100_000, seed=7)
    ivs = [iv for _, _, iv in smile]
    assert max(ivs) - min(ivs) > 0.03, "expected a visible smile/skew spread of at least 3 vol points"
