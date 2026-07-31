import pytest

from advanced_models.merton_jump_diffusion import merton_price, merton_smile
from pricing.black_scholes import bs_price


def test_zero_jump_intensity_reduces_exactly_to_black_scholes():
    """The defining sanity check for this module: turn off jumps and it must be
    algebraically identical to Black-Scholes, not just numerically close."""
    S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.2
    merton = merton_price(S, K, T, r, sigma, jump_intensity=0.0, jump_mean=-0.1, jump_vol=0.3, option_type="call")
    bs = bs_price(S, K, T, r, sigma, "call")
    assert merton == pytest.approx(bs, abs=1e-9)


def test_jumps_increase_price_relative_to_pure_diffusion():
    """Adding jump risk (with nonzero jump volatility) should increase the option's fair
    value relative to pure diffusion at the same base sigma — extra uncertainty, extra value."""
    S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.15
    bs = bs_price(S, K, T, r, sigma, "call")
    merton = merton_price(S, K, T, r, sigma, jump_intensity=1.0, jump_mean=-0.05, jump_vol=0.25, option_type="call")
    assert merton > bs


def test_merton_smile_is_not_flat():
    S, T, r, sigma = 100, 0.25, 0.05, 0.15
    smile = merton_smile(S, [80, 90, 100, 110, 120], T, r, sigma, jump_intensity=1.0, jump_mean=-0.1, jump_vol=0.25)
    ivs = [iv for _, _, iv in smile]
    assert all(iv is not None for iv in ivs)
    assert max(ivs) - min(ivs) > 0.03


def test_merton_smile_is_u_shaped_for_symmetric_jumps():
    """With a symmetric (zero-mean) jump distribution, the smile should be roughly
    U-shaped/symmetric: both wings priced above the ATM implied vol."""
    S, T, r, sigma = 100, 0.25, 0.05, 0.15
    smile = merton_smile(S, [85, 100, 115], T, r, sigma, jump_intensity=1.5, jump_mean=0.0, jump_vol=0.2)
    iv_low, iv_atm, iv_high = [iv for _, _, iv in smile]
    assert iv_low > iv_atm
    assert iv_high > iv_atm
