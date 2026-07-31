import pytest

from pricing.black_scholes import bs_price
from pricing.monte_carlo import antithetic_mc_price, control_variate_mc_price, naive_mc_price


@pytest.fixture
def params():
    return dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")


def test_naive_mc_close_to_black_scholes(params):
    bs_ref = bs_price(params["S"], params["K"], params["T"], params["r"], params["sigma"], params["option_type"])
    result = naive_mc_price(**params, n_paths=200_000, seed=42)
    assert result.price == pytest.approx(bs_ref, abs=4 * result.std_error)


def test_variance_reduction_actually_reduces_variance(params):
    """The point of this module: antithetic and control-variate estimators must have a
    smaller standard error than naive MC for the same random-draw budget — not just
    implement the techniques and assume it worked."""
    n_paths = 100_000
    naive = naive_mc_price(**params, n_paths=n_paths, seed=7)
    antithetic = antithetic_mc_price(**params, n_paths=n_paths, seed=7)
    control = control_variate_mc_price(**params, n_paths=n_paths, seed=7)

    assert antithetic.std_error < naive.std_error
    assert control.std_error < naive.std_error


def test_all_estimators_agree_with_black_scholes(params):
    bs_ref = bs_price(params["S"], params["K"], params["T"], params["r"], params["sigma"], params["option_type"])
    n_paths = 200_000
    for estimator in (naive_mc_price, antithetic_mc_price, control_variate_mc_price):
        result = estimator(**params, n_paths=n_paths, seed=123)
        assert result.price == pytest.approx(bs_ref, abs=5 * result.std_error), estimator.__name__


def test_put_call_via_control_variate(params):
    put_params = {**params, "option_type": "put"}
    bs_ref = bs_price(put_params["S"], put_params["K"], put_params["T"], put_params["r"], put_params["sigma"], "put")
    result = control_variate_mc_price(**put_params, n_paths=200_000, seed=99)
    assert result.price == pytest.approx(bs_ref, abs=5 * result.std_error)
