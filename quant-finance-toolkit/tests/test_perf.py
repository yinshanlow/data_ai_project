import pytest

from pricing.black_scholes import bs_price

mc_pricer_cpp = pytest.importorskip(
    "perf.mc_pricer_cpp",
    reason="C++ extension not built — run: cd perf && python setup.py build_ext --inplace",
)


@pytest.fixture
def bs_ref():
    return bs_price(100, 100, 1.0, 0.05, 0.2, "call")


def test_single_threaded_price_close_to_black_scholes(bs_ref):
    result = mc_pricer_cpp.mc_price(100, 100, 1.0, 0.05, 0.2, 500_000, "call", 1)
    assert result.price == pytest.approx(bs_ref, abs=6 * result.std_error)


def test_parallel_price_close_to_black_scholes(bs_ref):
    result = mc_pricer_cpp.mc_price_parallel(100, 100, 1.0, 0.05, 0.2, 500_000, "call", 1, 0)
    assert result.price == pytest.approx(bs_ref, abs=6 * result.std_error)


def test_single_threaded_and_parallel_agree_statistically(bs_ref):
    """Different code paths (one loop vs. many threads combined) computing the same
    quantity should land in the same place, not just each be individually close to BS."""
    single = mc_pricer_cpp.mc_price(100, 100, 1.0, 0.05, 0.2, 1_000_000, "call", 1)
    parallel = mc_pricer_cpp.mc_price_parallel(100, 100, 1.0, 0.05, 0.2, 1_000_000, "call", 1, 0)
    combined_se = (single.std_error**2 + parallel.std_error**2) ** 0.5
    assert single.price == pytest.approx(parallel.price, abs=6 * combined_se)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_pricer_cpp.mc_price(100, 100, 1.0, 0.05, 0.2, 1000, "straddle", 1)


def test_parallel_matches_single_thread_request():
    """n_threads=1 should behave like the single-threaded version, statistically."""
    result = mc_pricer_cpp.mc_price_parallel(100, 100, 1.0, 0.05, 0.2, 200_000, "call", 1, 1)
    bs = bs_price(100, 100, 1.0, 0.05, 0.2, "call")
    assert result.price == pytest.approx(bs, abs=6 * result.std_error)
