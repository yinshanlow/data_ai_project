import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk.var import historical_var, kupiec_test, monte_carlo_var, parametric_var


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.02, 20_000))


def test_var_methods_agree_on_normal_data(normal_returns):
    """On genuinely normal data, all three methods should agree closely — the parametric
    method's normality assumption is exactly correct here."""
    analytic = -(normal_returns.mean() + stats.norm.ppf(0.05) * normal_returns.std(ddof=1))
    h = historical_var(normal_returns, 0.95)
    p = parametric_var(normal_returns, 0.95)
    m = monte_carlo_var(normal_returns, 0.95, seed=1)

    assert h == pytest.approx(analytic, abs=0.002)
    assert p == pytest.approx(analytic, abs=1e-6)
    assert m == pytest.approx(analytic, abs=0.002)


def test_var_increases_with_confidence(normal_returns):
    for fn in (historical_var, parametric_var):
        var_95 = fn(normal_returns, 0.95)
        var_99 = fn(normal_returns, 0.99)
        assert var_99 > var_95


def test_kupiec_accepts_well_calibrated_model():
    """A model whose breach rate exactly matches its stated confidence level should not be rejected."""
    rng = np.random.default_rng(1)
    n = 2000
    breaches = rng.random(n) < 0.05  # exactly the expected 5% breach rate, by construction
    result = kupiec_test(breaches, confidence=0.95)
    assert not result.reject_at_5pct
    assert result.p_value > 0.05


def test_kupiec_rejects_badly_miscalibrated_model():
    """A model that breaches 20% of the time when it claims 95% confidence (5% expected)
    should be flagged as miscalibrated."""
    rng = np.random.default_rng(2)
    n = 500
    breaches = rng.random(n) < 0.20
    result = kupiec_test(breaches, confidence=0.95)
    assert result.reject_at_5pct
    assert result.p_value < 0.05


def test_kupiec_handles_zero_breaches():
    breaches = np.zeros(500, dtype=bool)
    result = kupiec_test(breaches, confidence=0.95)
    assert result.n_breaches == 0
    assert np.isfinite(result.lr_statistic)
