"""Value-at-Risk, computed three ways, plus a Kupiec proportion-of-failures backtest.

VaR is reported as a positive number: the loss (as a fraction of portfolio
value) that should not be exceeded with probability `confidence` over one day.
All three methods are then backtested the same way — a rolling out-of-sample
VaR estimate compared against the next day's *actual* return — because the
whole point of computing VaR three ways is finding out which one's breach
rate actually matches what it claims, not just producing three numbers.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return float(-np.percentile(returns, 100 * (1 - confidence)))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def monte_carlo_var(returns: pd.Series, confidence: float = 0.95, n_sims: int = 100_000, seed: int | None = None) -> float:
    """Fits a normal distribution to historical returns, then simulates — distinct from
    parametric VaR in that it goes through actual simulation rather than the closed-form
    normal quantile, which matters once this is extended to non-normal/simulated factors."""
    mu, sigma = returns.mean(), returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    simulated = rng.normal(mu, sigma, n_sims)
    return float(-np.percentile(simulated, 100 * (1 - confidence)))


@dataclass
class KupiecResult:
    n_obs: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    lr_statistic: float
    p_value: float
    reject_at_5pct: bool


def kupiec_test(breaches: pd.Series | np.ndarray, confidence: float = 0.95) -> KupiecResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    breaches: boolean series, True where the actual loss exceeded the VaR estimate for that day.
    H0: the true breach probability equals (1 - confidence). Rejecting H0 means the VaR
    model's claimed confidence level doesn't match its actual empirical performance.
    """
    breaches = np.asarray(breaches, dtype=bool)
    n = len(breaches)
    x = int(breaches.sum())
    p_expected = 1 - confidence
    p_observed = x / n if n > 0 else 0.0

    if x == 0:
        log_l0 = n * np.log(1 - p_expected)
        log_l1 = n * np.log(1 - p_observed) if p_observed < 1 else 0.0
    elif x == n:
        log_l0 = n * np.log(p_expected)
        log_l1 = n * np.log(p_observed)
    else:
        log_l0 = x * np.log(p_expected) + (n - x) * np.log(1 - p_expected)
        log_l1 = x * np.log(p_observed) + (n - x) * np.log(1 - p_observed)

    lr_stat = -2 * (log_l0 - log_l1)
    p_value = float(1 - stats.chi2.cdf(lr_stat, df=1))

    return KupiecResult(
        n_obs=n, n_breaches=x, breach_rate=p_observed, expected_rate=p_expected,
        lr_statistic=float(lr_stat), p_value=p_value, reject_at_5pct=p_value < 0.05,
    )


def rolling_var_backtest(
    returns: pd.Series, method: str, window: int = 250, confidence: float = 0.95, seed: int | None = None,
) -> pd.DataFrame:
    """For each day after the initial window, compute VaR from the trailing `window` days
    and record whether the *next* day's actual loss breached it. Returns a DataFrame with
    one row per out-of-sample day: var_estimate, actual_return, breached.
    """
    estimator = {"historical": historical_var, "parametric": parametric_var}.get(method)
    rows = []
    for i in range(window, len(returns)):
        train = returns.iloc[i - window:i]
        actual = returns.iloc[i]
        if method == "monte_carlo":
            var_est = monte_carlo_var(train, confidence, n_sims=20_000, seed=seed)
        else:
            var_est = estimator(train, confidence)
        breached = bool(-actual > var_est)
        rows.append({"date": returns.index[i], "var_estimate": var_est, "actual_return": actual, "breached": breached})
    return pd.DataFrame(rows).set_index("date")
