"""Conditional VaR (Expected Shortfall) and historical-scenario stress testing.

CVaR answers the question VaR doesn't: given that the loss exceeds VaR, how
bad is it on average? Reported as a positive loss fraction, same convention
as var.py.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from risk.var import historical_var


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    var = historical_var(returns, confidence)
    tail_losses = returns[returns <= -var]
    if len(tail_losses) == 0:
        return var
    return float(-tail_losses.mean())


def parametric_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(confidence)
    mu_loss = -mu
    return float(mu_loss + sigma * stats.norm.pdf(z) / (1 - confidence))


@dataclass
class StressTestResult:
    scenario_name: str
    portfolio_shock: float  # fractional portfolio loss under the scenario
    asset_shocks: pd.Series


def apply_historical_scenario(
    weights: pd.Series, scenario_returns: pd.Series,
) -> StressTestResult:
    """Apply a historical crisis period's per-asset returns to the current portfolio weights.

    scenario_returns: total (not annualized) return per asset over the crisis window,
    indexed the same way as `weights`.
    """
    common = weights.index.intersection(scenario_returns.index)
    if len(common) == 0:
        raise ValueError("weights and scenario_returns share no assets")
    w = weights.loc[common]
    shocks = scenario_returns.loc[common]
    portfolio_shock = float((w * shocks).sum())
    return StressTestResult(scenario_name="custom", portfolio_shock=portfolio_shock, asset_shocks=shocks)


def crisis_period_total_return(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Peak-to-trough (or start-to-end) total return per asset over a historical window
    already present in `prices` — e.g. a COVID-crash or rate-shock window."""
    window = prices.loc[start:end]
    if len(window) < 2:
        raise ValueError(f"Not enough data in [{start}, {end}] — check the price history covers this window")
    return window.iloc[-1] / window.iloc[0] - 1
