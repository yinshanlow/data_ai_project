"""Markowitz mean-variance portfolio optimization and the efficient frontier.

Long-only, fully-invested portfolios (weights sum to 1, no shorting) — the
standard textbook constraint set. Uses scipy's SLSQP rather than a dedicated
QP library to keep the dependency footprint small; for the portfolio sizes
here (a handful of assets) it's exact enough and fast.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioResult:
    weights: pd.Series
    expected_return: float  # annualized
    volatility: float  # annualized
    sharpe_ratio: float


def _annualize(mean_daily: pd.Series, cov_daily: pd.DataFrame, periods: int = 252):
    return mean_daily * periods, cov_daily * periods


def _portfolio_stats(weights: np.ndarray, mean_annual: np.ndarray, cov_annual: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(weights @ mean_annual)
    vol = float(np.sqrt(weights @ cov_annual @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def min_variance_portfolio(returns: pd.DataFrame, rf: float = 0.0) -> PortfolioResult:
    mean_annual, cov_annual = _annualize(returns.mean(), returns.cov())
    n = len(returns.columns)
    mean_arr, cov_arr = mean_annual.values, cov_annual.values

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.full(n, 1 / n)

    result = minimize(lambda w: w @ cov_arr @ w, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    ret, vol, sharpe = _portfolio_stats(result.x, mean_arr, cov_arr, rf)
    return PortfolioResult(pd.Series(result.x, index=returns.columns), ret, vol, sharpe)


def max_sharpe_portfolio(returns: pd.DataFrame, rf: float = 0.0) -> PortfolioResult:
    mean_annual, cov_annual = _annualize(returns.mean(), returns.cov())
    n = len(returns.columns)
    mean_arr, cov_arr = mean_annual.values, cov_annual.values

    def neg_sharpe(w):
        ret, vol, sharpe = _portfolio_stats(w, mean_arr, cov_arr, rf)
        return -sharpe

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.full(n, 1 / n)

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    ret, vol, sharpe = _portfolio_stats(result.x, mean_arr, cov_arr, rf)
    return PortfolioResult(pd.Series(result.x, index=returns.columns), ret, vol, sharpe)


def efficient_frontier(returns: pd.DataFrame, n_points: int = 40, rf: float = 0.0) -> pd.DataFrame:
    """Minimum-variance portfolio at each of n_points target returns spanning the achievable range."""
    mean_annual, cov_annual = _annualize(returns.mean(), returns.cov())
    n = len(returns.columns)
    mean_arr, cov_arr = mean_annual.values, cov_annual.values

    target_returns = np.linspace(mean_arr.min(), mean_arr.max(), n_points)
    frontier_rows = []
    bounds = [(0.0, 1.0)] * n

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mean_arr - t},
        ]
        x0 = np.full(n, 1 / n)
        result = minimize(lambda w: w @ cov_arr @ w, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            ret, vol, sharpe = _portfolio_stats(result.x, mean_arr, cov_arr, rf)
            frontier_rows.append({"target_return": target, "return": ret, "volatility": vol, "sharpe": sharpe})

    return pd.DataFrame(frontier_rows)
