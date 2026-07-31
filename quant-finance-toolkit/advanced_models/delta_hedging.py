"""Discrete delta-hedging simulation.

A classic "do you actually understand the mechanics, not just the formula"
interview question: Black-Scholes assumes continuous, costless rebalancing,
which perfectly replicates the option and eliminates the seller's risk. No
real hedger rebalances continuously. This simulates an option seller who
delta-hedges at realistic discrete intervals and measures the resulting P&L
variance — and shows it shrinking toward zero as rebalancing frequency
increases, which is the concrete, simulated version of "continuous hedging
eliminates risk," rather than a claim taken on faith.
"""
from dataclasses import dataclass

import numpy as np

from pricing.black_scholes import bs_greeks, bs_price


@dataclass
class HedgeSimResult:
    rebalances_per_year: int
    pnl_mean: float
    pnl_std: float
    pnl: np.ndarray


def simulate_discrete_hedge(
    S0: float, K: float, T: float, r: float, sigma: float,
    rebalances_per_year: int, n_paths: int = 20_000, option_type: str = "call", seed: int | None = None,
) -> HedgeSimResult:
    """Seller sells one option at the Black-Scholes price and delta-hedges it at
    `rebalances_per_year` evenly spaced points until expiry. Returns the seller's
    hedging P&L (positive = seller ends up ahead) across n_paths simulated GBM paths.
    """
    n_steps = max(int(round(T * rebalances_per_year)), 1)
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    premium = bs_price(S0, K, T, r, sigma, option_type)
    S = np.full(n_paths, S0, dtype=float)

    delta0 = bs_greeks(S0, K, T, r, sigma, option_type).delta
    shares_held = np.full(n_paths, delta0)
    cash = premium - shares_held * S0  # sold the option, bought delta0 shares, rest in cash (can be negative = borrowing)

    for step in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        cash = cash * np.exp(r * dt)  # cash/borrowing accrues at the risk-free rate

        time_remaining = T - step * dt
        if time_remaining > 1e-8:
            greeks = bs_greeks(S, K, time_remaining, r, sigma, option_type)
            new_shares = greeks.delta
        else:
            new_shares = np.zeros(n_paths)  # at expiry, unwind the stock position entirely

        cash = cash - (new_shares - shares_held) * S  # buy/sell the stock needed to rebalance
        shares_held = new_shares

    portfolio_value = shares_held * S + cash
    payoff = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
    pnl = portfolio_value - payoff  # seller's final P&L: what they have, minus what they owe

    return HedgeSimResult(
        rebalances_per_year=rebalances_per_year, pnl_mean=float(pnl.mean()), pnl_std=float(pnl.std(ddof=1)), pnl=pnl,
    )


def hedging_frequency_study(
    S0: float, K: float, T: float, r: float, sigma: float,
    frequencies: tuple[int, ...] = (12, 52, 252, 1000), n_paths: int = 20_000, option_type: str = "call", seed: int | None = 0,
) -> list[HedgeSimResult]:
    return [
        simulate_discrete_hedge(S0, K, T, r, sigma, freq, n_paths, option_type, seed=None if seed is None else seed + freq)
        for freq in frequencies
    ]
