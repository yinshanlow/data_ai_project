"""Heston (1993) stochastic volatility model, priced by Monte Carlo.

The point of this module: Black-Scholes (Part A) assumes constant volatility,
which forces a flat implied volatility curve across strikes. Real option
markets never look like that — they show a "smile" or "skew". Heston lets
volatility itself be a random, mean-reverting process, correlated with the
asset's own returns, which is enough to reproduce a realistic smile. See
heston_smile() and tests/test_heston.py, which check the smile is real
curvature, not implement-and-hope.

Simulation scheme, stated honestly: variance is simulated with the "full
truncation" Euler scheme (Lord, Koekkoek & van Dijk, 2010) — the negative
variance values that plain Euler discretization can produce are truncated to
zero before use, rather than a full Milstein or exact (Broadie-Kaya /
Andersen QE) scheme. Full truncation is simple, standard, and adequate for
demonstrating the smile; it introduces a small discretization bias relative
to the exact Heston characteristic-function price, which a production
pricer would use instead.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from pricing.black_scholes import bs_price


@dataclass
class HestonParams:
    v0: float      # initial variance
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol-of-vol
    rho: float     # correlation between asset and variance shocks


def simulate_heston_paths(
    S0: float, T: float, r: float, params: HestonParams, n_paths: int, n_steps: int, seed: int | None = None,
) -> np.ndarray:
    """Returns terminal asset prices S_T (shape: (n_paths,)) under the Heston model."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S = np.full(n_paths, S0, dtype=float)
    v = np.full(n_paths, params.v0, dtype=float)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_v = z1
        z_s = params.rho * z1 + np.sqrt(1 - params.rho**2) * z2

        v_pos = np.maximum(v, 0.0)  # full truncation: use max(v,0) wherever v appears
        S = S * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * np.sqrt(dt) * z_s)
        v = v + params.kappa * (params.theta - v_pos) * dt + params.xi * np.sqrt(v_pos) * np.sqrt(dt) * z_v

    return S


def heston_price_mc(
    S0: float, K: float, T: float, r: float, params: HestonParams,
    n_paths: int = 100_000, n_steps: int = 100, option_type: str = "call", seed: int | None = None,
) -> tuple[float, float]:
    """Returns (price, std_error)."""
    S_T = simulate_heston_paths(S0, T, r, params, n_paths, n_steps, seed)
    payoff = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
    discounted = np.exp(-r * T) * payoff
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def implied_volatility(price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float | None:
    """Inverts the Black-Scholes formula for sigma given a market (or model) price.
    Returns None if no solution exists in a reasonable range (e.g. the price is outside
    arbitrage bounds, which can happen for noisy Monte Carlo prices deep out-of-the-money)."""
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if price <= intrinsic * np.exp(-r * T) + 1e-10:
        return None

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        return None


def heston_smile(
    S0: float, strikes: list[float], T: float, r: float, params: HestonParams,
    n_paths: int = 200_000, n_steps: int = 100, option_type: str = "call", seed: int | None = 0,
) -> list[tuple[float, float, float]]:
    """Returns [(strike, heston_price, implied_vol), ...] — the implied vol column is what
    produces the smile plot when compared against Black-Scholes' flat assumed sigma."""
    results = []
    for i, K in enumerate(strikes):
        price, _ = heston_price_mc(S0, K, T, r, params, n_paths, n_steps, option_type, seed=None if seed is None else seed + i)
        iv = implied_volatility(price, S0, K, T, r, option_type)
        results.append((K, price, iv))
    return results
