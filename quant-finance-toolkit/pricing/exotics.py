"""Path-dependent exotic options, priced by Monte Carlo simulation.

Both payoffs depend on the whole simulated path, not just S_T, which is
exactly why they need Monte Carlo (or a PDE in more dimensions) rather than
a closed-form formula — the natural next step up in complexity from Part A's
vanilla pricers.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ExoticResult:
    price: float
    std_error: float
    n_paths: int


def _simulate_paths(S: float, T: float, r: float, sigma: float, q: float, n_paths: int, n_steps: int, seed: int | None) -> np.ndarray:
    """Returns an (n_paths, n_steps + 1) array of simulated GBM paths, including S at t=0."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.log(S) + np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = np.exp(log_paths)
    return paths


def barrier_down_and_out_call(
    S: float, K: float, T: float, r: float, sigma: float, barrier: float,
    n_paths: int = 100_000, n_steps: int = 100, q: float = 0.0, seed: int | None = None,
) -> ExoticResult:
    """Down-and-out call: knocked out (worthless) if the path ever touches or falls below `barrier`."""
    if barrier >= S:
        raise ValueError("barrier must be below the current spot for a down-and-out option")

    paths = _simulate_paths(S, T, r, sigma, q, n_paths, n_steps, seed)
    knocked_out = (paths <= barrier).any(axis=1)
    S_T = paths[:, -1]
    payoff = np.where(knocked_out, 0.0, np.maximum(S_T - K, 0.0))
    discounted = np.exp(-r * T) * payoff
    return ExoticResult(price=float(discounted.mean()), std_error=float(discounted.std(ddof=1) / np.sqrt(n_paths)), n_paths=n_paths)


def asian_arithmetic_call(
    S: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 100_000, n_steps: int = 100, q: float = 0.0, seed: int | None = None,
) -> ExoticResult:
    """Fixed-strike arithmetic-average Asian call: payoff = max(mean(path) - K, 0)."""
    paths = _simulate_paths(S, T, r, sigma, q, n_paths, n_steps, seed)
    average_price = paths[:, 1:].mean(axis=1)  # average of the monitored (post-t0) prices
    payoff = np.maximum(average_price - K, 0.0)
    discounted = np.exp(-r * T) * payoff
    return ExoticResult(price=float(discounted.mean()), std_error=float(discounted.std(ddof=1) / np.sqrt(n_paths)), n_paths=n_paths)
