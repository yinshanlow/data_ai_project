"""Monte Carlo pricer for European options, with variance reduction.

Three estimators are implemented on purpose, not just the "best" one:
naive MC, antithetic variates, and a control variate (using the terminal
asset price, whose risk-neutral expectation is known in closed form). The
point of this module is comparing their variance for a fixed random-number
budget — see tests/test_monte_carlo.py, which asserts the reduction is real
rather than just implementing the techniques and asserting nothing.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int


def _terminal_prices(S: float, T: float, r: float, sigma: float, q: float, z: np.ndarray) -> np.ndarray:
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def _payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def naive_mc_price(
    S: float, K: float, T: float, r: float, sigma: float, n_paths: int,
    option_type: str = "call", q: float = 0.0, seed: int | None = None,
) -> MCResult:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = _terminal_prices(S, T, r, sigma, q, z)
    discounted = np.exp(-r * T) * _payoff(S_T, K, option_type)
    return MCResult(price=float(discounted.mean()), std_error=float(discounted.std(ddof=1) / np.sqrt(n_paths)), n_paths=n_paths)


def antithetic_mc_price(
    S: float, K: float, T: float, r: float, sigma: float, n_paths: int,
    option_type: str = "call", q: float = 0.0, seed: int | None = None,
) -> MCResult:
    """n_paths total simulated paths (n_paths // 2 antithetic pairs) — same random-draw budget as naive_mc_price."""
    rng = np.random.default_rng(seed)
    n_pairs = n_paths // 2
    z = rng.standard_normal(n_pairs)

    S_T_pos = _terminal_prices(S, T, r, sigma, q, z)
    S_T_neg = _terminal_prices(S, T, r, sigma, q, -z)
    payoff_pos = _payoff(S_T_pos, K, option_type)
    payoff_neg = _payoff(S_T_neg, K, option_type)

    pair_avg = np.exp(-r * T) * 0.5 * (payoff_pos + payoff_neg)
    return MCResult(price=float(pair_avg.mean()), std_error=float(pair_avg.std(ddof=1) / np.sqrt(n_pairs)), n_paths=n_paths)


def control_variate_mc_price(
    S: float, K: float, T: float, r: float, sigma: float, n_paths: int,
    option_type: str = "call", q: float = 0.0, seed: int | None = None,
) -> MCResult:
    """Control variate = terminal asset price S_T, whose risk-neutral mean E[S_T] = S*exp((r-q)T) is known exactly."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = _terminal_prices(S, T, r, sigma, q, z)
    payoff = _payoff(S_T, K, option_type)

    control_mean = S * np.exp((r - q) * T)
    cov = np.cov(payoff, S_T, ddof=1)
    b_star = cov[0, 1] / cov[1, 1]

    adjusted = payoff - b_star * (S_T - control_mean)
    discounted = np.exp(-r * T) * adjusted
    return MCResult(price=float(discounted.mean()), std_error=float(discounted.std(ddof=1) / np.sqrt(n_paths)), n_paths=n_paths)
