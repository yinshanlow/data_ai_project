"""Merton (1976) jump-diffusion model — closed form as a Poisson-weighted sum
of Black-Scholes prices.

Where Heston lets *volatility* be random, Merton keeps volatility constant but
lets the asset *jump* — a different, complementary departure from Black-
Scholes' pure diffusion. Constant-volatility diffusion (Part A) cannot
produce the fat tails and the smile lift jumps cause, especially for
short-dated options, where the market weights the small probability of a
large sudden move much more heavily than a pure diffusion process would.
"""
import math

import numpy as np
from scipy.optimize import brentq

from pricing.black_scholes import bs_price


def merton_price(
    S: float, K: float, T: float, r: float, sigma: float,
    jump_intensity: float, jump_mean: float, jump_vol: float,
    option_type: str = "call", n_terms: int = 40,
) -> float:
    """jump_intensity (lambda): expected number of jumps per year.
    jump_mean, jump_vol: mean and std dev of the log jump size, log(Y) ~ N(jump_mean, jump_vol^2).
    n_terms: number of Poisson terms to sum — 40 is overkill for realistic lambda*T (Poisson
    weights decay factorially), included for headroom rather than because it's needed.
    """
    k = np.exp(jump_mean + 0.5 * jump_vol**2) - 1  # E[Y - 1]
    lam_prime = jump_intensity * (1 + k)

    total = 0.0
    for n in range(n_terms):
        poisson_weight = np.exp(-lam_prime * T) * (lam_prime * T) ** n / math.factorial(n)
        sigma_n = np.sqrt(sigma**2 + n * jump_vol**2 / T)
        r_n = r - jump_intensity * k + n * (jump_mean + 0.5 * jump_vol**2) / T
        total += poisson_weight * bs_price(S, K, T, r_n, sigma_n, option_type)

    return float(total)


def merton_implied_vol(
    S: float, K: float, T: float, r: float, sigma: float,
    jump_intensity: float, jump_mean: float, jump_vol: float, option_type: str = "call",
) -> float | None:
    price = merton_price(S, K, T, r, sigma, jump_intensity, jump_mean, jump_vol, option_type)
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if price <= intrinsic * np.exp(-r * T) + 1e-10:
        return None

    def objective(iv):
        return bs_price(S, K, T, r, iv, option_type) - price

    try:
        return brentq(objective, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        return None


def merton_smile(
    S: float, strikes: list[float], T: float, r: float, sigma: float,
    jump_intensity: float, jump_mean: float, jump_vol: float, option_type: str = "call",
) -> list[tuple[float, float, float]]:
    results = []
    for K in strikes:
        price = merton_price(S, K, T, r, sigma, jump_intensity, jump_mean, jump_vol, option_type)
        iv = merton_implied_vol(S, K, T, r, sigma, jump_intensity, jump_mean, jump_vol, option_type)
        results.append((K, price, iv))
    return results
