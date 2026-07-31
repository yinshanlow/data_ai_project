"""Closed-form Black-Scholes-Merton pricing and Greeks.

Assumes a single underlying following geometric Brownian motion with constant
volatility, constant risk-free rate, and a constant continuous dividend yield.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class BSResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European option.

    S: spot, K: strike, T: time to expiry in years, r: risk-free rate,
    sigma: annualized volatility, q: continuous dividend yield.
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> BSResult:
    """Analytical Greeks. Theta and Rho are per-year; divide Theta by 365 for a daily figure."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    price = bs_price(S, K, T, r, sigma, option_type, q)
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return BSResult(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
