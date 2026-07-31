"""Vasicek (1977) short-rate model.

dr_t = kappa*(theta - r_t) dt + sigma dW_t — mean-reverting, Gaussian short
rate, with a genuine closed-form zero-coupon bond price. The closed-form
price and a Monte Carlo simulation of the same model are compared directly
in tests/test_short_rate.py, the same cross-validation pattern used
everywhere else in this repo (Part A's PDE vs. tree, Part B's zero-jump
Merton vs. Black-Scholes): two independent ways of computing the same
quantity should agree, and that agreement is itself the test.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class VasicekParams:
    kappa: float  # mean-reversion speed
    theta: float  # long-run mean short rate
    sigma: float  # short-rate volatility


def vasicek_bond_price(r0: float, T: float, params: VasicekParams) -> float:
    """Closed-form zero-coupon bond price P(0,T) = A(T) * exp(-B(T) * r0)."""
    kappa, theta, sigma = params.kappa, params.theta, params.sigma

    if kappa == 0:
        B = T
    else:
        B = (1 - np.exp(-kappa * T)) / kappa

    if kappa == 0:
        A_log = -theta * T**2 / 2 + sigma**2 * T**3 / 6
    else:
        A_log = (theta - sigma**2 / (2 * kappa**2)) * (B - T) - (sigma**2 / (4 * kappa)) * B**2

    return float(np.exp(A_log - B * r0))


def vasicek_yield(r0: float, T: float, params: VasicekParams) -> float:
    """Continuously-compounded yield implied by the closed-form bond price."""
    price = vasicek_bond_price(r0, T, params)
    return float(-np.log(price) / T)


def simulate_vasicek_paths(r0: float, T: float, params: VasicekParams, n_paths: int, n_steps: int, seed: int | None = None) -> np.ndarray:
    """Euler discretization. Returns the full short-rate path, shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    r = np.full((n_paths, n_steps + 1), r0, dtype=float)

    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        r[:, step + 1] = r[:, step] + params.kappa * (params.theta - r[:, step]) * dt + params.sigma * np.sqrt(dt) * z

    return r


def monte_carlo_bond_price(r0: float, T: float, params: VasicekParams, n_paths: int = 100_000, n_steps: int = 200, seed: int | None = None) -> tuple[float, float]:
    """Prices the same zero-coupon bond by simulating the short rate and discounting
    along each path: P(0,T) = E[exp(-integral of r_t dt)]. Returns (price, std_error)."""
    paths = simulate_vasicek_paths(r0, T, params, n_paths, n_steps, seed)
    dt = T / n_steps
    integrated_r = np.trapezoid(paths, dx=dt, axis=1)  # trapezoidal integral of r_t over [0, T] per path
    discounted = np.exp(-integrated_r)
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))
