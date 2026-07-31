"""Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

Built specifically for American options with early exercise — the natural
place a PDE grid earns its keep over the closed-form Black-Scholes formula,
which only prices European options. The American price here is sanity-checked
against the binomial tree's American price in tests/test_pde_solver.py.

Simplification, stated honestly: early exercise is enforced by taking the
elementwise max of the Crank-Nicolson solution against the intrinsic payoff
after each time step, rather than solving the full linear complementarity
problem with projected SOR. This is a standard, widely-used approximation
that converges to the correct price on a reasonably fine grid, but a
production system would use PSOR (or an operator-splitting method) for a
numerically cleaner treatment of the free exercise boundary.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class PDEResult:
    price: float
    S_grid: np.ndarray
    V_grid: np.ndarray  # option value at t=0 across the full S grid


def crank_nicolson_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "put", exercise: str = "american", q: float = 0.0,
    S_max_mult: float = 3.0, M: int = 200, N: int = 200,
) -> PDEResult:
    """M: number of spatial (S) steps. N: number of time steps."""
    S_max = S_max_mult * K
    dS = S_max / M
    dt = T / N
    S_grid = np.linspace(0, S_max, M + 1)
    i = np.arange(1, M)  # interior node indices

    if option_type == "call":
        payoff = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    V = payoff.copy()  # terminal condition V(S, T)

    a = 0.25 * dt * (sigma**2 * i**2 - (r - q) * i)
    b = -0.5 * dt * (sigma**2 * i**2 + r)
    c = 0.25 * dt * (sigma**2 * i**2 + (r - q) * i)

    M_left = np.zeros((3, M - 1))  # banded matrix for the implicit (n+1) side
    M_left[0, 1:] = -c[:-1]
    M_left[1, :] = 1 - b
    M_left[2, :-1] = -a[1:]

    for step in range(N - 1, -1, -1):
        rhs = a * V[:-2] + (1 + b) * V[1:-1] + c * V[2:]

        tau = T - step * dt  # time-to-expiry at the boundary being applied
        if option_type == "call":
            V0, VM = 0.0, S_max - K * np.exp(-r * tau)
        else:
            V0, VM = K * np.exp(-r * tau), 0.0
        rhs[0] += a[0] * V0
        rhs[-1] += c[-1] * VM

        V_interior = solve_banded((1, 1), M_left, rhs)
        V = np.concatenate(([V0], V_interior, [VM]))

        if exercise == "american":
            V = np.maximum(V, payoff)

    price = float(np.interp(S, S_grid, V))
    return PDEResult(price=price, S_grid=S_grid, V_grid=V)
