"""Cox-Ross-Rubinstein binomial tree pricer for European and American options.

The point of this module isn't just the tree itself — it's demonstrating that
it converges to the Black-Scholes closed-form price as the step count grows,
which is the standard way to sanity-check a tree implementation. See
convergence_to_bs() and tests/test_binomial_tree.py.
"""
import numpy as np


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_type: str = "call",
    exercise: str = "european",
    q: float = 0.0,
) -> float:
    """CRR binomial tree price.

    exercise: "european" or "american". American options are checked for
    early exercise at every node on the backward pass.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside (0,1) — n_steps too small or "
            "parameters imply arbitrage at this step size."
        )

    j = np.arange(n_steps + 1)
    terminal_S = S * (u ** (n_steps - j)) * (d**j)
    if option_type == "call":
        values = np.maximum(terminal_S - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - terminal_S, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if exercise == "american":
            j = np.arange(step + 1)
            S_at_step = S * (u ** (step - j)) * (d**j)
            intrinsic = np.maximum(S_at_step - K, 0.0) if option_type == "call" else np.maximum(K - S_at_step, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])


def convergence_to_bs(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0,
    step_counts: tuple[int, ...] = (5, 10, 25, 50, 100, 200, 400, 800),
) -> list[tuple[int, float]]:
    """Binomial (European) price at increasing step counts — for plotting convergence to BS."""
    return [
        (n, binomial_price(S, K, T, r, sigma, n, option_type, exercise="european", q=q))
        for n in step_counts
    ]
