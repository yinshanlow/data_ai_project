"""Generates the proof plots referenced in pricing/README.md.

Run with: python -m pricing.generate_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricing.binomial_tree import convergence_to_bs
from pricing.black_scholes import bs_price
from pricing.monte_carlo import antithetic_mc_price, control_variate_mc_price, naive_mc_price

FIG_DIR = Path(__file__).parent / "figures"


def plot_binomial_convergence():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    bs_ref = bs_price(S, K, T, r, sigma, "call")
    results = convergence_to_bs(S, K, T, r, sigma, "call", step_counts=tuple(2**k for k in range(2, 11)))
    steps, prices = zip(*results)
    errors = [abs(p - bs_ref) for p in prices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(steps, prices, marker="o", label="Binomial tree price")
    ax1.axhline(bs_ref, color="crimson", linestyle="--", label=f"Black-Scholes = {bs_ref:.4f}")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Number of tree steps (n)")
    ax1.set_ylabel("Option price")
    ax1.set_title("Binomial tree price vs. step count")
    ax1.legend()

    ax2.plot(steps, errors, marker="o", color="darkorange")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of tree steps (n)")
    ax2.set_ylabel("|Tree price - BS price|")
    ax2.set_title("Convergence error (log scale)")

    fig.suptitle("Binomial tree convergence to Black-Scholes (ATM call, S=K=100, T=1y)")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "binomial_convergence.png", dpi=150)
    plt.close(fig)


def plot_mc_variance_reduction():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_paths = 100_000
    estimators = {
        "Naive MC": naive_mc_price,
        "Antithetic variates": antithetic_mc_price,
        "Control variate": control_variate_mc_price,
    }

    seeds = range(30)
    prices_by_estimator = {name: [] for name in estimators}
    for seed in seeds:
        for name, fn in estimators.items():
            result = fn(S, K, T, r, sigma, n_paths, "call", seed=seed)
            prices_by_estimator[name].append(result.price)

    bs_ref = bs_price(S, K, T, r, sigma, "call")
    std_devs = {name: np.std(prices, ddof=1) for name, prices in prices_by_estimator.items()}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(std_devs.keys())
    values = [std_devs[n] for n in names]
    bars = ax.bar(names, values, color=["#888888", "#4C72B0", "#55A868"])
    ax.set_ylabel("Std dev of price estimate across 30 independent runs")
    ax.set_title(f"Monte Carlo variance reduction (n_paths={n_paths:,}, BS ref={bs_ref:.4f})")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "mc_variance_reduction.png", dpi=150)
    plt.close(fig)

    return std_devs


if __name__ == "__main__":
    plot_binomial_convergence()
    std_devs = plot_mc_variance_reduction()
    print("Saved figures to", FIG_DIR)
    print("MC estimator std devs across 30 runs:", {k: round(v, 5) for k, v in std_devs.items()})
