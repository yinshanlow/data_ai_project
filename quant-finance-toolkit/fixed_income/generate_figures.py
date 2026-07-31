"""Generates the proof plots referenced in fixed_income/README.md.

Run with: python -m fixed_income.generate_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fixed_income.bonds import bond_price, bond_risk_metrics, estimate_price_change
from fixed_income.short_rate import VasicekParams, simulate_vasicek_paths, vasicek_yield
from fixed_income.yield_curve import ParInstrument, bootstrap_zero_curve

FIG_DIR = Path(__file__).parent / "figures"


def plot_bootstrapped_curve():
    instruments = [
        ParInstrument(maturity_years=0.5, par_yield=0.045, freq=1),
        ParInstrument(maturity_years=1.0, par_yield=0.047, freq=1),
        ParInstrument(maturity_years=2.0, par_yield=0.049, freq=2),
        ParInstrument(maturity_years=3.0, par_yield=0.050, freq=2),
        ParInstrument(maturity_years=5.0, par_yield=0.052, freq=2),
        ParInstrument(maturity_years=10.0, par_yield=0.053, freq=2),
    ]
    curve = bootstrap_zero_curve(instruments)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([i.maturity_years for i in instruments], [i.par_yield for i in instruments], marker="s", label="Input par yields (market quotes)")
    ax.plot([t for t, _ in curve], [z for _, z in curve], marker="o", label="Bootstrapped zero-coupon curve")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Rate")
    ax.set_title("Yield curve bootstrapping")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "bootstrapped_curve.png", dpi=150)
    plt.close(fig)


def plot_duration_convexity_approximation():
    metrics = bond_risk_metrics(100, 0.05, 0.05, 10, freq=2)
    dys = np.linspace(-0.03, 0.03, 25)
    exact = [bond_price(100, 0.05, 0.05 + dy, 10, freq=2) - metrics.price for dy in dys]
    approx = [estimate_price_change(metrics, dy) for dy in dys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dys * 10000, exact, label="Exact repriced change", color="black", linewidth=2)
    ax.plot(dys * 10000, approx, label="Duration + convexity estimate", color="darkorange", linestyle="--")
    ax.set_xlabel("Yield change (bps)")
    ax.set_ylabel("Bond price change")
    ax.set_title("Duration+convexity is a local approximation —\nerror grows with the size of the yield move")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "duration_convexity_approximation.png", dpi=150)
    plt.close(fig)


def plot_vasicek_paths_and_curve():
    params = VasicekParams(kappa=0.5, theta=0.04, sigma=0.02)
    r0 = 0.03
    paths = simulate_vasicek_paths(r0, T=10, params=params, n_paths=30, n_steps=500, seed=1)
    time_grid = np.linspace(0, 10, 501)

    maturities = np.linspace(0.25, 15, 40)
    yields = [vasicek_yield(r0, T, params) for T in maturities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in paths:
        ax1.plot(time_grid, path, alpha=0.4, linewidth=0.8)
    ax1.axhline(params.theta, color="black", linestyle="--", label=f"Long-run mean θ={params.theta}")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Short rate")
    ax1.set_title("Simulated Vasicek short-rate paths\n(mean-reverting toward θ)")
    ax1.legend()

    ax2.plot(maturities, yields, color="steelblue")
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("Yield")
    ax2.set_title(f"Vasicek-implied yield curve\n(r0={r0} < θ={params.theta} → upward sloping)")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "vasicek_paths_and_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_bootstrapped_curve()
    plot_duration_convexity_approximation()
    plot_vasicek_paths_and_curve()
    print("Saved figures to", FIG_DIR)
