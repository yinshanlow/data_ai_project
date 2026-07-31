"""Generates the proof plots referenced in advanced_models/README.md.

Run with: python -m advanced_models.generate_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt

from advanced_models.delta_hedging import hedging_frequency_study
from advanced_models.heston import HestonParams, heston_smile
from advanced_models.merton_jump_diffusion import merton_smile

FIG_DIR = Path(__file__).parent / "figures"


def plot_volatility_smile_comparison():
    S0, T, r = 100, 1.0, 0.05
    bs_flat_vol = 0.20
    strikes = [70, 80, 90, 100, 110, 120, 130]

    heston_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.7)
    heston_results = heston_smile(S0, strikes, T, r, heston_params, n_paths=200_000, seed=10)
    heston_ivs = [iv for _, _, iv in heston_results]

    merton_results = merton_smile(S0, strikes, T, r, sigma=0.20, jump_intensity=0.8, jump_mean=-0.08, jump_vol=0.2)
    merton_ivs = [iv for _, _, iv in merton_results]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(bs_flat_vol, color="crimson", linestyle="--", label="Black-Scholes (Part A) — flat, by construction")
    ax.plot(strikes, heston_ivs, marker="o", color="steelblue", label="Heston (stochastic vol, ρ=-0.7)")
    ax.plot(strikes, merton_ivs, marker="s", color="darkorange", label="Merton (jump-diffusion)")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title("The smile Black-Scholes cannot produce\n(same option chain: S=100, T=1y, r=5%)")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "volatility_smile_comparison.png", dpi=150)
    plt.close(fig)

    return heston_ivs, merton_ivs


def plot_hedging_frequency_study():
    results = hedging_frequency_study(100, 100, 1.0, 0.05, 0.2, frequencies=(12, 26, 52, 126, 252, 1000), n_paths=30_000, seed=0)
    freqs = [r.rebalances_per_year for r in results]
    stds = [r.pnl_std for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(freqs, stds, marker="o", color="darkgreen")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rebalancing frequency (times per year)")
    ax.set_ylabel("Std dev of hedging P&L")
    ax.set_title("Discrete delta-hedging: P&L variance shrinks toward\nthe continuous-hedging (zero-risk) limit")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "hedging_frequency_study.png", dpi=150)
    plt.close(fig)

    return results


if __name__ == "__main__":
    heston_ivs, merton_ivs = plot_volatility_smile_comparison()
    print("Heston implied vols:", [round(v, 4) for v in heston_ivs])
    print("Merton implied vols:", [round(v, 4) for v in merton_ivs])

    hedge_results = plot_hedging_frequency_study()
    print()
    for r in hedge_results:
        print(f"  rebalances/yr={r.rebalances_per_year:5d}  pnl_mean={r.pnl_mean:+.4f}  pnl_std={r.pnl_std:.4f}")

    print("\nSaved figures to", FIG_DIR)
