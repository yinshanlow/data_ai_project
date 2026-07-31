"""Generates the proof plots referenced in risk/README.md.

Run with: python -m risk.generate_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from risk.cvar import apply_historical_scenario, crisis_period_total_return
from risk.data import fetch_prices, prices_to_returns
from risk.portfolio import efficient_frontier, max_sharpe_portfolio, min_variance_portfolio
from risk.var import kupiec_test, rolling_var_backtest

FIG_DIR = Path(__file__).parent / "figures"
CRISIS_START, CRISIS_END = "2025-03-15", "2025-05-15"


def plot_efficient_frontier(returns: pd.DataFrame, is_synthetic: bool):
    frontier = efficient_frontier(returns, n_points=40)
    mv = min_variance_portfolio(returns)
    ms = max_sharpe_portfolio(returns, rf=0.04)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(frontier["volatility"], frontier["return"], color="steelblue", label="Efficient frontier (all target returns)")
    ax.scatter([mv.volatility], [mv.expected_return], color="crimson", zorder=5, s=80, label="Min-variance portfolio")
    ax.scatter([ms.volatility], [ms.expected_return], color="darkorange", zorder=5, s=80, label="Max-Sharpe portfolio (rf=4%)")

    individual_ret = returns.mean() * 252
    individual_vol = returns.std() * (252**0.5)
    ax.scatter(individual_vol, individual_ret, color="gray", marker="x", label="Individual assets")
    for ticker in returns.columns:
        ax.annotate(ticker, (individual_vol[ticker], individual_ret[ticker]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    title = "Markowitz efficient frontier"
    if is_synthetic:
        title += " (SYNTHETIC data — live fetch unavailable)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "efficient_frontier.png", dpi=150)
    plt.close(fig)


def plot_var_backtest_comparison(returns: pd.DataFrame, is_synthetic: bool):
    portfolio_returns = returns.mean(axis=1)
    methods = ["historical", "parametric", "monte_carlo"]
    full_rates, crisis_rates, p_values = [], [], []

    for method in methods:
        bt = rolling_var_backtest(portfolio_returns, method, window=250, confidence=0.95, seed=1)
        kt = kupiec_test(bt["breached"], confidence=0.95)
        full_rates.append(kt.breach_rate)
        p_values.append(kt.p_value)
        crisis_bt = bt.loc[CRISIS_START:CRISIS_END]
        crisis_rates.append(crisis_bt["breached"].mean() if len(crisis_bt) else float("nan"))

    x = range(len(methods))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], full_rates, width, label="Full-sample breach rate", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], crisis_rates, width, label=f"Crisis-window breach rate\n({CRISIS_START} to {CRISIS_END})", color="#C44E52")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="Expected rate (5%, 95% VaR)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Breach rate")
    title = "VaR backtest: breach rate clusters in the crisis window"
    if is_synthetic:
        title += " (SYNTHETIC data)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "var_backtest_comparison.png", dpi=150)
    plt.close(fig)

    return full_rates, crisis_rates, p_values


if __name__ == "__main__":
    prices, is_synthetic = fetch_prices(period="5y")
    returns = prices_to_returns(prices)

    plot_efficient_frontier(returns, is_synthetic)
    full_rates, crisis_rates, p_values = plot_var_backtest_comparison(returns, is_synthetic)

    print("Saved figures to", FIG_DIR)
    print("synthetic data:", is_synthetic)
    for method, fr, cr, p in zip(["historical", "parametric", "monte_carlo"], full_rates, crisis_rates, p_values):
        print(f"  {method:12s} full-sample breach rate={fr:.3f} (p={p:.3f})  crisis-window breach rate={cr:.3f}")

    port = max_sharpe_portfolio(returns, rf=0.04)
    scenario = crisis_period_total_return(prices, CRISIS_START, CRISIS_END)
    stress = apply_historical_scenario(port.weights, scenario)
    print(f"\nStress test: max-Sharpe portfolio under the {CRISIS_START}-{CRISIS_END} scenario -> {stress.portfolio_shock:+.2%}")
