"""Generates the proof plots referenced in ai_augmented/README.md.

Run with: python -m ai_augmented.generate_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt

from ai_augmented.features import FEATURE_COLUMNS, build_factor_panel
from ai_augmented.interpretability import explain_model
from ai_augmented.risk_model import RISK_FEATURE_COLUMNS, build_volatility_panel, train_volatility_model
from ai_augmented.signal_model import chronological_split, compare_baseline_vs_ml
from risk.data import fetch_prices, prices_to_returns

FIG_DIR = Path(__file__).parent / "figures"


def plot_signal_comparison(baseline, ml):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    names = [baseline.name, ml.name]
    colors = ["#888888", "#4C72B0"]

    axes[0].bar(names, [baseline.ic, ml.ic], color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Out-of-sample Spearman IC")
    axes[0].set_title("Signal quality: baseline vs. ML")
    axes[0].tick_params(axis="x", labelrotation=15)

    axes[1].bar(names, [baseline.long_short_sharpe, ml.long_short_sharpe], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Long-top / short-bottom Sharpe (annualized, approx.)")
    axes[1].set_title("Resulting portfolio Sharpe")
    axes[1].tick_params(axis="x", labelrotation=15)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "signal_comparison.png", dpi=150)
    plt.close(fig)


def plot_shap_importance(importance_df, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1], color="#55A868")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    prices, is_synthetic = fetch_prices(period="5y")

    panel = build_factor_panel(prices)
    baseline, ml, model = compare_baseline_vs_ml(panel)
    _, test = chronological_split(panel)
    signal_importance, _ = explain_model(model, test[FEATURE_COLUMNS])

    vol_panel = build_volatility_panel(prices)
    vol_model, _, vol_test = train_volatility_model(vol_panel)
    risk_importance, _ = explain_model(vol_model, vol_test[RISK_FEATURE_COLUMNS])

    plot_signal_comparison(baseline, ml)
    plot_shap_importance(signal_importance, "SHAP feature importance — return-signal model", "shap_signal_importance.png")
    plot_shap_importance(risk_importance, "SHAP feature importance — volatility-forecast model", "shap_risk_importance.png")

    print("synthetic data:", is_synthetic)
    print(baseline)
    print(ml)
    print()
    print("Signal model SHAP importance:\n", signal_importance)
    print()
    print("Volatility model SHAP importance:\n", risk_importance)
