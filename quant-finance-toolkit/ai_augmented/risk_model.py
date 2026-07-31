"""A small ML model predicting forward realized volatility — the interpretability
target for the risk side of Part E (SHAP applied to a risk model, as distinct
from SHAP applied to the return-signal model in signal_model.py).

Predicting volatility rather than trying to force SHAP onto the VaR formulas
in risk/var.py directly: those are closed-form/empirical-quantile calculations,
not fitted models, so there's nothing for SHAP to attribute. A volatility
forecasting model is the natural fitted-model stand-in for "explain what
drives this portfolio's risk" that Part D's VaR figures are downstream of.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

RISK_FEATURE_COLUMNS = ["trailing_vol_21", "trailing_vol_63", "trailing_skew_21", "trailing_mean_return_21", "market_vol_21"]


def build_volatility_panel(prices: pd.DataFrame, market_ticker: str = "SPY") -> pd.DataFrame:
    """Target: forward_vol_21 = realized annualized volatility over the NEXT 21 trading
    days for the equal-weighted portfolio of non-market tickers. Features are all
    backward-looking (computed only from data available at each date)."""
    returns = prices.pct_change()
    asset_tickers = [c for c in prices.columns if c != market_ticker]
    portfolio_returns = returns[asset_tickers].mean(axis=1)
    market_returns = returns[market_ticker]

    df = pd.DataFrame(index=prices.index)
    df["trailing_vol_21"] = portfolio_returns.rolling(21).std() * np.sqrt(252)
    df["trailing_vol_63"] = portfolio_returns.rolling(63).std() * np.sqrt(252)
    df["trailing_skew_21"] = portfolio_returns.rolling(21).skew()
    df["trailing_mean_return_21"] = portfolio_returns.rolling(21).mean() * 252
    df["market_vol_21"] = market_returns.rolling(21).std() * np.sqrt(252)
    df["forward_vol_21"] = portfolio_returns.shift(-21).rolling(21).std() * np.sqrt(252)

    return df.dropna(subset=RISK_FEATURE_COLUMNS + ["forward_vol_21"]).reset_index().rename(columns={"index": "date"})


def train_volatility_model(panel: pd.DataFrame, train_frac: float = 0.7) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    cutoff_idx = int(len(panel) * train_frac)
    train, test = panel.iloc[:cutoff_idx], panel.iloc[cutoff_idx:]

    model = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[RISK_FEATURE_COLUMNS], train["forward_vol_21"])
    return model, train, test
