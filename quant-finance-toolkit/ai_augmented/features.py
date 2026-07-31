"""Price/volume-based factor features for the ML signal model.

Note on provenance: `quant-research-lab` (referenced in this repo's top-level
README) isn't available in this workspace to import its RFM-style factors
directly, so this module reconstructs comparable factors in the same spirit
— momentum, volatility, and short-term reversal — computed from the same
market data pipeline `risk/data.py` already built for Part D, rather than
importing code that isn't actually present.

Every feature here is a legitimate, textbook cross-sectional equity factor:
momentum (12-1 style, at shorter horizons for the data window available),
realized volatility, and short-term reversal — the well-documented families
academic and practitioner factor research keeps rediscovering.
"""
import numpy as np
import pandas as pd


def build_factor_panel(prices: pd.DataFrame, market_ticker: str = "SPY") -> pd.DataFrame:
    """Returns a long-format panel: one row per (date, ticker), with factor columns
    and a forward_return_5d label column (the next 5 trading days' return — the
    prediction target). Rows with NaN features or labels (start/end of sample) are dropped.
    """
    returns = prices.pct_change()
    market_returns = returns[market_ticker]
    asset_tickers = [c for c in prices.columns if c != market_ticker]

    rows = []
    for ticker in asset_tickers:
        r = returns[ticker]
        df = pd.DataFrame(index=prices.index)
        df["ticker"] = ticker
        df["mom_21"] = prices[ticker].pct_change(21)
        df["mom_63"] = prices[ticker].pct_change(63)
        df["vol_21"] = r.rolling(21).std() * np.sqrt(252)
        df["reversal_5"] = -prices[ticker].pct_change(5)
        df["rel_strength_21"] = prices[ticker].pct_change(21) - prices[market_ticker].pct_change(21)
        df["forward_return_5d"] = prices[ticker].pct_change(5).shift(-5)
        rows.append(df)

    panel = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    feature_cols = ["mom_21", "mom_63", "vol_21", "reversal_5", "rel_strength_21"]
    panel = panel.dropna(subset=feature_cols + ["forward_return_5d"])
    return panel.reset_index(drop=True)


FEATURE_COLUMNS = ["mom_21", "mom_63", "vol_21", "reversal_5", "rel_strength_21"]
