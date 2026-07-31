"""Historical price data for the risk module.

Fetches adjusted close prices directly from Yahoo Finance's public chart
endpoint (the same data `yfinance` wraps) rather than through the `yfinance`
library itself — in this environment, `yfinance`'s internal session/crumb
handshake was unreliable (repeatedly 429'd) while a plain HTTP GET with a
browser User-Agent succeeded. If the environment running this has no network
access at all (or Yahoo blocks the IP outright), fetch_prices() falls back to
a synthetic, clearly-labeled multi-asset price series so the rest of the risk
module remains runnable and testable offline.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]


def _fetch_one(ticker: str, range_: str, interval: str) -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    resp = requests.get(
        url, params={"range": range_, "interval": interval},
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]
    index = pd.to_datetime(timestamps, unit="s").normalize()
    return pd.Series(adj_close, index=index, name=ticker)


def _synthetic_prices(tickers: list[str], n_days: int, seed: int = 0) -> pd.DataFrame:
    """Correlated GBM price paths — used only when live data is unavailable.

    Every consumer of this data must be told plainly when it's synthetic;
    fetch_prices() logs a clear warning and callers should surface that
    upstream (the risk module's README states this explicitly too).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    corr = np.full((n_assets, n_assets), 0.4)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    annual_vol = rng.uniform(0.18, 0.35, n_assets)
    annual_drift = rng.uniform(0.04, 0.12, n_assets)
    dt = 1 / 252

    z = rng.standard_normal((n_days, n_assets)) @ chol.T
    daily_returns = (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * z
    log_prices = np.log(100.0) + np.cumsum(daily_returns, axis=0)
    prices = np.exp(log_prices)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def fetch_prices(
    tickers: list[str] = DEFAULT_TICKERS, period: str = "3y", interval: str = "1d", use_cache: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Returns (prices_df, is_synthetic). Caches successful live fetches to CSV."""
    cache_path = CACHE_DIR / f"prices_{'_'.join(tickers)}_{period}_{interval}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df, False

    try:
        series_list = [_fetch_one(t, period, interval) for t in tickers]
        df = pd.concat(series_list, axis=1).dropna()
        if df.empty or len(df) < 30:
            raise ValueError("fetched series too short or empty")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        return df, False
    except Exception as exc:
        print(f"[risk.data] Live fetch failed ({exc!r}) — falling back to SYNTHETIC price data. "
              f"Results using this data are illustrative only, not real market behaviour.")
        n_days = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}.get(period, 756)
        return _synthetic_prices(tickers, n_days), True


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
