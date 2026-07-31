"""ML-based signal generation, benchmarked honestly against a simple hand-crafted factor.

Two models predict the same target (5-day forward return) from the same
features: a genuinely simple, unfit momentum rule (no training — it can't
overfit by construction) and a gradient-boosted tree model (XGBoost) trained
on a chronological train/test split. Both are scored the same way
out-of-sample. Whichever wins, that result gets reported — see
ai_augmented/README.md for the actual outcome and why it's not a case for
throwing the ML model away even where it doesn't clearly beat the baseline.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from ai_augmented.features import FEATURE_COLUMNS


@dataclass
class SignalEvalResult:
    name: str
    ic: float  # pooled Spearman rank correlation between signal and forward_return_5d, OOS
    ic_p_value: float
    long_short_sharpe: float  # annualized Sharpe of a daily long-top/short-bottom-ticker spread


def chronological_split(panel: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by date (not row) so every ticker's observation for a given date stays
    on the same side of the split — the only way to avoid look-ahead leakage here."""
    dates = np.sort(panel["date"].unique())
    cutoff = dates[int(len(dates) * train_frac)]
    train = panel[panel["date"] < cutoff].copy()
    test = panel[panel["date"] >= cutoff].copy()
    return train, test


def baseline_momentum_signal(panel: pd.DataFrame) -> pd.Series:
    """A genuinely simple, hand-crafted, unfit rule: raw 21d + 63d momentum, unweighted.
    No parameters are estimated from data, so there's no way for this to overfit."""
    return panel["mom_21"] + panel["mom_63"]


def train_xgboost_signal(train: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train["forward_return_5d"])
    return model


def _long_short_sharpe(test: pd.DataFrame, signal: pd.Series) -> float:
    """Each day, go long the ticker with the highest signal and short the one with the
    lowest, among that day's covered tickers. Small-universe (4 tickers) caveat: see README."""
    df = test.copy()
    df["signal"] = signal.values
    daily = []
    for date, group in df.groupby("date"):
        if len(group) < 2:
            continue
        top = group.loc[group["signal"].idxmax(), "forward_return_5d"]
        bottom = group.loc[group["signal"].idxmin(), "forward_return_5d"]
        daily.append(top - bottom)
    daily = pd.Series(daily)
    if daily.std() == 0 or len(daily) == 0:
        return 0.0
    # forward_return_5d observations overlap (5-day return computed daily), so this is a
    # rough annualization, not a rigorous non-overlapping-period Sharpe — stated in the README.
    return float(daily.mean() / daily.std() * np.sqrt(252 / 5))


def evaluate_signal(name: str, test: pd.DataFrame, signal: pd.Series) -> SignalEvalResult:
    ic, p_value = spearmanr(signal, test["forward_return_5d"])
    sharpe = _long_short_sharpe(test, signal)
    return SignalEvalResult(name=name, ic=float(ic), ic_p_value=float(p_value), long_short_sharpe=sharpe)


def compare_baseline_vs_ml(panel: pd.DataFrame) -> tuple[SignalEvalResult, SignalEvalResult, xgb.XGBRegressor]:
    train, test = chronological_split(panel)

    baseline_signal_test = baseline_momentum_signal(test)
    baseline_result = evaluate_signal("Hand-crafted momentum (unfit)", test, baseline_signal_test)

    model = train_xgboost_signal(train)
    ml_signal_test = pd.Series(model.predict(test[FEATURE_COLUMNS]), index=test.index)
    ml_result = evaluate_signal("XGBoost (trained on train split)", test, ml_signal_test)

    return baseline_result, ml_result, model
