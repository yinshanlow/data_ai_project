import numpy as np
import pandas as pd
import pytest

from ai_augmented.features import FEATURE_COLUMNS, build_factor_panel
from ai_augmented.interpretability import explain_model
from ai_augmented.risk_model import RISK_FEATURE_COLUMNS, build_volatility_panel, train_volatility_model
from ai_augmented.signal_model import (
    baseline_momentum_signal,
    chronological_split,
    compare_baseline_vs_ml,
    evaluate_signal,
)


@pytest.fixture
def synthetic_prices():
    rng = np.random.default_rng(11)
    n = 800
    dates = pd.bdate_range("2022-01-03", periods=n)
    tickers = ["A", "B", "C", "SPY"]
    corr = np.full((4, 4), 0.3)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, 4)) @ chol.T
    daily_returns = 0.0003 + 0.015 * z
    log_prices = np.log(100) + np.cumsum(daily_returns, axis=0)
    return pd.DataFrame(np.exp(log_prices), index=dates, columns=tickers)


def test_factor_panel_has_no_nans(synthetic_prices):
    panel = build_factor_panel(synthetic_prices)
    assert not panel[FEATURE_COLUMNS + ["forward_return_5d"]].isna().any().any()
    assert set(panel["ticker"]) == {"A", "B", "C"}


def test_chronological_split_no_leakage(synthetic_prices):
    panel = build_factor_panel(synthetic_prices)
    train, test = chronological_split(panel)
    assert train["date"].max() < test["date"].min()
    assert len(train) > 0 and len(test) > 0


def test_baseline_signal_matches_manual_calculation(synthetic_prices):
    panel = build_factor_panel(synthetic_prices)
    signal = baseline_momentum_signal(panel)
    expected = panel["mom_21"] + panel["mom_63"]
    pd.testing.assert_series_equal(signal, expected, check_names=False)


def test_evaluate_signal_ic_in_valid_range(synthetic_prices):
    panel = build_factor_panel(synthetic_prices)
    _, test = chronological_split(panel)
    signal = baseline_momentum_signal(test)
    result = evaluate_signal("test", test, signal)
    assert -1.0 <= result.ic <= 1.0
    assert 0.0 <= result.ic_p_value <= 1.0


def test_compare_baseline_vs_ml_runs_end_to_end(synthetic_prices):
    panel = build_factor_panel(synthetic_prices)
    baseline, ml, model = compare_baseline_vs_ml(panel)
    assert -1.0 <= baseline.ic <= 1.0
    assert -1.0 <= ml.ic <= 1.0
    assert hasattr(model, "predict")


def test_shap_importance_shape_matches_features(synthetic_prices):
    panel = build_factor_panel(synthetic_prices)
    _, ml, model = compare_baseline_vs_ml(panel)
    _, test = chronological_split(panel)
    importance_df, shap_values = explain_model(model, test[FEATURE_COLUMNS])

    assert set(importance_df["feature"]) == set(FEATURE_COLUMNS)
    assert shap_values.shape == (len(test), len(FEATURE_COLUMNS))
    assert (importance_df["mean_abs_shap"] >= 0).all()
    assert importance_df["mean_abs_shap"].is_monotonic_decreasing


def test_volatility_model_and_shap(synthetic_prices):
    vol_panel = build_volatility_panel(synthetic_prices)
    assert not vol_panel[RISK_FEATURE_COLUMNS + ["forward_vol_21"]].isna().any().any()

    model, train, test = train_volatility_model(vol_panel)
    assert len(train) > 0 and len(test) > 0

    importance_df, shap_values = explain_model(model, test[RISK_FEATURE_COLUMNS])
    assert set(importance_df["feature"]) == set(RISK_FEATURE_COLUMNS)
    assert shap_values.shape == (len(test), len(RISK_FEATURE_COLUMNS))
