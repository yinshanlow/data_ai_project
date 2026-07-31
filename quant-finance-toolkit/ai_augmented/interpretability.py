"""SHAP-based interpretability for both AI-augmented models in this module:
the return-signal XGBoost model (signal_model.py) and the volatility-forecast
XGBoost model (risk_model.py). Directly targets "model interpretability" as
named in the 2026 hiring research this project is built against — a systematic
trading or risk desk cannot deploy a model it can't explain.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass
class ImportanceResult:
    feature: str
    mean_abs_shap: float
    direction: str  # "increases" or "decreases" — sign of the average SHAP value, plain-English framing


def explain_model(model, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (importance_df sorted by mean |SHAP value| descending, raw shap_values array)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    rows = []
    for i, col in enumerate(X.columns):
        mean_abs = float(np.abs(shap_values[:, i]).mean())
        mean_signed = float(shap_values[:, i].mean())
        rows.append(ImportanceResult(feature=col, mean_abs_shap=mean_abs, direction="increases" if mean_signed > 0 else "decreases"))

    importance_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance_df, shap_values
