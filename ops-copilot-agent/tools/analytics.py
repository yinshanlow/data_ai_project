"""Analytics tool: read-only lookups over RetailPulse's scored customer data.

Deliberately NOT an open text-to-SQL interface — that would let an LLM (or a
malformed mock-mode regex) construct arbitrary queries against a CSV with no
validation, which is a real injection/hallucination surface for a demo this
size. Instead this exposes a small, fixed set of typed functions. See the
README for why that trade-off was made on purpose, not out of laziness.

Depends on sg-data-ai-demo already having been run at least once
(python data/generate_data.py && python etl/transform_pandas_quickstart.py &&
python ml/train_churn_model.py) so data/processed/customer_scored.csv and
ml/artifacts/*.csv exist.
"""
import json
from pathlib import Path

import pandas as pd

RETAILPULSE_DIR = Path(__file__).resolve().parent.parent.parent / "sg-data-ai-demo"
CUSTOMER_SCORED_PATH = RETAILPULSE_DIR / "data" / "processed" / "customer_scored.csv"
FEATURE_IMPORTANCE_PATH = RETAILPULSE_DIR / "ml" / "artifacts" / "feature_importance.csv"
MODEL_METRICS_PATH = RETAILPULSE_DIR / "ml" / "artifacts" / "model_metrics.json"

_customers_df: pd.DataFrame | None = None


def _load_customers() -> pd.DataFrame:
    global _customers_df
    if _customers_df is None:
        if not CUSTOMER_SCORED_PATH.exists():
            raise FileNotFoundError(
                f"RetailPulse scored data not found at {CUSTOMER_SCORED_PATH}. "
                "Run the sg-data-ai-demo pipeline first: python data/generate_data.py "
                "&& python etl/transform_pandas_quickstart.py && python ml/train_churn_model.py"
            )
        _customers_df = pd.read_csv(CUSTOMER_SCORED_PATH)
    return _customers_df


def valid_countries() -> list[str]:
    return sorted(_load_customers()["country"].unique().tolist())


def get_customer_churn_risk(customer_id: str) -> dict:
    df = _load_customers()
    match = df[df["customer_id"].str.upper() == customer_id.strip().upper()]
    if match.empty:
        return {"error": f"No customer found with ID '{customer_id}'."}

    r = match.iloc[0]
    return {
        "customer_id": r["customer_id"],
        "country": r["country"],
        "segment": r["segment"],
        "churn_probability": round(float(r["churn_probability"]), 4),
        "churn_risk_band": r["churn_risk_band"],
        "recency_days": int(r["recency_days"]),
        "frequency": int(r["frequency"]),
        "monetary_sgd": round(float(r["monetary_sgd"]), 2),
        "tenure_days": int(r["tenure_days"]),
    }


def get_country_kpis(country: str) -> dict:
    df = _load_customers()
    subset = df[df["country"].str.lower() == country.strip().lower()]
    if subset.empty:
        return {"error": f"No data for country '{country}'. Valid markets: {', '.join(valid_countries())}."}

    return {
        "country": subset["country"].iloc[0],
        "customer_count": int(len(subset)),
        "avg_churn_probability": round(float(subset["churn_probability"].mean()), 4),
        "high_risk_count": int((subset["churn_risk_band"] == "High").sum()),
        "total_monetary_sgd": round(float(subset["monetary_sgd"].sum()), 2),
    }


def get_churn_risk_drivers(top_n: int = 5) -> dict:
    if not FEATURE_IMPORTANCE_PATH.exists() or not MODEL_METRICS_PATH.exists():
        return {"error": "Churn model artifacts not found. Run sg-data-ai-demo/ml/train_churn_model.py first."}

    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH).sort_values("importance", ascending=False).head(top_n)
    metrics = json.loads(MODEL_METRICS_PATH.read_text())
    return {
        "top_drivers": [
            {"feature": row["feature"], "importance": round(float(row["importance"]), 4)}
            for _, row in fi.iterrows()
        ],
        "model_accuracy": metrics["accuracy"],
        "model_roc_auc": metrics["roc_auc"],
    }
