"""
transform_pandas_quickstart.py
-------------------------------
A pandas re-implementation of transform_pyspark.py, with IDENTICAL output
columns and business logic. This exists purely so the dashboard can be run
in under a minute without needing a local Spark/Java install.

Use this for: a fast local demo, or running in this sandbox.
Use transform_pyspark.py for: what you'd actually show/describe in an
interview as the "production" version - same RFM logic, but expressed the
way a client's data platform team would expect (Spark, partitioned Parquet).

Output:
    data/processed/customer_features.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"

AS_OF_DATE = pd.Timestamp("2026-07-29")


def load_raw_tables():
    customers = pd.read_csv(RAW_DATA_DIR / "customers.csv", parse_dates=["signup_date"])
    transactions = pd.read_csv(RAW_DATA_DIR / "transactions.csv", parse_dates=["transaction_date"])
    return customers, transactions


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.dropna(subset=["customer_id", "transaction_date", "amount_sgd"])
    transactions = transactions[transactions["amount_sgd"] > 0]
    return transactions


def build_rfm_features(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    agg = transactions.groupby("customer_id").agg(
        last_purchase_date=("transaction_date", "max"),
        frequency=("transaction_id", "count"),
        monetary_sgd=("amount_sgd", "sum"),
        avg_basket_sgd=("amount_sgd", "mean"),
        categories_purchased=("category", "nunique"),
    ).reset_index()

    agg["monetary_sgd"] = agg["monetary_sgd"].round(2)
    agg["avg_basket_sgd"] = agg["avg_basket_sgd"].round(2)
    agg["recency_days"] = (AS_OF_DATE - agg["last_purchase_date"]).dt.days

    features = customers.merge(agg, on="customer_id", how="left")

    # Tenure (days since signup) - used as an ML feature instead of
    # recency_days, since recency_days is used to DEFINE the churn label
    # below. Feeding a label-defining column back in as a model feature is
    # a classic data-leakage bug - excluding it here on purpose.
    features["tenure_days"] = (AS_OF_DATE - features["signup_date"]).dt.days

    features[["frequency", "categories_purchased"]] = (
        features[["frequency", "categories_purchased"]].fillna(0).astype(int)
    )
    features[["monetary_sgd", "avg_basket_sgd"]] = features[["monetary_sgd", "avg_basket_sgd"]].fillna(0.0)
    features["recency_days"] = features["recency_days"].fillna(9999).astype(int)

    # Same business rule as the PySpark version: churned = no purchase in 90
    # days but has purchased before; brand-new signups are "unknown" (NaN).
    conditions = [
        (features["frequency"] == 0),
        (features["recency_days"] > 90) & (features["frequency"] > 0),
    ]
    choices = [np.nan, 1]
    features["churned"] = np.select(conditions, choices, default=0)

    # Quartile scores (1 = best, 4 = worst) for each RFM dimension.
    features["recency_days_quartile"] = pd.qcut(
        features["recency_days"], 4, labels=[1, 2, 3, 4], duplicates="drop"
    ).astype("Int64")
    features["frequency_quartile"] = pd.qcut(
        features["frequency"].rank(method="first"), 4, labels=[4, 3, 2, 1]
    ).astype("Int64")
    features["monetary_sgd_quartile"] = pd.qcut(
        features["monetary_sgd"].rank(method="first"), 4, labels=[4, 3, 2, 1]
    ).astype("Int64")

    return features


def main():
    customers, transactions = load_raw_tables()
    transactions = clean_transactions(transactions)
    features = build_rfm_features(customers, transactions)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(features):,} customer feature rows -> {OUTPUT_PATH}")
    summary = features.groupby("country").agg(
        customers=("customer_id", "count"),
        avg_ltv_sgd=("monetary_sgd", "mean"),
        churn_rate=("churned", "mean"),
    ).round(3)
    print(summary)


if __name__ == "__main__":
    main()
