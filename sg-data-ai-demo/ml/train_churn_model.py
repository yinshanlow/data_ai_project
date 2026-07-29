"""
train_churn_model.py
---------------------
Trains a churn-risk classifier on the RFM feature table, and writes out:
    ml/artifacts/churn_model.pkl      - the trained model
    ml/artifacts/feature_importance.csv
    ml/artifacts/model_metrics.json
    data/processed/customer_scored.csv (features + churn probability, used by the dashboard)

Model choice: Random Forest. This is a deliberate presales talking point -
it's easy to explain to a non-technical stakeholder ("the model learned that
recency and purchase frequency matter most"), and feature_importance_ gives
you an instant, defensible slide for a client meeting without needing SHAP.

Run:
    python ml/train_churn_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
SCORED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_scored.csv"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

NUMERIC_FEATURES = [
    "tenure_days", "frequency", "monetary_sgd", "avg_basket_sgd", "categories_purchased",
]
CATEGORICAL_FEATURES = ["country", "segment", "age_group"]
TARGET = "churned"

# NOTE ON LEAKAGE: recency_days is deliberately excluded from the feature
# set. The "churned" label is defined as recency_days > 90, so including
# recency_days as a model input would let the model trivially re-learn that
# threshold (accuracy/AUC ~1.0) instead of genuinely predicting risk from
# purchase behaviour. tenure_days (signup age) is used instead - it's a
# legitimate, label-independent signal.


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    # Only customers with a purchase history have a known churn label -
    # brand-new signups (churned == NaN) are excluded from training, but
    # kept in the scored output so the dashboard can still show them.
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_feature_names(pipeline: Pipeline) -> list:
    cat_encoder: OneHotEncoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_training_data()

    labelled = df.dropna(subset=[TARGET]).copy()
    labelled[TARGET] = labelled[TARGET].astype(int)

    X = labelled[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = labelled[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate_in_data": round(y.mean(), 4),
    }
    print("Model performance:", json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Feature importance -> the slide you'd actually show a client
    importances = pipeline.named_steps["model"].feature_importances_
    feature_names = get_feature_names(pipeline)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
    print("\nTop 5 churn drivers:\n", importance_df.head())

    joblib.dump(pipeline, ARTIFACTS_DIR / "churn_model.pkl")
    with open(ARTIFACTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Score EVERY customer (including unlabelled new signups) for the dashboard.
    all_X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df["churn_probability"] = pipeline.predict_proba(all_X)[:, 1].round(4)
    df["churn_risk_band"] = pd.cut(
        df["churn_probability"], bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )
    df.to_csv(SCORED_OUTPUT_PATH, index=False)
    print(f"\nScored {len(df):,} customers -> {SCORED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
