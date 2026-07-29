"""
app.py - RetailPulse Customer Intelligence Dashboard
------------------------------------------------------
The client-facing artifact for this project. This is what you'd actually
screen-share in an interview: "here's a PoC I built end-to-end - synthetic
data, PySpark ETL, a churn model, and this dashboard on top."

Run:
    pip install streamlit plotly
    streamlit run dashboard/app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORED_PATH = PROJECT_ROOT / "data" / "processed" / "customer_scored.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "ml" / "artifacts" / "feature_importance.csv"

st.set_page_config(
    page_title="RetailPulse Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    scored_path = SCORED_PATH if SCORED_PATH.exists() else FEATURES_PATH
    df = pd.read_csv(scored_path)
    importance = pd.read_csv(IMPORTANCE_PATH) if IMPORTANCE_PATH.exists() else None
    return df, importance


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("RetailPulse")
    st.sidebar.caption("Customer Intelligence PoC — demo build")

    countries = sorted(df["country"].unique())
    segments = sorted(df["segment"].unique())

    selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
    selected_segments = st.sidebar.multiselect("Segment", segments, default=segments)

    st.sidebar.divider()
    st.sidebar.markdown("### Demo talking points")
    st.sidebar.markdown(
        "- Data: synthetic, 6,000 customers / 42,000 transactions across SG/MY/ID/TH\n"
        "- Pipeline: PySpark ETL → RFM features → Random Forest churn model\n"
        "- **Ask me**: how the churn definition, feature choices, or model "
        "evaluation would need to change for a real client's data"
    )

    filtered = df[df["country"].isin(selected_countries) & df["segment"].isin(selected_segments)]
    return filtered


def render_kpis(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Total Revenue (SGD)", f"${df['monetary_sgd'].sum():,.0f}")
    col3.metric("Avg. Customer LTV", f"${df['monetary_sgd'].mean():,.0f}")

    known_churn = df["churned"].dropna()
    churn_rate = known_churn.mean() if len(known_churn) else float("nan")
    col4.metric("Churn Rate (known customers)", f"{churn_rate:.1%}" if churn_rate == churn_rate else "n/a")


def render_revenue_charts(df: pd.DataFrame):
    left, right = st.columns(2)

    with left:
        st.subheader("Revenue by Country")
        by_country = df.groupby("country", as_index=False)["monetary_sgd"].sum().sort_values(
            "monetary_sgd", ascending=False
        )
        fig = px.bar(by_country, x="country", y="monetary_sgd", text_auto=".2s")
        fig.update_layout(yaxis_title="Revenue (SGD)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Customers by Segment")
        by_segment = df["segment"].value_counts().reset_index()
        by_segment.columns = ["segment", "customers"]
        fig = px.pie(by_segment, names="segment", values="customers", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)


def render_churn_section(df: pd.DataFrame, importance: pd.DataFrame | None):
    st.subheader("Churn Risk")

    if "churn_risk_band" not in df.columns:
        st.info("Run `python ml/train_churn_model.py` to generate churn scores for this section.")
        return

    left, right = st.columns([2, 1])

    with left:
        st.markdown("**Churn risk distribution by country**")
        risk_counts = (
            df.groupby(["country", "churn_risk_band"], observed=True)
            .size()
            .reset_index(name="customers")
        )
        fig = px.bar(
            risk_counts, x="country", y="customers", color="churn_risk_band",
            barmode="stack",
            category_orders={"churn_risk_band": ["Low", "Medium", "High"]},
            color_discrete_map={"Low": "#2ca02c", "Medium": "#ff9f1c", "High": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if importance is not None:
            st.markdown("**What drives churn risk?**")
            top5 = importance.head(5).sort_values("importance")
            fig = px.bar(top5, x="importance", y="feature", orientation="h")
            fig.update_layout(yaxis_title="", xaxis_title="Model importance")
            st.plotly_chart(fig, use_container_width=True)


def render_customer_lookup(df: pd.DataFrame):
    st.subheader("Customer Lookup")
    st.caption("Pull up any single customer, the way you would live in front of a client.")

    customer_id = st.selectbox("Customer ID", df["customer_id"].sort_values())
    row = df[df["customer_id"] == customer_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", row["segment"])
    c2.metric("Country", row["country"])
    c3.metric("Lifetime Spend", f"${row['monetary_sgd']:,.0f}")
    c4.metric("Last Purchase (days ago)", f"{int(row['recency_days'])}" if pd.notna(row["recency_days"]) else "n/a")

    if "churn_probability" in df.columns and pd.notna(row.get("churn_probability")):
        st.progress(
            min(float(row["churn_probability"]), 1.0),
            text=f"Predicted churn probability: {row['churn_probability']:.0%} "
                 f"({row.get('churn_risk_band', 'n/a')} risk)",
        )


def main():
    df, importance = load_data()

    st.title("RetailPulse — Customer Intelligence")
    st.caption(
        "Demo PoC: synthetic Southeast Asian e-commerce data → PySpark ETL → "
        "churn model → this dashboard. Built to show how I'd scope and deliver "
        "a first client PoC end-to-end."
    )

    filtered = render_sidebar(df)
    render_kpis(filtered)
    st.divider()
    render_revenue_charts(filtered)
    st.divider()
    render_churn_section(filtered, importance)
    st.divider()
    render_customer_lookup(filtered)


if __name__ == "__main__":
    main()
