"""
generate_data.py
-----------------
Generates a synthetic dataset for "RetailPulse" - a fictional Southeast Asian
e-commerce retailer used as the demo scenario for this project.

Business narrative (use this when presenting the demo):
    RetailPulse sells across Singapore, Malaysia, Indonesia, and Thailand
    through both an online storefront and physical outlets. Leadership wants
    to understand which customers are at risk of churning and where revenue
    is concentrated, so the growth team can prioritise retention campaigns.

This script has zero external dependencies beyond numpy/pandas, so it runs
anywhere without needing an API key or internet access.

Outputs (written to data/raw/):
    customers.csv     - one row per customer
    transactions.csv  - one row per transaction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG_SEED = 42
N_CUSTOMERS = 6000
N_TRANSACTIONS = 42000
OUTPUT_DIR = Path(__file__).parent / "raw"

FIRST_NAMES = [
    "Wei Ming", "Hui Ling", "Arif", "Nur Aisyah", "Siti", "Kumar", "Priya",
    "Jia Xin", "Farhan", "Mei Ling", "Ahmad", "Divya", "Zhi Hao", "Aina",
    "Ravi", "Xin Yi", "Haziq", "Suresh", "Chen Wei", "Nabila",
]
LAST_NAMES = [
    "Tan", "Lim", "Bin Abdullah", "Binte Ismail", "Rahman", "Wong", "Lee",
    "Kaur", "Chong", "Nair", "Yusof", "Chua", "Osman", "Muthu", "Goh",
]

COUNTRIES = ["Singapore", "Malaysia", "Indonesia", "Thailand"]
COUNTRY_WEIGHTS = [0.35, 0.30, 0.20, 0.15]  # SG/MY skew, matches the SEA job market focus

SEGMENTS = ["Budget", "Mainstream", "Premium"]
SEGMENT_WEIGHTS = [0.45, 0.40, 0.15]

CHANNELS = ["Online", "In-Store"]
CATEGORIES = [
    "Electronics", "Fashion", "Groceries", "Home & Living",
    "Beauty", "Sports & Outdoors", "Toys & Baby",
]

# Average basket size differs by category - makes the revenue charts look real
CATEGORY_PRICE_RANGE = {
    "Electronics": (80, 900),
    "Fashion": (15, 150),
    "Groceries": (5, 60),
    "Home & Living": (20, 300),
    "Beauty": (10, 120),
    "Sports & Outdoors": (20, 250),
    "Toys & Baby": (10, 100),
}

TODAY = datetime(2026, 7, 29)
SIGNUP_WINDOW_DAYS = 730       # customers signed up within the last 2 years
TXN_WINDOW_DAYS = 365          # transactions span the last 12 months


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    customer_ids = [f"CUST{100000 + i}" for i in range(N_CUSTOMERS)]

    names = [
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        for _ in range(N_CUSTOMERS)
    ]

    signup_offsets = rng.integers(0, SIGNUP_WINDOW_DAYS, size=N_CUSTOMERS)
    signup_dates = [TODAY - timedelta(days=int(d)) for d in signup_offsets]

    countries = rng.choice(COUNTRIES, size=N_CUSTOMERS, p=COUNTRY_WEIGHTS)
    segments = rng.choice(SEGMENTS, size=N_CUSTOMERS, p=SEGMENT_WEIGHTS)
    age_groups = rng.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"],
        size=N_CUSTOMERS,
        p=[0.15, 0.32, 0.28, 0.16, 0.09],
    )

    return pd.DataFrame({
        "customer_id": customer_ids,
        "customer_name": names,
        "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
        "country": countries,
        "segment": segments,
        "age_group": age_groups,
    })


def generate_transactions(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    # Give Premium customers a higher chance of transacting (more realistic RFM spread)
    segment_purchase_weight = customers["segment"].map(
        {"Premium": 2.2, "Mainstream": 1.3, "Budget": 0.8}
    ).to_numpy()
    segment_purchase_weight = segment_purchase_weight / segment_purchase_weight.sum()

    customer_idx = rng.choice(
        len(customers), size=N_TRANSACTIONS, p=segment_purchase_weight
    )
    chosen_customers = customers.iloc[customer_idx].reset_index(drop=True)

    txn_offsets = rng.integers(0, TXN_WINDOW_DAYS, size=N_TRANSACTIONS)
    txn_dates = [TODAY - timedelta(days=int(d)) for d in txn_offsets]

    categories = rng.choice(CATEGORIES, size=N_TRANSACTIONS)
    channels = rng.choice(CHANNELS, size=N_TRANSACTIONS, p=[0.62, 0.38])

    amounts = np.array([
        rng.uniform(*CATEGORY_PRICE_RANGE[c]) for c in categories
    ]).round(2)

    return pd.DataFrame({
        "transaction_id": [f"TXN{500000 + i}" for i in range(N_TRANSACTIONS)],
        "customer_id": chosen_customers["customer_id"],
        "transaction_date": [d.strftime("%Y-%m-%d") for d in txn_dates],
        "category": categories,
        "channel": channels,
        "amount_sgd": amounts,
    })


def main():
    rng = np.random.default_rng(RNG_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(rng)
    transactions = generate_transactions(rng, customers)

    customers.to_csv(OUTPUT_DIR / "customers.csv", index=False)
    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False)

    print(f"Generated {len(customers):,} customers -> {OUTPUT_DIR/'customers.csv'}")
    print(f"Generated {len(transactions):,} transactions -> {OUTPUT_DIR/'transactions.csv'}")


if __name__ == "__main__":
    main()
