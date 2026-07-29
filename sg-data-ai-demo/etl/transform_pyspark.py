"""
transform_pyspark.py
---------------------
PySpark ETL layer: reads the raw customer/transaction CSVs, cleans them, and
builds a customer-level RFM (Recency / Frequency / Monetary) feature table.

This is the layer you'd point to in an interview when a job post asks for
"hands-on ETL / PySpark / modern data platform experience" - it mirrors what
you'd build on Databricks or EMR, just running locally here.

Run locally:
    pip install pyspark
    python etl/transform_pyspark.py

Run on Databricks:
    Upload data/raw/*.csv to a Volume or DBFS path, update RAW_DATA_DIR below,
    and run this as a notebook/job - the logic is unchanged.

Output:
    data/processed/customer_features/  (Parquet, partitioned by country)
"""

from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "customer_features"

# The "as of" date the demo pretends to run on - keeps recency calculations
# stable and reproducible instead of depending on wall-clock time.
AS_OF_DATE = "2026-07-29"


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("RetailPulseCustomerFeatures")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")  # small dataset, keep it light locally
        .getOrCreate()
    )


def load_raw_tables(spark: SparkSession):
    customers = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(str(RAW_DATA_DIR / "customers.csv"))
    )
    transactions = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv(str(RAW_DATA_DIR / "transactions.csv"))
    )
    return customers, transactions


def clean_transactions(transactions):
    """Basic data-quality pass: drop nulls/negatives, cast types.
    In a real client engagement this is where you'd also log rejected rows
    to a quarantine table for the client's data-governance team.
    """
    return (
        transactions
        .dropna(subset=["customer_id", "transaction_date", "amount_sgd"])
        .filter(F.col("amount_sgd") > 0)
        .withColumn("transaction_date", F.to_date("transaction_date"))
    )


def build_rfm_features(customers, transactions):
    """Aggregate transactions into per-customer Recency / Frequency / Monetary
    features, then join back onto the customer master table.
    """
    as_of = F.to_date(F.lit(AS_OF_DATE))

    agg = (
        transactions.groupBy("customer_id")
        .agg(
            F.max("transaction_date").alias("last_purchase_date"),
            F.count("transaction_id").alias("frequency"),
            F.round(F.sum("amount_sgd"), 2).alias("monetary_sgd"),
            F.round(F.avg("amount_sgd"), 2).alias("avg_basket_sgd"),
            F.countDistinct("category").alias("categories_purchased"),
        )
        .withColumn("recency_days", F.datediff(as_of, F.col("last_purchase_date")))
    )

    features = (
        customers
        .withColumn("signup_date", F.to_date("signup_date"))
        .withColumn("tenure_days", F.datediff(as_of, F.col("signup_date")))
        .join(agg, on="customer_id", how="left")
        # Customers with zero transactions are exactly the ones a client cares
        # about most - never drop them, fill sensible defaults instead.
        .fillna({
            "frequency": 0,
            "monetary_sgd": 0.0,
            "avg_basket_sgd": 0.0,
            "categories_purchased": 0,
            "recency_days": 9999,
        })
    )

    # Business rule (documented so it's easy to defend in a client Q&A):
    # "churn risk" = no purchase in the last 90 days AND at least one
    # historical purchase (brand-new signups aren't "churned", they just
    # haven't bought yet).
    features = features.withColumn(
        "churned",
        F.when((F.col("recency_days") > 90) & (F.col("frequency") > 0), 1)
         .when(F.col("frequency") == 0, F.lit(None).cast("int"))  # unknown / not yet a customer
         .otherwise(0),
    )

    # Simple RFM score (1-4 quartile per metric) - a classic presales talking
    # point: shows you can translate raw numbers into a business-friendly score.
    for col_name, ascending in [("recency_days", True), ("frequency", False), ("monetary_sgd", False)]:
        w = Window.orderBy(F.col(col_name).asc() if ascending else F.col(col_name).desc())
        features = features.withColumn(
            f"{col_name}_quartile",
            F.ntile(4).over(w),
        )

    return features


def main():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    customers, transactions = load_raw_tables(spark)
    transactions = clean_transactions(transactions)
    features = build_rfm_features(customers, transactions)

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    (
        features.write.mode("overwrite")
        .partitionBy("country")
        .parquet(str(OUTPUT_DIR))
    )

    print(f"Wrote customer feature table to {OUTPUT_DIR}")
    features.groupBy("country").agg(
        F.count("*").alias("customers"),
        F.round(F.avg("monetary_sgd"), 2).alias("avg_ltv_sgd"),
        F.round(F.avg(F.col("churned").cast("double")), 3).alias("churn_rate"),
    ).orderBy("country").show()

    spark.stop()


if __name__ == "__main__":
    main()
