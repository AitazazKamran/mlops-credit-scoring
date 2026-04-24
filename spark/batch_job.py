"""
Spark Batch Job — Medallion Pipeline (Bronze → Silver → Gold)
Reads from Kafka batch topic and processes through 3 layers:
  Bronze  → Raw Parquet in MinIO
  Silver  → Cleaned + validated data
  Gold    → Feature engineered → PostgreSQL (gold_features table)
"""

import os
import json
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from kafka import KafkaConsumer

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
KAFKA_BOOTSTRAP   = "kafka:29092"
BATCH_TOPIC       = "loan-applications-batch"
KAFKA_GROUP_ID    = "spark-batch-consumer"
KAFKA_TIMEOUT_MS  = 10000   # stop consuming after 10s of no messages

MINIO_ENDPOINT    = "http://minio:9000"
MINIO_BUCKET      = "mlflow-artifacts"
BRONZE_PREFIX     = "bronze/loan_applications"

PG_HOST           = "postgres"
PG_PORT           = 5432
PG_DB             = "credit_scoring"
PG_USER           = "mlops"
PG_PASSWORD       = "mlops123"

PRINT_EVERY       = 1000

# Loan statuses we treat as DEFAULT (bad loan = 1)
DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
}


# ─────────────────────────────────────────
# BRONZE — Read from Kafka, save raw Parquet
# ─────────────────────────────────────────
def bronze_layer() -> pd.DataFrame:
    print("\n" + "="*55)
    print("  BRONZE LAYER — Reading from Kafka")
    print("="*55)

    consumer = KafkaConsumer(
        BATCH_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=KAFKA_TIMEOUT_MS,
    )

    records = []
    for i, msg in enumerate(consumer, 1):
        records.append(msg.value)
        if i % PRINT_EVERY == 0:
            print(f"  Read {i:,} records from Kafka...")

    consumer.close()
    print(f"✅ Bronze: consumed {len(records):,} records from Kafka")

    if not records:
        raise RuntimeError("❌ No records found in Kafka batch topic. Run producer.py first.")

    df = pd.DataFrame(records)

    # Save as parquet locally inside container
    os.makedirs("/tmp/bronze", exist_ok=True)
    parquet_path = "/tmp/bronze/loan_applications.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Bronze: saved {len(df):,} rows to {parquet_path}")

    # Upload to MinIO via boto3
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "admin"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "password123"),
        )
        s3.upload_file(parquet_path, MINIO_BUCKET, f"{BRONZE_PREFIX}/loan_applications.parquet")
        print(f"✅ Bronze: uploaded to MinIO s3://{MINIO_BUCKET}/{BRONZE_PREFIX}/")
    except Exception as e:
        print(f"⚠️  MinIO upload skipped (will continue): {e}")

    return df


# ─────────────────────────────────────────
# SILVER — Clean, validate, label
# ─────────────────────────────────────────
def silver_layer(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*55)
    print("  SILVER LAYER — Cleaning & Validating")
    print("="*55)

    original_count = len(df)

    # 1. Keep only relevant columns
    keep_cols = [
        "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
        "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "issue_d", "loan_status", "purpose",
        "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "initial_list_status", "application_type", "issue_year",
    ]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    # 2. Fix data types
    numeric_cols = [
        "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
        "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Clean string columns
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "").str.strip()
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "").str.strip()
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.replace(" months", "").str.strip()
        df["term"] = pd.to_numeric(df["term"], errors="coerce")

    # 4. Drop rows missing critical fields
    critical_cols = ["loan_amnt", "annual_inc", "dti", "loan_status"]
    critical_available = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=critical_available)

    # 5. Create binary label (1 = default, 0 = fully paid)
    if "loan_status" in df.columns:
        df["label"] = df["loan_status"].apply(
            lambda s: 1 if s in DEFAULT_STATUSES else 0
        )
        # Keep only clear outcomes (drop "Current", "In Grace Period" etc.)
        clear_statuses = DEFAULT_STATUSES | {"Fully Paid"}
        df = df[df["loan_status"].isin(clear_statuses)]

    # 6. Validation checks
    print(f"\n  Validation Report:")
    print(f"  Rows before cleaning : {original_count:,}")
    print(f"  Rows after cleaning  : {len(df):,}")
    print(f"  Rows dropped         : {original_count - len(df):,}")

    if "label" in df.columns:
        label_dist = df["label"].value_counts()
        print(f"  Label distribution   : {label_dist.to_dict()}")

    assert len(df) > 100, "❌ Validation failed: too few rows after cleaning"
    assert df["loan_amnt"].min() > 0, "❌ Validation failed: negative loan amounts found"
    print("✅ Silver: all validations passed")

    # Save silver parquet
    os.makedirs("/tmp/silver", exist_ok=True)
    silver_path = "/tmp/silver/loan_applications_clean.parquet"
    df.to_parquet(silver_path, index=False)
    print(f"✅ Silver: saved {len(df):,} rows to {silver_path}")

    return df


# ─────────────────────────────────────────
# GOLD — Feature engineering → PostgreSQL
# ─────────────────────────────────────────
def gold_layer(df: pd.DataFrame) -> None:
    print("\n" + "="*55)
    print("  GOLD LAYER — Feature Engineering → PostgreSQL")
    print("="*55)

    df = df.copy()

    # 1. Feature: debt-to-income risk band
    if "dti" in df.columns:
        df["dti_risk"] = pd.cut(
            df["dti"],
            bins=[-1, 10, 20, 30, float("inf")],
            labels=["low", "medium", "high", "very_high"]
        ).astype(str)

    # 2. Feature: annual income band
    if "annual_inc" in df.columns:
        df["income_band"] = pd.cut(
            df["annual_inc"],
            bins=[0, 40000, 80000, 120000, float("inf")],
            labels=["low", "medium", "high", "very_high"]
        ).astype(str)

    # 3. Feature: loan to income ratio
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"].replace(0, 1)).round(4)

    # 4. Feature: FICO midpoint
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_mid"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).round(0)

    # 5. Fill remaining nulls
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("unknown")

    print(f"  Features engineered: dti_risk, income_band, loan_to_income, fico_mid")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Total rows   : {len(df):,}")

    # 6. Write to PostgreSQL
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )
    cur = conn.cursor()

    # Create table
    cur.execute("DROP TABLE IF EXISTS gold_features;")
    cur.execute("""
        CREATE TABLE gold_features (
            loan_amnt        FLOAT,
            funded_amnt      FLOAT,
            term             FLOAT,
            int_rate         FLOAT,
            installment      FLOAT,
            grade            TEXT,
            sub_grade        TEXT,
            emp_length       TEXT,
            home_ownership   TEXT,
            annual_inc       FLOAT,
            verification_status TEXT,
            issue_d          TEXT,
            loan_status      TEXT,
            purpose          TEXT,
            dti              FLOAT,
            delinq_2yrs      FLOAT,
            fico_range_low   FLOAT,
            fico_range_high  FLOAT,
            open_acc         FLOAT,
            pub_rec          FLOAT,
            revol_bal        FLOAT,
            revol_util       FLOAT,
            total_acc        FLOAT,
            initial_list_status TEXT,
            application_type TEXT,
            issue_year       FLOAT,
            label            INTEGER,
            dti_risk         TEXT,
            income_band      TEXT,
            loan_to_income   FLOAT,
            fico_mid         FLOAT
        );
    """)

    # Insert rows in batches
    cols = [c for c in df.columns if c in [
        "loan_amnt","funded_amnt","term","int_rate","installment",
        "grade","sub_grade","emp_length","home_ownership","annual_inc",
        "verification_status","issue_d","loan_status","purpose",
        "dti","delinq_2yrs","fico_range_low","fico_range_high",
        "open_acc","pub_rec","revol_bal","revol_util","total_acc",
        "initial_list_status","application_type","issue_year","label",
        "dti_risk","income_band","loan_to_income","fico_mid"
    ]]

    df_insert = df[cols].copy()
    rows = [tuple(row) for row in df_insert.itertuples(index=False)]

    insert_sql = f"INSERT INTO gold_features ({', '.join(cols)}) VALUES %s"
    execute_values(cur, insert_sql, rows, page_size=500)

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Gold: inserted {len(rows):,} rows into PostgreSQL table 'gold_features'")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Spark Batch Job — Medallion Pipeline")
    print("=" * 55)

    # Bronze
    df_bronze = bronze_layer()

    # Silver
    df_silver = silver_layer(df_bronze)

    # Gold
    gold_layer(df_silver)

    print("\n" + "="*55)
    print("  ✅ Pipeline Complete!")
    print("  Bronze → MinIO")
    print("  Silver → Validated & cleaned")
    print("  Gold   → PostgreSQL (gold_features)")
    print("="*55)


if __name__ == "__main__":
    main()