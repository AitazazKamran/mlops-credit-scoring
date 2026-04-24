"""
Kafka Producer — Credit Scoring Pipeline
Reads accepted_2007_to_2018Q4.csv.gz and sends:
  - 2015 data → loan-applications-batch  (training baseline)
  - 2018 data → loan-applications-stream (simulates live drift)
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
BOOTSTRAP_SERVERS = "localhost:9092"
BATCH_TOPIC       = "loan-applications-batch"
STREAM_TOPIC      = "loan-applications-stream"
DATA_PATH         = "../data/raw/accepted_2007_to_2018Q4.csv.gz"
STREAM_DELAY      = 0.0   # seconds between stream messages
PRINT_EVERY       = 100   # print progress every N rows

# Columns we actually need (keeps memory low on 1.8M row file)
USECOLS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "initial_list_status", "application_type",
]


# ─────────────────────────────────────────
# Helper: serialise row to JSON bytes
# ─────────────────────────────────────────
def row_to_bytes(row: dict) -> bytes:
    return json.dumps(row, default=str).encode("utf-8")


# ─────────────────────────────────────────
# Helper: connect with retry
# ─────────────────────────────────────────
def connect_producer(retries: int = 5, delay: int = 5) -> KafkaProducer:
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=row_to_bytes,
                acks="all",
                retries=3,
            )
            print(f"✅ Connected to Kafka at {BOOTSTRAP_SERVERS}")
            return producer
        except NoBrokersAvailable:
            print(f"⚠️  Kafka not ready (attempt {attempt}/{retries}). Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("❌ Could not connect to Kafka after multiple attempts.")


# ─────────────────────────────────────────
# Send batch (2015 data → baseline topic)
# ─────────────────────────────────────────
def send_batch(producer: KafkaProducer, df: pd.DataFrame) -> None:
    df_2015 = df[df["issue_year"] == 2015].copy()
    total = len(df_2015)
    print(f"\n📦 Sending {total:,} rows (2015) → topic: {BATCH_TOPIC}")

    for i, (_, row) in enumerate(df_2015.iterrows(), 1):
        producer.send(BATCH_TOPIC, value=row.to_dict())
        if i % PRINT_EVERY == 0:
            print(f"  batch: {i:,}/{total:,} rows sent")

    producer.flush()
    print(f"✅ Batch complete — {total:,} rows sent to {BATCH_TOPIC}\n")


# ─────────────────────────────────────────
# Send stream (2018 data → stream topic)
# ─────────────────────────────────────────
def send_stream(producer: KafkaProducer, df: pd.DataFrame) -> None:
  #  df_2018 = df[df["issue_year"] == 2018].copy()
    df_2018 = df[df["issue_year"] == 2018].head(5000).copy()

    total = len(df_2018)
    print(f"🌊 Streaming {total:,} rows (2018) → topic: {STREAM_TOPIC}")
    print(f"   Delay: {STREAM_DELAY}s per row  |  ETA: ~{total * STREAM_DELAY / 60:.1f} min\n")

    for i, (_, row) in enumerate(df_2018.iterrows(), 1):
        producer.send(STREAM_TOPIC, value=row.to_dict())
        if i % PRINT_EVERY == 0:
            print(f"  stream: {i:,}/{total:,} rows sent")
        time.sleep(STREAM_DELAY)

    producer.flush()
    print(f"✅ Stream complete — {total:,} rows sent to {STREAM_TOPIC}\n")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Kafka Producer — Credit Scoring Pipeline")
    print("=" * 55)

    # 1. Load dataset (only needed columns, low_memory safe)
    print(f"\n📂 Loading dataset from: {DATA_PATH}")
    print("   (this may take 1-2 minutes for the full 1.8M row file...)\n")

    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda c: c in USECOLS,   # load only what we need
        low_memory=False,
        compression="gzip",
    )

    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # 2. Parse issue year from issue_d column (format: "Jan-2015")
    df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year
    df = df.dropna(subset=["issue_year"])
    df["issue_year"] = df["issue_year"].astype(int)

    year_counts = df["issue_year"].value_counts().sort_index()
    print(f"\n📅 Rows by year:\n{year_counts.to_string()}\n")

    # 3. Connect to Kafka
    producer = connect_producer()

    # 4. Send 2015 data to batch topic (training baseline)
    send_batch(producer, df)

    # 5. Send 2018 data to stream topic (simulates drift)
    send_stream(producer, df)

    producer.close()
    print("🏁 Producer finished. Check Kafka UI at http://localhost:8080")


if __name__ == "__main__":
    main()