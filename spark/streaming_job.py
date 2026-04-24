"""
Spark Streaming Job — Real-time Credit Scoring Predictions
Reads from Kafka stream topic every 10 seconds,
loads MLflow model, runs predictions,
writes results to PostgreSQL streaming_predictions table.
"""

import os
import json
import time
import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from kafka import KafkaConsumer
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
KAFKA_BOOTSTRAP     = "kafka:29092"
STREAM_TOPIC        = "loan-applications-stream"
KAFKA_GROUP_ID      = "spark-stream-consumer"
POLL_TIMEOUT_MS     = 10000   # wait 10s for messages per poll
BATCH_SIZE          = 100     # process 100 messages at a time

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME          = "CreditScoringModel"
MODEL_STAGE         = "Staging"

PG_HOST             = "postgres"
PG_PORT             = 5432
PG_DB               = "credit_scoring"
PG_USER             = "mlops"
PG_PASSWORD         = "mlops123"

NUMERIC_FEATURES = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "loan_to_income", "fico_mid",
]

CATEGORICAL_FEATURES = [
    "grade", "home_ownership", "verification_status",
    "purpose", "dti_risk", "income_band", "application_type",
]


def configure_artifact_store_env() -> None:
    """Ensure MLflow S3 artifact downloads from MinIO have credentials."""
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ROOT_USER", "admin")
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_ROOT_PASSWORD", "password123")

    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


# ─────────────────────────────────────────
# Step 1 — Setup PostgreSQL table
# ─────────────────────────────────────────
def setup_predictions_table():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS streaming_predictions (
            id               SERIAL PRIMARY KEY,
            prediction       INTEGER,
            probability      FLOAT,
            loan_amnt        FLOAT,
            annual_inc       FLOAT,
            dti              FLOAT,
            grade            TEXT,
            int_rate         FLOAT,
            fico_mid         FLOAT,
            loan_to_income   FLOAT,
            processed_at     TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ streaming_predictions table ready")


# ─────────────────────────────────────────
# Step 2 — Load MLflow model
# ─────────────────────────────────────────
def load_model():
    print(f"\n📦 Loading model '{MODEL_NAME}' from MLflow (stage: {MODEL_STAGE})...")
    configure_artifact_store_env()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Model loaded successfully")
    return model


# ─────────────────────────────────────────
# Step 3 — Preprocess a batch of messages
# ─────────────────────────────────────────
def preprocess_batch(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # Engineer features (same as batch job Gold layer)
    if "dti" in df.columns:
        df["dti"] = pd.to_numeric(df["dti"], errors="coerce").fillna(0)
        df["dti_risk"] = pd.cut(
            df["dti"],
            bins=[-1, 10, 20, 30, float("inf")],
            labels=["low", "medium", "high", "very_high"]
        ).astype(str)

    if "annual_inc" in df.columns:
        df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce").fillna(0)
        df["income_band"] = pd.cut(
            df["annual_inc"],
            bins=[0, 40000, 80000, 120000, float("inf")],
            labels=["low", "medium", "high", "very_high"]
        ).astype(str)

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0)
        df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"].replace(0, 1)).round(4)

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_range_low"]  = pd.to_numeric(df["fico_range_low"],  errors="coerce").fillna(0)
        df["fico_range_high"] = pd.to_numeric(df["fico_range_high"], errors="coerce").fillna(0)
        df["fico_mid"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).round(0)

    # Fix numeric features
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    # Fix categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")
        else:
            df[col] = "unknown"
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[feature_cols]


# ─────────────────────────────────────────
# Step 4 — Save predictions to PostgreSQL
# ─────────────────────────────────────────
def save_predictions(records: list, predictions: list, probabilities: list):
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )
    cur = conn.cursor()

    rows = []
    for record, pred, prob in zip(records, predictions, probabilities):
        rows.append((
            int(pred),
            float(prob),
            float(record.get("loan_amnt", 0) or 0),
            float(record.get("annual_inc", 0) or 0),
            float(record.get("dti", 0) or 0),
            str(record.get("grade", "unknown")),
            float(record.get("int_rate", 0) or 0),
            float(record.get("fico_range_low", 0) or 0),
            float(record.get("loan_amnt", 0) or 0) / max(float(record.get("annual_inc", 1) or 1), 1),
        ))

    execute_values(cur, """
        INSERT INTO streaming_predictions
            (prediction, probability, loan_amnt, annual_inc, dti,
             grade, int_rate, fico_mid, loan_to_income)
        VALUES %s
    """, rows)

    conn.commit()
    cur.close()
    conn.close()


# ─────────────────────────────────────────
# Main streaming loop
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Spark Streaming — Real-time Credit Scoring")
    print("=" * 55)

    # Setup
    setup_predictions_table()
    model = load_model()

    # Connect to Kafka
    consumer = KafkaConsumer(
        STREAM_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=POLL_TIMEOUT_MS,
    )

    print(f"\n🌊 Listening on topic: {STREAM_TOPIC}")
    print(f"   Processing in batches of {BATCH_SIZE} messages")
    print(f"   Press Ctrl+C to stop\n")

    total_processed = 0
    batch = []

    try:
        for msg in consumer:
            batch.append(msg.value)

            if len(batch) >= BATCH_SIZE:
                # Preprocess
                X = preprocess_batch(batch)

                # Predict
                predictions   = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]

                # Save
                save_predictions(batch, predictions, probabilities)

                total_processed += len(batch)
                defaults = sum(predictions)
                print(f"  ✅ Batch processed: {len(batch)} records | "
                      f"defaults: {defaults} | "
                      f"total: {total_processed:,}")

                batch = []

        # Process remaining messages
        if batch:
            X = preprocess_batch(batch)
            predictions   = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
            save_predictions(batch, predictions, probabilities)
            total_processed += len(batch)
            print(f"  ✅ Final batch: {len(batch)} records processed")

    except KeyboardInterrupt:
        print(f"\n⛔ Stopped by user")

    finally:
        consumer.close()
        print(f"\n🏁 Streaming complete — {total_processed:,} total predictions saved")
        print(f"   Check PostgreSQL: SELECT COUNT(*) FROM streaming_predictions;")


if __name__ == "__main__":
    main()