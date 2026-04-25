"""
Airflow DAG — MLOps Credit Scoring Pipeline
Runs daily at 2 AM with this task chain:
ingest_bronze >> validate_silver >> process_gold >> check_drift >> train_model
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ─────────────────────────────────────────
# Default args
# ─────────────────────────────────────────
default_args = {
    "owner"           : "mlops",
    "depends_on_past" : False,
    "start_date"      : datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry"  : False,
    "retries"         : 2,
    "retry_delay"     : timedelta(minutes=5),
}

# ─────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────
dag = DAG(
    dag_id="credit_scoring_pipeline",
    default_args=default_args,
    description="Daily MLOps pipeline: Bronze → Silver → Gold → Drift Check → Train",
    schedule_interval="0 2 * * *",   # every day at 2 AM
    catchup=False,
    tags=["mlops", "credit-scoring"],
)


# ─────────────────────────────────────────
# Task 1 — Ingest Bronze
# Read from Kafka batch topic → save raw parquet to MinIO
# ─────────────────────────────────────────
def ingest_bronze(**context):
    import json
    import os
    import pandas as pd
    from kafka import KafkaConsumer

    print("🥉 Starting Bronze ingestion from Kafka...")

    consumer = KafkaConsumer(
        "loan-applications-batch",
        bootstrap_servers="kafka:29092",
        group_id="airflow-bronze-consumer",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=30000,
    )

    records = []
    for msg in consumer:
        records.append(msg.value)

    consumer.close()

    if not records:
        raise ValueError("❌ No records in Kafka batch topic!")

    df = pd.DataFrame(records)
    os.makedirs("/tmp/bronze", exist_ok=True)
    df.to_parquet("/tmp/bronze/loan_applications.parquet", index=False)

    print(f"✅ Bronze: {len(df):,} records saved")
    return len(df)


# ─────────────────────────────────────────
# Task 2 — Validate Silver
# Clean + validate data
# ─────────────────────────────────────────
def validate_silver(**context):
    import pandas as pd

    DEFAULT_STATUSES = {
        "Charged Off", "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "Late (31-120 days)",
    }

    print("🥈 Starting Silver validation...")

    df = pd.read_parquet("/tmp/bronze/loan_applications.parquet")
    original = len(df)

    # Fix types
    numeric_cols = [
        "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
        "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%", "").str.strip(),
                errors="coerce"
            )

    if "term" in df.columns:
        df["term"] = pd.to_numeric(
            df["term"].astype(str).str.replace(" months", "").str.strip(),
            errors="coerce"
        )

    # Drop nulls
    critical = [c for c in ["loan_amnt", "annual_inc", "dti", "loan_status"] if c in df.columns]
    df = df.dropna(subset=critical)

    # Label
    if "loan_status" in df.columns:
        clear = DEFAULT_STATUSES | {"Fully Paid"}
        df = df[df["loan_status"].isin(clear)]
        df["label"] = df["loan_status"].apply(lambda s: 1 if s in DEFAULT_STATUSES else 0)

    # Parse year
    if "issue_d" in df.columns:
        df["issue_year"] = pd.to_datetime(
            df["issue_d"], format="%b-%Y", errors="coerce"
        ).dt.year.fillna(0).astype(int)

    # Validations
    assert len(df) > 100, "Too few rows after cleaning!"
    assert df["loan_amnt"].min() > 0, "Negative loan amounts found!"

    import os
    os.makedirs("/tmp/silver", exist_ok=True)
    df.to_parquet("/tmp/silver/loan_applications_clean.parquet", index=False)

    print(f"✅ Silver: {original:,} → {len(df):,} rows ({original - len(df):,} dropped)")
    return len(df)


# ─────────────────────────────────────────
# Task 3 — Process Gold
# Feature engineering → PostgreSQL
# ─────────────────────────────────────────
def process_gold(**context):
    import pandas as pd
    import psycopg2
    from psycopg2.extras import execute_values

    print("🥇 Starting Gold feature engineering...")

    df = pd.read_parquet("/tmp/silver/loan_applications_clean.parquet")

    # Engineer features
    if "dti" in df.columns:
        df["dti_risk"] = pd.cut(
            df["dti"], bins=[-1, 10, 20, 30, float("inf")],
            labels=["low", "medium", "high", "very_high"]
        ).astype(str)

    if "annual_inc" in df.columns:
        df["income_band"] = pd.cut(
            df["annual_inc"], bins=[0, 40000, 80000, 120000, float("inf")],
            labels=["low", "medium", "high", "very_high"]
        ).astype(str)

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"].replace(0, 1)).round(4)

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_mid"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).round(0)

    # Fill nulls
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("unknown")

    # Write to PostgreSQL
    conn = psycopg2.connect(
        host="postgres", port=5432, dbname="credit_scoring",
        user="mlops", password="mlops123"
    )
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE gold_features;")

    cols = [c for c in df.columns if c in [
        "loan_amnt","funded_amnt","term","int_rate","installment",
        "grade","sub_grade","emp_length","home_ownership","annual_inc",
        "verification_status","issue_d","loan_status","purpose",
        "dti","delinq_2yrs","fico_range_low","fico_range_high",
        "open_acc","pub_rec","revol_bal","revol_util","total_acc",
        "initial_list_status","application_type","issue_year","label",
        "dti_risk","income_band","loan_to_income","fico_mid"
    ]]

    rows = [tuple(row) for row in df[cols].itertuples(index=False)]
    execute_values(cur, f"INSERT INTO gold_features ({', '.join(cols)}) VALUES %s", rows, page_size=500)
    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Gold: {len(rows):,} rows inserted into PostgreSQL")
    return len(rows)


# ─────────────────────────────────────────
# Task 4 — Check Drift
# Calculate PSI and fail if drift detected
# ─────────────────────────────────────────
def check_drift(**context):
    import numpy as np
    import psycopg2

    print("🔍 Checking model drift (PSI)...")

    conn = psycopg2.connect(
        host="postgres", port=5432, dbname="credit_scoring",
        user="mlops", password="mlops123"
    )
    cur = conn.cursor()

    cur.execute("SELECT loan_amnt FROM gold_features WHERE issue_year = 2015 AND loan_amnt > 0")
    baseline = np.array([r[0] for r in cur.fetchall()], dtype=float)

    cur.execute("SELECT loan_amnt FROM streaming_predictions WHERE loan_amnt > 0")
    stream = np.array([r[0] for r in cur.fetchall()], dtype=float)

    cur.close()
    conn.close()

    if len(baseline) < 10 or len(stream) < 10:
        print("⚠️  Not enough data for PSI calculation, skipping.")
        return 0.0

    # PSI calculation
    breakpoints  = np.unique(np.percentile(baseline, np.linspace(0, 100, 11)))
    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    stream_pct   = np.histogram(stream,   bins=breakpoints)[0] / len(stream)
    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    stream_pct   = np.where(stream_pct   == 0, 1e-6, stream_pct)
    psi = float(np.sum((stream_pct - baseline_pct) * np.log(stream_pct / baseline_pct)))

    print(f"   PSI Score : {psi:.4f}")

    if psi > 0.25:
        raise ValueError(f"🚨 DRIFT DETECTED! PSI={psi:.4f} exceeds threshold 0.25. Triggering retrain.")

    print(f"✅ Drift check passed — PSI={psi:.4f} is STABLE")
    return psi


# ─────────────────────────────────────────
# Task 5 — Train Model
# Retrain if drift detected or scheduled
# ─────────────────────────────────────────
def train_model(**context):
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    import psycopg2
    import numpy as np
    from mlflow.tracking import MlflowClient
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    print("🤖 Starting model training...")

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

    conn = psycopg2.connect(
        host="postgres", port=5432, dbname="credit_scoring",
        user="mlops", password="mlops123"
    )
    cols = ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["label"])
    df = pd.read_sql(f"SELECT {cols} FROM gold_features WHERE label IS NOT NULL", conn)
    conn.close()

    # Preprocess
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if col in df else 0)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str).fillna("unknown"))

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc_roc" : round(roc_auc_score(y_test, y_pred_proba), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }

    # Log to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("credit-scoring")

    with mlflow.start_run(run_name="airflow-scheduled-retrain"):
        mlflow.log_params({"n_estimators": 100, "max_depth": 10})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )
        

    print(f"✅ Model trained — AUC: {metrics['auc_roc']} | Accuracy: {metrics['accuracy']}")
    return metrics


# ─────────────────────────────────────────
# Define tasks
# ─────────────────────────────────────────
t1_ingest_bronze = PythonOperator(
    task_id="ingest_bronze",
    python_callable=ingest_bronze,
    dag=dag,
)

t2_validate_silver = PythonOperator(
    task_id="validate_silver",
    python_callable=validate_silver,
    dag=dag,
)

t3_process_gold = PythonOperator(
    task_id="process_gold",
    python_callable=process_gold,
    dag=dag,
)

t4_check_drift = PythonOperator(
    task_id="check_drift",
    python_callable=check_drift,
    dag=dag,
)

t5_train_model = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

# ─────────────────────────────────────────
# Task chain
# ─────────────────────────────────────────
t1_ingest_bronze >> t2_validate_silver >> t3_process_gold >> t4_check_drift >> t5_train_model