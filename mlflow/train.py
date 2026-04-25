"""
MLflow Training Pipeline — Credit Scoring Model
Reads gold_features from PostgreSQL, trains RandomForestClassifier,
logs metrics to MLflow, registers model to MLflow Model Registry.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME     = "credit-scoring"
MODEL_NAME          = "CreditScoringModel"

PG_HOST             = "postgres"
PG_PORT             = 5432
PG_DB               = "credit_scoring"
PG_USER             = "mlops"
PG_PASSWORD         = "mlops123"

# Features to use for training
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

TARGET = "label"

# Model hyperparameters
RF_PARAMS = {
    "n_estimators"  : 100,
    "max_depth"     : 10,
    "min_samples_split": 5,
    "min_samples_leaf" : 2,
    "random_state"  : 42,
    "n_jobs"        : -1,
}


def configure_artifact_store_env() -> None:
    """Ensure MLflow has S3-compatible settings for MinIO artifact uploads."""
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    # Fallback to MinIO credentials when AWS variables are not provided.
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ROOT_USER", "admin")
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_ROOT_PASSWORD", "password123")

    # Silence non-blocking git metadata warnings from MLflow's git integration.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


# ─────────────────────────────────────────
# Step 1 — Load data from PostgreSQL
# ─────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("\n" + "="*55)
    print("  STEP 1 — Loading gold_features from PostgreSQL")
    print("="*55)

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )

    query = f"""
        SELECT {', '.join(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])}
        FROM gold_features
        WHERE label IS NOT NULL
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"   Label distribution: {df[TARGET].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────
# Step 2 — Preprocess
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  STEP 2 — Preprocessing")
    print("="*55)

    df = df.copy()

    # Fill numeric nulls with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET]

    print(f"✅ Features ready: {len(feature_cols)} columns")
    print(f"   Numeric    : {len([c for c in NUMERIC_FEATURES if c in df.columns])}")
    print(f"   Categorical: {len([c for c in CATEGORICAL_FEATURES if c in df.columns])}")

    return X, y, feature_cols, encoders


# ─────────────────────────────────────────
# Step 3 — Train & evaluate
# ─────────────────────────────────────────
def train_and_evaluate(X, y):
    print("\n" + "="*55)
    print("  STEP 3 — Training RandomForestClassifier")
    print("="*55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train size : {len(X_train):,}")
    print(f"   Test size  : {len(X_test):,}")

    model = RandomForestClassifier(**RF_PARAMS)
    print(f"\n   Training with {RF_PARAMS['n_estimators']} trees... (this takes 2-3 min)")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred     = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "auc_roc"   : round(roc_auc_score(y_test, y_pred_proba), 4),
        "f1_score"  : round(f1_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
    }

    print(f"\n  📊 Model Performance:")
    for k, v in metrics.items():
        print(f"     {k:<12}: {v}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:\n{cm}")

    return model, metrics, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# Step 4 — Log to MLflow & register model
# ─────────────────────────────────────────
def log_to_mlflow(model, metrics, feature_cols):
    print("\n" + "="*55)
    print("  STEP 4 — Logging to MLflow")
    print("="*55)

    configure_artifact_store_env()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="random-forest-baseline") as run:
        # Log hyperparameters
        mlflow.log_params(RF_PARAMS)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("features", ", ".join(feature_cols))

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id
        print(f"✅ MLflow run logged: {run_id}")
        print(f"   Experiment : {EXPERIMENT_NAME}")
        print(f"   Model name : {MODEL_NAME}")

    # Transition model to Staging
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])

    if latest:
        version = latest[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
        )
        print(f"✅ Model v{version} transitioned to → Staging")

    print(f"\n   View at: http://localhost:5000/#/experiments")
    return run_id

def ensure_model_registry():
    """Create model registry entry if it doesn't exist."""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.create_registered_model(MODEL_NAME)
        print(f"✅ Model registry '{MODEL_NAME}' created")
    except Exception:
        print(f"✅ Model registry '{MODEL_NAME}' already exists")
# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  MLflow Training Pipeline — Credit Scoring")
    print("=" * 55)
    # 0. Ensure model registry exists   ← ADD THIS
    configure_artifact_store_env()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    ensure_model_registry()
    # 1. Load
    df = load_data()

    # 2. Preprocess
    X, y, feature_cols, encoders = preprocess(df)

    # 3. Train
    model, metrics, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)

    # 4. Log to MLflow
    run_id = log_to_mlflow(model, metrics, feature_cols)

    print("\n" + "="*55)
    print("  ✅ Training Complete!")
    print(f"  AUC-ROC  : {metrics['auc_roc']}")
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  F1 Score : {metrics['f1_score']}")
    print("="*55)


if __name__ == "__main__":
    main()