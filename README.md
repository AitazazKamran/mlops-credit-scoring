# 🏦 MLOps Credit Scoring Pipeline

A production-grade MLOps pipeline for real-time credit scoring with automated drift detection, model retraining, and full observability.

---

## 📐 Architecture

```
Raw Data (Lending Club CSV)
         │
         ▼
   Kafka Producer
    ┌────┴────┐
    │         │
    ▼         ▼
 Batch      Stream
 Topic      Topic
 (2015)     (2018)
    │         │
    ▼         │
Spark Batch   │
Pipeline      │
 Bronze       │
 Silver       │
 Gold ──► PostgreSQL
    │         │
    ▼         ▼
MLflow    Spark Streaming
Training  (Live Predictions)
    │         │
    ▼         ▼
 Model     PostgreSQL
Registry  (streaming_predictions)
              │
              ▼
         PSI Exporter
              │
              ▼
         Prometheus
              │
              ▼
          Grafana
         Dashboard
              │
         (Drift Alert)
              │
              ▼
         Airflow DAG
      (Daily Retrain)
```

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Data Streaming | Apache Kafka |
| Batch Processing | PySpark (Medallion Architecture) |
| Feature Store | PostgreSQL |
| Object Storage | MinIO (S3-compatible) |
| ML Training | scikit-learn (RandomForest) |
| Experiment Tracking | MLflow |
| Drift Detection | PSI + KL Divergence |
| Metrics | Prometheus |
| Dashboards | Grafana |
| Orchestration | Apache Airflow |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (16GB RAM allocated, 4 CPUs)
- Git
- Python 3.10+
- WSL2 enabled (Windows only)

### 1. Clone the repository

```bash
git clone https://github.com/AitazazKamran/mlops-credit-scoring.git
cd mlops-credit-scoring
```

### 2. Download the dataset

Download the Lending Club dataset from Kaggle:
- [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Download `accepted_2007_to_2018Q4.csv.gz`
- Place it at: `data/raw/accepted_2007_to_2018Q4.csv.gz`

### 3. Create environment file

```bash
cp .env.example .env
```

Or create `.env` manually:

```env
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops123
POSTGRES_DB=credit_scoring
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password123
GRAFANA_ADMIN_PASSWORD=admin
```

### 4. Build custom Docker images

```bash
docker build -f Dockerfile.spark -t mlops-spark:latest .
docker build -f Dockerfile.airflow -t mlops-airflow:latest .
```

### 5. Start all services

```bash
docker compose up -d
```

Wait 90 seconds for all services to initialize.

### 6. Verify all services are running

| Service | URL | Login |
|---|---|---|
| Kafka UI | http://localhost:8080 | — |
| MinIO | http://localhost:9001 | admin / password123 |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8081 | airflow / airflow |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

### 7. Create MinIO bucket

- Open http://localhost:9001
- Login → Create Bucket → name: `mlflow-artifacts`

### 8. Create Kafka topics

```bash
docker exec -it kafka kafka-topics --create --bootstrap-server localhost:9092 --topic loan-applications-batch --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics --create --bootstrap-server localhost:9092 --topic loan-applications-stream --partitions 3 --replication-factor 1
```

### 9. Install Python dependencies

```bash
cd kafka
pip install kafka-python-ng pandas
cd ..
```

### 10. Run the Kafka producer

```bash
python kafka/producer.py
```

This sends 2015 data to the batch topic and 5,000 rows of 2018 data to the stream topic.

### 11. Run the Spark batch pipeline

```bash
docker exec -it spark bash -c "python3 /opt/spark-jobs/batch_job.py"
```

This runs the full Bronze → Silver → Gold medallion pipeline (~5 minutes).

### 12. Train the ML model

```bash
docker exec -it spark bash -c "python3 /opt/mlflow-jobs/train.py"
```

View results at http://localhost:5000

### 13. Run Spark streaming

```bash
docker exec -it spark bash -c "python3 /opt/spark-jobs/streaming_job.py"
```

### 14. Fix Prometheus scrape config

Get Spark container IP:
```bash
docker inspect spark --format "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"
```

Update `monitoring/prometheus.yml` with that IP:
```yaml
- job_name: "psi-exporter"
  static_configs:
    - targets: ["YOUR_SPARK_IP:8000"]
```

Then restart Prometheus:
```bash
docker compose restart prometheus
```

> **Note:** If you recreate the Spark container, its IP may change. Repeat this step if Grafana shows "No data".

### 15. Start PSI drift monitoring

```bash
docker exec -it spark bash -c "python3 /opt/monitoring/psi_exporter.py"
```

### 16. Import Grafana dashboard

- Open http://localhost:3000
- Dashboards → New → Import
- Upload `monitoring/grafana_dashboard.json`
- Select Prometheus as data source → Import

---

## 📊 Pipeline Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project structure setup | ✅ |
| Phase 2 | Docker stack (9 services) | ✅ |
| Phase 3 | Kafka producer (batch + stream) | ✅ |
| Phase 4 | Spark medallion pipeline (Bronze/Silver/Gold) | ✅ |
| Phase 7 | MLflow model training + registry | ✅ |
| Phase 5 | Spark streaming predictions | ✅ |
| Phase 8 | PSI drift exporter (Prometheus) | ✅ |
| Phase 9 | Grafana dashboard | ✅ |
| Phase 10 | Airflow DAG orchestration | ✅ |
| Phase 11 | CI/CD + Dockerfiles | ✅ |

---

## 🗂️ Project Structure

```
mlops-credit-scoring/
├── airflow/
│   └── dags/
│       └── medallion_dag.py      # Airflow DAG (daily pipeline)
├── data/
│   └── raw/                      # Place dataset here (not in git)
├── kafka/
│   ├── producer.py               # Sends data to Kafka topics
│   └── requirements.txt
├── mlflow/
│   └── train.py                  # Model training + MLflow logging
├── monitoring/
│   ├── psi_exporter.py           # PSI drift exporter (Prometheus)
│   ├── prometheus.yml            # Prometheus scrape config
│   └── grafana_dashboard.json    # Grafana dashboard (importable)
├── spark/
│   ├── batch_job.py              # Bronze → Silver → Gold pipeline
│   └── streaming_job.py          # Real-time predictions
├── tests/
│   ├── test_pipeline.py          # Unit tests (CI/CD)
│   └── drift_simulator.py        # Injects fake drift data for demo
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions pipeline
├── Dockerfile.airflow            # Custom Airflow image
├── Dockerfile.spark              # Custom Spark image
├── docker-compose.yml            # All 9 services
├── .env                          # Secrets (not in git)
└── .env.example                  # Template for .env
```

---

## 🔍 Drift Detection

The PSI (Population Stability Index) exporter compares:
- **Baseline**: 2015 loan applications (training distribution)
- **Current**: 2018 streaming predictions

| PSI Value | Status |
|---|---|
| < 0.10 | ✅ No drift |
| 0.10 – 0.25 | ⚠️ Moderate drift |
| > 0.25 | 🚨 Significant drift — retrain triggered |

### Testing Drift Detection

Use the drift simulator to inject fake high-risk loan data and trigger drift:

```bash
# Inject drifted data (run 3-4 times to push PSI above 0.25)
docker exec -it spark bash -c "python3 /opt/tests/drift_simulator.py"

# Watch PSI exporter detect drift
docker exec -it spark bash -c "python3 /opt/monitoring/psi_exporter.py"
```

You should see PSI cross 0.25 and show `🚨 DRIFT DETECTED`.

Reset drift data after demo:
```bash
docker exec -it postgres psql -U mlops -d credit_scoring -c "DELETE FROM streaming_predictions WHERE probability > 0.7;"
```

---

## 🤖 Airflow DAG

The DAG `credit_scoring_pipeline` runs daily at 2 AM:

```
ingest_bronze → validate_silver → process_gold → check_drift → train_model
```

- **Retries**: 2 per task
- **Retry delay**: 5 minutes
- **Drift threshold**: PSI > 0.25 triggers retrain

---

## 🧪 Running Tests

```bash
pip install pytest pandas numpy scikit-learn
pytest tests/ -v
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Accuracy | 80.3% |
| AUC-ROC | 74.5% |
| F1 Score | 18.3% |

> Note: Low F1 is due to class imbalance (80% fully paid vs 20% default). This can be improved with SMOTE or class weighting.

---

## 🛑 Stopping the Pipeline

```bash
docker compose down
```

Data is preserved in Docker volumes. Restart anytime with `docker compose up -d`.

---

## 🎬 Demo Checklist

```
1. docker compose up -d
2. Get Spark IP and update prometheus.yml
3. docker exec -it spark bash -c "python3 /opt/monitoring/psi_exporter.py"
4. Open Grafana → show STABLE dashboard
5. Run drift simulator 3-4 times
6. Watch PSI turn red → DRIFT DETECTED
7. Trigger Airflow DAG → show automatic retrain
8. Open MLflow → show new training run logged
```

---

## 📝 License

MIT License — feel free to use this project as a template for your own MLOps pipelines.
