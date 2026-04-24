"""
PSI Drift Exporter
Compares 2015 baseline (gold_features) vs 2018 stream predictions.
Calculates PSI + KL Divergence every 5 minutes.
Exposes metrics on port 8000 for Prometheus to scrape.
"""

import time
import numpy as np
import psycopg2
from prometheus_client import start_http_server, Gauge

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
PG_HOST         = "postgres"
PG_PORT         = 5432
PG_DB           = "credit_scoring"
PG_USER         = "mlops"
PG_PASSWORD     = "mlops123"

EXPORTER_PORT   = 8000
SCRAPE_INTERVAL = 300   # seconds (5 minutes)
N_BINS          = 10

PSI_THRESHOLD   = 0.25  # > 0.25 = significant drift
KL_THRESHOLD    = 0.50

# ─────────────────────────────────────────
# Prometheus Gauges
# ─────────────────────────────────────────
psi_gauge = Gauge("model_psi_score",      "Population Stability Index score")
kl_gauge  = Gauge("model_kl_divergence",  "KL Divergence between baseline and stream")
drift_gauge = Gauge("model_drift_detected", "1 if drift detected (PSI > 0.25), 0 otherwise")
baseline_count_gauge = Gauge("baseline_record_count", "Number of baseline records")
stream_count_gauge   = Gauge("stream_record_count",   "Number of stream records")


# ─────────────────────────────────────────
# DB helper
# ─────────────────────────────────────────
def get_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        dbname="credit_scoring",
        user="mlops",
        password="mlops123"
    )


# ─────────────────────────────────────────
# Load baseline (2015 gold_features)
# ─────────────────────────────────────────
def load_baseline() -> np.ndarray:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT loan_amnt
        FROM gold_features
        WHERE issue_year = 2015
          AND loan_amnt IS NOT NULL
          AND loan_amnt > 0
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return np.array([r[0] for r in rows], dtype=float)


# ─────────────────────────────────────────
# Load stream (streaming_predictions)
# ─────────────────────────────────────────
def load_stream() -> np.ndarray:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT loan_amnt
        FROM streaming_predictions
        WHERE loan_amnt IS NOT NULL
          AND loan_amnt > 0
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return np.array([r[0] for r in rows], dtype=float)


# ─────────────────────────────────────────
# Calculate PSI
# ─────────────────────────────────────────
def calculate_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    PSI < 0.10 : no drift
    PSI 0.10-0.25 : moderate drift
    PSI > 0.25 : significant drift
    """
    # Create bins based on baseline distribution
    breakpoints = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    breakpoints  = np.unique(breakpoints)  # remove duplicates

    # Count observations per bin
    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts  = np.histogram(current,  bins=breakpoints)[0]

    # Convert to percentages, avoid division by zero
    baseline_pct = baseline_counts / len(baseline)
    current_pct  = current_counts  / len(current)

    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    current_pct  = np.where(current_pct  == 0, 1e-6, current_pct)

    # PSI formula
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return round(float(psi), 6)


# ─────────────────────────────────────────
# Calculate KL Divergence
# ─────────────────────────────────────────
def calculate_kl_divergence(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    breakpoints    = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    breakpoints    = np.unique(breakpoints)

    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts  = np.histogram(current,  bins=breakpoints)[0]

    baseline_pct = baseline_counts / len(baseline)
    current_pct  = current_counts  / len(current)

    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    current_pct  = np.where(current_pct  == 0, 1e-6, current_pct)

    kl = np.sum(baseline_pct * np.log(baseline_pct / current_pct))
    return round(float(kl), 6)


# ─────────────────────────────────────────
# Main monitoring loop
# ─────────────────────────────────────────
def monitor():
    print("=" * 55)
    print("  PSI Drift Exporter")
    print(f"  Prometheus metrics on port {EXPORTER_PORT}")
    print(f"  Scrape interval: {SCRAPE_INTERVAL}s")
    print("=" * 55)

    start_http_server(EXPORTER_PORT)
    print(f"✅ Metrics server started at http://localhost:{EXPORTER_PORT}/metrics\n")

    while True:
        try:
            print(f"🔍 Calculating drift metrics...")

            baseline = load_baseline()
            stream   = load_stream()

            print(f"   Baseline records : {len(baseline):,}")
            print(f"   Stream records   : {len(stream):,}")

            if len(baseline) < 10:
                print("⚠️  Not enough baseline data. Skipping.")
                time.sleep(SCRAPE_INTERVAL)
                continue

            if len(stream) < 10:
                print("⚠️  Not enough stream data. Skipping.")
                time.sleep(SCRAPE_INTERVAL)
                continue

            psi = calculate_psi(baseline, stream, N_BINS)
            kl  = calculate_kl_divergence(baseline, stream, N_BINS)

            # Update Prometheus gauges
            psi_gauge.set(psi)
            kl_gauge.set(kl)
            drift_gauge.set(1 if psi > PSI_THRESHOLD else 0)
            baseline_count_gauge.set(len(baseline))
            stream_count_gauge.set(len(stream))

            # Print status
            drift_status = "🚨 DRIFT DETECTED" if psi > PSI_THRESHOLD else "✅ STABLE"
            print(f"\n  PSI Score     : {psi}  (threshold: {PSI_THRESHOLD})")
            print(f"  KL Divergence : {kl}  (threshold: {KL_THRESHOLD})")
            print(f"  Status        : {drift_status}")
            print(f"  Next check in : {SCRAPE_INTERVAL}s\n")

        except Exception as e:
            print(f"❌ Error during monitoring: {e}")

        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    monitor()