"""
Drift Simulator — Injects fake data with different distribution
to trigger PSI > 0.25 and test the drift detection pipeline.

Run this script, then check:
- PSI exporter logs → should show 🚨 DRIFT DETECTED
- Grafana dashboard → PSI gauge should turn red
- Airflow → check_drift task should fail and trigger retrain
"""

import json
import random
import psycopg2
from psycopg2.extras import execute_values

PG_HOST     = "postgres"
PG_PORT     = 5432
PG_DB       = "credit_scoring"
PG_USER     = "mlops"
PG_PASSWORD = "mlops123"

N_ROWS = 2000  # number of fake drift rows to inject


def generate_drifted_rows(n: int) -> list:
    """
    Generate fake loan applications with VERY different distribution:
    - loan_amnt: much higher (50k-100k vs normal 5k-35k)
    - annual_inc: much lower (10k-30k vs normal 40k-80k)
    - dti: much higher (35-60 vs normal 10-25)
    - int_rate: much higher (25-35% vs normal 8-20%)
    This simulates a riskier population → triggers drift
    """
    rows = []
    grades = ["D", "E", "F", "G"]  # only bad grades (normally mix of A-G)

    for _ in range(n):
        loan_amnt  = random.uniform(50000, 100000)   # much higher
        annual_inc = random.uniform(10000, 30000)    # much lower
        dti        = random.uniform(35, 60)          # much higher
        int_rate   = random.uniform(25, 35)          # much higher
        fico_low   = random.uniform(580, 650)        # lower FICO
        fico_high  = fico_low + 4
        fico_mid   = (fico_low + fico_high) / 2

        loan_to_income = loan_amnt / annual_inc

        # DTI risk (all high/very_high due to high DTI)
        if dti <= 10:
            dti_risk = "low"
        elif dti <= 20:
            dti_risk = "medium"
        elif dti <= 30:
            dti_risk = "high"
        else:
            dti_risk = "very_high"

        # Income band (all low due to low income)
        if annual_inc <= 40000:
            income_band = "low"
        elif annual_inc <= 80000:
            income_band = "medium"
        elif annual_inc <= 120000:
            income_band = "high"
        else:
            income_band = "very_high"

        rows.append((
            round(loan_amnt, 2),        # loan_amnt
            round(annual_inc, 2),       # annual_inc
            round(dti, 2),              # dti
            round(int_rate, 2),         # int_rate
            round(fico_low, 0),         # fico_range_low
            round(fico_high, 0),        # fico_range_high
            round(fico_mid, 0),         # fico_mid
            round(loan_to_income, 4),   # loan_to_income
            random.choice(grades),      # grade
            dti_risk,                   # dti_risk
            income_band,                # income_band
            1,                          # prediction (all defaults)
            round(random.uniform(0.7, 0.95), 4),  # probability (high default prob)
        ))

    return rows


def inject_drift():
    print("=" * 55)
    print("  Drift Simulator")
    print(f"  Injecting {N_ROWS} drifted rows into streaming_predictions")
    print("=" * 55)

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )
    cur = conn.cursor()

    # Check current count
    cur.execute("SELECT COUNT(*) FROM streaming_predictions")
    before = cur.fetchone()[0]
    print(f"\n  Before injection: {before:,} rows")

    # Generate drifted rows
    rows = generate_drifted_rows(N_ROWS)

    # Insert into streaming_predictions
    execute_values(cur, """
        INSERT INTO streaming_predictions
            (loan_amnt, annual_inc, dti, int_rate,
             fico_range_low, fico_range_high, fico_mid,
             loan_to_income, grade, dti_risk, income_band,
             prediction, probability)
        VALUES %s
    """, rows)

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM streaming_predictions")
    after = cur.fetchone()[0]
    print(f"  After injection  : {after:,} rows")
    print(f"  Rows added       : {after - before:,}")

    cur.close()
    conn.close()

    print(f"""
✅ Drift data injected successfully!

Now watch the PSI exporter — next cycle it should show:
  PSI Score > 0.25 → 🚨 DRIFT DETECTED

Also check:
  Grafana  → http://localhost:3000 (PSI gauge turns red)
  Airflow  → manually trigger DAG to test check_drift task
    """)


if __name__ == "__main__":
    inject_drift()