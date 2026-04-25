"""
FastAPI Backend — MLOps Credit Scoring Demo
Exposes endpoints to:
- GET  /metrics      → current PSI, KL, stream count
- POST /inject-drift → injects fake drift data
- POST /reset-drift  → removes fake drift data
- POST /trigger-dag  → triggers Airflow DAG
- GET  /health       → health check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import psycopg2
import numpy as np
import random
import requests
from psycopg2.extras import execute_values

app = FastAPI(title="MLOps Credit Scoring Demo API")

# Allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
PG_HOST     = "postgres"
PG_PORT     = 5432
PG_DB       = "credit_scoring"
PG_USER     = "mlops"
PG_PASSWORD = "mlops123"

AIRFLOW_URL  = "http://airflow-webserver:8080"
AIRFLOW_USER = "airflow"
AIRFLOW_PASS = "airflow"
DAG_ID       = "credit_scoring_pipeline"

N_BINS = 10


# ─────────────────────────────────────────
# DB helper
# ─────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )


# ─────────────────────────────────────────
# PSI calculation
# ─────────────────────────────────────────
def calculate_psi(baseline, current):
    breakpoints  = np.unique(np.percentile(baseline, np.linspace(0, 100, N_BINS + 1)))
    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    current_pct  = np.histogram(current,  bins=breakpoints)[0] / len(current)
    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    current_pct  = np.where(current_pct  == 0, 1e-6, current_pct)
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def calculate_kl(baseline, current):
    breakpoints  = np.unique(np.percentile(baseline, np.linspace(0, 100, N_BINS + 1)))
    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    current_pct  = np.histogram(current,  bins=breakpoints)[0] / len(current)
    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    current_pct  = np.where(current_pct  == 0, 1e-6, current_pct)
    return float(np.sum(baseline_pct * np.log(baseline_pct / current_pct)))


# ─────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────
@app.get("/metrics")
def get_metrics():
    try:
        conn = get_conn()
        cur  = conn.cursor()

        cur.execute("SELECT loan_amnt FROM gold_features WHERE issue_year = 2015 AND loan_amnt > 0")
        baseline = np.array([r[0] for r in cur.fetchall()], dtype=float)

        cur.execute("SELECT loan_amnt FROM streaming_predictions WHERE loan_amnt > 0")
        stream = np.array([r[0] for r in cur.fetchall()], dtype=float)

        cur.execute("SELECT COUNT(*) FROM streaming_predictions")
        stream_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM gold_features WHERE issue_year = 2015")
        baseline_count = cur.fetchone()[0]

        cur.close()
        conn.close()

        if len(baseline) < 10 or len(stream) < 10:
            return {
                "psi": 0.0, "kl": 0.0,
                "stream_count": int(stream_count),
                "baseline_count": int(baseline_count),
                "status": "insufficient_data"
            }

        psi = round(calculate_psi(baseline, stream), 6)
        kl  = round(calculate_kl(baseline, stream), 6)

        if psi > 0.25:
            status = "drift"
        elif psi > 0.10:
            status = "moderate"
        else:
            status = "stable"

        return {
            "psi"            : psi,
            "kl"             : kl,
            "stream_count"   : int(stream_count),
            "baseline_count" : int(baseline_count),
            "status"         : status,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────
# POST /inject-drift
# ─────────────────────────────────────────
@app.post("/inject-drift")
def inject_drift(rows: int = 2000):
    try:
        conn = get_conn()
        cur  = conn.cursor()

        grades = ["D", "E", "F", "G"]
        data = []
        for _ in range(rows):
            loan_amnt  = random.uniform(50000, 100000)
            annual_inc = random.uniform(10000, 30000)
            dti        = random.uniform(35, 60)
            int_rate   = random.uniform(25, 35)
            fico_low   = random.uniform(580, 650)
            fico_high  = fico_low + 4
            prob       = random.uniform(0.7, 0.95)

            data.append((
                round(loan_amnt, 2),
                round(annual_inc, 2),
                round(dti, 2),
                round(int_rate, 2),
                round(fico_low, 0),
                round(fico_high, 0),
                round((fico_low + fico_high) / 2, 0),
                round(loan_amnt / annual_inc, 4),
                random.choice(grades),
                "very_high",
                "low",
                1,
                round(prob, 4),
            ))

        execute_values(cur, """
            INSERT INTO streaming_predictions
                (loan_amnt, annual_inc, dti, int_rate,
                 fico_range_low, fico_range_high, fico_mid,
                 loan_to_income, grade, dti_risk, income_band,
                 prediction, probability)
            VALUES %s
        """, data)

        conn.commit()
        cur.close()
        conn.close()

        return {"success": True, "rows_injected": rows, "message": f"{rows} drifted rows injected"}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────
# POST /reset-drift
# ─────────────────────────────────────────
@app.post("/reset-drift")
def reset_drift():
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("DELETE FROM streaming_predictions WHERE probability > 0.7")
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "rows_deleted": deleted, "message": "Drift data removed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────
# POST /trigger-dag
# ─────────────────────────────────────────
@app.post("/trigger-dag")
def trigger_dag():
    try:
        response = requests.post(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns",
            json={"conf": {}},
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=10,
        )
        if response.status_code in [200, 201]:
            return {"success": True, "message": "DAG triggered successfully", "run_id": response.json().get("dag_run_id")}
        else:
            return {"success": False, "message": f"Airflow returned {response.status_code}", "detail": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────
# GET / — Demo dashboard HTML
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLOps Credit Scoring — Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 14px; margin-bottom: 2rem; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1.5rem; }
  .metric { background: #1e2433; border-radius: 10px; padding: 16px 20px; border: 1px solid #2d3748; }
  .metric-label { font-size: 12px; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 26px; font-weight: 600; }
  .metric-value.stable { color: #48bb78; }
  .metric-value.moderate { color: #ed8936; }
  .metric-value.drift { color: #fc8181; }
  .psi-wrap { background: #1e2433; border-radius: 10px; padding: 20px; margin-bottom: 1.5rem; border: 1px solid #2d3748; }
  .psi-header { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 14px; color: #94a3b8; }
  .psi-track { height: 12px; background: #2d3748; border-radius: 6px; overflow: hidden; margin-bottom: 8px; }
  .psi-fill { height: 100%; border-radius: 6px; transition: width 0.8s ease, background 0.4s; }
  .psi-markers { display: flex; position: relative; height: 18px; font-size: 11px; color: #64748b; }
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 1.5rem; }
  .panel { background: #1e2433; border-radius: 10px; padding: 16px 20px; border: 1px solid #2d3748; }
  .panel-title { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
  .mini-chart { display: flex; align-items: flex-end; gap: 5px; height: 60px; }
  .bar { flex: 1; border-radius: 3px 3px 0 0; transition: height 0.5s, background 0.4s; min-height: 3px; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1rem; }
  .btn { padding: 10px 22px; border-radius: 8px; border: none; font-size: 14px; font-weight: 500; cursor: pointer; transition: opacity 0.15s, transform 0.1s; }
  .btn:active { transform: scale(0.97); }
  .btn-danger { background: #e53e3e; color: white; }
  .btn-reset  { background: #2d3748; color: #e2e8f0; }
  .btn-dag    { background: #3182ce; color: white; }
  .btn:hover  { opacity: 0.85; }
  .log { background: #0d1117; border-radius: 8px; padding: 12px 16px; font-size: 12px; font-family: 'SF Mono', monospace; color: #94a3b8; max-height: 140px; overflow-y: auto; border: 1px solid #2d3748; }
  .log-line { margin-bottom: 4px; line-height: 1.5; }
  .log-alert { color: #fc8181; }
  .log-ok    { color: #68d391; }
  .log-info  { color: #63b3ed; }
  .status-pill { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; }
  .pill-stable   { background: #1a3a2a; color: #68d391; }
  .pill-moderate { background: #3a2a1a; color: #f6ad55; }
  .pill-drift    { background: #3a1a1a; color: #fc8181; }
</style>
</head>
<body>

<h1>MLOps Credit Scoring Pipeline</h1>
<p class="subtitle">Live drift detection demo — inject data and watch metrics update in real time</p>

<div class="metrics">
  <div class="metric">
    <div class="metric-label">PSI Score</div>
    <div class="metric-value stable" id="psi-val">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">KL Divergence</div>
    <div class="metric-value stable" id="kl-val">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">Stream Records</div>
    <div class="metric-value" id="stream-val">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">Status</div>
    <div id="status-pill" class="status-pill pill-stable">Loading...</div>
  </div>
</div>

<div class="psi-wrap">
  <div class="psi-header">
    <span>PSI Score (threshold: 0.25)</span>
    <span id="psi-pct">—</span>
  </div>
  <div class="psi-track">
    <div class="psi-fill" id="psi-fill" style="width:0%; background:#48bb78;"></div>
  </div>
  <div class="psi-markers">
    <span style="position:absolute;left:20%;">0.10 moderate</span>
    <span style="position:absolute;left:50%;">0.25 drift ▲</span>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <div class="panel-title">PSI over time</div>
    <div class="mini-chart" id="psi-chart"></div>
  </div>
  <div class="panel">
    <div class="panel-title">KL divergence over time</div>
    <div class="mini-chart" id="kl-chart"></div>
  </div>
</div>

<div class="controls">
  <button class="btn btn-danger" onclick="injectDrift()">Inject drift data</button>
  <button class="btn btn-reset"  onclick="resetDrift()">Reset to stable</button>
  <button class="btn btn-dag"    onclick="triggerDag()">Trigger Airflow DAG</button>
</div>

<div class="log" id="log">
  <div class="log-line log-info">[system] Dashboard loaded. Fetching metrics...</div>
</div>

<script>
var psiHistory = [];
var klHistory  = [];

function ts() {
  var d = new Date();
  return '[' + d.getHours() + ':' + String(d.getMinutes()).padStart(2,'0') + ':' + String(d.getSeconds()).padStart(2,'0') + ']';
}

function log(msg, cls) {
  var el = document.getElementById('log');
  var line = document.createElement('div');
  line.className = 'log-line ' + (cls || '');
  line.textContent = ts() + ' ' + msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function barColor(v, thresholds) {
  if (v > thresholds[1]) return '#fc8181';
  if (v > thresholds[0]) return '#f6ad55';
  return '#68d391';
}

function renderChart(id, data, maxVal, thresholds) {
  var el = document.getElementById(id);
  el.innerHTML = '';
  var max = Math.max(maxVal, Math.max.apply(null, data.concat([0.001])));
  data.forEach(function(v) {
    var bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = Math.round((v / max) * 55) + 'px';
    bar.style.background = barColor(v, thresholds);
    el.appendChild(bar);
  });
}

function updateUI(data) {
  var psi = data.psi;
  var kl  = data.kl;
  var status = data.status;

  psiHistory.push(psi); if (psiHistory.length > 12) psiHistory.shift();
  klHistory.push(kl);   if (klHistory.length  > 12) klHistory.shift();

  var cls = status === 'drift' ? 'drift' : status === 'moderate' ? 'moderate' : 'stable';
  document.getElementById('psi-val').textContent = psi.toFixed(4);
  document.getElementById('psi-val').className = 'metric-value ' + cls;
  document.getElementById('kl-val').textContent = kl.toFixed(4);
  document.getElementById('stream-val').textContent = data.stream_count.toLocaleString();

  var pill = document.getElementById('status-pill');
  pill.className = 'status-pill pill-' + cls;
  pill.textContent = status === 'drift' ? 'DRIFT DETECTED' : status === 'moderate' ? 'MODERATE DRIFT' : 'STABLE';

  var pct = Math.min(Math.round((psi / 0.50) * 100), 100);
  document.getElementById('psi-fill').style.width = pct + '%';
  document.getElementById('psi-fill').style.background = barColor(psi, [0.10, 0.25]);
  document.getElementById('psi-pct').textContent = psi.toFixed(4) + ' / 0.50';

  renderChart('psi-chart', psiHistory, 0.50, [0.10, 0.25]);
  renderChart('kl-chart',  klHistory,  1.00, [0.25, 0.50]);
}

function fetchMetrics() {
  fetch('/metrics')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (!data.error) updateUI(data);
    })
    .catch(function() {});
}

function injectDrift() {
  log('Injecting 2000 drifted rows...', 'log-info');
  fetch('/inject-drift', { method: 'POST' })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.success) {
        log(data.rows_injected + ' drifted rows injected into streaming_predictions', 'log-alert');
        setTimeout(fetchMetrics, 500);
      } else {
        log('Error: ' + data.error, 'log-alert');
      }
    });
}

function resetDrift() {
  log('Removing drift data...', 'log-info');
  fetch('/reset-drift', { method: 'POST' })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.success) {
        log(data.rows_deleted + ' rows removed — PSI reset to stable', 'log-ok');
        setTimeout(fetchMetrics, 500);
      } else {
        log('Error: ' + data.error, 'log-alert');
      }
    });
}

function triggerDag() {
  log('Triggering Airflow DAG: credit_scoring_pipeline...', 'log-info');
  fetch('/trigger-dag', { method: 'POST' })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.success) {
        log('DAG triggered — run_id: ' + data.run_id, 'log-ok');
        log('Check Airflow UI: http://YOUR_SERVER_IP:8081', 'log-info');
      } else {
        log('DAG trigger failed: ' + data.message, 'log-alert');
      }
    });
}

fetchMetrics();
setInterval(fetchMetrics, 15000);
</script>
</body>
</html>
"""
