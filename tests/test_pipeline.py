"""
Basic pipeline tests — run by GitHub Actions CI
"""

import pytest
import pandas as pd
import numpy as np


# ─────────────────────────────────────────
# Test 1 — PSI calculation
# ─────────────────────────────────────────
def calculate_psi(baseline, current, n_bins=10):
    breakpoints  = np.unique(np.percentile(baseline, np.linspace(0, 100, n_bins + 1)))
    baseline_pct = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    current_pct  = np.histogram(current,  bins=breakpoints)[0] / len(current)
    baseline_pct = np.where(baseline_pct == 0, 1e-6, baseline_pct)
    current_pct  = np.where(current_pct  == 0, 1e-6, current_pct)
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def test_psi_same_distribution():
    """PSI should be near 0 for identical distributions."""
    data = np.random.normal(50000, 10000, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.01, f"PSI should be ~0 for same distribution, got {psi}"


def test_psi_different_distribution():
    """PSI should be high for very different distributions."""
    baseline = np.random.normal(10000, 1000, 1000)
    current  = np.random.normal(50000, 1000, 1000)
    psi = calculate_psi(baseline, current)
    assert psi > 0.25, f"PSI should be high for different distributions, got {psi}"


def test_psi_threshold():
    """PSI below 0.25 should be considered stable."""
    baseline = np.random.normal(50000, 10000, 1000)
    current  = np.random.normal(51000, 10000, 1000)  # slightly different
    psi = calculate_psi(baseline, current)
    assert psi < 0.25, f"Similar distributions should be STABLE, got PSI={psi}"


# ─────────────────────────────────────────
# Test 2 — Silver layer validation logic
# ─────────────────────────────────────────
DEFAULT_STATUSES = {
    "Charged Off", "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
}

def test_label_creation():
    """Label should be 1 for defaults, 0 for fully paid."""
    df = pd.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off", "Default", "Fully Paid"]
    })
    df["label"] = df["loan_status"].apply(lambda s: 1 if s in DEFAULT_STATUSES else 0)
    assert list(df["label"]) == [0, 1, 1, 0]


def test_numeric_cleaning():
    """Numeric columns should be converted properly."""
    df = pd.DataFrame({
        "int_rate": ["10.5%", "15.2%", "bad_value"],
        "loan_amnt": ["5000", "10000", "abc"],
    })
    df["int_rate"] = pd.to_numeric(
        df["int_rate"].str.replace("%", ""), errors="coerce"
    )
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")

    assert df["int_rate"].iloc[0] == 10.5
    assert df["loan_amnt"].iloc[0] == 5000.0
    assert pd.isna(df["int_rate"].iloc[2])


# ─────────────────────────────────────────
# Test 3 — Feature engineering
# ─────────────────────────────────────────
def test_loan_to_income():
    """loan_to_income should be loan_amnt / annual_inc."""
    df = pd.DataFrame({
        "loan_amnt" : [10000, 20000],
        "annual_inc": [50000, 100000],
    })
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    assert df["loan_to_income"].iloc[0] == pytest.approx(0.2)
    assert df["loan_to_income"].iloc[1] == pytest.approx(0.2)


def test_dti_risk_bands():
    """dti_risk should categorize DTI correctly."""
    df = pd.DataFrame({"dti": [5, 15, 25, 35]})
    df["dti_risk"] = pd.cut(
        df["dti"],
        bins=[-1, 10, 20, 30, float("inf")],
        labels=["low", "medium", "high", "very_high"]
    ).astype(str)
    assert df["dti_risk"].iloc[0] == "low"
    assert df["dti_risk"].iloc[1] == "medium"
    assert df["dti_risk"].iloc[2] == "high"
    assert df["dti_risk"].iloc[3] == "very_high"