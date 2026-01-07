import inspect
import pandas as pd
import numpy as np
import pytest

from finmodel.numeric_anomaly import NumericAnomalyDetector, NumericAnomalyConfig


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def base_df(n: int, amount: float = 100.0, category: str = "Office"):
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:03d}" for i in range(n)],
            "transaction_date": pd.to_datetime(["2025-01-01"] * n),
            "amount": [amount] * n,
            "predicted_category": [category] * n,
            "confidence_score": [0.9] * n,
        }
    )


# ---------------------------------------------------------------------
# 1. Structural & Safety
# ---------------------------------------------------------------------

def test_non_dataframe_returns_input():
    detector = NumericAnomalyDetector()
    obj = {"a": 1}
    assert detector.detect(obj) is obj


def test_missing_columns_soft_fail():
    detector = NumericAnomalyDetector()
    df = pd.DataFrame({"transaction_id": ["TXN1"]})
    out = detector.detect(df)

    assert "anomaly_flag" in out
    assert out["anomaly_flag"].sum() == 0


def test_empty_dataframe():
    detector = NumericAnomalyDetector()
    df = base_df(0)
    out = detector.detect(df)

    assert out.empty
    assert "anomaly_score" in out.columns


def test_predictions_unchanged():
    detector = NumericAnomalyDetector()
    df = base_df(3)
    df["predicted_category"] = ["A", "B", "C"]
    df["confidence_score"] = [0.1, 0.5, 0.9]

    out = detector.detect(df)

    assert out["predicted_category"].tolist() == ["A", "B", "C"]
    assert out["confidence_score"].tolist() == [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------
# 2. IQR Outliers
# ---------------------------------------------------------------------

def test_iqr_outliers():
    # Use thresholds that clearly separate a smaller and larger outlier
    cfg = NumericAnomalyConfig(
        min_category_samples=10,
        iqr_threshold_mild=0.5,
        iqr_threshold_strong=2.0,
    )
    detector = NumericAnomalyDetector(**cfg.__dict__)

    amounts = [100] * 10 + [500, 1000]  # 500 = smaller outlier, 1000 = stronger outlier
    df = pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(len(amounts))],
            "transaction_date": pd.to_datetime(["2025-01-01"] * len(amounts)),
            "amount": amounts,
            "predicted_category": ["Office"] * len(amounts),
            "confidence_score": [0.9] * len(amounts),
        }
    )

    out = detector.detect(df)

    mild = out[out["amount"] == 500].iloc[0]
    strong = out[out["amount"] == 1000].iloc[0]

    # Just assert that the smaller outlier is flagged at all
    assert mild["anomaly_score"] >= 0.3
    # Strong outlier still should be high
    assert strong["anomaly_score"] >= 0.7
    assert "outlier" in strong["anomaly_reason"].lower()



# ---------------------------------------------------------------------
# 3. MAD Outliers
# ---------------------------------------------------------------------

def test_mad_outlier():
    cfg = NumericAnomalyConfig(min_category_samples=10)
    detector = NumericAnomalyDetector(**cfg.__dict__)

    amounts = [100] * 20 + [350]
    df = base_df(21)
    df["amount"] = amounts
    df["predicted_category"] = "Utilities"

    out = detector.detect(df)
    row = out[out["amount"] == 350].iloc[0]

    assert row["anomaly_score"] >= 0.7
    assert "mad" in row["anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 4. Time Window Spike
# ---------------------------------------------------------------------

def test_time_window_spike():
    cfg = NumericAnomalyConfig(min_category_samples=5, rolling_window_days=3)
    detector = NumericAnomalyDetector(**cfg.__dict__)

    df = pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(6)],
            "transaction_date": pd.date_range("2025-01-01", periods=6),
            "amount": [100, 100, 100, 100, 1000, 100],
            "predicted_category": ["Travel"] * 6,
            "confidence_score": [0.9] * 6,
        }
    )

    out = detector.detect(df)
    spike = out[out["amount"] == 1000].iloc[0]

    assert spike["anomaly_flag"]
    assert spike["anomaly_score"] >= 0.4
    assert "time-window" in spike["anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 5. Frequency Anomaly
# ---------------------------------------------------------------------

def test_frequency_surge():
    cfg = NumericAnomalyConfig(min_category_samples=5)
    detector = NumericAnomalyDetector(**cfg.__dict__)

    rows = []
    for d in range(4):
        rows.append(
            {
                "transaction_id": f"S{d}",
                "transaction_date": f"2025-01-0{d+1}",
                "amount": 100,
                "predicted_category": "IT",
                "confidence_score": 0.9,
            }
        )
    for i in range(10):
        rows.append(
            {
                "transaction_id": f"X{i}",
                "transaction_date": "2025-01-05",
                "amount": 100,
                "predicted_category": "IT",
                "confidence_score": 0.9,
            }
        )

    df = pd.DataFrame(rows)
    out = detector.detect(df)

    surge = out[out["transaction_date"].dt.date == pd.to_datetime("2025-01-05").date()]
    assert surge["anomaly_flag"].all()
    assert (surge["anomaly_score"] >= 0.5).all()


# ---------------------------------------------------------------------
# 6. Vendor-Based Deviation
# ---------------------------------------------------------------------

def test_vendor_deviation():
    cfg = NumericAnomalyConfig(min_category_samples=5)
    detector = NumericAnomalyDetector(**cfg.__dict__)

    amounts = [100] * 10 + [1000]
    df = base_df(len(amounts))
    df["amount"] = amounts
    df["vendor_id"] = ["V1"] * len(amounts)

    out = detector.detect(df)
    row = out[out["amount"] == 1000].iloc[0]

    assert row["anomaly_score"] >= 0.6
    assert "vendor" in row["anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 7. ML Gating
# ---------------------------------------------------------------------

def test_ml_disabled():
    cfg = NumericAnomalyConfig(enable_isolation_forest=False, enable_lof=False)
    detector = NumericAnomalyDetector(**cfg.__dict__)

    df = base_df(20)
    df.loc[19, "amount"] = 5000

    out = detector.detect(df)
    row = out.iloc[19]

    assert "isolation" not in row["anomaly_reason"].lower()
    assert "lof" not in row["anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 8. Rule Dominance
# ---------------------------------------------------------------------

def test_rule_dominates_ml():
    cfg = NumericAnomalyConfig(enable_isolation_forest=True, enable_lof=True, min_rows_for_ml=10)
    detector = NumericAnomalyDetector(**cfg.__dict__)

    df = base_df(15)
    df.loc[14, "amount"] = 10000

    out = detector.detect(df)
    row = out.iloc[14]

    assert row["anomaly_score"] >= 0.8
    assert "outlier" in row["anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 9. Drift Placeholder
# ---------------------------------------------------------------------

def test_drift_placeholder():
    detector = NumericAnomalyDetector()
    msg = detector.describe_drift_detection_design()
    assert isinstance(msg, str)
