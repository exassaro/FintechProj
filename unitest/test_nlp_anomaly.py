import numpy as np
import pandas as pd
import pytest

from finmodel.nlp_anomaly import NLPAnomalyDetector


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def base_nlp_df():
    # Ensure deterministic behavior across runs
    np.random.seed(42)

    return pd.DataFrame(
        {
            "transaction_id": [f"TXN{i}" for i in range(1, 51)],
            "transaction_date": pd.date_range("2025-01-01", periods=50),
            "amount": np.concatenate(
                [
                    np.random.normal(10000, 1000, 45),
                    [50000, 60000, 70000, 80000, 90000],
                ]
            ),
            "predicted_category": ["Office Supplies"] * 50,
            "confidence_score": [0.95] * 50,
            "pipeline_used": ["NLP"] * 50,
            "vendor_name": ["VendorA"] * 45 + ["VendorB"] * 5,
        }
    )


@pytest.fixture
def detector_baseline():
    return NLPAnomalyDetector(
        min_category_samples=10,
        enable_isolation_forest=False,
        enable_lof=False,
    )


@pytest.fixture
def detector_with_ml():
    return NLPAnomalyDetector(
        min_category_samples=10,
        enable_isolation_forest=True,
        enable_lof=True,
        min_rows_for_ml=30,
        anomaly_threshold=0.5,
    )


# ---------------------------------------------------------------------
# Baseline Rule Tests
# ---------------------------------------------------------------------

def test_iqr_outlier_detection(base_nlp_df, detector_baseline):
    out = detector_baseline.detect(base_nlp_df)

    assert "anomaly_flag" in out.columns
    assert "anomaly_score" in out.columns

    # Expect anomalies from extreme values
    assert out["anomaly_flag"].sum() > 0
    assert out["anomaly_score"].max() >= 0.4


def test_category_amount_threshold():
    df = pd.DataFrame(
        {
            "transaction_id": ["TXN1"],
            "transaction_date": ["2025-01-01"],
            "amount": [300000],
            "predicted_category": ["Meals"],
            "confidence_score": [0.9],
            "pipeline_used": ["NLP"],
        }
    )

    detector = NLPAnomalyDetector(
        category_amount_thresholds={"Meals": {"high_amount": 50000}},
    )

    out = detector.detect(df)
    row = out.iloc[0]

    assert bool(row["anomaly_flag"]) is True
    assert row["anomaly_score"] >= 0.5
    assert "category" in row["anomaly_reason"].lower()


def test_vendor_rarity_detection(base_nlp_df, detector_baseline):
    df = base_nlp_df.copy()
    df.loc[49, "vendor_name"] = "RareVendor"

    out = detector_baseline.detect(df)
    rare_row = out.iloc[49]

    assert bool(rare_row["anomaly_flag"]) is True
    assert rare_row["anomaly_score"] >= 0.3
    assert (
    "vendor" in rare_row["anomaly_reason"].lower()
    or "outlier" in rare_row["anomaly_reason"].lower()
)



def test_no_vendor_column_safe():
    df = pd.DataFrame(
        {
            "transaction_id": ["TXN1"],
            "transaction_date": ["2025-01-01"],
            "amount": [10000],
            "predicted_category": ["Office Supplies"],
            "confidence_score": [0.95],
            "pipeline_used": ["NLP"],
        }
    )

    detector = NLPAnomalyDetector()
    out = detector.detect(df)

    assert "anomaly_flag" in out.columns
    assert "anomaly_score" in out.columns
    assert out["anomaly_score"].iloc[0] >= 0.0


# ---------------------------------------------------------------------
# Advanced Detector Tests (Isolation Forest + LOF)
# ---------------------------------------------------------------------

def test_isolation_forest_and_lof_enabled(base_nlp_df, detector_with_ml):
    out = detector_with_ml.detect(base_nlp_df)

    assert "anomaly_score" in out.columns
    assert out["anomaly_score"].max() <= 1.0
    assert out["anomaly_score"].max() > 0.0


def test_ml_detectors_gated_for_small_dataset():
    df = pd.DataFrame(
        {
            "transaction_id": ["TXN1", "TXN2"],
            "transaction_date": ["2025-01-01", "2025-01-02"],
            "amount": [1000, 2000],
            "predicted_category": ["Office Supplies", "Office Supplies"],
            "confidence_score": [0.9, 0.9],
            "pipeline_used": ["NLP", "NLP"],
        }
    )

    detector = NLPAnomalyDetector(
        enable_isolation_forest=True,
        enable_lof=True,
        min_rows_for_ml=10,
    )

    out = detector.detect(df)

    # ML detectors should be skipped
    assert out["anomaly_score"].max() <= 0.8
    assert "isolation" not in " ".join(out["anomaly_reason"].astype(str)).lower()
    assert "lof" not in " ".join(out["anomaly_reason"].astype(str)).lower()


# ---------------------------------------------------------------------
# Determinism Test
# ---------------------------------------------------------------------

def test_deterministic_output(base_nlp_df, detector_with_ml):
    out1 = detector_with_ml.detect(base_nlp_df)
    out2 = detector_with_ml.detect(base_nlp_df)

    pd.testing.assert_frame_equal(
        out1.sort_index(axis=1),
        out2.sort_index(axis=1),
    )


# ---------------------------------------------------------------------
# Fail-soft Behavior Test
# ---------------------------------------------------------------------

def test_missing_required_columns_fail_soft():
    df = pd.DataFrame({"foo": [1], "bar": [2]})

    detector = NLPAnomalyDetector()
    out = detector.detect(df)

    assert "anomaly_flag" in out.columns
    assert "anomaly_score" in out.columns
    assert bool(out["anomaly_flag"].iloc[0]) is False
    assert out["anomaly_score"].iloc[0] == 0.0
