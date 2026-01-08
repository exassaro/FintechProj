# tests/test_hsn_sac_anomaly.py

import copy
import pandas as pd
import numpy as np
import pytest
import json

from finmodel.hsn_sac_anomaly import HsnSacAnomalyDetector, HsnSacAnomalyConfig


# ---------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def basic_rules(tmp_path):
    """
    Create a minimal hsn_sac.json-like rules file for tests.

    Codes:
      - 8703: HSN, Passenger motor vehicles, gst_slab=28, itc_eligible=False
      - 3004: HSN, Pharmaceutical products, gst_slab=12, itc_eligible=True
      - 9983: SAC, Professional services, gst_slab=18, itc_eligible=True
    """
    rules = {
        "metadata": {"version": "1.0.0"},
        "hsn_rules": {
            "8703": {
                "gst_slab": 28,
                "tax_category": "Passenger motor vehicles",
                "itc_eligible": False,
            },
            "3004": {
                "gst_slab": 12,
                "tax_category": "Pharmaceutical products",
                "itc_eligible": True,
            },
        },
        "sac_rules": {
            "9983": {
                "gst_slab": 18,
                "tax_category": "Professional services",
                "itc_eligible": True,
            }
        },
    }
    path = tmp_path / "hsn_sac.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return str(path)


@pytest.fixture
def detector_with_rules(basic_rules):
    cfg = HsnSacAnomalyConfig(hsn_sac_rules_path=basic_rules, min_vendor_category_samples=5)
    return HsnSacAnomalyDetector(config=cfg)


def _base_df():
    return pd.DataFrame(
        [
            {
                "transaction_id": "TXN1",
                "hsn_sac_code": "8703",
                "predicted_category": "Passenger motor vehicles",
                "gst_slab": 28,
                "itc_eligible": False,
                "confidence_score": 0.9,
            }
        ]
    )


# ---------------------------------------------------------------------
# 1. Input Validation & Soft-Fail Behavior
# ---------------------------------------------------------------------


def test_non_dataframe_returns_input_unchanged(detector_with_rules):
    obj = {"a": 1}
    out = detector_with_rules.detect(obj)
    assert out is obj


def test_empty_dataframe_returns_empty_anomaly_columns(detector_with_rules):
    df = pd.DataFrame(
        columns=[
            "transaction_id",
            "hsn_sac_code",
            "predicted_category",
            "gst_slab",
            "itc_eligible",
            "confidence_score",
        ]
    )
    out = detector_with_rules.detect(df)
    assert out.empty
    assert "hsn_anomaly_score" in out.columns
    assert "hsn_anomaly_flag" in out.columns
    assert "hsn_anomaly_reason" in out.columns


def test_missing_required_columns_soft_fail():
    detector = HsnSacAnomalyDetector()
    df = pd.DataFrame({"transaction_id": ["TXN1"]})
    out = detector.detect(df)
    assert "hsn_anomaly_score" in out.columns
    assert "hsn_anomaly_flag" in out.columns
    assert "hsn_anomaly_reason" in out.columns
    assert (out["hsn_anomaly_score"] == 0.0).all()
    assert (out["hsn_anomaly_flag"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------
# 2. Invalid / Unknown HSN/SAC Codes
# ---------------------------------------------------------------------


def test_unknown_hsn_sac_code_flagged(detector_with_rules):
    df = _base_df()
    df["hsn_sac_code"] = "999999"  # not present in rules

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    assert row["hsn_anomaly_score"] > 0.5  # high anomaly score
    assert "unknown" in row["hsn_anomaly_reason"].lower() or "missing" in row["hsn_anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 3. Category vs HSN/SAC Mismatch
# ---------------------------------------------------------------------


def test_category_mismatch_flagged(detector_with_rules):
    df = _base_df()
    # Valid code 8703 but wrong predicted category
    df["predicted_category"] = "Office Supplies"

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    assert row["hsn_anomaly_score"] >= detector_with_rules.config.score_category_mismatch
    assert "mismatch" in row["hsn_anomaly_reason"].lower()
    assert "category" in row["hsn_anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 4. GST Slab Inconsistency
# ---------------------------------------------------------------------


def test_gst_slab_inconsistency_flagged(detector_with_rules):
    df = _base_df()
    # Correct code 8703 but wrong slab
    df["gst_slab"] = 18

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    assert row["hsn_anomaly_score"] >= detector_with_rules.config.score_gst_slab_inconsistency
    assert "gst slab" in row["hsn_anomaly_reason"].lower() or "gst" in row["hsn_anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 5. ITC Eligibility Violation
# ---------------------------------------------------------------------


def test_itc_violation_blocked_code_marked_eligible(detector_with_rules):
    df = _base_df()
    # 8703 is non-eligible in rules; mark eligible → strong violation
    df["itc_eligible"] = True

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    assert row["hsn_anomaly_score"] >= detector_with_rules.config.score_itc_violation_strong
    assert "itc violation" in row["hsn_anomaly_reason"].lower()


def test_itc_violation_eligible_code_marked_non_eligible(detector_with_rules):
    df = _base_df()
    # Switch to 3004 which is eligible
    df["hsn_sac_code"] = "3004"
    df["predicted_category"] = "Pharmaceutical products"
    df["gst_slab"] = 12
    df["itc_eligible"] = False  # should be True per rules

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    assert row["hsn_anomaly_score"] >= detector_with_rules.config.score_itc_violation_mild
    assert "underclaim" in row["hsn_anomaly_reason"].lower() or "allows itc" in row["hsn_anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 6. HSN vs SAC Structural Violation
# ---------------------------------------------------------------------


def test_sac_used_for_goods_category_structural_violation(detector_with_rules):
    df = _base_df()
    # Use SAC code 9983 but a goods-like category
    df["hsn_sac_code"] = "9983"
    df["predicted_category"] = "Goods - Machinery"

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    assert "structural" in row["hsn_anomaly_reason"].lower()
    assert "sac" in row["hsn_anomaly_reason"].lower()


def test_hsn_used_for_service_category_structural_violation(detector_with_rules):
    df = _base_df()
    # Use HSN 3004 but a service-like category
    df["hsn_sac_code"] = "3004"
    df["predicted_category"] = "Professional services"

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    assert "structural" in row["hsn_anomaly_reason"].lower()
    assert "hsn" in row["hsn_anomaly_reason"].lower() or "service-like" in row["hsn_anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 7. Low Confidence Regulatory Risk
# ---------------------------------------------------------------------


def test_low_confidence_regulatory_risk_flagged(detector_with_rules):
    df = _base_df()
    df["confidence_score"] = 0.1  # very low

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    # At least low_confidence score applied
    assert row["hsn_anomaly_score"] >= detector_with_rules.config.score_low_conf_risk
    assert "low confidence" in row["hsn_anomaly_reason"].lower()


# ---------------------------------------------------------------------
# 8. Vendor-Level HSN/SAC Inconsistency
# ---------------------------------------------------------------------


def test_vendor_level_inconsistency_flagged(detector_with_rules):
    cfg = copy.deepcopy(detector_with_rules.config)
    cfg.min_vendor_category_samples = 5
    detector = HsnSacAnomalyDetector(config=cfg)

    # Same vendor, same category, dominant code 8703, minority code 3004
    rows = []
    # 8 rows with 8703
    for i in range(8):
        rows.append(
            {
                "transaction_id": f"TXN{i}",
                "hsn_sac_code": "8703",
                "predicted_category": "Passenger motor vehicles",
                "gst_slab": 28,
                "itc_eligible": False,
                "confidence_score": 0.9,
                "vendor_id": "V001",
            }
        )
    # 4 rows with 3004 (minority codes)
    for i in range(4):
        rows.append(
            {
                "transaction_id": f"TXN_M{i}",
                "hsn_sac_code": "3004",
                "predicted_category": "Passenger motor vehicles",
                "gst_slab": 28,
                "itc_eligible": False,
                "confidence_score": 0.9,
                "vendor_id": "V001",
            }
        )

    df = pd.DataFrame(rows)

    out = detector.detect(df)
    minority_rows = out[out["hsn_sac_code"] == "3004"]

    assert (minority_rows["hsn_anomaly_flag"]).any()
    assert (minority_rows["hsn_anomaly_score"] >= detector.config.score_vendor_inconsistency).any()
    assert minority_rows["hsn_anomaly_reason"].str.lower().str.contains("vendor-level").any()


# ---------------------------------------------------------------------
# 9. Multiple Anomalies Aggregation
# ---------------------------------------------------------------------


def test_multiple_anomalies_aggregated(detector_with_rules):
    df = _base_df()
    # Create multiple anomalies:
    # - Wrong category
    # - Wrong GST slab
    # - ITC violation
    df["predicted_category"] = "Office Supplies"
    df["gst_slab"] = 5
    df["itc_eligible"] = True

    out = detector_with_rules.detect(df)
    row = out.iloc[0]

    assert bool(row["hsn_anomaly_flag"]) is True
    # Score should be at least the max of category_mismatch and itc_violation_strong
    expected_min = max(
        detector_with_rules.config.score_category_mismatch,
        detector_with_rules.config.score_itc_violation_strong,
        detector_with_rules.config.score_gst_slab_inconsistency,
    )
    assert row["hsn_anomaly_score"] >= expected_min

    reason = row["hsn_anomaly_reason"].lower()
    assert "mismatch" in reason
    assert "gst" in reason or "slab" in reason
    assert "itc" in reason
    # No dupes of the same phrase
    parts = [p.strip() for p in reason.split(";") if p.strip()]
    assert len(parts) == len(set(parts))


# ---------------------------------------------------------------------
# 10. Numeric Independence
# ---------------------------------------------------------------------


def test_amount_changes_do_not_affect_hsn_anomalies(detector_with_rules):
    # Base DF with an amount column that should be ignored by HSN/SAC detector
    df = _base_df()
    df["amount"] = 1000.0

    out1 = detector_with_rules.detect(df)

    # Change amount drastically; HSN anomalies must remain identical
    df2 = df.copy()
    df2["amount"] = 1_000_000.0
    out2 = detector_with_rules.detect(df2)

    # Compare only HSN anomaly columns
    cols = ["hsn_anomaly_flag", "hsn_anomaly_score", "hsn_anomaly_reason"]
    pd.testing.assert_series_equal(out1[cols[0]], out2[cols[0]])
    pd.testing.assert_series_equal(out1[cols[1]], out2[cols[1]])
    pd.testing.assert_series_equal(out1[cols[2]], out2[cols[2]])
