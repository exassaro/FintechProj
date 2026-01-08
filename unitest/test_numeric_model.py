import numpy as np
import pandas as pd
import pytest

import finmodel.numeric_model as nm


# -------------------------------------------------
# 1. Rule helper tests
# -------------------------------------------------
def test_map_slab_to_tax_category():
    assert nm.map_slab_to_tax_category("0%") == "exempt"
    assert nm.map_slab_to_tax_category("exempt") == "exempt"
    assert nm.map_slab_to_tax_category("5%") == "reduced"
    assert nm.map_slab_to_tax_category("12%") == "standard_lower"
    assert nm.map_slab_to_tax_category("18%") == "standard"
    assert nm.map_slab_to_tax_category("28%") == "higher"
    assert nm.map_slab_to_tax_category("unknown") == "unknown"


def test_compute_itc_eligibility():
    assert nm.compute_itc_eligibility("18%", True) is True
    assert nm.compute_itc_eligibility("0%", True) is False
    assert nm.compute_itc_eligibility("exempt", True) is False
    assert nm.compute_itc_eligibility("18%", False) is False


# -------------------------------------------------
# 2. Dummy transaction row
# -------------------------------------------------
def _make_dummy_row():
    return pd.Series(
        {
            "transaction_id": "TXN1",
            "amount": 5000.0,
            "amount_log": np.log1p(5000.0),
            "vendor_txn_count_log": np.log1p(10),
            "amount_bucket": "1k-5k",
            "txn_month": 4,
            "txn_quarter": 2,
            "txn_dow": 1,
            "financial_year": 2024,
            "vendor_id": "VENDOR1",
            "vendor_primary_slab": "18%",
            "gst_applicable": True,
        }
    )


# -------------------------------------------------
# 3. Inference helpers (MOCKED MODELS)
# -------------------------------------------------
def test_predict_gst_slab_row(monkeypatch):
    row = _make_dummy_row()

    # ---- Mock pipeline ----
    class DummyPipe:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])

    # ---- Mock LabelEncoder ----
    class DummyLE:
        def inverse_transform(self, idxs):
            return np.array(["18%"])

    monkeypatch.setattr(nm, "pipe", DummyPipe())
    monkeypatch.setattr(nm, "le", DummyLE())

    out = nm.predict_gst_slab_row(row)

    assert out["predicted_gst_slab"] == "18%"
    assert out["confidence_score"] == pytest.approx(0.9)


def test_flag_anomaly(monkeypatch):
    row = _make_dummy_row()

    class DummyISO:
        def decision_function(self, X):
            return np.array([-0.5])

    monkeypatch.setattr(nm, "iso", DummyISO())

    assert nm.flag_anomaly(row) is True


def test_predict_transaction_full_normal_case(monkeypatch):
    row = _make_dummy_row()

    # ---- Mock slab predictor ----
    monkeypatch.setattr(
        nm,
        "predict_gst_slab_row",
        lambda r: {"predicted_gst_slab": "18%", "confidence_score": 0.95},
    )

    # ---- Mock anomaly ----
    monkeypatch.setattr(nm, "flag_anomaly", lambda r: False)

    out = nm.predict_transaction_full(row)

    assert out["transaction_id"] == "TXN1"
    assert out["predicted_gst_slab"] == "18%"
    assert out["confidence_score"] == 0.95
    assert out["needs_review"] is False
    assert out["tax_category"] == "standard"
    assert out["itc_eligible"] is True
    assert out["anomaly_flag"] is False


def test_predict_transaction_full_needs_review(monkeypatch):
    row = _make_dummy_row()

    monkeypatch.setattr(
        nm,
        "predict_gst_slab_row",
        lambda r: {"predicted_gst_slab": "18%", "confidence_score": 0.4},
    )
    monkeypatch.setattr(nm, "flag_anomaly", lambda r: True)

    out = nm.predict_transaction_full(row)

    assert out["needs_review"] is True
