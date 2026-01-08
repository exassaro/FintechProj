import numpy as np
import pytest

import finmodel.description_only_model as dom


# -------------------------------------------------
# 1. Tests for clean_description
# -------------------------------------------------
def test_clean_description_basic():
    text = "UPI Payment INV123 FY24"
    cleaned = dom.clean_description(text)

    assert isinstance(cleaned, str)
    assert "upi" in cleaned
    assert "inv" not in cleaned
    assert "fy" not in cleaned


def test_clean_description_non_string():
    assert dom.clean_description(None) == ""
    assert dom.clean_description(123) == ""


def test_clean_description_removes_noise():
    text = "Bill/1234 Ref 9988 01-04-2024"
    cleaned = dom.clean_description(text)

    assert "bill" in cleaned
    assert "ref" not in cleaned
    assert "2024" not in cleaned


# -------------------------------------------------
# 2. Inference helper tests (MOCKED)
# -------------------------------------------------
def test_predict_description_only_output_schema(monkeypatch):
    """
    Test inference helper without real model fitting.
    """

    # ---- Mock TF-IDF ----
    class DummyTFIDF:
        def transform(self, X):
            # Return fake sparse-like array
            return np.zeros((1, 10))

    # ---- Mock XGBoost ----
    class DummyXGB:
        def predict_proba(self, X):
            return np.array([[0.1, 0.7, 0.2]])

    # ---- Mock LabelEncoder ----
    class DummyLE:
        def inverse_transform(self, idxs):
            return np.array(["Rent"])

    monkeypatch.setattr(dom, "tfidf", DummyTFIDF())
    monkeypatch.setattr(dom, "xgb_desc_only", DummyXGB())
    monkeypatch.setattr(dom, "le", DummyLE())

    out = dom.predict_description_only("Monthly rent via UPI")

    assert isinstance(out, dict)
    assert set(out.keys()) == {
        "predicted_category",
        "confidence_score",
        "needs_review",
    }

    assert out["predicted_category"] == "Rent"
    assert 0.0 <= out["confidence_score"] <= 1.0
    assert out["needs_review"] is False


def test_predict_description_only_confidence_value(monkeypatch):
    class DummyTFIDF:
        def transform(self, X):
            return np.zeros((1, 5))

    class DummyXGB:
        def predict_proba(self, X):
            return np.array([[0.05, 0.95]])

    class DummyLE:
        def inverse_transform(self, idxs):
            return np.array(["Utilities"])

    monkeypatch.setattr(dom, "tfidf", DummyTFIDF())
    monkeypatch.setattr(dom, "xgb_desc_only", DummyXGB())
    monkeypatch.setattr(dom, "le", DummyLE())

    out = dom.predict_description_only("Electricity bill")

    assert out["predicted_category"] == "Utilities"
    assert out["confidence_score"] == pytest.approx(0.95, rel=1e-6)
