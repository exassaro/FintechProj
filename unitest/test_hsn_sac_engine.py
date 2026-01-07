import inspect
import json
from typing import Any, Dict

import pandas as pd
import pytest

import finmodel.service as service


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rules_path(tmp_path_factory) -> str:
    """
    Provide a real hsn_sac_rules.json file for tests.
    """
    tmp_dir = tmp_path_factory.mktemp("rules")
    path = tmp_dir / "hsn_sac_rules.json"

    rules: Dict[str, Any] = {
        "metadata": {
            "version": "1.0.0",
            "effective_from_date": "2026-04-01",
            "last_updated": "2026-04-01",
            "authority": "GST Council / CBIC",
            "notes": ["Test configuration for HSN/SAC rule engine unit tests"],
        },
        "hsn_rules": {
            "8703": {
                "gst_slab": 28,
                "tax_category": "Passenger motor vehicles",
                "itc_eligible": False,
                "itc_conditions": ["ITC blocked for passenger motor vehicles"],
                "exemption_type": "blocked_itc",
                "description": "Motor vehicles for transport of persons",
            },
            "3004": {
                "gst_slab": 12,
                "tax_category": "Pharmaceutical products",
                "itc_eligible": True,
                "itc_conditions": ["ITC allowed for taxable business supplies"],
                "exemption_type": "none",
                "description": "Medicaments and pharmaceutical products",
            },
            "1001": {
                "gst_slab": 0,
                "tax_category": "Food grains",
                "itc_eligible": False,
                "itc_conditions": ["No ITC for exempt supplies"],
                "exemption_type": "absolute_exemption",
                "description": "Exempt food grains",
            },
        },
        "sac_rules": {
            "9983": {
                "gst_slab": 18,
                "tax_category": "Professional, technical and business services",
                "itc_eligible": True,
                "itc_conditions": ["ITC allowed for business services"],
            },
            "9965": {
                "gst_slab": 5,
                "tax_category": "Transport services",
                "itc_eligible": False,
                "itc_conditions": ["ITC restricted for transport services"],
            },
            "9993": {
                "gst_slab": 0,
                "tax_category": "Healthcare services",
                "itc_eligible": False,
                "itc_conditions": ["Healthcare services exempt"],
            },
        },
        "default_rules": {
            "unknown_code": {
                "default_gst_slab": None,
                "default_itc_eligible": None,
                "force_human_review": True,
            },
            "missing_code": {
                "default_gst_slab": None,
                "default_itc_eligible": None,
                "force_human_review": True,
            },
            "invalid_code": {
                "default_gst_slab": None,
                "default_itc_eligible": None,
                "force_human_review": True,
            },
        },
        "validation_rules": {
            "allowed_gst_slabs": [0, 5, 12, 18, 28],
            "hsn_code": {
                "allowed_lengths": [2, 4, 6, 8],
                "pattern": "^[0-9]{2,8}$",
            },
            "sac_code": {
                "allowed_lengths": [4, 6],
                "pattern": "^[0-9]{4,6}$",
            },
        },
        "confidence_policy": {
            "hsn_sac_present": 1.0,
            "unknown_or_invalid_code": 0.0,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

    return str(path)


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "transaction_id": "TXN001",
                "transaction_date": "2025-04-01",
                "hsn_sac_code": "8703",
                "code_type": "HSN",
                "taxable_amount": 100000.0,
            }
        ]
    )


# ---------------------------------------------------------------------------
# 1. VALID HSN CODE TESTS
# ---------------------------------------------------------------------------

def test_valid_hsn_8703_blocked_itc(rules_path, base_df):
    out = service.apply_hsn_sac_rules(base_df, rules_path)
    row = out.iloc[0]

    assert row["gst_slab"] == 28
    assert row["tax_category"] == "Passenger motor vehicles"
    assert not row["itc_eligible"]
    assert row["confidence_score"] == 1.0
    assert not row["needs_review"]
    assert row["rule_applied"] == "HSN_8703"


def test_valid_hsn_3004_full_itc(rules_path, base_df):
    df = base_df.copy()
    df["hsn_sac_code"] = "3004"

    out = service.apply_hsn_sac_rules(df, rules_path)
    row = out.iloc[0]

    assert row["gst_slab"] == 12
    assert row["itc_eligible"]



def test_valid_hsn_1001_exempt(rules_path, base_df):
    df = base_df.copy()
    df["hsn_sac_code"] = "1001"

    out = service.apply_hsn_sac_rules(df, rules_path)
    row = out.iloc[0]

    assert row["gst_slab"] == 0
    assert row["gst_amount"] == 0.0


# ---------------------------------------------------------------------------
# 2. VALID SAC CODE TESTS
# ---------------------------------------------------------------------------

def test_valid_sac_9983(rules_path, base_df):
    df = base_df.copy()
    df["hsn_sac_code"] = "9983"
    df["code_type"] = "SAC"
    df["taxable_amount"] = 250000.0

    out = service.apply_hsn_sac_rules(df, rules_path)
    row = out.iloc[0]

    assert row["gst_slab"] == 18
    assert row["itc_eligible"]
    assert row["gst_amount"] == round(250000 * 0.18, 2)


# ---------------------------------------------------------------------------
# 3. UNKNOWN / INVALID / MISSING CODE TESTS
# ---------------------------------------------------------------------------

def test_unknown_code(rules_path, base_df):
    df = base_df.copy()
    df["hsn_sac_code"] = "999999"

    out = service.apply_hsn_sac_rules(df, rules_path)
    row = out.iloc[0]

    assert row["gst_slab"] is None
    assert row["needs_review"]
    assert row["confidence_score"] == 0.0


@pytest.mark.parametrize("code", ["ABCD", "12AB", ""])
def test_invalid_or_missing_code(rules_path, base_df, code):
    df = base_df.copy()
    df["hsn_sac_code"] = code

    out = service.apply_hsn_sac_rules(df, rules_path)
    row = out.iloc[0]

    assert row["needs_review"]
    assert row["confidence_score"] == 0.0


# ---------------------------------------------------------------------------
# 4. NON-ML SAFETY & DETERMINISM
# ---------------------------------------------------------------------------

def test_no_ml_imports():
    source = inspect.getsource(service)
    forbidden = ["sklearn", "tensorflow", "torch", "xgboost"]
    for lib in forbidden:
        assert lib not in source


def test_deterministic_output(rules_path, base_df):
    out1 = service.apply_hsn_sac_rules(base_df, rules_path)
    out2 = service.apply_hsn_sac_rules(base_df, rules_path)

    pd.testing.assert_frame_equal(out1, out2)
