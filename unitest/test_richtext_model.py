import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from finmodel.richtext_model import (
    clean_text,
    clean_vendor,
    TEXTCOL,
)


# -------------------------------------------------
# 1. Text cleaning tests
# -------------------------------------------------
def test_clean_text_basic():
    s = "UPI AWFIS RCPT2404002 FY24 - RENT EXP REF 6160"
    out = clean_text(s)
    assert "upi" not in out
    assert "fy24" not in out
    assert "rcpt2404002" not in out
    assert "rent" in out
    assert "exp" in out


def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_clean_vendor_basic():
    v = "Awfis India Pvt Ltd."
    out = clean_vendor(v)
    assert out == "AWFIS"


def test_clean_vendor_non_string():
    assert clean_vendor(None) == ""


# -------------------------------------------------
# 2. Dummy dataset
# -------------------------------------------------
def _make_dummy_df():
    return pd.DataFrame(
        [
            {
                "transaction_date": "2024-04-01",
                "description": "UPI AWFIS RCPT2404002 FY24 - RENT EXP REF 6160",
                "vendor_name": "AWFIS INDIA PVT LTD",
                "amount": 58560.14,
                "category_label": "Rent",
                "currency": "INR",
            },
            {
                "transaction_date": "2024-04-01",
                "description": "IMPS ZOMATO INV2404003 Q2 - MEALS EXP",
                "vendor_name": "ZOMATO",
                "amount": 2184.95,
                "category_label": "Meals",
                "currency": "INR",
            },
        ]
    )


# -------------------------------------------------
# 3. Dummy classifier and pipeline tests
# -------------------------------------------------
class DummyClf:
    """Simple majority-class classifier used only in tests."""

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.majority_ = values[counts.argmax()]
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.majority_, dtype=int)


def _build_dummy_pipe_and_encoder():
    df = _make_dummy_df()

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["description_clean"] = df["description"].apply(clean_text)
    df["vendor_clean"] = df["vendor_name"].apply(clean_vendor)
    df["month"] = df["transaction_date"].dt.month
    df["dow"] = df["transaction_date"].dt.dayofweek

    X = pd.DataFrame(
        {
            TEXTCOL: df["description_clean"] + " " + df["vendor_clean"],
            "amount": df["amount"],
            "month": df["month"],
            "dow": df["dow"],
        }
    )
    y = df["category_label"]

    tfidf = TfidfVectorizer(min_df=1)
    preprocess = ColumnTransformer(
        [
            ("text", tfidf, TEXTCOL),
            ("num", "passthrough", ["amount", "month", "dow"]),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", DummyClf()),
        ]
    )

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pipe.fit(X, y_enc)

    return pipe, le


def test_pipeline_fit_and_predict_smoke():
    pipe, le = _build_dummy_pipe_and_encoder()
    df = _make_dummy_df()

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["description_clean"] = df["description"].apply(clean_text)
    df["vendor_clean"] = df["vendor_name"].apply(clean_vendor)
    df["month"] = df["transaction_date"].dt.month
    df["dow"] = df["transaction_date"].dt.dayofweek

    X = pd.DataFrame(
        {
            TEXTCOL: df["description_clean"] + " " + df["vendor_clean"],
            "amount": df["amount"],
            "month": df["month"],
            "dow": df["dow"],
        }
    )

    y_pred_enc = pipe.predict(X)
    y_pred = le.inverse_transform(y_pred_enc)

    assert len(y_pred) == len(df)
    assert set(y_pred).issubset(set(df["category_label"]))


def test_predict_category_interface_like_path():
    """
    Test an inference path equivalent to predict_category, using the dummy
    pipeline and encoder, without calling richtext_model.predict_category.
    """
    pipe, le = _build_dummy_pipe_and_encoder()

    description = "UPI AWFIS RCPT2404002 FY24 - RENT EXP REF 6160"
    vendorname = "AWFIS INDIA PVT LTD"
    amount = 58560.14
    transactiondate = "01-04-2024"

    dt = pd.to_datetime(transactiondate, format="%d-%m-%Y", errors="coerce")
    month = int(dt.month) if pd.notnull(dt) else 0
    dow = int(dt.dayofweek) if pd.notnull(dt) else 0

    desc_clean = clean_text(description)
    vend_clean = clean_vendor(vendorname)
    textcombo = f"{desc_clean} {vend_clean}".strip()

    x_row = pd.DataFrame(
        [
            {
                TEXTCOL: textcombo,
                "amount": float(amount),
                "month": month,
                "dow": dow,
            }
        ]
    )

    pred_enc = pipe.predict(x_row)[0]
    cat = le.inverse_transform([pred_enc])[0]

    assert isinstance(cat, str)
    assert len(cat) > 0
