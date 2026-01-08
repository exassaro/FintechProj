import numpy as np
import pandas as pd

from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.ensemble import IsolationForest

from xgboost import XGBClassifier


RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.75

# -------------------------------------------------
# 1. Feature configuration
# -------------------------------------------------
ID_COL = "transaction_id"
TARGET_COL = "gst_slab"

NUM_FEATURES = [
    "amount",
    "amount_log",
    "vendor_txn_count_log",
]

CAT_FEATURES = [
    "amount_bucket",
    "txn_month",
    "txn_quarter",
    "txn_dow",
    "financial_year",
    "vendor_id",
    "vendor_primary_slab",
]

ANOM_NUM_FEATURES = [
    "amount",
    "amount_log",
    "vendor_txn_count_log",
]

# -------------------------------------------------
# 2. Model components (DEFINED, NOT TRAINED)
# -------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES),
    ]
)

xgb_clf = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", xgb_clf),
    ]
)

le = LabelEncoder()

iso = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=RANDOM_STATE,
)

# -------------------------------------------------
# 3. Rule helpers (DETERMINISTIC)
# -------------------------------------------------
def map_slab_to_tax_category(slab: str) -> str:
    s = str(slab).strip().lower()
    if s in ["0%", "exempt", "nil"]:
        return "exempt"
    if s == "5%":
        return "reduced"
    if s == "12%":
        return "standard_lower"
    if s == "18%":
        return "standard"
    if s == "28%":
        return "higher"
    return "unknown"


def compute_itc_eligibility(slab: str, gst_applicable: bool) -> bool:
    if not gst_applicable:
        return False
    s = str(slab).strip().lower()
    if s in ["0%", "exempt", "nil"]:
        return False
    return True


# -------------------------------------------------
# 4. Inference helpers (ASSUME FITTED MODELS)
# -------------------------------------------------
def predict_gst_slab_row(row: pd.Series) -> Dict[str, Any]:
    X_row = row[NUM_FEATURES + CAT_FEATURES].to_frame().T

    proba = pipe.predict_proba(X_row)[0]
    idx = int(np.argmax(proba))
    slab_pred = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])

    return {
        "predicted_gst_slab": slab_pred,
        "confidence_score": confidence,
    }


def flag_anomaly(row: pd.Series, threshold: float = -0.2) -> bool:
    X_row = row[ANOM_NUM_FEATURES].to_frame().T.values
    score = iso.decision_function(X_row)[0]
    return bool(score < threshold)   # ✅ FIXED


def predict_transaction_full(row: pd.Series) -> Dict[str, Any]:
    base = predict_gst_slab_row(row)

    slab_pred = base["predicted_gst_slab"]
    confidence = base["confidence_score"]

    anomaly_flag = flag_anomaly(row)

    gst_applicable = bool(row.get("gst_applicable", True))
    tax_category = map_slab_to_tax_category(slab_pred)
    itc_eligible = compute_itc_eligibility(slab_pred, gst_applicable)

    needs_review = (confidence < CONFIDENCE_THRESHOLD) or anomaly_flag

    return {
        "transaction_id": row[ID_COL],
        "predicted_gst_slab": slab_pred,
        "confidence_score": confidence,
        "needs_review": bool(needs_review),
        "tax_category": tax_category,
        "itc_eligible": bool(itc_eligible),
        "anomaly_flag": bool(anomaly_flag),
    }
