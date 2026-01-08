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
    return score < threshold


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


# -------------------------------------------------
# 5. Training + evaluation entry point
# -------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv("datasets/dataset2_non_text_transactions.csv")
    print("Initial shape:", df.shape)

    # -----------------------------
    # Basic cleaning
    # -----------------------------
    df = df.drop_duplicates()

    if "currency" in df.columns:
        df = df[df["currency"].astype(str).str.upper() == "INR"].copy()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], format="%d-%m-%Y", errors="coerce"
    )
    df = df.dropna(subset=["transaction_date"])

    df["gst_slab"] = df["gst_slab"].astype(str).str.strip()
    df = df[df["gst_slab"].str.lower() != "unknown"].copy()
    df["gst_slab"] = df["gst_slab"].replace({"exempt": "0%"})

    if "vendor_id" in df.columns:
        df["vendor_id"] = df["vendor_id"].fillna("UNKNOWN_VENDOR")

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df_feat = df.copy()
    dt = df_feat["transaction_date"]

    df_feat["txn_month"] = dt.dt.month
    df_feat["txn_quarter"] = dt.dt.quarter
    df_feat["txn_dow"] = dt.dt.dayofweek
    df_feat["financial_year"] = np.where(dt.dt.month >= 4, dt.dt.year, dt.dt.year - 1)

    df_feat["amount_log"] = np.log1p(df_feat["amount"])

    bins = [0, 1000, 5000, 20000, np.inf]
    labels = ["0-1k", "1k-5k", "5k-20k", "20k+"]
    df_feat["amount_bucket"] = pd.cut(
        df_feat["amount"], bins=bins, labels=labels, include_lowest=True
    )

    if "vendor_id" in df_feat.columns:
        vc = df_feat["vendor_id"].value_counts()
        df_feat["vendor_txn_count"] = df_feat["vendor_id"].map(vc).fillna(1)
        df_feat["vendor_txn_count_log"] = np.log1p(df_feat["vendor_txn_count"])

        slab_mode = (
            df_feat.groupby("vendor_id")["gst_slab"]
            .agg(lambda x: x.mode().iloc[0])
        )
        df_feat["vendor_primary_slab"] = df_feat["vendor_id"].map(slab_mode)
    else:
        df_feat["vendor_txn_count_log"] = 0.0
        df_feat["vendor_primary_slab"] = "UNKNOWN"

    # -----------------------------
    # Train / evaluate
    # -----------------------------
    X = df_feat[NUM_FEATURES + CAT_FEATURES]
    y = df_feat[TARGET_COL]

    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=RANDOM_STATE,
    )

    pipe.fit(X_train, y_train)

    y_pred_enc = pipe.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    y_test_str = le.inverse_transform(y_test)

    print("Accuracy:", accuracy_score(y_test_str, y_pred))
    print("Macro F1:", f1_score(y_test_str, y_pred, average="macro"))
    print("\nClassification report:")
    print(classification_report(y_test_str, y_pred))

    # -----------------------------
    # Train anomaly model
    # -----------------------------
    iso.fit(df_feat[ANOM_NUM_FEATURES].values)

    # -----------------------------
    # Optional: save artifacts
    # -----------------------------
    # import joblib
    # joblib.dump(pipe, "artifacts/nontext_gst_slab_pipe_v1.pkl")
    # joblib.dump(le, "artifacts/nontext_gst_slab_label_encoder_v1.pkl")
    # joblib.dump(iso, "artifacts/nontext_gst_slab_iso_v1.pkl")
