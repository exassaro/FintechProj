import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier


# -----------------------------
# 1. Load and basic cleaning
# -----------------------------
df = pd.read_csv("datasets/dataset1_text_rich_transactions_harder_v2.csv")

# Parse dates (dd-mm-YYYY)
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

# Drop rows with critical missing values
df = df.dropna(subset=["description", "category_label", "transaction_date"])

# keep only INR
if "currency" in df.columns:
    df = df[df["currency"].astype(str).str.upper() == "INR"].copy()

# Fill missing vendor with placeholder
if "vendor_name" in df.columns:
    df["vendor_name"] = df["vendor_name"].fillna("UNKNOWNVENDOR")


# -----------------------------
# 2. Text cleaning functions
# -----------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # Noise tokens similar to your POC notebooks
    noise_tokens = [
        r"\bfy24\b", r"\bfy25\b", r"\bq1\b", r"\bq2\b",
        r"inv\d*", r"rcpt\d*", r"taxinv\d*", r"\bref\s*\d+",
        r"\bsubs\b", r"\badv\b", r"\bimps\b", r"\bupi\b",
        r"\btxnid\b", r"\btxn\b", r"\brcpt\b"
    ]
    for pat in noise_tokens:
        s = re.sub(pat, " ", s)
    # Keep alphanumerics and spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_vendor(v: str) -> str:
    if not isinstance(v, str):
        return ""
    v = v.upper()
    # Remove punctuation and common suffixes
    v = re.sub(r"[^A-Z0-9]+", " ", v)
    v = re.sub(r"\bPVT\b|\bPRIVATE\b|\bLTD\b|\bLIMITED\b|\bINDIA\b", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v

df["description_clean"] = df["description"].apply(clean_text)
df["vendor_clean"] = df["vendor_name"].apply(clean_vendor)

# -----------------------------
# 3. Numeric & date features
# -----------------------------
df["month"] = df["transaction_date"].dt.month
df["dow"] = df["transaction_date"].dt.dayofweek  # 0=Monday

# Amount (ensure numeric)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.dropna(subset=["amount"])

# -----------------------------
# 4. Combine text fields
# -----------------------------
df["textcombo"] = (
    df["description_clean"].fillna("") + " " +
    df["vendor_clean"].fillna("")
)

TEXTCOL = "textcombo"
NUMCOLS = ["amount", "month", "dow"]
TARGETCOL = "category_label"

X = df[[TEXTCOL] + NUMCOLS].copy()
y = df[TARGETCOL].copy()

# -----------------------------
# 5. Label encoding & split
# -----------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# -----------------------------
# 6. Preprocess + XGBoost model
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3
)

preprocess = ColumnTransformer(
    transformers=[
        ("text", tfidf, TEXTCOL),
        ("num", "passthrough", NUMCOLS),
    ],
    remainder="drop"
)

xgb_clf = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    max_depth=8,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", xgb_clf),
    ]
)

# -----------------------------
# 7. Train and evaluate
# -----------------------------
pipe.fit(X_train, y_train)

y_pred_enc = pipe.predict(X_test)
y_test_str = le.inverse_transform(y_test)
y_pred_str = le.inverse_transform(y_pred_enc)

print("TF-IDF + XGBoost (Type 1)")
print(classification_report(y_test_str, y_pred_str))
print("Macro F1:",
      f1_score(y_test_str, y_pred_str, average="macro"))







#Non numeric

#features_nontext.py

def clean_transactions(df):
    df = df.drop_duplicates()
    df = df[df["currency"].astype(str).str.upper() == "INR"]
    df = df.dropna(subset=["amount", "transaction_date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df["vendor_id"] = df["vendor_id"].fillna("UNKNOWN_VENDOR")
    return df


def add_date_features(df):
    dt = pd.to_datetime(df["transaction_date"], format="%d-%m-%Y", errors="coerce")
    df["txn_month"] = dt.dt.month
    df["txn_quarter"] = dt.dt.quarter
    df["txn_dow"] = dt.dt.dayofweek
    df["financial_year"] = np.where(dt.dt.month >= 4, dt.dt.year, dt.dt.year - 1)
    return df


def add_amount_features(df):
    df["amount_log"] = np.log1p(df["amount"])

    bins = [0, 1000, 5000, 20000, np.inf]
    labels = ["0-1k", "1k-5k", "5k-20k", "20k+"]
    df["amount_bucket"] = pd.cut(df["amount"], bins=bins, labels=labels, include_lowest=True)
    return df


def add_vendor_features(df):
    vc = df["vendor_id"].value_counts()
    df["vendor_txn_count"] = df["vendor_id"].map(vc).fillna(1)
    df["vendor_txn_count_log"] = np.log1p(df["vendor_txn_count"])

    # historical primary slab
    slab_mode = df.groupby("vendor_id")["gst_slab"].agg(lambda x: x.mode().iloc[0])
    df["vendor_primary_slab"] = df["vendor_id"].map(slab_mode)
    return df



#gst_slab_model.py


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

NUM_FEATURES = ["amount", "amount_log", "vendor_txn_count_log"]
CAT_FEATURES = ["amount_bucket", "txn_month", "txn_quarter", "txn_dow", "financial_year", "vendor_id", "vendor_primary_slab"]

def build_preprocessor():
    num_trans = ("num", StandardScaler(), NUM_FEATURES)
    cat_trans = ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
    return ColumnTransformer([num_trans, cat_trans])


def build_model():
    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )
    preprocessor = build_preprocessor()
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])


#gst_rules.py

def map_slab_to_tax_category(slab: str) -> str:
    if slab in ["0%", "exempt", "nil"]:
        return "exempt"
    elif slab in ["5%"]:
        return "reduced"
    elif slab in ["12%"]:
        return "standard_lower"
    elif slab in ["18%"]:
        return "standard"
    elif slab in ["28%"]:
        return "higher"
    else:
        return "unknown"


def compute_itc_eligibility(slab: str, gst_applicable: bool, context: dict) -> bool:
    if not gst_applicable:
        return False
    if slab in ["0%", "exempt", "nil"]:
        return False

    # future: add blocked credit rules based on context
    return True


#anomaly_nontext.py

from sklearn.ensemble import IsolationForest

def train_anomaly_model(X_num):
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_num)
    return iso

def flag_anomaly(iso, X_num_row, threshold=-0.2):
    score = iso.decision_function(X_num_row)[0]
    return score < threshold



#core_nontext_pipeline.py

def predict_transaction(row, clf, iso, class_labels, conf_thresh=0.75):
    X_row = row[NUM_FEATURES + CAT_FEATURES].to_frame().T

    # GST slab prediction
    proba = clf.predict_proba(X_row)[0]
    idx = proba.argmax()
    slab_pred = class_labels[idx]
    confidence = float(proba[idx])

    # anomaly
    X_num = row[["amount", "amount_log", "vendor_txn_count_log"]].to_frame().T
    anomaly_flag = flag_anomaly(iso, X_num)

    # rules
    tax_category = map_slab_to_tax_category(slab_pred)
    gst_applicable = bool(row["gst_applicable"])
    itc_eligible = compute_itc_eligibility(slab_pred, gst_applicable, context={})

    needs_review = (confidence < conf_thresh) or anomaly_flag

    return {
        "transaction_id": row["transaction_id"],
        "predicted_gst_slab": slab_pred,
        "confidence_score": confidence,
        "needs_review": needs_review,
        "tax_category": tax_category,
        "itc_eligible": itc_eligible,
        "anomaly_flag": anomaly_flag,
    }


