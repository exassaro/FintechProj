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
