import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier


# -----------------------------
# 1. Text cleaning functions
# -----------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    noise_tokens = [
        r"\bfy24\b", r"\bfy25\b", r"\bq1\b", r"\bq2\b",
        r"inv\d*", r"rcpt\d*", r"taxinv\d*", r"\bref\s*\d+",
        r"\bsubs\b", r"\badv\b", r"\bimps\b", r"\bupi\b",
        r"\btxnid\b", r"\btxn\b", r"\brcpt\b",
    ]
    for pat in noise_tokens:
        s = re.sub(pat, " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_vendor(v: str) -> str:
    if not isinstance(v, str):
        return ""
    v = v.upper()
    v = re.sub(r"[^A-Z0-9]+", " ", v)
    v = re.sub(r"\bPVT\b|\bPRIVATE\b|\bLTD\b|\bLIMITED\b|\bINDIA\b", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


# -----------------------------
# 2. Model components (NO TRAINING HERE)
# -----------------------------
TEXTCOL = "textcombo"
NUMCOLS = ["amount", "month", "dow"]

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,  # keep small for tests and smoke runs
)

preprocess = ColumnTransformer(
    transformers=[
        ("text", tfidf, TEXTCOL),
        ("num", "passthrough", NUMCOLS),
    ],
    remainder="drop",
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
    n_jobs=-1,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", xgb_clf),
    ]
)

le = LabelEncoder()


# -----------------------------
# 3. Inference function
# -----------------------------
def predict_category(
    description: str,
    vendorname: str,
    amount: float,
    transactiondate: str,
) -> str:
    """
    Assumes `pipe` and `le` have already been fitted.
    """
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
    return le.inverse_transform([pred_enc])[0]


# -----------------------------
# 4. Training entry point
# -----------------------------
if __name__ == "__main__":
    # Adjust the path to your actual CSV
    csv_path = r"C:\Users\USER\Documents\Capstone_project\unitest\datasets\dataset1_text_rich_transactions_harder_v3_prophet_ready.csv"

    df = pd.read_csv(csv_path)

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["description", "category_label", "transaction_date"])

    if "currency" in df.columns:
        df = df[df["currency"].astype(str).str.upper() == "INR"].copy()

    if "vendor_name" in df.columns:
        df["vendor_name"] = df["vendor_name"].fillna("UNKNOWNVENDOR")

    df["description_clean"] = df["description"].apply(clean_text)
    df["vendor_clean"] = df["vendor_name"].apply(clean_vendor)

    df["month"] = df["transaction_date"].dt.month
    df["dow"] = df["transaction_date"].dt.dayofweek

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    df[TEXTCOL] = df["description_clean"] + " " + df["vendor_clean"]

    X = df[[TEXTCOL] + NUMCOLS]
    y = df["category_label"]

    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42,
    )

    pipe.fit(X_train, y_train)

    y_pred_enc = pipe.predict(X_test)
    y_test_str = le.inverse_transform(y_test)
    y_pred_str = le.inverse_transform(y_pred_enc)

    print("TF-IDF + XGBoost (Type 1)")
    print(classification_report(y_test_str, y_pred_str))
    print("Macro F1:", f1_score(y_test_str, y_pred_str, average="macro"))
