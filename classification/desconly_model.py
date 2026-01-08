import re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score

from xgboost import XGBClassifier


RANDOM_STATE = 42

# -------------------------------------------------
# 1. Text cleaning logic
# -------------------------------------------------
def clean_description(s: str) -> str:
    """Deterministic description cleaning (NO vendor usage)."""
    if not isinstance(s, str):
        return ""
    s = s.lower()

    replacements = {
        r"\bupi\b": " upi ",
        r"\bimps\b": " imps ",
        r"\bneft\b": " neft ",
        r"\brtgs\b": " rtgs ",
        r"\bcard\b": " card ",
        r"\bauto[- ]?debit\b": " autodebit ",
        r"\bbill\b": " bill ",
        r"\binv\b": " invoice ",
        r"\btaxinv\b": " taxinvoice ",
        r"\brcpt\b": " receipt ",
    }
    for pat, repl in replacements.items():
        s = re.sub(pat, repl, s)

    noise_tokens = [
        r"fy\d{2}",
        r"q[1-4]",
        r"ref\s*\d+",
        r"inv/?\d+",
        r"bill/?\d+",
        r"rcpt/?\d+",
        r"taxinv/?\d+",
        r"\d{2}-\d{2}-\d{4}",
    ]
    for pat in noise_tokens:
        s = re.sub(pat, " ", s)

    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------------------------------
# 2. Model components (DEFINED, NOT TRAINED)
# -------------------------------------------------
TEXTCOL = "description_clean"
TARGETCOL = "category_label"

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
)

xgb_desc_only = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    max_depth=7,
    learning_rate=0.1,
    n_estimators=250,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

le = LabelEncoder()


# -------------------------------------------------
# 3. Inference helper (ASSUMES FITTED MODELS)
# -------------------------------------------------
def predict_description_only(description: str) -> dict:
    """
    Single-description inference helper.
    Assumes tfidf, xgb_desc_only, le are already fitted.
    """
    desc_clean = clean_description(description)
    X_tfidf = tfidf.transform([desc_clean])

    proba = xgb_desc_only.predict_proba(X_tfidf)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(np.max(proba))

    return {
        "predicted_category": pred_label,
        "confidence_score": confidence,
        "needs_review": False,
    }


# -------------------------------------------------
# 4. Training + evaluation entry point
# -------------------------------------------------
if __name__ == "__main__":

    # Adjust dataset path as needed
    df = pd.read_csv("datasets/dataset1_text_rich_transactions_harder.csv")

    # Basic filtering
    df = df.dropna(subset=["description", "category_label"]).copy()

    print("Rows after dropna:", len(df))

    # Clean text
    df["description_clean"] = df["description"].apply(clean_description)

    # Prepare X / y
    X_text = df[TEXTCOL].astype(str)
    y = df[TARGETCOL].astype(str)

    y_enc = le.fit_transform(y)

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_text,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=RANDOM_STATE,
    )

    print("Train size:", len(X_train), "Val size:", len(X_val))

    # TF-IDF transform
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    print("TF-IDF shapes:", X_train_tfidf.shape, X_val_tfidf.shape)

    # Train model
    xgb_desc_only.fit(X_train_tfidf, y_train)

    # Evaluate
    y_val_proba = xgb_desc_only.predict_proba(X_val_tfidf)
    y_val_pred = y_val_proba.argmax(axis=1)

    y_val_true_str = le.inverse_transform(y_val)
    y_val_pred_str = le.inverse_transform(y_val_pred)

    print("Description-only TF-IDF + XGBoost")
    print(classification_report(y_val_true_str, y_val_pred_str))
    print("Macro F1:", f1_score(y_val_true_str, y_val_pred_str, average="macro"))

    # -------------------------------------------------
    # Optional: save artifacts
    # -------------------------------------------------
    # MODEL_VERSION = "v1"
    # joblib.dump(tfidf, f"artifacts/description_only_vectorizer_{MODEL_VERSION}.pkl")
    # joblib.dump(xgb_desc_only, f"artifacts/description_only_xgb_model_{MODEL_VERSION}.pkl")
    # joblib.dump(le, f"artifacts/description_only_label_encoder_{MODEL_VERSION}.pkl")
    # print("Saved artifacts:", MODEL_VERSION)
