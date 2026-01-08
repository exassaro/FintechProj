from __future__ import annotations

import re
from typing import Dict, List, Tuple
import pandas as pd


# ============================================================
# CANONICAL COLUMN ROLES (LOCKED CONTRACT WITH MODELS)
# ============================================================

ROLE_SYNONYMS: Dict[str, List[str]] = {
    "transaction_id": [
        r"^txn_?id$", r"^transaction_?id$", r"^voucher_?id$",
        r"^doc_?no$", r"^doc_?id$", r"^document_?number$",
        r"^ref_?no$", r"^id$",
    ],
    "description": [
        r"^description$", r"^narration$", r"^details?$",
        r"^remark$", r"^remarks$", r"^memo$", r"^particulars$",
    ],
    "vendor_name": [
        r"^vendor$", r"^vendor_?name$", r"^party_?name$",
        r"^supplier$", r"^customer$", r"^counterparty$",
        r"^payee$", r"^account_?name$",
    ],
    "amount": [
        r"^amount$", r"^txn_?amount$", r"^txn_?amt$",
        r"^transaction_?amount$", r"^value$", r"^net_?amount$",
        r"^total$", r"^debit$", r"^credit$",
    ],
    "transaction_date": [
        r"^date$", r"^txn_?date$", r"^transaction_?date$",
        r"^posting_?date$", r"^voucher_?date$",
        r"^doc_?date$", r"^entry_?date$",
    ],
    "hsn_sac_code": [
        r"^hsn$", r"^hsn_?code$", r"^sac$", r"^sac_?code$",
        r"^hsn_?sac$",
    ],
    "gst_slab": [
        r"^gst_?slab$", r"^gst_?rate$", r"^gst_?percent$",
        r"^tax_?rate$",
    ],
    "itc_eligible": [
        r"^itc_?eligible$", r"^is_?itc$", r"^itc_?flag$",
        r"^input_?tax_?credit$",
    ],
    "confidence_score": [
        r"^confidence$", r"^confidence_?score$",
        r"^model_?confidence$",
    ],
}


# ============================================================
# CORE NORMALIZATION LOGIC
# ============================================================

def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize dataframe column names into canonical roles expected by models.

    This function:
    - Uses deterministic regex + synonym matching
    - Replaces column names in-place (returns a copy)
    - Detects ambiguity and surfaces it (does NOT guess)
    - Does NOT inspect values
    - Does NOT run models

    Returns:
        normalized_df: pd.DataFrame
        metadata: dict with mapping details and review flags
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("normalize_columns expects a pandas DataFrame")

    original_columns = list(df.columns)
    normalized_df = df.copy()

    column_mapping: Dict[str, str] = {}
    reverse_mapping: Dict[str, List[str]] = {}
    ambiguity: Dict[str, List[str]] = {}

    for col in original_columns:
        col_l = col.strip().lower()
        matched_roles = []

        for role, patterns in ROLE_SYNONYMS.items():
            for pattern in patterns:
                if re.match(pattern, col_l):
                    matched_roles.append(role)
                    break

        if len(matched_roles) == 1:
            role = matched_roles[0]
            column_mapping[col] = role
            reverse_mapping.setdefault(role, []).append(col)

        elif len(matched_roles) > 1:
            ambiguity[col] = matched_roles

    # Detect multiple columns mapping to same role
    conflicting_roles = {
        role: cols
        for role, cols in reverse_mapping.items()
        if len(cols) > 1
    }

    needs_review = bool(ambiguity or conflicting_roles)

    # Apply renaming only if safe
    if not needs_review:
        normalized_df = normalized_df.rename(columns=column_mapping)

    metadata = {
        "original_columns": original_columns,
        "column_mapping": column_mapping,
        "ambiguous_columns": ambiguity,
        "conflicting_roles": conflicting_roles,
        "needs_review": needs_review,
    }

    return normalized_df, metadata


# ============================================================
# HELPER (OPTIONAL, FOR ROUTER SAFETY)
# ============================================================

def get_detected_roles(df: pd.DataFrame) -> List[str]:
    """
    Lightweight helper to list detected canonical roles
    without mutating the dataframe.
    """
    roles = set()
    for col in df.columns:
        col_l = col.lower()
        for role, patterns in ROLE_SYNONYMS.items():
            if any(re.match(p, col_l) for p in patterns):
                roles.add(role)
                break
    return sorted(roles)
