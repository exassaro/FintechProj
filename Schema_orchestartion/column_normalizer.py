# ingestion/column_normalizer.py

from __future__ import annotations

from typing import Dict, Tuple

import difflib
import pandas as pd


# Canonical column names used by the system
_CANONICAL_COLUMNS = {
    "description",
    "vendor_name",
    "amount",
    "transaction_date",
    "hsn_sac_code",
    "gst_slab",
    "itc_eligible",
    "confidence_score",
}

# Synonym dictionary, derived from notebook patterns and common finance schemas.
# Keys are lowercase synonyms, values are canonical column names.
_SYNONYMS: Dict[str, str] = {
    # description
    "description": "description",
    "txn_description": "description",
    "transaction_description": "description",
    "narration": "description",
    "details": "description",
    "remarks": "description",
    "txn_desc": "description",
    "desc": "description",
    "transactiondetails": "description",
    "transaction_detail": "description",
    "particulars": "description",
    # vendor / merchant
    "vendor_name": "vendor_name",
    "vendor": "vendor_name",
    "merchant": "vendor_name",
    "payee": "vendor_name",
    "party_name": "vendor_name",
    "counterparty": "vendor_name",
    "beneficiary": "vendor_name",
    "merchant_name": "vendor_name",
    "supplier": "vendor_name",
    "customer": "vendor_name",
    # amount
    "amount": "amount",
    "txn_amount": "amount",
    "transaction_amount": "amount",
    "amt": "amount",
    "debit": "amount",
    "credit": "amount",
    "value": "amount",
    "net_amount": "amount",
    "gross_amount": "amount",
    "invoice_amount": "amount",
    # transaction_date
    "transaction_date": "transaction_date",
    "txn_date": "transaction_date",
    "date": "transaction_date",
    "posting_date": "transaction_date",
    "value_date": "transaction_date",
    "booking_date": "transaction_date",
    "doc_date": "transaction_date",
    "invoice_date": "transaction_date",
    # hsn_sac_code
    "hsn_sac_code": "hsn_sac_code",
    "hsn": "hsn_sac_code",
    "hsn_code": "hsn_sac_code",
    "sac": "hsn_sac_code",
    "sac_code": "hsn_sac_code",
    "hsn_sac": "hsn_sac_code",
    "tax_code": "hsn_sac_code",
    # gst_slab
    "gst_slab": "gst_slab",
    "gst_rate": "gst_slab",
    "gst_percentage": "gst_slab",
    "gst%": "gst_slab",
    "tax_rate": "gst_slab",
    "tax_percentage": "gst_slab",
    # itc_eligible
    "itc_eligible": "itc_eligible",
    "eligible_for_itc": "itc_eligible",
    "itc_flag": "itc_eligible",
    "itc": "itc_eligible",
    "input_tax_credit": "itc_eligible",
    # confidence_score
    "confidence_score": "confidence_score",
    "model_confidence": "confidence_score",
    "prediction_confidence": "confidence_score",
    "score": "confidence_score",
    "probability": "confidence_score",
}


def _normalize_name(name: str) -> str:
    """
    Normalize a column name for comparison:
      - strip whitespace
      - lowercase
      - replace spaces and common separators with underscore
    """
    n = name.strip().lower()
    for ch in (" ", "-", ".", "/"):
        n = n.replace(ch, "_")
    return n


def _exact_or_case_insensitive_match(col: str) -> str | None:
    """
    Check for exact and case-insensitive matches against canonical columns.
    Returns the canonical name if matched, else None.
    """
    # Exact match
    if col in _CANONICAL_COLUMNS:
        return col

    # Case-insensitive match
    lower_col = col.lower()
    for canon in _CANONICAL_COLUMNS:
        if lower_col == canon.lower():
            return canon

    return None


def _synonym_match(col: str) -> str | None:
    """
    Check against known synonyms dictionary.
    Returns canonical column name if found, else None.
    """
    normalized = _normalize_name(col)
    return _SYNONYMS.get(normalized)


def _fuzzy_match(col: str, threshold: float = 0.8) -> str | None:
    """
    Fuzzy string matching against canonical column names.

    Uses simple difflib.SequenceMatcher ratio; no ML.
    Returns canonical column name if best match exceeds threshold, else None.
    """
    normalized = _normalize_name(col)
    best_canon = None
    best_score = 0.0
    for canon in _CANONICAL_COLUMNS:
        score = difflib.SequenceMatcher(None, normalized, canon).ratio()
        if score > best_score:
            best_score = score
            best_canon = canon
    if best_canon is not None and best_score >= threshold:
        return best_canon
    return None


def _semantic_match_placeholder(col: str) -> str | None:
    """
    Placeholder for future semantic similarity.

    This intentionally does nothing and always returns None.
    No ML models or embeddings are used.
    """
    _ = col  # unused
    return None


def _resolve_canonical_name(col: str, already_mapped: set[str]) -> str | None:
    """
    Resolve a single source column to a canonical name using the following priority:
      1. Exact / case-insensitive match
      2. Synonym dictionary
      3. Fuzzy match
      4. Semantic placeholder (currently no-op)

    Ensures that a canonical column is mapped at most once.
    """
    # 1. Exact / case-insensitive
    target = _exact_or_case_insensitive_match(col)
    if target and target not in already_mapped:
        return target

    # 2. Synonym dictionary
    target = _synonym_match(col)
    if target and target not in already_mapped:
        return target

    # 3. Fuzzy matching (last resort)
    target = _fuzzy_match(col)
    if target and target not in already_mapped:
        return target

    # 4. Semantic placeholder (no-op for now)
    target = _semantic_match_placeholder(col)
    if target and target not in already_mapped:
        return target

    return None


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize DataFrame column names to canonical names used by the system.

    Responsibilities:
      - Only rename columns; never modify data values or drop rows.
      - Use deterministic matching strategies:
          1. Exact/case-insensitive match
          2. Synonym dictionary
          3. Fuzzy string similarity (difflib ratio)
          4. Semantic similarity placeholder (no-op)
      - Avoid ambiguous mappings: each canonical column is assigned at most once.
      - Unmatched columns are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with arbitrary column names.

    Returns
    -------
    normalized_df : pd.DataFrame
        DataFrame with normalized column names where matches were found.
    mapping_report : dict
        Mapping from original column names to canonical names, e.g.:
        {
            "txn_desc": "description",
            "merchant": "vendor_name",
            "amt": "amount",
        }
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("normalize_columns expects a pandas DataFrame")

    original_columns = list(df.columns)
    mapping: Dict[str, str] = {}
    used_canonical: set[str] = set()

    # Determine mapping for each original column name
    for col in original_columns:
        target = _resolve_canonical_name(str(col), already_mapped=used_canonical)
        if target is not None:
            mapping[col] = target
            used_canonical.add(target)

    # Apply renaming; do not mutate original DataFrame
    normalized_df = df.rename(columns=mapping, copy=True)

    return normalized_df, mapping
