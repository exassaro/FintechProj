# routing/schema_router.py

from __future__ import annotations

from typing import Dict

import pandas as pd


def _has_column(df: pd.DataFrame, name: str) -> bool:
    """Return True if the DataFrame has the given column name."""
    return name in df.columns


def _detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Infer a high-level dataset type from the schema.

    This does not call any models and is purely column-based.
    """
    has_description = _has_column(df, "description")
    has_vendor = _has_column(df, "vendor_name")
    has_amount = _has_column(df, "amount")
    has_hsn = _has_column(df, "hsn_sac_code")

    # HSN/SAC-first datasets
    if has_hsn:
        if has_amount:
            return "hsn_sac_with_amount"
        return "hsn_sac_only"

    # Pure numeric (no text)
    if not has_description and has_amount:
        return "numeric_only"

    # Text-based datasets
    if has_description and has_vendor and has_amount:
        return "text_rich_with_amount"
    if has_description and has_amount and not has_vendor:
        return "text_only_with_amount"
    if has_description and not has_amount:
        # Text-only, no amount (e.g. classification-only use cases)
        if has_vendor:
            return "text_rich_no_amount"
        return "text_only_no_amount"

    # Fallback when schema is incomplete or ambiguous
    return "unknown"


def _classification_route_for(dataset_type: str) -> str:
    """
    Map dataset_type to a classification route.

    Routes:
      - "nlp": description-based classifier (with or without vendor)
      - "numeric": numeric-only classifier
      - "hsn_sac": deterministic rule-based HSN/SAC engine
    """
    if dataset_type.startswith("hsn_sac"):
        return "hsn_sac"
    if dataset_type.startswith("numeric"):
        return "numeric"
    if dataset_type.startswith("text"):
        return "nlp"
    # Unknown schema: no classification route decided
    return "unknown"


def _anomaly_flags_for(dataset_type: str) -> Dict[str, bool]:
    """
    Decide which anomaly detectors should be enabled for a given dataset type.

    Rules (from locked design):
      - Text (NLP) datasets:
          * NLP anomaly: enabled
          * Numeric anomaly: enabled only if amount exists
          * HSN/SAC anomaly: disabled
      - Numeric-only datasets:
          * Numeric anomaly: enabled
          * NLP anomaly: disabled
          * HSN/SAC anomaly: disabled
      - HSN/SAC datasets:
          * HSN/SAC anomaly: enabled
          * Numeric anomaly: enabled only if amount exists
          * NLP anomaly: disabled
    """
    enable_nlp = False
    enable_numeric = False
    enable_hsn = False

    if dataset_type.startswith("text"):
        enable_nlp = True
        if "with_amount" in dataset_type:
            enable_numeric = True

    elif dataset_type.startswith("numeric"):
        enable_numeric = True

    elif dataset_type.startswith("hsn_sac"):
        enable_hsn = True
        if "with_amount" in dataset_type:
            enable_numeric = True

    return {
        "enable_nlp_anomaly": enable_nlp,
        "enable_numeric_anomaly": enable_numeric,
        "enable_hsn_anomaly": enable_hsn,
    }


def _reason_for(dataset_type: str) -> str:
    """
    Provide a human-readable reason for the routing decision.

    This is purely descriptive and must remain deterministic.
    """
    if dataset_type == "text_rich_with_amount":
        return "Description and vendor_name present with amount column; route via NLP with numeric and NLP anomalies"
    if dataset_type == "text_only_with_amount":
        return "Description present with amount column; route via NLP with numeric and NLP anomalies"
    if dataset_type == "text_rich_no_amount":
        return "Description and vendor_name present without amount; route via NLP with NLP anomaly only"
    if dataset_type == "text_only_no_amount":
        return "Description present without amount; route via NLP with NLP anomaly only"
    if dataset_type == "numeric_only":
        return "No text fields; numeric-only dataset; route via numeric classifier with numeric anomaly only"
    if dataset_type == "hsn_sac_with_amount":
        return "HSN/SAC codes present with amount; route via HSN/SAC rule engine with HSN anomaly and numeric anomaly"
    if dataset_type == "hsn_sac_only":
        return "HSN/SAC codes present without amount; route via HSN/SAC rule engine with HSN anomaly only"
    return "Schema does not match known patterns; routing marked as unknown"


def detect_schema(df: pd.DataFrame) -> Dict[str, object]:
    """
    Detect dataset schema characteristics and return a routing decision.

    This function:
      - Does not call any models.
      - Does not mutate the input DataFrame.
      - Is fully deterministic and based only on column presence.
      - Is designed to be unit-test and FastAPI friendly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. Only column names are inspected; values are not used.

    Returns
    -------
    dict
        Routing decision with keys:
          - dataset_type: str
          - classification_route: str ("nlp", "numeric", "hsn_sac", or "unknown")
          - enable_nlp_anomaly: bool
          - enable_numeric_anomaly: bool
          - enable_hsn_anomaly: bool
          - reason: str (human-readable explanation)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("detect_schema expects a pandas DataFrame")

    dataset_type = _detect_dataset_type(df)
    classification_route = _classification_route_for(dataset_type)
    anomaly_flags = _anomaly_flags_for(dataset_type)
    reason = _reason_for(dataset_type)

    decision: Dict[str, object] = {
        "dataset_type": dataset_type,
        "classification_route": classification_route,
        **anomaly_flags,
        "reason": reason,
    }
    return decision
