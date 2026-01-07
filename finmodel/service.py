"""
service.py

Deterministic rule-based GST engine for datasets containing only HSN/SAC codes.
No ML, no API, no DB. Pure Python + Pandas, ready to be wrapped by FastAPI later.

Public entry point:
    apply_hsn_sac_rules(df: pd.DataFrame, rules_path: str) -> pd.DataFrame
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd


class RulesConfigError(Exception):
    """Raised when the HSN/SAC rules JSON is invalid or inconsistent."""


class DataValidationError(Exception):
    """Raised when the input DataFrame fails mandatory validation checks."""


def _load_rules(rules_path: str) -> Dict[str, Any]:
    """
    Load and validate the HSN/SAC rules configuration JSON.

    Raises:
        RulesConfigError: if JSON cannot be parsed or mandatory sections are missing.
    """
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except Exception as exc:
        raise RulesConfigError(f"Failed to load rules JSON: {exc}") from exc

    # Basic structural checks for mandatory sections
    for key in ("metadata", "hsn_rules", "sac_rules", "default_rules", "validation_rules"):
        if key not in rules:
            raise RulesConfigError(f"Missing top-level key in rules JSON: '{key}'")

    # Validate allowed slabs presence
    validation_rules = rules.get("validation_rules", {})
    allowed_slabs = validation_rules.get("allowed_gst_slabs")
    if not isinstance(allowed_slabs, list) or not allowed_slabs:
        raise RulesConfigError("validation_rules.allowed_gst_slabs must be a non-empty list")

    return rules


def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure the input DataFrame contains all mandatory columns.

    Raises:
        DataValidationError: if any required column is missing.
    """
    required_cols = {"transaction_id", "transaction_date", "hsn_sac_code", "code_type", "taxable_amount"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns in input DataFrame: {sorted(missing)}")


def _get_validation_patterns(rules: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract HSN and SAC validation config (lengths + regex patterns) from rules JSON.
    """
    vr = rules.get("validation_rules", {})
    hsn_cfg = vr.get("hsn_code", {})
    sac_cfg = vr.get("sac_code", {})
    return hsn_cfg, sac_cfg


def _is_valid_code_format(code: str, code_type: str, rules: Dict[str, Any]) -> bool:
    """
    Validate the format of a single HSN or SAC code using validation_rules.

    Args:
        code: Raw code string from transaction.
        code_type: 'HSN' or 'SAC'
        rules: Parsed rules JSON.

    Returns:
        True if code passes validation, False otherwise.
    """
    if not isinstance(code, str) or not code.strip():
        return False

    code = code.strip()
    hsn_cfg, sac_cfg = _get_validation_patterns(rules)

    if code_type.upper() == "HSN":
        allowed_lengths = hsn_cfg.get("allowed_lengths", [])
        pattern = hsn_cfg.get("pattern")
    elif code_type.upper() == "SAC":
        allowed_lengths = sac_cfg.get("allowed_lengths", [])
        pattern = sac_cfg.get("pattern")
    else:
        # Unknown code_type is invalid by design
        return False

    if allowed_lengths and len(code) not in allowed_lengths:
        return False

    if pattern:
        if not re.match(pattern, code):
            return False

    return True


def _lookup_rule(
    code: str,
    code_type: str,
    rules: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Look up the concrete rule for a given HSN/SAC code.

    Args:
        code: Normalized code string.
        code_type: 'HSN' or 'SAC'.
        rules: Parsed rules JSON.

    Returns:
        (rule_dict_or_none, rule_applied_label)

        rule_dict_or_none:
            - dict from hsn_rules/sac_rules when found
            - None if not found (default rules must be used)
        rule_applied_label:
            - e.g. 'HSN_8703', 'SAC_9983', or 'DEFAULT_UNKNOWN', etc.
    """
    code = code.strip()
    code_type_u = code_type.upper()

    if code_type_u == "HSN":
        table = rules.get("hsn_rules", {})
        rule = table.get(code)
        if rule is not None:
            return rule, f"HSN_{code}"
        return None, "DEFAULT_UNKNOWN"

    if code_type_u == "SAC":
        table = rules.get("sac_rules", {})
        rule = table.get(code)
        if rule is not None:
            return rule, f"SAC_{code}"
        return None, "DEFAULT_UNKNOWN"

    # Unknown code_type should not happen if validation is done before;
    # treat as invalid -> default invalid_code
    return None, "DEFAULT_INVALID_CODE_TYPE"


def _apply_default_rule(
    default_key: str,
    rules: Dict[str, Any],
) -> Tuple[Optional[float], Optional[bool], bool, str]:
    """
    Apply one of the default rules: unknown_code, missing_code, invalid_code.

    Args:
        default_key: key inside default_rules ('unknown_code', 'missing_code', 'invalid_code').
        rules: Parsed rules JSON.

    Returns:
        (gst_slab_or_none, itc_eligible_or_none, needs_review, rule_applied_label)
    """
    default_rules = rules.get("default_rules", {})
    dr = default_rules.get(default_key, {})

    gst_slab = dr.get("default_gst_slab")
    itc_eligible = dr.get("default_itc_eligible")
    force_human_review = bool(dr.get("force_human_review", True))
    rule_label = f"DEFAULT_{default_key.upper()}"

    return gst_slab, itc_eligible, force_human_review, rule_label


def _get_confidence_score(code_present: bool, valid_and_known: bool, rules: Dict[str, Any]) -> float:
    """
    Derive a deterministic confidence_score based on confidence_policy in rules.

    - Typically 1.0 when a known HSN/SAC rule is applied.
    - 0.0 when default unknown/invalid rules are applied.

    Args:
        code_present: True if a non-empty code is present.
        valid_and_known: True if code format is valid and rule is found.
        rules: Parsed rules JSON.

    Returns:
        Confidence score as a float between 0.0 and 1.0.
    """
    policy = rules.get("confidence_policy", {})
    if valid_and_known:
        return float(policy.get("hsn_sac_present", 1.0))
    if not code_present:
        return float(policy.get("missing_code", 0.0))
    return float(policy.get("unknown_or_invalid_code", 0.0))


def _compute_gst_amount(taxable_amount: Any, gst_slab: Optional[float]) -> float:
    """
    Compute GST amount given taxable_amount and gst_slab.

    If gst_slab is None or taxable_amount is not numeric, return 0.0.

    Note: This engine only computes total GST; splitting into CGST/SGST/IGST
    is handled by higher-level tax engine if required.
    """
    try:
        amt = float(taxable_amount)
    except (TypeError, ValueError):
        return 0.0

    if gst_slab is None:
        return 0.0

    return round(amt * float(gst_slab) / 100.0, 2)


def apply_hsn_sac_rules(df: pd.DataFrame, rules_path: str) -> pd.DataFrame:
    """
    Apply deterministic HSN/SAC GST rules to a transaction DataFrame.

    Args:
        df: Pandas DataFrame with at least:
            - transaction_id
            - transaction_date
            - hsn_sac_code
            - code_type ('HSN' or 'SAC')
            - taxable_amount
        rules_path: Path to hsn_sac_rules.json configuration file.

    Returns:
        A new DataFrame with enriched GST fields, preserving original columns:
            - gst_slab
            - gst_amount
            - tax_category
            - itc_eligible
            - confidence_score
            - needs_review
            - rule_applied

    Raises:
        RulesConfigError: on invalid JSON or missing rule sections.
        DataValidationError: on missing DataFrame columns.
    """
    rules = _load_rules(rules_path)
    _validate_required_columns(df)

    # Copy to avoid mutating caller's DataFrame
    out_df = df.copy()

    # Prepare columns for outputs
    out_df["gst_slab"] = None
    out_df["gst_amount"] = 0.0
    out_df["tax_category"] = None
    out_df["itc_eligible"] = None
    out_df["confidence_score"] = 0.0
    out_df["needs_review"] = False
    out_df["rule_applied"] = None

    for idx, row in out_df.iterrows():
        raw_code = row.get("hsn_sac_code")
        code_type = row.get("code_type")
        taxable_amount = row.get("taxable_amount")

        code_present = isinstance(raw_code, str) and raw_code.strip() != ""
        code_type_present = isinstance(code_type, str) and code_type.strip() != ""

        # Default values for this row
        gst_slab: Optional[float] = None
        tax_category: Optional[str] = None
        itc_eligible: Optional[bool] = None
        confidence_score: float = 0.0
        needs_review: bool = False
        rule_applied: str = ""

        if not code_type_present:
            # Missing code_type: treat as invalid_code
            gst_slab, itc_eligible, needs_review, rule_applied = _apply_default_rule(
                "invalid_code", rules
            )
            confidence_score = _get_confidence_score(code_present=False, valid_and_known=False, rules=rules)

        elif not code_present:
            # Code missing but type present: missing_code default
            gst_slab, itc_eligible, needs_review, rule_applied = _apply_default_rule(
                "missing_code", rules
            )
            confidence_score = _get_confidence_score(code_present=False, valid_and_known=False, rules=rules)

        else:
            # We have a code and code_type; validate format
            normalized_code = str(raw_code).strip()
            code_type_u = str(code_type).upper()

            is_valid_format = _is_valid_code_format(normalized_code, code_type_u, rules)

            if not is_valid_format:
                # invalid_code default
                gst_slab, itc_eligible, needs_review, rule_applied = _apply_default_rule(
                    "invalid_code", rules
                )
                confidence_score = _get_confidence_score(
                    code_present=True, valid_and_known=False, rules=rules
                )
            else:
                # Try to find concrete rule
                rule, concrete_label = _lookup_rule(normalized_code, code_type_u, rules)

                if rule is None:
                    # unknown_code default
                    gst_slab, itc_eligible, needs_review, rule_applied = _apply_default_rule(
                        "unknown_code", rules
                    )
                    confidence_score = _get_confidence_score(
                        code_present=True, valid_and_known=False, rules=rules
                    )
                else:
                    # Known rule: apply deterministic GST fields
                    gst_slab = rule.get("gst_slab")
                    # For HSN we use 'tax_category'; for SAC we may also use 'tax_category' or 'service_category'
                    tax_category = (
                        rule.get("tax_category")
                        or rule.get("category_name")
                        or rule.get("service_category")
                    )
                    itc_eligible = rule.get("itc_eligible")
                    # ITC conditions are preserved in a separate column if caller wants to inspect them.
                    # For now, we keep them as-is and let higher-level modules decide how to store/use them.
                    # If you want, you can add: out_df["itc_conditions"] later.
                    rule_applied = concrete_label
                    confidence_score = _get_confidence_score(
                        code_present=True, valid_and_known=True, rules=rules
                    )
                    # For known rules, needs_review remains False by default; anomaly/HITL layers
                    # may still mark the transaction later based on external policies.

        # Compute GST amount from taxable_amount and gst_slab
        gst_amount = _compute_gst_amount(taxable_amount, gst_slab)

        # Assign back to DataFrame
        out_df.at[idx, "gst_slab"] = gst_slab
        out_df.at[idx, "gst_amount"] = gst_amount
        out_df.at[idx, "tax_category"] = tax_category
        out_df.at[idx, "itc_eligible"] = itc_eligible
        out_df.at[idx, "confidence_score"] = confidence_score
        out_df.at[idx, "needs_review"] = bool(needs_review)
        out_df.at[idx, "rule_applied"] = rule_applied

    return out_df


# ---------------------------------------------------------------------------
# Usage example (commented):
#
# import pandas as pd
#
# df = pd.read_csv("synthetic_gst_hsn_sac_only.csv")
# enriched_df = apply_hsn_sac_rules(df, rules_path="hsn_sac_rules.json")
# print(enriched_df.head())
#
# ---------------------------------------------------------------------------
