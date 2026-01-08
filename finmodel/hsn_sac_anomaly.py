# anomaly/hsn_sac_anomaly.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import json
import pandas as pd


@dataclass
class HsnSacAnomalyConfig:
    """
    Configuration for HSN/SAC anomaly detection.

    All thresholds are rule-based and deterministic.
    """

    # Required columns
    required_columns: Tuple[str, ...] = (
        "transaction_id",
        "hsn_sac_code",
        "predicted_category",
        "gst_slab",
        "itc_eligible",
        "confidence_score",
    )

    # Confidence thresholds
    low_confidence_threshold: float = 0.4  # very low confidence
    high_confidence_threshold: float = 0.8  # used for some consistency checks

    # Scoring levels
    score_invalid_code: float = 1.0
    score_unknown_code: float = 0.9
    score_category_mismatch: float = 0.7
    score_gst_slab_inconsistency: float = 0.7
    score_itc_violation_strong: float = 0.8
    score_itc_violation_mild: float = 0.5
    score_structural_violation: float = 0.6
    score_low_conf_risk: float = 0.4
    score_vendor_inconsistency: float = 0.3

    # Allowed GST slab mismatch tolerance (if GST slab in rule is a single value)
    allow_gst_slab_null_as_unknown: bool = True

    # Vendor consistency (optional)
    min_vendor_category_samples: int = 20
    vendor_inconsistent_ratio_threshold: float = 0.2  # fraction of non-dominant codes to trigger anomaly

    # Paths / rule behavior
    hsn_sac_rules_path: Optional[str] = None  # if provided, used to validate codes


class HsnSacAnomalyDetector:
    """
    Rule-first, deterministic HSN/SAC anomaly detector.

    Focus:
      - Regulatory / logical anomalies around HSN/SAC codes:
        * Invalid/unknown codes
        * HSN/SAC vs category mismatch
        * GST slab inconsistency
        * ITC eligibility violations
        * Structural HSN vs SAC misuse
        * Low-confidence regulatory risk
        * Vendor-level HSN/SAC inconsistency (optional)
    Explicitly does NOT:
      - Inspect numeric amounts
      - Duplicate NumericAnomalyDetector behavior
    """

    def __init__(self, config: Optional[HsnSacAnomalyConfig] = None) -> None:
        self.config = config or HsnSacAnomalyConfig()
        self._rules: Optional[Dict[str, Any]] = None
        self._hsn_rules: Dict[str, Dict[str, Any]] = {}
        self._sac_rules: Dict[str, Dict[str, Any]] = {}
        self._category_map: Dict[str, Dict[str, Any]] = {}
        self._code_type_hint: Dict[str, str] = {}  # HSN/SAC hint per code
        if self.config.hsn_sac_rules_path:
            self._load_rules(self.config.hsn_sac_rules_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run HSN/SAC anomaly detection on the given DataFrame.

        Soft-fail behavior:
          - If input is not a DataFrame: return input as-is.
          - If required columns are missing: return copy with zero anomaly flags/scores.
        """
        if not isinstance(df, pd.DataFrame):
            return df

        missing = set(self.config.required_columns) - set(df.columns)
        if missing:
            out = df.copy()
            n = len(out)
            out["hsn_anomaly_score"] = 0.0
            out["hsn_anomaly_flag"] = False
            out["hsn_anomaly_reason"] = ["" for _ in range(n)]
            return out

        out = df.copy()
        n = len(out)
        scores = [0.0] * n
        reasons: list[list[str]] = [[] for _ in range(n)]

        # Load rules lazily if path is configured
        if self._rules is None and self.config.hsn_sac_rules_path:
            self._load_rules(self.config.hsn_sac_rules_path)

        # Normalize some fields
        out["hsn_sac_code"] = out["hsn_sac_code"].astype(str).str.strip()
        out["predicted_category"] = out["predicted_category"].astype(str).str.strip()
        out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce")

        # Per-row checks
        for idx, row in out.iterrows():
            row_index = out.index.get_loc(idx)
            self._apply_row_rules(row, row_index, scores, reasons)

        # Vendor-level consistency (optional)
        if "vendor_id" in out.columns:
            self._apply_vendor_consistency(out, scores, reasons)

        # Build final columns
        final_reasons = []
        for r in reasons:
            # Stable, unique reasons
            r_unique = list(dict.fromkeys(r))
            final_reasons.append("; ".join(r_unique))

        out["hsn_anomaly_score"] = scores
        out["hsn_anomaly_flag"] = [s > 0.0 for s in scores]
        out["hsn_anomaly_reason"] = final_reasons

        return out

    # ------------------------------------------------------------------
    # Rule loading and helpers
    # ------------------------------------------------------------------

    def _load_rules(self, path: str) -> None:
        """
        Load HSN/SAC rules JSON.

        Expected structure aligned with hsn_sac.json used by tax engine:
          - hsn_rules: {code -> {gst_slab, tax_category, itc_eligible, ...}}
          - sac_rules: {code -> {gst_slab, tax_category, itc_eligible, ...}}
        """
        with open(path, "r", encoding="utf-8") as f:
            self._rules = json.load(f)

        self._hsn_rules = self._rules.get("hsn_rules", {})
        self._sac_rules = self._rules.get("sac_rules", {})

        # Build a simple category map for HSN/SAC → expected tax_category/itc/gst
        self._category_map = {}
        for code, rule in self._hsn_rules.items():
            cat = str(rule.get("tax_category", "")).strip()
            if not cat:
                continue
            self._category_map.setdefault(code, {})
            self._category_map[code]["tax_category"] = cat
            self._category_map[code]["gst_slab"] = rule.get("gst_slab")
            self._category_map[code]["itc_eligible"] = rule.get("itc_eligible")
            self._code_type_hint[code] = "HSN"

        for code, rule in self._sac_rules.items():
            cat = str(rule.get("tax_category", "")).strip()
            if not cat:
                continue
            self._category_map.setdefault(code, {})
            self._category_map[code]["tax_category"] = cat
            self._category_map[code]["gst_slab"] = rule.get("gst_slab")
            self._category_map[code]["itc_eligible"] = rule.get("itc_eligible")
            self._code_type_hint[code] = "SAC"

    def _lookup_code_rule(self, code: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Return (rule_dict, code_type_hint) for a given code, if present.
        """
        if self._rules is None:
            return None, None
        if code in self._hsn_rules:
            return self._hsn_rules[code], "HSN"
        if code in self._sac_rules:
            return self._sac_rules[code], "SAC"
        return None, None

    # ------------------------------------------------------------------
    # Per-row rules
    # ------------------------------------------------------------------

    def _apply_row_rules(
        self,
        row: pd.Series,
        pos: int,
        scores: list[float],
        reasons: list[list[str]],
    ) -> None:
        """
        Apply all per-row HSN/SAC anomaly rules to a single row.
        """
        code = str(row.get("hsn_sac_code", "")).strip()
        category = str(row.get("predicted_category", "")).strip()
        gst_slab = row.get("gst_slab")
        itc_eligible = row.get("itc_eligible")
        conf = row.get("confidence_score")
        vendor_id = row.get("vendor_id", None)

        rule, code_type_hint = self._lookup_code_rule(code)

        # 1) Invalid / Unknown HSN/SAC codes
        # If rules are loaded and code is not found, treat as unknown code
        if self._rules is not None:
            if not code or rule is None:
                # Unknown or missing code
                scores[pos] = max(scores[pos], self.config.score_unknown_code)
                reasons[pos].append("Unknown or missing HSN/SAC code in master reference")
                # Further checks are not meaningful for unknown code
                return

        # From this point, rule should be present if rules file is configured
        # If rules not configured, skip code-based checks and focus on generic ones
        expected_cat = None
        expected_gst = None
        expected_itc = None

        if rule is not None:
            expected_cat = str(rule.get("tax_category", "")).strip()
            expected_gst = rule.get("gst_slab")
            expected_itc = rule.get("itc_eligible")

        # 2) HSN/SAC vs Category mismatch
        if expected_cat and category:
            if category.lower() != expected_cat.lower():
                scores[pos] = max(scores[pos], self.config.score_category_mismatch)
                reasons[pos].append(
                    f"HSN/SAC vs category mismatch: code maps to '{expected_cat}', but predicted '{category}'"
                )

        # 3) GST slab inconsistency
        if expected_gst is not None:
            try:
                tx_slab = float(gst_slab)
            except (TypeError, ValueError):
                tx_slab = None
            if tx_slab is not None and tx_slab != float(expected_gst):
                scores[pos] = max(scores[pos], self.config.score_gst_slab_inconsistency)
                reasons[pos].append(
                    f"GST slab inconsistency for code {code}: expected {expected_gst}, got {gst_slab}"
                )
        else:
            # If rules know nothing about slab, optionally treat as unknown
            if self.config.allow_gst_slab_null_as_unknown and rule is not None:
                # No explicit expected slab; no inconsistency flag
                pass

        # 4) ITC eligibility violations
        if expected_itc is not None:
            tx_itc = None
            if isinstance(itc_eligible, bool):
                tx_itc = itc_eligible
            elif isinstance(itc_eligible, str):
                tx_itc = itc_eligible.strip().lower() in {"true", "1", "yes", "y"}

            # Strong violation: rule says blocked (False) but transaction says eligible
            if expected_itc is False and tx_itc is True:
                scores[pos] = max(scores[pos], self.config.score_itc_violation_strong)
                reasons[pos].append(
                    f"ITC violation: code {code} is non-eligible but transaction marked eligible"
                )
            # Mild violation: rule says eligible but transaction says not
            elif expected_itc is True and tx_itc is False:
                scores[pos] = max(scores[pos], self.config.score_itc_violation_mild)
                reasons[pos].append(
                    f"ITC potential underclaim: code {code} allows ITC but transaction marked non-eligible"
                )

        # 5) HSN vs SAC structural violations
        # Use code_type_hint and observed category semantics (very coarse)
        if code_type_hint is not None and category:
            cat_lower = category.lower()
            is_goods_like = any(k in cat_lower for k in ["goods", "motor", "vehicle", "product", "pharma", "grain"])
            is_service_like = any(k in cat_lower for k in ["services", "service", "consult", "professional", "transport"])
            if code_type_hint == "SAC" and is_goods_like:
                scores[pos] = max(scores[pos], self.config.score_structural_violation)
                reasons[pos].append(
                    f"Structural violation: SAC code {code} used for goods-like category '{category}'"
                )
            if code_type_hint == "HSN" and is_service_like:
                scores[pos] = max(scores[pos], self.config.score_structural_violation)
                reasons[pos].append(
                    f"Structural violation: HSN code {code} used for service-like category '{category}'"
                )

        # 6) Low confidence regulatory risk (no amount-based logic here)
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            conf_val = None

        if conf_val is not None and conf_val < self.config.low_confidence_threshold:
            # Only attach regulatory risk reason if other anomalies exist or if rule present
            scores[pos] = max(scores[pos], self.config.score_low_conf_risk)
            reasons[pos].append(
                f"Low confidence HSN/SAC classification (confidence={conf_val:.2f})"
            )

    # ------------------------------------------------------------------
    # Vendor-level consistency
    # ------------------------------------------------------------------

    def _apply_vendor_consistency(
        self,
        df: pd.DataFrame,
        scores: list[float],
        reasons: list[list[str]],
    ) -> None:
        """
        Vendor-level HSN/SAC inconsistency:

        For each (vendor_id, predicted_category), if there is a dominant HSN/SAC code
        historically and a significant fraction of rows deviate from it, mark those
        deviating rows as anomalies (soft signal).
        """
        vendor_col = df["vendor_id"].astype(str).str.strip()
        cat_col = df["predicted_category"].astype(str).str.strip()
        code_col = df["hsn_sac_code"].astype(str).str.strip()

        tmp = df.copy()
        tmp["vendor_id"] = vendor_col
        tmp["predicted_category"] = cat_col
        tmp["hsn_sac_code"] = code_col

        grouped = tmp.groupby(["vendor_id", "predicted_category"])
        for (vendor_id, cat), g in grouped:
            if len(g) < self.config.min_vendor_category_samples:
                continue

            code_counts = g["hsn_sac_code"].value_counts()
            if code_counts.empty:
                continue

            dominant_code = code_counts.index[0]
            dominant_count = code_counts.iloc[0]
            total = len(g)
            non_dominant = total - dominant_count
            if total <= 0:
                continue

            ratio = non_dominant / float(total)
            if ratio < self.config.vendor_inconsistent_ratio_threshold:
                continue

            # Flag all non-dominant-code rows as vendor inconsistency
            mask = g["hsn_sac_code"] != dominant_code
            for idx in g[mask].index:
                pos = df.index.get_loc(idx)
                scores[pos] = max(scores[pos], self.config.score_vendor_inconsistency)
                reasons[pos].append(
                    f"Vendor-level HSN/SAC inconsistency: vendor {vendor_id} "
                    f"uses multiple codes for category '{cat}'"
                )

    # ------------------------------------------------------------------
    # Drift design placeholder
    # ------------------------------------------------------------------

    def describe_regulatory_drift_design(self) -> str:
        """
        Design-only placeholder for future regulatory drift detection.

        Possible approaches (not implemented):
          - Monitor changes in HSN/SAC usage patterns across categories over time.
          - Compare GST slab distributions per HSN/SAC code across periods.
          - Track shifts in ITC eligibility patterns for the same codes.
          - Use rule-version metadata (from hsn_sac.json) to trigger revalidation
            when GST Council updates HSN/SAC or ITC provisions.

        No side effects or DataFrame mutation.
        """
        return (
            "Regulatory drift design: monitor HSN/SAC usage, GST slab, and ITC eligibility "
            "patterns over time and compare against rule versions; raise alerts when "
            "patterns deviate from historical norms or when configuration versions change."
        )
