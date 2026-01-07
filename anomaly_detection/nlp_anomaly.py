# anomaly/nlp_anomaly.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class NLPAnomalyConfig:
    # -------- Baseline rules --------
    min_category_samples: int = 20
    iqr_threshold_mild: float = 1.5
    iqr_threshold_strong: float = 3.0

    category_amount_thresholds: Optional[Dict[str, Dict[str, float]]] = None

    rarity_min_vendor_count: int = 5
    rarity_min_vendor_category_count: int = 1

    # -------- ML-based (optional) --------
    enable_isolation_forest: bool = False
    enable_lof: bool = False

    min_rows_for_ml: int = 300
    anomaly_threshold: float = 0.5

    # Isolation Forest
    isolation_forest_contamination: float = 0.05
    random_state: int = 42

    # LOF
    lof_n_neighbors: int = 20


# ============================================================
# DETECTOR
# ============================================================

class NLPAnomalyDetector:
    """
    Deterministic, rule-first anomaly detector for NLP-based
    transaction classification outputs.
    """

    def __init__(self, **kwargs):
        self.config = NLPAnomalyConfig(**kwargs)
        self.category_amount_thresholds = self.config.category_amount_thresholds or {}

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        # Soft-fail: non DataFrame
        if not isinstance(df, pd.DataFrame):
            return df

        # Missing required columns → fail soft
        required = {
            "transaction_id",
            "transaction_date",
            "amount",
            "predicted_category",
            "confidence_score",
        }
        if not required.issubset(df.columns):
            return self._attach_output_columns(
                df.copy(),
                scores=np.zeros(len(df)),
                reasons=[""] * len(df),
                flags=np.zeros(len(df), dtype=bool),
            )

        if df.empty:
            return self._attach_output_columns(
                df.copy(),
                scores=np.zeros(0),
                reasons=[],
                flags=np.zeros(0, dtype=bool),
            )

        out_df = df.copy()
        n = len(out_df)

        baseline_scores = np.zeros(n, dtype=float)
        baseline_reasons = [[] for _ in range(n)]
        iso_scores = np.zeros(n, dtype=float)
        lof_scores = np.zeros(n, dtype=float)

        # ---------- Baseline rules ----------
        self._apply_amount_iqr_outliers(out_df, baseline_scores, baseline_reasons)
        self._apply_category_amount_thresholds(out_df, baseline_scores, baseline_reasons)
        self._apply_vendor_category_rarity(out_df, baseline_scores, baseline_reasons)

        # ---------- Optional ML ----------
        if _SKLEARN_AVAILABLE and len(out_df) >= self.config.min_rows_for_ml:
            if self.config.enable_isolation_forest:
                self._apply_isolation_forest(out_df, iso_scores)
            if self.config.enable_lof:
                self._apply_lof(out_df, lof_scores)

        # ---------- Merge ----------
        final_scores = np.maximum.reduce([baseline_scores, iso_scores, lof_scores])
        flags = final_scores >= float(self.config.anomaly_threshold)

        reasons = []
        for i in range(n):
            msgs = list(dict.fromkeys(baseline_reasons[i]))
            if iso_scores[i] > baseline_scores[i] and iso_scores[i] > 0:
                msgs.append("Isolation Forest detected numeric anomaly")
            if lof_scores[i] > baseline_scores[i] and lof_scores[i] > 0:
                msgs.append("LOF detected local density anomaly")
            reasons.append("; ".join(msgs) if msgs else "")

        return self._attach_output_columns(out_df, final_scores, reasons, flags)

    # --------------------------------------------------------
    # Output helper
    # --------------------------------------------------------
    def _attach_output_columns(
        self,
        df: pd.DataFrame,
        scores: np.ndarray,
        reasons: Any,
        flags: np.ndarray,
    ) -> pd.DataFrame:
        df["anomaly_flag"] = flags.astype(bool)
        df["anomaly_score"] = scores.astype(float)
        df["anomaly_reason"] = list(reasons)
        return df

    # ========================================================
    # BASELINE RULES
    # ========================================================

    def _apply_amount_iqr_outliers(self, df, scores, reasons):
        amounts = pd.to_numeric(df["amount"], errors="coerce")
        cats = df["predicted_category"].astype(str)

        valid = amounts.notna()
        for cat, idxs in df[valid].groupby(cats[valid]).groups.items():
            if len(idxs) < self.config.min_category_samples:
                continue

            vals = amounts.loc[idxs]
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue

            mild_low = q1 - self.config.iqr_threshold_mild * iqr
            mild_high = q3 + self.config.iqr_threshold_mild * iqr
            strong_low = q1 - self.config.iqr_threshold_strong * iqr
            strong_high = q3 + self.config.iqr_threshold_strong * iqr

            for i in idxs:
                v = amounts.iloc[i]
                if v < strong_low or v > strong_high:
                    scores[i] = max(scores[i], 0.8)
                    reasons[i].append(f"Strong amount outlier for category: {cat}")
                elif v < mild_low or v > mild_high:
                    scores[i] = max(scores[i], 0.4)
                    reasons[i].append(f"Mild amount outlier for category: {cat}")

    def _apply_category_amount_thresholds(self, df, scores, reasons):
        if not self.category_amount_thresholds:
            return

        amounts = pd.to_numeric(df["amount"], errors="coerce")
        cats = df["predicted_category"].astype(str)

        for i in range(len(df)):
            cfg = self.category_amount_thresholds.get(cats.iloc[i])
            if not cfg or pd.isna(amounts.iloc[i]):
                continue

            if amounts.iloc[i] > cfg.get("high_amount", float("inf")):
                scores[i] = max(scores[i], 0.5)
                reasons[i].append("Category–amount threshold exceeded")

    def _apply_vendor_category_rarity(self, df, scores, reasons):
        if "vendor_name" not in df.columns:
            return

        vendor = df["vendor_name"].astype(str).replace("", np.nan)
        category = df["predicted_category"].astype(str)
        valid = vendor.notna()

        if not valid.any():
            return

        v_counts = vendor[valid].value_counts()
        vc_counts = (
            pd.DataFrame({"v": vendor[valid], "c": category[valid]})
            .groupby(["v", "c"])
            .size()
        )

        for i in np.where(valid)[0]:
            v = vendor.iloc[i]
            c = category.iloc[i]
            if v_counts.get(v, 0) < self.config.rarity_min_vendor_count:
                continue
            if vc_counts.get((v, c), 0) < self.config.rarity_min_vendor_category_count:
                scores[i] = max(scores[i], 0.3)
                reasons[i].append("Rare vendor–category combination")

    # ========================================================
    # ML RULES
    # ========================================================

    def _build_numeric_matrix(self, df):
        amt = pd.to_numeric(df["amount"], errors="coerce")
        valid = amt.notna()
        X = np.column_stack([
            amt.fillna(0.0),
            np.log1p(np.clip(amt, 0, None)).fillna(0.0),
        ])
        return X[valid], np.where(valid)[0]

    def _apply_isolation_forest(self, df, scores):
        X, idx = self._build_numeric_matrix(df)
        if len(X) < self.config.min_rows_for_ml:
            return

        model = IsolationForest(
            contamination=self.config.isolation_forest_contamination,
            random_state=self.config.random_state,
        )
        raw = -model.fit(X).score_samples(X)
        norm = self._normalize(raw)
        for i, s in zip(idx, norm):
            scores[i] = max(scores[i], s)

    def _apply_lof(self, df, scores):
        X, idx = self._build_numeric_matrix(df)
        if len(X) <= self.config.lof_n_neighbors:
            return

        lof = LocalOutlierFactor(n_neighbors=self.config.lof_n_neighbors)
        lof.fit(X)
        raw = -lof.negative_outlier_factor_
        norm = self._normalize(raw)
        for i, s in zip(idx, norm):
            scores[i] = max(scores[i], s)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        mn, mx = float(arr.min()), float(arr.max())
        return np.zeros_like(arr) if mx <= mn else (arr - mn) / (mx - mn)
