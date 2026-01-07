# anomaly/numeric_anomaly.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


@dataclass
class NumericAnomalyConfig:
    min_category_samples: int = 20
    iqr_threshold_mild: float = 1.5
    iqr_threshold_strong: float = 3.0
    mad_threshold_mild: float = 2.5
    mad_threshold_strong: float = 4.0
    rolling_window_days: int = 7
    freq_surge_multiplier: float = 3.0
    vendor_amount_mad_threshold: float = 3.0

    ratio_mild: float = 1.5
    ratio_strong: float = 3.0

    enable_isolation_forest: bool = False
    enable_lof: bool = False
    min_rows_for_ml: int = 300
    anomaly_threshold: float = 0.5

    isolation_forest_contamination: float = 0.05
    random_state: int = 42
    lof_n_neighbors: int = 20


class NumericAnomalyDetector:

    def __init__(self, **kwargs: Any) -> None:
        self.config = NumericAnomalyConfig(**kwargs)

    # --------------------------------------------------

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            return df

        required = {
            "transaction_id",
            "transaction_date",
            "amount",
            "predicted_category",
            "confidence_score",
        }
        if not required.issubset(df.columns):
            n = len(df)
            return self._attach(df.copy(), np.zeros(n), [""] * n, np.zeros(n, bool))

        if df.empty:
            return self._attach(df.copy(), np.zeros(0), [], np.zeros(0, bool))

        out = df.copy()
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
        out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")

        n = len(out)
        scores = np.zeros(n)
        reasons = [[] for _ in range(n)]

        self._iqr_outliers(out, scores, reasons)
        self._mad_outliers(out, scores, reasons)
        self._time_window_spikes(out, scores, reasons)
        self._frequency(out, scores, reasons)
        self._vendor(out, scores, reasons)

        if _SKLEARN_AVAILABLE and n >= self.config.min_rows_for_ml:
            if self.config.enable_isolation_forest:
                self._apply_isolation_forest(out, scores)
            if self.config.enable_lof:
                self._apply_lof(out, scores)

        for i in range(n):
            if scores[i] > 0 and not reasons[i]:
                reasons[i].append("Numeric anomaly detected")

        final_reasons = []
        for r in reasons:
            r = list(dict.fromkeys(r))
            r.sort(key=lambda x: ("outlier" not in x.lower()))
            final_reasons.append("; ".join(r))

        flags = scores >= self.config.anomaly_threshold
        return self._attach(out, scores, final_reasons, flags)

    # --------------------------------------------------

    @staticmethod
    def _attach(df, scores, reasons, flags):
        df["anomaly_score"] = scores
        df["anomaly_flag"] = flags
        df["anomaly_reason"] = reasons
        return df

    # --------------------------------------------------
    # IQR
    # --------------------------------------------------

    def _iqr_outliers(self, df, scores, reasons):
        for _, g in df.groupby("predicted_category"):
            vals = g["amount"]
            med = vals.median()
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1

            for idx, val in vals.items():
                pos = df.index.get_loc(idx)
                if pd.isna(val) or pd.isna(med):
                    continue

                if iqr > 0:
                    if val > q3 + self.config.iqr_threshold_strong * iqr:
                        scores[pos] = max(scores[pos], 0.8)
                        reasons[pos].append("Strong amount outlier (IQR)")
                    elif val > q3 + self.config.iqr_threshold_mild * iqr:
                        scores[pos] = max(scores[pos], 0.4)
                        reasons[pos].append("Mild amount outlier (IQR)")
                else:
                    if val >= self.config.ratio_strong * med:
                        scores[pos] = max(scores[pos], 0.8)
                        reasons[pos].append("Strong amount outlier (ratio)")
                    elif val >= self.config.ratio_mild * med:
                        scores[pos] = max(scores[pos], 0.4)
                        reasons[pos].append("Mild amount outlier (ratio)")

    # --------------------------------------------------
    # MAD
    # --------------------------------------------------

    def _mad_outliers(self, df, scores, reasons):
        for _, g in df.groupby("predicted_category"):
            vals = g["amount"]
            med = vals.median()
            mad = np.median(np.abs(vals - med))

            for idx, val in vals.items():
                pos = df.index.get_loc(idx)
                if pd.isna(val) or pd.isna(med):
                    continue

                if mad > 0:
                    z = abs(val - med) / mad
                    if z >= self.config.mad_threshold_strong:
                        scores[pos] = max(scores[pos], 0.7)
                        reasons[pos].append("Strong amount outlier (MAD)")
                else:
                    if val >= self.config.ratio_strong * med:
                        scores[pos] = max(scores[pos], 0.7)
                        reasons[pos].append("Strong amount outlier (MAD fallback)")

    # --------------------------------------------------
    # Time window
    # --------------------------------------------------

    def _time_window_spikes(self, df, scores, reasons):
        if self.config.rolling_window_days <= 0:
            return

        df_sorted = df.sort_values("transaction_date")

        for _, g in df_sorted.groupby("predicted_category"):
            if len(g) <= self.config.rolling_window_days:
                continue

            g = g.reset_index()
            for i in range(self.config.rolling_window_days, len(g)):
                window = g.loc[i - self.config.rolling_window_days:i - 1, "amount"]
                med = window.median()
                if pd.isna(med):
                    continue

                cur_idx = g.loc[i, "index"]
                cur_val = g.loc[i, "amount"]
                pos = df.index.get_loc(cur_idx)

                if cur_val >= self.config.ratio_strong * med:
                    scores[pos] = max(scores[pos], 0.7)
                    reasons[pos].append("Time-window spike")

    # --------------------------------------------------
    # Frequency
    # --------------------------------------------------

    def _frequency(self, df, scores, reasons):
        df = df.copy()
        df["date"] = pd.to_datetime(df["transaction_date"]).dt.date

        for _, g in df.groupby("predicted_category"):
            counts = g.groupby("date").size()
            if counts.empty:
                continue
            med = counts.median()

            for day in counts[counts > med * self.config.freq_surge_multiplier].index:
                for idx in g[g["date"] == day].index:
                    pos = df.index.get_loc(idx)
                    scores[pos] = max(scores[pos], 0.5)
                    reasons[pos].append("Frequency surge")

    # --------------------------------------------------
    # Vendor
    # --------------------------------------------------

    def _vendor(self, df, scores, reasons):
        if "vendor_id" not in df.columns:
            return

        for _, g in df.groupby("vendor_id"):
            med = g["amount"].median()
            if pd.isna(med):
                continue

            for idx, val in g["amount"].items():
                if val >= self.config.ratio_strong * med:
                    pos = df.index.get_loc(idx)
                    scores[pos] = max(scores[pos], 0.6)
                    reasons[pos].append("Vendor amount deviation")

    # --------------------------------------------------
    # ML
    # --------------------------------------------------

    def _build_X(self, df) -> Tuple[np.ndarray, np.ndarray]:
        amt = df["amount"].fillna(0.0).to_numpy()
        X = np.column_stack([amt, np.log1p(np.clip(amt, 0, None))])
        return X, np.arange(len(df))

    def _apply_isolation_forest(self, df, scores):
        X, idx = self._build_X(df)
        raw = -IsolationForest(
            contamination=self.config.isolation_forest_contamination,
            random_state=self.config.random_state,
        ).fit(X).score_samples(X)

        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        for i, s in zip(idx, norm):
            scores[i] = max(scores[i], float(s))

    def _apply_lof(self, df, scores):
        X, idx = self._build_X(df)
        if len(X) <= 1:
            return
        lof = LocalOutlierFactor(n_neighbors=min(self.config.lof_n_neighbors, len(X) - 1))
        raw = -lof.fit_predict(X).astype(float)
        for i, s in zip(idx, raw):
            scores[i] = max(scores[i], float(s))

    # --------------------------------------------------

    def describe_drift_detection_design(self) -> str:
        return "Design only: PSI / KS-based distribution drift checks."

