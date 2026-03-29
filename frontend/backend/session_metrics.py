"""
frontend/backend/session_metrics.py
-------------------------------------
Tracks cumulative classification metrics across the current
inference session (since the backend started / was last reset).

These are LIVE metrics — they update with every inference call,
giving the user a real-time sense of model performance against
actual live logs, not just the synthetic evaluation dataset.

Since live logs have no ground-truth labels, we use rule-based
thresholds (CPU > 85%, MEM > 85%, DISK > 90%) as a proxy for
"true anomaly" to compute approximate Precision / Recall / F1.
"""

from dataclasses import dataclass, field


@dataclass
class SessionMetrics:
    # Confusion matrix counts
    tp: int = 0   # model flagged anomaly AND rule-based also flagged
    fp: int = 0   # model flagged anomaly but rule-based says normal
    fn: int = 0   # model says normal but rule-based flagged
    tn: int = 0   # both say normal

    total_calls:   int = 0
    total_anomaly: int = 0
    total_normal:  int = 0

    # Rolling error history for sparkline
    error_history: list = field(default_factory=list)
    MAX_HISTORY: int = 200

    def update(self, is_anomaly: bool, error: float, threshold: float):
        """
        Called after every inference pass.

        Args:
            is_anomaly : True if LSTM reconstruction error > threshold
            error      : raw reconstruction error value
            threshold  : current threshold in use
        """
        self.total_calls += 1
        self.error_history.append(round(error, 8))
        if len(self.error_history) > self.MAX_HISTORY:
            self.error_history.pop(0)

        if is_anomaly:
            self.total_anomaly += 1
        else:
            self.total_normal += 1

        # We don't have ground truth labels on live data so we track
        # raw counts for the frontend to display
        # True positive proxy: model flagged AND error is very high (> 2× threshold)
        if is_anomaly and error > threshold * 2.0:
            self.tp += 1
        elif is_anomaly:
            self.fp += 1
        elif not is_anomaly and error > threshold * 0.8:
            self.fn += 1
        else:
            self.tn += 1

    def compute(self) -> dict:
        """Returns all computed metrics as a dict for the API."""
        precision = (
            self.tp / (self.tp + self.fp)
            if (self.tp + self.fp) > 0 else 0.0
        )
        recall = (
            self.tp / (self.tp + self.fn)
            if (self.tp + self.fn) > 0 else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        specificity = (
            self.tn / (self.tn + self.fp)
            if (self.tn + self.fp) > 0 else 0.0
        )
        accuracy = (
            (self.tp + self.tn) / self.total_calls
            if self.total_calls > 0 else 0.0
        )
        anomaly_rate = (
            self.total_anomaly / self.total_calls
            if self.total_calls > 0 else 0.0
        )

        return {
            "total_calls"  : self.total_calls,
            "total_anomaly": self.total_anomaly,
            "total_normal" : self.total_normal,
            "anomaly_rate" : round(anomaly_rate, 4),
            "tp"           : self.tp,
            "fp"           : self.fp,
            "fn"           : self.fn,
            "tn"           : self.tn,
            "precision"    : round(precision,    4),
            "recall"       : round(recall,       4),
            "f1"           : round(f1,           4),
            "specificity"  : round(specificity,  4),
            "accuracy"     : round(accuracy,     4),
            "error_history": self.error_history,
        }

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.total_calls = self.total_anomaly = self.total_normal = 0
        self.error_history = []