"""
frontend/backend/evaluator.py
------------------------------
Runs full evaluation of the LSTM Autoencoder against
synthetic_logs.csv and returns all data needed by the
frontend to render ROC, PR curves, confusion matrix,
and error distribution charts.

This is a heavier operation — called on demand via GET /evaluate,
not on every inference tick.
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix,
)


SEQ_LEN    = 60
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, ai_model_dir: Path, saved_dir: Path, threshold: float):
        self.ai_model_dir = Path(ai_model_dir)
        self.saved_dir    = Path(saved_dir)
        self.threshold    = threshold

    def run(self) -> dict:
        """
        Loads model + dataset, runs all windows through the model,
        computes every metric, and returns chart-ready data.

        Returns a dict with:
            metrics        : scalar metrics (accuracy, f1, etc.)
            confusion_matrix: [[tn, fp], [fn, tp]]
            roc_curve      : {fpr: [...], tpr: [...], auc: float}
            pr_curve       : {precision: [...], recall: [...], ap: float}
            error_dist     : per-incident {errors: [...]} for histogram
            per_incident   : detection rate per incident type
        """
        # ── Load artefacts ────────────────────────────────────────────────
        with open(self.saved_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        from model.lstm_autoencoder import LSTMAutoencoder
        from model.dataset import SystemLogsDataset

        model = LSTMAutoencoder(
            input_size=3, hidden_size=64, latent_size=32,
            num_layers=2, dropout=0.2, seq_len=SEQ_LEN
        )
        model.load_state_dict(
            torch.load(
                str(self.saved_dir / "lstm_autoencoder.pth"),
                map_location=DEVICE, weights_only=True
            )
        )
        model.to(DEVICE)
        model.eval()

        # ── Dataset ───────────────────────────────────────────────────────
        dataset = SystemLogsDataset(
            seq_len   = SEQ_LEN,
            stride    = 1,
            scaler    = scaler,
            data_path = str(
                self.ai_model_dir / "data" / "synthetic_logs.csv"
            ),
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # ── Inference ─────────────────────────────────────────────────────
        all_errors = []
        with torch.no_grad():
            for batch in loader:
                errs = model.reconstruction_error(batch.to(DEVICE))
                all_errors.extend(errs.cpu().numpy().tolist())

        all_errors  = np.array(all_errors)
        true_labels = dataset.window_labels
        pred_labels = (all_errors > self.threshold).astype(int)
        win_types   = dataset.window_types

        # ── Scalar metrics ────────────────────────────────────────────────
        metrics = {
            "accuracy"   : round(float(accuracy_score(true_labels, pred_labels)), 4),
            "precision"  : round(float(precision_score(true_labels, pred_labels, zero_division=0)), 4),
            "recall"     : round(float(recall_score(true_labels, pred_labels, zero_division=0)), 4),
            "f1"         : round(float(f1_score(true_labels, pred_labels, zero_division=0)), 4),
            "roc_auc"    : round(float(roc_auc_score(true_labels, all_errors)), 4),
            "avg_prec"   : round(float(average_precision_score(true_labels, all_errors)), 4),
        }

        # ── Confusion matrix ──────────────────────────────────────────────
        cm       = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()

        # ── ROC Curve — downsample to 200 pts for JSON size ───────────────
        fpr_arr, tpr_arr, _ = roc_curve(true_labels, all_errors)
        step = max(1, len(fpr_arr) // 200)
        roc_data = {
            "fpr": fpr_arr[::step].tolist(),
            "tpr": tpr_arr[::step].tolist(),
            "auc": metrics["roc_auc"],
        }

        # ── PR Curve — downsample to 200 pts ─────────────────────────────
        prec_arr, rec_arr, _ = precision_recall_curve(true_labels, all_errors)
        step = max(1, len(prec_arr) // 200)
        pr_data = {
            "precision": prec_arr[::step].tolist(),
            "recall"   : rec_arr[::step].tolist(),
            "ap"       : metrics["avg_prec"],
        }

        # ── Error distribution per incident type ──────────────────────────
        incident_types = ["normal", "cpu_spike", "memory_leak", "disk_near_full"]
        error_dist = {}
        for inc in incident_types:
            idx = [i for i, t in enumerate(win_types) if t == inc]
            if idx:
                errs = all_errors[idx].tolist()
                # Downsample to 500 pts max per type for JSON
                step_e = max(1, len(errs) // 500)
                error_dist[inc] = errs[::step_e]

        # ── Per-incident detection rate ───────────────────────────────────
        per_incident = {}
        for inc in ["cpu_spike", "memory_leak", "disk_near_full"]:
            idx = np.array([i for i, t in enumerate(win_types) if t == inc])
            if len(idx) == 0:
                continue
            detected = int(pred_labels[idx].sum())
            total    = len(idx)
            per_incident[inc] = {
                "detected" : detected,
                "total"    : total,
                "rate"     : round(detected / total, 4),
                "mean_err" : round(float(all_errors[idx].mean()), 6),
                "max_err"  : round(float(all_errors[idx].max()),  6),
            }

        return {
            "threshold"      : self.threshold,
            "total_windows"  : len(all_errors),
            "metrics"        : metrics,
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp),
            },
            "roc_curve"      : roc_data,
            "pr_curve"       : pr_data,
            "error_dist"     : error_dist,
            "per_incident"   : per_incident,
        }