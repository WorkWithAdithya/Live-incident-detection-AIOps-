"""
evaluate.py
-----------
Comprehensive evaluation of the trained LSTM Autoencoder.

Metrics computed:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix
  - ROC-AUC Score
  - Per-incident-type detection rate
  - Classification Report (full breakdown)
  - Optimal threshold suggestion (via Youden's J statistic)

Plots saved to ../saved/:
  - evaluation.png        : error distribution + anomaly timeline
  - confusion_matrix.png  : visual confusion matrix
  - roc_curve.png         : ROC curve with AUC
  - error_boxplot.png     : per-incident error spread

Usage:
    python -m model.evaluate
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

from model.lstm_autoencoder import LSTMAutoencoder
from model.dataset import SystemLogsDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR       = os.path.join(os.path.dirname(__file__), "..", "saved")
MODEL_PATH     = os.path.join(SAVE_DIR, "lstm_autoencoder.pth")
SCALER_PATH    = os.path.join(SAVE_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(SAVE_DIR, "threshold.txt")

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN    = 60
BATCH_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INCIDENT_COLORS = {
    "normal"        : "#4A90D9",
    "cpu_spike"     : "#E74C3C",
    "memory_leak"   : "#F39C12",
    "disk_near_full": "#8E44AD",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_threshold() -> float:
    with open(THRESHOLD_PATH) as f:
        return float(f.readline().strip())


def load_threshold_stats() -> dict:
    stats = {}
    with open(THRESHOLD_PATH) as f:
        lines = f.read().strip().splitlines()
    stats["threshold"] = float(lines[0])
    for line in lines[1:]:
        k, v = line.split("=")
        stats[k.strip()] = float(v.strip())
    return stats


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_all_metrics(true_labels: np.ndarray,
                        pred_labels: np.ndarray,
                        all_errors:  np.ndarray) -> dict:
    """
    Computes the full suite of classification metrics.
    Returns a dict with all values for printing and plotting.
    """
    tp = int(((pred_labels == 1) & (true_labels == 1)).sum())
    fp = int(((pred_labels == 1) & (true_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (true_labels == 1)).sum())
    tn = int(((pred_labels == 0) & (true_labels == 0)).sum())

    accuracy  = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall    = recall_score(true_labels, pred_labels, zero_division=0)
    f1        = f1_score(true_labels, pred_labels, zero_division=0)
    roc_auc   = roc_auc_score(true_labels, all_errors)
    avg_prec  = average_precision_score(true_labels, all_errors)

    # False Positive Rate and False Negative Rate
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Matthews Correlation Coefficient — robust for imbalanced datasets
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc   = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy"   : accuracy,
        "precision"  : precision,
        "recall"     : recall,
        "f1"         : f1,
        "roc_auc"    : roc_auc,
        "avg_prec"   : avg_prec,
        "mcc"        : mcc,
        "specificity": specificity,
        "fpr"        : fpr_val,
        "fnr"        : fnr_val,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def find_optimal_threshold(true_labels: np.ndarray,
                           all_errors:  np.ndarray) -> float:
    """
    Uses Youden's J statistic (sensitivity + specificity - 1) to find
    the threshold that maximises the trade-off between TPR and FPR.
    Useful if you want to tune away from the mean+3σ default.
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(true_labels, all_errors)
    j_scores  = tpr_arr - fpr_arr          # Youden's J
    best_idx  = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def per_incident_stats(true_labels:  np.ndarray,
                       pred_labels:  np.ndarray,
                       all_errors:   np.ndarray,
                       window_types: list) -> dict:
    """
    Breaks down detection rate, mean error, and false negative count
    per incident type.
    """
    results = {}
    for inc_type in ["cpu_spike", "memory_leak", "disk_near_full"]:
        idx = np.array([i for i, t in enumerate(window_types) if t == inc_type])
        if len(idx) == 0:
            continue
        detected   = int(pred_labels[idx].sum())
        total      = len(idx)
        missed     = total - detected
        mean_err   = float(all_errors[idx].mean())
        max_err    = float(all_errors[idx].max())
        results[inc_type] = {
            "total"    : total,
            "detected" : detected,
            "missed"   : missed,
            "rate"     : detected / total,
            "mean_err" : mean_err,
            "max_err"  : max_err,
        }
    return results


# ── Print functions ───────────────────────────────────────────────────────────

def print_full_report(metrics:      dict,
                      true_labels:  np.ndarray,
                      pred_labels:  np.ndarray,
                      threshold:    float,
                      opt_threshold:float,
                      per_incident: dict,
                      n_windows:    int):

    sep = "=" * 62

    print(f"\n{sep}")
    print("  LSTM AUTOENCODER — FULL EVALUATION REPORT")
    print(sep)
    print(f"  Total windows evaluated : {n_windows:,}")
    print(f"  True anomaly windows    : {int(metrics['tp'] + metrics['fn']):,}  "
          f"({100*(metrics['tp']+metrics['fn'])/n_windows:.1f}%)")
    print(f"  Applied threshold       : {threshold:.6f}  (mean + 3σ)")
    print(f"  Optimal threshold (J)   : {opt_threshold:.6f}  (Youden's J statistic)")

    print(f"\n{'─'*62}")
    print("  CORE ACCURACY METRICS")
    print(f"{'─'*62}")

    def bar(val, width=20):
        filled = int(val * width)
        return "█" * filled + "░" * (width - filled)

    metrics_to_print = [
        ("Accuracy",              metrics["accuracy"],    "⚠️  Can be misleading (class imbalance)"),
        ("Precision",             metrics["precision"],   "Of flagged alerts, how many are real"),
        ("Recall  (Sensitivity)", metrics["recall"],      "Of real incidents, how many detected  ← most important"),
        ("F1 Score",              metrics["f1"],          "Balanced precision-recall score"),
        ("Specificity",           metrics["specificity"], "Of normal windows, how many stay unflagged"),
        ("ROC-AUC",               metrics["roc_auc"],     "Overall separability (1.0 = perfect)"),
        ("Avg Precision (PR-AUC)",metrics["avg_prec"],    "Area under Precision-Recall curve"),
        ("MCC",                   metrics["mcc"],         "Matthews Corr. Coeff — best for imbalanced data"),
    ]

    for name, val, note in metrics_to_print:
        print(f"\n  {name:<26} {val:.4f}  {bar(val)}")
        print(f"  {'':26} └─ {note}")

    print(f"\n{'─'*62}")
    print("  ERROR RATES")
    print(f"{'─'*62}")
    print(f"  False Positive Rate  : {metrics['fpr']:.4f}  "
          f"({metrics['fpr']*100:.2f}% of normal windows falsely flagged)")
    print(f"  False Negative Rate  : {metrics['fnr']:.4f}  "
          f"({metrics['fnr']*100:.2f}% of incidents missed)")

    print(f"\n{'─'*62}")
    print("  CONFUSION MATRIX")
    print(f"{'─'*62}")
    print(f"                   Predicted Normal   Predicted Anomaly")
    print(f"  Actual Normal  :    TN={metrics['tn']:>6,}            FP={metrics['fp']:>6,}")
    print(f"  Actual Anomaly :    FN={metrics['fn']:>6,}            TP={metrics['tp']:>6,}")

    print(f"\n{'─'*62}")
    print("  SKLEARN CLASSIFICATION REPORT")
    print(f"{'─'*62}")
    report = classification_report(
        true_labels, pred_labels,
        target_names=["Normal", "Anomaly"],
        digits=4
    )
    for line in report.splitlines():
        print(f"  {line}")

    print(f"\n{'─'*62}")
    print("  PER-INCIDENT-TYPE BREAKDOWN")
    print(f"{'─'*62}")
    print(f"  {'Incident':<20} {'Detected':>8} {'Total':>7} {'Rate':>7} "
          f"{'Mean Error':>12} {'Max Error':>11}")
    print(f"  {'─'*20} {'─'*8} {'─'*7} {'─'*7} {'─'*12} {'─'*11}")
    for inc_type, s in per_incident.items():
        rate_bar = "█" * int(s['rate'] * 10)
        print(f"  {inc_type:<20} {s['detected']:>8,} {s['total']:>7,} "
              f"{s['rate']*100:>6.1f}% {s['mean_err']:>12.6f} {s['max_err']:>11.6f}  {rate_bar}")

    print(f"\n{'─'*62}")
    print("  VERDICT")
    print(f"{'─'*62}")
    if metrics["recall"] >= 0.95 and metrics["f1"] >= 0.85:
        print("  ✅ Model is PRODUCTION READY.")
        print("     High recall ensures incidents are not missed.")
    elif metrics["recall"] >= 0.90:
        print("  ⚠️  Model is GOOD but could be improved.")
        print("     Consider tuning the threshold or retraining.")
    else:
        print("  ❌ Model needs improvement before production use.")
        print("     Consider more training data or architecture tuning.")

    if opt_threshold < threshold:
        improvement = (1 - metrics["fpr"]) * 100
        print(f"\n  💡 Tip: Lowering threshold to {opt_threshold:.6f} (Youden's J)")
        print(f"     may reduce false positives while preserving recall.")

    print(f"\n{sep}\n")


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_error_distribution_and_timeline(all_errors, window_types,
                                          threshold, metrics, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Left: Error distribution ──
    ax = axes[0]
    for inc_type, color in INCIDENT_COLORS.items():
        idx = np.array([i for i, t in enumerate(window_types) if t == inc_type])
        if len(idx) == 0:
            continue
        ax.hist(all_errors[idx], bins=80, alpha=0.70, color=color,
                label=inc_type, edgecolor="none")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Distribution by Incident Type", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: Anomaly timeline ──
    ax = axes[1]
    step   = max(1, len(all_errors) // 4000)
    idx_s  = np.arange(0, len(all_errors), step)
    colors = [INCIDENT_COLORS.get(window_types[i], "#cccccc") for i in idx_s]
    ax.scatter(idx_s, all_errors[idx_s], c=colors, s=4, alpha=0.65)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold:.4f})")
    patches = [mpatches.Patch(color=c, label=t)
               for t, c in INCIDENT_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("Window Index", fontsize=11)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_title("Anomaly Timeline", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    title = (f"LSTM Autoencoder Evaluation  |  "
             f"F1={metrics['f1']:.3f}  "
             f"Prec={metrics['precision']:.3f}  "
             f"Rec={metrics['recall']:.3f}  "
             f"AUC={metrics['roc_auc']:.3f}")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "evaluation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Error distribution + timeline → {path}")


def plot_confusion_matrix(metrics, save_dir):
    cm     = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest",
                   cmap=plt.cm.Blues)  # type: ignore
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Anomaly"], fontsize=11)
    ax.set_yticklabels(["Actual Normal", "Actual Anomaly"], fontsize=11)

    thresh_cm = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh_cm else "black"
            ax.text(j, i,
                    f"{labels[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix          → {path}")


def plot_roc_curve(true_labels, all_errors, metrics, save_dir):
    fpr_arr, tpr_arr, _ = roc_curve(true_labels, all_errors)
    prec_arr, rec_arr, _ = precision_recall_curve(true_labels, all_errors)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── ROC Curve ──
    ax = axes[0]
    ax.plot(fpr_arr, tpr_arr, color="#4A90D9", linewidth=2.5,
            label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1.2, label="Random Classifier")
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.10, color="#4A90D9")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # ── Precision-Recall Curve ──
    ax = axes[1]
    ax.plot(rec_arr, prec_arr, color="#E67E22", linewidth=2.5,
            label=f"PR Curve (AP = {metrics['avg_prec']:.4f})")
    ax.fill_between(rec_arr, prec_arr, alpha=0.10, color="#E67E22")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.suptitle("Model Discrimination Ability", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 ROC + PR curves           → {path}")


def plot_error_boxplot(all_errors, window_types, threshold, save_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    incident_order = ["normal", "cpu_spike", "memory_leak", "disk_near_full"]
    data_by_type   = []
    labels_bp      = []
    colors_bp      = []

    for inc_type in incident_order:
        idx = np.array([i for i, t in enumerate(window_types) if t == inc_type])
        if len(idx) == 0:
            continue
        data_by_type.append(all_errors[idx])
        labels_bp.append(inc_type.replace("_", "\n"))
        colors_bp.append(INCIDENT_COLORS[inc_type])

    bp = ax.boxplot(
        data_by_type,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
        medianprops=dict(color="black", linewidth=2),
    )

    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(threshold, color="red", linestyle="--",
               linewidth=1.8, label=f"Threshold ({threshold:.4f})")
    ax.set_xticklabels(labels_bp, fontsize=11)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_title("Reconstruction Error per Incident Type",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "error_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Error boxplot             → {path}")


def plot_accuracy_card(metrics, threshold, opt_threshold, save_dir):
    """
    Standalone accuracy-focused PNG showing:
      - Accuracy gauge
      - Accuracy vs other core metrics side by side
      - Why accuracy alone is insufficient (imbalance note)
      - Confusion matrix numbers embedded
    """
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#0F1923")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.38)

    # ── Shared style ──────────────────────────────────────────────────────
    ACCENT   = "#00E5FF"
    GOOD     = "#2ECC71"
    WARN     = "#F39C12"
    CRITICAL = "#E74C3C"
    TEXT     = "#ECEFF1"
    PANEL    = "#1A2535"

    def panel_ax(ax, title=""):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2C3E50")
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.title.set_color(TEXT)
        ax.title.set_fontsize(11)
        ax.title.set_fontweight("bold")
        if title:
            ax.set_title(title)
        return ax

    # ── 1. Accuracy Gauge (donut chart) ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    panel_ax(ax1)
    ax1.set_aspect("equal")

    acc_val  = metrics["accuracy"]
    rem_val  = 1.0 - acc_val
    colors_d = [ACCENT, "#1A2535"]
    wedges, _ = ax1.pie(
        [acc_val, rem_val],
        colors=colors_d,
        startangle=90,
        wedgeprops=dict(width=0.38, edgecolor="#0F1923", linewidth=2),
        counterclock=False,
    )
    ax1.text(0, 0.08, f"{acc_val*100:.2f}%",
             ha="center", va="center", fontsize=20,
             fontweight="bold", color=ACCENT)
    ax1.text(0, -0.22, "A C C U R A C Y",
             ha="center", va="center", fontsize=9,
             color=TEXT, fontweight="bold")

    # Imbalance warning
    ax1.text(0, -0.58,
             "⚠ High due to class imbalance\n   Use F1 / Recall for AIOps",
             ha="center", va="center", fontsize=7.5,
             color=WARN, style="italic")
    ax1.set_title("Accuracy", color=TEXT, fontsize=11, fontweight="bold", pad=8)

    # ── 2. Core metrics horizontal bar ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    panel_ax(ax2, "Accuracy vs Core Metrics")

    bar_metrics = {
        "Accuracy"   : (metrics["accuracy"],    "#95A5A6", "⚠ misleading alone"),
        "Precision"  : (metrics["precision"],   WARN,      ""),
        "Recall"     : (metrics["recall"],      GOOD,      "← most critical"),
        "F1 Score"   : (metrics["f1"],          ACCENT,    ""),
        "Specificity": (metrics["specificity"], "#8E44AD", ""),
        "ROC-AUC"    : (metrics["roc_auc"],     "#1ABC9C", ""),
    }

    names  = list(bar_metrics.keys())
    vals   = [v[0] for v in bar_metrics.values()]
    colors = [v[1] for v in bar_metrics.values()]
    notes  = [v[2] for v in bar_metrics.values()]
    y_pos  = range(len(names))

    bars = ax2.barh(list(y_pos), vals, color=colors,
                    edgecolor="#0F1923", linewidth=0.8, height=0.55)
    ax2.set_xlim(0, 1.18)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(names, fontsize=10, color=TEXT)
    ax2.set_xlabel("Score", color=TEXT, fontsize=9)
    ax2.axvline(1.0, color="#2C3E50", linestyle="--", linewidth=1)
    ax2.grid(axis="x", alpha=0.15, color=TEXT)

    for bar, val, note, color in zip(bars, vals, notes, colors):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}  {note}",
                 va="center", fontsize=8.5, color=color)

    # ── 3. Confusion matrix tile ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    panel_ax(ax3, "Confusion Matrix")

    tp, fp = metrics["tp"], metrics["fp"]
    fn, tn = metrics["fn"], metrics["tn"]
    cm_data  = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_norm  = cm_data / cm_data.sum()

    im = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    cell_vals   = [[tn, fp], [fn, tp]]
    cell_colors = [["#2ECC71", "#E74C3C"], ["#E74C3C", "#2ECC71"]]

    for i in range(2):
        for j in range(2):
            ax3.text(j, i,
                     f"{cell_labels[i][j]}\n{cell_vals[i][j]:,}",
                     ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color=cell_colors[i][j])

    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred Normal", "Pred Anomaly"], fontsize=8, color=TEXT)
    ax3.set_yticklabels(["Act Normal", "Act Anomaly"],   fontsize=8, color=TEXT)

    # ── 4. FPR / FNR / accuracy breakdown ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    panel_ax(ax4, "Error Rate Breakdown")

    rate_labels = ["False\nPositive\nRate", "False\nNegative\nRate",
                   "Miss\nRate\n(FNR)"]
    rate_vals   = [metrics["fpr"], metrics["fnr"], metrics["fnr"]]
    rate_colors = [WARN, CRITICAL, CRITICAL]

    b = ax4.bar(rate_labels[:2],
                [metrics["fpr"], metrics["fnr"]],
                color=[WARN, CRITICAL],
                edgecolor="#0F1923", width=0.45)
    ax4.set_ylim(0, max(metrics["fpr"], metrics["fnr"]) * 1.6 + 0.01)
    ax4.set_ylabel("Rate", color=TEXT, fontsize=9)
    ax4.grid(axis="y", alpha=0.15, color=TEXT)
    for bar, val in zip(b, [metrics["fpr"], metrics["fnr"]]):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{val*100:.2f}%",
                 ha="center", fontsize=10,
                 fontweight="bold", color=TEXT)

    # ── 5. Threshold comparison ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    panel_ax(ax5, "Threshold Comparison")

    t_labels = ["Applied\n(mean+3σ)", "Optimal\n(Youden's J)"]
    t_vals   = [threshold, opt_threshold]
    t_colors = [ACCENT, GOOD]

    b2 = ax5.bar(t_labels, t_vals, color=t_colors,
                 edgecolor="#0F1923", width=0.4)
    ax5.set_ylabel("Threshold Value", color=TEXT, fontsize=9)
    ax5.grid(axis="y", alpha=0.15, color=TEXT)
    for bar, val in zip(b2, t_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.00002,
                 f"{val:.6f}",
                 ha="center", fontsize=9,
                 fontweight="bold", color=TEXT)

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"AIOps LSTM Autoencoder — Accuracy Report  |  "
        f"Accuracy={metrics['accuracy']*100:.2f}%   "
        f"F1={metrics['f1']:.3f}   "
        f"Recall={metrics['recall']:.3f}   "
        f"MCC={metrics['mcc']:.3f}",
        fontsize=12, fontweight="bold", color=TEXT, y=1.01
    )

    path = os.path.join(save_dir, "accuracy_report.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 Accuracy report           → {path}")


def plot_metrics_dashboard(metrics, save_dir):
    """
    Single-page visual summary of all key metrics as a bar chart.
    """
    metric_names = [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "Specificity", "ROC-AUC",
        "Avg Precision", "MCC"
    ]
    metric_vals = [
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"],       metrics["specificity"], metrics["roc_auc"],
        metrics["avg_prec"], metrics["mcc"],
    ]
    bar_colors = [
        "#95A5A6",   # accuracy  — grey (use with caution)
        "#E67E22",   # precision
        "#27AE60",   # recall    — most important, green
        "#4A90D9",   # f1
        "#8E44AD",   # specificity
        "#2ECC71",   # roc-auc
        "#F39C12",   # avg precision
        "#1ABC9C",   # mcc
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(metric_names, metric_vals, color=bar_colors,
                  edgecolor="white", linewidth=1.2, width=0.6)

    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance — All Metrics at a Glance",
                 fontsize=13, fontweight="bold")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=10)

    # Annotate recall bar specially
    ax.annotate("← Most critical\n   for AIOps",
                xy=(2, metrics["recall"]),
                xytext=(3.2, metrics["recall"] - 0.12),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=9, color="green")

    plt.tight_layout()
    path = os.path.join(save_dir, "metrics_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Metrics dashboard         → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load artefacts ────────────────────────────────────────────────────
    print("📦 Loading model, scaler, threshold...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    threshold      = load_threshold()
    threshold_stats = load_threshold_stats()

    model = LSTMAutoencoder(
        input_size=3, hidden_size=64, latent_size=32,
        num_layers=2, dropout=0.2, seq_len=SEQ_LEN
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────
    print("📂 Loading full synthetic dataset...")
    dataset = SystemLogsDataset(seq_len=SEQ_LEN, stride=1, scaler=scaler)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Forward pass — compute reconstruction errors ───────────────────
    print("🔄 Running inference on all windows...")
    all_errors = []
    with torch.no_grad():
        for batch in loader:
            errors = model.reconstruction_error(batch.to(DEVICE))
            all_errors.extend(errors.cpu().numpy().tolist())

    all_errors   = np.array(all_errors)
    true_labels  = dataset.window_labels
    pred_labels  = (all_errors > threshold).astype(int)
    window_types = dataset.window_types
    n_windows    = len(all_errors)

    # ── Compute all metrics ───────────────────────────────────────────────
    metrics       = compute_all_metrics(true_labels, pred_labels, all_errors)
    opt_threshold = find_optimal_threshold(true_labels, all_errors)
    per_incident  = per_incident_stats(true_labels, pred_labels,
                                       all_errors, window_types)

    # ── Print full report ─────────────────────────────────────────────────
    print_full_report(
        metrics, true_labels, pred_labels,
        threshold, opt_threshold, per_incident, n_windows
    )

    # ── Generate all plots ────────────────────────────────────────────────
    print("🎨 Generating plots...")
    plot_accuracy_card(metrics, threshold, opt_threshold, SAVE_DIR)
    plot_error_distribution_and_timeline(
        all_errors, window_types, threshold, metrics, SAVE_DIR
    )
    plot_confusion_matrix(metrics, SAVE_DIR)
    plot_roc_curve(true_labels, all_errors, metrics, SAVE_DIR)
    plot_error_boxplot(all_errors, window_types, threshold, SAVE_DIR)
    plot_metrics_dashboard(metrics, SAVE_DIR)

    print("\n✅ Evaluation complete.")
    print(f"   All plots saved to: {os.path.abspath(SAVE_DIR)}/\n")


if __name__ == "__main__":
    main()