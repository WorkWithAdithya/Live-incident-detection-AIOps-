"""
evaluate_forecaster_classification.py
--------------------------------------
Evaluates the LSTM Forecaster as a BREACH PREDICTOR — converts
the regression task into binary classification:

    Predicted value > threshold  →  1 (breach predicted)
    Predicted value <= threshold →  0 (no breach predicted)

    Actual future value > threshold  →  1 (breach actually happened)
    Actual future value <= threshold →  0 (no breach happened)

This lets us compute:
    - Accuracy, Precision, Recall, F1
    - ROC Curve + AUC
    - Precision-Recall Curve + Average Precision
    - Confusion Matrix
    - Per-metric and per-horizon-step breakdown

Two threshold modes:
    1. User-set limits  (cpu_warning, cpu_critical etc.)
    2. Statistical      (mean + 1.5σ of the metric — auto-computed)

Uses real NeonDB data if available, synthetic CSV as fallback.

Usage:
    cd ai_model
    python -m model.evaluate_forecaster_classification
    python -m model.evaluate_forecaster_classification --mode statistical
    python -m model.evaluate_forecaster_classification --cpu-warn 50 --mem-warn 38 --disk-warn 25
"""

import os
import sys
import pickle
import argparse
import numpy as np
import psycopg2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).resolve().parent
_ROOT = _DIR.parent
sys.path.insert(0, str(_ROOT))

_ENV_PATH = _ROOT.parent / "log_generator" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)
DATABASE_URL = os.getenv("DATABASE_URL")

from model.lstm_forecaster import LSTMForecaster

SAVE_DIR     = _ROOT / "saved"
MODEL_PATH   = SAVE_DIR / "lstm_forecaster.pth"
SCALER_PATH  = SAVE_DIR / "scaler_real.pkl"   # prefer real scaler
if not SCALER_PATH.exists():
    SCALER_PATH = SAVE_DIR / "scaler.pkl"

DATA_PATH    = _ROOT / "data" / "synthetic_logs.csv"

LOOKBACK     = 60
HORIZON      = 12
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL_SECONDS", 1))

FEATURES     = ["cpu_usage", "memory_usage", "disk_usage"]
FEAT_LABELS  = ["CPU Usage", "Memory Usage", "Disk Usage"]
FEAT_COLORS  = ["#a78bfa", "#38bdf8", "#fb923c"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

def fetch_from_db(min_rows: int = 100):
    """Fetch real logs from NeonDB."""
    if not DATABASE_URL:
        return None, None
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cpu_usage, memory_usage, disk_usage "
            "FROM system_logs ORDER BY timestamp ASC"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if len(rows) < LOOKBACK + HORIZON + 10:
            return None, None
        raw = np.array([[float(r[0]), float(r[1]), float(r[2])]
                        for r in rows], dtype=np.float32)
        return raw, "NeonDB"
    except Exception as e:
        print(f"   DB fetch failed: {e}")
        return None, None


def fetch_from_csv():
    """Fallback: load synthetic CSV."""
    if not DATA_PATH.exists():
        return None, None
    import pandas as pd
    df  = pd.read_csv(DATA_PATH)
    raw = df[FEATURES].values.astype(np.float32)
    return raw, "synthetic_logs.csv"


def load_data():
    """Try DB first, fall back to CSV."""
    raw, source = fetch_from_db()
    if raw is None:
        raw, source = fetch_from_csv()
    if raw is None:
        sys.exit("❌ No data available. Run logger or generate synthetic CSV.")
    return raw, source


# ── Build windows ─────────────────────────────────────────────────────────────

def build_windows(normalised: np.ndarray, stride: int = 3):
    X_list, y_list = [], []
    total = LOOKBACK + HORIZON
    for start in range(0, len(normalised) - total + 1, stride):
        X_list.append(normalised[start          : start + LOOKBACK])
        y_list.append(normalised[start + LOOKBACK : start + total])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


# ── Threshold modes ───────────────────────────────────────────────────────────

def get_thresholds_statistical(raw: np.ndarray) -> dict:
    """Auto-compute thresholds as mean + 1.5σ per metric."""
    thresholds = {}
    for i, feat in enumerate(FEATURES):
        mean = float(raw[:, i].mean())
        std  = float(raw[:, i].std())
        thresholds[feat] = round(mean + 1.5 * std, 2)
    return thresholds


def get_thresholds_user(cpu_warn, cpu_crit, mem_warn, mem_crit,
                        disk_warn, disk_crit) -> dict:
    """Use user-specified thresholds. Use warning if critical not set."""
    return {
        "cpu_usage"    : cpu_crit  or cpu_warn,
        "memory_usage" : mem_crit  or mem_warn,
        "disk_usage"   : disk_crit or disk_warn,
    }


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_forecaster(model, X: np.ndarray) -> np.ndarray:
    all_pred = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start:start+batch_size]).to(DEVICE)
            pred  = model.predict(batch)
            all_pred.append(pred.cpu().numpy())
    return np.concatenate(all_pred, axis=0)   # (W, horizon, 3)


def compute_classification_metrics(
    y_pred_pct: np.ndarray,
    y_true_pct: np.ndarray,
    thresholds: dict,
) -> dict:
    """
    Converts regression predictions to binary classification per metric.
    Returns dict with per-metric and overall metrics.
    """
    results = {}

    for i, feat in enumerate(FEATURES):
        thresh = thresholds.get(feat)
        if thresh is None:
            continue

        # Binary labels across all windows and all horizon steps
        y_true_bin = (y_true_pct[:, :, i] > thresh).astype(int).flatten()
        y_pred_bin = (y_pred_pct[:, :, i] > thresh).astype(int).flatten()

        # Soft score: how far above threshold (for ROC/PR curves)
        # Use predicted value directly as score — higher = more likely breach
        y_score = y_pred_pct[:, :, i].flatten()

        # Skip if only one class present (can't compute ROC)
        if len(np.unique(y_true_bin)) < 2:
            print(f"   ⚠️  {FEAT_LABELS[i]}: no true positives at threshold {thresh}% "
                  f"— threshold may be too high for this data")
            results[feat] = None
            continue

        acc   = accuracy_score(y_true_bin, y_pred_bin)
        prec  = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec   = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1    = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        auc   = roc_auc_score(y_true_bin, y_score)
        ap    = average_precision_score(y_true_bin, y_score)
        cm    = confusion_matrix(y_true_bin, y_pred_bin)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true_bin, y_score)
        prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_true_bin, y_score)

        # Per-step metrics
        step_metrics = []
        for t in range(HORIZON):
            y_t_bin  = (y_true_pct[:, t, i] > thresh).astype(int)
            y_p_bin  = (y_pred_pct[:, t, i] > thresh).astype(int)
            y_p_score = y_pred_pct[:, t, i]

            if len(np.unique(y_t_bin)) < 2:
                step_metrics.append(None)
                continue

            step_metrics.append({
                "step"     : t + 1,
                "seconds"  : (t + 1) * LOG_INTERVAL,
                "accuracy" : accuracy_score(y_t_bin, y_p_bin),
                "precision": precision_score(y_t_bin, y_p_bin, zero_division=0),
                "recall"   : recall_score(y_t_bin, y_p_bin, zero_division=0),
                "f1"       : f1_score(y_t_bin, y_p_bin, zero_division=0),
                "auc"      : roc_auc_score(y_t_bin, y_p_score)
                             if len(np.unique(y_t_bin)) > 1 else None,
            })

        results[feat] = {
            "label"      : FEAT_LABELS[i],
            "color"      : FEAT_COLORS[i],
            "threshold"  : thresh,
            "accuracy"   : acc,
            "precision"  : prec,
            "recall"     : rec,
            "f1"         : f1,
            "auc"        : auc,
            "ap"         : ap,
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
            "y_true_bin" : y_true_bin,
            "y_pred_bin" : y_pred_bin,
            "y_score"    : y_score,
            "fpr"        : fpr_arr,
            "tpr"        : tpr_arr,
            "prec_curve" : prec_arr,
            "rec_curve"  : rec_arr,
            "step_metrics": step_metrics,
        }

    return results


# ── Print report ──────────────────────────────────────────────────────────────

def print_report(results: dict, thresholds: dict, source: str, n_windows: int):
    print(f"\n{'='*65}")
    print(f"  LSTM FORECASTER — CLASSIFICATION EVALUATION REPORT")
    print(f"{'='*65}")
    print(f"  Data source     : {source}")
    print(f"  Eval windows    : {n_windows:,}")
    print(f"  Horizon         : {HORIZON} steps × {LOG_INTERVAL}s = {HORIZON*LOG_INTERVAL}s ahead")
    print(f"  Classification  : predicted value > threshold = breach\n")

    for feat, r in results.items():
        if r is None:
            print(f"  {feat}: ⚠️ skipped (no positive class at this threshold)\n")
            continue

        print(f"  {'─'*63}")
        print(f"  {r['label']}  (breach threshold: {r['threshold']}%)")
        print(f"  {'─'*63}")
        print(f"  {'Metric':<20} {'Value':>8}   {'Bar':}")

        for name, val in [
            ("Accuracy",  r["accuracy"]),
            ("Precision", r["precision"]),
            ("Recall",    r["recall"]),
            ("F1 Score",  r["f1"]),
            ("ROC-AUC",   r["auc"]),
            ("Avg Prec",  r["ap"]),
        ]:
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {name:<20} {val:>8.4f}   {bar}")

        print(f"\n  Confusion Matrix:")
        print(f"    TP={r['tp']:>5,}  FP={r['fp']:>5,}")
        print(f"    FN={r['fn']:>5,}  TN={r['tn']:>5,}")

        print(f"\n  Per-step F1 and AUC:")
        print(f"  {'Step':>4}  {'Sec':>5}  {'Acc':>7}  {'Prec':>7}  "
              f"{'Rec':>7}  {'F1':>7}  {'AUC':>7}")
        print(f"  {'─'*4}  {'─'*5}  {'─'*7}  {'─'*7}  "
              f"{'─'*7}  {'─'*7}  {'─'*7}")
        for sm in r["step_metrics"]:
            if sm is None:
                continue
            auc_str = f"{sm['auc']:.4f}" if sm["auc"] is not None else "   N/A"
            print(f"  {sm['step']:>4}  {sm['seconds']:>5}  "
                  f"{sm['accuracy']:>7.4f}  {sm['precision']:>7.4f}  "
                  f"{sm['recall']:>7.4f}  {sm['f1']:>7.4f}  {auc_str:>7}")
        print()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict):
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 5))
    if len(valid) == 1:
        axes = [axes]

    for ax, (feat, r) in zip(axes, valid.items()):
        ax.plot(r["fpr"], r["tpr"], color=r["color"], linewidth=2.5,
                label=f"ROC  (AUC = {r['auc']:.4f})")
        ax.plot([0, 1], [0, 1], color="#555", linestyle="--",
                linewidth=1.2, label="Random")
        ax.fill_between(r["fpr"], r["tpr"], alpha=0.1, color=r["color"])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"{r['label']}\n(threshold: {r['threshold']}%)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.suptitle(
        "LSTM Forecaster — ROC Curves (Breach Prediction)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_roc.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 ROC curves               → {path}")


def plot_pr_curves(results: dict):
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 5))
    if len(valid) == 1:
        axes = [axes]

    for ax, (feat, r) in zip(axes, valid.items()):
        ax.plot(r["rec_curve"], r["prec_curve"], color=r["color"],
                linewidth=2.5, label=f"PR  (AP = {r['ap']:.4f})")
        ax.fill_between(r["rec_curve"], r["prec_curve"],
                        alpha=0.1, color=r["color"])
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"{r['label']}\n(threshold: {r['threshold']}%)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.suptitle(
        "LSTM Forecaster — Precision-Recall Curves (Breach Prediction)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_pr.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Precision-Recall curves  → {path}")


def plot_confusion_matrices(results: dict):
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4))
    if len(valid) == 1:
        axes = [axes]

    for ax, (feat, r) in zip(axes, valid.items()):
        cm_arr = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        im = ax.imshow(cm_arr, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax)
        labels = [["TN", "FP"], ["FN", "TP"]]
        thresh_cm = cm_arr.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{labels[i][j]}\n{cm_arr[i,j]:,}",
                        ha="center", va="center", fontsize=12,
                        fontweight="bold",
                        color="white" if cm_arr[i,j] > thresh_cm else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: No Breach", "Pred: Breach"])
        ax.set_yticklabels(["Act: No Breach", "Act: Breach"])
        ax.set_title(f"{r['label']}  (thresh: {r['threshold']}%)",
                     fontsize=10, fontweight="bold")

    plt.suptitle(
        "LSTM Forecaster — Confusion Matrices (Breach Prediction)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_confusion.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrices       → {path}")


def plot_metrics_summary(results: dict):
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return

    metrics_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg Prec"]
    metric_keys   = ["accuracy", "precision", "recall", "f1", "auc", "ap"]

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (feat, r) in zip(axes, valid.items()):
        vals   = [r[k] for k in metric_keys]
        colors = [
            "#95A5A6",   # accuracy — grey
            "#E67E22",   # precision
            "#27AE60",   # recall — most important
            r["color"],  # f1 — metric color
            "#2ECC71",   # roc-auc
            "#F39C12",   # avg prec
        ]
        bars = ax.bar(metrics_names, vals, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold"
            )
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title(f"{r['label']}\n(thresh: {r['threshold']}%)",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8, rotation=20)

    plt.suptitle(
        "LSTM Forecaster — Classification Metrics Summary",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_metrics_summary.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Metrics summary          → {path}")


def plot_per_step_metrics(results: dict):
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return

    fig, axes = plt.subplots(2, len(valid),
                             figsize=(6 * len(valid), 8))
    if len(valid) == 1:
        axes = axes[:, np.newaxis]

    steps_sec = [(t + 1) * LOG_INTERVAL for t in range(HORIZON)]

    for col, (feat, r) in enumerate(valid.items()):
        step_mets = [sm for sm in r["step_metrics"] if sm is not None]
        if not step_mets:
            continue

        secs  = [sm["seconds"]   for sm in step_mets]
        f1s   = [sm["f1"]        for sm in step_mets]
        precs = [sm["precision"]  for sm in step_mets]
        recs  = [sm["recall"]     for sm in step_mets]
        aucs  = [sm["auc"]       for sm in step_mets
                 if sm["auc"] is not None]
        auc_secs = [sm["seconds"] for sm in step_mets
                    if sm["auc"] is not None]

        # Top: F1, Precision, Recall per step
        ax = axes[0, col]
        ax.plot(secs, f1s,   label="F1",        color=r["color"],  linewidth=2)
        ax.plot(secs, precs, label="Precision",  color="#E67E22",   linewidth=1.5, linestyle="--")
        ax.plot(secs, recs,  label="Recall",     color="#27AE60",   linewidth=1.5, linestyle=":")
        ax.set_title(f"{r['label']}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Bottom: AUC per step
        ax2 = axes[1, col]
        if aucs:
            ax2.plot(auc_secs, aucs, color=r["color"], linewidth=2, marker="o", markersize=4)
            ax2.axhline(0.5, color="#555", linestyle="--", linewidth=1, label="Random")
            ax2.set_title(f"ROC-AUC per Step")
            ax2.set_xlabel("Seconds ahead")
            ax2.set_ylabel("AUC")
            ax2.set_ylim(0, 1.05)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

    plt.suptitle(
        "LSTM Forecaster — Classification Metrics per Horizon Step",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_per_step_metrics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Per-step metrics         → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LSTM Forecaster Classification Evaluation"
    )
    parser.add_argument(
        "--mode", choices=["statistical", "user"], default="statistical",
        help="Threshold mode: statistical (auto mean+1.5σ) or user (manual)"
    )
    parser.add_argument("--cpu-warn",  type=float, default=None,
                        help="CPU warning threshold %%")
    parser.add_argument("--cpu-crit",  type=float, default=None,
                        help="CPU critical threshold %%")
    parser.add_argument("--mem-warn",  type=float, default=None,
                        help="Memory warning threshold %%")
    parser.add_argument("--mem-crit",  type=float, default=None,
                        help="Memory critical threshold %%")
    parser.add_argument("--disk-warn", type=float, default=None,
                        help="Disk warning threshold %%")
    parser.add_argument("--disk-crit", type=float, default=None,
                        help="Disk critical threshold %%")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        sys.exit(f"❌ {MODEL_PATH} not found. Train forecaster first.")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"   Scaler: {SCALER_PATH.name}")

    model = LSTMForecaster(
        input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        dropout=DROPOUT, lookback=LOOKBACK, horizon=HORIZON,
    )
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n📡 Loading data...")
    raw, source = load_data()
    print(f"   Source : {source}")
    print(f"   Rows   : {len(raw):,}")

    # ── Thresholds ────────────────────────────────────────────────────────────
    user_provided = any([
        args.cpu_warn, args.cpu_crit,
        args.mem_warn, args.mem_crit,
        args.disk_warn, args.disk_crit,
    ])

    if user_provided or args.mode == "user":
        thresholds = get_thresholds_user(
            args.cpu_warn,  args.cpu_crit,
            args.mem_warn,  args.mem_crit,
            args.disk_warn, args.disk_crit,
        )
        thresh_mode = "user-defined"
    else:
        thresholds = get_thresholds_statistical(raw)
        thresh_mode = "statistical (mean + 1.5σ)"

    print(f"\n   Threshold mode : {thresh_mode}")
    for feat, thresh in thresholds.items():
        if thresh is not None:
            print(f"   {feat:<20}: {thresh}%")

    # ── Normalise + build windows ─────────────────────────────────────────────
    normalised = scaler.transform(raw)
    X, y_true  = build_windows(normalised, stride=3)
    print(f"\n   Eval windows   : {len(X):,}")

    # ── Run forecaster ────────────────────────────────────────────────────────
    print("🔄 Running forecaster...")
    y_pred = run_forecaster(model, X)   # (W, horizon, 3)

    # Inverse transform → % values
    W = len(y_pred)
    pred_pct = np.stack([scaler.inverse_transform(y_pred[:, t, :])
                         for t in range(HORIZON)], axis=1)
    true_pct = np.stack([scaler.inverse_transform(y_true[:, t, :])
                         for t in range(HORIZON)], axis=1)

    # ── Compute classification metrics ────────────────────────────────────────
    results = compute_classification_metrics(pred_pct, true_pct, thresholds)

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(results, thresholds, source, len(X))

    # ── Generate plots ────────────────────────────────────────────────────────
    print("🎨 Generating plots...")
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_confusion_matrices(results)
    plot_metrics_summary(results)
    plot_per_step_metrics(results)

    print(f"\n✅ Classification evaluation complete.")
    print(f"   All plots saved to {SAVE_DIR}/\n")
    print("  Tips:")
    print("  • Run with your actual limits:")
    print("    python -m model.evaluate_forecaster_classification \\")
    print("      --cpu-warn 50 --mem-warn 38 --disk-warn 25")
    print("  • Use statistical mode for auto-thresholds:")
    print("    python -m model.evaluate_forecaster_classification --mode statistical")


if __name__ == "__main__":
    main()