"""
evaluate_forecaster_live.py
---------------------------
Evaluates the LSTM Forecaster against REAL system logs from NeonDB.

Unlike evaluate_forecaster.py (which uses synthetic CSV), this script:
  - Pulls actual rows from your NeonPostgreSQL system_logs table
  - Builds (input, target) pairs from real data
  - Runs the forecaster and compares predictions to what actually happened
  - Reports MAE, RMSE, MAPE per metric and per horizon step
  - Saves evaluation plots to ai_model/saved/

This tells you how well the model trained on synthetic data
generalises to your real system's actual behaviour.

Usage:
    cd ai_model
    python -m model.evaluate_forecaster_live
    python -m model.evaluate_forecaster_live --min-rows 200
"""

import os
import sys
import pickle
import argparse
import numpy as np
import psycopg2
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).resolve().parent          # ai_model/model/
_ROOT = _DIR.parent                              # ai_model/

sys.path.insert(0, str(_ROOT))

# Load .env from log_generator/
_ENV_PATH = _ROOT.parent / "log_generator" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)
DATABASE_URL = os.getenv("DATABASE_URL")

from model.lstm_forecaster import LSTMForecaster

SAVE_DIR        = _ROOT / "saved"
MODEL_PATH      = SAVE_DIR / "lstm_forecaster.pth"
SCALER_PATH     = SAVE_DIR / "scaler.pkl"

LOOKBACK        = 60
HORIZON         = 24
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.2
LOG_INTERVAL    = 5    # seconds between readings

FEATURES        = ["cpu_usage", "memory_usage", "disk_usage"]
FEATURE_LABELS  = ["CPU Usage", "Memory Usage", "Disk Usage"]
FEATURE_COLORS  = ["#a78bfa", "#38bdf8", "#fb923c"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── DB fetch ──────────────────────────────────────────────────────────────────

def fetch_real_logs(min_rows: int = 200) -> tuple:
    """
    Fetches all rows from system_logs ordered by timestamp ASC.
    Returns (raw np.ndarray (N,3), timestamps list).
    """
    if not DATABASE_URL:
        raise EnvironmentError(
            "DATABASE_URL not set. Check log_generator/.env"
        )

    conn   = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, cpu_usage, memory_usage, disk_usage
        FROM system_logs
        ORDER BY timestamp ASC
        """
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(rows) < LOOKBACK + HORIZON:
        raise ValueError(
            f"Not enough real data to evaluate.\n"
            f"Need at least {LOOKBACK + HORIZON} rows, have {len(rows)}.\n"
            f"Let the logger run longer and try again."
        )

    if len(rows) < min_rows:
        print(
            f"⚠️  Only {len(rows)} rows available "
            f"(recommended: {min_rows}+). "
            f"Results may not be representative."
        )

    timestamps = [r[0] for r in rows]
    raw = np.array(
        [[float(r[1]), float(r[2]), float(r[3])] for r in rows],
        dtype=np.float32
    )
    return raw, timestamps


# ── Build evaluation pairs ────────────────────────────────────────────────────

def build_eval_pairs(normalised: np.ndarray, stride: int = 5):
    """
    Builds (X, y) pairs from real normalised data.
    stride=5 means every 5th window (avoids massive redundancy).
    """
    X_list, y_list = [], []
    total = LOOKBACK + HORIZON
    n     = len(normalised)

    for start in range(0, n - total + 1, stride):
        X_list.append(normalised[start            : start + LOOKBACK])
        y_list.append(normalised[start + LOOKBACK : start + total])

    return (
        np.array(X_list, dtype=np.float32),   # (W, lookback, 3)
        np.array(y_list, dtype=np.float32),   # (W, horizon,  3)
    )


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(min_rows: int = 200):
    print(f"\n{'='*60}")
    print("  LSTM Forecaster — Live Data Evaluation")
    print(f"{'='*60}")
    print(f"  Device   : {DEVICE}")
    print(f"  Lookback : {LOOKBACK} steps ({LOOKBACK * LOG_INTERVAL}s)")
    print(f"  Horizon  : {HORIZON} steps ({HORIZON * LOG_INTERVAL}s ahead)\n")

    # ── 1. Load artefacts ─────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        sys.exit(f"❌ {MODEL_PATH} not found. Run python -m model.train_forecaster first.")
    if not SCALER_PATH.exists():
        sys.exit(f"❌ {SCALER_PATH} not found. Run python -m model.train first.")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = LSTMForecaster(
        input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        dropout=DROPOUT, lookback=LOOKBACK, horizon=HORIZON,
    )
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    # ── 2. Fetch real data ────────────────────────────────────────────────────
    print("📡 Fetching real logs from NeonDB...")
    raw, timestamps = fetch_real_logs(min_rows=min_rows)
    print(f"   Fetched  : {len(raw):,} rows")
    print(f"   From     : {timestamps[0]}")
    print(f"   To       : {timestamps[-1]}")
    print(f"   Duration : {(timestamps[-1] - timestamps[0])}")

    # ── 3. Normalise with existing scaler ─────────────────────────────────────
    normalised = scaler.transform(raw)   # (N, 3)

    # ── 4. Build eval windows ─────────────────────────────────────────────────
    X, y_true = build_eval_pairs(normalised, stride=5)
    print(f"\n   Eval windows : {len(X):,}  (stride=5)")

    # ── 5. Run forecaster ─────────────────────────────────────────────────────
    print("🔄 Running forecaster on real data...")
    all_pred = []

    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch     = torch.tensor(X[start:start+batch_size]).to(DEVICE)
            pred_norm = model.predict(batch)
            all_pred.append(pred_norm.cpu().numpy())

    y_pred = np.concatenate(all_pred, axis=0)   # (W, horizon, 3)

    # ── 6. Inverse transform → % values ──────────────────────────────────────
    W = len(y_pred)
    pred_pct = np.stack([
        scaler.inverse_transform(y_pred[:, t, :]) for t in range(HORIZON)
    ], axis=1)   # (W, horizon, 3)

    true_pct = np.stack([
        scaler.inverse_transform(y_true[:, t, :]) for t in range(HORIZON)
    ], axis=1)   # (W, horizon, 3)

    # ── 7. Compute metrics ────────────────────────────────────────────────────
    diff = pred_pct - true_pct   # (W, horizon, 3)

    print(f"\n{'─'*60}")
    print(f"  {'Metric':<14}  {'MAE (%)':>9}  {'RMSE (%)':>9}  {'MAPE (%)':>9}  {'MaxErr (%)':>11}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*11}")

    metric_results = {}
    for i, name in enumerate(FEATURE_LABELS):
        mae   = float(np.mean(np.abs(diff[:, :, i])))
        rmse  = float(np.sqrt(np.mean(diff[:, :, i] ** 2)))
        maxe  = float(np.max(np.abs(diff[:, :, i])))
        # MAPE — avoid division by zero
        mask  = true_pct[:, :, i] > 0.5   # only where value > 0.5%
        mape  = float(np.mean(
            np.abs(diff[:, :, i][mask] / true_pct[:, :, i][mask])
        )) * 100 if mask.any() else 0.0

        metric_results[name] = {
            "mae": mae, "rmse": rmse, "mape": mape, "max_err": maxe
        }
        print(f"  {name:<14}  {mae:9.3f}  {rmse:9.3f}  {mape:9.2f}  {maxe:11.3f}")

    # ── 8. Per-step MAE ───────────────────────────────────────────────────────
    step_maes = np.mean(np.abs(diff), axis=(0, 2))   # (horizon,) — avg across windows and metrics
    step_maes_per = np.mean(np.abs(diff), axis=0)    # (horizon, 3) — per metric

    print(f"\n  Per-step accuracy (avg across all metrics):")
    print(f"  {'Step':>4}  {'Sec':>5}  {'MAE (%)':>9}  Gauge")
    print(f"  {'─'*4}  {'─'*5}  {'─'*9}  {'─'*20}")
    for t in range(HORIZON):
        mae_t = float(step_maes[t])
        bar   = "█" * int(mae_t * 2) + "░" * max(0, 20 - int(mae_t * 2))
        print(f"  {t+1:4d}  {(t+1)*LOG_INTERVAL:5d}  {mae_t:9.3f}  {bar}")

    # ── 9. Grading ────────────────────────────────────────────────────────────
    overall_mae = float(np.mean([v["mae"] for v in metric_results.values()]))
    print(f"\n  Overall MAE : {overall_mae:.3f}%")
    if overall_mae < 3.0:
        grade = "✅ EXCELLENT — model generalises well to real data"
    elif overall_mae < 7.0:
        grade = "🟡 GOOD — minor gap between synthetic training and real behaviour"
    elif overall_mae < 15.0:
        grade = "🟠 FAIR — model predicts direction but values are off"
    else:
        grade = "🔴 POOR — synthetic data doesn't match real system well enough"
    print(f"  Grade       : {grade}")

    print(f"\n{'='*60}\n")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    _plot_per_step_mae(step_maes_per)
    _plot_sample_predictions(pred_pct, true_pct, timestamps, normalised)
    _plot_error_distribution(diff)

    print("✅ Live evaluation complete.")
    print(f"   Plots saved to {SAVE_DIR}/\n")


def _plot_per_step_mae(step_maes_per: np.ndarray):
    """MAE vs forecast horizon per metric."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    steps = [(t + 1) * LOG_INTERVAL for t in range(HORIZON)]

    for i, (name, color, ax) in enumerate(zip(FEATURE_LABELS, FEATURE_COLORS, axes)):
        ax.plot(steps, step_maes_per[:, i],
                marker="o", linewidth=2, color=color, markersize=5)
        ax.fill_between(steps, step_maes_per[:, i], alpha=0.12, color=color)
        ax.set_xlabel("Seconds ahead")
        ax.set_ylabel("MAE (% points)")
        ax.set_title(f"{name}")
        ax.grid(alpha=0.3)
        ax.set_xticks(steps)

    plt.suptitle(
        "LSTM Forecaster — MAE per Horizon Step (Real NeonDB Data)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_live_mae.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 MAE plot            → {path}")


def _plot_sample_predictions(pred_pct, true_pct, timestamps, normalised):
    """Shows 3 sample windows: actual vs predicted trajectory."""
    n_samples  = min(3, len(pred_pct))
    indices    = np.linspace(0, len(pred_pct) - 1, n_samples, dtype=int)
    steps      = [(t + 1) * LOG_INTERVAL for t in range(HORIZON)]

    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row_idx, win_idx in enumerate(indices):
        for col_idx, (name, color) in enumerate(zip(FEATURE_LABELS, FEATURE_COLORS)):
            ax    = axes[row_idx, col_idx]
            true  = true_pct[win_idx, :, col_idx]
            pred  = pred_pct[win_idx, :, col_idx]

            ax.plot(steps, true, label="Actual",    color=color,   linewidth=2)
            ax.plot(steps, pred, label="Predicted", color=color,
                    linewidth=2, linestyle="--", alpha=0.8)

            mae_w = float(np.mean(np.abs(pred - true)))
            ax.fill_between(steps,
                            pred - mae_w, pred + mae_w,
                            alpha=0.12, color=color, label=f"±{mae_w:.1f}%")

            ax.set_title(f"Window {win_idx} — {name}" if row_idx == 0 else name)
            ax.set_xlabel("Seconds ahead")
            ax.set_ylabel("(%)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xticks(steps)

    plt.suptitle(
        "LSTM Forecaster — Sample Predictions vs Actual (Real Data)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_live_samples.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Sample predictions  → {path}")


def _plot_error_distribution(diff: np.ndarray):
    """Histogram of prediction errors per metric."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for i, (name, color, ax) in enumerate(zip(FEATURE_LABELS, FEATURE_COLORS, axes)):
        errors = diff[:, :, i].flatten()
        ax.hist(errors, bins=60, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(np.mean(errors),  color="yellow", linewidth=1.2,
                   label=f"Mean {np.mean(errors):+.2f}%")
        ax.axvline(np.median(errors), color="cyan",  linewidth=1.2,
                   linestyle=":", label=f"Median {np.median(errors):+.2f}%")
        ax.set_xlabel("Prediction Error (% points)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name} Error Distribution")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    plt.suptitle(
        "LSTM Forecaster — Prediction Error Distribution (Real Data)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = SAVE_DIR / "forecaster_live_errors.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Error distribution  → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-rows", type=int, default=200,
        help="Minimum DB rows required (default: 200)"
    )
    args = parser.parse_args()
    evaluate(min_rows=args.min_rows)