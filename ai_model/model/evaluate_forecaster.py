"""
evaluate_forecaster.py
----------------------
Evaluates the trained LSTM Forecaster on the full synthetic dataset.

Metrics per metric (cpu, memory, disk):
    MAE   — Mean Absolute Error (in % points after inverse transform)
    RMSE  — Root Mean Squared Error
    MAPE  — Mean Absolute Percentage Error

Metrics per forecast step (step 1 through 12):
    MAE at each step — shows how accuracy degrades further into the future

Plots saved to ai_model/saved/:
    forecaster_evaluation.png   — MAE per horizon step
    forecaster_predictions.png  — sample actual vs predicted trajectories

Usage:
    cd ai_model
    python -m model.evaluate_forecaster
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..")
sys.path.insert(0, _ROOT)

from model.lstm_forecaster    import LSTMForecaster
from model.forecaster_dataset import ForecasterDataset

LOOKBACK   = 60
HORIZON    = 12
BATCH_SIZE = 256
SAVE_DIR   = os.path.join(_ROOT, "saved")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES   = ["CPU", "Memory", "Disk"]
COLORS     = ["#a78bfa", "#38bdf8", "#fb923c"]


def load_model() -> LSTMForecaster:
    path = os.path.join(SAVE_DIR, "lstm_forecaster.pth")
    if not os.path.exists(path):
        sys.exit(f"❌ lstm_forecaster.pth not found. Run train_forecaster.py first.")
    model = LSTMForecaster(
        input_size=3, hidden_size=128, num_layers=2,
        dropout=0.2, lookback=LOOKBACK, horizon=HORIZON
    )
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    return model


def main():
    print("📂 Loading dataset and model...")
    dataset = ForecasterDataset(lookback=LOOKBACK, horizon=HORIZON, stride=5)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model   = load_model()
    scaler  = dataset.scaler

    # ── Run all predictions ───────────────────────────────────────────────────
    all_pred, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model.predict(X_batch.to(DEVICE))   # (B, horizon, 3)
            all_pred.append(y_pred.cpu().numpy())
            all_true.append(y_batch.numpy())

    all_pred = np.concatenate(all_pred, axis=0)   # (N, horizon, 3)
    all_true = np.concatenate(all_true, axis=0)   # (N, horizon, 3)

    # ── Inverse transform to % ────────────────────────────────────────────────
    N = all_pred.shape[0]
    pred_pct = np.stack([
        scaler.inverse_transform(all_pred[:, t, :]) for t in range(HORIZON)
    ], axis=1)   # (N, horizon, 3)

    true_pct = np.stack([
        scaler.inverse_transform(all_true[:, t, :]) for t in range(HORIZON)
    ], axis=1)   # (N, horizon, 3)

    # ── Overall metrics ───────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  LSTM FORECASTER — EVALUATION REPORT")
    print(f"{'='*58}")
    print(f"  Total windows : {N:,}")
    print(f"  Horizon       : {HORIZON} steps × 5s = {HORIZON*5}s ahead\n")

    print(f"  {'Metric':<10}  {'MAE (%)':>9}  {'RMSE (%)':>9}  {'MAPE (%)':>9}")
    print(f"  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}")

    for i, name in enumerate(FEATURES):
        diff = pred_pct[:, :, i] - true_pct[:, :, i]
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        # MAPE — avoid division by zero
        nonzero = true_pct[:, :, i] != 0
        mape = float(np.mean(
            np.abs(diff[nonzero] / true_pct[:, :, i][nonzero])
        )) * 100
        print(f"  {name:<10}  {mae:9.3f}  {rmse:9.3f}  {mape:9.2f}")

    print()

    # ── Per-step MAE ──────────────────────────────────────────────────────────
    print(f"  {'Step':>4}  {'Time':>7}  {'CPU MAE':>9}  {'MEM MAE':>9}  {'DISK MAE':>10}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*10}")

    step_maes = np.zeros((HORIZON, 3))
    for t in range(HORIZON):
        for i in range(3):
            step_maes[t, i] = np.mean(np.abs(
                pred_pct[:, t, i] - true_pct[:, t, i]
            ))
        print(
            f"  {t+1:4d}  {(t+1)*5:>5}s    "
            f"{step_maes[t,0]:9.3f}  {step_maes[t,1]:9.3f}  {step_maes[t,2]:10.3f}"
        )

    print(f"{'='*58}\n")

    # ── Plot 1: MAE per horizon step ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    steps = [(t+1)*5 for t in range(HORIZON)]   # seconds ahead

    for i, (name, color, ax) in enumerate(zip(FEATURES, COLORS, axes)):
        ax.plot(steps, step_maes[:, i], marker="o", linewidth=2,
                color=color, markersize=5)
        ax.fill_between(steps, step_maes[:, i], alpha=0.15, color=color)
        ax.set_xlabel("Seconds ahead")
        ax.set_ylabel("MAE (% points)")
        ax.set_title(f"{name} — MAE vs Forecast Horizon")
        ax.grid(alpha=0.3)
        ax.set_xticks(steps)

    plt.suptitle("LSTM Forecaster — Accuracy Degrades with Horizon",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, "forecaster_evaluation.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"📊 Evaluation plot → {p1}")

    # ── Plot 2: Sample predictions vs actuals ─────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    sample_idx = 100   # pick a mid-dataset example

    for i, (name, color, ax) in enumerate(zip(FEATURES, COLORS, axes)):
        t_axis = [(t+1)*5 for t in range(HORIZON)]
        ax.plot(t_axis, true_pct[sample_idx, :, i],
                label="Actual", linewidth=2, color=color)
        ax.plot(t_axis, pred_pct[sample_idx, :, i],
                label="Predicted", linewidth=2, color=color,
                linestyle="--", alpha=0.8)
        ax.fill_between(
            t_axis,
            pred_pct[sample_idx, :, i] - step_maes[:, i],
            pred_pct[sample_idx, :, i] + step_maes[:, i],
            alpha=0.1, color=color, label="±MAE band"
        )
        ax.set_ylabel(f"{name} (%)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Seconds ahead")
    plt.suptitle("LSTM Forecaster — Sample Prediction vs Actual",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, "forecaster_predictions.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"📊 Prediction plot → {p2}")
    print("✅ Evaluation complete.\n")


if __name__ == "__main__":
    main()