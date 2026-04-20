"""
train_forecaster.py  (UPDATED — 5-minute horizon on synthetic data)
--------------------------------------------------------------------
Trains the LSTM Forecaster on synthetic_logs.csv.

KEY CHANGE from original:
  LOG_INTERVAL_SEC = 1   (your real logger runs at 1s)
  HORIZON  = 300         (300 × 1s = 5 minutes ahead)
  LOOKBACK = 300         (300 × 1s = 5 minutes of history)

With 10,000 synthetic rows:
  Training windows = 10000 - 300 - 300 + 1 = 9,401 windows  ✅ plenty

The model learns temporal patterns from synthetic data
(CPU spikes, memory leaks, disk fills) and generalises to
predict whether real metrics will breach thresholds in the
next 5 minutes.

Usage:
    cd ai_model
    python -m model.train_forecaster
    python -m model.train_forecaster --epochs 100
    python -m model.train_forecaster --horizon 180  # 3 min ahead
    python -m model.train_forecaster --horizon 60   # 1 min ahead
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path

_DIR  = Path(__file__).resolve().parent
_ROOT = _DIR.parent
sys.path.insert(0, str(_ROOT))

from model.lstm_forecaster    import LSTMForecaster
from model.forecaster_dataset import ForecasterDataset

# ── Hyperparameters ───────────────────────────────────────────────────────────
LOG_INTERVAL_SEC = 1      # matches LOG_INTERVAL_SECONDS=1 in your .env

LOOKBACK     = 300        # 300 × 1s = 5 minutes of history
HORIZON      = 300        # 300 × 1s = 5 minutes ahead   ← KEY CHANGE
STRIDE       = 1
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
VAL_SPLIT    = 0.10

TF_START        = 0.5
TF_DECAY_EPOCHS = 30

SAVE_DIR = _ROOT / "saved"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def teacher_forcing_ratio(epoch: int) -> float:
    if epoch >= TF_DECAY_EPOCHS:
        return 0.0
    return TF_START * (1.0 - epoch / TF_DECAY_EPOCHS)


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
          horizon: int = HORIZON, lookback: int = LOOKBACK):

    os.makedirs(SAVE_DIR, exist_ok=True)

    forecast_sec = horizon * LOG_INTERVAL_SEC
    history_sec  = lookback * LOG_INTERVAL_SEC

    print(f"\n🖥️  Device   : {DEVICE}")
    print(f"   Horizon  : {horizon} steps × {LOG_INTERVAL_SEC}s = "
          f"{forecast_sec}s ({forecast_sec//60}m {forecast_sec%60}s ahead)")
    print(f"   Lookback : {lookback} steps × {LOG_INTERVAL_SEC}s = "
          f"{history_sec}s of history")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("\n📂 Loading synthetic data...")
    dataset = ForecasterDataset(
        lookback = lookback,
        horizon  = horizon,
        stride   = STRIDE,
    )

    print(f"   Total windows : {len(dataset):,}")

    if len(dataset) < 100:
        print(
            f"\n❌ Only {len(dataset)} training windows — not enough.\n"
            f"   With horizon={horizon} and lookback={lookback} you need at least\n"
            f"   {lookback + horizon + 100:,} rows in synthetic_logs.csv.\n"
            f"   Run: python data/generate_synthetic_data.py first."
        )
        sys.exit(1)

    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"   Train windows : {train_size:,}")
    print(f"   Val windows   : {val_size:,}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = LSTMForecaster(
        input_size  = 3,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        lookback    = lookback,
        horizon     = horizon,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params  : {total_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ── 3. Training loop ──────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"🏋️  Training for {epochs} epochs...")
    print(f"{'─'*60}")
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'TF':>6}  {'LR':>10}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        tf = teacher_forcing_ratio(epoch - 1)

        model.train()
        batch_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_b, teacher_forcing=tf, target=y_b)
            loss   = criterion(y_pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                y_pred   = model(X_b, teacher_forcing=0.0)
                val_batch_losses.append(criterion(y_pred, y_b).item())

        val_loss = float(np.mean(val_batch_losses))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       SAVE_DIR / "lstm_forecaster.pth")
            saved = "  💾"

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {train_loss:12.6f}  {val_loss:10.6f}  "
                  f"{tf:6.3f}  {current_lr:10.2e}{saved}")

    print(f"{'─'*60}")
    print(f"\n✅ Training complete!  Best val loss: {best_val_loss:.6f}")

    # ── 4. Save forecaster config ─────────────────────────────────────────────
    config_path = SAVE_DIR / "forecaster_config.txt"
    with open(config_path, "w") as f:
        f.write(f"horizon={horizon}\n")
        f.write(f"lookback={lookback}\n")
        f.write(f"log_interval_sec={LOG_INTERVAL_SEC}\n")
        f.write(f"forecast_seconds={forecast_sec}\n")
        f.write(f"forecast_minutes={forecast_sec / 60:.1f}\n")

    print(f"💾 Model    → {SAVE_DIR}/lstm_forecaster.pth")
    print(f"💾 Config   → {config_path}")
    print(f"\n   Forecasts {forecast_sec}s "
          f"({forecast_sec//60}m {forecast_sec%60}s) ahead\n")

    # ── 5. Loss plot ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs+1), train_losses,
             label="Train Loss", linewidth=2, color="#4A90D9")
    plt.plot(range(1, epochs+1), val_losses,
             label="Val Loss",   linewidth=2, color="#E67E22")
    if TF_DECAY_EPOCHS < epochs:
        plt.axvline(TF_DECAY_EPOCHS, color="#888", linestyle=":",
                    linewidth=1.2,
                    label=f"TF ends (epoch {TF_DECAY_EPOCHS})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(
        f"LSTM Forecaster — Horizon: {forecast_sec}s ahead "
        f"({forecast_sec//60}m {forecast_sec%60}s)"
    )
    plt.legend()
    plt.tight_layout()
    plot_path = SAVE_DIR / "forecaster_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Loss plot → {plot_path}")

    # ── 6. Sanity check ───────────────────────────────────────────────────────
    print("\n🔍 Sanity check — sample prediction:")
    model.load_state_dict(
        torch.load(str(SAVE_DIR / "lstm_forecaster.pth"),
                   map_location=DEVICE, weights_only=True)
    )
    model.eval()

    X_s, y_s = dataset[0]
    with torch.no_grad():
        y_pred = model.predict(
            X_s.unsqueeze(0).to(DEVICE)
        ).squeeze(0).cpu().numpy()

    scaler      = dataset.scaler
    y_true_pct  = scaler.inverse_transform(y_s.numpy())
    y_pred_pct  = scaler.inverse_transform(y_pred)

    print(f"\n  {'Step':>4}  {'Sec':>5}  "
          f"{'PredCPU':>8} {'TrueCPU':>8}  "
          f"{'PredMEM':>8} {'TrueMEM':>8}  "
          f"{'PredDISK':>9} {'TrueDISK':>9}")
    print(f"  {'─'*4}  {'─'*5}  "
          f"{'─'*8} {'─'*8}  "
          f"{'─'*8} {'─'*8}  "
          f"{'─'*9} {'─'*9}")

    # Show first 5 steps and last 5 steps
    show_idx = list(range(5)) + list(range(horizon-5, horizon))
    for i in show_idx:
        if i == 5:
            print(f"  {'...':^4}  {'...':^5}  "
                  f"{'...':^8} {'...':^8}  "
                  f"{'...':^8} {'...':^8}  "
                  f"{'...':^9} {'...':^9}")
        secs = (i+1) * LOG_INTERVAL_SEC
        print(f"  {i+1:4d}  {secs:5d}  "
              f"{y_pred_pct[i,0]:7.2f}% {y_true_pct[i,0]:7.2f}%  "
              f"{y_pred_pct[i,1]:7.2f}% {y_true_pct[i,1]:7.2f}%  "
              f"{y_pred_pct[i,2]:8.2f}% {y_true_pct[i,2]:8.2f}%")

    mae = np.mean(np.abs(y_pred_pct - y_true_pct), axis=0)
    print(f"\n  MAE — CPU: {mae[0]:.2f}%  "
          f"MEM: {mae[1]:.2f}%  DISK: {mae[2]:.2f}%")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--horizon", type=int, default=HORIZON,
        help=f"Forecast steps (default: {HORIZON} = "
             f"{HORIZON * LOG_INTERVAL_SEC}s = "
             f"{HORIZON * LOG_INTERVAL_SEC // 60}m ahead)"
    )
    parser.add_argument(
        "--lookback", type=int, default=LOOKBACK,
        help=f"History steps (default: {LOOKBACK} = "
             f"{LOOKBACK * LOG_INTERVAL_SEC}s)"
    )
    args = parser.parse_args()
    train(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        horizon    = args.horizon,
        lookback   = args.lookback,
    )