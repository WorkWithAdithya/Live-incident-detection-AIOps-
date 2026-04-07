"""
train_forecaster.py
-------------------
Trains the LSTM Forecaster on synthetic_logs.csv.

Key design decisions:
  - Uses EXISTING scaler.pkl (never refits) — same normalisation as autoencoder
  - Teacher forcing schedule: starts at 0.5, decays to 0.0 over training
    (helps the model learn without becoming dependent on ground truth at inference)
  - Trains on ALL windows (including anomalies) — forecaster must predict
    both normal and anomalous future values accurately
  - Saves only the model weights (scaler already exists)

Outputs to ai_model/saved/:
    lstm_forecaster.pth     ← model weights
    forecaster_loss.png     ← training curve

Usage:
    cd ai_model
    python -m model.train_forecaster
    python -m model.train_forecaster --epochs 100 --batch-size 64
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..")
sys.path.insert(0, _ROOT)

from model.lstm_forecaster    import LSTMForecaster
from model.forecaster_dataset import ForecasterDataset

# ── Hyperparameters ───────────────────────────────────────────────────────────
LOOKBACK     = 60       # input window  — must match LSTM Autoencoder
HORIZON      = 12       # forecast steps — 12 × 5s = 60s ahead
STRIDE       = 1        # sliding window stride
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
VAL_SPLIT    = 0.10

# Teacher forcing: probability of using true target as next decoder input
# Decays from TF_START → 0.0 linearly over TF_DECAY_EPOCHS
TF_START       = 0.5
TF_DECAY_EPOCHS = 30    # by epoch 30, teacher forcing is 0

SAVE_DIR = os.path.join(_ROOT, "saved")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def teacher_forcing_ratio(epoch: int) -> float:
    """Linear decay from TF_START to 0 over TF_DECAY_EPOCHS."""
    if epoch >= TF_DECAY_EPOCHS:
        return 0.0
    return TF_START * (1.0 - epoch / TF_DECAY_EPOCHS)


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\n🖥️  Device : {DEVICE}")
    print(f"   Lookback: {LOOKBACK} steps ({LOOKBACK * 5}s history)")
    print(f"   Horizon : {HORIZON} steps ({HORIZON * 5}s = {HORIZON * 5 // 60}m {HORIZON * 5 % 60}s forecast)\n")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("📂 Loading ForecasterDataset (using existing scaler.pkl)...")
    dataset = ForecasterDataset(
        lookback  = LOOKBACK,
        horizon   = HORIZON,
        stride    = STRIDE,
    )

    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"   Train windows : {train_size:,}")
    print(f"   Val   windows : {val_size:,}\n")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = LSTMForecaster(
        input_size  = 3,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        lookback    = LOOKBACK,
        horizon     = HORIZON,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params  : {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ── 3. Training loop ──────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"🏋️  Training for {epochs} epochs...\n")
    print(f"{'─'*65}")
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  {'TF Ratio':>10}  {'LR':>10}")
    print(f"{'─'*65}")

    for epoch in range(1, epochs + 1):
        tf_ratio = teacher_forcing_ratio(epoch - 1)

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)   # (B, lookback, 3)
            y_batch = y_batch.to(DEVICE)   # (B, horizon,  3)

            optimizer.zero_grad()

            # Forward with teacher forcing during training
            y_pred = model(
                X_batch,
                teacher_forcing = tf_ratio,
                target          = y_batch,
            )   # (B, horizon, 3)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_batch_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                # No teacher forcing at validation
                y_pred  = model(X_batch, teacher_forcing=0.0)
                loss    = criterion(y_pred, y_batch)
                val_batch_losses.append(loss.item())

        val_loss = float(np.mean(val_batch_losses))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # ── Checkpoint ──────────────────────────────────────────────────────
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "lstm_forecaster.pth")
            )
            saved = " 💾"

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  {epoch:5d}  {train_loss:12.6f}  {val_loss:10.6f}  "
                f"{tf_ratio:10.3f}  {current_lr:10.2e}{saved}"
            )

    print(f"{'─'*65}")
    print(f"\n✅ Training complete!")
    print(f"   Best val loss : {best_val_loss:.6f}")
    print(f"   Saved         → {SAVE_DIR}/lstm_forecaster.pth\n")

    # ── 4. Loss curve plot ────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs + 1), train_losses,
             label="Train Loss", linewidth=2, color="#4A90D9")
    plt.plot(range(1, epochs + 1), val_losses,
             label="Val Loss",   linewidth=2, color="#E67E22")

    # Mark where teacher forcing ends
    if TF_DECAY_EPOCHS < epochs:
        plt.axvline(
            TF_DECAY_EPOCHS, color="#888", linestyle=":", linewidth=1.2,
            label=f"TF ends (epoch {TF_DECAY_EPOCHS})"
        )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"LSTM Forecaster — Training Curve  "
              f"(horizon={HORIZON} steps = {HORIZON*5}s)")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, "forecaster_loss.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Loss plot → {plot_path}")

    # ── 5. Quick sanity check on a single batch ───────────────────────────────
    print("\n🔍 Sanity check — sample predictions vs actuals:")
    model.load_state_dict(
        torch.load(
            os.path.join(SAVE_DIR, "lstm_forecaster.pth"),
            map_location=DEVICE,
            weights_only=True,
        )
    )
    model.eval()

    X_sample, y_sample = dataset[0]
    X_t = X_sample.unsqueeze(0).to(DEVICE)   # (1, lookback, 3)

    with torch.no_grad():
        y_pred = model.predict(X_t).squeeze(0).cpu().numpy()   # (horizon, 3)

    y_true = y_sample.numpy()   # (horizon, 3)
    scaler = dataset.scaler

    # Inverse transform to % values for readability
    y_pred_pct = scaler.inverse_transform(y_pred)
    y_true_pct = scaler.inverse_transform(y_true)

    print(f"\n  {'Step':>4}  {'Pred CPU':>9} {'True CPU':>9}  "
          f"{'Pred MEM':>9} {'True MEM':>9}  "
          f"{'Pred DISK':>10} {'True DISK':>10}")
    print(f"  {'─'*4}  {'─'*9} {'─'*9}  {'─'*9} {'─'*9}  {'─'*10} {'─'*10}")

    for i in range(min(6, HORIZON)):
        print(
            f"  {i+1:4d}  "
            f"{y_pred_pct[i,0]:8.2f}% {y_true_pct[i,0]:8.2f}%  "
            f"{y_pred_pct[i,1]:8.2f}% {y_true_pct[i,1]:8.2f}%  "
            f"{y_pred_pct[i,2]:9.2f}% {y_true_pct[i,2]:9.2f}%"
        )

    # Overall MAE in % points
    mae_cpu  = float(np.mean(np.abs(y_pred_pct[:,0] - y_true_pct[:,0])))
    mae_mem  = float(np.mean(np.abs(y_pred_pct[:,1] - y_true_pct[:,1])))
    mae_disk = float(np.mean(np.abs(y_pred_pct[:,2] - y_true_pct[:,2])))
    print(f"\n  MAE (% points) — CPU: {mae_cpu:.2f}  MEM: {mae_mem:.2f}  DISK: {mae_disk:.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Forecaster")
    parser.add_argument("--epochs",     type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)