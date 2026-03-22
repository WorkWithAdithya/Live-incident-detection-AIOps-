"""
train.py
--------
Trains the LSTM Autoencoder on synthetic_logs.csv.

Key design: trains ONLY on normal windows — so the model
learns exclusively what healthy system behaviour looks like.
Anomalous windows are excluded from training but used during
threshold calibration and evaluation.

Outputs (saved to ../saved/):
    lstm_autoencoder.pth    ← best model weights (by val loss)
    scaler.pkl              ← fitted MinMaxScaler (needed at inference)
    threshold.txt           ← anomaly decision boundary
    training_loss.png       ← loss curve plot

Usage:
    python -m model.train
    python -m model.train --epochs 100 --batch-size 32
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model.lstm_autoencoder import LSTMAutoencoder
from model.dataset import SystemLogsDataset

# ── Defaults ─────────────────────────────────────────────────────────────────
SEQ_LEN      = 60
STRIDE       = 1
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
HIDDEN_SIZE  = 64
LATENT_SIZE  = 32
NUM_LAYERS   = 2
DROPOUT      = 0.2
VAL_SPLIT    = 0.10
THRESHOLD_K  = 3.0      # threshold = mean_error + K * std_error

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epochs=EPOCHS, batch_size=BATCH_SIZE):
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\n🖥️  Device : {DEVICE}")

    # ── 1. Load dataset ───────────────────────────────────────────────────
    print("\n📂 Loading synthetic training data...")
    full_dataset = SystemLogsDataset(seq_len=SEQ_LEN, stride=STRIDE)

    # Train ONLY on normal windows — core principle of autoencoder anomaly detection
    normal_dataset = full_dataset.get_normal_subset()

    val_size   = max(1, int(len(normal_dataset) * VAL_SPLIT))
    train_size = len(normal_dataset) - val_size
    train_ds, val_ds = random_split(normal_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n   Total windows  : {len(full_dataset):,}")
    print(f"   Training on    : {train_size:,} normal windows")
    print(f"   Validating on  : {val_size:,} normal windows")

    # ── 2. Model, loss, optimiser ─────────────────────────────────────────
    model = LSTMAutoencoder(
        input_size  = 3,
        hidden_size = HIDDEN_SIZE,
        latent_size = LATENT_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        seq_len     = SEQ_LEN,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params   : {total_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ── 3. Training loop ──────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"🏋️  Training for {epochs} epochs...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        batch_losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon = model(batch)
                loss  = criterion(recon, batch)
                val_batch_losses.append(loss.item())

        val_loss = float(np.mean(val_batch_losses))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_autoencoder.pth"))
            status = "  💾 saved"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  |  "
                  f"Train: {train_loss:.6f}  |  Val: {val_loss:.6f}{status}")

    print("-" * 65)
    print(f"✅ Best val loss : {best_val_loss:.6f}")

    # ── 4. Save scaler ────────────────────────────────────────────────────
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(full_dataset.scaler, f)
    print(f"💾 Scaler saved  → {scaler_path}")

    # ── 5. Calibrate anomaly threshold on normal training windows ─────────
    print("\n📐 Calibrating anomaly threshold on training data...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "lstm_autoencoder.pth"),
                                     map_location=DEVICE))
    model.eval()

    all_errors = []
    cal_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in cal_loader:
            batch  = batch.to(DEVICE)
            errors = model.reconstruction_error(batch)
            all_errors.extend(errors.cpu().numpy().tolist())

    all_errors = np.array(all_errors)
    mean_err   = float(all_errors.mean())
    std_err    = float(all_errors.std())
    threshold  = mean_err + THRESHOLD_K * std_err

    threshold_path = os.path.join(SAVE_DIR, "threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(f"{threshold:.8f}\n")
        f.write(f"mean={mean_err:.8f}\n")
        f.write(f"std={std_err:.8f}\n")
        f.write(f"k={THRESHOLD_K}\n")

    print(f"   Mean error  : {mean_err:.6f}")
    print(f"   Std  error  : {std_err:.6f}")
    print(f"   Threshold   : {threshold:.6f}  (mean + {THRESHOLD_K}σ)")
    print(f"💾 Threshold    → {threshold_path}")

    # ── 6. Loss curve plot ────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss", linewidth=2, color="#4A90D9")
    plt.plot(range(1, epochs+1), val_losses,   label="Val Loss",   linewidth=2, color="#E67E22")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"Anomaly Threshold ({threshold:.5f})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Autoencoder — Training Curve (Synthetic Data)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "training_loss.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Loss plot     → {plot_path}")
    print("\n✅ Training complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)