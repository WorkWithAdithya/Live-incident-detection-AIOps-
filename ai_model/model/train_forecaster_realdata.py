"""
train_forecaster_realdata.py  (UPDATED — configurable horizon)
--------------------------------------------------------------
Trains the LSTM Forecaster on real NeonDB data with a configurable
prediction horizon.

Default: HORIZON=300 → predicts 5 minutes ahead (at 1s interval)

Minimum data needed:
    horizon=60  (1 min)  → ~400 rows  (~7 min logging)
    horizon=180 (3 min)  → ~800 rows  (~14 min logging)
    horizon=300 (5 min)  → ~1200 rows (~20 min logging)
    horizon=600 (10 min) → ~2400 rows (~40 min logging)

Usage:
    cd ai_model

    # 5 minutes ahead (default):
    python -m model.train_forecaster_realdata

    # 1 minute ahead (less data needed):
    python -m model.train_forecaster_realdata --horizon 60 --lookback 120

    # 3 minutes ahead:
    python -m model.train_forecaster_realdata --horizon 180 --lookback 180

    # Check how many rows you have first:
    python -m model.train_forecaster_realdata --check-only
"""

import os
import sys
import pickle
import argparse
import numpy as np
import psycopg2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).resolve().parent
_ROOT = _DIR.parent

sys.path.insert(0, str(_ROOT))

_ENV_PATH = _ROOT.parent / "log_generator" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)
DATABASE_URL     = os.getenv("DATABASE_URL")
LOG_INTERVAL_SEC = int(os.getenv("LOG_INTERVAL_SECONDS", 1))

from model.lstm_forecaster import LSTMForecaster

SAVE_DIR = _ROOT / "saved"

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULT_HORIZON  = 300    # 300 × 1s = 5 minutes ahead
DEFAULT_LOOKBACK = 300    # 300 × 1s = 5 minutes of history
STRIDE           = 1
BATCH_SIZE       = 32
EPOCHS           = 100
LR               = 5e-4
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.2
VAL_SPLIT        = 0.15
TF_START         = 0.5
TF_DECAY_EPOCHS  = 40
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES         = ["cpu_usage", "memory_usage", "disk_usage"]


# ── DB fetch ──────────────────────────────────────────────────────────────────

def fetch_all_logs() -> np.ndarray:
    conn   = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT cpu_usage, memory_usage, disk_usage "
        "FROM system_logs ORDER BY timestamp ASC"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return np.array([[float(r[0]), float(r[1]), float(r[2])]
                     for r in rows], dtype=np.float32)


def check_data_sufficiency(rows: int, lookback: int, horizon: int):
    """Prints a clear report of whether you have enough data."""
    needed     = lookback + horizon + 200
    windows    = max(0, rows - lookback - horizon + 1)
    minutes    = rows * LOG_INTERVAL_SEC / 60
    needed_min = needed * LOG_INTERVAL_SEC / 60

    print(f"\n{'─'*55}")
    print(f"  Data sufficiency check")
    print(f"{'─'*55}")
    print(f"  Current rows      : {rows:,}  ({minutes:.1f} min of data)")
    print(f"  Lookback          : {lookback} steps = {lookback*LOG_INTERVAL_SEC}s")
    print(f"  Horizon           : {horizon} steps = "
          f"{horizon*LOG_INTERVAL_SEC}s ({horizon*LOG_INTERVAL_SEC//60}m "
          f"{horizon*LOG_INTERVAL_SEC%60}s ahead)")
    print(f"  Training windows  : {windows:,}")
    print(f"  Minimum needed    : {needed:,} rows ({needed_min:.1f} min of logging)")
    print()

    if windows < 50:
        print(f"  ❌ NOT ENOUGH DATA")
        print(f"     Need {needed - rows:,} more rows "
              f"= {(needed - rows) * LOG_INTERVAL_SEC / 60:.1f} more minutes of logging")
        print()
        print(f"  Options:")
        print(f"    1. Let logger run {(needed-rows)//60 + 1} more minutes, then retrain")
        print(f"    2. Use a shorter horizon:")

        for h, l in [(60, 120), (120, 180), (180, 180)]:
            w = max(0, rows - l - h + 1)
            if w >= 50:
                secs = h * LOG_INTERVAL_SEC
                print(f"       --horizon {h} --lookback {l}  "
                      f"→ {secs}s ({secs//60}m) ahead, {w} windows  ✅")
        return False
    elif windows < 200:
        print(f"  ⚠️  MARGINAL — {windows} windows (recommended: 200+)")
        print(f"     Training will work but accuracy may be limited.")
        print(f"     Let logger run longer for better results.")
        return True
    else:
        print(f"  ✅ SUFFICIENT — {windows} training windows")
        return True


# ── Dataset ───────────────────────────────────────────────────────────────────

class RealDataForecasterDataset(Dataset):
    def __init__(self, normalised: np.ndarray,
                 lookback: int, horizon: int, stride: int):
        self.X, self.y = [], []
        total = lookback + horizon
        for start in range(0, len(normalised) - total + 1, stride):
            self.X.append(normalised[start           : start + lookback])
            self.y.append(normalised[start + lookback : start + total])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ── Teacher forcing ───────────────────────────────────────────────────────────

def tf_ratio(epoch: int) -> float:
    if epoch >= TF_DECAY_EPOCHS:
        return 0.0
    return TF_START * (1.0 - epoch / TF_DECAY_EPOCHS)


# ── Training ──────────────────────────────────────────────────────────────────

def train(horizon: int, lookback: int, epochs: int, check_only: bool):
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LSTM Forecaster — Real Data Training")
    print(f"{'='*60}")
    print(f"  Horizon  : {horizon} steps × {LOG_INTERVAL_SEC}s "
          f"= {horizon*LOG_INTERVAL_SEC}s "
          f"({horizon*LOG_INTERVAL_SEC//60}m "
          f"{horizon*LOG_INTERVAL_SEC%60}s ahead)")
    print(f"  Lookback : {lookback} steps × {LOG_INTERVAL_SEC}s "
          f"= {lookback*LOG_INTERVAL_SEC}s of history")
    print(f"  Device   : {DEVICE}")

    # ── Fetch data ────────────────────────────────────────────────────────────
    print("\n📡 Fetching real logs from NeonDB...")
    raw = fetch_all_logs()
    print(f"   Rows fetched : {len(raw):,}  "
          f"({len(raw)*LOG_INTERVAL_SEC/60:.1f} min of data)")

    sufficient = check_data_sufficiency(len(raw), lookback, horizon)

    if check_only:
        return

    if not sufficient:
        sys.exit(
            "\n❌ Not enough data to train. "
            "Let the logger run longer and try again.\n"
            "   Or use a shorter horizon with --horizon and --lookback flags."
        )

    # ── Print real data ranges ────────────────────────────────────────────────
    print(f"\n  Real data ranges (model will learn these):")
    for i, name in enumerate(["CPU", "Memory", "Disk"]):
        print(f"    {name:<8}  min={raw[:,i].min():.2f}%  "
              f"avg={raw[:,i].mean():.2f}%  "
              f"max={raw[:,i].max():.2f}%  "
              f"std={raw[:,i].std():.2f}%")

    # ── Fit scaler on real data ───────────────────────────────────────────────
    real_scaler = MinMaxScaler(feature_range=(0, 1))
    normalised  = real_scaler.fit_transform(raw)

    scaler_path = SAVE_DIR / "scaler_real.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(real_scaler, f)
    print(f"\n💾 Scaler saved → {scaler_path}")

    # ── Save horizon config so inference_engine knows what to expect ──────────
    horizon_config_path = SAVE_DIR / "forecaster_config.txt"
    with open(horizon_config_path, "w") as f:
        f.write(f"horizon={horizon}\n")
        f.write(f"lookback={lookback}\n")
        f.write(f"log_interval_sec={LOG_INTERVAL_SEC}\n")
        f.write(f"forecast_seconds={horizon * LOG_INTERVAL_SEC}\n")
        f.write(f"forecast_minutes={horizon * LOG_INTERVAL_SEC / 60:.1f}\n")
    print(f"💾 Horizon config saved → {horizon_config_path}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset    = RealDataForecasterDataset(normalised, lookback, horizon, STRIDE)
    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"\n   Training windows : {train_size:,}")
    print(f"   Val windows      : {val_size:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LSTMForecaster(
        input_size  = 3,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        lookback    = lookback,
        horizon     = horizon,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params     : {total_params:,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"🏋️  Training for {epochs} epochs...")
    print(f"{'─'*58}")
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  {'TF':>6}")
    print(f"{'─'*58}")

    for epoch in range(1, epochs + 1):
        tf = tf_ratio(epoch - 1)

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
            saved = " 💾"

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:5d}  {train_loss:12.6f}  "
                  f"{val_loss:10.6f}  {tf:6.3f}{saved}")

    print(f"{'─'*58}")
    print(f"\n✅ Training complete!  Best val loss: {best_val_loss:.6f}")
    print(f"   Model → {SAVE_DIR}/lstm_forecaster.pth")
    print(f"   Forecasts {horizon*LOG_INTERVAL_SEC}s "
          f"({horizon*LOG_INTERVAL_SEC//60}m "
          f"{horizon*LOG_INTERVAL_SEC%60}s) ahead\n")

    # ── Loss plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train", linewidth=2, color="#4A90D9")
    plt.plot(val_losses,   label="Val",   linewidth=2, color="#E67E22")
    if TF_DECAY_EPOCHS < epochs:
        plt.axvline(TF_DECAY_EPOCHS, color="#888", linestyle=":",
                    linewidth=1.2,
                    label=f"TF ends (epoch {TF_DECAY_EPOCHS})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(
        f"LSTM Forecaster (Real Data) — "
        f"Horizon: {horizon*LOG_INTERVAL_SEC}s ahead"
    )
    plt.legend()
    plt.tight_layout()
    plot_path = SAVE_DIR / "forecaster_realdata_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Loss plot → {plot_path}")
    print(f"\n  Next steps:")
    print(f"    1. Reload the forecaster: curl -X POST http://localhost:8000/model/reload-forecaster")
    print(f"    2. Evaluate: python -m model.evaluate_forecaster_live\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM Forecaster on real NeonDB data"
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"Forecast steps ahead (default: {DEFAULT_HORIZON} = "
             f"{DEFAULT_HORIZON*1}s = "
             f"{DEFAULT_HORIZON//60}m ahead)"
    )
    parser.add_argument(
        "--lookback", type=int, default=DEFAULT_LOOKBACK,
        help=f"History steps to use as input (default: {DEFAULT_LOOKBACK})"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Just check if you have enough data — don't train"
    )
    args = parser.parse_args()

    train(
        horizon    = args.horizon,
        lookback   = args.lookback,
        epochs     = args.epochs,
        check_only = args.check_only,
    )