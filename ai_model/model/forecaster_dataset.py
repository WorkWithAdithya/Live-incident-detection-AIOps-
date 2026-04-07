"""
forecaster_dataset.py
---------------------
Builds supervised (X, y) pairs for LSTM Forecaster training.

Each sample:
    X : (lookback, 3)  — input window of past readings  (normalised)
    y : (horizon, 3)   — target window of future readings (normalised)

Uses the EXISTING scaler.pkl — never refits it.
This is critical: both models must share the same normalisation.

Example at default settings (lookback=60, horizon=12, stride=1):
    Row 0:   X = rows[0:60],   y = rows[60:72]
    Row 1:   X = rows[1:61],   y = rows[61:73]
    ...
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ["cpu_usage", "memory_usage", "disk_usage"]

# Default paths — relative to ai_model/ root
_DIR      = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.join(_DIR, "..")
DATA_PATH = os.path.join(_ROOT, "data", "synthetic_logs.csv")
SCALER_PATH = os.path.join(_ROOT, "saved", "scaler.pkl")


def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Loads the existing fitted MinMaxScaler.
    Must use the same scaler as the LSTM Autoencoder.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"❌ scaler.pkl not found at {scaler_path}\n"
            f"   Run python -m model.train first to generate it."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"✅ Loaded existing scaler from {os.path.basename(scaler_path)}")
    return scaler


class ForecasterDataset(Dataset):
    """
    Supervised dataset for sequence-to-sequence forecasting.

    Args:
        lookback    : Input window length  (default 60 — same as autoencoder)
        horizon     : Forecast horizon     (default 12 — 1 min at 5s interval)
        stride      : Step between samples (default 1 — full overlap)
        data_path   : Path to synthetic_logs.csv
        scaler_path : Path to existing scaler.pkl

    Attributes:
        X           : np.ndarray (N, lookback, 3)  — input windows
        y           : np.ndarray (N, horizon, 3)   — target windows
        scaler      : The loaded MinMaxScaler (shared with autoencoder)
    """

    def __init__(
        self,
        lookback    : int = 60,
        horizon     : int = 12,
        stride      : int = 1,
        data_path   : str = DATA_PATH,
        scaler_path : str = SCALER_PATH,
    ):
        self.lookback = lookback
        self.horizon  = horizon
        self.stride   = stride

        # ── 1. Load raw data ──────────────────────────────────────────────────
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"❌ synthetic_logs.csv not found at {data_path}\n"
                f"   Run: python data/generate_synthetic_data.py"
            )
        df  = pd.read_csv(data_path, parse_dates=["timestamp"])
        raw = df[FEATURES].values.astype(np.float32)
        print(f"📦 Loaded {len(raw):,} rows from {os.path.basename(data_path)}")

        # ── 2. Normalise with EXISTING scaler (transform only, never fit) ─────
        self.scaler  = load_scaler(scaler_path)
        normalised   = self.scaler.transform(raw)   # (N, 3) in [0, 1]

        # ── 3. Build (X, y) pairs ─────────────────────────────────────────────
        self.X, self.y = self._build_pairs(normalised)
        print(
            f"🪟  Forecaster windows: {len(self.X):,}  "
            f"(lookback={lookback}, horizon={horizon}, stride={stride})"
        )

    def _build_pairs(self, data: np.ndarray):
        """
        Slides a window of (lookback + horizon) across the data.
        First `lookback` steps → X (input)
        Next  `horizon`  steps → y (target to predict)
        """
        total   = self.lookback + self.horizon
        n       = len(data)
        X_list, y_list = [], []

        for start in range(0, n - total + 1, self.stride):
            X_list.append(data[start          : start + self.lookback])
            y_list.append(data[start + self.lookback : start + total])

        return (
            np.array(X_list, dtype=np.float32),   # (W, lookback, 3)
            np.array(y_list, dtype=np.float32),   # (W, horizon,  3)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),   # (lookback, 3)
            torch.tensor(self.y[idx]),   # (horizon,  3)
        )

    def get_stats(self):
        """Returns summary statistics for sanity checking."""
        return {
            "n_samples"   : len(self.X),
            "lookback"    : self.lookback,
            "horizon"     : self.horizon,
            "X_shape"     : self.X.shape,
            "y_shape"     : self.y.shape,
            "X_min"       : float(self.X.min()),
            "X_max"       : float(self.X.max()),
            "y_min"       : float(self.y.min()),
            "y_max"       : float(self.y.max()),
        }


if __name__ == "__main__":
    ds = ForecasterDataset(lookback=60, horizon=12, stride=5)
    x, y = ds[0]
    print(f"\nSample X shape : {x.shape}  (input window)")
    print(f"Sample y shape : {y.shape}  (target forecast)")
    print(f"X[0]  : {x[0].numpy()}  (first step: cpu, mem, disk normalised)")
    print(f"y[0]  : {y[0].numpy()}  (first predicted step)")
    import pprint
    pprint.pprint(ds.get_stats())