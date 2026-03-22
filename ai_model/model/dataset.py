"""
dataset.py
----------
Loads synthetic_logs.csv and builds overlapping sliding-window
sequences for LSTM Autoencoder training.

Each sample  →  tensor of shape (seq_len, 3):
    [cpu_usage, memory_usage, disk_usage]  — normalised to [0, 1]

The 'label' and 'incident_type' columns are kept separately for
evaluation purposes but are NEVER fed into the model during training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logs.csv")
FEATURES  = ["cpu_usage", "memory_usage", "disk_usage"]


def load_csv(path: str = DATA_PATH) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Reads the synthetic CSV.

    Returns:
        raw     : np.ndarray (N, 3)  — raw float values
        labels  : np.ndarray (N,)    — 0=normal, 1=anomaly
        types   : list[str]          — incident type per row
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Synthetic data not found at:\n   {path}\n"
            f"   Run:  python data/generate_synthetic_data.py"
        )
    df      = pd.read_csv(path, parse_dates=["timestamp"])
    raw     = df[FEATURES].values.astype(np.float32)
    labels  = df["label"].values.astype(int)
    types   = df["incident_type"].tolist()
    print(f"📦 Loaded {len(df):,} rows from {os.path.basename(path)}")
    return raw, labels, types


class SystemLogsDataset(Dataset):
    """
    Sliding-window dataset over normalised synthetic metric sequences.

    Args:
        seq_len     : Window length  (default 60 = 5-min @ 5s interval)
        stride      : Step between windows (1 = max overlap for training)
        scaler      : Pass a fitted MinMaxScaler for val/test; None = fit here
        data_path   : Override default CSV path

    Attributes:
        scaler          : Fitted MinMaxScaler  — save with the model
        sequences       : np.ndarray (W, seq_len, 3)
        window_labels   : np.ndarray (W,)  — 1 if ANY row in window is anomaly
        window_types    : list[str]         — dominant incident type per window
    """

    def __init__(
        self,
        seq_len   : int  = 60,
        stride    : int  = 1,
        scaler           = None,
        data_path : str  = DATA_PATH,
    ):
        self.seq_len = seq_len
        self.stride  = stride

        # ── 1. Load raw data ───────────────────────────────────────────────
        raw, labels, types = load_csv(data_path)

        # ── 2. Normalise ───────────────────────────────────────────────────
        if scaler is None:
            self.scaler  = MinMaxScaler(feature_range=(0, 1))
            normalised   = self.scaler.fit_transform(raw)
        else:
            self.scaler  = scaler
            normalised   = self.scaler.transform(raw)

        # ── 3. Build sliding windows ───────────────────────────────────────
        self.sequences, self.window_labels, self.window_types = \
            self._build_windows(normalised, labels, types)

        n_anomaly = int(self.window_labels.sum())
        print(f"🪟  Windows: {len(self.sequences):,}  "
              f"(normal={len(self.sequences)-n_anomaly:,}, "
              f"anomaly={n_anomaly:,})  "
              f"seq_len={seq_len}, stride={stride}")

    # ── Window builder ─────────────────────────────────────────────────────
    def _build_windows(
        self,
        data   : np.ndarray,
        labels : np.ndarray,
        types  : list,
    ):
        seqs, win_labels, win_types = [], [], []
        n = len(data)
        for start in range(0, n - self.seq_len + 1, self.stride):
            end = start + self.seq_len
            seqs.append(data[start:end])

            # Window label: 1 if any step in window is anomalous
            win_labels.append(int(labels[start:end].any()))

            # Dominant incident type (most frequent non-normal)
            window_type_list = types[start:end]
            non_normal = [t for t in window_type_list if t != "normal"]
            if non_normal:
                win_types.append(max(set(non_normal), key=non_normal.count))
            else:
                win_types.append("normal")

        return (
            np.array(seqs, dtype=np.float32),
            np.array(win_labels, dtype=int),
            win_types,
        )

    # ── PyTorch interface ──────────────────────────────────────────────────
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Returns tensor (seq_len, 3) — label NOT returned; autoencoder is unsupervised
        return torch.tensor(self.sequences[idx])

    def get_normal_subset(self):
        """Returns a Dataset containing only normal windows (for training)."""
        normal_idx = np.where(self.window_labels == 0)[0]
        subset = _IndexedSubset(self, normal_idx)
        print(f"   Normal-only subset: {len(subset):,} windows")
        return subset


class _IndexedSubset(Dataset):
    """Lightweight wrapper to index into a parent Dataset."""
    def __init__(self, parent: SystemLogsDataset, indices: np.ndarray):
        self.parent  = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]


if __name__ == "__main__":
    ds = SystemLogsDataset(seq_len=60, stride=5)
    print(f"\nSample shape : {ds[0].shape}")
    print(f"Sample[0][0] : {ds[0][0].numpy()}  (cpu, mem, disk — normalised)")