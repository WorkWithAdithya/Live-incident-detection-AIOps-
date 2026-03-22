"""
inference.py
------------
Real-time incident detection engine.

After training on synthetic data, this module:
  1. Loads the trained model + scaler + threshold from ../saved/
  2. Polls your NeonPostgreSQL system_logs table every LOG_INTERVAL_SECONDS
  3. Fetches the latest `seq_len` rows as a sliding window
  4. Normalises with the same scaler used during training
  5. Computes reconstruction error via the LSTM Autoencoder
  6. Classifies: NORMAL / WARNING / CRITICAL

Usage:
    python -m model.inference           # continuous loop
    python -m model.inference --once    # single pass and exit
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import psycopg2
import torch
from datetime import datetime
from dotenv import load_dotenv

from model.lstm_autoencoder import LSTMAutoencoder

load_dotenv()
DATABASE_URL     = os.getenv("DATABASE_URL")
LOG_INTERVAL_SEC = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

SAVE_DIR       = os.path.join(os.path.dirname(__file__), "..", "saved")
MODEL_PATH     = os.path.join(SAVE_DIR, "lstm_autoencoder.pth")
SCALER_PATH    = os.path.join(SAVE_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(SAVE_DIR, "threshold.txt")

SEQ_LEN      = 60
HIDDEN_SIZE  = 64
LATENT_SIZE  = 32
NUM_LAYERS   = 2
DROPOUT      = 0.2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_model() -> LSTMAutoencoder:
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"❌ Model not found. Run `python -m model.train` first.")
    model = LSTMAutoencoder(
        input_size=3, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT, seq_len=SEQ_LEN
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

def load_threshold() -> float:
    with open(THRESHOLD_PATH) as f:
        return float(f.readline().strip())


# ── DB fetch ──────────────────────────────────────────────────────────────────

def fetch_latest_window(seq_len: int):
    """
    Fetches the most recent `seq_len` rows from your NeonDB system_logs.
    Returns (raw_array, latest_row_dict) or (None, None) if not enough data.
    """
    conn   = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, cpu_usage, memory_usage, disk_usage
        FROM system_logs
        ORDER BY timestamp DESC
        LIMIT %s
        """,
        (seq_len,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if len(rows) < seq_len:
        return None, None

    rows   = list(reversed(rows))       # chronological order
    latest = {
        "timestamp"    : rows[-1][0],
        "cpu_usage"    : rows[-1][1],
        "memory_usage" : rows[-1][2],
        "disk_usage"   : rows[-1][3],
    }
    raw = np.array([[r[1], r[2], r[3]] for r in rows], dtype=np.float32)
    return raw, latest


# ── Rule-based metric flags (complement to ML score) ─────────────────────────

def flag_metrics(raw: np.ndarray) -> list[str]:
    last = raw[-1]
    flags = []
    if last[0] > 85:  flags.append(f"CPU {last[0]:.1f}% > 85%")
    if last[1] > 85:  flags.append(f"Memory {last[1]:.1f}% > 85%")
    if last[2] > 90:  flags.append(f"Disk {last[2]:.1f}% > 90%")
    return flags


# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(model, scaler, threshold: float) -> dict:
    raw, latest = fetch_latest_window(SEQ_LEN)

    if raw is None:
        return {"status": "insufficient_data"}

    normalised = scaler.transform(raw)
    x          = torch.tensor(normalised).unsqueeze(0).to(DEVICE)   # (1, 60, 3)
    error      = model.reconstruction_error(x).item()

    if error < threshold:
        severity, is_anomaly = "NORMAL",   False
    elif error < threshold * 1.5:
        severity, is_anomaly = "WARNING",  True
    else:
        severity, is_anomaly = "CRITICAL", True

    return {
        "status"         : "ok",
        "timestamp"      : latest["timestamp"],
        "cpu"            : latest["cpu_usage"],
        "memory"         : latest["memory_usage"],
        "disk"           : latest["disk_usage"],
        "error"          : round(error, 6),
        "threshold"      : round(threshold, 6),
        "is_anomaly"     : is_anomaly,
        "severity"       : severity,
        "flagged_metrics": flag_metrics(raw) if is_anomaly else [],
    }


def print_result(result: dict):
    if result["status"] == "insufficient_data":
        print(f"[{datetime.now():%H:%M:%S}]  ⏳  Waiting for {SEQ_LEN} rows in DB...")
        return

    icons = {"NORMAL": "✅", "WARNING": "⚠️ ", "CRITICAL": "🔴"}
    icon  = icons[result["severity"]]
    print(
        f"[{result['timestamp']}]  {icon}  {result['severity']:<8} | "
        f"CPU:{result['cpu']:5.1f}%  MEM:{result['memory']:5.1f}%  "
        f"DISK:{result['disk']:5.1f}%  | "
        f"Err:{result['error']:.6f}  Thresh:{result['threshold']:.6f}"
    )
    if result["flagged_metrics"]:
        print(f"           ↳ 🚨 {', '.join(result['flagged_metrics'])}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Single inference pass then exit")
    args = parser.parse_args()

    print("🔍 Loading model artefacts...")
    model     = load_model()
    scaler    = load_scaler()
    threshold = load_threshold()
    print(f"   Threshold : {threshold:.6f}")
    print(f"   Window    : {SEQ_LEN} steps × {LOG_INTERVAL_SEC}s = "
          f"{SEQ_LEN * LOG_INTERVAL_SEC}s\n")

    if args.once:
        result = run_inference(model, scaler, threshold)
        print_result(result)
        return result

    print(f"🚨 Real-time detection active (every {LOG_INTERVAL_SEC}s)...\n")
    while True:
        try:
            print_result(run_inference(model, scaler, threshold))
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}]  ❌  {e}")
        time.sleep(LOG_INTERVAL_SEC)


if __name__ == "__main__":
    main()