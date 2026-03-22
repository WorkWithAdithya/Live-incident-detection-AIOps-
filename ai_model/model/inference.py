"""
inference.py
------------
Real-time incident detection engine for AIOps LSTM Autoencoder.

After training on synthetic data, this module:
  1. Loads the trained model + scaler + threshold from ../saved/
  2. Resolves the .env from log_generator/ (one level up from ai_model/)
  3. Polls your NeonPostgreSQL system_logs table every LOG_INTERVAL_SECONDS
  4. Fetches the latest `seq_len` rows as a sliding window
  5. Normalises with the same scaler used during training
  6. Computes reconstruction error via the LSTM Autoencoder
  7. Classifies: NORMAL / WARNING / CRITICAL and prints result

Project structure assumed:
    your_project/
    ├── log_generator/
    │   ├── .env                   ← DATABASE_URL lives here
    │   └── src/
    └── ai_model/
        ├── model/
        │   └── inference.py       ← this file
        └── saved/
            ├── lstm_autoencoder.pth
            ├── scaler.pkl
            └── threshold.txt

Usage:
    python -m model.inference           # runs continuously
    python -m model.inference --once    # single pass then exit
    python -m model.inference --debug   # single pass with extra detail
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
from pathlib import Path
from dotenv import load_dotenv

from model.lstm_autoencoder import LSTMAutoencoder

# ── Resolve .env path ─────────────────────────────────────────────────────────
# inference.py  →  ai_model/model/inference.py
# parent        →  ai_model/model/
# parent.parent →  ai_model/
# parent.parent.parent → your_project/
# / log_generator / .env

_ENV_PATH = (
    Path(__file__).resolve()
    .parent       # ai_model/model/
    .parent       # ai_model/
    .parent       # your_project/
    / "log_generator"
    / ".env"
)

load_dotenv(dotenv_path=_ENV_PATH)

# ── Validate env loaded correctly ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError(
        f"\n❌ DATABASE_URL not found in environment.\n"
        f"   Looked for .env at : {_ENV_PATH}\n"
        f"   File exists        : {_ENV_PATH.exists()}\n\n"
        f"   Fix: ensure log_generator/.env contains DATABASE_URL=...\n"
        f"   Or adjust the path in inference.py if your folder structure differs."
    )

LOG_INTERVAL_SEC = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

# ── Saved artefact paths ──────────────────────────────────────────────────────
_BASE_DIR      = Path(__file__).resolve().parent.parent   # ai_model/
SAVE_DIR       = _BASE_DIR / "saved"
MODEL_PATH     = SAVE_DIR / "lstm_autoencoder.pth"
SCALER_PATH    = SAVE_DIR / "scaler.pkl"
THRESHOLD_PATH = SAVE_DIR / "threshold.txt"

# ── Model hyperparameters (must match train.py) ───────────────────────────────
SEQ_LEN      = 60
HIDDEN_SIZE  = 64
LATENT_SIZE  = 32
NUM_LAYERS   = 2
DROPOUT      = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_model() -> LSTMAutoencoder:
    """Loads trained LSTM Autoencoder weights from saved/."""
    if not MODEL_PATH.exists():
        sys.exit(
            f"❌ Model weights not found at:\n   {MODEL_PATH}\n"
            f"   Run `python -m model.train` first."
        )
    model = LSTMAutoencoder(
        input_size  = 3,
        hidden_size = HIDDEN_SIZE,
        latent_size = LATENT_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        seq_len     = SEQ_LEN,
    )
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    return model


def load_scaler():
    """Loads the MinMaxScaler fitted during training."""
    if not SCALER_PATH.exists():
        sys.exit(
            f"❌ Scaler not found at:\n   {SCALER_PATH}\n"
            f"   Run `python -m model.train` first."
        )
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


def load_threshold() -> tuple[float, float, float]:
    """
    Loads anomaly threshold from saved/threshold.txt.
    Returns (threshold, mean_error, std_error).
    """
    if not THRESHOLD_PATH.exists():
        sys.exit(
            f"❌ Threshold not found at:\n   {THRESHOLD_PATH}\n"
            f"   Run `python -m model.train` first."
        )
    with open(THRESHOLD_PATH) as f:
        lines = f.read().strip().splitlines()

    threshold = float(lines[0])
    stats     = {}
    for line in lines[1:]:
        if "=" in line:
            k, v = line.split("=")
            stats[k.strip()] = float(v.strip())

    return threshold, stats.get("mean", 0.0), stats.get("std", 0.0)


# ── Database ──────────────────────────────────────────────────────────────────

def fetch_latest_window(seq_len: int) -> tuple:
    """
    Fetches the most recent rows from system_logs — works from row 1.

    Strategy:
      - If DB has >= seq_len rows  -> use the last seq_len rows (full window)
      - If DB has <  seq_len rows  -> use all available rows, front-pad by
                                      repeating the first row so the tensor
                                      is always (seq_len, 3)

    This means inference starts on the VERY FIRST log entry — no waiting.

    Returns:
        raw         : np.ndarray (seq_len, 3) — padded if needed
        latest      : dict  — most recent row values for display
        actual_rows : int   — real rows fetched (before padding)
    """
    try:
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

    except psycopg2.OperationalError as e:
        raise ConnectionError(f"❌ Cannot connect to NeonDB: {e}") from e

    # Nothing in DB at all yet
    if len(rows) == 0:
        return None, None, 0

    rows        = list(reversed(rows))   # DESC -> ASC (chronological)
    actual_rows = len(rows)

    latest = {
        "timestamp"    : rows[-1][0],
        "cpu_usage"    : float(rows[-1][1]),
        "memory_usage" : float(rows[-1][2]),
        "disk_usage"   : float(rows[-1][3]),
    }

    raw = np.array(
        [[float(r[1]), float(r[2]), float(r[3])] for r in rows],
        dtype=np.float32
    )   # (actual_rows, 3)

    # Pad to seq_len if fewer rows available
    # Front-pad by repeating the first row — neutral padding that the
    # model reconstructs easily and does not inflate the anomaly score
    if actual_rows < seq_len:
        pad_count = seq_len - actual_rows
        pad       = np.tile(raw[0], (pad_count, 1))   # (pad_count, 3)
        raw       = np.vstack([pad, raw])              # (seq_len, 3)

    return raw, latest, actual_rows


# ── Rule-based metric flags ───────────────────────────────────────────────────

def flag_metrics(raw: np.ndarray) -> list[str]:
    """
    Checks the most recent reading against hard thresholds.
    These are complementary to the ML anomaly score — they give
    a human-readable reason for why an alert fired.
    """
    last   = raw[-1]   # [cpu, memory, disk]
    flags  = []
    if last[0] > 85:
        flags.append(f"CPU {last[0]:.1f}% (> 85%)")
    if last[1] > 85:
        flags.append(f"Memory {last[1]:.1f}% (> 85%)")
    if last[2] > 90:
        flags.append(f"Disk {last[2]:.1f}% (> 90%)")
    return flags


def trend_summary(raw: np.ndarray) -> str:
    """
    Quick trend description over the last 12 steps (~1 min).
    Used in --debug mode.
    """
    window  = raw[-12:] if len(raw) >= 12 else raw
    deltas  = window[-1] - window[0]
    parts   = []
    labels  = ["CPU", "MEM", "DISK"]
    for i, (label, delta) in enumerate(zip(labels, deltas)):
        if delta > 5:
            parts.append(f"{label} ↑{delta:+.1f}%")
        elif delta < -5:
            parts.append(f"{label} ↓{delta:+.1f}%")
        else:
            parts.append(f"{label} ~stable")
    return "  |  ".join(parts)


# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(model, scaler, threshold: float) -> dict:
    """
    Runs one complete inference pass:
      fetch → normalise → forward pass → classify → return result dict.

    Returns:
        {
            status          : "ok" | "insufficient_data" | "db_error"
            timestamp       : datetime
            cpu             : float
            memory          : float
            disk            : float
            error           : float   (reconstruction MSE)
            threshold       : float
            error_ratio     : float   (error / threshold — useful for dashboards)
            is_anomaly      : bool
            severity        : "NORMAL" | "WARNING" | "CRITICAL"
            flagged_metrics : list[str]
            raw_window      : np.ndarray  (seq_len, 3)
        }
    """
    try:
        raw, latest, actual_rows = fetch_latest_window(SEQ_LEN)
    except ConnectionError as e:
        return {"status": "db_error", "message": str(e)}

    if raw is None:
        return {"status": "no_data"}

    # Warmup mode: fewer than SEQ_LEN real rows — model still runs but
    # accuracy improves as the window fills up over time
    warming_up = actual_rows < SEQ_LEN

    # Normalise using training scaler
    normalised = scaler.transform(raw)                                   # (seq_len, 3)
    x          = torch.tensor(normalised,
                               dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, seq_len, 3)

    # Reconstruction error = anomaly score
    error       = model.reconstruction_error(x).item()
    error_ratio = error / threshold

    # ── Severity classification ──
    # Normal   : error < threshold
    # Warning  : threshold ≤ error < 1.5× threshold  (early warning)
    # Critical : error ≥ 1.5× threshold              (definite incident)
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
        "error"          : round(error, 8),
        "threshold"      : round(threshold, 8),
        "error_ratio"    : round(error_ratio, 4),
        "is_anomaly"     : is_anomaly,
        "severity"       : severity,
        "flagged_metrics": flag_metrics(raw) if is_anomaly else [],
        "raw_window"     : raw,
        "actual_rows"    : actual_rows,
        "warming_up"     : warming_up,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

SEVERITY_ICONS = {
    "NORMAL"  : "✅",
    "WARNING" : "⚠️ ",
    "CRITICAL": "🔴",
}

SEVERITY_COLORS = {
    "NORMAL"  : "\033[92m",   # green
    "WARNING" : "\033[93m",   # yellow
    "CRITICAL": "\033[91m",   # red
}
RESET = "\033[0m"


def print_result(result: dict, debug: bool = False):
    """Formatted console output for one inference result."""

    if result["status"] == "no_data":
        print(
            f"[{datetime.now():%H:%M:%S}]  ⏳  Waiting for first log entry "
            f"in system_logs..."
        )
        return

    if result["status"] == "db_error":
        print(f"[{datetime.now():%H:%M:%S}]  {result['message']}")
        return

    sev   = result["severity"]
    icon  = SEVERITY_ICONS[sev]
    color = SEVERITY_COLORS[sev]
    ts    = result["timestamp"]

    # ── Warmup indicator ──
    warmup_tag = ""
    if result.get("warming_up"):
        pct        = int((result["actual_rows"] / SEQ_LEN) * 20)
        bar        = "█" * pct + "░" * (20 - pct)
        warmup_tag = (
            f"  [WARMUP {result['actual_rows']:>2}/{SEQ_LEN} rows  {bar}]"
        )

    # ── Main line ──
    print(
        f"[{ts}]  {icon}  "
        f"{color}{sev:<8}{RESET}  |  "
        f"CPU: {result['cpu']:5.1f}%  "
        f"MEM: {result['memory']:5.1f}%  "
        f"DISK: {result['disk']:5.1f}%  |  "
        f"Err: {result['error']:.6f}  "
        f"({result['error_ratio']:.2f}x thresh)"
        f"{warmup_tag}"
    )

    # ── Flagged metrics (anomaly only) ──
    if result["flagged_metrics"]:
        flags = ", ".join(result["flagged_metrics"])
        print(f"           ↳ 🚨 Flagged: {flags}")

    # ── Debug mode: trend + raw values ──
    if debug and result.get("raw_window") is not None:
        trend = trend_summary(result["raw_window"])
        print(f"           ↳ 📈 Trend (last 1 min): {trend}")
        last  = result["raw_window"][-1]
        print(
            f"           ↳ 🔢 Raw:  CPU={last[0]:.2f}%  "
            f"MEM={last[1]:.2f}%  DISK={last[2]:.2f}%"
        )


def print_startup_banner(threshold: float, mean_err: float, std_err: float):
    """Prints a startup summary to confirm everything loaded correctly."""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║      AIOps LSTM Autoencoder — Inference Engine      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Device           : {DEVICE}")
    print(f"  .env loaded from : {_ENV_PATH}")
    print(f"  DB connected     : {'✅' if DATABASE_URL else '❌'}")
    print(f"  Model            : {MODEL_PATH.name}")
    print(f"  Sequence window  : {SEQ_LEN} steps × {LOG_INTERVAL_SEC}s "
          f"= {SEQ_LEN * LOG_INTERVAL_SEC}s (pads from row 1 — no waiting)")
    print(f"  Threshold        : {threshold:.8f}  "
          f"(mean={mean_err:.6f}  std={std_err:.6f})")
    print(f"  Severity bands   :")
    print(f"    NORMAL   → error < {threshold:.6f}")
    print(f"    WARNING  → {threshold:.6f} ≤ error < {threshold*1.5:.6f}")
    print(f"    CRITICAL → error ≥ {threshold*1.5:.6f}")
    print(f"  Poll interval    : every {LOG_INTERVAL_SEC}s")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AIOps LSTM Autoencoder — Real-time Inference Engine"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single inference pass and exit"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show trend and raw values on each inference pass"
    )
    args = parser.parse_args()

    # ── Load artefacts ────────────────────────────────────────────────────
    print("🔍 Loading model artefacts...")
    model                     = load_model()
    scaler                    = load_scaler()
    threshold, mean_err, std_err = load_threshold()

    print_startup_banner(threshold, mean_err, std_err)

    # ── Single pass ───────────────────────────────────────────────────────
    if args.once or args.debug:
        result = run_inference(model, scaler, threshold)
        print_result(result, debug=True)
        return result

    # ── Continuous loop ───────────────────────────────────────────────────
    print(f"🚨 Real-time detection active  (Ctrl+C to stop)\n")
    consecutive_anomalies = 0

    while True:
        try:
            result = run_inference(model, scaler, threshold)
            print_result(result, debug=args.debug)

            # Track consecutive anomalies — useful for escalation logic
            if result.get("is_anomaly"):
                consecutive_anomalies += 1
                if consecutive_anomalies >= 3:
                    print(
                        f"           ↳ ⚡ ALERT: {consecutive_anomalies} consecutive "
                        f"anomalous windows detected!"
                    )
            else:
                if consecutive_anomalies > 0:
                    print(
                        f"           ↳ ✅ Resolved after "
                        f"{consecutive_anomalies} anomalous windows."
                    )
                consecutive_anomalies = 0

        except KeyboardInterrupt:
            print("\n\n🛑 Inference stopped by user.")
            break
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}]  ❌  Unexpected error: {e}")

        time.sleep(LOG_INTERVAL_SEC)


if __name__ == "__main__":
    main()