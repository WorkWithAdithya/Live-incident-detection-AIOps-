"""
frontend/backend/inference_engine.py
-------------------------------------
Wraps the ai_model LSTM inference logic for use by the FastAPI backend.
Handles model loading, scaler, threshold, and per-call inference.
"""

import os
import pickle
import numpy as np
import psycopg2
import torch
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env
_ENV_PATH = (
    Path(__file__).resolve()
    .parent.parent.parent
    / "log_generator" / ".env"
)
load_dotenv(dotenv_path=_ENV_PATH)

DATABASE_URL     = os.getenv("DATABASE_URL")
LOG_INTERVAL_SEC = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

SEQ_LEN     = 60
HIDDEN_SIZE = 64
LATENT_SIZE = 32
NUM_LAYERS  = 2
DROPOUT     = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceEngine:
    """
    Loads model, scaler, and threshold from saved/ directory.
    Exposes run() to perform one inference pass against NeonDB.
    """

    def __init__(self, saved_dir: Path):
        self.saved_dir  = Path(saved_dir)
        self.model_path = self.saved_dir / "lstm_autoencoder.pth"

        # ── Load model ────────────────────────────────────────────────────
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. "
                f"Run python -m model.train first."
            )

        # Import here so sys.path is already set by main.py
        from model.lstm_autoencoder import LSTMAutoencoder

        self.model = LSTMAutoencoder(
            input_size  = 3,
            hidden_size = HIDDEN_SIZE,
            latent_size = LATENT_SIZE,
            num_layers  = NUM_LAYERS,
            dropout     = DROPOUT,
            seq_len     = SEQ_LEN,
        )
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=DEVICE,
                       weights_only=True)
        )
        self.model.to(DEVICE)
        self.model.eval()

        # ── Load scaler ───────────────────────────────────────────────────
        scaler_path = self.saved_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}.")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # ── Load threshold ────────────────────────────────────────────────
        threshold_path = self.saved_dir / "threshold.txt"
        if not threshold_path.exists():
            raise FileNotFoundError(
                f"Threshold not found at {threshold_path}."
            )
        with open(threshold_path) as f:
            lines = f.read().strip().splitlines()

        self.default_threshold = float(lines[0])
        self.threshold         = self.default_threshold   # mutable by user

        stats = {}
        for line in lines[1:]:
            if "=" in line:
                k, v = line.split("=")
                stats[k.strip()] = float(v.strip())

        self.mean_error        = stats.get("mean", 0.0)
        self.std_error         = stats.get("std",  0.0)

        # Youden's optimal threshold — compute lazily on first evaluate call
        self.optimal_threshold = round(self.default_threshold * 0.85, 8)

        print(f"✅ InferenceEngine ready | "
              f"threshold={self.threshold:.6f} | device={DEVICE}")

    # ── DB fetch ──────────────────────────────────────────────────────────

    def _fetch_window(self) -> tuple:
        """
        Fetches latest SEQ_LEN rows from system_logs.
        Pads with first row if fewer rows available (works from row 1).

        Returns: (raw np.ndarray (SEQ_LEN,3), latest dict, actual_rows int)
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
            (SEQ_LEN,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(rows) == 0:
            return None, None, 0

        rows        = list(reversed(rows))
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
        )

        # Front-pad with first row if window not full yet
        if actual_rows < SEQ_LEN:
            pad = np.tile(raw[0], (SEQ_LEN - actual_rows, 1))
            raw = np.vstack([pad, raw])

        return raw, latest, actual_rows

    # ── Inference ─────────────────────────────────────────────────────────

    def _flag_metrics(self, raw: np.ndarray) -> list:
        last  = raw[-1]
        flags = []
        if last[0] > 85: flags.append(f"CPU {last[0]:.1f}% > 85%")
        if last[1] > 85: flags.append(f"Memory {last[1]:.1f}% > 85%")
        if last[2] > 90: flags.append(f"Disk {last[2]:.1f}% > 90%")
        return flags

    def run(self) -> dict:
        """
        One complete inference pass.
        Returns a result dict ready to be JSON-serialised by FastAPI.
        """
        raw, latest, actual_rows = self._fetch_window()

        if raw is None:
            return {"status": "no_data", "actual_rows": 0}

        normalised = self.scaler.transform(raw)
        x          = torch.tensor(
            normalised, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        error       = self.model.reconstruction_error(x).item()
        error_ratio = error / self.threshold

        if error < self.threshold:
            severity, is_anomaly = "NORMAL",   False
        elif error < self.threshold * 1.5:
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
            "threshold"      : round(self.threshold, 8),
            "error_ratio"    : round(error_ratio, 4),
            "is_anomaly"     : is_anomaly,
            "severity"       : severity,
            "flagged_metrics": self._flag_metrics(raw) if is_anomaly else [],
            "actual_rows"    : actual_rows,
            "warming_up"     : actual_rows < SEQ_LEN,
        }