"""
frontend/backend/inference_engine.py
--------------------------------------
Runs BOTH LSTM models on every inference cycle:

  1. LSTM Autoencoder  → reconstruction error → anomaly DETECTION (now)
  2. LSTM Forecaster   → 12 predicted readings → anomaly PREDICTION (future)

The forecaster output is returned as a list of 12 dicts:
    [
      { "step": 1, "seconds_ahead": 5,  "cpu": 34.2, "memory": 52.1, "disk": 44.8 },
      { "step": 2, "seconds_ahead": 10, "cpu": 35.1, "memory": 53.4, "disk": 44.9 },
      ...
      { "step": 12,"seconds_ahead": 60, "cpu": 41.3, "memory": 61.2, "disk": 45.1 },
    ]

The FastAPI SSE stream attaches this to every event so the frontend
PredictionPanel can show real LSTM-predicted future values.

Scaler: single shared scaler.pkl used for BOTH models.
"""

import os
import sys
import pickle
import numpy as np
import psycopg2
import torch
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ── Paths & env ───────────────────────────────────────────────────────────────
_ENV_PATH = (
    Path(__file__).resolve()
    .parent.parent.parent
    / "log_generator" / ".env"
)
load_dotenv(dotenv_path=_ENV_PATH)

DATABASE_URL     = os.getenv("DATABASE_URL")
LOG_INTERVAL_SEC = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

# ── Model constants ────────────────────────────────────────────────────────────
SEQ_LEN          = 60     # lookback window — shared by both models
HORIZON          = 12     # forecaster output steps
FEATURES         = ["cpu", "memory", "disk"]

# Autoencoder arch
AE_HIDDEN_SIZE   = 64
AE_LATENT_SIZE   = 32
AE_NUM_LAYERS    = 2
AE_DROPOUT       = 0.2

# Forecaster arch
FC_HIDDEN_SIZE   = 128
FC_NUM_LAYERS    = 2
FC_DROPOUT       = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceEngine:
    """
    Loads both LSTM models and the shared scaler.
    Exposes run() which returns detection + forecast results in one dict.
    """

    def __init__(self, saved_dir: Path):
        self.saved_dir = Path(saved_dir)

        # ── Load shared scaler ────────────────────────────────────────────────
        scaler_path = self.saved_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"scaler.pkl not found at {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        print(f"   scaler.pkl loaded")

        # ── Load LSTM Autoencoder (detection) ─────────────────────────────────
        from model.lstm_autoencoder import LSTMAutoencoder

        ae_path = self.saved_dir / "lstm_autoencoder.pth"
        if not ae_path.exists():
            raise FileNotFoundError(
                f"lstm_autoencoder.pth not found at {ae_path}.\n"
                f"Run: python -m model.train"
            )
        self.autoencoder = LSTMAutoencoder(
            input_size  = 3,
            hidden_size = AE_HIDDEN_SIZE,
            latent_size = AE_LATENT_SIZE,
            num_layers  = AE_NUM_LAYERS,
            dropout     = AE_DROPOUT,
            seq_len     = SEQ_LEN,
        )
        self.autoencoder.load_state_dict(
            torch.load(str(ae_path), map_location=DEVICE, weights_only=True)
        )
        self.autoencoder.to(DEVICE)
        self.autoencoder.eval()
        print(f"   lstm_autoencoder.pth loaded")

        # ── Load LSTM Forecaster (prediction) — optional ──────────────────────
        self.forecaster         = None
        self.forecaster_ready   = False

        fc_path = self.saved_dir / "lstm_forecaster.pth"
        print(f"   Looking for forecaster at : {fc_path}")
        print(f"   forecaster file exists    : {fc_path.exists()}")

        if fc_path.exists():
            try:
                # Ensure ai_model/ is on sys.path so model.lstm_forecaster resolves
                import sys as _sys
                _ai_model_dir = str(self.saved_dir.parent)
                if _ai_model_dir not in _sys.path:
                    _sys.path.insert(0, _ai_model_dir)
                    print(f"   Added to sys.path: {_ai_model_dir}")

                from model.lstm_forecaster import LSTMForecaster
                self.forecaster = LSTMForecaster(
                    input_size  = 3,
                    hidden_size = FC_HIDDEN_SIZE,
                    num_layers  = FC_NUM_LAYERS,
                    dropout     = FC_DROPOUT,
                    lookback    = SEQ_LEN,
                    horizon     = HORIZON,
                )
                self.forecaster.load_state_dict(
                    torch.load(str(fc_path), map_location=DEVICE, weights_only=True)
                )
                self.forecaster.to(DEVICE)
                self.forecaster.eval()
                self.forecaster_ready = True
                print(f"   lstm_forecaster.pth loaded  ✅ Prediction enabled")
            except Exception as e:
                import traceback
                print(f"   ❌ lstm_forecaster.pth found but failed to load:")
                print(f"      Error: {e}")
                traceback.print_exc()
                print(f"   Prediction disabled — fix the error above and reload.")
        else:
            print(
                f"   ⚠️  lstm_forecaster.pth not found at {fc_path}\n"
                f"   Run: cd ai_model && python -m model.train_forecaster"
            )

        # ── Load threshold ────────────────────────────────────────────────────
        threshold_path = self.saved_dir / "threshold.txt"
        if not threshold_path.exists():
            raise FileNotFoundError(f"threshold.txt not found at {threshold_path}")
        with open(threshold_path) as f:
            lines = f.read().strip().splitlines()

        self.default_threshold = float(lines[0])
        self.threshold         = self.default_threshold

        stats = {}
        for line in lines[1:]:
            if "=" in line:
                k, v = line.split("=")
                stats[k.strip()] = float(v.strip())
        self.mean_error        = stats.get("mean", 0.0)
        self.std_error         = stats.get("std",  0.0)
        self.optimal_threshold = round(self.default_threshold * 0.85, 8)

        print(
            f"\n✅ InferenceEngine ready | "
            f"threshold={self.threshold:.6f} | device={DEVICE} | "
            f"forecaster={'ON' if self.forecaster_ready else 'OFF'}"
        )

    # ── DB fetch ──────────────────────────────────────────────────────────────

    def _fetch_window(self) -> tuple:
        """
        Fetches latest SEQ_LEN rows from system_logs, front-padded if needed.
        Returns (raw, latest_row_dict, actual_rows_count).
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

        if actual_rows < SEQ_LEN:
            pad = np.tile(raw[0], (SEQ_LEN - actual_rows, 1))
            raw = np.vstack([pad, raw])

        return raw, latest, actual_rows

    # ── Autoencoder detection ─────────────────────────────────────────────────

    def _run_autoencoder(self, normalised: np.ndarray) -> dict:
        """
        Runs LSTM Autoencoder on the normalised window.
        Returns detection result dict.
        """
        x     = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        error = self.autoencoder.reconstruction_error(x).item()
        ratio = error / self.threshold

        if error < self.threshold:
            severity, is_anomaly = "NORMAL",   False
        elif error < self.threshold * 1.5:
            severity, is_anomaly = "WARNING",  True
        else:
            severity, is_anomaly = "CRITICAL", True

        return {
            "error"      : round(error, 8),
            "threshold"  : round(self.threshold, 8),
            "error_ratio": round(ratio, 4),
            "is_anomaly" : is_anomaly,
            "severity"   : severity,
        }

    # ── Forecaster prediction ─────────────────────────────────────────────────

    def _run_forecaster(
        self,
        normalised : np.ndarray,
        latest_ts  : datetime,
    ) -> list:
        """
        Runs LSTM Forecaster on the normalised window.

        Returns a list of HORIZON dicts, each representing one future step:
            {
                step          : int    (1-based step index)
                seconds_ahead : int    (step × LOG_INTERVAL_SEC)
                predicted_at  : str    (ISO timestamp of predicted moment)
                cpu           : float  (predicted CPU %)
                memory        : float  (predicted Memory %)
                disk          : float  (predicted Disk %)
            }
        Returns empty list if forecaster is not loaded.
        """
        if not self.forecaster_ready:
            return []

        x = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_norm = self.forecaster.predict(x)   # (1, horizon, 3)

        pred_norm_np = pred_norm.squeeze(0).cpu().numpy()   # (horizon, 3)

        # Inverse transform → back to % values
        pred_pct = self.scaler.inverse_transform(pred_norm_np)   # (horizon, 3)

        # Build result list
        base_ts = latest_ts if isinstance(latest_ts, datetime) else \
                  datetime.fromisoformat(str(latest_ts))

        forecast = []
        for step_idx in range(HORIZON):
            secs_ahead    = (step_idx + 1) * LOG_INTERVAL_SEC
            predicted_at  = base_ts + timedelta(seconds=secs_ahead)

            # Clamp to [0, 100] — model can occasionally predict slightly out of range
            cpu    = float(np.clip(pred_pct[step_idx, 0], 0.0, 100.0))
            memory = float(np.clip(pred_pct[step_idx, 1], 0.0, 100.0))
            disk   = float(np.clip(pred_pct[step_idx, 2], 0.0, 100.0))

            forecast.append({
                "step"         : step_idx + 1,
                "seconds_ahead": secs_ahead,
                "predicted_at" : predicted_at.isoformat(),
                "cpu"          : round(cpu,    2),
                "memory"       : round(memory, 2),
                "disk"         : round(disk,   2),
            })

        return forecast

    def _check_forecast_vs_limits(
        self,
        forecast : list,
        limits   : dict,
    ) -> list:
        """
        Checks each forecast step against user-set Warning/Critical limits.
        Returns a list of breach predictions sorted by time.

        Args:
            forecast : output from _run_forecaster()
            limits   : {
                cpu_warning, cpu_critical,
                memory_warning, memory_critical,
                disk_warning, disk_critical
            }

        Returns list of dicts:
            {
                metric         : "cpu" | "memory" | "disk"
                label          : "CPU Usage" etc.
                severity       : "WARNING" | "CRITICAL"
                predicted_value: float
                limit          : float
                seconds_ahead  : int
                predicted_at   : str  (ISO timestamp)
                criteria       : str  (human-readable)
            }
        """
        if not forecast or not limits:
            return []

        breaches = []
        metric_labels = {
            "cpu"   : "CPU Usage",
            "memory": "Memory Usage",
            "disk"  : "Disk Usage",
        }

        for step in forecast:
            for metric, label in metric_labels.items():
                val   = step[metric]
                c_lim = limits.get(f"{metric}_critical")
                w_lim = limits.get(f"{metric}_warning")

                if c_lim is not None and val > c_lim:
                    breaches.append({
                        "metric"         : metric,
                        "label"          : label,
                        "severity"       : "CRITICAL",
                        "predicted_value": round(val, 2),
                        "limit"          : c_lim,
                        "seconds_ahead"  : step["seconds_ahead"],
                        "predicted_at"   : step["predicted_at"],
                        "criteria"       : f"{label} > {c_lim}% (critical)",
                    })
                elif w_lim is not None and val > w_lim:
                    breaches.append({
                        "metric"         : metric,
                        "label"          : label,
                        "severity"       : "WARNING",
                        "predicted_value": round(val, 2),
                        "limit"          : w_lim,
                        "seconds_ahead"  : step["seconds_ahead"],
                        "predicted_at"   : step["predicted_at"],
                        "criteria"       : f"{label} > {w_lim}% (warning)",
                    })

        # Keep only earliest breach per metric+severity combo
        seen = set()
        unique_breaches = []
        for b in sorted(breaches, key=lambda x: x["seconds_ahead"]):
            key = (b["metric"], b["severity"])
            if key not in seen:
                seen.add(key)
                unique_breaches.append(b)

        return unique_breaches

    # ── Public run() ─────────────────────────────────────────────────────────

    def run(self, limits: dict = None) -> dict:
        """
        Full inference cycle — runs both models.

        Args:
            limits : user-set metric limits from the frontend
                     { cpu_warning, cpu_critical, memory_warning, ... }
                     Pass None if no limits are set yet.

        Returns dict:
            status           : "ok" | "no_data"
            timestamp        : datetime of latest DB row
            cpu, memory, disk: current metric values
            --- Autoencoder (detection) ---
            error            : reconstruction MSE
            threshold        : current anomaly threshold
            error_ratio      : error / threshold
            is_anomaly       : bool
            severity         : "NORMAL" | "WARNING" | "CRITICAL"
            flagged_metrics  : list of human-readable breach descriptions
            actual_rows      : real rows in window (< 60 during warmup)
            warming_up       : bool
            --- Forecaster (prediction) ---
            forecast         : list of 12 predicted readings (empty if not trained)
            forecast_breaches: list of predicted threshold violations
            forecaster_ready : bool — whether the forecaster model is loaded
        """
        raw, latest, actual_rows = self._fetch_window()

        if raw is None:
            return {"status": "no_data", "actual_rows": 0,
                    "forecaster_ready": self.forecaster_ready}

        # Normalise with shared scaler
        normalised = self.scaler.transform(raw)

        # ── 1. Autoencoder detection ──────────────────────────────────────────
        ae_result = self._run_autoencoder(normalised)

        # ── 2. Forecaster prediction ──────────────────────────────────────────
        forecast = self._run_forecaster(normalised, latest["timestamp"])

        # ── 3. Check forecast against user limits ─────────────────────────────
        forecast_breaches = self._check_forecast_vs_limits(
            forecast, limits or {}
        )

        return {
            "status"          : "ok",
            "timestamp"       : latest["timestamp"],
            "cpu"             : latest["cpu_usage"],
            "memory"          : latest["memory_usage"],
            "disk"            : latest["disk_usage"],
            # Detection
            "error"           : ae_result["error"],
            "threshold"       : ae_result["threshold"],
            "error_ratio"     : ae_result["error_ratio"],
            "is_anomaly"      : ae_result["is_anomaly"],
            "severity"        : ae_result["severity"],
            "flagged_metrics" : self._flag_metrics(raw) if ae_result["is_anomaly"] else [],
            "actual_rows"     : actual_rows,
            "warming_up"      : actual_rows < SEQ_LEN,
            # Prediction
            "forecast"        : forecast,
            "forecast_breaches": forecast_breaches,
            "forecaster_ready": self.forecaster_ready,
        }

    def _flag_metrics(self, raw: np.ndarray) -> list:
        last  = raw[-1]
        flags = []
        if last[0] > 85: flags.append(f"CPU {last[0]:.1f}% > 85%")
        if last[1] > 85: flags.append(f"Memory {last[1]:.1f}% > 85%")
        if last[2] > 90: flags.append(f"Disk {last[2]:.1f}% > 90%")
        return flags