"""
frontend/backend/inference_engine.py
--------------------------------------
Runs BOTH LSTM models on every inference cycle.
Docker-compatible: reads SAVED_DIR from environment variable.

FIXES:
  - Separate scalers: ae_scaler (synthetic) and fc_scaler (real data)
  - _fetch_window() pads to requested window_size, not hardcoded 60
  - Each model normalizes with its own scaler
  - Sanity check at startup
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
# Docker: ENV_PATH is set, or .env is loaded via env_file in docker-compose
# Local:  resolve relative to this file
_ENV_PATH = os.getenv("ENV_PATH")
if _ENV_PATH:
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    _local_env = (
        Path(__file__).resolve()
        .parent.parent.parent
        / "log_generator" / ".env"
    )
    if _local_env.exists():
        load_dotenv(dotenv_path=_local_env)

DATABASE_URL     = os.getenv("DATABASE_URL")
LOG_INTERVAL_SEC = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

# ── Model constants ────────────────────────────────────────────────────────────
SEQ_LEN          = 60
FEATURES         = ["cpu", "memory", "disk"]

AE_HIDDEN_SIZE   = 64
AE_LATENT_SIZE   = 32
AE_NUM_LAYERS    = 2
AE_DROPOUT       = 0.2

FC_HIDDEN_SIZE   = 128
FC_NUM_LAYERS    = 2
FC_DROPOUT       = 0.2
FC_HORIZON_DEFAULT  = 12
FC_LOOKBACK_DEFAULT = 60


def _load_forecaster_config(saved_dir: Path) -> tuple:
    config_path = saved_dir / "forecaster_config.txt"
    if not config_path.exists():
        return FC_HORIZON_DEFAULT, FC_LOOKBACK_DEFAULT

    config = {}
    with open(config_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=")
                config[k.strip()] = v.strip()

    horizon  = int(config.get("horizon",  FC_HORIZON_DEFAULT))
    lookback = int(config.get("lookback", FC_LOOKBACK_DEFAULT))
    return horizon, lookback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_scaler_info(label, scaler):
    try:
        mins = scaler.data_min_
        maxs = scaler.data_max_
        print(f"   {label} ranges: "
              f"CPU [{mins[0]:.1f}-{maxs[0]:.1f}]  "
              f"MEM [{mins[1]:.1f}-{maxs[1]:.1f}]  "
              f"DISK [{mins[2]:.1f}-{maxs[2]:.1f}]")
    except Exception:
        pass


class InferenceEngine:

    def __init__(self, saved_dir: Path):
        self.saved_dir = Path(saved_dir)
        print(f"   Loading models from: {self.saved_dir}")

        # ── Scalers (separate per model) ──────────────────────────────────────
        real_scaler_path = self.saved_dir / "scaler_real.pkl"
        scaler_path      = self.saved_dir / "scaler.pkl"

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.ae_scaler = pickle.load(f)
            print(f"   Autoencoder scaler: scaler.pkl (synthetic)")
            _log_scaler_info("AE scaler", self.ae_scaler)
        else:
            raise FileNotFoundError(f"scaler.pkl not found at {scaler_path}")

        if real_scaler_path.exists():
            with open(real_scaler_path, "rb") as f:
                self.fc_scaler = pickle.load(f)
            print(f"   Forecaster scaler : scaler_real.pkl (real data)")
            _log_scaler_info("FC scaler", self.fc_scaler)
        else:
            self.fc_scaler = self.ae_scaler
            print(f"   Forecaster scaler : scaler.pkl (synthetic, shared)")

        self.scaler = self.ae_scaler  # legacy compat

        # ── Autoencoder ───────────────────────────────────────────────────────
        from model.lstm_autoencoder import LSTMAutoencoder

        ae_path = self.saved_dir / "lstm_autoencoder.pth"
        if not ae_path.exists():
            raise FileNotFoundError(f"lstm_autoencoder.pth not found at {ae_path}")

        self.autoencoder = LSTMAutoencoder(
            input_size=3, hidden_size=AE_HIDDEN_SIZE, latent_size=AE_LATENT_SIZE,
            num_layers=AE_NUM_LAYERS, dropout=AE_DROPOUT, seq_len=SEQ_LEN,
        )
        self.autoencoder.load_state_dict(
            torch.load(str(ae_path), map_location=DEVICE, weights_only=True)
        )
        self.autoencoder.to(DEVICE)
        self.autoencoder.eval()
        print(f"   lstm_autoencoder.pth loaded")

        # ── Forecaster ────────────────────────────────────────────────────────
        self.forecaster       = None
        self.forecaster_ready = False

        fc_path = self.saved_dir / "lstm_forecaster.pth"
        if fc_path.exists():
            try:
                _ai_model_dir = str(self.saved_dir.parent)
                if _ai_model_dir not in sys.path:
                    sys.path.insert(0, _ai_model_dir)

                self.fc_horizon, self.fc_lookback = _load_forecaster_config(self.saved_dir)
                print(f"   Forecaster: horizon={self.fc_horizon} lookback={self.fc_lookback} "
                      f"({self.fc_horizon * LOG_INTERVAL_SEC}s ahead)")

                from model.lstm_forecaster import LSTMForecaster
                self.forecaster = LSTMForecaster(
                    input_size=3, hidden_size=FC_HIDDEN_SIZE, num_layers=FC_NUM_LAYERS,
                    dropout=FC_DROPOUT, lookback=self.fc_lookback, horizon=self.fc_horizon,
                )
                self.forecaster.load_state_dict(
                    torch.load(str(fc_path), map_location=DEVICE, weights_only=True)
                )
                self.forecaster.to(DEVICE)
                self.forecaster.eval()
                self.forecaster_ready = True
                print(f"   lstm_forecaster.pth loaded")
                self._forecaster_sanity_check()
            except Exception as e:
                import traceback
                print(f"   Forecaster load failed: {e}")
                traceback.print_exc()
        else:
            print(f"   lstm_forecaster.pth not found — prediction disabled")

        # ── Threshold ─────────────────────────────────────────────────────────
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

        print(f"\n   InferenceEngine ready | threshold={self.threshold:.6f} | "
              f"device={DEVICE} | forecaster={'ON' if self.forecaster_ready else 'OFF'}")

    def _forecaster_sanity_check(self):
        try:
            try:
                high_vals = self.fc_scaler.data_max_ * 0.9 + self.fc_scaler.data_min_ * 0.1
            except Exception:
                high_vals = np.array([80.0, 80.0, 80.0])

            test_raw  = np.tile(high_vals, (self.fc_lookback, 1)).astype(np.float32)
            test_norm = self.fc_scaler.transform(test_raw)
            x = torch.tensor(test_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = self.forecaster.predict(x).squeeze(0).cpu().numpy()
            pred_pct = self.fc_scaler.inverse_transform(pred)

            print(f"   Sanity: input=flat CPU {high_vals[0]:.1f}% -> "
                  f"step1={pred_pct[0,0]:.1f}% last={pred_pct[-1,0]:.1f}%")
            if abs(pred_pct[0, 0] - high_vals[0]) > high_vals[0] * 0.5:
                print(f"   ** FORECAST WARNING: model may not have learned meaningful patterns")
        except Exception as e:
            print(f"   Sanity check skipped: {e}")

    def _fetch_window(self, window_size=None):
        if window_size is None:
            window_size = SEQ_LEN
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, cpu_usage, memory_usage, disk_usage "
            "FROM system_logs ORDER BY timestamp DESC LIMIT %s",
            (window_size,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return None, None, 0

        rows        = list(reversed(rows))
        actual_rows = len(rows)

        latest = {
            "timestamp":    rows[-1][0],
            "cpu_usage":    float(rows[-1][1]),
            "memory_usage": float(rows[-1][2]),
            "disk_usage":   float(rows[-1][3]),
        }

        raw = np.array(
            [[float(r[1]), float(r[2]), float(r[3])] for r in rows],
            dtype=np.float32
        )

        if actual_rows < window_size:
            pad = np.tile(raw[0], (window_size - actual_rows, 1))
            raw = np.vstack([pad, raw])

        return raw, latest, actual_rows

    def _run_autoencoder(self, raw):
        ae_raw = raw[-SEQ_LEN:]
        normalised = self.ae_scaler.transform(ae_raw)
        x     = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        error = self.autoencoder.reconstruction_error(x).item()
        ratio = error / self.threshold

        if error < self.threshold:
            severity, is_anomaly = "NORMAL", False
        elif error < self.threshold * 1.5:
            severity, is_anomaly = "WARNING", True
        else:
            severity, is_anomaly = "CRITICAL", True

        return {
            "error":       round(error, 8),
            "threshold":   round(self.threshold, 8),
            "error_ratio": round(ratio, 4),
            "is_anomaly":  is_anomaly,
            "severity":    severity,
        }

    def _run_forecaster(self, raw, latest_ts):
        if not self.forecaster_ready:
            return []

        fc_raw     = raw[-self.fc_lookback:]
        normalised = self.fc_scaler.transform(fc_raw)
        x = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_norm = self.forecaster.predict(x)
        pred_pct = self.fc_scaler.inverse_transform(
            pred_norm.squeeze(0).cpu().numpy()
        )

        base_ts = latest_ts if isinstance(latest_ts, datetime) else \
                  datetime.fromisoformat(str(latest_ts))

        forecast = []
        for i in range(self.fc_horizon):
            secs = (i + 1) * LOG_INTERVAL_SEC
            forecast.append({
                "step":          i + 1,
                "seconds_ahead": secs,
                "predicted_at":  (base_ts + timedelta(seconds=secs)).isoformat(),
                "cpu":           round(float(np.clip(pred_pct[i, 0], 0, 100)), 2),
                "memory":        round(float(np.clip(pred_pct[i, 1], 0, 100)), 2),
                "disk":          round(float(np.clip(pred_pct[i, 2], 0, 100)), 2),
            })
        return forecast

    def _check_forecast_vs_limits(self, forecast, limits):
        if not forecast or not limits:
            return []

        breaches = []
        labels = {"cpu": "CPU Usage", "memory": "Memory Usage", "disk": "Disk Usage"}

        for step in forecast:
            for metric, label in labels.items():
                val   = step[metric]
                c_lim = limits.get(f"{metric}_critical")
                w_lim = limits.get(f"{metric}_warning")

                if c_lim is not None and val > c_lim:
                    breaches.append({
                        "metric": metric, "label": label, "severity": "CRITICAL",
                        "predicted_value": round(val, 2), "limit": c_lim,
                        "seconds_ahead": step["seconds_ahead"],
                        "predicted_at": step["predicted_at"],
                        "criteria": f"{label} > {c_lim}% (critical)",
                    })
                elif w_lim is not None and val > w_lim:
                    breaches.append({
                        "metric": metric, "label": label, "severity": "WARNING",
                        "predicted_value": round(val, 2), "limit": w_lim,
                        "seconds_ahead": step["seconds_ahead"],
                        "predicted_at": step["predicted_at"],
                        "criteria": f"{label} > {w_lim}% (warning)",
                    })

        seen = set()
        unique = []
        for b in sorted(breaches, key=lambda x: x["seconds_ahead"]):
            key = (b["metric"], b["severity"])
            if key not in seen:
                seen.add(key)
                unique.append(b)
        return unique

    def run(self, limits=None):
        needed = SEQ_LEN
        if self.forecaster_ready:
            needed = max(SEQ_LEN, self.fc_lookback)

        raw, latest, actual_rows = self._fetch_window(window_size=needed)
        if raw is None:
            return {"status": "no_data", "actual_rows": 0,
                    "forecaster_ready": self.forecaster_ready}

        ae_result = self._run_autoencoder(raw)
        forecast  = self._run_forecaster(raw, latest["timestamp"])
        breaches  = self._check_forecast_vs_limits(forecast, limits or {})

        return {
            "status":           "ok",
            "timestamp":        latest["timestamp"],
            "cpu":              latest["cpu_usage"],
            "memory":           latest["memory_usage"],
            "disk":             latest["disk_usage"],
            "error":            ae_result["error"],
            "threshold":        ae_result["threshold"],
            "error_ratio":      ae_result["error_ratio"],
            "is_anomaly":       ae_result["is_anomaly"],
            "severity":         ae_result["severity"],
            "flagged_metrics":  self._flag_metrics(raw) if ae_result["is_anomaly"] else [],
            "actual_rows":      actual_rows,
            "warming_up":       actual_rows < SEQ_LEN,
            "forecast":         forecast,
            "forecast_breaches": breaches,
            "forecaster_ready": self.forecaster_ready,
        }

    def _flag_metrics(self, raw):
        last = raw[-1]
        flags = []
        if last[0] > 85: flags.append(f"CPU {last[0]:.1f}% > 85%")
        if last[1] > 85: flags.append(f"Memory {last[1]:.1f}% > 85%")
        if last[2] > 90: flags.append(f"Disk {last[2]:.1f}% > 90%")
        return flags