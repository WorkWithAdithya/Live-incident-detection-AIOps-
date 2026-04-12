"""
frontend/backend/main.py
------------------------
FastAPI backend — Phase 2 update.

Changes from Phase 1:
  - AppState now stores user limits (forwarded to InferenceEngine.run())
  - POST /limits    : frontend sends current user-set metric limits
  - GET  /forecast  : returns latest forecast snapshot on demand
  - SSE stream      : each event now includes forecast + forecast_breaches
  - InferenceEngine.run(limits) called with current limits every cycle
"""

import os
import sys
import json
import asyncio
import psycopg2
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AI_MODEL_DIR = _PROJECT_ROOT / "ai_model"
_SAVED_DIR    = _AI_MODEL_DIR / "saved"
_ENV_PATH     = _PROJECT_ROOT / "log_generator" / ".env"

sys.path.insert(0, str(_AI_MODEL_DIR))
load_dotenv(dotenv_path=_ENV_PATH)

DATABASE_URL     = os.getenv("DATABASE_URL")
LOG_INTERVAL_SEC = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

from inference_engine import InferenceEngine
from evaluator        import Evaluator
from session_metrics  import SessionMetrics
from email_notifier   import notifier as email_notifier


# ── App state ─────────────────────────────────────────────────────────────────
class AppState:
    engine:          Optional[InferenceEngine] = None
    session_metrics: SessionMetrics            = SessionMetrics()
    alerts:          list                      = []
    latest_forecast: list                      = []   # last forecaster output
    # User-set limits — updated by POST /limits, passed to engine.run()
    limits: dict = {
        "cpu_warning":    None,
        "cpu_critical":   None,
        "memory_warning": None,
        "memory_critical":None,
        "disk_warning":   None,
        "disk_critical":  None,
    }
    MAX_ALERTS = 100

state = AppState()


# ── DB helpers ────────────────────────────────────────────────────────────────

def db_fetch_logs(limit: int = 200, after_id: int = 0) -> list:
    conn   = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    if after_id > 0:
        cursor.execute(
            """
            SELECT id, timestamp, cpu_usage, memory_usage, disk_usage
            FROM system_logs WHERE id > %s
            ORDER BY timestamp ASC LIMIT %s
            """,
            (after_id, limit)
        )
    else:
        cursor.execute(
            """
            SELECT id, timestamp, cpu_usage, memory_usage, disk_usage
            FROM system_logs
            ORDER BY timestamp DESC LIMIT %s
            """,
            (limit,)
        )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "id"       : r[0],
            "timestamp": r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1]),
            "cpu"      : float(r[2]),
            "memory"   : float(r[3]),
            "disk"     : float(r[4]),
        })
    if after_id == 0:
        result = list(reversed(result))
    return result


def db_max_id() -> int:
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM system_logs")
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return int(row[0]) if row and row[0] else 0
    except Exception:
        return 0


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AIOps backend starting...")
    try:
        state.engine = InferenceEngine(_SAVED_DIR)
    except Exception as e:
        print(f"⚠️  Model not auto-loaded: {e}")
    yield
    print("🛑 Backend shutting down.")


app = FastAPI(title="AIOps Incident Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class ThresholdRequest(BaseModel):
    threshold: float

class LimitsRequest(BaseModel):
    cpu_warning:     Optional[float] = None
    cpu_critical:    Optional[float] = None
    memory_warning:  Optional[float] = None
    memory_critical: Optional[float] = None
    disk_warning:    Optional[float] = None
    disk_critical:   Optional[float] = None

class RuleAlertRequest(BaseModel):
    severity:       str
    cpu:            float
    memory:         float
    disk:           float
    exceeded:       list
    threshold_info: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status"          : "ok",
        "service"         : "AIOps Backend",
        "forecaster_ready": state.engine.forecaster_ready if state.engine else False,
        "time"            : datetime.now().isoformat(),
    }


@app.get("/model/status")
def model_status():
    if state.engine is None:
        return {"loaded": False, "threshold": None, "default_threshold": None,
                "forecaster_ready": False}
    return {
        "loaded"           : True,
        "threshold"        : state.engine.threshold,
        "default_threshold": state.engine.default_threshold,
        "mean_error"       : state.engine.mean_error,
        "std_error"        : state.engine.std_error,
        "optimal_threshold": state.engine.optimal_threshold,
        "forecaster_ready" : state.engine.forecaster_ready,
    }


@app.post("/model/load")
def load_model():
    try:
        state.engine = InferenceEngine(_SAVED_DIR)
        return {
            "success"         : True,
            "threshold"       : state.engine.threshold,
            "forecaster_ready": state.engine.forecaster_ready,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload-forecaster")
def reload_forecaster():
    """
    Hot-reloads just the forecaster without restarting the server.
    Call this after running python -m model.train_forecaster.
    """
    if state.engine is None:
        raise HTTPException(status_code=400, detail="Load the main model first")
    try:
        fc_path = _SAVED_DIR / "lstm_forecaster.pth"
        if not fc_path.exists():
            return {
                "success": False,
                "message": f"lstm_forecaster.pth not found at {fc_path}. Train it first.",
                "looked_at": str(fc_path),
            }
        import sys as _sys, torch, traceback
        _ai_model_dir = str(_SAVED_DIR.parent)
        if _ai_model_dir not in _sys.path:
            _sys.path.insert(0, _ai_model_dir)
        from model.lstm_forecaster import LSTMForecaster
        fc = LSTMForecaster(
            input_size=3, hidden_size=128, num_layers=2,
            dropout=0.2, lookback=60, horizon=12,
        )
        fc.load_state_dict(torch.load(str(fc_path), map_location="cpu", weights_only=True))
        fc.eval()
        state.engine.forecaster       = fc
        state.engine.forecaster_ready = True
        return {
            "success"         : True,
            "forecaster_ready": True,
            "message"         : f"Forecaster loaded from {fc_path}",
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"success": False, "message": str(e), "traceback": tb}


@app.get("/model/debug-paths")
def debug_paths():
    """Returns resolved paths so you can verify the backend is looking in the right place."""
    import os
    saved_contents = []
    if _SAVED_DIR.exists():
        saved_contents = [f.name for f in _SAVED_DIR.iterdir()]
    return {
        "project_root"    : str(_PROJECT_ROOT),
        "ai_model_dir"    : str(_AI_MODEL_DIR),
        "saved_dir"       : str(_SAVED_DIR),
        "saved_dir_exists": _SAVED_DIR.exists(),
        "saved_contents"  : sorted(saved_contents),
        "forecaster_file" : str(_SAVED_DIR / "lstm_forecaster.pth"),
        "forecaster_exists": (_SAVED_DIR / "lstm_forecaster.pth").exists(),
        "forecaster_ready": state.engine.forecaster_ready if state.engine else False,
    }


@app.post("/threshold")
def set_threshold(req: ThresholdRequest):
    if state.engine is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    if req.threshold <= 0:
        raise HTTPException(status_code=422, detail="Threshold must be > 0")
    state.engine.threshold = req.threshold
    return {"success": True, "threshold": req.threshold}


@app.post("/limits")
def set_limits(req: LimitsRequest):
    """
    Receives the user-set metric limits from the frontend.
    Stored in AppState and forwarded to engine.run() on every SSE cycle.
    """
    state.limits = req.dict()
    return {"success": True, "limits": state.limits}


@app.get("/limits")
def get_limits():
    return {"limits": state.limits}


@app.get("/logs")
def get_logs(n: int = 200):
    try:
        rows = db_fetch_logs(limit=n)
        return {"logs": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
def get_alerts(n: int = 50):
    return {"alerts": state.alerts[-n:]}


@app.get("/forecast")
def get_forecast():
    """
    Returns the latest LSTM forecast snapshot.
    The frontend can poll this on demand (e.g. on page load).
    """
    return {
        "forecast"        : state.latest_forecast,
        "forecaster_ready": state.engine.forecaster_ready if state.engine else False,
        "timestamp"       : datetime.now().isoformat(),
    }


@app.get("/session/metrics")
def get_session_metrics():
    return state.session_metrics.compute()


@app.get("/evaluate")
def run_evaluation():
    if state.engine is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    try:
        ev = Evaluator(_AI_MODEL_DIR, _SAVED_DIR, state.engine.threshold)
        return ev.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alert/rule")
async def rule_alert(req: RuleAlertRequest):
    if req.severity not in ("WARNING", "CRITICAL"):
        return {"sent": False, "reason": "severity must be WARNING or CRITICAL"}
    email_notifier.notify(
        severity       = req.severity,
        metrics        = {"cpu": req.cpu, "memory": req.memory, "disk": req.disk},
        exceeded       = req.exceeded,
        threshold_info = req.threshold_info,
    )
    return {
        "sent"    : email_notifier.enabled,
        "severity": req.severity,
        "to"      : os.getenv("ALERT_EMAIL_TO", "not configured"),
    }


@app.get("/alert/email-status")
def email_status():
    return {
        "enabled"  : email_notifier.enabled,
        "to"       : os.getenv("ALERT_EMAIL_TO",  "") or "not configured",
        "from"     : os.getenv("ALERT_EMAIL_FROM", "") or "not configured",
        "smtp_host": os.getenv("ALERT_SMTP_HOST",  "smtp.gmail.com"),
        "cooldown" : int(os.getenv("ALERT_COOLDOWN_SECONDS", "300")),
    }


# ── SSE Stream ────────────────────────────────────────────────────────────────

async def inference_stream():
    """
    SSE generator — every LOG_INTERVAL_SECONDS:
      1. Fetch new DB rows since last poll (by id)
      2. Run BOTH models via engine.run(limits)
      3. Push each new row enriched with detection + forecast data
    """
    last_id = db_max_id()

    while True:
        await asyncio.sleep(LOG_INTERVAL_SEC)

        if state.engine is None:
            yield f"data: {json.dumps({'status': 'model_not_loaded'})}\n\n"
            continue

        try:
            # ── Fetch new rows ────────────────────────────────────────────────
            new_rows = db_fetch_logs(limit=50, after_id=last_id)

            if not new_rows:
                yield f"data: {json.dumps({'status': 'no_new_rows', 'last_id': last_id})}\n\n"
                continue

            # ── Run both models with current user limits ───────────────────────
            result = state.engine.run(limits=state.limits)

            if result.get("status") != "ok":
                yield f"data: {json.dumps({'status': result.get('status', 'error')})}\n\n"
                continue

            # Cache latest forecast for GET /forecast
            state.latest_forecast = result.get("forecast", [])

            # ── Push one SSE event per new DB row ─────────────────────────────
            for row in new_rows:
                last_id = max(last_id, row["id"])

                entry = {
                    "status"           : "ok",
                    "id"               : row["id"],
                    "timestamp"        : row["timestamp"],
                    "cpu"              : row["cpu"],
                    "memory"           : row["memory"],
                    "disk"             : row["disk"],
                    # Detection (autoencoder)
                    "error"            : result["error"],
                    "threshold"        : result["threshold"],
                    "error_ratio"      : result["error_ratio"],
                    "severity"         : result["severity"],
                    "is_anomaly"       : result["is_anomaly"],
                    "actual_rows"      : result["actual_rows"],
                    "warming_up"       : result["warming_up"],
                    "flagged"          : result["flagged_metrics"],
                    # Prediction (forecaster)
                    "forecast"         : result["forecast"],
                    "forecast_breaches": result["forecast_breaches"],
                    "forecaster_ready" : result["forecaster_ready"],
                }

                # Track session metrics
                state.session_metrics.update(
                    is_anomaly = entry["is_anomaly"],
                    error      = entry["error"],
                    threshold  = entry["threshold"],
                )

                # Store LSTM anomaly alerts
                if entry["is_anomaly"]:
                    state.alerts.append(entry)
                    if len(state.alerts) > state.MAX_ALERTS:
                        state.alerts.pop(0)

                # Email on LSTM WARNING/CRITICAL
                if entry["severity"] in ("WARNING", "CRITICAL"):
                    email_notifier.notify(
                        severity       = entry["severity"],
                        metrics        = {
                            "cpu":    entry["cpu"],
                            "memory": entry["memory"],
                            "disk":   entry["disk"],
                        },
                        exceeded       = entry.get("flagged", []),
                        threshold_info = {
                            "LSTM Threshold": round(entry["threshold"], 6)
                        },
                    )

                yield f"data: {json.dumps(entry)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"


@app.get("/stream")
async def stream():
    return StreamingResponse(
        inference_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"              : "no-cache",
            "X-Accel-Buffering"          : "no",
            "Access-Control-Allow-Origin": "*",
        },
    )