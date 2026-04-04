"""
frontend/backend/main.py
------------------------
FastAPI backend for the AIOps Incident Detection Dashboard.

Key fix: /logs fetches all rows directly from NeonDB so the frontend
chart always reflects real DB data, not just in-memory state.
SSE stream tracks last_seen_id and only pushes genuinely new rows.
"""

import os
import sys
import json
import time
import asyncio
import pickle
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import psycopg2
import torch
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
from evaluator import Evaluator
from session_metrics import SessionMetrics
from email_notifier import notifier as email_notifier


# ── App state ─────────────────────────────────────────────────────────────────
class AppState:
    engine:          Optional[InferenceEngine] = None
    session_metrics: SessionMetrics            = SessionMetrics()
    alerts:          list                      = []
    MAX_ALERTS = 100

state = AppState()


# ── DB helpers ────────────────────────────────────────────────────────────────

def db_fetch_logs(limit: int = 200, after_id: int = 0) -> list:
    """
    Fetch rows from system_logs directly from NeonDB.
    If after_id > 0, only fetch rows with id > after_id (new rows only).
    Returns list of dicts ordered by timestamp ASC.
    """
    conn   = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    if after_id > 0:
        cursor.execute(
            """
            SELECT id, timestamp, cpu_usage, memory_usage, disk_usage
            FROM system_logs
            WHERE id > %s
            ORDER BY timestamp ASC
            LIMIT %s
            """,
            (after_id, limit)
        )
    else:
        cursor.execute(
            """
            SELECT id, timestamp, cpu_usage, memory_usage, disk_usage
            FROM system_logs
            ORDER BY timestamp DESC
            LIMIT %s
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

    # If fetching latest (no after_id), reverse to get ASC order
    if after_id == 0:
        result = list(reversed(result))

    return result


def db_max_id() -> int:
    """Returns the current max id in system_logs."""
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
        print("✅ Model auto-loaded on startup")
    except Exception as e:
        print(f"⚠️  Model not auto-loaded: {e}")
    yield
    print("🛑 Backend shutting down.")


app = FastAPI(title="AIOps Incident Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic ──────────────────────────────────────────────────────────────────
class ThresholdRequest(BaseModel):
    threshold: float

class RuleAlertRequest(BaseModel):
    severity:       str          # "WARNING" | "CRITICAL"
    cpu:            float
    memory:         float
    disk:           float
    exceeded:       list         # list of breach description strings
    threshold_info: dict         # { label: value } of active limits


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "AIOps Backend", "time": datetime.now().isoformat()}


@app.get("/model/status")
def model_status():
    if state.engine is None:
        return {"loaded": False, "threshold": None, "default_threshold": None}
    return {
        "loaded"           : True,
        "threshold"        : state.engine.threshold,
        "default_threshold": state.engine.default_threshold,
        "mean_error"       : state.engine.mean_error,
        "std_error"        : state.engine.std_error,
        "optimal_threshold": state.engine.optimal_threshold,
    }


@app.post("/model/load")
def load_model():
    try:
        state.engine = InferenceEngine(_SAVED_DIR)
        return {"success": True, "threshold": state.engine.threshold}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/threshold")
def set_threshold(req: ThresholdRequest):
    if state.engine is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    if req.threshold <= 0:
        raise HTTPException(status_code=422, detail="Threshold must be > 0")
    state.engine.threshold = req.threshold
    return {"success": True, "threshold": req.threshold}


@app.get("/logs")
def get_logs(n: int = 200):
    """
    Fetches the last n rows DIRECTLY from NeonDB system_logs.
    This is what the frontend uses to hydrate charts on load —
    it reflects the actual database, not in-memory state.
    """
    try:
        rows = db_fetch_logs(limit=n)
        return {"logs": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
def get_alerts(n: int = 50):
    return {"alerts": state.alerts[-n:]}


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
    """
    Called by the frontend whenever a user-set metric limit is breached.
    Triggers an email notification if not on cooldown.
    """
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
    """Returns whether email alerting is configured and enabled."""
    return {
        "enabled"  : email_notifier.enabled,
        "to"       : os.getenv("ALERT_EMAIL_TO",   "") or "not configured",
        "from"     : os.getenv("ALERT_EMAIL_FROM",  "") or "not configured",
        "smtp_host": os.getenv("ALERT_SMTP_HOST",   "smtp.gmail.com"),
        "cooldown" : int(os.getenv("ALERT_COOLDOWN_SECONDS", "300")),
    }


# ── SSE Stream ────────────────────────────────────────────────────────────────

async def inference_stream():
    """
    SSE generator. Every LOG_INTERVAL_SECONDS:
      1. Fetches all new DB rows since last poll (by id)
      2. Runs LSTM inference on the latest window
      3. Pushes each new row enriched with LSTM result to the frontend

    This ensures every distinct DB row is shown in the frontend —
    no duplicates, no missing rows.
    """
    last_id = db_max_id()   # start from current max so we only push new rows

    while True:
        await asyncio.sleep(LOG_INTERVAL_SEC)

        if state.engine is None:
            yield f"data: {json.dumps({'status': 'model_not_loaded'})}\n\n"
            continue

        try:
            # ── 1. Fetch new rows from DB since last poll ──────────────────
            new_rows = db_fetch_logs(limit=50, after_id=last_id)

            if not new_rows:
                # No new rows yet — push a heartbeat so frontend knows stream is alive
                yield f"data: {json.dumps({'status': 'no_new_rows', 'last_id': last_id})}\n\n"
                continue

            # ── 2. Run LSTM inference (uses the latest full window) ────────
            lstm_result = state.engine.run()

            # ── 3. Enrich each new row with LSTM result and push ──────────
            for row in new_rows:
                last_id = max(last_id, row["id"])

                entry = {
                    "status"     : "ok",
                    "id"         : row["id"],
                    "timestamp"  : row["timestamp"],
                    "cpu"        : row["cpu"],
                    "memory"     : row["memory"],
                    "disk"       : row["disk"],
                    # LSTM fields — same for all rows in this batch
                    # (they share the same inference window)
                    "error"      : lstm_result.get("error", 0),
                    "threshold"  : lstm_result.get("threshold", state.engine.threshold),
                    "error_ratio": lstm_result.get("error_ratio", 0),
                    "severity"   : lstm_result.get("severity", "NORMAL"),
                    "is_anomaly" : lstm_result.get("is_anomaly", False),
                    "actual_rows": lstm_result.get("actual_rows", 0),
                    "warming_up" : lstm_result.get("warming_up", False),
                    "flagged"    : lstm_result.get("flagged_metrics", []),
                }

                # Track session metrics
                state.session_metrics.update(
                    is_anomaly=entry["is_anomaly"],
                    error=entry["error"],
                    threshold=entry["threshold"],
                )

                # Store alerts
                if entry["is_anomaly"]:
                    state.alerts.append(entry)
                    if len(state.alerts) > state.MAX_ALERTS:
                        state.alerts.pop(0)

                # Email on LSTM WARNING/CRITICAL
                if entry["severity"] in ("WARNING", "CRITICAL"):
                    email_notifier.notify(
                        severity       = entry["severity"],
                        metrics        = {"cpu": entry["cpu"], "memory": entry["memory"], "disk": entry["disk"]},
                        exceeded       = entry.get("flagged", []),
                        threshold_info = {"LSTM Threshold": round(entry["threshold"], 6)},
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