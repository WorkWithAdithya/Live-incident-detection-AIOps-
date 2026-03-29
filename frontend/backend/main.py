"""
frontend/backend/main.py
------------------------
FastAPI backend for the AIOps Incident Detection Dashboard.

Endpoints:
    GET  /                       health check
    GET  /model/status           model loaded status + current threshold
    POST /model/load             load model from saved/
    POST /threshold              update anomaly threshold
    GET  /stream                 SSE stream — pushes inference result every 5s
    GET  /history                last N inference results (chart hydration)
    GET  /evaluate               run full evaluation, return all metrics + curves
    GET  /alerts                 last N alerts

Run:
    cd frontend/backend
    uvicorn main:app --reload --port 8000
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
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Resolve paths ─────────────────────────────────────────────────────────────
# frontend/backend/main.py
# → parent       = frontend/backend/
# → parent.parent = frontend/
# → parent.parent.parent = your_project/

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AI_MODEL_DIR = _PROJECT_ROOT / "ai_model"
_SAVED_DIR    = _AI_MODEL_DIR / "saved"
_ENV_PATH     = _PROJECT_ROOT / "log_generator" / ".env"

# Add ai_model to Python path so we can import model modules
sys.path.insert(0, str(_AI_MODEL_DIR))

load_dotenv(dotenv_path=_ENV_PATH)

# Now safe to import ai_model modules
from model.lstm_autoencoder import LSTMAutoencoder
from inference_engine import InferenceEngine
from evaluator import Evaluator
from session_metrics import SessionMetrics

# ── App state ─────────────────────────────────────────────────────────────────
class AppState:
    engine:          Optional[InferenceEngine] = None
    session_metrics: SessionMetrics            = SessionMetrics()
    history:         list                      = []   # rolling last 200 results
    alerts:          list                      = []   # rolling last 100 alerts
    MAX_HISTORY      = 200
    MAX_ALERTS       = 100

state = AppState()


# ── Lifespan: auto-load model on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AIOps backend starting...")
    try:
        state.engine = InferenceEngine(_SAVED_DIR)
        print("✅ Model auto-loaded on startup")
    except Exception as e:
        print(f"⚠️  Model not auto-loaded: {e}")
        print("   Call POST /model/load to load it manually.")
    yield
    print("🛑 Backend shutting down.")


app = FastAPI(title="AIOps Incident Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class ThresholdRequest(BaseModel):
    threshold: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status" : "ok",
        "service": "AIOps Incident Detection Backend",
        "time"   : datetime.now().isoformat(),
    }


@app.get("/model/status")
def model_status():
    if state.engine is None:
        return {
            "loaded"            : False,
            "threshold"         : None,
            "default_threshold" : None,
            "mean_error"        : None,
            "std_error"         : None,
        }
    return {
        "loaded"            : True,
        "threshold"         : state.engine.threshold,
        "default_threshold" : state.engine.default_threshold,
        "mean_error"        : state.engine.mean_error,
        "std_error"         : state.engine.std_error,
        "optimal_threshold" : state.engine.optimal_threshold,
        "model_path"        : str(state.engine.model_path),
    }


@app.post("/model/load")
def load_model():
    try:
        state.engine = InferenceEngine(_SAVED_DIR)
        return {
            "success"   : True,
            "message"   : "Model loaded successfully",
            "threshold" : state.engine.threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/threshold")
def set_threshold(req: ThresholdRequest):
    if state.engine is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    if req.threshold <= 0:
        raise HTTPException(status_code=422, detail="Threshold must be > 0")

    state.engine.threshold = req.threshold
    return {
        "success"  : True,
        "threshold": req.threshold,
        "message"  : f"Threshold updated to {req.threshold:.8f}",
    }


@app.get("/history")
def get_history(n: int = 100):
    """Returns last n inference results for chart hydration on page load."""
    return {"history": state.history[-n:]}


@app.get("/alerts")
def get_alerts(n: int = 50):
    """Returns last n alerts."""
    return {"alerts": state.alerts[-n:]}


@app.get("/session/metrics")
def get_session_metrics():
    """Returns live session-level F1, Precision, Recall, etc."""
    return state.session_metrics.compute()


@app.get("/evaluate")
def run_evaluation():
    """
    Runs full evaluation against synthetic_logs.csv.
    Returns all metric data needed to render ROC, PR curves, confusion matrix.
    This is a heavier call — run on demand, not every poll.
    """
    if state.engine is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    try:
        ev = Evaluator(_AI_MODEL_DIR, _SAVED_DIR, state.engine.threshold)
        return ev.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── SSE Stream ────────────────────────────────────────────────────────────────

async def inference_stream():
    """
    Server-Sent Events generator.
    Runs inference every LOG_INTERVAL_SECONDS and pushes the result.
    """
    interval = int(os.getenv("LOG_INTERVAL_SECONDS", 5))

    while True:
        if state.engine is None:
            payload = json.dumps({
                "status" : "model_not_loaded",
                "time"   : datetime.now().isoformat(),
            })
            yield f"data: {payload}\n\n"
            await asyncio.sleep(interval)
            continue

        try:
            result = state.engine.run()

            # ── Store history ──
            if result.get("status") == "ok":
                entry = {
                    "timestamp"  : result["timestamp"].isoformat()
                                   if hasattr(result["timestamp"], "isoformat")
                                   else str(result["timestamp"]),
                    "cpu"        : result["cpu"],
                    "memory"     : result["memory"],
                    "disk"       : result["disk"],
                    "error"      : result["error"],
                    "threshold"  : result["threshold"],
                    "error_ratio": result["error_ratio"],
                    "severity"   : result["severity"],
                    "is_anomaly" : result["is_anomaly"],
                    "actual_rows": result["actual_rows"],
                    "warming_up" : result["warming_up"],
                    "flagged"    : result.get("flagged_metrics", []),
                }

                state.history.append(entry)
                if len(state.history) > state.MAX_HISTORY:
                    state.history.pop(0)

                # ── Update session metrics ──
                state.session_metrics.update(
                    is_anomaly=result["is_anomaly"],
                    error=result["error"],
                    threshold=result["threshold"],
                )

                # ── Store alerts (anomaly only, or first of each severity) ──
                if result["is_anomaly"] or result["severity"] != "NORMAL":
                    alert = {
                        **entry,
                        "flagged": result.get("flagged_metrics", []),
                    }
                    state.alerts.append(alert)
                    if len(state.alerts) > state.MAX_ALERTS:
                        state.alerts.pop(0)

                payload = json.dumps({**entry, "status": "ok"})

            else:
                payload = json.dumps({
                    "status"     : result.get("status", "unknown"),
                    "time"       : datetime.now().isoformat(),
                    "actual_rows": result.get("actual_rows", 0),
                })

        except Exception as e:
            payload = json.dumps({
                "status" : "error",
                "message": str(e),
                "time"   : datetime.now().isoformat(),
            })

        yield f"data: {payload}\n\n"
        await asyncio.sleep(interval)


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