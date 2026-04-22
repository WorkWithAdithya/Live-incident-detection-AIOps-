# AIOps — Intelligent Incident Detection Using Log Analysis

A real-time AIOps system that monitors system metrics (CPU, Memory, Disk), detects anomalies using an LSTM Autoencoder, predicts future threshold breaches using an LSTM Forecaster, sends email alerts, and displays everything on a live React dashboard.

![Dashboard](https://img.shields.io/badge/Dashboard-React_18-61DAFB?style=flat&logo=react)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)
![ML](https://img.shields.io/badge/ML-PyTorch_LSTM-EE4C2C?style=flat&logo=pytorch)
![DB](https://img.shields.io/badge/Database-NeonPostgreSQL-4169E1?style=flat&logo=postgresql)
![Docker](https://img.shields.io/badge/Containerized-Docker-2496ED?style=flat&logo=docker)

---

## What It Does

The system collects real system metrics every second, runs two LSTM models on every cycle, and streams results to a live dashboard:

**Layer 1 — Rule-Based Alerting:** Users set warning thresholds per metric. Alerts fire immediately when values cross limits.

**Layer 2 — LSTM Autoencoder (Detection):** Learns normal system behavior. Flags anomalies when reconstruction error exceeds the calibrated threshold. Achieves F1 = 0.900, ROC-AUC = 0.999 on evaluation data.

**Layer 3 — LSTM Forecaster (Prediction):** Predicts future metric values using an Encoder-Decoder seq2seq architecture. Warns about threshold breaches *before* they happen.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Log Generator   │────▶│  NeonPostgreSQL  │◀────│   AI Model      │
│  (psutil, 1s)    │     │  (Cloud DB)      │     │  (Training)     │
└─────────────────┘     └────────┬─────────┘     └────────┬────────┘
                                 │                         │
                                 ▼                         ▼
                        ┌────────────────┐        ┌──────────────┐
                        │    Backend     │◀───────│ model_artifacts│
                        │   (FastAPI)    │        │   (volume)    │
                        │  Autoencoder   │        └──────────────┘
                        │  Forecaster    │
                        │  SSE Stream    │
                        └───────┬────────┘
                                │ SSE
                                ▼
                        ┌────────────────┐
                        │   Frontend     │
                        │  (React+Vite)  │
                        │  Live Charts   │
                        │  Alert Feed    │
                        │  Predictions   │
                        └────────────────┘
```

---

## Project Structure

```
Live-incident-detection-AIOps/
├── log_generator/              # Collects real system metrics → NeonDB
│   ├── Dockerfile
│   ├── .env                    # DATABASE_URL, LOG_INTERVAL_SECONDS=1
│   ├── requirements.txt
│   └── src/
│       ├── main.py             # Entry point, runs logger loop
│       ├── config.py           # Loads .env
│       ├── logger/
│       │   ├── log_generator.py
│       │   ├── db_writer.py    # Connection-pooled inserts
│       │   ├── db_init.py      # Creates system_logs table
│       │   └── db_cleanup.py   # Retention-based cleanup
│       └── metrics/
│           ├── cpu.py           # psutil.cpu_percent()
│           ├── memory.py        # psutil.virtual_memory().percent
│           └── disk.py          # psutil.disk_usage('/').percent
│
├── ai_model/                   # All ML code
│   ├── Dockerfile
│   ├── docker-entrypoint.sh    # Routes training commands
│   ├── requirements.txt
│   ├── data/
│   │   ├── generate_synthetic_data.py   # 10,000 rows with anomaly patterns
│   │   └── synthetic_logs.csv
│   ├── model/
│   │   ├── lstm_autoencoder.py          # Encoder-Decoder LSTM (detection)
│   │   ├── lstm_forecaster.py           # Seq2Seq LSTM (prediction)
│   │   ├── dataset.py                   # Autoencoder dataset
│   │   ├── forecaster_dataset.py        # Forecaster dataset
│   │   ├── train.py                     # Train autoencoder
│   │   ├── train_forecaster.py          # Train forecaster (synthetic)
│   │   ├── train_forecaster_realdata.py # Train forecaster (real NeonDB data)
│   │   ├── evaluate.py                  # F1, ROC, PR, confusion matrix
│   │   └── inference.py                 # Standalone CLI inference
│   └── saved/                           # Generated after training
│       ├── lstm_autoencoder.pth
│       ├── lstm_forecaster.pth
│       ├── scaler.pkl                   # MinMaxScaler (synthetic)
│       ├── scaler_real.pkl              # MinMaxScaler (real data)
│       ├── threshold.txt
│       └── forecaster_config.txt
│
├── frontend/
│   ├── backend/                # FastAPI inference server
│   │   ├── Dockerfile
│   │   ├── main.py             # Routes + SSE stream
│   │   ├── inference_engine.py # Runs both LSTM models per cycle
│   │   ├── evaluator.py        # On-demand evaluation
│   │   ├── session_metrics.py  # Live F1/Precision/Recall
│   │   ├── email_notifier.py   # Gmail SMTP alerts
│   │   └── requirements.txt
│   └── ui/                     # React dashboard
│       ├── Dockerfile
│       ├── package.json
│       ├── vite.config.js
│       └── src/
│           ├── App.jsx
│           ├── api.js
│           ├── index.css        # Dark theme design tokens
│           └── components/
│               ├── Header.jsx
│               ├── ControlPanel.jsx
│               ├── AlertFeed.jsx
│               ├── LogTable.jsx
│               ├── MetricChart.jsx
│               └── PredictionPanel.jsx
│
├── docker-compose.yml
├── Makefile
├── .dockerignore
└── .gitignore
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- A [NeonDB](https://neon.tech) account (free tier works)

### 1. Clone & Configure

```bash
git clone https://github.com/your-username/Live-incident-detection-AIOps.git
cd Live-incident-detection-AIOps
```

Create `log_generator/.env`:

```env
DATABASE_URL=postgresql://user:password@your-neon-host/dbname?sslmode=require
LOG_INTERVAL_SECONDS=1
LOG_RETENTION_HOURS=24
CLEANUP_INTERVAL_SECONDS=3600

# Email alerts (optional)
ALERT_EMAIL_FROM=sender@gmail.com
ALERT_EMAIL_PASSWORD=your-16-char-app-password
ALERT_EMAIL_TO=admin@example.com
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_COOLDOWN_SECONDS=60
```

### 2. Build

```bash
make build
```

### 3. Train Models

```bash
make train
```

This generates synthetic data, trains the LSTM Autoencoder, and trains the LSTM Forecaster. Model artifacts are saved to a shared Docker volume.

### 4. Start the System

```bash
make up
```

- **Dashboard:** http://localhost:5173
- **API:** http://localhost:8000

### 5. Set Thresholds

Open the dashboard and set warning percentages for CPU, Memory, and Disk in the Control Panel. The system will start alerting when values exceed your limits.

---

## Docker Commands

| Command | What it does |
|---------|-------------|
| `make build` | Build all Docker images |
| `make up` | Start log_generator + backend + frontend |
| `make down` | Stop all services |
| `make logs` | Follow all container logs |
| `make train` | Train all models (synthetic data) |
| `make train-ae` | Train autoencoder only |
| `make train-fc` | Train forecaster (synthetic) |
| `make train-fc-real` | Train forecaster (real NeonDB data) |
| `make check-data` | Check if enough real data exists for training |
| `make reload` | Hot-reload forecaster without restart |
| `make status` | Show container status + model info |
| `make health` | Quick API health check |
| `make clean` | Remove containers, volumes, and images |

---

## How the Models Work

### LSTM Autoencoder (Anomaly Detection)

Learns what **normal** looks like. If current behavior deviates from what the model learned, the reconstruction error spikes and an anomaly is flagged.

```
Input (60 readings) → Encoder LSTM → Latent Vector → Decoder LSTM → Reconstructed Output
                                                                          │
                                                              MSE(input, output) = anomaly score
                                                              score > threshold? → ANOMALY
```

- Trained only on normal data (unsupervised)
- Threshold = mean + 3σ of training reconstruction errors
- F1 = 0.900 | Precision = 0.819 | Recall = 0.998 | ROC-AUC = 0.999

### LSTM Forecaster (Prediction)

Predicts **future** metric values and checks them against your warning thresholds.

```
Input (N past readings) → Encoder LSTM → Hidden State → Decoder LSTM (autoregressive) → N future predictions
                                                                                              │
                                                                          predicted_value > limit? → BREACH ALERT
```

- Encoder-Decoder seq2seq architecture
- Autoregressive decoding (each prediction feeds into the next)
- Teacher forcing during training for stability
- Configurable lookback and horizon via `forecaster_config.txt`

---

## API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Health check |
| GET | `/model/status` | Model loaded, threshold, forecaster_ready |
| POST | `/model/load` | Load models from saved/ |
| POST | `/model/reload-forecaster` | Hot-reload forecaster only |
| POST | `/threshold` | Set LSTM anomaly threshold |
| POST | `/limits` | Sync user metric limits to backend |
| GET | `/logs?n=200` | Last N rows from NeonDB |
| GET | `/stream` | SSE — pushes new rows + LSTM results |
| GET | `/forecast` | Latest forecaster snapshot |
| GET | `/alerts?n=50` | Last N LSTM alerts |
| GET | `/session/metrics` | Live F1/Precision/Recall |
| GET | `/evaluate` | Full autoencoder evaluation |
| POST | `/alert/rule` | Trigger rule-based email alert |
| GET | `/alert/email-status` | Email configured? |

---

## SSE Event Shape

Every SSE event is one new DB row enriched with both model results:

```json
{
  "status": "ok",
  "id": 128,
  "timestamp": "2026-04-19T15:22:07",
  "cpu": 84.7,
  "memory": 61.4,
  "disk": 23.9,
  "error": 0.003421,
  "threshold": 0.001797,
  "error_ratio": 1.904,
  "severity": "WARNING",
  "is_anomaly": true,
  "forecast": [
    {"step": 1, "seconds_ahead": 1, "cpu": 83.2, "memory": 61.1, "disk": 23.9},
    {"step": 2, "seconds_ahead": 2, "cpu": 82.8, "memory": 61.3, "disk": 23.9}
  ],
  "forecast_breaches": [
    {
      "metric": "cpu",
      "label": "CPU Usage",
      "severity": "WARNING",
      "predicted_value": 83.2,
      "limit": 50.0,
      "seconds_ahead": 1,
      "predicted_at": "2026-04-19T15:22:08",
      "criteria": "CPU Usage > 50.0% (warning)"
    }
  ],
  "forecaster_ready": true
}
```

---

## Training on Real Data

For better forecaster accuracy, train on real system data collected during actual load:

```bash
# 1. Let the system run under load for 20+ minutes
stress --cpu 4 --timeout 1200

# 2. Check data availability
make check-data

# 3. Train on real data
make train-fc-real

# 4. Hot-reload without restart
make reload
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Metrics Collection | Python psutil |
| Database | NeonPostgreSQL (cloud) |
| ML Models | PyTorch — LSTM Autoencoder + LSTM Forecaster (seq2seq) |
| ML Evaluation | scikit-learn (F1, ROC-AUC, PR curves, confusion matrix) |
| Backend | FastAPI + Uvicorn + SSE |
| Frontend | React 18 + Vite + Recharts |
| Email Alerts | Python smtplib + Gmail App Password |
| Containerization | Docker + Docker Compose |

---

## Key Design Decisions

- **DB-first SSE:** Backend tracks `last_seen_id`, fetches only new rows (`WHERE id > last_id`). No duplicates, no stale values.
- **Separate scalers:** Autoencoder uses `scaler.pkl` (synthetic), Forecaster uses `scaler_real.pkl` (real data). Each model gets data normalized with the scaler it was trained with.
- **Adaptive windowing:** If fewer rows than needed exist in the DB, the system front-pads with the first available row so inference works from row 1.
- **Sticky predictions:** Forecast breach alerts persist on the dashboard until their predicted timestamp passes, then auto-clear.
- **Hot-reload:** `POST /model/reload-forecaster` reloads the forecaster weights without restarting the server.
- **Host network for metrics:** The log_generator container uses `network_mode: host` so psutil reads the actual host machine's CPU/memory/disk.

---

## License

This project was developed as a Mini Project for M.E in Computer Science and Engineering (Cloud Computing) at Manipal School of Information Sciences, MAHE.

---

## Team

| Name | Reg. Number |
|------|------------|
| Adithya B S | 251100680014 |
| Amrutha V M | 251100680015 |
| Aditya Amlapure | 251100680031 |

**Guide:** Dr. Sathyendranath Malli, Assistant Professor, MSIS, MAHE, Manipal
