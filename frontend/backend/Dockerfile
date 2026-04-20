# frontend/backend/Dockerfile
# FastAPI backend — loads trained LSTM models from the shared saved/ volume
# and serves inference results via REST API + SSE stream.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# Port for FastAPI
EXPOSE 8000

# Run uvicorn — 1 worker (model is loaded in memory, not multiprocess-safe)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]