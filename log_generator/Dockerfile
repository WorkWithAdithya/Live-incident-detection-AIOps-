# log_generator/Dockerfile
# Lightweight Python container that collects system metrics via psutil
# and writes them to NeonDB every LOG_INTERVAL_SECONDS.
#
# IMPORTANT: This container MUST run with network_mode: host
# so that psutil reads the HOST machine's CPU/memory/disk, not the container's.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Run the logger
CMD ["python", "-u", "src/main.py"]