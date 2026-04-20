# ai_model/Dockerfile
# Trains the LSTM Autoencoder and LSTM Forecaster models.
#
# This is a BUILD-TIME container — it runs training, writes model artifacts
# to the /app/saved/ directory (bind-mounted), then exits.
#
# The saved/ directory is shared with the backend container via a Docker volume,
# so trained models are immediately available for inference without rebuilding.
#
# Usage in docker-compose:
#   docker compose run ai_model              # train both models
#   docker compose run ai_model train-ae     # train autoencoder only
#   docker compose run ai_model train-fc     # train forecaster (synthetic) only
#   docker compose run ai_model train-fc-real # train forecaster (real data) only

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all ai_model source
COPY . .

# Create saved directory
RUN mkdir -p saved

# Entrypoint script that routes to the right training command
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]

# Default: train everything
CMD ["train-all"]