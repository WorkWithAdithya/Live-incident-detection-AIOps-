# frontend/ui/Dockerfile
# React + Vite frontend — runs in dev mode with HMR.
# Proxies /api and /stream to the backend container.

FROM node:20-slim

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy source
COPY . .

# Vite dev server port
EXPOSE 5173

# Run Vite dev server
# VITE_API_URL is set via docker-compose environment
CMD ["npm", "run", "dev"]