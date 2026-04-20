# ══════════════════════════════════════════════════════════
#  AIOps — Docker Commands
# ══════════════════════════════════════════════════════════

.PHONY: build up down logs restart train train-ae train-fc train-fc-real check-data evaluate reload status clean

# ── Build & Run ───────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up -d
	@echo ""
	@echo "Dashboard: http://localhost:5173"
	@echo "API:       http://localhost:8000"
	@echo ""

down:
	docker compose down

restart:
	docker compose restart backend frontend

logs:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-generator:
	docker compose logs -f log_generator

# ── Model Training ────────────────────────────────────────

train:
	docker compose run --rm ai_model train-all

train-ae:
	docker compose run --rm ai_model train-ae

train-fc:
	docker compose run --rm ai_model train-fc

train-fc-real:
	docker compose run --rm ai_model train-fc-real

check-data:
	docker compose run --rm ai_model check-data

evaluate:
	docker compose run --rm ai_model evaluate

# ── Hot Reload ────────────────────────────────────────────

reload:
	@echo "Hot-reloading forecaster model..."
	curl -s -X POST http://localhost:8000/model/reload-forecaster | python3 -m json.tool
	@echo ""

reload-full:
	@echo "Reloading all models..."
	curl -s -X POST http://localhost:8000/model/load | python3 -m json.tool
	@echo ""

# ── Status ────────────────────────────────────────────────

status:
	@echo "=== Containers ==="
	docker compose ps
	@echo ""
	@echo "=== Model Status ==="
	@curl -s http://localhost:8000/model/status | python3 -m json.tool 2>/dev/null || echo "Backend not reachable"
	@echo ""
	@echo "=== Debug Paths ==="
	@curl -s http://localhost:8000/model/debug-paths | python3 -m json.tool 2>/dev/null || echo "Backend not reachable"

health:
	@curl -s http://localhost:8000/ | python3 -m json.tool

# ── Cleanup ───────────────────────────────────────────────

clean:
	docker compose down -v --rmi local
	@echo "Cleaned up containers, volumes, and local images."