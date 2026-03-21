# Chatbot with OpenLLMetry → Elastic
# Run: make setup  then edit .env (ELASTIC_APM_SERVER_URL) and run: make up

.PHONY: setup up build down dev help

help:
	@echo "Targets:"
	@echo "  make setup   - Copy .env.example to .env if missing (then edit .env and set ELASTIC_APM_SERVER_URL)"
	@echo "  make build   - Build Docker images"
	@echo "  make up      - Start stack (docker compose up -d)"
	@echo "  make down    - Stop stack"
	@echo "  make dev     - Local dev: create venv, install deps (activate with: source .venv/bin/activate)"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example. Edit .env and set ELASTIC_APM_SERVER_URL (and optionally ELASTIC_APM_SECRET_TOKEN)."; \
	else \
		echo ".env already exists; no change."; \
	fi

build:
	docker compose build

up: setup
	docker compose up -d

down:
	docker compose down

dev:
	python -m venv .venv
	@echo "Run: source .venv/bin/activate  then: pip install -r requirements.txt"
