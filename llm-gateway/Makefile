.PHONY: setup install dev test test-cov lint format type-check clean run docker-up docker-down docker-build help

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
UVICORN := $(VENV)/bin/uvicorn

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)LLM Gateway - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Create virtual environment and install all dependencies
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Setup complete!$(NC)"

install: ## Install production dependencies only
	$(PIP) install -e .

dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=src/llm_gateway --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v

lint: ## Run linting checks
	@echo "$(BLUE)Running linter...$(NC)"
	$(RUFF) check src/ tests/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

type-check: ## Run type checking
	@echo "$(BLUE)Running type checker...$(NC)"
	$(MYPY) src/

check: lint type-check test ## Run all checks (lint, type-check, test)

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run: ## Run the development server
	@echo "$(BLUE)Starting development server...$(NC)"
	$(UVICORN) llm_gateway.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the production server
	$(UVICORN) llm_gateway.main:app --host 0.0.0.0 --port 8000 --workers 4

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t llm-gateway:latest .

docker-up: ## Start services with Docker Compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started! API available at http://localhost:8000$(NC)"

docker-down: ## Stop Docker Compose services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose down

docker-logs: ## View Docker Compose logs
	docker-compose logs -f

docker-restart: docker-down docker-up ## Restart Docker Compose services

# Development helpers
shell: ## Start Python shell with project imported
	$(VENV)/bin/python -c "from llm_gateway import *; import code; code.interact(local=locals())"

docs: ## Generate API documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	$(VENV)/bin/python -c "from llm_gateway.main import app; import json; print(json.dumps(app.openapi(), indent=2))" > openapi.json
	@echo "$(GREEN)OpenAPI spec written to openapi.json$(NC)"
