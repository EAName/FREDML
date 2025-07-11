.PHONY: help install test lint format clean build run deploy

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/

build: ## Build Docker image
	docker build -t fred-ml .

run: ## Run application locally
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run-docker: ## Run with Docker Compose
	docker-compose up --build

deploy: ## Deploy to Kubernetes
	kubectl apply -f kubernetes/
	helm install fred-ml helm/

logs: ## View application logs
	docker-compose logs -f fred-ml

shell: ## Open shell in container
	docker-compose exec fred-ml bash

migrate: ## Run database migrations
	alembic upgrade head

setup-dev: install format lint test ## Setup development environment 