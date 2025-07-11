.PHONY: help install test lint format clean build run deploy

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -e .
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=xml

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
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

build: ## Build Docker image
	docker build -t fred-ml .

run: ## Run application locally
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run-docker: ## Run with Docker Compose (development)
	docker-compose -f deploy/docker/docker-compose.dev.yml up --build

run-prod: ## Run with Docker Compose (production)
	docker-compose -f deploy/docker/docker-compose.prod.yml up --build

deploy: ## Deploy to Kubernetes
	kubectl apply -f deploy/kubernetes/

deploy-helm: ## Deploy with Helm
	helm install fred-ml deploy/helm/

logs: ## View application logs
	docker-compose -f deploy/docker/docker-compose.dev.yml logs -f fred-ml

shell: ## Open shell in container
	docker-compose -f deploy/docker/docker-compose.dev.yml exec fred-ml bash

migrate: ## Run database migrations
	alembic upgrade head

setup-dev: install format lint test ## Setup development environment

ci: test lint format ## Run CI checks locally

package: clean build ## Build package for distribution
	python -m build

publish: package ## Publish to PyPI
	twine upload dist/* 