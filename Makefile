# Enterprise-Grade Makefile for FRED ML
# Comprehensive build, test, and deployment automation

.PHONY: help install test clean build deploy lint format docs setup dev prod

# Default target
help: ## Show this help message
	@echo "FRED ML - Enterprise Economic Analytics Platform"
	@echo "================================================"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables:"
	@echo "  FRED_API_KEY     - Your FRED API key"
	@echo "  AWS_ACCESS_KEY_ID - AWS access key for cloud features"
	@echo "  AWS_SECRET_ACCESS_KEY - AWS secret key"
	@echo "  ENVIRONMENT      - Set to 'production' for production mode"

# Development setup
setup: ## Initial project setup
	@echo "🚀 Setting up FRED ML development environment..."
	python scripts/setup_venv.py
	@echo "✅ Development environment setup complete!"

venv-create: ## Create virtual environment
	@echo "🏗️ Creating virtual environment..."
	python scripts/setup_venv.py
	@echo "✅ Virtual environment created!"

venv-activate: ## Activate virtual environment
	@echo "🔌 Activating virtual environment..."
	@if [ -d ".venv" ]; then \
		echo "Virtual environment found at .venv/"; \
		echo "To activate, run: source .venv/bin/activate"; \
		echo "Or on Windows: .venv\\Scripts\\activate"; \
	else \
		echo "❌ Virtual environment not found. Run 'make venv-create' first."; \
	fi

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Dependencies installed!"

# Testing targets
test: ## Run all tests
	@echo "🧪 Running comprehensive test suite..."
	python tests/run_tests.py
	@echo "✅ All tests completed!"

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	python -m pytest tests/unit/ -v --tb=short
	@echo "✅ Unit tests completed!"

test-integration: ## Run integration tests only
	@echo "🔗 Running integration tests..."
	python -m pytest tests/integration/ -v --tb=short
	@echo "✅ Integration tests completed!"

test-e2e: ## Run end-to-end tests only
	@echo "🚀 Running end-to-end tests..."
	python -m pytest tests/e2e/ -v --tb=short
	@echo "✅ End-to-end tests completed!"

test-coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated!"

# Code quality targets
lint: ## Run linting checks
	@echo "🔍 Running code linting..."
	flake8 src/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "✅ Linting completed!"

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black src/ tests/ scripts/ --line-length=88
	isort src/ tests/ scripts/ --profile=black
	@echo "✅ Code formatting completed!"

type-check: ## Run type checking with mypy
	@echo "🔍 Running type checks..."
	mypy src/ --ignore-missing-imports --disallow-untyped-defs
	@echo "✅ Type checking completed!"

# Cleanup targets
clean: ## Clean up build artifacts and cache
	@echo "🧹 Cleaning up build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/
	@echo "✅ Cleanup completed!"

clean-redundant: ## Clean up redundant test files
	@echo "🗑️ Cleaning up redundant files..."
	python scripts/cleanup_redundant_files.py --live
	@echo "✅ Redundant files cleaned up!"

# Build targets
build: clean ## Build the project
	@echo "🔨 Building FRED ML..."
	python setup.py sdist bdist_wheel
	@echo "✅ Build completed!"

build-docker: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t fred-ml:latest .
	@echo "✅ Docker image built!"

# Development targets
dev: ## Start development environment
	@echo "🚀 Starting development environment..."
	@echo "Make sure you have set FRED_API_KEY environment variable"
	streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0

dev-local: ## Start local development server
	@echo "🏠 Starting local development server..."
	streamlit run frontend/app.py --server.port=8501

# Production targets
prod: ## Start production environment
	@echo "🏭 Starting production environment..."
	ENVIRONMENT=production streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0

# Documentation targets
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	python scripts/generate_docs.py
	@echo "✅ Documentation generated!"

docs-serve: ## Serve documentation locally
	@echo "📖 Serving documentation..."
	python -m http.server 8000 --directory docs/
	@echo "📖 Documentation available at http://localhost:8000"

# Deployment targets
deploy-local: ## Deploy locally
	@echo "🚀 Deploying locally..."
	python scripts/deploy_local.py
	@echo "✅ Local deployment completed!"

deploy-aws: ## Deploy to AWS
	@echo "☁️ Deploying to AWS..."
	python scripts/deploy_aws.py
	@echo "✅ AWS deployment completed!"

deploy-streamlit: ## Deploy to Streamlit Cloud
	@echo "☁️ Deploying to Streamlit Cloud..."
	@echo "Make sure your repository is connected to Streamlit Cloud"
	@echo "Set the main file path to: streamlit_app.py"
	@echo "Add environment variables for FRED_API_KEY and AWS credentials"
	@echo "✅ Streamlit Cloud deployment instructions provided!"

# Quality assurance targets
qa: lint format type-check test ## Run full quality assurance suite
	@echo "✅ Quality assurance completed!"

pre-commit: format lint type-check test ## Run pre-commit checks
	@echo "✅ Pre-commit checks completed!"

# Monitoring and logging targets
logs: ## View application logs
	@echo "📋 Viewing application logs..."
	tail -f logs/fred_ml.log

logs-clear: ## Clear application logs
	@echo "🗑️ Clearing application logs..."
	rm -f logs/*.log
	@echo "✅ Logs cleared!"

# Backup and restore targets
backup: ## Create backup of current state
	@echo "💾 Creating backup..."
	tar -czf backup/fred_ml_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='.git' --exclude='.venv' --exclude='__pycache__' \
		--exclude='*.pyc' --exclude='.pytest_cache' --exclude='htmlcov' .
	@echo "✅ Backup created!"

restore: ## Restore from backup (specify BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "❌ Please specify BACKUP_FILE=path/to/backup.tar.gz"; \
		exit 1; \
	fi
	@echo "🔄 Restoring from backup: $(BACKUP_FILE)"
	tar -xzf $(BACKUP_FILE)
	@echo "✅ Restore completed!"

# Health check targets
health: ## Check system health
	@echo "🏥 Checking system health..."
	python scripts/health_check.py
	@echo "✅ Health check completed!"

# Configuration targets
config-validate: ## Validate configuration
	@echo "🔍 Validating configuration..."
	python -c "from config.settings import get_config; config = get_config(); print('✅ Configuration valid!')"
	@echo "✅ Configuration validation completed!"

config-show: ## Show current configuration
	@echo "📋 Current configuration:"
	python -c "from config.settings import get_config; import json; config = get_config(); print(json.dumps(config.to_dict(), indent=2))"

# Database targets
db-migrate: ## Run database migrations
	@echo "🗄️ Running database migrations..."
	python scripts/db_migrate.py
	@echo "✅ Database migrations completed!"

db-seed: ## Seed database with initial data
	@echo "🌱 Seeding database..."
	python scripts/db_seed.py
	@echo "✅ Database seeding completed!"

# Analytics targets
analytics-run: ## Run analytics pipeline
	@echo "📊 Running analytics pipeline..."
	python scripts/run_analytics.py
	@echo "✅ Analytics pipeline completed!"

analytics-cache-clear: ## Clear analytics cache
	@echo "🗑️ Clearing analytics cache..."
	rm -rf data/cache/*
	@echo "✅ Analytics cache cleared!"

# Security targets
security-scan: ## Run security scan
	@echo "🔒 Running security scan..."
	bandit -r src/ -f json -o security_report.json || true
	@echo "✅ Security scan completed!"

security-audit: ## Run security audit
	@echo "🔍 Running security audit..."
	safety check
	@echo "✅ Security audit completed!"

# Performance targets
performance-test: ## Run performance tests
	@echo "⚡ Running performance tests..."
	python scripts/performance_test.py
	@echo "✅ Performance tests completed!"

performance-profile: ## Profile application performance
	@echo "📊 Profiling application performance..."
	python -m cProfile -o profile_output.prof scripts/profile_app.py
	@echo "✅ Performance profiling completed!"

# All-in-one targets
all: setup install qa test build ## Complete setup and testing
	@echo "🎉 Complete setup and testing completed!"

production-ready: clean qa test-coverage security-scan performance-test ## Prepare for production
	@echo "🏭 Production readiness check completed!"

# Helpers
version: ## Show version information
	@echo "FRED ML Version: $(shell python -c "import src; print(src.__version__)" 2>/dev/null || echo "Unknown")"
	@echo "Python Version: $(shell python --version)"
	@echo "Pip Version: $(shell pip --version)"

status: ## Show project status
	@echo "📊 Project Status:"
	@echo "  - Python files: $(shell find src/ -name '*.py' | wc -l)"
	@echo "  - Test files: $(shell find tests/ -name '*.py' | wc -l)"
	@echo "  - Lines of code: $(shell find src/ -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "  - Test coverage: $(shell python -m pytest tests/ --cov=src --cov-report=term-missing | tail -1 || echo "Not available")"

# Default target
.DEFAULT_GOAL := help 