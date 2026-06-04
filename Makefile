# Home Loan Default Prediction — Makefile
# Usage: make <target>

.PHONY: train app api test lint clean docker-up docker-down help

## Train model with full pipeline (all 7 tables)
train:
	python src/models/train.py

## Launch Streamlit dashboard
app:
	streamlit run app/app.py

## Launch FastAPI service
api:
	uvicorn api.main:app --reload --port 8000

## Run test suite
test:
	pytest tests/ -v --tb=short

## Run linting
lint:
	flake8 src/ app/ api/ tests/ --max-line-length=100 --ignore=E203,W503

## Remove compiled Python files and test cache
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

## Start all Docker services (API + Streamlit)
docker-up:
	docker-compose up -d

## Stop all Docker services
docker-down:
	docker-compose down

## Generate SHAP plots and feature importance (run after make train)
explain:
	python src/models/run_explain.py

## Run Optuna hyperparameter tuning (default: xgboost, 50 trials)
tune:
	python src/models/tune.py --model xgboost --n-trials 50

## Rebuild master feature table from raw CSVs (force rebuild even if cache exists)
build-features:
	python -c "from src.features.pipeline import load_or_build_master; load_or_build_master(force_rebuild=True)"

## Show this help
help:
	@echo "Available targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
