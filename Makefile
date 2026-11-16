.DEFAULT_GOAL := help
PYTHON ?= python3

.PHONY: help install run dev test cov lint format typecheck check train clean

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install runtime and development dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

run:  ## Start the server on http://127.0.0.1:8000
	$(PYTHON) -m app.main

dev:  ## Start the server with auto-reload
	$(PYTHON) -m app.main --reload

test:  ## Run the test suite
	$(PYTHON) -m pytest

cov:  ## Run the tests with a coverage report
	$(PYTHON) -m pytest --cov --cov-report=term-missing

lint:  ## Lint (includes bandit-style security rules) and check formatting
	ruff check .
	ruff format --check .

format:  ## Auto-format and apply safe lint fixes
	ruff check --fix .
	ruff format .

typecheck:  ## Static type checking (mypy --strict)
	mypy

check: lint typecheck test  ## Everything CI runs

train:  ## Retrain the classifier and print cross-validated metrics
	$(PYTHON) -m scripts.train_classifier --errors

clean:  ## Remove caches and the generated model artefact
	rm -rf data/model .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
