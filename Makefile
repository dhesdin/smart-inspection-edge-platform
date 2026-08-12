.PHONY: help setup test lint clean

PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

ifeq ($(strip $(PYTHON)),)
$(error "Python 3.10 or higher version is required. Please install Python >=3.10  : https://www.python.org/downloads/")
endif


help:
	@echo "Makefile commands:"
	@echo "  setup 	- Set up the virtual environment and install dependencies"
	@echo "  test	-  Run unitary tests"
	@echo "  lint 	- Run ruff check and format"
	@echo "  clean 	- Clean up generated files and caches"

setup:
	@echo "Setting up the virtual environment and installing dependencies..."
	$(PYTHON) -c "import sys ; sys.exit('Python 3.10 or higher is required. Upgrade your Python version') if sys.version_info < (3,10) else None"
	@echo "Using ${PYTHON} version $(shell $(PYTHON) --version)"

	$(PYTHON) -m venv .venv && \
	.venv/bin/pip install --upgrade pip && \
	.venv/bin/pip install -e .[dev,edge]
	@echo "Setup complete. Activate the venv with: source .venv/bin/activate"


test: 
	@echo "Running unit tests..."
	.venv/bin/python -m pytest tests/ -q

lint:
	@echo "Runnin lint tests..."
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format: 
	@echo "Running code formatting..."
	.venv/bin/ruff format .

clean:
	@echo "WARNING ! : This will delete all models, and figures generate, all __pycache__ folders."
	@echo "Use : Ctrl+C to cancel."
	@echo "Running clean generated files and caches..."
	@sleep 5
	find . -type d -name "__pycache__" -exec rm -rf {} + && \
	rm -rf .pytest_cache .ruff_cache && \
	rm -rf outputs/models/* outputs/figures/* outputs/reports/* outputs/logs/*