.PHONY: help setup test lint format clean

UV := $(shell command -v uv 2>/dev/null)

# Workstation = every extra + the CUDA 13.2 PyTorch build. Override for a smaller target,
# e.g. `make setup EXTRAS=serve,edge` or `make setup EXTRAS=train,cpu,dev` (no NVIDIA GPU).
EXTRAS ?= train,cu132,export,serve,dashboard,edge,dev
comma := ,
EXTRA_FLAGS := $(foreach extra,$(subst $(comma), ,$(EXTRAS)),--extra $(extra))

help:
	@echo "Makefile commands:"
	@echo "  setup 	- Set up the virtual environment and install dependencies (EXTRAS=$(EXTRAS))"
	@echo "  test	-  Run unitary tests"
	@echo "  lint 	- Run ruff check and format"
	@echo "  clean 	- Clean up generated files and caches"

setup:
ifeq ($(strip $(UV)),)
	$(error "uv is required: https://docs.astral.sh/uv/getting-started/installation/")
endif
	@echo "Syncing .venv with uv.lock (extras: $(EXTRAS))..."
	uv sync --locked $(EXTRA_FLAGS)
	@echo "Setup complete. Activate the venv with: source .venv/bin/activate"

test: 
	@echo "Running unit tests..."
	.venv/bin/python -m pytest tests/ -q

lint:
	@echo "Runnin lint tests..."
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format: 
	@echo "Running code formatting..."
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .

clean:
	@echo "WARNING ! : This will delete all models, and figures generate, all __pycache__ folders."
	@echo "Use : Ctrl+C to cancel."
	@echo "Running clean generated files and caches..."
	@sleep 5
	find . -type d -name "__pycache__" -exec rm -rf {} + && \
	rm -rf .pytest_cache .ruff_cache && \
	rm -rf outputs/models/* outputs/figures/* outputs/reports/* outputs/logs/*