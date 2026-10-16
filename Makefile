.PHONY: help install test lint eval build clean

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@echo "make install   - create venv and install the package with dev extras"
	@echo "make test      - run the test suite"
	@echo "make eval      - run the evaluation harness and print the report"
	@echo "make lint      - compile-check every source file"
	@echo "make build     - build the wheel and sdist"
	@echo "make clean     - remove build and cache artifacts"

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

install: $(BIN)/python
	$(BIN)/python -m pip install -e ".[dev]"

test:
	$(BIN)/python -m pytest -q

eval:
	$(BIN)/citekit eval evals/golden.json

lint:
	$(BIN)/python -m compileall -q src

build: clean
	$(BIN)/python -m pip install --upgrade build
	$(BIN)/python -m build

clean:
	rm -rf build dist .pytest_cache htmlcov *.egg-info src/*.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
