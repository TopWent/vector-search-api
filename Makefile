.PHONY: install test lint format bench seed docker clean

VENV ?= .venv
PY   := $(VENV)/bin/python

install:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

bench:
	$(PY) scripts/benchmark.py

seed:
	$(PY) scripts/seed.py

docker:
	docker build -t vector-search-api:dev .

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache build dist *.egg-info data/*.bin data/*.jsonl data/*.ids
