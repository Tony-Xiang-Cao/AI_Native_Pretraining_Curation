.PHONY: install test experiments figures report lint all clean

install:
	pip install -e ".[judge,dev]"

test:
	pytest -q

experiments:
	python scripts/run_experiments.py

figures:
	python scripts/make_figures.py

report: experiments figures

lint:
	ruff check src tests

all: experiments figures test

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
