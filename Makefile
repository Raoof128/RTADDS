.PHONY: install dev lint format test run clean

install:
python -m pip install --upgrade pip
pip install -r requirements.txt

dev: install
pip install -r requirements-dev.txt

lint:
ruff check .
black --check .

format:
black .
ruff check --fix .

test:
pytest

run:
uvicorn backend.main:app --host 0.0.0.0 --port 8000

clean:
rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ models/*.keras assets/*.pdf logs/*
