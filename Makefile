PY := .venv/bin/python
LATEST = $(shell cat results/LATEST)

.PHONY: install test lint demo chat smoke eval report all data

install:            ## create .venv and install the package with dev tools
	uv venv && uv pip install -e '.[dev]'

lint:
	.venv/bin/ruff check src evals scripts tests && .venv/bin/ruff format --check src evals scripts tests

test:               ## unit + integration tests (scripted model, real MCP server, no API calls)
	.venv/bin/pytest -q

demo:               ## two-turn demo: a two-part request, then a follow-up that needs memory
	.venv/bin/triage-agent

chat:               ## interactive multi-turn chat
	.venv/bin/triage-agent --chat

smoke:              ## 2 cases per category, 1 repeat, all variants
	$(PY) -m evals.run_eval --limit 2

eval:               ## full eval: 40 cases x 3 variants x 1 repeat -> results/<timestamp>/, then sync README
	$(PY) -m evals.run_eval --concurrency 3
	$(PY) scripts/sync_readme.py

report:             ## failure-mode breakdown of the latest run
	$(PY) -m evals.failure_report results/$(LATEST)

all: test demo eval ## everything, in one command

data:               ## rebuild data/ from GitHub (needs an authenticated `gh`), then the eval cases
	$(PY) scripts/build_data.py
	$(PY) scripts/build_cases.py
