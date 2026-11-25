PY := .venv/bin/python
LATEST = $(shell cat results/LATEST)

.PHONY: install test lint demo chat smoke eval report all data serve web web-check compose-up compose-down kind-up load-test kind-down

install:            ## create .venv and install the package with dev tools and the HTTP server
	uv venv && uv pip install -e '.[dev,server]'

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

serve:              ## HTTP API with SSE on :8000 (TRIAGE_MODEL picks the model)
	.venv/bin/uvicorn triage_agent.server:app --port 8000

web:                ## frontend dev server on :5173, proxying /api to :8000
	npm --prefix web ci && npm --prefix web run dev

web-check:          ## frontend typecheck + production build
	npm --prefix web ci && npm --prefix web run build

compose-up:         ## gateway + api + web on http://localhost:8080 (needs GEMINI_API_KEY or ZHIPUAI_API_KEY)
	docker compose up --build -d

compose-down:
	docker compose down

kind-up:            ## kind cluster + ingress-nginx + metrics-server + the app, one command
	deploy/kind_up.sh

load-test:          ## k6 load through the Ingress while recording the HPA (needs kind-up)
	deploy/load_test.sh

kind-down:
	kind delete cluster --name triage-agent
