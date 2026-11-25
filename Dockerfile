# API image: FastAPI + LangGraph agent + the MCP issue-tracker server (spawned over stdio).
# Build: docker build -t triage-agent-api:0.1.0 .

FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
# Dependencies only (the project itself runs from /app/src via PYTHONPATH).
RUN uv sync --frozen --no-install-project --extra server

FROM python:3.13-slim
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src ./src
COPY data ./data
COPY results ./results
RUN python -m compileall -q src
ENV PATH=/app/.venv/bin:$PATH PYTHONPATH=/app/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001
EXPOSE 8000
CMD ["uvicorn", "triage_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
