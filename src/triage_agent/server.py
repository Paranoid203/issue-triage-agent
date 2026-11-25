"""HTTP service around the LangGraph agent.

    uvicorn triage_agent.server:app --port 8000

POST /api/threads/{thread_id}/messages   one user turn, streamed back as Server-Sent Events
GET  /api/docs/search?q=...              BM25 documentation search (the agent's search_docs tool)
GET  /api/metrics                        metrics of the latest eval run (results/LATEST)
GET  /healthz                            liveness: the process is up
GET  /readyz                             readiness: the model is configured and the MCP tools are loaded

SSE events of a turn, in order: `node` and `plan` (the subtasks), then any number of `node`,
`tool_call`, `tool_result`, `submission` (a decision the submit tool accepted) and `text`,
then `done` (the turn trace summary) or `error`.

Conversation memory is the graph checkpointer, keyed by thread_id and held in process memory, so a
thread must keep hitting the same replica (the Kubernetes Ingress hashes on the request path).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from triage_agent.agent import MCP_TOOLS, _parse, extract_turn
from triage_agent.config import DEFAULT_MODEL, MODELS, ROOT
from triage_agent.graph import build_graph
from triage_agent.tools import FINAL_TOOL, local_tools, mcp_tools, normalize_decision, search_docs

RESULTS = ROOT / "results"
ToolsFactory = Callable[[AsyncExitStack], Awaitable[list[BaseTool]]]


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


async def default_tools(stack: AsyncExitStack) -> list[BaseTool]:
    return local_tools() + await stack.enter_async_context(mcp_tools())


def default_model() -> BaseChatModel:
    return MODELS[os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)].chat_model()


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def preview(result: Any) -> Any:
    """A compact view of a tool result for the UI: ids and titles, not whole documents."""
    keep = ("number", "title", "state", "created_at", "labels", "chunk_id", "path", "heading")
    if isinstance(result, list):
        return [{k: r[k] for k in keep if k in r} if isinstance(r, dict) else r for r in result[:5]]
    if isinstance(result, dict):
        return {k: result[k] for k in keep if k in result}
    return str(result)[:300]


def update_events(node: str, update: dict, calls: dict[str, dict]) -> list[str]:
    """Translate one graph-node update into SSE events; `calls` remembers tool calls by id."""
    events = [sse("node", {"name": node})]
    if node == "planner":
        events.append(sse("plan", {"subtasks": update["plan"]}))
    for m in update.get("messages", []):
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                source = "mcp" if call["name"] in MCP_TOOLS else "local"
                calls[call["id"]] = call
                events.append(sse("tool_call", {"id": call["id"], "name": call["name"], "args": call["args"], "source": source}))
            if m.text and not m.tool_calls:
                events.append(sse("text", {"text": m.text}))
        elif isinstance(m, ToolMessage):
            ok = m.status != "error"
            body = {"id": m.tool_call_id, "name": m.name, "ok": ok}
            body |= {"result": preview(_parse(m.content))} if ok else {"error": str(m.content)[:300]}
            events.append(sse("tool_result", body))
            if ok and m.name == FINAL_TOOL and m.tool_call_id in calls:
                events.append(sse("submission", normalize_decision(calls[m.tool_call_id]["args"])))
    return events


def create_app(
    model_factory: Callable[[], BaseChatModel] = default_model, tools_factory: ToolsFactory = default_tools
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph, app.state.not_ready = None, "starting"
        async with AsyncExitStack() as stack:
            try:
                tools = await tools_factory(stack)
                app.state.graph = build_graph(model_factory(), tools)
                app.state.tools = [t.name for t in tools]
                app.state.not_ready = None
            except Exception as exc:  # stay alive so /healthz works and /readyz explains why
                while isinstance(exc, BaseExceptionGroup) and exc.exceptions:  # e.g. from the MCP task group
                    exc = exc.exceptions[0]
                app.state.not_ready = f"{type(exc).__name__}: {exc}"[:300]
            yield

    app = FastAPI(title="triage-agent", lifespan=lifespan)
    locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        if app.state.not_ready:
            return JSONResponse({"status": "not ready", "reason": app.state.not_ready}, status_code=503)
        return JSONResponse({"status": "ready", "tools": app.state.tools})

    @app.get("/api/docs/search")
    def docs_search(q: str, k: int = 5) -> list[dict]:  # sync: FastAPI runs it in a worker thread
        return search_docs.invoke({"query": q, "k": k})

    @app.get("/api/metrics")
    def metrics() -> dict:
        latest = RESULTS / "LATEST"
        if not latest.exists():
            raise HTTPException(404, "no eval run yet")
        run = latest.read_text().strip()
        load = lambda name: json.loads((RESULTS / run / name).read_text())  # noqa: E731
        return {"run": run, "config": load("config.json"), "metrics": load("metrics.json")}

    @app.post("/api/threads/{thread_id}/messages")
    async def send(thread_id: str, msg: Message) -> StreamingResponse:
        if app.state.not_ready:
            raise HTTPException(503, app.state.not_ready)
        lock = locks[thread_id]
        if lock.locked():
            raise HTTPException(409, "this thread is still answering the previous message")
        return StreamingResponse(
            stream_turn(app.state.graph, thread_id, msg.text, lock),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def stream_turn(graph, thread_id: str, text: str, lock: asyncio.Lock) -> AsyncIterator[str]:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
    start, calls = time.monotonic(), {}
    async with lock:
        try:
            async for chunk in graph.astream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
                for node, update in chunk.items():
                    for event in update_events(node, update or {}, calls):
                        yield event
            trace = extract_turn((await graph.aget_state(config)).values, text)
        except Exception as exc:
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"[:500]})
            return
        summary = trace.to_dict() | {"seconds": round(time.monotonic() - start, 2)}
        summary.pop("steps")
        yield sse("done", summary)


app = create_app()
