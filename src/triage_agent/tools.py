"""Tools offered to the model: local LangChain tools plus the MCP issue-tracker tools."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field

from triage_agent.config import DATA_DIR
from triage_agent.retrieval import Corpus

FINAL_TOOL = "submit_triage"
IssueType = Literal["question", "bug", "feature_request", "documentation"]


@cache
def docs_corpus() -> Corpus:
    return Corpus(DATA_DIR / "doc_chunks.jsonl", lambda r: f"{r['heading']} {r['text']}")


@tool
def search_docs(query: str, k: int = 5) -> list[dict]:
    """Search the official Pydantic documentation (BM25 keyword search).

    Use it for how-to questions and to check documented behaviour. Returns doc chunks.
    """
    k = max(1, min(int(k), 8))
    return [
        {"chunk_id": r["id"], "path": r["path"], "heading": r["heading"], "text": r["text"][:900]}
        for r in docs_corpus().search(query, k)
    ]


class TriageDecision(BaseModel):
    subtask: int = Field(1, description="Number of the subtask in the plan that this decision answers (1-based)")
    type: IssueType = Field(
        description="question = usage question; bug = defect report; "
        "feature_request = new capability; documentation = docs problem"
    )
    duplicate_of: int | None = Field(None, description="Number of an existing issue this duplicates, else null")
    answer: str = Field(description="Reply to the user: the answer, or the triage summary")
    references: list[str] = Field(default_factory=list, description="Doc chunk ids or issue numbers you relied on")


@tool(FINAL_TOOL, args_schema=TriageDecision)
def submit_triage(**decision) -> str:
    """Submit the final answer for one subtask. Call it once per subtask in the plan."""
    return f"recorded subtask {decision.get('subtask', 1)}"


def normalize_decision(args: dict) -> dict:
    """The decision as the tool validated it (e.g. duplicate_of "7687" -> 7687), not the raw model output."""
    try:
        return TriageDecision.model_validate(args).model_dump()
    except ValueError:
        return dict(args)


def local_tools(use_docs: bool = True) -> list[BaseTool]:
    return [search_docs, submit_triage] if use_docs else [submit_triage]


@asynccontextmanager
async def mcp_tools(tracker_before: int | None = None):
    """Start the issue-tracker MCP server over stdio and yield its tools as LangChain tools.

    `tracker_before` hides every issue filed at or after that issue number (used by the eval
    so the agent sees the tracker as it was when the reported issue was filed).
    """
    # The MCP SDK starts stdio servers with a minimal environment (no PYTHONPATH), so point the child
    # at this package explicitly; that also covers running from source, as the container image does.
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    if tracker_before:
        env["TRACKER_BEFORE"] = str(tracker_before)
    connection = {"transport": "stdio", "command": sys.executable, "args": ["-m", "triage_agent.mcp_server"], "env": env}
    client = MultiServerMCPClient({"issues": connection})
    async with client.session("issues") as session:
        yield await load_mcp_tools(session)
