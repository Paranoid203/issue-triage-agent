"""MCP server exposing a read-only GitHub issue tracker snapshot of pydantic/pydantic.

Run standalone (stdio transport):  python -m triage_agent.mcp_server
Any MCP client (this agent, Claude Desktop, Cursor, ...) can then call its tools.
"""

from __future__ import annotations

import os
from functools import cache

from mcp.server.fastmcp import FastMCP

from triage_agent.config import DATA_DIR
from triage_agent.retrieval import Corpus

mcp = FastMCP("pydantic-issue-tracker", log_level="WARNING")

# Optional time cutoff: only issues filed before this issue number are visible. The eval sets it
# per case so the agent sees the tracker as it was when the reported issue was filed.
BEFORE = int(os.environ.get("TRACKER_BEFORE", "0")) or None


@cache
def tracker() -> Corpus:
    # Title is repeated so that title words weigh more than body words.
    corpus = Corpus(DATA_DIR / "issues.jsonl", lambda r: f"{r['title']} {r['title']} {r['body']}")
    if BEFORE:
        corpus.restrict(lambda r: r["number"] < BEFORE)
    return corpus


@mcp.tool()
def search_issues(query: str, k: int = 5) -> list[dict]:
    """Keyword search over existing pydantic GitHub issues (open and closed).

    Use it to find whether a reported problem or request already has an issue.
    Returns issue number, title, state, labels and a short body excerpt.
    """
    k = max(1, min(int(k), 10))
    return [
        {
            "number": r["number"],
            "title": r["title"],
            "state": r["state"],
            "labels": r["labels"],
            "created_at": r["created_at"],
            "excerpt": r["body"][:300],
        }
        for r in tracker().search(query, k)
    ]


@mcp.tool()
def get_issue(number: int) -> dict:
    """Fetch one issue by number with its full (truncated to 1500 chars) body and URL."""
    for r in tracker().rows:
        if r["number"] == int(number) and (not BEFORE or r["number"] < BEFORE):
            return r
    return {"error": f"issue #{number} not found in the tracker snapshot"}


if __name__ == "__main__":
    mcp.run()
