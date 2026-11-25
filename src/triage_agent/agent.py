"""A multi-turn session on top of the LangGraph agent, with a per-turn trace for evaluation."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from triage_agent.config import ModelSpec
from triage_agent.graph import PLANNER_PROMPT, SYSTEM_PROMPT, build_graph, submitted_subtasks
from triage_agent.tools import FINAL_TOOL, normalize_decision

MCP_TOOLS = {"search_issues", "get_issue"}


@dataclass
class Step:
    tool: str
    arguments: dict
    valid: bool
    error: str | None
    result: Any
    source: str  # "local" or "mcp"


@dataclass
class TurnTrace:
    user: str
    plan: list[str]
    steps: list[Step] = field(default_factory=list)
    submissions: list[dict] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str = ""  # closed | text_answer | budget | error: ...
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Session:
    """One conversation. Each `send` is one user turn; memory lives in the graph checkpointer."""

    def __init__(
        self,
        model: ModelSpec | BaseChatModel,
        tools: list[BaseTool],
        system_prompt: str = SYSTEM_PROMPT,
        planner_prompt: str = PLANNER_PROMPT,
    ) -> None:
        chat = model.chat_model() if isinstance(model, ModelSpec) else model
        self.tool_names = [t.name for t in tools]
        self.graph = build_graph(chat, tools, system_prompt, planner_prompt)
        self.config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 80}

    async def send(self, text: str) -> TurnTrace:
        start = time.monotonic()
        try:
            state = await self.graph.ainvoke({"messages": [HumanMessage(text)]}, self.config)
        except Exception as exc:  # recorded as a failed turn; the eval keeps going
            trace = TurnTrace(text, [], stop_reason=f"error: {type(exc).__name__}: {exc}"[:500])
            trace.seconds = round(time.monotonic() - start, 2)
            return trace
        trace = extract_turn(state, text)
        trace.seconds = round(time.monotonic() - start, 2)
        return trace


def _parse(content: Any) -> Any:
    """Tool results arrive as JSON text (local tools) or MCP content blocks; decode both."""
    if isinstance(content, list):
        texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
        parsed = [_parse(t) for t in texts]
        if len(parsed) == 1:
            return parsed[0]
        return parsed
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return content
        return value.get("result", value) if isinstance(value, dict) and set(value) == {"result"} else value
    return content


def extract_turn(state: dict, text: str) -> TurnTrace:
    turn = state["messages"][state["turn_start"] :]
    trace = TurnTrace(text, state["plan"])
    if usage := state.get("planner_usage"):
        trace.llm_calls += 1
        trace.input_tokens += usage.get("input_tokens", 0)
        trace.output_tokens += usage.get("output_tokens", 0)
    results = {m.tool_call_id: m for m in turn if isinstance(m, ToolMessage)}
    for m in turn:
        if not isinstance(m, AIMessage):
            continue
        trace.llm_calls += 1
        if m.usage_metadata:
            trace.input_tokens += m.usage_metadata.get("input_tokens", 0)
            trace.output_tokens += m.usage_metadata.get("output_tokens", 0)
        if m.text and not m.tool_calls:
            trace.final_text = m.text
        for call in m.tool_calls:
            res = results.get(call["id"])
            ok = res is not None and res.status != "error"
            trace.steps.append(
                Step(
                    call["name"],
                    call["args"],
                    ok,
                    None if ok else (str(res.content)[:300] if res else "no tool result"),
                    _parse(res.content) if ok else None,
                    "mcp" if call["name"] in MCP_TOOLS else "local",
                )
            )
            if ok and call["name"] == FINAL_TOOL:
                trace.submissions.append(normalize_decision(call["args"]))
    done = set(range(1, len(trace.plan) + 1)) <= submitted_subtasks(turn)
    trace.stop_reason = "closed" if done else ("text_answer" if trace.final_text else "budget")
    return trace
