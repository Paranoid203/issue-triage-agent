"""LangGraph orchestration: plan the user's message into subtasks, then run a tool-calling loop
until every subtask has a submitted decision.

    START -> planner -> executor -> tools -> executor ... -> END
                           |  (no tool call or budget spent, subtasks open) -> nudge -> executor
                           |  (all subtasks submitted, or reminded twice) -> END

A checkpointer keeps the whole conversation per thread, so follow-up turns can refer back to
earlier answers and tool results.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from triage_agent.tools import FINAL_TOOL

SYSTEM_PROMPT = """\
You are the support and triage assistant for the open-source Python library Pydantic.
Users ask how to do something, file issue reports (bug, feature request, docs problem), or ask
follow-up questions about earlier answers in the conversation.

Work with the tools:
- For how-to questions, ground your answer in the documentation (search_docs) and name the exact
  API, parameter or config setting the user needs.
- For issue reports, search the existing issues (search_issues) with a few different keyword
  queries. If a candidate looks like the same problem or request, open it with get_issue and
  compare before you call it a duplicate. Only set duplicate_of when you are confident.
- For questions about an issue (its date, state, labels), look it up with get_issue.
- Classify each subtask as question, bug, feature_request or documentation.
The current message has been split into the numbered subtasks below. Call submit_triage once per
subtask, with its number, as soon as you have the answer. Use at most about three searches per
subtask; do not repeat a search you already ran. Keep answers short and concrete."""

PLANNER_PROMPT = """\
You split a user's latest message to a Pydantic support assistant into independent subtasks.
Most messages are a single subtask. Split only when the message clearly contains separate
requests (for example a how-to question and a separate bug report). Resolve references to the
earlier conversation ("that", "the issue you linked") into explicit words in the subtask.
Plan only what the latest user message asks; earlier requests are already handled.
Reply with a JSON array of short strings, nothing else."""

# v2 adds one rule, written after the failure analysis of the first full run: GLM-4.7-FlashX split
# single issue reports into "investigate / reproduce / fix" subtasks and then never checked for
# duplicates (docs/failure_analysis.md, section 3.1).
PLANNER_PROMPT_V2 = (
    PLANNER_PROMPT
    + """
An issue report (a bug, a feature request, a docs problem) is exactly one subtask: triage it.
Never split one report into steps such as investigating, reproducing, or fixing it."""
)
PLANNER_PROMPTS = {"v1": PLANNER_PROMPT, "v2": PLANNER_PROMPT_V2}

MAX_SUBTASKS = 4
MAX_NUDGES = 2


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    turn_start: int  # index in `messages` where the current user turn begins
    steps: int  # executor model calls in the current turn
    nudges: int  # reminders sent in the current turn to submit open subtasks
    planner_usage: dict | None  # token usage of the planner call in the current turn


def submitted_subtasks(messages: list[AnyMessage]) -> set[int]:
    """Subtask numbers that have a successful submit_triage call in `messages`."""
    ok_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.status != "error"}
    return {
        int(call["args"].get("subtask", 1))
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call["name"] == FINAL_TOOL and call["id"] in ok_ids
    }


def parse_plan(text: str, fallback: str) -> list[str]:
    match = re.search(r"\[.*\]", text or "", re.S)
    try:
        plan = [str(s).strip() for s in json.loads(match.group(0)) if str(s).strip()] if match else []
    except json.JSONDecodeError:
        plan = []
    return plan[:MAX_SUBTASKS] or [fallback]


def transcript(messages: list[AnyMessage]) -> str:
    """User messages and submitted answers only: enough context for the planner, no tool noise."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            lines += [f"ASSISTANT: {c['args'].get('answer', '')}" for c in m.tool_calls if c["name"] == FINAL_TOOL]
    return "\n\n".join(lines)


def build_graph(
    model: BaseChatModel, tools: list[BaseTool], system_prompt: str = SYSTEM_PROMPT, planner_prompt: str = PLANNER_PROMPT
):
    final_tool = next(t for t in tools if t.name == FINAL_TOOL)
    with_tools = model.bind_tools(tools)
    final_only = model.bind_tools([final_tool])

    def budget(plan: list[str]) -> int:
        return 6 + 4 * (len(plan) - 1)

    async def planner(state: AgentState) -> dict:
        latest = state["messages"][-1].content
        reply = await model.ainvoke(
            [SystemMessage(planner_prompt), HumanMessage(f"Conversation so far:\n\n{transcript(state['messages'])}")]
        )
        return {
            "plan": parse_plan(reply.text, latest),
            "turn_start": len(state["messages"]) - 1,
            "steps": 0,
            "nudges": 0,
            "messages": [],
            "planner_usage": reply.usage_metadata,
        }

    async def executor(state: AgentState) -> dict:
        plan = state["plan"]
        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan, 1))
        prompt = [SystemMessage(f"{system_prompt}\n\nSubtasks of the current message:\n{plan_text}")]
        # Budget spent or already reminded: only the submit tool is offered, so the turn closes.
        out_of_budget = state["steps"] >= budget(plan) - 1 or state["nudges"] > 0
        bound = final_only if out_of_budget else with_tools
        reply = await bound.ainvoke(prompt + state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def nudge(state: AgentState) -> dict:
        open_ = sorted(set(range(1, len(state["plan"]) + 1)) - submitted_subtasks(_turn(state)))
        note = f"Stop searching. Call {FINAL_TOOL} now for subtask(s) {open_} with your best answer."
        return {"messages": [HumanMessage(note)], "nudges": state["nudges"] + 1}

    def after_executor(state: AgentState) -> str:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END if _all_submitted(state) or state["nudges"] >= MAX_NUDGES else "nudge"

    def after_tools(state: AgentState) -> str:
        if _all_submitted(state):
            return END
        if state["steps"] >= budget(state["plan"]):
            return "nudge" if state["nudges"] < MAX_NUDGES else END
        return "executor"

    def _turn(state: AgentState) -> list[AnyMessage]:
        return state["messages"][state["turn_start"] :]

    def _all_submitted(state: AgentState) -> bool:
        return set(range(1, len(state["plan"]) + 1)) <= submitted_subtasks(_turn(state))

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("nudge", nudge)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", after_executor, ["tools", "nudge", END])
    graph.add_conditional_edges("tools", after_tools, ["executor", "nudge", END])
    graph.add_edge("nudge", "executor")
    return graph.compile(checkpointer=InMemorySaver())
